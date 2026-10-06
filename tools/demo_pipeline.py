"""Demonstrations for the polar teacher: search -> collect -> behaviour cloning (BC).

--mode search  : for each egg position (r, theta) on a grid, search a cable-command motion that
                 lands the egg in the bucket (CEM; egg AND pedestal are moved together).
--mode collect : replay those motions (source=search) or roll out an existing policy on its
                 task (source=policy); keep only SUCCESSFUL episodes as (observation, action) pairs.
--mode bc      : start from a checkpoint and train its policy network to copy the demonstrations
                 (supervised learning), then save a normal checkpoint for PPO fine-tuning.
"""
import sys, os, json, time, argparse, math
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from tensordict import TensorDict
import polar_tasks  # noqa: F401  (registers the polar tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["search", "collect", "bc"], required=True)
ap.add_argument("--out", required=True, help="output folder (Drive)")
ap.add_argument("--task", default="Mjlab-SpiRob-Polar-Wide", help="env used for search / search demos")
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--angles", default="40,45,50,55,60,65,70,75,80")
ap.add_argument("--radii", default="135,155,175")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--segments", type=int, default=10)
ap.add_argument("--seg_len", type=int, default=50)
ap.add_argument("--horizon", type=int, default=600)
ap.add_argument("--sigma", type=float, default=0.4)
ap.add_argument("--init_motion", default="", help="json with a 'motion' (e.g. the 97% nominal motion)")
ap.add_argument("--table", default="demo_table.json")
ap.add_argument("--source", choices=["search", "policy"], default="search")
ap.add_argument("--min_success", type=float, default=0.3)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--jitter_mm", type=float, default=1.5)
ap.add_argument("--checkpoint", default="")
ap.add_argument("--policy_task", default="Mjlab-SpiRob-EggToBucket-Stage2Sector-Arc1")
ap.add_argument("--demos", default="", help="bc: comma-separated npz files")
ap.add_argument("--bc_steps", type=int, default=3000)
ap.add_argument("--bc_lr", type=float, default=3e-4)
ap.add_argument("--bc_batch", type=int, default=4096)
ap.add_argument("--save_ckpt", default="")
ap.add_argument("--name", default="demos")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda:0"
K, SEG, T = a.segments, a.seg_len, a.horizon
D = 2 * K
torch.manual_seed(0)
BUCKET = (-0.05, 0.15)


# ---------------------------------------------------------------- environment helpers
def make_env(task, n):
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = n
    cfg.auto_reset = False
    cfg.seed = 1001
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
    vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
    return env, vec


def place(env, xy):
    """Put egg AND (mocap) pedestal at local xy (N, 2) -- same method as sir's polar reset."""
    N = env.num_envs
    ids = torch.arange(N, device=DEV)
    egg = env.scene["egg"]
    st = egg.data.default_root_state.clone()
    off = xy - st[:, :2]
    st[:, :3] += env.scene.env_origins
    st[:, :2] += off
    egg.write_root_state_to_sim(st, env_ids=ids)
    ped = env.scene["pedestal"]
    pst = ped.data.default_root_state
    pose = torch.empty((N, 7), device=DEV)
    pose[:, :3] = pst[:, :3] + env.scene.env_origins
    pose[:, :2] += off
    pose[:, 3:7] = pst[:, 3:7]
    ped.write_mocap_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()


def obs_now(env):
    return TensorDict(env.observation_manager.compute(update_history=True), batch_size=[env.num_envs])


def flags(env):
    tm = env.termination_manager
    g = lambda n: tm.get_term(n).bool()
    return g("success_egg_inside_bucket"), g("egg_fell") | g("egg_oob"), g("nan_state")


def rollout(env, vec, xy, act_fn, steps, record=False):
    """One synchronous batch. xy=None keeps the task's own random spawns."""
    N = env.num_envs
    vec.reset()
    if xy is not None:
        place(env, xy)
    obs = obs_now(env)
    egg, bucket = env.scene["egg"], env.scene["bucket"]
    sid = bucket.find_sites(("bucket_site",), preserve_order=True)[0]
    done = torch.zeros(N, dtype=torch.bool, device=DEV)
    succ, fell, crash = done.clone(), done.clone(), done.clone()
    min_d = torch.full((N,), float("inf"), device=DEV)
    t_end = torch.full((N,), steps, device=DEV, dtype=torch.long)
    O = A = None
    if record:
        O = torch.zeros((steps, N, obs["actor"].shape[1]), device=DEV)
        A = torch.zeros((steps, N, 2), device=DEV)
    for t in range(steps):
        with torch.no_grad():
            act = act_fn(t, obs)
        if record:
            O[t] = obs["actor"]; A[t] = act
        obs, _, dones, _ = vec.step(act)
        d = torch.linalg.norm(egg.data.root_link_pos_w - bucket.data.site_pos_w[:, sid].squeeze(1), dim=-1)
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))
        min_d = torch.where(done, min_d, torch.minimum(min_d, d))
        s, f, n = flags(env)
        new = ~done & (s | f | n)
        succ |= new & s; fell |= new & f & ~s; crash |= new & n & ~s
        t_end = torch.where(new, torch.full_like(t_end, t + 1), t_end)
        done |= new
        if bool(done.all()):
            break
        ended = torch.nonzero(dones.bool()).flatten()
        if len(ended):
            env.reset(env_ids=ended)
    return dict(succ=succ, fell=fell, crash=crash, md=min_d * 1000, t_end=t_end, O=O, A=A)


def grid_positions():
    pos = []
    for th in [float(v) for v in a.angles.split(",")]:
        for r in [float(v) for v in a.radii.split(",")]:
            x, y = r / 1000 * math.cos(math.radians(th)), r / 1000 * math.sin(math.radians(th))
            robot_ok = abs(x) >= 0.038
            bucket_ok = math.hypot(x - BUCKET[0], y - BUCKET[1]) >= 0.055
            if robot_ok and bucket_ok:
                pos.append((r, th, x, y))
    return pos


def motion_act(X):
    Xv = X.view(X.shape[0], K, 2)
    return lambda t, obs: Xv[:, min(t // SEG, K - 1), :]


def keep_successes(res, name):
    ok = res["succ"]
    O, A, te = res["O"], res["A"], res["t_end"]
    obs_l, act_l = [], []
    for i in torch.nonzero(ok).flatten().tolist():
        n = int(te[i])
        obs_l.append(O[:n, i].cpu()); act_l.append(A[:n, i].cpu())
    return obs_l, act_l


# ---------------------------------------------------------------- modes
if a.mode == "search":
    pos = grid_positions()
    G = len(pos)
    env, vec = make_env(a.task, a.num_envs)
    N = env.num_envs
    per = N // G
    assert per >= 8, f"{G} positions need at least {8 * G} envs (have {N}); reduce --angles/--radii"
    gidx = torch.arange(G, device=DEV).repeat_interleave(per)
    gidx = torch.cat([gidx, torch.zeros(N - len(gidx), dtype=torch.long, device=DEV)])
    gxy = torch.tensor([[p[2], p[3]] for p in pos], device=DEV)
    xy = gxy[gidx]
    if a.init_motion and os.path.isfile(a.init_motion):
        m0 = torch.tensor(json.load(open(a.init_motion))["motion"], device=DEV)
        print("warm start from", a.init_motion, flush=True)
    else:
        m0 = torch.zeros(D, device=DEV)
    mu = m0.repeat(G, 1)
    sig = torch.full((G, D), a.sigma, device=DEV)
    best = [{"F": -1e9, "motion": m0.tolist()} for _ in range(G)]
    print(f"searching {G} positions x {per} tries each, {a.iters} iterations", flush=True)
    for it in range(a.iters):
        t0 = time.time()
        X = (mu[gidx] + sig[gidx] * torch.randn(N, D, device=DEV)).clamp(-1, 1)
        if it == 0:
            X[gidx.new_tensor([g * per for g in range(G)])] = m0     # keep the warm start in the batch
        r = rollout(env, vec, xy, motion_act(X), T)
        F = torch.where(r["succ"], 1000.0 - 0.1 * r["t_end"].float(), -r["md"] - 50.0 * r["crash"].float())
        hit = 0
        for g in range(G):
            m = gidx == g
            Fg, Xg = F[m], X[m]
            top = torch.topk(Fg, max(2, int(0.15 * len(Fg)))).indices
            el = Xg[top]
            mu[g] = 0.3 * mu[g] + 0.7 * el.mean(0)
            sig[g] = (0.3 * sig[g] + 0.7 * el.std(0)).clamp(min=0.05)
            if float(Fg[top[0]]) > best[g]["F"]:
                best[g] = {"F": float(Fg[top[0]]), "motion": Xg[top[0]].tolist()}
            hit += int(r["succ"][m].any())
        print(f"iter {it + 1:2d}/{a.iters} | {time.time() - t0:5.0f}s | successes {int(r['succ'].sum())}/{N} | "
              f"positions with >=1 success: {hit}/{G} | sigma {float(sig.mean()):.2f}", flush=True)
    # exact re-test: every env replays its position's best motion
    Xb = torch.tensor([best[g]["motion"] for g in range(G)], device=DEV)[gidx]
    r = rollout(env, vec, xy, motion_act(Xb), T)
    table = []
    print("\n r(mm) theta | exact success | closest(mm)")
    for g, (rr, th, x, y) in enumerate(pos):
        m = gidx == g
        sr = float(r["succ"][m].float().mean())
        table.append({"r_mm": rr, "theta": th, "x": x, "y": y, "motion": best[g]["motion"],
                      "exact_success": sr, "closest_mm": float(r["md"][m].min())})
        print(f" {rr:5.0f} {th:5.0f} | {sr:12.0%} | {float(r['md'][m].min()):8.1f}")
    good = sum(t["exact_success"] >= a.min_success for t in table)
    print(f"\npositions with exact success >= {a.min_success:.0%}: {good}/{G}")
    json.dump({"segments": K, "seg_len": SEG, "horizon": T, "table": table},
              open(f"{a.out}/{a.table}", "w"), indent=1)
    print("saved", f"{a.out}/{a.table}")

elif a.mode == "collect":
    obs_all, act_all = [], []
    if a.source == "search":
        tab = json.load(open(f"{a.out}/{a.table}"))
        rows = [t for t in tab["table"] if t["exact_success"] >= a.min_success]
        assert rows, "no position reached --min_success; lower it or search longer"
        env, vec = make_env(a.task, a.num_envs)
        N = env.num_envs
        G = len(rows)
        gidx = torch.arange(N, device=DEV) % G
        base = torch.tensor([[t["x"], t["y"]] for t in rows], device=DEV)[gidx]
        X = torch.tensor([t["motion"] for t in rows], device=DEV)[gidx]
        print(f"replaying {G} motions (success >= {a.min_success:.0%}) with +-{a.jitter_mm} mm jitter", flush=True)
        for rd in range(a.rounds):
            xy = base + (torch.rand_like(base) * 2 - 1) * a.jitter_mm / 1000
            r = rollout(env, vec, xy, motion_act(X), T, record=True)
            o, ac = keep_successes(r, a.name); obs_all += o; act_all += ac
            print(f"round {rd + 1}/{a.rounds}: {int(r['succ'].sum())}/{N} successful episodes kept", flush=True)
    else:
        assert a.checkpoint and os.path.isfile(a.checkpoint), "give --checkpoint for source=policy"
        env, vec = make_env(a.policy_task, a.num_envs)
        runner_cls = load_runner_cls(a.policy_task) or MjlabOnPolicyRunner
        runner = runner_cls(vec, asdict(load_rl_cfg(a.policy_task)), device=DEV)
        runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=DEV)
        policy = runner.get_inference_policy(device=DEV)
        steps = int(env.max_episode_length) + 1
        for rd in range(a.rounds):
            r = rollout(env, vec, None, lambda t, obs: policy(obs), steps, record=True)
            o, ac = keep_successes(r, a.name); obs_all += o; act_all += ac
            print(f"round {rd + 1}/{a.rounds}: {int(r['succ'].sum())}/{env.num_envs} successful episodes kept", flush=True)
    assert obs_all, "no successful episodes collected"
    O = torch.cat(obs_all).numpy(); A_ = torch.cat(act_all).numpy()
    f = f"{a.out}/{a.name}.npz"
    np.savez_compressed(f, obs=O, act=A_, episodes=len(obs_all))
    print(f"saved {f}: {len(obs_all)} episodes, {len(O)} (observation, action) pairs")

else:  # bc
    assert a.checkpoint and os.path.isfile(a.checkpoint) and a.save_ckpt, "give --checkpoint and --save_ckpt"
    files = [p for p in a.demos.split(",") if p]
    data = []
    for p in files:
        d = np.load(p)
        data.append((torch.tensor(d["obs"], device=DEV, dtype=torch.float32),
                     torch.tensor(d["act"], device=DEV, dtype=torch.float32)))
        print(f"{os.path.basename(p)}: {int(d['episodes'])} episodes, {len(d['obs'])} pairs")
    env, vec = make_env(a.policy_task, 64)
    runner_cls = load_runner_cls(a.policy_task) or MjlabOnPolicyRunner
    runner = runner_cls(vec, asdict(load_rl_cfg(a.policy_task)), device=DEV)
    runner.load(a.checkpoint, load_cfg={"actor": True, "critic": True}, strict=True, map_location=DEV)
    actor = getattr(runner.alg, "_raw_actor", runner.alg.actor)
    actor.eval()                                   # keep the observation normalizer frozen
    opt = torch.optim.Adam(actor.mlp.parameters(), lr=a.bc_lr)
    per_src = max(1, a.bc_batch // len(data))      # equal share of every batch from each demo file

    CLIP = load_rl_cfg(a.policy_task).clip_actions or 1.0
    print(f"action range used by the env: +-{CLIP}", flush=True)
    data = [(O, A_.clamp(-CLIP, CLIP)) for O, A_ in data]   # the env clips anyway

    def bc_loss(pred, tgt):
        # beyond the clip limit every value acts the same, so a target at the limit
        # only penalises predictions that are NOT at/beyond that limit
        err = (pred - tgt) ** 2
        at_hi = (tgt >= CLIP - 1e-4) & (pred >= tgt)
        at_lo = (tgt <= -CLIP + 1e-4) & (pred <= tgt)
        return torch.where(at_hi | at_lo, torch.zeros_like(err), err).mean()

    def batch():
        obs, act = [], []
        for O, A_ in data:
            i = torch.randint(len(O), (per_src,), device=DEV)
            obs.append(O[i]); act.append(A_[i])
        return torch.cat(obs), torch.cat(act)

    with torch.no_grad():
        ob, ac = batch()
        l0 = float(bc_loss(actor(TensorDict({"actor": ob}, batch_size=[len(ob)])), ac))
    print(f"BC start: mean squared action error {l0:.4f}", flush=True)
    for step in range(1, a.bc_steps + 1):
        ob, ac = batch()
        pred = actor(TensorDict({"actor": ob}, batch_size=[len(ob)]))
        loss = bc_loss(pred, ac)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == a.bc_steps:
            print(f"BC step {step:5d}/{a.bc_steps} | action error {loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(a.save_ckpt), exist_ok=True)
    ck = torch.load(a.checkpoint, map_location=DEV, weights_only=False)   # keep critic/optimizer as they were
    ck["actor_state_dict"] = actor.state_dict()
    torch.save(ck, a.save_ckpt)
    print("saved BC checkpoint", a.save_ckpt)
