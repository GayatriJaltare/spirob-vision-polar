"""Trajectory search on the polar (moving-pedestal) setup: is the LEFT side deliverable at all?

Same method as tools/traj_search.py (no RL, no neural network):
  a "motion" = K segments x 2 cable commands in [-1, 1], each held for SEG control steps;
  the cross-entropy method (CEM) keeps the best 10 % of tries and samples around them;
  success = sir's own termination check (egg inside the bucket).
New here: the egg AND its pedestal are placed together at a chosen (r, theta), exactly the
way sir's reset_stage2_sector_egg_and_pedestal moves them. All targets are searched at once,
each with its own share of the parallel environments.

Modes
  hold   : robot held still -- does the egg stay on its pedestal at every target?
  search : CEM per target, then re-checks each target's best motion (exact and with noise)
  video  : record the best motion found for one target
Standalone: none of sir's files are modified.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
import spirob_mjlab  # noqa: F401
from spirob_mjlab import mdp
from spirob_mjlab.entities import movable_pedestal_cfg
from spirob_mjlab.env_cfgs import spirob_egg_to_bucket_stage1_env_cfg
from spirob_mjlab.rl_cfg import spirob_ppo_runner_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["hold", "search", "video"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--targets", default="L130_145:145:130,L130_165:165:130,L140_145:145:140,L140_165:165:140,"
                                      "L150_145:145:150,L150_165:165:150,R65:158:65,R45:158:45",
                help="name:r_mm:theta_deg, comma separated")
ap.add_argument("--num_envs", type=int, default=512)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--segments", type=int, default=10)
ap.add_argument("--seg_len", type=int, default=50, help="control steps per segment (0.01 s each)")
ap.add_argument("--horizon", type=int, default=600, help="control steps per try (6 s)")
ap.add_argument("--sigma", type=float, default=0.6)
ap.add_argument("--elite", type=float, default=0.1)
ap.add_argument("--video_target", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda:0"
K, SEG, T = a.segments, a.seg_len, a.horizon
D = 2 * K
torch.manual_seed(0)
TARGETS = []
for s in a.targets.split(","):
    n, r, th = s.split(":")
    TARGETS.append((n, float(r), float(th)))
CLIP = spirob_ppo_runner_cfg(stage2=True).clip_actions


def make_env(n, render=False):
    cfg = spirob_egg_to_bucket_stage1_env_cfg(play=False)
    cfg.scene.num_envs = n
    cfg.auto_reset = False
    cfg.seed = 1001
    cfg.scene.entities["pedestal"] = movable_pedestal_cfg()
    # sir's sector reset (it needs the mocap pedestal); every egg is then moved to its target below
    cfg.events["stage2_egg_pedestal_spawn"] = EventTermCfg(
        func=mdp.reset_stage2_sector_egg_and_pedestal, mode="reset",
        params={"radius_range_m": (0.139, 0.179), "angle_range_deg": (50.8, 75.8), "radial_strata": 1,
                "angular_strata": 1, "schedule_length": 10, "nominal_slots": 1, "retention_slots": 3,
                "max_resample_attempts": 256, "min_robot_centerline_clearance": 0.038,
                "min_bucket_center_clearance": 0.055})
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array" if render else None)
    return env


def target_xy(groups):
    """(N, 2) egg positions in metres (env-local) for a group index per env."""
    rt = torch.tensor([[r, th] for _, r, th in TARGETS], device=DEV)
    r, th = rt[groups, 0] / 1000.0, torch.deg2rad(rt[groups, 1])
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1)


def place(env, xy):
    """Move egg and pedestal together to xy (env-local), as sir's sector reset does."""
    ids = torch.arange(env.num_envs, device=DEV)
    egg, ped = env.scene["egg"], env.scene["pedestal"]
    es = egg.data.default_root_state.clone()
    off = xy - es[:, :2]
    es[:, :3] += env.scene.env_origins
    es[:, :2] += off
    egg.write_root_state_to_sim(es, env_ids=ids)
    ps = ped.data.default_root_state
    pose = torch.empty((env.num_envs, 7), device=DEV)
    pose[:, :3] = ps[:, :3] + env.scene.env_origins
    pose[:, :2] += off
    pose[:, 3:7] = ps[:, 3:7]
    ped.write_mocap_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()


def rollout(env, vec, motions, xy, steps=T):
    """motions: (N, K, 2). Returns success, fell, crash, min egg-bucket distance (mm), end step, start-z drop (mm)."""
    N = env.num_envs
    vec.reset()
    place(env, xy)
    egg, bucket = env.scene["egg"], env.scene["bucket"]
    sid = bucket.find_sites(("bucket_site",), preserve_order=True)[0]
    tm = env.termination_manager
    z0 = egg.data.root_link_pos_w[:, 2].clone()
    done = torch.zeros(N, dtype=torch.bool, device=DEV)
    succ, fell, crash = done.clone(), done.clone(), done.clone()
    min_d = torch.full((N,), float("inf"), device=DEV)
    min_z = z0.clone()
    t_end = torch.full((N,), float(steps), device=DEV)
    for t in range(steps):
        _, _, dones, _ = vec.step(motions[:, min(t // SEG, K - 1), :])
        d = torch.linalg.norm(egg.data.root_link_pos_w - bucket.data.site_pos_w[:, sid].squeeze(1), dim=-1)
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))
        min_d = torch.where(done, min_d, torch.minimum(min_d, d))
        ez = egg.data.root_link_pos_w[:, 2]
        min_z = torch.where(done, min_z, torch.minimum(min_z, ez))
        s = tm.get_term("success_egg_inside_bucket").bool()
        f = tm.get_term("egg_fell").bool() | tm.get_term("egg_oob").bool()
        n = tm.get_term("nan_state").bool()
        new = ~done & (s | f | n)
        succ |= new & s
        fell |= new & f & ~s
        crash |= new & n & ~s
        t_end = torch.where(new, torch.full_like(t_end, float(t)), t_end)
        done |= new
        if bool(done.all()):
            break
        ended = torch.nonzero(dones.bool()).flatten()
        if len(ended):
            env.reset(env_ids=ended)
    return succ, fell, crash, min_d * 1000.0, t_end, (z0 - min_z) * 1000.0


def fitness(succ, crash, md, te):
    return torch.where(succ, 1000.0 - 0.1 * te, -md - 50.0 * crash.float())


def two_phase_seeds(n):
    pats = torch.tensor([[-1, 1], [1, -1], [-1, -1], [1, 1], [0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]],
                        dtype=torch.float32, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(1)
    out = []
    for _ in range(n):
        p1, p2 = pats[torch.randint(len(pats), (2,), generator=g)]
        sw = int(torch.randint(1, K, (1,), generator=g))
        out.append(torch.stack([p1 if k < sw else p2 for k in range(K)]).reshape(-1))
    return torch.stack(out)


def groups_for(N, G):
    gidx = torch.arange(G, device=DEV).repeat_interleave(N // G)
    return torch.cat([gidx, torch.arange(N - len(gidx), device=DEV) % G])


meta = {"segments": K, "seg_len": SEG, "horizon": T}
G = len(TARGETS)

if a.mode in ("hold", "search"):
    env = make_env(a.num_envs)
    vec = RslRlVecEnvWrapper(env, clip_actions=CLIP)
    N = env.num_envs
    gidx = groups_for(N, G)
    xy = target_xy(gidx)

if a.mode == "hold":
    succ, fell, crash, md, te, drop = rollout(env, vec, torch.zeros(N, K, 2, device=DEV), xy, steps=150)
    print("robot held still for 1.5 s: does the egg stay on its pedestal?")
    for g, (name, r, th) in enumerate(TARGETS):
        m = gidx == g
        ok = (~fell[m] & ~crash[m] & (drop[m] < 5)).float().mean()
        print(f"  {name:9s} r {r:3.0f} mm  theta {th:3.0f} deg | stays {float(ok):4.0%} | "
              f"egg-bucket distance {float(md[m].min()):5.0f} mm")

elif a.mode == "search":
    print(f"{G} targets x {N // G} tries each per iteration, motion = {K} x {SEG} steps, {a.iters} iterations", flush=True)
    mu = torch.zeros(G, D, device=DEV)
    sig = torch.full((G, D), a.sigma, device=DEV)
    seeds = two_phase_seeds(N // G // 3)
    best = [{"F": -1e9} for _ in range(G)]
    t_all = time.time()
    for it in range(a.iters):
        x = mu[gidx] + sig[gidx] * torch.randn(N, D, device=DEV)
        if it == 0:
            for g in range(G):                       # same hand-made starting ideas for every target
                rows = torch.nonzero(gidx == g).flatten()[: len(seeds)]
                x[rows] = seeds[: len(rows)]
        x = x.clamp(-1, 1)
        t0 = time.time()
        succ, fell, crash, md, te, _ = rollout(env, vec, x.view(N, K, 2), xy)
        F = fitness(succ, crash, md, te)
        parts = []
        for g in range(G):
            m = gidx == g
            Fg, xg = F[m], x[m]
            top = torch.topk(Fg, max(2, int(a.elite * len(Fg)))).indices
            el = xg[top]
            mu[g] = 0.3 * mu[g] + 0.7 * el.mean(0)
            sig[g] = (0.3 * sig[g] + 0.7 * el.std(0)).clamp(min=0.05)
            if float(Fg[top[0]]) > best[g]["F"]:
                best[g] = {"F": float(Fg[top[0]]), "motion": xg[top[0]].tolist(),
                           "success": bool(succ[m][top[0]]), "min_dist_mm": float(md[m][top[0]])}
            parts.append(f"{TARGETS[g][0]} {int(succ[m].sum())}/{int(m.sum())} ({min(best[g]['min_dist_mm'], 999):.0f}mm)")
        print(f"iter {it + 1:2d}/{a.iters} | {time.time() - t0:4.0f}s | " + " | ".join(parts), flush=True)
        json.dump({**meta, "targets": TARGETS, "iter": it + 1, "best": best}, open(f"{a.out}/search_progress.json", "w"))
    print(f"\nsearch done in {(time.time() - t_all) / 60:.0f} min. Re-checking each target's best motion "
          f"(half the envs exact, half with noise 0.05)...", flush=True)
    xb = torch.stack([torch.tensor(best[g]["motion"], device=DEV) for g in range(G)])[gidx]
    noisy = torch.arange(N, device=DEV) % 2 == 1
    xr = torch.where(noisy[:, None], (xb + 0.05 * torch.randn_like(xb)).clamp(-1, 1), xb)
    succ, fell, crash, md, te, _ = rollout(env, vec, xr.view(N, K, 2), xy)
    res = []
    print("\ntarget      r    theta | best try reached | exact success | noisy success | fell")
    for g, (name, r, th) in enumerate(TARGETS):
        m = gidx == g
        ex = float(succ[m & ~noisy].float().mean()); nz = float(succ[m & noisy].float().mean())
        fl = float(fell[m].float().mean())
        reached = "IN BUCKET" if best[g]["success"] else f"{best[g]['min_dist_mm']:.0f} mm away"
        print(f"{name:9s} {r:4.0f} {th:6.0f}  | {reached:>16s} | {ex:13.0%} | {nz:13.0%} | {fl:4.0%}")
        res.append({"name": name, "r_mm": r, "theta_deg": th, **best[g], "exact_success": ex,
                    "noisy_success": nz, "fell": fl})
    json.dump({**meta, "results": res}, open(f"{a.out}/left_search.json", "w"), indent=1)
    print("saved", f"{a.out}/left_search.json")

else:  # video
    from mjlab.utils.wrappers import VideoRecorder
    res = {r["name"]: r for r in json.load(open(f"{a.out}/left_search.json"))["results"]}
    tg = res[a.video_target]
    TARGETS = [(tg["name"], tg["r_mm"], tg["theta_deg"])]
    base = make_env(1, render=True)
    vdir = f"{a.out}/videos"
    rec = VideoRecorder(base, video_folder=vdir, episode_trigger=lambda e: e == 0,
                        video_length=None, name_prefix=f"motion_{a.video_target}")
    vec = RslRlVecEnvWrapper(rec, clip_actions=CLIP)
    x = torch.tensor(tg["motion"], device=DEV).view(1, K, 2)
    succ, fell, crash, md, te, _ = rollout(base, vec, x, target_xy(torch.zeros(1, dtype=torch.long, device=DEV)))
    print(f"video run {a.video_target}: success={bool(succ[0])} fell={bool(fell[0])} "
          f"closest={float(md[0]):.1f} mm at step {int(te[0])}")
    vec.close()
    print("video folder:", vdir, os.listdir(vdir) if os.path.isdir(vdir) else "(missing)")
