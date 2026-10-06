"""Trajectory search teacher for SpiRob Stage-1 (no RL, no neural network).

A "motion" = K segments x 2 cable commands in [-1, 1], each held for SEG control steps.
Every parallel environment tries a different motion; the cross-entropy method (CEM)
keeps the best 10 % and samples around them. Success is sir's own termination check.

Modes
  nominal : find a motion that puts the egg in the bucket from the trained egg spot
  offsets : starting from that motion, find one motion per egg offset (the teacher table)
            and check whether ONE fixed motion already works everywhere
  video   : record a video of a found motion
"""
import sys, os, json, time, argparse
sys.path.insert(0, "/content/spirob-stage1/tools")
import numpy as np
import torch
import evaluate_stage1 as S1
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["nominal", "offsets", "video"], required=True)
ap.add_argument("--out", required=True, help="output folder (Drive)")
ap.add_argument("--num_envs", type=int, default=512)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--segments", type=int, default=10)
ap.add_argument("--seg_len", type=int, default=50, help="control steps per segment (0.01 s each)")
ap.add_argument("--horizon", type=int, default=600, help="control steps per try (6 s)")
ap.add_argument("--sigma", type=float, default=0.6)
ap.add_argument("--elite", type=float, default=0.1)
ap.add_argument("--offsets_mm", default="0:0,5:0,-5:0,0:5,0:-5,10:0,-10:0,0:10,0:-10")
ap.add_argument("--eval_noise", type=float, default=0.0,
                help="robust search: each try is executed with this much extra random noise")
ap.add_argument("--continue_from_nominal", action="store_true",
                help="start the nominal search from the saved nominal.json motion")
ap.add_argument("--video_key", default="nominal", help="'nominal' or an offset like '5:0'")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda:0"
K, SEG, T = a.segments, a.seg_len, a.horizon
D = 2 * K
torch.manual_seed(0)


def make_env(n, render=False):
    cfg = load_env_cfg(S1.TASK_ID, play=False)
    cfg.scene.num_envs = n
    cfg.auto_reset = False
    cfg.seed = S1.S1_EVALUATION_SEED
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    return ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array" if render else None)


def place_eggs(env, off_mm):
    """Same method as sir's evaluator, for all envs at once. off_mm: (N, 2) tensor in mm."""
    egg = env.scene["egg"]
    ids = torch.arange(env.num_envs, device=DEV)
    rs = egg.data.default_root_state.clone()
    rs[:, 0:3] += env.scene.env_origins
    rs[:, 0:2] += off_mm / 1000.0
    egg.write_root_state_to_sim(rs, env_ids=ids)
    env.sim.forward()


def rollout(env, vec, motions, off_mm):
    """motions: (N, K, 2). Returns per-env success, fell, crash, min egg-bucket distance (mm), end step."""
    N = env.num_envs
    vec.reset()
    place_eggs(env, off_mm)
    egg, bucket = env.scene["egg"], env.scene["bucket"]
    sid = bucket.find_sites(("bucket_site",), preserve_order=True)[0]
    tm = env.termination_manager
    done = torch.zeros(N, dtype=torch.bool, device=DEV)
    succ, fell, crash = done.clone(), done.clone(), done.clone()
    min_d = torch.full((N,), float("inf"), device=DEV)
    t_end = torch.full((N,), float(T), device=DEV)
    for t in range(T):
        _, _, dones, _ = vec.step(motions[:, min(t // SEG, K - 1), :])
        d = torch.linalg.norm(egg.data.root_link_pos_w - bucket.data.site_pos_w[:, sid].squeeze(1), dim=-1)
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))
        min_d = torch.where(done, min_d, torch.minimum(min_d, d))
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
        # mjlab 1.4 requires finished envs to be reset before the next step;
        # their result is already recorded above and they stay masked out
        ended = torch.nonzero(dones.bool()).flatten()
        if len(ended):
            env.reset(env_ids=ended)
    return succ, fell, crash, min_d * 1000.0, t_end


def fitness(succ, crash, md, te):
    # success first (earlier is better); otherwise: how close the egg ever got to the bucket
    return torch.where(succ, 1000.0 - 0.1 * te, -md - 50.0 * crash.float())


def two_phase_seeds(n):
    """Hand-made starting ideas: one cable pattern, then a switch to another."""
    pats = torch.tensor([[-1, 1], [1, -1], [-1, -1], [1, 1], [0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]],
                        dtype=torch.float32, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(1)
    out = []
    for _ in range(n):
        p1, p2 = pats[torch.randint(len(pats), (2,), generator=g)]
        sw = int(torch.randint(1, K, (1,), generator=g))
        m = torch.stack([p1 if k < sw else p2 for k in range(K)])
        out.append(m.reshape(-1))
    return torch.stack(out)


def search(env, vec, mu, sig, group_off, iters, seeds=None):
    """CEM with G independent groups (one per egg offset) sharing each batch."""
    G = mu.shape[0]
    N = env.num_envs
    gidx = torch.arange(G, device=DEV).repeat_interleave(N // G)
    gidx = torch.cat([gidx, torch.zeros(N - len(gidx), dtype=torch.long, device=DEV)])
    off = group_off[gidx]
    best = [{"F": -1e9} for _ in range(G)]
    for it in range(iters):
        x = mu[gidx] + sig[gidx] * torch.randn(N, D, device=DEV)
        if it == 0 and seeds is not None:
            x[: len(seeds)] = seeds
        x = x.clamp(-1, 1)
        t0 = time.time()
        x_run = (x + a.eval_noise * torch.randn_like(x)).clamp(-1, 1) if a.eval_noise > 0 else x
        succ, fell, crash, md, te = rollout(env, vec, x_run.view(N, K, 2), off)
        F = fitness(succ, crash, md, te)
        line = []
        for g in range(G):
            m = gidx == g
            Fg, xg = F[m], x[m]
            ne = max(2, int(a.elite * len(Fg)))
            top = torch.topk(Fg, ne).indices
            el = xg[top]
            mu[g] = 0.3 * mu[g] + 0.7 * el.mean(0)
            sig[g] = (0.3 * sig[g] + 0.7 * el.std(0)).clamp(min=0.05)
            if float(Fg[top[0]]) > best[g]["F"]:
                best[g] = {"F": float(Fg[top[0]]), "motion": xg[top[0]].tolist(),
                           "success": bool(succ[m][top[0]]), "min_dist_mm": float(md[m][top[0]])}
            line.append(f"{int(succ[m].sum())}/{int(m.sum())}")
        nb = ~succ
        close = float(md[nb].min()) if bool(nb.any()) else float("nan")
        print(f"iter {it + 1:2d}/{iters} | {time.time() - t0:5.1f}s | successes in batch: "
              f"{' '.join(line)} | fell {int(fell.sum())} crash {int(crash.sum())} | "
              f"closest non-success {close:6.1f} mm | sigma {float(sig.mean()):.2f}", flush=True)
    return mu, sig, best


def success_rate(env, vec, motion, off_mm, noise=0.0):
    N = env.num_envs
    x = torch.tensor(motion, device=DEV).repeat(N, 1)
    if noise > 0:
        x = (x + noise * torch.randn_like(x)).clamp(-1, 1)
    off = torch.tensor(off_mm, dtype=torch.float32, device=DEV).repeat(N, 1)
    succ, fell, crash, md, te = rollout(env, vec, x.view(N, K, 2), off)
    return float(succ.float().mean()), float(fell.float().mean()), float(crash.float().mean())


meta = {"segments": K, "seg_len": SEG, "horizon": T}

if a.mode == "nominal":
    env = make_env(a.num_envs)
    vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(S1.TASK_ID).clip_actions)
    print(f"searching with {a.num_envs} parallel tries per iteration, motion = {K} x {SEG} steps", flush=True)
    mu = torch.zeros(1, D, device=DEV)
    sig = torch.full((1, D), a.sigma, device=DEV)
    seeds = two_phase_seeds(min(128, a.num_envs // 4))
    if a.continue_from_nominal:
        prev = json.load(open(f"{a.out}/nominal.json"))
        json.dump(prev, open(f"{a.out}/nominal_previous.json", "w"), indent=1)
        mu = torch.tensor(prev["motion"], device=DEV).view(1, D)
        seeds = mu.clone()          # keep the known-good motion in the first batch
        print(f"continuing from saved motion (exact {prev.get('exact_success')}, noisy {prev.get('noisy_success')})", flush=True)
    if a.eval_noise > 0:
        print(f"robust search: every try runs with extra noise {a.eval_noise}", flush=True)
    mu, sig, best = search(env, vec, mu, sig, torch.zeros(1, 2, device=DEV), a.iters, seeds)
    cands = {"best_try": best[0]["motion"], "cem_mean": mu[0].tolist()}
    if a.continue_from_nominal:
        cands["start"] = prev["motion"]
    results = {}
    for name, mot in cands.items():
        ex = success_rate(env, vec, mot, [0, 0])
        nz = success_rate(env, vec, mot, [0, 0], noise=0.05)
        results[name] = (ex[0], nz[0], mot)
        print(f"{name:9s} exact {ex[0]:.0%} | noisy {nz[0]:.0%}", flush=True)
    pick = max(results, key=lambda k: (results[k][1], results[k][0]))
    ex, nz, mot = results[pick]
    print("chosen:", pick)
    out = {**meta, "motion": mot, "exact_success": ex}
    out.update({"noisy_success": nz, "chosen": pick})
    json.dump(out, open(f"{a.out}/nominal.json", "w"), indent=1)
    print("saved", f"{a.out}/nominal.json")

elif a.mode == "offsets":
    nom = json.load(open(f"{a.out}/nominal.json"))
    assert (nom["segments"], nom["seg_len"]) == (K, SEG), "use the same --segments/--seg_len as the nominal search"
    offs = [tuple(float(v) for v in s.split(":")) for s in a.offsets_mm.split(",")]
    env = make_env(a.num_envs)
    vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(S1.TASK_ID).clip_actions)
    print("A) does ONE fixed motion (the nominal one) work at every egg offset?", flush=True)
    fixed = {}
    for o in offs:
        r = success_rate(env, vec, nom["motion"], list(o))
        fixed[f"{o[0]:g}:{o[1]:g}"] = r[0]
        print(f"   offset {o[0]:+5.0f},{o[1]:+5.0f} mm -> success {r[0]:.0%}", flush=True)
    print("\nB) searching one motion per offset (the teacher table)", flush=True)
    G = len(offs)
    mu = torch.tensor(nom["motion"], device=DEV).repeat(G, 1)
    sig = torch.full((G, D), 0.25, device=DEV)
    goff = torch.tensor(offs, dtype=torch.float32, device=DEV)
    mu, sig, best = search(env, vec, mu, sig, goff, a.iters)
    table = {}
    print("\noffset (mm)   fixed-motion   own-motion (exact)   own-motion (noisy)")
    for o, b in zip(offs, best):
        key = f"{o[0]:g}:{o[1]:g}"
        ex = success_rate(env, vec, b["motion"], list(o))[0]
        nz = success_rate(env, vec, b["motion"], list(o), noise=0.05)[0]
        table[key] = {**b, "exact_success": ex, "noisy_success": nz, "fixed_motion_success": fixed[key]}
        print(f"{key:>10}   {fixed[key]:>10.0%}   {ex:>16.0%}   {nz:>17.0%}", flush=True)
    json.dump({**meta, "table": table}, open(f"{a.out}/teacher_table.json", "w"), indent=1)
    print("saved", f"{a.out}/teacher_table.json")

else:  # video
    from mjlab.utils.wrappers import VideoRecorder
    if a.video_key == "nominal":
        src = json.load(open(f"{a.out}/nominal.json"))
        motion, off = src["motion"], [0.0, 0.0]
    elif a.video_key in ("previous", "v1"):
        fn = "nominal_previous.json" if a.video_key == "previous" else "nominal_v1_97pct.json"
        src = json.load(open(f"{a.out}/{fn}"))
        motion, off = src["motion"], [0.0, 0.0]
    else:
        tab = json.load(open(f"{a.out}/teacher_table.json"))["table"]
        motion = tab[a.video_key]["motion"]
        off = [float(v) for v in a.video_key.split(":")]
    base = make_env(1, render=True)
    vdir = f"{a.out}/videos"
    name = "motion_" + a.video_key.replace(":", "_").replace("-", "m")
    rec = VideoRecorder(base, video_folder=vdir, episode_trigger=lambda e: e == 0,
                        video_length=None, name_prefix=name)
    vec = RslRlVecEnvWrapper(rec, clip_actions=load_rl_cfg(S1.TASK_ID).clip_actions)
    x = torch.tensor(motion, device=DEV).view(1, K, 2)
    succ, fell, crash, md, te = rollout(base, vec, x, torch.tensor([off], device=DEV))
    print(f"video run: success={bool(succ[0])} fell={bool(fell[0])} crash={bool(crash[0])} "
          f"closest={float(md[0]):.1f} mm at step {int(te[0])}")
    vec.close()
    print("video folder:", vdir, os.listdir(vdir) if os.path.isdir(vdir) else "(missing)")
