"""Compatibility check for a (new) robot model, before any training. Takes ~2-5 minutes.

Run it from sir's repo folder, on the branch / model you want to test:
    uv run python <this repo>/tools/check_model.py --teacher <teacher.pt> [--student <student.pt>]

It checks, on the C1 task (125-185 mm, 51-80 deg):
  1. observation and action sizes (teacher expects 93 inputs, 2 outputs)
  2. the teacher checkpoint loads into the new model with matching shapes
  3. the egg stays on its pedestal when the robot is held still
  4. the sensors respond when the arm moves: cable lengths, arm keypoints, touch sensors
  5. (optional) the student checkpoint's input layout matches
Nothing is trained and no files are changed.
"""
import os, sys, argparse
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tensordict import TensorDict
import polar_tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-SpiRob-Polar-C1")
ap.add_argument("--teacher", default="")
ap.add_argument("--student", default="")
ap.add_argument("--num_envs", type=int, default=64)
a = ap.parse_args()
DEV = "cuda:0"
# teacher observation layout (93): tendon_len 0:2 | tendon_vel 2:4 | egg_to_bucket 4:7 |
# last_action 7:9 | robot_keypoints_xy 9:51 | touch 51:93
SL = {"cable lengths": slice(0, 2), "arm keypoints (21 x 2)": slice(9, 51), "touch sensors (42)": slice(51, 93)}
results = []


def report(ok, name, detail):
    tag = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    results.append(tag)
    print(f"[{tag}] {name}: {detail}", flush=True)


cfg = load_env_cfg(a.task, play=False)
cfg.scene.num_envs = a.num_envs
cfg.auto_reset = False
cfg.seed = 1001
cfg.observations["actor"].enable_corruption = False
cfg.observations["critic"].enable_corruption = False
env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(a.task).clip_actions)
N = env.num_envs

# 1. sizes
vec.reset()
obs = env.observation_manager.compute(update_history=True)
n_actor, n_critic = obs["actor"].shape[-1], obs["critic"].shape[-1]
n_act = getattr(vec, "num_actions", None)
report(n_actor == 93, "teacher inputs", f"{n_actor} numbers (expected 93); critic {n_critic}")
report(n_act == 2 if n_act is not None else None, "actions", f"{n_act} cable commands (expected 2)")
try:
    terms = env.observation_manager.active_terms["actor"]
    dims = env.observation_manager.group_obs_term_dim["actor"]
    print("       actor input terms:", ", ".join(f"{t} {tuple(d)}" for t, d in zip(terms, dims)))
except Exception:
    pass

# 2. teacher checkpoint
if a.teacher:
    try:
        runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
        runner = runner_cls(vec, asdict(load_rl_cfg(a.task)), device=DEV)
        runner.load(a.teacher, load_cfg={"actor": True}, strict=True, map_location=DEV)
        report(True, "teacher checkpoint", f"loads with matching shapes ({os.path.basename(a.teacher)})")
    except Exception as e:
        report(False, "teacher checkpoint", f"does not load: {str(e).splitlines()[0][:200]}")


def run(actions, steps):
    """Constant per-env actions; returns per-step actor observations and egg heights."""
    vec.reset()
    o = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[N])
    hist, z = [o["actor"].clone()], [env.scene["egg"].data.root_link_pos_w[:, 2].clone()]
    for _ in range(steps):
        o, _, dones, _ = vec.step(actions)
        hist.append(o["actor"].clone()); z.append(env.scene["egg"].data.root_link_pos_w[:, 2].clone())
        ended = torch.nonzero(dones.bool()).flatten()
        if len(ended):
            env.reset(env_ids=ended)
    return torch.stack(hist), torch.stack(z)


# 3. egg stays on its pedestal (robot held still for 1.5 s)
_, z = run(torch.zeros(N, 2, device=DEV), 150)
drop_mm = (z[0] - z.min(0).values) * 1000
stable = float((drop_mm < 5).float().mean())
report(stable >= 0.95, "egg on pedestal", f"stays in {stable:.0%} of spawns (largest drop {float(drop_mm.max()):.0f} mm)")

# 4. sensors respond while the arm moves (four cable patterns, 3 s each)
pats = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]], device=DEV)
H, _ = run(pats[torch.arange(N, device=DEV) % 4], 300)
for name, s in SL.items():
    span = (H[:, :, s].amax(0) - H[:, :, s].amin(0))          # (N, channels): how much each reading changed
    moving = int((span.amax(0) > 1e-6).sum())
    total = span.shape[-1]
    if "touch" in name:
        report(True if moving > 0 else None, name, f"{moving}/{total} sensors registered contact at least once "
               "(0: no contact in this test, or the touch sites do not sit on the new surfaces -- check the XML)")
    else:
        report(moving == total, name, f"{moving}/{total} readings change when the arm moves")
nan = bool(torch.isnan(H).any())
report(not nan, "numerical stability", "no NaN in observations" if not nan else "NaN found in observations")

# 5. student layout
if a.student:
    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    keep, cam = ck.get("keep"), ck.get("cam")
    ok = keep is not None and max(keep) < n_actor
    report(ok, "student checkpoint", f"uses {len(keep)} of the teacher inputs, camera {tuple(cam)} -> "
           f"{'compatible' if ok else 'input layout changed'}")

print("\nSUMMARY:", f"{results.count('PASS')} pass, {results.count('WARN')} warn, {results.count('FAIL')} fail")
print("All good: go on to the zero-shot evaluation." if "FAIL" not in results else
      "Something failed: fix it before training (send the lines above).")
