"""Camera student for SpiRob: DAgger from a teacher policy, evaluation, and camera ablation tests.

Modes: check (camera images), train (DAgger), eval (student alone, optionally with a broken image:
black / frozen / swap / farswap). The student sees an overhead camera image plus the teacher's
inputs except the egg position (and optionally the arm keypoints)."""
import sys, os, json, time, math, argparse
from dataclasses import asdict, dataclass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
import polar_tasks  # noqa: F401  (registers the polar tasks)
from polar_tasks import polar_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["check", "train", "eval"], required=True)
ap.add_argument("--out", required=True, help="Drive folder for students, logs, pictures")
ap.add_argument("--task", default="Mjlab-SpiRob-Polar-Arc15", help="polar task (spawn ranges + teacher runner cfg)")
ap.add_argument("--teacher", default="", help="teacher checkpoint (train mode)")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--betas", default="1,0.5,0.3,0.2,0.1,0", help="share of envs driven by the teacher, per round (last value repeats)")
ap.add_argument("--rec_every", type=int, default=3, help="record every k-th step")
ap.add_argument("--max_samples", type=int, default=120000, help="dataset cap (GPU memory; 150000 ran out on a T4)")
ap.add_argument("--train_steps", type=int, default=1500)
ap.add_argument("--first_train_steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--no_keypoints", action="store_true", help="also hide the arm-shape keypoints from the student")
ap.add_argument("--resume", action="store_true", help="train: continue from student_last.pt in --out")
ap.add_argument("--init", default="", help="train: start a NEW run from these student weights")
ap.add_argument("--student", default="", help="eval: student checkpoint")
ap.add_argument("--label", default="student")
ap.add_argument("--ablate", choices=["none", "black", "frozen", "swap", "farswap"], default="none")
ap.add_argument("--eval_rounds", type=int, default=2)
ap.add_argument("--cam_w", type=int, default=80)
ap.add_argument("--cam_h", type=int, default=60)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda:0"
CAM = "overhead_cam"
GROUP = {0: "nominal", 1: "retention", 2: "sector"}
torch.manual_seed(0)

# teacher actor observation layout (93): tendon_len 0:2 | tendon_vel 2:4 | egg_to_bucket 4:7 |
# last_action 7:9 | robot_keypoints_xy 9:51 | touch 51:93
EGG = slice(4, 7)
HIDE = set(range(4, 7)) | (set(range(9, 51)) if a.no_keypoints else set())
KEEP = [i for i in range(93) if i not in HIDE]

@dataclass
class RobotCamCfg(CameraSensorCfg):
    """Camera fixed to the robot's base frame, so every parallel env gets its own view."""
    def build(self):
        return RobotCam(self)


class RobotCam(CameraSensor):
    def edit_spec(self, scene_spec, entities):
        names = [b.name for b in scene_spec.bodies]
        parent = next((n for n in ("robot/mocap_base", "robot/robot_base") if n in names), None)
        if parent is None:
            raise RuntimeError(f"no robot base body found; bodies: {names[:15]}")
        print(f"[camera] attached to body '{parent}'", flush=True)
        self.cfg.parent_body = parent
        super().edit_spec(scene_spec, entities)


def make_env(n):
    r, th = polar_tasks.POLAR_TASKS[a.task]
    cfg = polar_env_cfg(r, th, play=False)
    # straight down, centred over the bucket + spawn arc; ~4.5 mm per pixel at egg height
    cam = RobotCamCfg(name=CAM, pos=(0.06, 0.135, 0.50), quat=(1.0, 0.0, 0.0, 0.0), fovy=34.0,
                      width=a.cam_w, height=a.cam_h, data_types=("rgb",))
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (cam,)
    cfg.scene.num_envs = n
    cfg.auto_reset = False
    cfg.seed = 1001 if a.mode == "eval" else int(time.time()) % 100000
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
    vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(a.task).clip_actions)
    return env, vec


def image(env):
    return env.scene[CAM].data.rgb.permute(0, 3, 1, 2).contiguous()      # (N, 3, H, W) uint8


def flags(env):
    tm = env.termination_manager
    g = lambda n: tm.get_term(n).bool()
    return g("success_egg_inside_bucket"), g("egg_fell") | g("egg_oob"), g("nan_state")


def spawn_info(env):
    xy = (env.scene["egg"].data.root_link_pos_w - env.scene.env_origins)[:, :2].clone()
    grp = getattr(env, "_stage2_spawn_group", torch.full((env.num_envs,), -1, device=DEV)).clone()
    return xy, grp


def load_teacher(vec, path):
    runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
    runner = runner_cls(vec, asdict(load_rl_cfg(a.task)), device=DEV)
    runner.load(path, load_cfg={"actor": True}, strict=True, map_location=DEV)
    return runner.get_inference_policy(device=DEV)


class SpatialSoftmax(nn.Module):
    """Turns each feature map into the (x, y) image position where it is most active."""
    def __init__(self, h, w):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1))
        self.register_buffer("ys", ys.reshape(-1))
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, f):
        B, K, h, w = f.shape
        p = torch.softmax(f.reshape(B, K, h * w) / self.log_temp.exp(), -1)
        return torch.cat([(p * self.xs).sum(-1), (p * self.ys).sum(-1)], -1)


class Student(nn.Module):
    def __init__(self, n_prop, h, w, n_kp=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ELU())
        self.kp = nn.Conv2d(64, n_kp, 3, 1, 1)
        with torch.no_grad():
            fh, fw = self.conv(torch.zeros(1, 3, h, w)).shape[-2:]
        self.ssm = SpatialSoftmax(fh, fw)
        n_img = 2 * n_kp + 64
        self.mlp = nn.Sequential(nn.Linear(n_img + n_prop, 256), nn.ELU(),
                                 nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 2))
        self.aux = nn.Sequential(nn.Linear(n_img, 128), nn.ELU(), nn.Linear(128, 3))   # egg position from image only
        self.register_buffer("p_mean", torch.zeros(n_prop))
        self.register_buffer("p_std", torch.ones(n_prop))

    def features(self, img_u8):
        f = self.conv(img_u8.float() / 255.0 - 0.5)
        return torch.cat([self.ssm(self.kp(f)), f.mean((-2, -1))], -1)

    def forward(self, img_u8, prop):
        z = self.features(img_u8)
        p = (prop - self.p_mean) / self.p_std
        return self.mlp(torch.cat([z, p], -1)), self.aux(z)


def act_loss(pred, tgt):
    """Actions beyond +-1 all mean 'cable at its limit'. Match the effective command,
    plus a small pull toward the teacher's raw value (it feeds back as 'last action')."""
    t = tgt.clamp(-1, 1)
    err = (pred - t) ** 2
    same_side = ((t >= 1 - 1e-4) & (pred >= 1)) | ((t <= -1 + 1e-4) & (pred <= -1))
    return torch.where(same_side, torch.zeros_like(err), err).mean() + 0.02 * F.smooth_l1_loss(pred, tgt)


def run_round(env, vec, teacher, student, beta, record, ablate="none"):
    N = env.num_envs
    vec.reset()
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[N])
    xy, grp = spawn_info(env)
    by_teacher = torch.rand(N, device=DEV) < beta
    first_img = image(env).clone()
    # image-swap tests: each robot is shown the camera image of ANOTHER robot (its egg is elsewhere)
    if ablate == "farswap":
        src_idx = torch.cdist(xy, xy).argmax(dim=1)            # the robot whose egg is farthest away
    else:
        src_idx = (torch.arange(N, device=DEV) - N // 2) % N   # a fixed other robot
    pair_dist = float((xy - xy[src_idx]).norm(dim=-1).mean() * 1000)
    T = int(env.max_episode_length) + 1
    done = torch.zeros(N, dtype=torch.bool, device=DEV)
    succ, fell, crash = done.clone(), done.clone(), done.clone()
    buf = {"img": [], "prop": [], "act": [], "egg": []}
    egg_err = []
    for t in range(T):
        img = image(env)
        if ablate == "black":
            img = torch.zeros_like(img)
        elif ablate == "frozen":
            img = first_img
        elif ablate in ("swap", "farswap"):
            img = img[src_idx]
        ob = obs["actor"]
        with torch.no_grad():
            a_t = teacher(obs) if teacher is not None else None
            if student is not None:
                a_s, egg_hat = student(img, ob[:, KEEP])
                egg_err.append(((egg_hat / 10 - ob[:, EGG]).norm(dim=-1))[~done])
            act = a_s if a_t is None else (a_t if student is None else torch.where(by_teacher[:, None], a_t, a_s))
        if record and t % a.rec_every == 0:
            m = ~done
            buf["img"].append(img[m].clone()); buf["prop"].append(ob[m][:, KEEP].clone())
            buf["act"].append(a_t[m].clone()); buf["egg"].append(ob[m][:, EGG].clone() * 10)
        obs, _, dones, _ = vec.step(act)
        s, f, n = flags(env)
        new = ~done & (s | f | n)
        succ |= new & s; fell |= new & f & ~s; crash |= new & n & ~s
        done |= new
        if bool(done.all()):
            break
        ended = torch.nonzero(dones.bool()).flatten()
        if len(ended):
            env.reset(env_ids=ended)
    data = {k: torch.cat(v) for k, v in buf.items()} if record else None
    res = dict(xy=xy.cpu().numpy(), group=grp.cpu().numpy(), by_teacher=by_teacher.cpu().numpy(),
               success=succ.cpu().numpy(), fell=fell.cpu().numpy(), crash=crash.cpu().numpy(),
               egg_err_mm=float(torch.cat(egg_err).mean() * 1000) if egg_err else float("nan"),
               pair_dist_mm=pair_dist)
    return res, data


def table(R, mask, title):
    th = np.degrees(np.arctan2(R["xy"][:, 1], R["xy"][:, 0]))
    out = {"groups": {}, "by_angle": []}
    print(f"  {title}")
    for g, name in GROUP.items():
        m = mask & (R["group"] == g)
        if m.any():
            out["groups"][name] = {"n": int(m.sum()), "success": float(R["success"][m].mean()), "fell": float(R["fell"][m].mean())}
            print(f"    {name:9s}: success {R['success'][m].mean():5.0%}  fell {R['fell'][m].mean():5.0%}  (n={m.sum()})")
    sec = mask & (R["group"] == 2)
    for lo in range(30, 150, 10):
        m = sec & (th >= lo) & (th < lo + 10)
        if m.any():
            out["by_angle"].append({"theta": lo, "n": int(m.sum()), "success": float(R["success"][m].mean())})
            print(f"      sector {lo:3d}-{lo + 10:3d} deg: success {R['success'][m].mean():5.0%}  (n={m.sum()})")
    return out

if a.mode == "check":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env, vec = make_env(8)
    vec.reset()
    img = image(env).permute(0, 2, 3, 1).cpu().numpy()
    xy, grp = spawn_info(env)
    obs = env.observation_manager.compute(update_history=True)
    print("camera image batch:", tuple(img.shape), "dtype", img.dtype, "| pixel range", img.min(), "-", img.max())
    print("teacher observation size:", obs["actor"].shape[-1], "| student sees", len(KEEP), "numbers + the image")
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for i, ax in enumerate(axs.flat):
        x, y = xy[i].tolist()
        ax.imshow(img[i], interpolation="nearest")
        ax.set_title(f"egg r={math.hypot(x, y) * 1000:.0f} mm, th={math.degrees(math.atan2(y, x)):.0f} deg", fontsize=9)
        ax.axis("off")
    plt.suptitle("overhead camera: what the student sees (each tile = one env)")
    plt.tight_layout()
    f = f"{a.out}/camera_check.png"
    plt.savefig(f, dpi=110)
    print("saved", f)

elif a.mode == "train":
    env, vec = make_env(a.num_envs)
    H, W = a.cam_h, a.cam_w
    assert a.teacher and os.path.isfile(a.teacher), "give --teacher"
    teacher = load_teacher(vec, a.teacher)
    student = Student(len(KEEP), H, W).to(DEV)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    betas = [float(b) for b in a.betas.split(",")]
    log_path = f"{a.out}/train_log.json"
    log, start = [], 0
    if a.resume and os.path.isfile(f"{a.out}/student_last.pt"):
        ck = torch.load(f"{a.out}/student_last.pt", map_location=DEV, weights_only=False)
        student.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["round"] + 1
        log = json.load(open(log_path)) if os.path.isfile(log_path) else []
        print(f"resumed after round {ck['round']} (dataset starts empty again)", flush=True)
    if a.init and not a.resume:
        ck0 = torch.load(a.init, map_location=DEV, weights_only=False)
        assert ck0["keep"] == KEEP and tuple(ck0["cam"]) == (H, W), "init student has a different input layout"
        student.load_state_dict(ck0["model"])          # weights + its input normalisation
        print(f"student init from {a.init}", flush=True)
    D = None
    print(f"student input: {H}x{W} image + {len(KEEP)} numbers (hidden: egg position"
          f"{', arm keypoints' if a.no_keypoints else ''}) | teacher: {a.teacher}", flush=True)
    for rd in range(start, a.rounds):
        beta = betas[min(rd, len(betas) - 1)]
        t0 = time.time()
        R, data = run_round(env, vec, teacher, student if (rd > 0 or start > 0 or a.init) else None, beta, record=True)
        t_roll = time.time() - t0
        # grow the dataset (random thinning when it passes the cap)
        D = data if D is None else {k: torch.cat([D[k], data[k]]) for k in D}
        n = len(D["act"])
        if n > a.max_samples:
            keep = torch.randperm(n, device=DEV)[: a.max_samples]
            D = {k: v[keep] for k, v in D.items()}
            n = a.max_samples
        if rd == start and not a.resume and not a.init:
            student.p_mean.copy_(D["prop"].mean(0)); student.p_std.copy_(D["prop"].std(0).clamp(min=1e-2))
        # train
        steps = a.first_train_steps if rd == start else a.train_steps
        t1, la, le = time.time(), [], []
        student.train()
        for s in range(steps):
            i = torch.randint(n, (a.batch,), device=DEV)
            pred, egg_hat = student(D["img"][i], D["prop"][i])
            l_act = act_loss(pred, D["act"][i])
            l_egg = F.mse_loss(egg_hat, D["egg"][i])
            loss = l_act + l_egg
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
            la.append(l_act.item()); le.append(l_egg.item())
        student.eval()
        egg_mm = math.sqrt(np.mean(le[-200:]) * 3) * 100      # rms egg-position error of the image head, mm
        stu = ~R["by_teacher"]
        row = {"round": rd, "beta": beta, "samples": n, "rollout_s": round(t_roll), "train_s": round(time.time() - t1),
               "action_loss": float(np.mean(la[-200:])), "egg_err_train_mm": egg_mm,
               "teacher_success": float(R["success"][R["by_teacher"]].mean()) if R["by_teacher"].any() else None,
               "student_success": float(R["success"][stu].mean()) if (stu.any() and (rd > 0 or bool(a.init))) else None,
               "student_egg_err_mm": R["egg_err_mm"]}
        log.append(row)
        ts = "-" if row["teacher_success"] is None else f"{row['teacher_success']:.0%}"
        ss = "-" if row["student_success"] is None else f"{row['student_success']:.0%}"
        print(f"round {rd + 1:2d}/{a.rounds} | teacher share {beta:.0%} | success: teacher-driven {ts:>4}, "
              f"student-driven {ss:>4} | egg found within ~{egg_mm:4.1f} mm | action loss {row['action_loss']:.3f} | "
              f"data {n} | {t_roll:.0f}s + {row['train_s']}s", flush=True)
        ck = {"model": student.state_dict(), "opt": opt.state_dict(), "round": rd, "keep": KEEP,
              "cam": (H, W), "no_keypoints": a.no_keypoints, "teacher": a.teacher}
        torch.save(ck, f"{a.out}/student_round{rd + 1:02d}.pt")
        torch.save(ck, f"{a.out}/student_last.pt")
        json.dump(log, open(log_path, "w"), indent=1)
    print("saved", f"{a.out}/student_last.pt")

else:  # eval
    assert a.student and os.path.isfile(a.student), "give --student"
    ck = torch.load(a.student, map_location=DEV, weights_only=False)
    assert ck["keep"] == KEEP, "use the same --no_keypoints setting as in training"
    env, vec = make_env(a.num_envs)
    student = Student(len(KEEP), *ck["cam"]).to(DEV)
    student.load_state_dict(ck["model"]); student.eval()
    rows = []
    for i in range(a.eval_rounds):
        t0 = time.time()
        R, _ = run_round(env, vec, None, student, 0.0, record=False, ablate=a.ablate)
        rows.append(R)
        print(f"round {i + 1}/{a.eval_rounds}: {time.time() - t0:.0f}s, success {R['success'].mean():.0%}", flush=True)
    R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0] if k not in ("egg_err_mm", "pair_dist_mm")}
    err = float(np.mean([r["egg_err_mm"] for r in rows]))
    print(f"\n{a.label} on {a.task} (image: {a.ablate}) | camera egg-position error {err:.1f} mm")
    out = table(R, np.ones(len(R["success"]), bool), "student alone")
    out.update({"label": a.label, "ablate": a.ablate, "egg_err_mm": err, "student": a.student,
                "pair_dist_mm": float(np.mean([r["pair_dist_mm"] for r in rows]))})
    f = f"{a.out}/eval_{a.label}_{a.ablate}.json"
    json.dump(out, open(f, "w"), indent=1)
    print("saved", f)