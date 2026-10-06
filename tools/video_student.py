"""Video of the camera student: same egg positions, camera ON (top row) vs camera OFF (bottom row).

--mode preview : one still frame from the video camera (check the viewing angle)
--mode video   : runs the student at several egg positions (a few tries each), picks for every
                 position a try where the camera-ON student succeeded, and writes
                 student_video.mp4 + student_keyframes.png to --out.
"""
import sys, os, math, time, argparse, subprocess
from dataclasses import dataclass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
import polar_tasks  # noqa: F401
from polar_tasks import polar_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.registry import load_rl_cfg
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["preview", "video"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--student", default="")
ap.add_argument("--task", default="Mjlab-SpiRob-Polar-Arc15")
ap.add_argument("--positions", default="158:72,165:64,152:60,170:56", help="r_mm:theta_deg list")
ap.add_argument("--tries", type=int, default=4, help="attempts per position (small jitter each)")
ap.add_argument("--vid_pos", default="0.26,-0.10,0.30", help="video camera position (robot base frame, m)")
ap.add_argument("--vid_target", default="0.0,0.14,0.07", help="point the video camera looks at")
ap.add_argument("--vid_fovy", type=float, default=50.0)
ap.add_argument("--vid_w", type=int, default=256)
ap.add_argument("--vid_h", type=int, default=192)
ap.add_argument("--every", type=int, default=3, help="record every k-th control step")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "cuda:0"
CAM, VID = "overhead_cam", "video_cam"


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



def look_at_quat(pos, target):
    """MuJoCo camera looks along its -z axis with +y up; build that frame and return (w, x, y, z)."""
    pos, target = np.array(pos, float), np.array(target, float)
    f = target - pos; f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    z = -f; x = right; y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def make_env(n, cam_hw):
    r, th = polar_tasks.POLAR_TASKS[a.task]
    cfg = polar_env_cfg(r, th, play=False)
    vp = [float(v) for v in a.vid_pos.split(",")]
    vt = [float(v) for v in a.vid_target.split(",")]
    cams = (RobotCamCfg(name=CAM, pos=(0.06, 0.135, 0.50), quat=(1.0, 0.0, 0.0, 0.0), fovy=34.0,
                        width=cam_hw[1], height=cam_hw[0], data_types=("rgb",)),
            RobotCamCfg(name=VID, pos=tuple(vp), quat=look_at_quat(vp, vt), fovy=a.vid_fovy,
                        width=a.vid_w, height=a.vid_h, data_types=("rgb",)))
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + cams
    cfg.scene.num_envs = n
    cfg.auto_reset = False
    cfg.seed = 7
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
    vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(a.task).clip_actions)
    return env, vec


def rgb(env, name):
    return env.scene[name].data.rgb.permute(0, 3, 1, 2).contiguous()


def place(env, xy):
    """Put egg AND (mocap) pedestal at local xy (N, 2), then re-render the cameras."""
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
    env.sim.sense()
    for s in env.scene.sensors.values():
        if hasattr(s, "_invalidate_cache"):
            s._invalidate_cache()


def flags(env):
    tm = env.termination_manager
    g = lambda n: tm.get_term(n).bool()
    return g("success_egg_inside_bucket"), g("egg_fell") | g("egg_oob") | g("nan_state")


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



def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.isfile(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


if a.mode == "preview":
    env, vec = make_env(2, (60, 80))
    vec.reset()
    v = rgb(env, VID)[0].permute(1, 2, 0).cpu().numpy()
    o = rgb(env, CAM)[0].permute(1, 2, 0).cpu().numpy()
    big = Image.fromarray(v).resize((a.vid_w * 2, a.vid_h * 2), Image.NEAREST)
    big.paste(Image.fromarray(o).resize((160, 120), Image.NEAREST), (6, 6))
    big.save(f"{a.out}/video_preview.png")
    print("saved", f"{a.out}/video_preview.png")
    sys.exit(0)

# ---------------------------------------------------------------- video
ck = torch.load(a.student, map_location=DEV, weights_only=False)
H, W = ck["cam"]
KEEP = ck["keep"]
student = Student(len(KEEP), H, W).to(DEV)
student.load_state_dict(ck["model"]); student.eval()

pos = [tuple(float(v) for v in p.split(":")) for p in a.positions.split(",")]
K, Tn = len(pos), a.tries
N = 2 * K * Tn                        # [camera ON: K x tries] + [camera OFF: same positions]
base = torch.tensor([[r / 1000 * math.cos(math.radians(t)), r / 1000 * math.sin(math.radians(t))] for r, t in pos], device=DEV)
g = torch.Generator(device=DEV).manual_seed(0)
xy_half = base.repeat_interleave(Tn, 0) + (torch.rand(K * Tn, 2, device=DEV, generator=g) * 2 - 1) * 0.002
xy = torch.cat([xy_half, xy_half])
blind = torch.zeros(N, dtype=torch.bool, device=DEV); blind[K * Tn:] = True

env, vec = make_env(N, (H, W))
vec.reset()
place(env, xy)
obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[N])
T = int(env.max_episode_length) + 1
done = torch.zeros(N, dtype=torch.bool, device=DEV)
succ, fell = done.clone(), done.clone()
frames, views, status = [], [], []                 # per recorded step: (N,3,h,w) uint8 on GPU, status codes
last_v, last_o = rgb(env, VID).clone(), rgb(env, CAM).clone()
t0 = time.time()
for t in range(T):
    img = rgb(env, CAM)
    img = torch.where(blind[:, None, None, None], torch.zeros_like(img), img)
    if t % a.every == 0:
        v = rgb(env, VID)
        last_v = torch.where(done[:, None, None, None], last_v, v)
        last_o = torch.where(done[:, None, None, None], last_o, img)
        frames.append(last_v.clone()); views.append(last_o.clone())
        status.append((succ.int() - fell.int()).clone())
    with torch.no_grad():
        act, _ = student(img, obs["actor"][:, KEEP])
    obs, _, dones, _ = vec.step(act)
    s, f = flags(env)
    new = ~done & (s | f)
    succ |= new & s; fell |= new & f & ~s
    done |= new | dones.bool()          # time-outs also end the clip (frame is frozen)
    if bool(done.all()):
        break
    ended = torch.nonzero(dones.bool()).flatten()
    if len(ended):
        env.reset(env_ids=ended)
for _ in range(15):                                  # hold the final frame ~0.5 s
    frames.append(last_v.clone()); views.append(last_o.clone()); status.append((succ.int() - fell.int()).clone())
print(f"rolled out {N} episodes in {time.time() - t0:.0f}s", flush=True)

S = succ.cpu().numpy(); Fl = fell.cpu().numpy()
pick = []
for k in range(K):
    on = [k * Tn + j for j in range(Tn)]
    j = next((i for i in on if S[i]), on[0])
    pick.append((j, j + K * Tn))
    print(f"position r={pos[k][0]:.0f} mm, theta={pos[k][1]:.0f} deg | camera ON success {S[on].mean():.0%} "
          f"| camera OFF success {S[[i + K * Tn for i in on]].mean():.0%} | shown: try {j - k * Tn + 1}", flush=True)

TW, TH, HDR, LBL = a.vid_w, a.vid_h, 44, 150
VW, VH = LBL + K * TW, HDR + 2 * TH
VW += VW % 2; VH += VH % 2
F1, F2, F3 = font(20), font(14), font(12)
COL = {1: (40, 170, 70), -1: (210, 50, 40), 0: (255, 255, 255)}


def compose(i):
    canvas = Image.new("RGB", (VW, VH), (18, 22, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 10), "Camera student (DAgger): same egg positions, camera ON vs camera OFF", fill=(235, 235, 235), font=F1)
    for row, label in ((0, "Camera ON"), (1, "Camera OFF\n(black image)")):
        d.multiline_text((10, HDR + row * TH + TH // 2 - 14), label, fill=(235, 235, 235), font=F2)
    st = status[i].cpu().numpy()
    for k, (jon, joff) in enumerate(pick):
        for row, j in ((0, jon), (1, joff)):
            x0, y0 = LBL + k * TW, HDR + row * TH
            tile = Image.fromarray(frames[i][j].permute(1, 2, 0).cpu().numpy())
            inset = Image.fromarray(views[i][j].permute(1, 2, 0).cpu().numpy()).resize((80, 60), Image.NEAREST)
            tile.paste(inset, (4, 4))
            td = ImageDraw.Draw(tile)
            td.rectangle([3, 3, 84, 64], outline=(255, 255, 255))
            c = COL[int(st[j])]
            td.rectangle([0, 0, TW - 1, TH - 1], outline=c, width=4 if st[j] else 1)
            txt = {1: "DELIVERED", -1: "DROPPED", 0: ""}[int(st[j])]
            if txt:
                td.text((TW - 100, TH - 22), txt, fill=c, font=F2)
            if row == 0:
                td.text((90, 6), f"r={pos[k][0]:.0f} mm  {pos[k][1]:.0f}°", fill=(255, 255, 255), font=F3)
            canvas.paste(tile, (x0, y0))
    return canvas


out_mp4 = f"{a.out}/student_video.mp4"
fps = max(1, int(round(1.0 / (env.step_dt * a.every))))
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                      "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", out_mp4],
                     stdin=subprocess.PIPE)
for i in range(len(frames)):
    p.stdin.write(np.asarray(compose(i)).tobytes())
p.stdin.close(); p.wait()
print(f"saved {out_mp4} ({len(frames)} frames, {fps} fps, {len(frames) / fps:.1f} s)", flush=True)

# key frames for the report: 5 moments of the first shown position
n_live = len(frames) - 15
idx = [int(round(q * (n_live - 1))) for q in (0, 0.25, 0.5, 0.75, 1.0)]
strip = Image.new("RGB", (5 * TW, 2 * TH), (18, 22, 30))
jon, joff = pick[1] if K > 1 else pick[0]
for c, i in enumerate(idx):
    for row, j in ((0, jon), (1, joff)):
        strip.paste(Image.fromarray(frames[i][j].permute(1, 2, 0).cpu().numpy()), (c * TW, row * TH))
strip.save(f"{a.out}/student_keyframes.png")
print("saved", f"{a.out}/student_keyframes.png (top: camera ON, bottom: camera OFF)")
