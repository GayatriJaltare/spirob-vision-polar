# Experiment log (after the v3 ablation failure)

Success = egg inside bucket (strict checker). Nominal egg: (0.05, 0.15) m → r 158 mm, θ 72°. Bucket at (−0.05, 0.15).

## 1. Why v3 failed
- Ablation of v3 (black / frozen / shuffled image, no proprioception): success unchanged → **camera not used**; the policy memorised one open-loop motion (egg always at the same spot).
- Randomised-spawn fine-tuning of v3 (three attempts): best checkpoint 8% (4 of 50).
- v7 (camera + PPO from scratch, randomised egg, 80×60 image, privileged critic): **used its camera** (black image +10.6 mm, shuffled pixels +19.3 mm egg-to-bucket distance, 60 paired episodes) but only ~7% success (4 of 60). v8 (10× reach reward) was worse.
- Conclusion: reward-only RL with a camera is too hard here → split into **privileged teacher + camera student**.

## 2. Reproducing  Stage-1 teacher (R002)
- Retraining from scratch: 0 reward on T4 and Blackwell. confirmed his run succeeded by seed luck.
- Used model_499.pt as the starting point instead (rebuilt from the unpacked checkpoint folder).

## 3. Trajectory search (no learning)
- Nominal spot: open-loop motion with 97% exact success, 8% under action noise; robust search 42% / 26%.
- Egg offsets: blind motion 26% vs position-aware motion 64% → **knowing the egg position is necessary** (motivates vision).

## 4. Workspace map
- At r = 158 mm the arm reaches θ 29–151°; up to ~310 mm around 60–120°; nothing at 0–10° / 170–180°.

## 5. Polar tasks (egg + pedestal spawned at (r, θ); 10% nominal, 30% retention, 60% sector)
- Baseline, model_499 on Wide (30–150°): nominal 78%, retention 63%, sector 3%.

## 6. Polar teacher (PPO fine-tune, Stage2FineTuneRunner, 256 envs, T4)

| Run | Start | Task | Iter | Nominal | Retention | Sector |
|---|---|---|---|---|---|---|
| Stage A | model_499 | Arc-1 (58–74°) | 300 | 96% | 67% | 52% |
| Stage B | A | Arc-2 (44–77°) | 300 | 90% | 73% | 20% (no gain) |
| A15 | A | Arc-1.5 (51–76°) | 500 | 69% | 78% | 39% |
| A15b | A15 | Arc-1.5 | 500 | 82% | 86% | 46% |
| **A15c** | A15b | Arc-1.5 | 500 | **80%** | **87%** | **54%** |

Lesson: widen the arc only where the policy already succeeds sometimes; new positions with ~0% success give no learning signal.

## 7. Demonstrations + behaviour cloning (negative result)
- CEM search at 23 positions: working motions at 55–75°, none at 40–50° (closest 37–100 mm).
- BC on Stage A's own successes: 96% nominal (pipeline correct).
- BC with search motions (all, or only 55–60°): **collapse to 0% nominal**. The search motions are open-loop (timed), so a feedback policy cannot imitate them; mixing two styles averages them.

## 8. Camera student (DAgger)
- Overhead camera 80×60 RGB, attached to the robot base frame, looking straight down from 0.5 m, ~4.5 mm/pixel.
- Student input: image + 90 proprioceptive numbers (tendon length/velocity, last action, touch, arm keypoints). egg_to_bucket hidden.
- Network: 3 conv layers → 32 spatial-softmax points + 64 pooled features → MLP 256-256-2; auxiliary head predicts the egg position (MSE, training only). ~190k weights, trained from scratch.
- DAgger: 256 envs, 10 rounds, teacher share 100 / 50 / 30 / 20 / 10 / 0%, every 3rd step recorded, dataset capped at 150k, ~8 min per round on a T4.
- Per-round success of student-driven robots (rounds 2–9): 56, 63, 62, 55, 60, 61, 58, 64%. Egg error during training: 8.0 → 6.0 mm.

Final evaluation on Arc-1.5, 512 episodes each (teacher = stageA15c):

| Policy / image | Nominal | Retention | Sector | 50–60° | 60–70° | 70–80° | Overall | Egg error |
|---|---|---|---|---|---|---|---|---|
| Teacher (given egg position) | 80% | 87% | 54% | 33% | 69% | 55% | 65% | – |
| Student, normal camera | 86% | 82% | 43% | 29% | 51% | 51% | 57% | 4.5 mm |
| Student, black image | 77% | 56% | 15% | 0% | 17% | 34% | 34% | 61 mm |
| Student, frozen first frame | 23% | 15% | 15% | 8% | 20% | 17% | 16% | 73 mm |
| Student, swapped image | 69% | 60% | 29% | 26% | 29% | 32% | 42% | 47 mm |

- The student reaches 88% of the teacher overall (57 / 65). 86% vs 80% at nominal is within noise (~50 episodes each).
- Black and frozen images show the student depends on its camera away from the nominal spot and throughout the motion.
- The swap test is the weakest: spawns are close together, so another robot's egg is often nearby. A far-swap version is planned.

## 9. Video
- `tools/video_student.py`: four egg positions (72°, 64°, 60°, 56°), camera on vs image blacked out. Chosen examples (best of four tries per position); the rates are the table above.

## 10. Open items
- Student without arm keypoints (`--no_keypoints`), far-swap ablation, more seeds, wider teacher (rest of the right side, then the left side), visual randomisation for sim-to-real.
