# Project history: how the current checkpoints were reached

All numbers are success rates in simulation.
- **nominal** = the egg at the original spot (72°, 158 mm);
- **retention** = small offsets around it;
- **sector** = random spots in the training region.

## 1. Starting point
Sir's Stage-1 policy `model_499` was trained on one fixed egg spot. It succeeds **78 %** at that spot, but only **3 %** across a sector of spawn positions: it learned one motion.

Earlier, the camera policy was trained directly with PPO (8 versions, about 26 configurations); this did not reach reliable success. That led to the teacher–student approach below.

## 2. Teacher: fine-tuning on growing spawn regions
These runs use sir's `Stage2FineTuneRunner` and spawn mix (10 % nominal, 30 % retention, 60 % sector).

| Run | Region | Iterations | Nominal | Retention | Sector |
|---|---|---|---|---|---|
| Stage B | Arc-2 (133–186 mm, 44–77°) | 300 | 90 % | 73 % | 20 % |
| A15 | Arc-1.5 (139–179 mm, 51–76°) | 500 | 69 % | 78 % | 39 % |
| A15b | Arc-1.5 | 500 | 82 % | 86 % | 46 % |
| **A15c** | Arc-1.5 | 500 | 80 % | 87 % | 54 % |

The jump straight to Arc-2 gave no gain: new positions with no success give no reward to learn from. So the region was widened in a smaller step (Arc-1.5).

## 3. Camera student v1 (DAgger from A15c, Arc-1.5)
This was 10 DAgger rounds. Student alone: **86 % nominal, 82 % retention, 43 % sector**, with a camera egg-position error of **4.5 mm**.

Image ablations show that the student uses the camera:

| Image | Nominal | Retention | Sector | Overall | Egg error |
|---|---|---|---|---|---|
| Normal | 86 % | 82 % | 43 % | 57 % | 4.5 mm |
| Swapped | 69 % | 60 % | 29 % | 42 % | 47 mm |
| Black | 77 % | 56 % | 15 % | 34 % | 61 mm |
| Far-swap | 50 % | 43 % | 20 % | 31 % | 50 mm |
| Frozen first frame | 23 % | 15 % | 15 % | 16 % | 73 mm |

## 4. Trajectory search (no learning)
A cross-entropy search over open-loop cable motions found that eggs at **40–50° cannot be delivered** to the bucket.

## 5. Curriculum from A15c (Phase 4)
Each stage was 500 iterations (run 3: 1,000). Gate: new part ≥ 40 % and home area ≥ 75 %.

| Stage | Run | Region | New part | Old region | Home area | Verdict |
|---|---|---|---|---|---|---|
| C1 | 1 | 125–195 mm, 51–80° | 17 % | 37 % | 81 % | train again |
| C1 | 2 | 125–195 mm | 25 % | 44 % | 91 % | train again |
| C1 | 3 (1,000 it) | 125–195 mm | 31 % | 55 % | 90 % | **kept: C1_a3** |
| C1 | 4 | 125–185 mm | 30 % | 49 % | 88 % | plateau |
| C2 | 1 | 125–185 mm, 46–80° | 5 % | 38 % | 89 % | stop: low-angle limit |
| CL | 1 | 125–185 mm, 51–150° | 0 % | 29 % | 78 % | stop: policy avoids the left |

**Findings:**
- **Outer edge:** success falls with radius past ~165 mm (19 % at 185–195 mm), so the region was trimmed to 125–185 mm.
- **Below ~50°:** not deliverable, which agrees with the trajectory search.
- **Left side:** never succeeds, so the policy learns to avoid it; it would need demonstrations.

**Whole wide sector, A15c → C1_a3:**
- nominal **79 % → 91 %**;
- egg dropped **64 % → 39 %**;
- 60–70° band **44 % → 54 %**;
- overall coverage unchanged at 14 %, because the low angles and the left side stay at 0 %.

## 6. Camera student v2 (DAgger from C1_a3, starting from v1)
This was 7 rounds; it stopped at round 8 when the T4 ran out of GPU memory. Evaluated on 125–185 mm, 51–80°, with the same egg positions for both students:

| | Nominal | Retention | Sector | Egg error |
|---|---|---|---|---|
| Student v1 | 85 % | 77 % | 41 % | 4.5 mm |
| **Student v2** | **92 %** | **90 %** | **50 %** | **4.2 mm** |
| Teacher C1_a3 (separate evaluation) | 93 % | 88 % | 50 % | – |
