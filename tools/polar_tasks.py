"""Polar-sector tasks built on sir's research/s2-spawn-randomization code.

Standalone: none of sir's files are modified. Importing this module registers:

  Mjlab-SpiRob-Polar-Right : r 120-200 mm, theta  30- 80 deg
  Mjlab-SpiRob-Polar-Left  : r 120-200 mm, theta 100-150 deg
  Mjlab-SpiRob-Polar-Wide  : r 120-200 mm, theta  30-150 deg  (coverage maps)
  Mjlab-SpiRob-Polar-Arc15 : r 139-179 mm, theta  51- 76 deg  (teacher A15c's region)
  Mjlab-SpiRob-Polar-C1    : r 125-185 mm, theta  51- 80 deg  (teacher C1_a3's region; main task now)
  Mjlab-SpiRob-Cur-R<lo>   : r 125-185 mm, theta lo- 80 deg   (curriculum, lo = 51, 46, 41, 36)
  Mjlab-SpiRob-Cur-W<lo>   : r 125-185 mm, theta lo-150 deg   (curriculum, right + left side)

Each reset places the egg AND its (mocap) pedestal together at one (r, theta) with sir's
reset_stage2_sector_egg_and_pedestal: 10 % nominal spot, 30 % small "retention" offsets around
it, 60 % sector spawns. Spawns within 38 mm of the robot centreline or 55 mm of the bucket are
redrawn (sir's clearance screen). Above ~72-79 deg every spawn falls in that centreline strip.

The robot model comes from sir's repo, so a new robot model is picked up automatically.
"""
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.registry import register_mjlab_task

import spirob_mjlab  # noqa: F401  (registers sir's tasks)
from spirob_mjlab import mdp
from spirob_mjlab.entities import movable_pedestal_cfg
from spirob_mjlab.env_cfgs import spirob_egg_to_bucket_stage1_env_cfg
from spirob_mjlab.rl_cfg import Stage2FineTuneRunner, spirob_ppo_runner_cfg

RADIUS_M = (0.120, 0.200)
CUR_RADIUS_M = (0.125, 0.185)
POLAR_TASKS = {
    "Mjlab-SpiRob-Polar-Right": (RADIUS_M, (30.0, 80.0)),
    "Mjlab-SpiRob-Polar-Left": (RADIUS_M, (100.0, 150.0)),
    "Mjlab-SpiRob-Polar-Wide": (RADIUS_M, (30.0, 150.0)),
    "Mjlab-SpiRob-Polar-Arc15": ((0.139, 0.179), (50.8, 75.8)),
    "Mjlab-SpiRob-Polar-C1": (CUR_RADIUS_M, (51.0, 80.0)),
}
for _lo in (51, 46, 41, 36):
    POLAR_TASKS[f"Mjlab-SpiRob-Cur-R{_lo}"] = (CUR_RADIUS_M, (float(_lo), 80.0))
    POLAR_TASKS[f"Mjlab-SpiRob-Cur-W{_lo}"] = (CUR_RADIUS_M, (float(_lo), 150.0))


def polar_env_cfg(radius_range_m, angle_range_deg, play=False):
    cfg = spirob_egg_to_bucket_stage1_env_cfg(play=play)
    if play:
        cfg.episode_length_s = 8.0
    cfg.scene.entities["pedestal"] = movable_pedestal_cfg()
    cfg.events["stage2_egg_pedestal_spawn"] = EventTermCfg(
        func=mdp.reset_stage2_sector_egg_and_pedestal,
        mode="reset",
        params={
            "radius_range_m": radius_range_m,
            "angle_range_deg": angle_range_deg,
            # 1 x 1 strata: a wide sector crosses the robot/bucket exclusion zones,
            # and fully-excluded strata would make stratified sampling fail
            "radial_strata": 1,
            "angular_strata": 1,
            "schedule_length": 10,
            "nominal_slots": 1,
            "retention_slots": 3,
            "max_resample_attempts": 256,
            "min_robot_centerline_clearance": 0.038,
            "min_bucket_center_clearance": 0.055,
        },
    )
    cfg.metrics = {
        "spawn_radius_mm": MetricsTermCfg(func=mdp.stage2_spawn_radius_mm, reduce="last"),
        "spawn_angle_deg": MetricsTermCfg(func=mdp.stage2_spawn_angle_deg, reduce="last"),
        "spawn_is_sector": MetricsTermCfg(func=mdp.stage2_spawn_is_sector, reduce="last"),
        "spawn_rejection_count": MetricsTermCfg(func=mdp.stage2_spawn_rejection_count, reduce="last"),
    }
    return cfg


for _task_id, (_r, _th) in POLAR_TASKS.items():
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=polar_env_cfg(_r, _th, play=False),
        play_env_cfg=polar_env_cfg(_r, _th, play=True),
        rl_cfg=spirob_ppo_runner_cfg(stage2=True),
        runner_cls=Stage2FineTuneRunner,
    )
