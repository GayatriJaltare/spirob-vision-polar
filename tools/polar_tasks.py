from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.registry import register_mjlab_task

import spirob_mjlab  # noqa: F401  (registers sir's tasks)
from spirob_mjlab import mdp
from spirob_mjlab.entities import movable_pedestal_cfg
from spirob_mjlab.env_cfgs import spirob_egg_to_bucket_stage1_env_cfg
from spirob_mjlab.rl_cfg import Stage2FineTuneRunner, spirob_ppo_runner_cfg

RADIUS_M = (0.120, 0.200)
POLAR_TASKS = {
    "Mjlab-SpiRob-Polar-Right": (RADIUS_M, (30.0, 80.0)),
    "Mjlab-SpiRob-Polar-Left": (RADIUS_M, (100.0, 150.0)),
    "Mjlab-SpiRob-Polar-Wide": (RADIUS_M, (30.0, 150.0)),
    "Mjlab-SpiRob-Polar-Arc15": ((0.139, 0.179), (50.8, 75.8)),   # halfway between sir's Arc-1 and Arc-2
}


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
