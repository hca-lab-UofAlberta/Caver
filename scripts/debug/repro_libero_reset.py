#!/usr/bin/env python3
from __future__ import annotations

from omegaconf import OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv


def main() -> int:
    cfg = OmegaConf.create(
        {
            "task_suite_name": "libero_goal",
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": 5,
            "max_episode_steps": 5,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": False,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": False,
            "video_cfg": {
                "save_video": False,
                "info_on_video": True,
                "video_base_dir": "/tmp/caver-libero-video",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
            },
            "task_ids": [0],
            "num_sample_tasks": None,
        }
    )

    env = LiberoEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    print("env created")

    try:
        obs, info = env.reset()
        print("reset ok", type(obs).__name__, type(info).__name__)
        return 0
    except Exception as exc:
        print(f"reset failed: {exc!r}")
        workers = getattr(getattr(env, "env", None), "workers", [])
        for idx, worker in enumerate(workers):
            process = getattr(worker, "process", None)
            alive = process.is_alive() if process is not None else None
            exitcode = process.exitcode if process is not None else None
            print(
                f"worker[{idx}] process_alive={alive} process_exitcode={exitcode}"
            )
        raise
    finally:
        try:
            env.close()
        except Exception as close_exc:
            print(f"close failed: {close_exc!r}")


if __name__ == "__main__":
    raise SystemExit(main())
