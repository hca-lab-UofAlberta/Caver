from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PIPER_ARM_DIM = 6
PIPER_ACTION_DIM = 7


@dataclass(frozen=True)
class PiperExecutionAdapterConfig:
    action_horizon: int = 4
    chunk_dt_s: float = 0.1
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.035
    joint_limits_rad: tuple[tuple[float, float] | None, ...] = (None, None, None, None, None, None)


class PiperExecutionAdapter:
    def __init__(self, config: PiperExecutionAdapterConfig | None = None) -> None:
        self.config = config or PiperExecutionAdapterConfig()

    def adapt(self, raw_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(raw_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            expected_size = self.config.action_horizon * PIPER_ACTION_DIM
            if chunk.size != expected_size:
                raise ValueError(
                    f"flat raw_chunk must have {expected_size} values, got {chunk.size}"
                )
            chunk = chunk.reshape(self.config.action_horizon, PIPER_ACTION_DIM)
        if chunk.shape != (self.config.action_horizon, PIPER_ACTION_DIM):
            raise ValueError(
                "raw_chunk must have shape "
                f"({self.config.action_horizon}, {PIPER_ACTION_DIM}), got {chunk.shape}"
            )

        adapted = np.array(chunk, copy=True)
        for joint_idx, limits in enumerate(self.config.joint_limits_rad):
            if limits is None:
                continue
            lower, upper = limits
            adapted[:, joint_idx] = np.clip(adapted[:, joint_idx], lower, upper)
        adapted[:, -1] = np.clip(
            adapted[:, -1], self.config.gripper_min_m, self.config.gripper_max_m
        )
        return adapted

    def hold_chunk(self, observed_joint: np.ndarray, observed_gripper_m: float) -> np.ndarray:
        joint = np.asarray(observed_joint, dtype=np.float32)
        if joint.shape != (PIPER_ARM_DIM,):
            raise ValueError(f"observed_joint must have shape ({PIPER_ARM_DIM},), got {joint.shape}")
        hold = np.zeros((self.config.action_horizon, PIPER_ACTION_DIM), dtype=np.float32)
        hold[:, :PIPER_ARM_DIM] = joint[None, :]
        hold[:, -1] = float(
            np.clip(observed_gripper_m, self.config.gripper_min_m, self.config.gripper_max_m)
        )
        return hold
