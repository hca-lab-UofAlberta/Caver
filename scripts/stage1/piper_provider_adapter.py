from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.stage1.piper_execution import PIPER_ACTION_DIM


@dataclass(frozen=True)
class PiperToGESimConfig:
    action_horizon: int = 4
    provider_action_dim: int = 16
    parked_right_pose: tuple[float, ...] = (0.80, -0.80, 0.30, 0.0, 0.0, 0.0, 1.0, 0.0)


class PiperToGESimAdapter:
    def __init__(self, config: PiperToGESimConfig | None = None) -> None:
        self.config = config or PiperToGESimConfig()

    def build_provider_chunk(
        self,
        execution_chunk: np.ndarray,
        left_pose_sequence: np.ndarray,
        right_pose_sequence: np.ndarray | None = None,
    ) -> np.ndarray:
        exec_chunk = np.asarray(execution_chunk, dtype=np.float32)
        if exec_chunk.shape != (self.config.action_horizon, PIPER_ACTION_DIM):
            raise ValueError(
                "execution_chunk must have shape "
                f"({self.config.action_horizon}, {PIPER_ACTION_DIM}), got {exec_chunk.shape}"
            )

        left = np.asarray(left_pose_sequence, dtype=np.float32)
        if left.shape == (self.config.action_horizon, 7):
            left = np.concatenate([left, exec_chunk[:, -1:]], axis=1)
        elif left.shape != (self.config.action_horizon, 8):
            raise ValueError(
                "left_pose_sequence must have shape "
                f"({self.config.action_horizon}, 7) or ({self.config.action_horizon}, 8), "
                f"got {left.shape}"
            )

        if right_pose_sequence is None:
            right = np.repeat(
                np.asarray(self.config.parked_right_pose, dtype=np.float32)[None, :],
                self.config.action_horizon,
                axis=0,
            )
        else:
            right = np.asarray(right_pose_sequence, dtype=np.float32)
            if right.shape == (8,):
                right = np.repeat(right[None, :], self.config.action_horizon, axis=0)
            elif right.shape != (self.config.action_horizon, 8):
                raise ValueError(
                    "right_pose_sequence must have shape (8,) or "
                    f"({self.config.action_horizon}, 8), got {right.shape}"
                )

        provider_chunk = np.concatenate([left, right], axis=1)
        if provider_chunk.shape != (self.config.action_horizon, self.config.provider_action_dim):
            raise ValueError(
                "provider_chunk must have shape "
                f"({self.config.action_horizon}, {self.config.provider_action_dim}), got {provider_chunk.shape}"
            )
        return provider_chunk
