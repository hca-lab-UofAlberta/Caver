from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.stage1.contracts import SafetyDecision
from scripts.stage1.piper_execution import PIPER_ACTION_DIM, PIPER_ARM_DIM


@dataclass(frozen=True)
class PiperSafetyConfig:
    action_horizon: int = 4
    chunk_dt_s: float = 0.1
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.035
    joint_limits_rad: tuple[tuple[float, float] | None, ...] = (None, None, None, None, None, None)
    arm_velocity_caps_rad_s: tuple[float, ...] = (1.0, 1.0, 1.0, 1.2, 1.2, 1.5)
    gripper_velocity_cap_m_s: float = 0.05
    arm_acceleration_caps_rad_s2: tuple[float, ...] = (4.0, 4.0, 4.0, 5.0, 5.0, 6.0)
    gripper_acceleration_cap_m_s2: float = 0.25
    workspace_x_m: tuple[float, float] = (0.18, 0.55)
    workspace_y_m: tuple[float, float] = (-0.28, 0.28)
    tool_height_m: tuple[float, float] = (-0.005, 0.30)


class PiperSafetyShield:
    def __init__(self, config: PiperSafetyConfig | None = None) -> None:
        self.config = config or PiperSafetyConfig()

    def evaluate(
        self,
        execution_chunk: np.ndarray,
        observed_joint: np.ndarray,
        observed_gripper_m: float,
        observed_joint_velocity: np.ndarray | None = None,
        observed_gripper_velocity_m_s: float | None = None,
        ee_pose_sequence: np.ndarray | None = None,
    ) -> SafetyDecision:
        chunk = np.asarray(execution_chunk, dtype=np.float32)
        if chunk.shape != (self.config.action_horizon, PIPER_ACTION_DIM):
            raise ValueError(
                "execution_chunk must have shape "
                f"({self.config.action_horizon}, {PIPER_ACTION_DIM}), got {chunk.shape}"
            )

        q_prev = np.asarray(observed_joint, dtype=np.float32)
        if q_prev.shape != (PIPER_ARM_DIM,):
            raise ValueError(f"observed_joint must have shape ({PIPER_ARM_DIM},), got {q_prev.shape}")
        dq_prev = (
            np.asarray(observed_joint_velocity, dtype=np.float32)
            if observed_joint_velocity is not None
            else np.zeros(PIPER_ARM_DIM, dtype=np.float32)
        )
        if dq_prev.shape != (PIPER_ARM_DIM,):
            raise ValueError(
                f"observed_joint_velocity must have shape ({PIPER_ARM_DIM},), got {dq_prev.shape}"
            )
        g_prev = float(observed_gripper_m)
        dg_prev = float(observed_gripper_velocity_m_s or 0.0)

        reasons: set[str] = set()

        for joint_idx, limits in enumerate(self.config.joint_limits_rad):
            if limits is None:
                continue
            lower, upper = limits
            if np.any(chunk[:, joint_idx] < lower) or np.any(chunk[:, joint_idx] > upper):
                reasons.add("joint_limit")

        if np.any(chunk[:, -1] < self.config.gripper_min_m) or np.any(chunk[:, -1] > self.config.gripper_max_m):
            reasons.add("gripper_range")

        dt = self.config.chunk_dt_s
        for row in chunk:
            dq = (row[:PIPER_ARM_DIM] - q_prev) / dt
            dg = (row[-1] - g_prev) / dt
            if np.any(np.abs(dq) > np.asarray(self.config.arm_velocity_caps_rad_s, dtype=np.float32)):
                reasons.add("velocity")
            if abs(dg) > self.config.gripper_velocity_cap_m_s:
                reasons.add("velocity")

            ddq = (dq - dq_prev) / dt
            ddg = (dg - dg_prev) / dt
            if np.any(np.abs(ddq) > np.asarray(self.config.arm_acceleration_caps_rad_s2, dtype=np.float32)):
                reasons.add("acceleration")
            if abs(ddg) > self.config.gripper_acceleration_cap_m_s2:
                reasons.add("acceleration")

            q_prev = row[:PIPER_ARM_DIM]
            dq_prev = dq
            g_prev = float(row[-1])
            dg_prev = float(dg)

        if ee_pose_sequence is not None:
            ee = np.asarray(ee_pose_sequence, dtype=np.float32)
            if ee.shape not in ((self.config.action_horizon, 7), (self.config.action_horizon, 8)):
                raise ValueError(
                    "ee_pose_sequence must have shape "
                    f"({self.config.action_horizon}, 7) or ({self.config.action_horizon}, 8), got {ee.shape}"
                )
            x = ee[:, 0]
            y = ee[:, 1]
            z = ee[:, 2]
            if np.any((x < self.config.workspace_x_m[0]) | (x > self.config.workspace_x_m[1])):
                reasons.add("workspace")
            if np.any((y < self.config.workspace_y_m[0]) | (y > self.config.workspace_y_m[1])):
                reasons.add("workspace")
            if np.any((z < self.config.tool_height_m[0]) | (z > self.config.tool_height_m[1])):
                reasons.add("workspace")

        return SafetyDecision(ok=not reasons, reasons=tuple(sorted(reasons)))
