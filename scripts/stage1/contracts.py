from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


def require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return array


@dataclass(frozen=True)
class ObservationBundle:
    task_id: str
    prompt: str
    camera_paths: Mapping[str, Path]
    joint_position: np.ndarray
    gripper_position_m: float
    joint_velocity: np.ndarray | None = None
    gripper_velocity_m_s: float | None = None


@dataclass(frozen=True)
class SafetyDecision:
    ok: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TaskOutcome:
    success: bool
    audit_required: bool
    reason: str
