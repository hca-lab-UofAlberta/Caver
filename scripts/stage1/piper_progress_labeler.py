from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class ProgressResult:
    value: float
    audit_required: bool
    reason: str


@dataclass(frozen=True)
class PiperProgressLabelerConfig:
    tray_height_margin_m: float = 0.030
    bowl_height_margin_m: float = 0.045
    stack_height_gap_m: float = 0.030
    marker_confidence_audit_threshold: float = 0.90
    marker_low_confidence_frame_limit: int = 3
    marker_audit_window_frames: int = 10
    distance_eps_m: float = 1e-4


class PiperProgressLabeler:
    """Deterministic Stage-F progress scores for CAVER+FASR readiness checks.

    These scores are not rewards. They are auditable geometric diagnostics used
    to decide whether a failed prefix is eligible for FASR repair admission.
    """

    def __init__(self, config: PiperProgressLabelerConfig | None = None) -> None:
        self.config = config or PiperProgressLabelerConfig()

    def audit_required(self, confidences: Iterable[float]) -> bool:
        values = np.asarray(list(confidences), dtype=np.float32)
        if values.size == 0:
            return True
        window = values[-self.config.marker_audit_window_frames :]
        low_count = int(np.sum(window < self.config.marker_confidence_audit_threshold))
        return low_count > self.config.marker_low_confidence_frame_limit

    def _distance_progress(self, initial_distance_m: float, current_distance_m: float) -> float:
        initial = max(float(initial_distance_m), self.config.distance_eps_m)
        return _clip01(1.0 - float(current_distance_m) / initial)

    def label_block_to_tray_progress(
        self,
        initial_xy_distance_m: float,
        current_xy_distance_m: float,
        inside_tray_xy: bool,
        height_above_tray_floor_m: float,
        marker_confidences: Iterable[float],
    ) -> ProgressResult:
        audit = self.audit_required(marker_confidences)
        distance_score = self._distance_progress(initial_xy_distance_m, current_xy_distance_m)
        inside_score = 1.0 if inside_tray_xy else 0.0
        height_score = _clip01(1.0 - float(height_above_tray_floor_m) / self.config.tray_height_margin_m)
        value = _clip01(0.70 * distance_score + 0.20 * inside_score + 0.10 * height_score)
        reason = "audit_required" if audit else "ok"
        return ProgressResult(value=value, audit_required=audit, reason=reason)

    def label_can_to_bowl_progress(
        self,
        initial_xy_distance_m: float,
        current_xy_distance_m: float,
        inside_bowl_xy: bool,
        height_above_bowl_floor_m: float,
        marker_confidences: Iterable[float],
    ) -> ProgressResult:
        audit = self.audit_required(marker_confidences)
        distance_score = self._distance_progress(initial_xy_distance_m, current_xy_distance_m)
        inside_score = 1.0 if inside_bowl_xy else 0.0
        height_score = _clip01(1.0 - float(height_above_bowl_floor_m) / self.config.bowl_height_margin_m)
        value = _clip01(0.70 * distance_score + 0.20 * inside_score + 0.10 * height_score)
        reason = "audit_required" if audit else "ok"
        return ProgressResult(value=value, audit_required=audit, reason=reason)

    def label_two_block_stack_progress(
        self,
        initial_planar_distance_m: float,
        current_planar_distance_m: float,
        height_gap_m: float,
        top_block_above_bottom: bool,
        stable_orientation: bool,
        marker_confidences: Iterable[float],
    ) -> ProgressResult:
        audit = self.audit_required(marker_confidences)
        distance_score = self._distance_progress(initial_planar_distance_m, current_planar_distance_m)
        above_score = 1.0 if top_block_above_bottom else 0.0
        height_score = _clip01(1.0 - abs(float(height_gap_m) - self.config.stack_height_gap_m) / self.config.stack_height_gap_m)
        stability_score = 1.0 if stable_orientation else 0.0
        value = _clip01(0.45 * distance_score + 0.25 * above_score + 0.20 * height_score + 0.10 * stability_score)
        reason = "audit_required" if audit else "ok"
        return ProgressResult(value=value, audit_required=audit, reason=reason)
