from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scripts.stage1.contracts import TaskOutcome


@dataclass(frozen=True)
class PiperLabelerConfig:
    tray_height_margin_m: float = 0.030
    bowl_height_margin_m: float = 0.045
    stack_planar_tolerance_m: float = 0.020
    stack_height_gap_m: float = 0.030
    marker_confidence_audit_threshold: float = 0.90
    marker_low_confidence_frame_limit: int = 3
    marker_audit_window_frames: int = 10


class PiperLabeler:
    def __init__(self, config: PiperLabelerConfig | None = None) -> None:
        self.config = config or PiperLabelerConfig()

    def audit_required(self, confidences: Iterable[float]) -> bool:
        values = np.asarray(list(confidences), dtype=np.float32)
        if values.size == 0:
            return True
        window = values[-self.config.marker_audit_window_frames :]
        low_count = int(np.sum(window < self.config.marker_confidence_audit_threshold))
        return low_count > self.config.marker_low_confidence_frame_limit

    def label_block_to_tray(
        self,
        inside_tray_xy: bool,
        height_above_tray_floor_m: float,
        marker_confidences: Iterable[float],
    ) -> TaskOutcome:
        audit = self.audit_required(marker_confidences)
        success = bool(inside_tray_xy and height_above_tray_floor_m <= self.config.tray_height_margin_m)
        reason = "ok" if success else "outside_tray_or_too_high"
        return TaskOutcome(success=success, audit_required=audit, reason=reason)

    def label_can_to_bowl(
        self,
        inside_bowl_xy: bool,
        height_above_bowl_floor_m: float,
        marker_confidences: Iterable[float],
    ) -> TaskOutcome:
        audit = self.audit_required(marker_confidences)
        success = bool(inside_bowl_xy and height_above_bowl_floor_m <= self.config.bowl_height_margin_m)
        reason = "ok" if success else "outside_bowl_or_too_high"
        return TaskOutcome(success=success, audit_required=audit, reason=reason)

    def label_two_block_stack(
        self,
        planar_distance_m: float,
        height_gap_m: float,
        top_block_above_bottom: bool,
        marker_confidences: Iterable[float],
    ) -> TaskOutcome:
        audit = self.audit_required(marker_confidences)
        success = bool(
            top_block_above_bottom
            and planar_distance_m <= self.config.stack_planar_tolerance_m
            and height_gap_m <= self.config.stack_height_gap_m
        )
        reason = "ok" if success else "stack_geometry_failed"
        return TaskOutcome(success=success, audit_required=audit, reason=reason)
