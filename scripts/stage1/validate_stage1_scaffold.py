#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.pi05_piper_h4 import load_config
from scripts.stage1.piper_execution import PiperExecutionAdapter
from scripts.stage1.piper_labeler import PiperLabeler
from scripts.stage1.piper_progress_labeler import PiperProgressLabeler
from scripts.stage1.piper_provider_adapter import PiperToGESimAdapter
from scripts.stage1.piper_safety import PiperSafetyShield


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}, got {type(payload).__name__}")
    return payload


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def main() -> None:
    policy_config = load_config()
    ensure(policy_config.action_horizon == 4, "policy action_horizon must be 4")
    ensure(policy_config.action_dim == 7, "policy action_dim must be 7")

    camera_config = load_yaml(REPO_ROOT / "configs" / "camera" / "piper_multiview.yaml")
    camera_views = set(camera_config.get("views", {}).keys())
    ensure(camera_views == {"head", "hand_left", "hand_right"}, "camera views must be head/hand_left/hand_right")

    task_config = load_yaml(REPO_ROOT / "configs" / "piper_tasks.yaml")
    task_names = set(task_config.get("tasks", {}).keys())
    ensure(task_names == {"block_to_tray", "can_to_bowl", "two_block_stack"}, "task set must match the proposal")

    manifest = load_yaml(REPO_ROOT / "manifest.lock")
    ensure(manifest.get("study_id") == "caver_piper_stage1_v1", "manifest study_id mismatch")
    ensure(manifest.get("ready_for_real_runs") is False, "scaffold manifest should not be marked ready")

    execution = PiperExecutionAdapter()
    raw_chunk = np.zeros((4, 7), dtype=np.float32)
    raw_chunk[:, -1] = 0.01
    execution_chunk = execution.adapt(raw_chunk)
    ensure(execution_chunk.shape == (4, 7), "execution chunk shape mismatch")

    ee_pose_sequence = np.tile(np.array([0.30, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1))

    provider = PiperToGESimAdapter()
    provider_chunk = provider.build_provider_chunk(execution_chunk, ee_pose_sequence)
    ensure(provider_chunk.shape == (4, 16), "provider chunk shape mismatch")

    safety = PiperSafetyShield()
    decision = safety.evaluate(
        execution_chunk,
        observed_joint=np.zeros(6, dtype=np.float32),
        observed_gripper_m=0.01,
        ee_pose_sequence=ee_pose_sequence,
    )
    ensure(decision.ok, f"sample safety decision should pass, got {decision.reasons}")

    labeler = PiperLabeler()
    outcome = labeler.label_block_to_tray(
        inside_tray_xy=True,
        height_above_tray_floor_m=0.02,
        marker_confidences=[0.95] * 10,
    )
    ensure(outcome.success and not outcome.audit_required, "sample label outcome should succeed without audit")

    progress_labeler = PiperProgressLabeler()
    progress = progress_labeler.label_block_to_tray_progress(
        initial_xy_distance_m=0.20,
        current_xy_distance_m=0.04,
        inside_tray_xy=True,
        height_above_tray_floor_m=0.01,
        marker_confidences=[0.95] * 10,
    )
    ensure(progress.value >= 0.75 and not progress.audit_required, "sample progress label should be high without audit")

    print(f"OK policy_config={policy_config.name}")
    print(f"OK camera_views={sorted(camera_views)}")
    print(f"OK task_names={sorted(task_names)}")
    print(f"OK provider_chunk_shape={provider_chunk.shape}")
    print(f"OK safety_reasons={decision.reasons}")
    print(f"OK label_outcome=success:{outcome.success},audit:{outcome.audit_required},reason:{outcome.reason}")
    print(f"OK progress_label=value:{progress.value:.3f},audit:{progress.audit_required},reason:{progress.reason}")


if __name__ == "__main__":
    main()
