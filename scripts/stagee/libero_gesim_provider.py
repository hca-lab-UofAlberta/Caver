from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


LIBERO_GESIM_BUNDLE_FORMAT = "libero_gesim_bundle_v1"
LIBERO_GESIM_HISTORY_LENGTH = 4
LIBERO_GESIM_SUMMARY_VIEW = "head"
LIBERO_OSC_POSE_POSITION_SCALE = np.array([0.05, 0.05, 0.05], dtype=np.float32)
LIBERO_OSC_POSE_ROTATION_SCALE = np.array([0.5, 0.5, 0.5], dtype=np.float32)
LIBERO_GESIM_DUMMY_RIGHT_ARM = np.array([0.80, -0.80, 0.30, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

# Stage-E LIBERO is a simulation proxy for the proposal's PiPER GE-Sim path.
# We alias the available LIBERO cameras into the GE-Sim example schema so the
# provider input bundle is explicit and reproducible even before full GE-Sim
# inference is turned on.
LIBERO_TO_GESIM_CAMERA_MAP: tuple[tuple[str, str], ...] = (
    ("frontview", "head"),
    ("agentview", "hand_left"),
    ("robot0_eye_in_hand", "hand_right"),
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_png(path: Path, image: np.ndarray) -> None:
    _ensure_parent(path)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def camera_intrinsic_from_fovy(
    *,
    height: int,
    width: int,
    fovy_degrees: float,
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("camera dimensions must be positive")
    half_fovy_radians = math.radians(float(fovy_degrees)) / 2.0
    fy = (0.5 * float(height)) / math.tan(half_fovy_radians)
    fx = (0.5 * float(width)) / math.tan(half_fovy_radians)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def get_libero_camera_extrinsic(env: Any, camera_name: str) -> np.ndarray:
    camera_id = int(env.sim.model.camera_name2id(camera_name))
    camera_rotation = np.asarray(env.sim.data.cam_xmat[camera_id], dtype=np.float32).reshape(3, 3)
    camera_position = np.asarray(env.sim.data.cam_xpos[camera_id], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = camera_rotation
    extrinsic[:3, 3] = camera_position
    return extrinsic


def extract_libero_gesim_provider_observation(
    obs: Mapping[str, Any],
    *,
    env: Any,
) -> dict[str, Any]:
    images: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    source_camera_names: dict[str, str] = {}

    for libero_camera_name, gesim_camera_name in LIBERO_TO_GESIM_CAMERA_MAP:
        obs_key = f"{libero_camera_name}_image"
        if obs_key not in obs:
            raise KeyError(f"required LIBERO observation key is missing for provider bundle: {obs_key}")
        image = np.ascontiguousarray(np.asarray(obs[obs_key])[::-1, ::-1])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB image for {obs_key}, got shape {image.shape}")
        image = np.asarray(image, dtype=np.uint8)
        camera_id = int(env.sim.model.camera_name2id(libero_camera_name))
        fovy_degrees = float(env.sim.model.cam_fovy[camera_id])
        intrinsic = camera_intrinsic_from_fovy(
            height=int(image.shape[0]),
            width=int(image.shape[1]),
            fovy_degrees=fovy_degrees,
        )
        extrinsic = get_libero_camera_extrinsic(env, libero_camera_name)
        images[gesim_camera_name] = image
        intrinsics[gesim_camera_name] = intrinsic
        extrinsics[gesim_camera_name] = extrinsic
        source_camera_names[gesim_camera_name] = libero_camera_name

    eef_pose = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        ),
        axis=0,
    )
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)

    return {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "eef_pose": eef_pose,
        "gripper_qpos": gripper_qpos,
        "source_camera_names": source_camera_names,
    }


def _normalize_gripper_command(raw_command: float) -> float:
    clipped = min(max(float(raw_command), -1.0), 1.0)
    return 0.5 * (clipped + 1.0)


def build_libero_gesim_action_array(
    candidate_chunk: Sequence[Sequence[float]],
    *,
    initial_eef_pose: Sequence[float],
) -> np.ndarray:
    eef_pose = np.asarray(initial_eef_pose, dtype=np.float32).reshape(-1)
    if eef_pose.size != 7:
        raise ValueError(f"expected initial eef pose with 7 values, got {eef_pose.size}")

    current_position = eef_pose[:3].astype(np.float32).copy()
    current_rotation = Rotation.from_quat(eef_pose[3:7].astype(np.float64))
    provider_actions: list[np.ndarray] = []

    for step_index, raw_step in enumerate(candidate_chunk):
        step = np.asarray(raw_step, dtype=np.float32).reshape(-1)
        if step.size < 7:
            raise ValueError(
                f"candidate chunk step {step_index} is too short for LIBERO OSC_POSE + gripper: {step.size}"
            )
        delta_position = np.clip(step[:3], -1.0, 1.0) * LIBERO_OSC_POSE_POSITION_SCALE
        delta_rotvec = np.clip(step[3:6], -1.0, 1.0) * LIBERO_OSC_POSE_ROTATION_SCALE
        current_position = current_position + delta_position.astype(np.float32)
        current_rotation = Rotation.from_rotvec(delta_rotvec.astype(np.float64)) * current_rotation
        left_arm_pose = np.concatenate(
            (
                current_position.astype(np.float32),
                current_rotation.as_quat().astype(np.float32),
                np.asarray([_normalize_gripper_command(float(step[6]))], dtype=np.float32),
            ),
            axis=0,
        )
        provider_actions.append(
            np.concatenate((left_arm_pose, LIBERO_GESIM_DUMMY_RIGHT_ARM.copy()), axis=0).astype(np.float32)
        )

    if not provider_actions:
        raise ValueError("candidate chunk must contain at least one action step")
    return np.stack(provider_actions, axis=0)


def write_libero_gesim_bundle(
    *,
    provider_observation_history: Sequence[Mapping[str, Any]],
    candidate_chunk: Sequence[Sequence[float]],
    output_dir: str | Path,
    context_id: str,
    policy_query_index: int,
    candidate_index: int,
) -> dict[str, Any]:
    if not provider_observation_history:
        raise ValueError("provider observation history must contain at least one entry")

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    raw_history = list(provider_observation_history)
    history = raw_history[-LIBERO_GESIM_HISTORY_LENGTH :]
    while len(history) < LIBERO_GESIM_HISTORY_LENGTH:
        history.insert(0, history[0])

    latest_observation = history[-1]
    provider_actions = build_libero_gesim_action_array(
        candidate_chunk,
        initial_eef_pose=latest_observation["eef_pose"],
    )
    exec_actions = np.asarray(candidate_chunk, dtype=np.float32)

    per_camera_image_counts: dict[str, int] = {}
    for gesim_camera_name in latest_observation["images"]:
        image_dir = resolved_output_dir / f"{gesim_camera_name}_color"
        image_dir.mkdir(parents=True, exist_ok=True)
        extrinsic_sequence = []
        for frame_index, history_item in enumerate(history):
            image_path = image_dir / f"{frame_index}.png"
            _write_png(image_path, history_item["images"][gesim_camera_name])
            extrinsic_sequence.append(np.asarray(history_item["extrinsics"][gesim_camera_name], dtype=np.float32))
        np.save(resolved_output_dir / f"extrinsic_{gesim_camera_name}.npy", np.stack(extrinsic_sequence, axis=0))
        np.save(
            resolved_output_dir / f"intrinsic_{gesim_camera_name}.npy",
            np.asarray(latest_observation["intrinsics"][gesim_camera_name], dtype=np.float32),
        )
        per_camera_image_counts[gesim_camera_name] = len(history)

    actions_path = resolved_output_dir / "actions.npy"
    exec_actions_path = resolved_output_dir / "exec_actions.npy"
    np.save(actions_path, provider_actions)
    np.save(exec_actions_path, exec_actions)

    metadata = {
        "bundle_format": LIBERO_GESIM_BUNDLE_FORMAT,
        "context_id": str(context_id),
        "policy_query_index": int(policy_query_index),
        "candidate_index": int(candidate_index),
        "memory_frame_count": len(history),
        "summary_view": LIBERO_GESIM_SUMMARY_VIEW,
        "gesim_camera_names": [gesim_camera_name for _, gesim_camera_name in LIBERO_TO_GESIM_CAMERA_MAP],
        "source_camera_names": dict(latest_observation["source_camera_names"]),
        "provider_action_shape": list(provider_actions.shape),
        "exec_action_shape": list(exec_actions.shape),
        "osc_pose_adapter": {
            "position_scale": LIBERO_OSC_POSE_POSITION_SCALE,
            "rotation_scale": LIBERO_OSC_POSE_ROTATION_SCALE,
            "right_arm_dummy_pose": LIBERO_GESIM_DUMMY_RIGHT_ARM,
            "gripper_mapping": "normalized_[0,1]_from_raw_[-1,1]",
        },
        "image_counts": per_camera_image_counts,
        "inference_status": "bundle_only",
        "notes": (
            "This bundle materializes the proposal-aligned GE-Sim input interface for Stage-E LIBERO contexts. "
            "Actual GE-Sim rollout inference still requires the frozen provider weights and runtime dependencies."
        ),
    }
    metadata_path = resolved_output_dir / "bundle_metadata.json"
    _write_json(metadata_path, metadata)

    return {
        "bundle_format": LIBERO_GESIM_BUNDLE_FORMAT,
        "output_dir": str(resolved_output_dir),
        "action_path": str(actions_path),
        "exec_action_path": str(exec_actions_path),
        "metadata_path": str(metadata_path),
        "summary_view": LIBERO_GESIM_SUMMARY_VIEW,
        "source_camera_names": dict(latest_observation["source_camera_names"]),
        "provider_action_shape": list(provider_actions.shape),
        "exec_action_shape": list(exec_actions.shape),
        "memory_frame_count": len(history),
        "inference_status": "bundle_only",
    }
