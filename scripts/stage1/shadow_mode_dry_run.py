#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.pi05_piper_h4 import load_config
from scripts.stage1.piper_execution import PiperExecutionAdapter
from scripts.stage1.piper_provider_adapter import PiperToGESimAdapter
from scripts.stage1.piper_safety import PiperSafetyShield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-hardware Stage-1 shadow-mode dry run. "
            "This writes execution/provider artifacts and a JSON trace without commanding any robot."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", default="block_to_tray")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=8765)
    parser.add_argument("--observed-joint", nargs=6, type=float, default=[0.0] * 6)
    parser.add_argument("--observed-gripper-m", type=float, default=0.01)
    parser.add_argument("--left-pose-npy", type=Path, default=None)
    parser.add_argument(
        "--synthetic-left-pose",
        action="store_true",
        help=(
            "Use a fixed in-workspace synthetic end-effector pose sequence for dry-run validation. "
            "This is only for scaffolding and must not replace real FK in the actual robot path."
        ),
    )
    parser.add_argument(
        "--record-policy-arrays",
        action="store_true",
        help="Write additional numpy arrays returned by the policy service when possible.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}, got {type(payload).__name__}")
    return payload


def build_synthetic_cameras(output_dir: Path, height: int, width: int) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    observations_dir = output_dir / "observations"
    observations_dir.mkdir(parents=True, exist_ok=True)

    images: dict[str, np.ndarray] = {}
    image_paths: dict[str, str] = {}
    base = np.zeros((height, width, 3), dtype=np.uint8)
    gradients = {
        "head": np.stack(
            [
                np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1)),
                np.zeros((height, width), dtype=np.uint8),
                np.full((height, width), 64, dtype=np.uint8),
            ],
            axis=-1,
        ),
        "hand_left": np.stack(
            [
                np.zeros((height, width), dtype=np.uint8),
                np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, None], (1, width)),
                np.full((height, width), 96, dtype=np.uint8),
            ],
            axis=-1,
        ),
        "hand_right": np.stack(
            [
                np.full((height, width), 48, dtype=np.uint8),
                np.flipud(np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, None], (1, width))),
                np.zeros((height, width), dtype=np.uint8),
            ],
            axis=-1,
        ),
    }

    for name in ("head", "hand_left", "hand_right"):
        image = np.array(base, copy=True)
        image[:] = gradients[name]
        path = observations_dir / f"{name}.npy"
        np.save(path, image)
        images[name] = image
        image_paths[name] = str(path)
    return images, image_paths


def build_prompt(task_id: str, task_config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    task_entry = task_config["tasks"][task_id]
    prompt = task_entry.get("language_instruction")
    if not prompt:
        raise ValueError(f"task {task_id!r} missing language_instruction in configs/piper_tasks.yaml")
    return str(prompt)


def query_policy(
    host: str,
    port: int,
    prompt: str,
    head_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(host, port)
    metadata = policy.get_server_metadata()
    observation = {
        "observation/image": head_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }
    start = time.perf_counter()
    payload = policy.infer(observation)
    latency_ms = (time.perf_counter() - start) * 1000.0
    actions = np.asarray(payload["actions"], dtype=np.float32)
    return actions, {"server_metadata": metadata, "payload": payload}, latency_ms


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "actions":
            continue
        if isinstance(value, dict):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    nested[nested_key] = nested_value
                else:
                    array = np.asarray(nested_value)
                    nested[nested_key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            summary[key] = nested
        elif isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        else:
            array = np.asarray(value)
            summary[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return summary


def sanitize_artifact_name(name: str) -> str:
    sanitized = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "artifact"


def maybe_write_payload_arrays(output_dir: Path, payload: dict[str, Any]) -> list[str]:
    written: list[str] = []
    payload_dir = output_dir / "policy_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for key, value in payload.items():
        if key == "actions":
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    continue
                array = np.asarray(nested_value)
                path = payload_dir / (
                    f"{sanitize_artifact_name(key)}__{sanitize_artifact_name(nested_key)}.npy"
                )
                np.save(path, array)
                written.append(str(path))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            continue
        else:
            array = np.asarray(value)
            path = payload_dir / f"{sanitize_artifact_name(key)}.npy"
            np.save(path, array)
            written.append(str(path))
    return written


def load_or_build_left_pose(args: argparse.Namespace, action_horizon: int, execution_chunk: np.ndarray) -> tuple[np.ndarray, str]:
    if args.left_pose_npy is not None:
        left_pose = np.load(args.left_pose_npy)
        return np.asarray(left_pose, dtype=np.float32), f"file:{args.left_pose_npy}"
    if args.synthetic_left_pose:
        synthetic = np.repeat(
            np.asarray([[0.30, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            action_horizon,
            axis=0,
        )
        synthetic[:, 0] += np.linspace(0.0, 0.02, action_horizon, dtype=np.float32)
        synthetic[:, 2] += np.linspace(0.0, 0.01, action_horizon, dtype=np.float32)
        return synthetic, "synthetic"
    raise SystemExit(
        "error: provide --left-pose-npy or --synthetic-left-pose so the dry run can build the provider chunk"
    )


def main() -> None:
    args = parse_args()

    policy_config = load_config()
    task_config = load_yaml(REPO_ROOT / "configs" / "piper_tasks.yaml")
    manifest = load_yaml(REPO_ROOT / "manifest.lock")

    if args.task_id not in task_config.get("tasks", {}):
        available = ", ".join(sorted(task_config.get("tasks", {}).keys()))
        raise SystemExit(f"error: unknown --task-id {args.task_id!r}; available tasks: {available}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images, image_paths = build_synthetic_cameras(output_dir, args.image_height, args.image_width)

    prompt = build_prompt(args.task_id, task_config, args.prompt)
    observed_joint = np.asarray(args.observed_joint, dtype=np.float32)
    observed_state = np.concatenate([observed_joint, np.asarray([args.observed_gripper_m], dtype=np.float32)])

    policy_source = "synthetic_zero_actions"
    policy_metadata: dict[str, Any] = {}
    policy_latency_ms: float | None = None

    if args.policy_host:
        raw_chunk, policy_result, policy_latency_ms = query_policy(
            args.policy_host,
            args.policy_port,
            prompt,
            images["head"],
            images["hand_left"],
            observed_state,
        )
        policy_source = f"websocket:{args.policy_host}:{args.policy_port}"
        policy_metadata = {
            "server_metadata": policy_result["server_metadata"],
            "payload_summary": summarize_payload(policy_result["payload"]),
        }
        if args.record_policy_arrays:
            policy_metadata["recorded_arrays"] = maybe_write_payload_arrays(output_dir, policy_result["payload"])
    else:
        raw_chunk = np.zeros((policy_config.action_horizon, policy_config.action_dim), dtype=np.float32)
        raw_chunk[:, -1] = float(args.observed_gripper_m)

    execution_adapter = PiperExecutionAdapter()
    execution_chunk = execution_adapter.adapt(raw_chunk)

    left_pose, left_pose_source = load_or_build_left_pose(args, policy_config.action_horizon, execution_chunk)
    safety = PiperSafetyShield()
    safety_decision = safety.evaluate(
        execution_chunk,
        observed_joint=observed_joint,
        observed_gripper_m=args.observed_gripper_m,
        ee_pose_sequence=left_pose,
    )

    provider_adapter = PiperToGESimAdapter()
    provider_chunk = provider_adapter.build_provider_chunk(execution_chunk, left_pose)

    raw_chunk_path = output_dir / "policy_raw_actions.npy"
    exec_chunk_path = output_dir / "exec_actions.npy"
    provider_chunk_path = output_dir / "actions.npy"
    np.save(raw_chunk_path, raw_chunk)
    np.save(exec_chunk_path, execution_chunk)
    np.save(provider_chunk_path, provider_chunk)

    trace = {
        "status": "shadow_mode_dry_run_complete",
        "scaffold_only": True,
        "task_id": args.task_id,
        "prompt": prompt,
        "policy_source": policy_source,
        "policy_latency_ms": policy_latency_ms,
        "policy_metadata": policy_metadata,
        "policy_config": {
            "name": policy_config.name,
            "action_horizon": policy_config.action_horizon,
            "action_dim": policy_config.action_dim,
            "provider_action_dim": policy_config.provider_action_dim,
        },
        "manifest_status": manifest.get("status"),
        "camera_paths": image_paths,
        "observed_state": {
            "joint_position": observed_joint.tolist(),
            "gripper_position_m": float(args.observed_gripper_m),
        },
        "left_pose_source": left_pose_source,
        "safety_decision": {
            "ok": safety_decision.ok,
            "reasons": list(safety_decision.reasons),
        },
        "artifacts": {
            "policy_raw_actions": str(raw_chunk_path),
            "exec_actions": str(exec_chunk_path),
            "provider_actions": str(provider_chunk_path),
        },
        "notes": [
            "Synthetic observation arrays are written as .npy files for the dry run.",
            "Synthetic left-pose mode is only for pre-hardware validation and must not replace real FK.",
            "This script never commands hardware.",
        ],
    }

    trace_path = output_dir / "shadow_context.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"OK output_dir={output_dir}")
    print(f"OK task_id={args.task_id}")
    print(f"OK policy_source={policy_source}")
    print(f"OK safety_ok={safety_decision.ok} reasons={list(safety_decision.reasons)}")
    print(f"OK exec_actions={exec_chunk_path}")
    print(f"OK provider_actions={provider_chunk_path}")
    print(f"OK trace={trace_path}")


if __name__ == "__main__":
    main()
