#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_TEMPLATE = REPO_ROOT / "third_party/src/Genie-Envisioner/configs/cosmos_model/acwm_cosmos.yaml"
DEFAULT_MODEL_CACHE_ROOT = REPO_ROOT / "third_party/model-cache/gesim"
DEFAULT_GESIM_CHECKPOINT_PATH = DEFAULT_MODEL_CACHE_ROOT / "ge_sim_cosmos_v0.1.safetensors"
DEFAULT_COSMOS_ASSETS_ROOT = DEFAULT_MODEL_CACHE_ROOT / "Cosmos-Predict2-2B-Video2World"
DEFAULT_COSMOS_REPO_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
DEFAULT_HF_TOKEN_FILE = Path.home() / ".config" / "huggingface_token"
DEFAULT_MODEL_SCOPE_URL = (
    "https://modelscope.cn/api/v1/models/agibot_world/Genie-Envisioner/repo"
    "?Revision=master&FilePath=ge_sim_cosmos_v0.1.safetensors"
)
EXPECTED_GESIM_CHECKPOINT_SIZE = 3938231616
EXPECTED_GESIM_CHECKPOINT_SHA256 = "0e49bbe4e83c2b6e380e0e2215f8f257ac760498b772b20e52f37a40b6649f8d"
COSMOS_REQUIRED_PATH_OPTIONS: dict[str, tuple[str, ...]] = {
    "model_index": ("model_index.json",),
    "scheduler_config": ("scheduler/scheduler_config.json",),
    "tokenizer_config": ("tokenizer/tokenizer_config.json",),
    "tokenizer_model": ("tokenizer/spiece.model",),
    "text_encoder_config": ("text_encoder/config.json",),
    "text_encoder_weights": (
        "text_encoder/model.safetensors",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/pytorch_model.bin",
    ),
    "vae_config": ("vae/config.json",),
    "vae_weights": (
        "vae/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors.index.json",
        "vae/diffusion_pytorch_model.bin",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect, optionally download, and render the local GE-Sim runtime configuration "
            "for the Stage-E LIBERO provider path."
        )
    )
    parser.add_argument(
        "--config-template-path",
        type=Path,
        default=DEFAULT_CONFIG_TEMPLATE,
        help="Template YAML config from Genie-Envisioner.",
    )
    parser.add_argument(
        "--output-config-path",
        type=Path,
        help="Optional output YAML path. Written only when the runtime is ready.",
    )
    parser.add_argument(
        "--status-json-path",
        type=Path,
        help="Optional JSON status report path.",
    )
    parser.add_argument(
        "--gesim-checkpoint-path",
        type=Path,
        default=DEFAULT_GESIM_CHECKPOINT_PATH,
        help="Path to ge_sim_cosmos_v0.1.safetensors.",
    )
    parser.add_argument(
        "--cosmos-assets-root",
        type=Path,
        default=DEFAULT_COSMOS_ASSETS_ROOT,
        help="Local snapshot directory for the gated Cosmos tokenizer/text-encoder/VAE assets.",
    )
    parser.add_argument(
        "--cosmos-repo-id",
        default=DEFAULT_COSMOS_REPO_ID,
        help="Hugging Face repo id for the Cosmos Video2World assets.",
    )
    parser.add_argument(
        "--download-cosmos-assets",
        action="store_true",
        help="Attempt gated Hugging Face snapshot download into --cosmos-assets-root.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Optional HF token; defaults to HF_TOKEN or HUGGING_FACE_HUB_TOKEN from the environment.",
    )
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=DEFAULT_HF_TOKEN_FILE if DEFAULT_HF_TOKEN_FILE.exists() else None,
        help=(
            "Optional file containing a Hugging Face token. Used only when --hf-token and the environment "
            "variables are unset."
        ),
    )
    parser.add_argument(
        "--verify-checkpoint-sha256",
        action="store_true",
        help="Compute SHA256 for the GE-Sim checkpoint when the file size matches the expected size.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless both the GE-Sim checkpoint and Cosmos assets are ready.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}, got {type(payload).__name__}")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_token_file(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    token = resolved.read_text(encoding="utf-8").strip()
    return token or None


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    if args.hf_token:
        return str(args.hf_token).strip() or None
    return read_token_file(args.hf_token_file)


def inspect_gesim_checkpoint(path: Path, *, verify_sha256: bool) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_size_bytes": EXPECTED_GESIM_CHECKPOINT_SIZE,
        "expected_sha256": EXPECTED_GESIM_CHECKPOINT_SHA256,
        "download_url": DEFAULT_MODEL_SCOPE_URL,
    }
    if not path.is_file():
        status["ready"] = False
        status["reason"] = "missing"
        return status

    size_bytes = path.stat().st_size
    status["size_bytes"] = size_bytes
    status["size_matches"] = size_bytes == EXPECTED_GESIM_CHECKPOINT_SIZE
    if verify_sha256 and status["size_matches"]:
        observed_sha256 = compute_sha256(path)
        status["sha256"] = observed_sha256
        status["sha256_matches"] = observed_sha256 == EXPECTED_GESIM_CHECKPOINT_SHA256
    else:
        status["sha256"] = None
        status["sha256_matches"] = None

    if not status["size_matches"]:
        if size_bytes < EXPECTED_GESIM_CHECKPOINT_SIZE:
            status["reason"] = "partial_download"
        else:
            status["reason"] = "unexpected_size"
        status["ready"] = False
        return status

    if verify_sha256 and status["sha256_matches"] is False:
        status["reason"] = "sha256_mismatch"
        status["ready"] = False
        return status

    status["reason"] = "ok"
    status["ready"] = True
    return status


def download_cosmos_assets(*, repo_id: str, local_dir: Path, hf_token: str | None) -> dict[str, Any]:
    if not hf_token:
        raise SystemExit(
            "error: --download-cosmos-assets requires a Hugging Face token via --hf-token, HF_TOKEN, "
            "or HUGGING_FACE_HUB_TOKEN"
        )

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=hf_token,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "vae/*",
        ],
    )
    return {"snapshot_path": str(Path(resolved).resolve())}


def resolve_required_entry(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative_path in candidates:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def inspect_cosmos_assets(root: Path, *, repo_id: str) -> dict[str, Any]:
    required: dict[str, Any] = {}
    missing_keys: list[str] = []
    for key, candidates in COSMOS_REQUIRED_PATH_OPTIONS.items():
        resolved = resolve_required_entry(root, candidates)
        required[key] = {
            "candidates": list(candidates),
            "resolved_path": (str(resolved) if resolved is not None else None),
            "present": resolved is not None,
        }
        if resolved is None:
            missing_keys.append(key)

    return {
        "root": str(root),
        "exists": root.exists(),
        "repo_id": repo_id,
        "required_entries": required,
        "missing_keys": missing_keys,
        "ready": not missing_keys,
    }


def render_runtime_config(
    *,
    template_path: Path,
    output_path: Path,
    cosmos_assets_root: Path,
    gesim_checkpoint_path: Path,
) -> None:
    payload = load_yaml(template_path)
    repo_root = template_path.resolve().parents[2]
    payload["pretrained_model_name_or_path"] = str(cosmos_assets_root.resolve())
    payload["tokenizer_pretrained_model_name_or_path"] = str(cosmos_assets_root.resolve())
    for key in (
        "vae_class_path",
        "diffusion_model_class_path",
        "diffusion_scheduler_class_path",
        "pipeline_class_path",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value) and ("/" in value or value.endswith(".py")):
            payload[key] = str((repo_root / value).resolve())
    diffusion_model = payload.setdefault("diffusion_model", {})
    if not isinstance(diffusion_model, dict):
        raise ValueError("expected 'diffusion_model' to be a YAML mapping")
    diffusion_model["model_path"] = str(gesim_checkpoint_path.resolve())
    dump_yaml(output_path, payload)


def main() -> int:
    args = parse_args()

    template_path = args.config_template_path.resolve()
    gesim_checkpoint_path = args.gesim_checkpoint_path.resolve()
    cosmos_assets_root = args.cosmos_assets_root.resolve()
    hf_token = resolve_hf_token(args)
    token_file_path = args.hf_token_file.expanduser().resolve() if args.hf_token_file is not None else None

    download_status: dict[str, Any] | None = None

    if args.download_cosmos_assets:
        try:
            download_result = download_cosmos_assets(
                repo_id=args.cosmos_repo_id,
                local_dir=cosmos_assets_root,
                hf_token=hf_token,
            )
            download_status = {
                "requested": True,
                "ok": True,
                **download_result,
            }
        except Exception as exc:  # pragma: no cover - operational error capture
            download_status = {
                "requested": True,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    checkpoint_status = inspect_gesim_checkpoint(
        gesim_checkpoint_path,
        verify_sha256=bool(args.verify_checkpoint_sha256),
    )
    cosmos_status = inspect_cosmos_assets(cosmos_assets_root, repo_id=args.cosmos_repo_id)

    runtime_ready = bool(checkpoint_status["ready"] and cosmos_status["ready"])
    rendered_config_path: str | None = None
    if args.output_config_path is not None and runtime_ready:
        output_config_path = args.output_config_path.resolve()
        render_runtime_config(
            template_path=template_path,
            output_path=output_config_path,
            cosmos_assets_root=cosmos_assets_root,
            gesim_checkpoint_path=gesim_checkpoint_path,
        )
        rendered_config_path = str(output_config_path)

    next_actions: list[str] = []
    if not checkpoint_status["ready"]:
        next_actions.append(
            "resume or restart the ModelScope checkpoint download until the expected size and checksum are reached"
        )
    if not cosmos_status["ready"]:
        next_actions.append("download the gated Cosmos tokenizer/text-encoder/VAE assets with an authorized Hugging Face account")
    if download_status and not download_status.get("ok", False):
        next_actions.append("request access to nvidia/Cosmos-Predict2-2B-Video2World for the Hugging Face account tied to the configured token")

    status = {
        "config_template_path": str(template_path),
        "output_config_path": rendered_config_path,
        "hugging_face": {
            "token_available": hf_token is not None,
            "token_source": (
                "environment"
                if args.hf_token
                else ("token_file" if token_file_path is not None and token_file_path.is_file() else "none")
            ),
            "token_file_path": (str(token_file_path) if token_file_path is not None else None),
        },
        "download_attempt": download_status,
        "gesim_checkpoint": checkpoint_status,
        "cosmos_assets": cosmos_status,
        "runtime_ready": runtime_ready,
        "next_actions": next_actions,
    }

    if args.status_json_path is not None:
        write_json(args.status_json_path.resolve(), status)

    print(json.dumps(status, indent=2, sort_keys=True))

    if args.require_ready and not runtime_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
