#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import safetensors
import safetensors.torch
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an RLinf FSDP actor checkpoint into an OpenPI-compatible "
            "PyTorch checkpoint directory with model.safetensors."
        )
    )
    parser.add_argument(
        "--actor-checkpoint-dir",
        required=True,
        help=(
            "RLinf actor checkpoint directory. Expected to contain "
            "model_state_dict/full_weights.pt."
        ),
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help=(
            "Reference OpenPI PyTorch checkpoint directory. "
            "Used for key allowlisting and asset/config copying."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Export directory to create/update.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary output.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing base checkpoint keys instead of failing.",
    )
    return parser.parse_args()


def candidate_key_variants(key: str) -> Iterable[str]:
    yield key
    prefixes = ("module.", "model.", "_orig_mod.", "actor.", "student_model.")
    for prefix in prefixes:
        if key.startswith(prefix):
            yield key[len(prefix) :]


def load_full_state_dict(actor_checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    weights_path = actor_checkpoint_dir / "model_state_dict" / "full_weights.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"RLinf actor checkpoint missing model_state_dict/full_weights.pt under {actor_checkpoint_dir}"
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")
    return state


def load_base_key_set(base_model_path: Path) -> set[str]:
    weight_path = base_model_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Base checkpoint missing model.safetensors under {base_model_path}")
    with safetensors.safe_open(str(weight_path), framework="pt") as handle:
        return set(handle.keys())


def copy_file_without_metadata(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f".{dst.name}.tmp")
    if tmp_dst.exists():
        tmp_dst.unlink()
    shutil.copyfile(src, tmp_dst)
    tmp_dst.replace(dst)


def symlink_or_copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, copy_function=shutil.copyfile)
        else:
            copy_file_without_metadata(src, dst)


def main() -> int:
    args = parse_args()

    actor_checkpoint_dir = Path(args.actor_checkpoint_dir).resolve()
    base_model_path = Path(args.base_model_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve() if args.summary_path else None

    full_state = load_full_state_dict(actor_checkpoint_dir)
    base_keys = load_base_key_set(base_model_path)

    exported_state: dict[str, torch.Tensor] = {}
    unmatched_keys: list[str] = []

    for raw_key, raw_value in full_state.items():
        if not torch.is_tensor(raw_value):
            continue
        matched_key = None
        for candidate_key in candidate_key_variants(raw_key):
            if candidate_key in base_keys:
                matched_key = candidate_key
                break
        if matched_key is None:
            unmatched_keys.append(raw_key)
            continue
        if matched_key in exported_state:
            raise ValueError(
                f"Multiple checkpoint entries resolved to the same base key {matched_key!r}"
            )
        exported_state[matched_key] = raw_value.detach().cpu()

    missing_base_keys = sorted(base_keys.difference(exported_state.keys()))
    if missing_base_keys and not args.allow_missing:
        sample = missing_base_keys[:20]
        raise ValueError(
            f"Exported checkpoint is missing {len(missing_base_keys)} base keys; sample={sample}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(exported_state, str(output_path / "model.safetensors"))

    config_src = base_model_path / "config.json"
    if config_src.is_file():
        symlink_or_copy_path(config_src, output_path / "config.json")

    assets_src = base_model_path / "assets"
    if assets_src.is_dir():
        symlink_or_copy_path(assets_src, output_path / "assets")

    summary = {
        "actor_checkpoint_dir": str(actor_checkpoint_dir),
        "base_model_path": str(base_model_path),
        "output_path": str(output_path),
        "base_key_count": len(base_keys),
        "exported_key_count": len(exported_state),
        "missing_base_key_count": len(missing_base_keys),
        "missing_base_key_sample": missing_base_keys[:20],
        "ignored_checkpoint_key_count": len(unmatched_keys),
        "ignored_checkpoint_key_sample": unmatched_keys[:20],
    }

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
