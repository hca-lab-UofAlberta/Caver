#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROVIDER_SUMMARY_VERSION = "gesim_future_summary_v1"
PROVIDER_SUMMARY_DIM = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a GE-Sim inference output directory into a compact provider feature vector."
    )
    parser.add_argument("--output-dir", required=True, help="GE-Sim output directory containing video.pt")
    parser.add_argument("--config-path", required=True, help="Rendered GE-Sim runtime YAML used for inference")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional output JSON path. Defaults to <output-dir>/provider_summary.json",
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
        raise ValueError(f"expected YAML mapping in {path}")
    return payload


def summarize_provider_video(
    frames_uint8: np.ndarray,
    *,
    frame_height: int,
    frame_width: int,
    view_count: int,
) -> dict[str, Any]:
    if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3:
        raise ValueError(f"expected frames with shape (t,h,w,3), got {frames_uint8.shape}")
    if frame_height <= 0 or frame_width <= 0 or view_count <= 0:
        raise ValueError("frame geometry must be positive")
    if frames_uint8.shape[1] < frame_height:
        raise ValueError("video frame height is smaller than configured GE-Sim sample height")
    if frames_uint8.shape[2] < frame_width * view_count:
        raise ValueError("video frame width is smaller than configured multi-view width")

    generated = frames_uint8[:, :frame_height, : frame_width * view_count, :]
    views = np.stack(
        [generated[:, :, index * frame_width : (index + 1) * frame_width, :] for index in range(view_count)],
        axis=1,
    ).astype(np.float32) / 255.0
    temporal_delta = np.abs(np.diff(views, axis=0)) if views.shape[0] > 1 else np.zeros_like(views[:1])
    last_first_delta = np.abs(views[-1:] - views[:1])

    per_view_mean = views.mean(axis=(0, 2, 3, 4))
    per_view_temporal = temporal_delta.mean(axis=(0, 2, 3, 4))
    per_view_last_first = last_first_delta.mean(axis=(0, 2, 3, 4))

    pixel_mean = float(views.mean())
    pixel_std = float(views.std())
    temporal_delta_mean = float(temporal_delta.mean()) if temporal_delta.size else 0.0
    temporal_delta_std = float(per_view_temporal.std()) if per_view_temporal.size else 0.0
    last_first_delta_mean = float(last_first_delta.mean()) if last_first_delta.size else 0.0
    last_first_delta_std = float(per_view_last_first.std()) if per_view_last_first.size else 0.0
    cross_view_mean_std = float(per_view_mean.std()) if per_view_mean.size else 0.0
    cross_view_temporal_std = float(per_view_temporal.std()) if per_view_temporal.size else 0.0
    provider_value_proxy = float(0.5 * temporal_delta_mean + 0.5 * last_first_delta_mean)
    provider_uncertainty_proxy = float(0.5 * temporal_delta_std + 0.5 * cross_view_temporal_std)

    summary_vector = [
        pixel_mean,
        pixel_std,
        temporal_delta_mean,
        temporal_delta_std,
        last_first_delta_mean,
        last_first_delta_std,
        cross_view_mean_std,
        cross_view_temporal_std,
        provider_value_proxy,
        provider_uncertainty_proxy,
    ]

    return {
        "summary_version": PROVIDER_SUMMARY_VERSION,
        "summary_dim": PROVIDER_SUMMARY_DIM,
        "summary_vector": [float(value) for value in summary_vector],
        "provider_value_proxy": provider_value_proxy,
        "provider_uncertainty_proxy": provider_uncertainty_proxy,
        "frame_count": int(views.shape[0]),
        "view_count": int(view_count),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "temporal_delta_mean": temporal_delta_mean,
        "temporal_delta_std": temporal_delta_std,
        "last_first_delta_mean": last_first_delta_mean,
        "last_first_delta_std": last_first_delta_std,
        "cross_view_mean_std": cross_view_mean_std,
        "cross_view_temporal_std": cross_view_temporal_std,
        "per_view_mean": [float(value) for value in per_view_mean.tolist()],
        "per_view_temporal_delta_mean": [float(value) for value in per_view_temporal.tolist()],
        "per_view_last_first_delta_mean": [float(value) for value in per_view_last_first.tolist()],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config_path).resolve()
    summary_path = (
        Path(args.summary_path).resolve() if args.summary_path else (output_dir / "provider_summary.json").resolve()
    )

    video_tensor_path = output_dir / "video.pt"
    if not video_tensor_path.exists():
        raise SystemExit(f"error: missing GE-Sim tensor dump: {video_tensor_path}")

    config = load_yaml(config_path)
    sample_height, sample_width = [int(value) for value in config["data"]["train"]["sample_size"]]
    valid_cam = list(config["data"]["train"]["valid_cam"])
    view_count = len(valid_cam)

    frames = torch.load(video_tensor_path, map_location="cpu")
    frames_uint8 = np.asarray(frames, dtype=np.uint8)
    summary = summarize_provider_video(
        frames_uint8,
        frame_height=sample_height,
        frame_width=sample_width,
        view_count=view_count,
    )
    summary.update(
        {
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "video_tensor_path": str(video_tensor_path),
            "summary_path": str(summary_path),
            "valid_cam": [str(value) for value in valid_cam],
        }
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
