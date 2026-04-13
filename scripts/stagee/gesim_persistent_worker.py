#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

from einops import rearrange
import numpy as np
import torch

import gesim_video_gen_examples.infer_gesim as gesim_infer

from summarize_gesim_output import summarize_provider_video
from summarize_gesim_output import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent GE-Sim worker that loads the model once and serves inference requests over JSONL."
    )
    parser.add_argument("--config-path", required=True, help="Rendered GE-Sim runtime YAML path")
    parser.add_argument(
        "--runtime-status-path",
        default=None,
        help="Optional rendered GE-Sim runtime status JSON copied into each request output directory.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device string. Defaults to cuda.")
    parser.add_argument(
        "--weight-dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Model weight dtype used for the persistent inference pipeline.",
    )
    return parser.parse_args()


def resolve_weight_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported weight dtype: {name}")


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def current_rss_gib() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        return float(parts[1]) / (1024.0 * 1024.0)
    return None


def trim_process_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def materialize_runtime_artifacts(
    *,
    output_dir: Path,
    config_path: Path,
    runtime_status_path: Path | None,
    runtime_status_payload: dict[str, Any] | None,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path = output_dir / "gesim_runtime.yaml"
    shutil.copyfile(config_path, config_copy_path)

    runtime_status_copy_path: Path | None = None
    if runtime_status_path is not None and runtime_status_path.exists():
        runtime_status_copy_path = output_dir / "gesim_runtime_status.json"
        shutil.copyfile(runtime_status_path, runtime_status_copy_path)
    elif runtime_status_payload is not None:
        runtime_status_copy_path = output_dir / "gesim_runtime_status.json"
        write_json(runtime_status_copy_path, runtime_status_payload)
    return config_copy_path, runtime_status_copy_path


class PersistentGesimWorker:
    def __init__(
        self,
        *,
        config_path: Path,
        runtime_status_path: Path | None,
        device: str,
        weight_dtype: torch.dtype,
    ) -> None:
        self.config_path = config_path
        self.runtime_status_path = runtime_status_path
        self.runtime_status_payload = load_json(runtime_status_path)
        self.device = device
        self.weight_dtype = weight_dtype
        self.request_counter = 0
        self.write_preview_artifacts = str(os.environ.get("CAVER_GESIM_WRITE_PREVIEW_ARTIFACTS", "0")).lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        self.write_video_tensor_artifacts = str(os.environ.get("CAVER_GESIM_WRITE_VIDEO_TENSOR", "0")).lower() not in {
            "",
            "0",
            "false",
            "no",
        }

        with contextlib.redirect_stdout(sys.stderr):
            self.runtime_args = gesim_infer.load_config(str(config_path))
            (
                self.tokenizer,
                self.text_encoder,
                self.vae,
                self.diffusion_model,
                self.scheduler,
                self.pipe,
            ) = gesim_infer.prepare_model(self.runtime_args, dtype=weight_dtype, device=device)

        self.sample_height = int(self.runtime_args.data["train"]["sample_size"][0])
        self.sample_width = int(self.runtime_args.data["train"]["sample_size"][1])
        self.valid_cam = [str(value) for value in self.runtime_args.data["train"]["valid_cam"]]
        self.valid_cams_with_color = [f"{camera_name}_color" for camera_name in self.valid_cam]
        self.n_previous = int(self.runtime_args.data["train"]["n_previous"])
        self.chunk = int(self.runtime_args.data["train"]["chunk"])

    def infer_bundle(
        self,
        *,
        bundle_dir: Path,
        output_dir: Path,
        prompt: str,
    ) -> dict[str, Any]:
        started = time.time()
        self.request_counter += 1
        config_copy_path, runtime_status_copy_path = materialize_runtime_artifacts(
            output_dir=output_dir,
            config_path=self.config_path,
            runtime_status_path=self.runtime_status_path,
            runtime_status_payload=self.runtime_status_payload,
        )
        provider_summary_path = output_dir / "provider_summary.json"
        video_tensor_path = output_dir / "video.pt"

        try:
            with contextlib.redirect_stdout(sys.stderr):
                summary_payload = self._run_inference(
                    bundle_dir=bundle_dir,
                    output_dir=output_dir,
                    provider_summary_path=provider_summary_path,
                    video_tensor_path=video_tensor_path,
                    config_copy_path=config_copy_path,
                    prompt=prompt,
                )
        finally:
            trim_process_memory()

        if self.request_counter == 1 or self.request_counter % 10 == 0:
            rss_gib = current_rss_gib()
            rss_suffix = f" rss_gib={rss_gib:.2f}" if rss_gib is not None else ""
            print(
                f"[gesim-worker] request={self.request_counter} duration_sec={time.time() - started:.2f}{rss_suffix}",
                file=sys.stderr,
                flush=True,
            )

        return {
            "event": "response",
            "request_id": None,
            "inference_status": "completed",
            "inference_returncode": 0,
            "inference_output_dir": str(output_dir),
            "provider_summary_path": str(provider_summary_path),
            "provider_summary": summary_payload,
            "provider_error": None,
            "duration_sec": time.time() - started,
            "runtime_config_path": str(config_copy_path),
            "runtime_status_path": (str(runtime_status_copy_path) if runtime_status_copy_path is not None else None),
        }

    def _run_inference(
        self,
        *,
        bundle_dir: Path,
        output_dir: Path,
        provider_summary_path: Path,
        video_tensor_path: Path,
        config_copy_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        args = self.runtime_args

        obs = None
        ori_sizes = None
        extrinsics = None
        intrinsics = None
        actions = None
        extrinsics_tensor = None
        intrinsics_tensor = None
        actions_tensor = None
        trajs = None
        original_trajs = None
        rays_o = None
        rays_d = None
        rays = None
        cond_to_concat = None
        chunk_conditions = None
        videos = None
        preds = None
        video_to_save = None
        frames_uint8 = None

        try:
            obs, ori_sizes = gesim_infer.load_images(
                args,
                str(bundle_dir),
                self.valid_cams_with_color,
                size=(self.sample_width, self.sample_height),
            )

            view_count, _channels, _frames, frame_height, frame_width = obs.shape
            extrinsics, intrinsics = gesim_infer.load_cam_infos(
                str(bundle_dir),
                str(bundle_dir),
                self.valid_cam,
                orisize=ori_sizes,
                size=args.data["train"]["sample_size"],
            )
            actions = np.load(bundle_dir / "actions.npy")

            extrinsics_tensor = torch.FloatTensor(extrinsics)
            intrinsics_tensor = torch.FloatTensor(intrinsics)
            actions_tensor = torch.FloatTensor(actions)

            trajs = gesim_infer.get_traj_maps(
                actions_tensor,
                torch.linalg.inv(extrinsics_tensor),
                extrinsics_tensor,
                intrinsics_tensor,
                args.data["train"]["sample_size"],
                radius_gen_func=gesim_infer.simple_radius_gen_func,
            )
            trajs = trajs * 2 - 1
            original_trajs = trajs.clone()

            rays_o, rays_d = gesim_infer.get_ray_maps(
                intrinsics_tensor.unsqueeze(dim=1)
                .repeat(1, extrinsics_tensor.shape[1], 1, 1)
                .reshape(-1, 3, 3),
                extrinsics_tensor.reshape(-1, 4, 4),
                self.sample_height,
                self.sample_width,
            )
            rays = torch.cat((rays_o, rays_d), dim=-1).reshape(
                trajs.shape[1],
                trajs.shape[2],
                rays_o.shape[1],
                rays_o.shape[2],
                -1,
            )
            rays = rays.permute(4, 0, 1, 2, 3)
            cond_to_concat = torch.cat((trajs, rays), dim=0)

            negative_prompt = (
                "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
                "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
                "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
                "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
                "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
                "Overall, the video is of poor quality."
            )

            nall = trajs.shape[2]
            nchunk = int(np.ceil((nall - self.n_previous) / self.chunk))
            videos = obs.clone()
            chunk_conditions = torch.cat(
                (
                    cond_to_concat[:, :, : self.n_previous],
                    cond_to_concat[:, :, self.n_previous : self.n_previous + self.chunk],
                ),
                dim=2,
            )

            with torch.inference_mode():
                for chunk_index in range(nchunk):
                    preds = self.pipe.infer(
                        video=obs.permute(0, 2, 1, 3, 4).to(self.device),
                        cond_to_concat=rearrange(chunk_conditions, "c v t h w -> v c t h w"),
                        prompt=[prompt],
                        negative_prompt=negative_prompt,
                        height=frame_height,
                        width=frame_width,
                        n_view=view_count,
                        num_frames=self.chunk,
                        num_inference_steps=args.num_inference_step,
                        n_prev=self.n_previous,
                        guidance_scale=1.0,
                        merge_view_into_width=False,
                        output_type="pt",
                        postprocess_video=False,
                    )["frames"]

                    videos = torch.cat((videos, preds.data.cpu()), dim=2)
                    videos = torch.clamp(videos, min=-1, max=1)

                    if chunk_index >= nchunk - 1:
                        continue

                    current_frames = videos.shape[2]
                    mem_idxes = list(np.linspace(0, current_frames - 1, self.n_previous).astype(np.int16))
                    obs = videos[:, :, mem_idxes].clone()
                    next_chunk_conditions = torch.cat(
                        (
                            cond_to_concat[:, :, mem_idxes],
                            cond_to_concat[
                                :,
                                :,
                                self.n_previous + (chunk_index + 1) * self.chunk : self.n_previous + (chunk_index + 2) * self.chunk,
                            ],
                        ),
                        dim=2,
                    )
                    if next_chunk_conditions.shape[2] < self.chunk + self.n_previous:
                        next_chunk_conditions = torch.cat(
                            [next_chunk_conditions] + [next_chunk_conditions[:, :, -1:]] * (
                                self.chunk - next_chunk_conditions.shape[2] - self.n_previous
                            ),
                            dim=2,
                        )
                    chunk_conditions = next_chunk_conditions

            video_to_save = torch.cat(
                (
                    rearrange(videos[:, :, : original_trajs.shape[2]], "v c t h w -> c t h (v w)", v=view_count),
                    rearrange(original_trajs, "c v t h w -> c t h (v w)", v=view_count),
                ),
                dim=2,
            )

            frames_uint8 = (
                ((torch.clamp(video_to_save, min=-1.0, max=1.0) + 1.0) / 2.0 * 255.0)
                .to(torch.uint8)
                .permute(1, 2, 3, 0)
                .contiguous()
                .cpu()
            )
            video_tensor_path_value: str | None = None
            if self.write_video_tensor_artifacts:
                torch.save(frames_uint8, video_tensor_path)
                video_tensor_path_value = str(video_tensor_path)
            if self.write_preview_artifacts:
                default_fps = 30
                if "action_chunk" in args.data["train"]:
                    video_fps = default_fps // (args.data["train"]["action_chunk"] // args.data["train"]["chunk"])
                else:
                    video_fps = default_fps
                gesim_infer.save_video(
                    video_to_save,
                    str(output_dir / "video.mp4"),
                    fps=video_fps,
                )

            summary = summarize_provider_video(
                frames_uint8.numpy(),
                frame_height=self.sample_height,
                frame_width=self.sample_width,
                view_count=len(self.valid_cam),
            )
            summary.update(
                {
                    "output_dir": str(output_dir),
                    "config_path": str(config_copy_path),
                    "video_tensor_path": video_tensor_path_value,
                    "summary_path": str(provider_summary_path),
                    "valid_cam": list(self.valid_cam),
                    "preview_artifacts_written": bool(self.write_preview_artifacts),
                    "video_tensor_written": bool(self.write_video_tensor_artifacts),
                }
            )
            write_json(provider_summary_path, summary)
            return summary
        finally:
            del (
                obs,
                ori_sizes,
                extrinsics,
                intrinsics,
                actions,
                extrinsics_tensor,
                intrinsics_tensor,
                actions_tensor,
                trajs,
                original_trajs,
                rays_o,
                rays_d,
                rays,
                cond_to_concat,
                chunk_conditions,
                videos,
                preds,
                video_to_save,
                frames_uint8,
            )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config_path).resolve()
    runtime_status_path = Path(args.runtime_status_path).resolve() if args.runtime_status_path else None

    worker_started = time.time()
    worker = PersistentGesimWorker(
        config_path=config_path,
        runtime_status_path=runtime_status_path,
        device=args.device,
        weight_dtype=resolve_weight_dtype(args.weight_dtype),
    )
    emit(
        {
            "event": "ready",
            "config_path": str(config_path),
            "runtime_status_path": (str(runtime_status_path) if runtime_status_path is not None else None),
            "device": args.device,
            "weight_dtype": args.weight_dtype,
            "pid": os.getpid(),
            "startup_sec": time.time() - worker_started,
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "preview_artifacts_written": bool(worker.write_preview_artifacts),
            "video_tensor_written": bool(worker.write_video_tensor_artifacts),
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id: str | None = None
        output_dir_value: str | None = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must decode to a JSON object")
            request_id = str(request.get("request_id")) if request.get("request_id") is not None else None
            action = str(request.get("action", "infer"))
            if action == "shutdown":
                emit({"event": "shutdown", "request_id": request_id, "status": "ok"})
                return 0
            if action != "infer":
                raise ValueError(f"unsupported request action: {action}")

            bundle_dir = Path(str(request["bundle_dir"])).resolve()
            output_dir = Path(str(request["output_dir"])).resolve()
            output_dir_value = str(output_dir)
            prompt = str(request["prompt"])

            response = worker.infer_bundle(bundle_dir=bundle_dir, output_dir=output_dir, prompt=prompt)
            response["request_id"] = request_id
            emit(response)
        except Exception as exc:  # noqa: BLE001
            emit(
                {
                    "event": "response",
                    "request_id": request_id,
                    "inference_status": "error",
                    "inference_returncode": 1,
                    "inference_output_dir": output_dir_value,
                    "provider_summary_path": (
                        f"{output_dir_value}/provider_summary.json" if output_dir_value is not None else None
                    ),
                    "provider_summary": None,
                    "provider_error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
