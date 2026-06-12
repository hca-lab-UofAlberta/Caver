#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


BACKGROUND = (248, 249, 251)
PANEL_BACKGROUND = (255, 255, 255)
TEXT = (22, 24, 29)
MUTED_TEXT = (90, 97, 106)
BORDER = (214, 219, 226)
ACCENT_INPUT = (51, 102, 204)
ACCENT_GESIM = (0, 128, 96)
ACCENT_GT = (204, 102, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a longer stitched GE-Sim-vs-LIBERO animation across multiple policy queries "
            "from the same Stage-E context."
        )
    )
    parser.add_argument("--trace-path", required=True, help="Stage-E chunk trace JSONL path.")
    parser.add_argument("--context-id", required=True, help="Trace context id.")
    parser.add_argument("--query-indices", required=True, help="Comma-separated 1-based query indices.")
    parser.add_argument(
        "--rerun-root",
        required=True,
        help="Root directory produced by rerun_gesim_trace_queries.sh containing query_XXX/gesim_rerun outputs.",
    )
    parser.add_argument("--gif-output-path", required=True, help="Animated GIF output path.")
    parser.add_argument("--mp4-output-path", default=None, help="Optional MP4 output path.")
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=350,
        help="Per-frame duration for the output animation.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def parse_query_indices(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"invalid query index: {value}")
        values.append(value)
    if not values:
        raise ValueError("query index list is empty")
    return values


def load_selected_trace_record(trace_path: Path, *, context_id: str, policy_query_index: int) -> dict:
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if context_id not in line:
                continue
            record = json.loads(line)
            if record.get("context_id") != context_id:
                continue
            if int(record.get("policy_query_index", -1)) != policy_query_index:
                continue
            return record
    raise FileNotFoundError(
        f"no trace record found in {trace_path} for context_id={context_id} policy_query_index={policy_query_index}"
    )


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return payload


def load_generated_frames(output_dir: Path) -> list[Image.Image]:
    tensor_path = output_dir / "video.pt"
    if tensor_path.exists():
        if torch is None:
            raise RuntimeError("video.pt exists but torch is unavailable; run under scripts/env/with_gesim_infer.sh")
        raw = torch.load(tensor_path, map_location="cpu")
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError(f"expected video tensor with shape (t,h,w,3), got {raw.shape}")
        return [Image.fromarray(frame.astype("uint8"), mode="RGB") for frame in raw]

    gif_path = output_dir / "video.gif"
    if gif_path.exists():
        with Image.open(gif_path) as image:
            return [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]

    raise FileNotFoundError(f"expected video.pt or video.gif under {output_dir}")


def crop_generated_view(
    frames: list[Image.Image],
    *,
    frame_height: int,
    frame_width: int,
    view_index: int,
) -> list[Image.Image]:
    left = view_index * frame_width
    upper = 0
    right = left + frame_width
    lower = frame_height
    return [frame.crop((left, upper, right, lower)) for frame in frames]


def array_to_image(value: list) -> Image.Image:
    if torch is None:
        raise RuntimeError("torch is unavailable; run under scripts/env/with_gesim_infer.sh")
    tensor = torch.as_tensor(value, dtype=torch.uint8)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"expected image array with shape (h,w,3), got {tuple(tensor.shape)}")
    return Image.fromarray(tensor.cpu().numpy(), mode="RGB")


def load_bundle_history(bundle_dir: Path, camera_name: str) -> list[Image.Image]:
    image_dir = bundle_dir / f"{camera_name}_color"
    frames: list[Image.Image] = []
    for image_path in sorted(image_dir.glob("*.png")):
        with Image.open(image_path) as image:
            frames.append(image.convert("RGB"))
    if not frames:
        raise FileNotFoundError(f"no bundle frames found under {image_dir}")
    return frames


def fit_tile(image: Image.Image, *, tile_width: int, tile_height: int) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), (tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height), PANEL_BACKGROUND)
    tile.paste(fitted, ((tile_width - fitted.width) // 2, (tile_height - fitted.height) // 2))
    return tile


def draw_labeled_panel(
    canvas: Image.Image,
    *,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    accent: tuple[int, int, int],
    image: Image.Image,
    title_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    draw.rounded_rectangle((x, y, x + width, y + height), radius=12, fill=PANEL_BACKGROUND, outline=BORDER, width=2)
    draw.rounded_rectangle((x + 14, y + 14, x + 24, y + 24), radius=4, fill=accent)
    draw.text((x + 32, y + 8), title, fill=TEXT, font=title_font)
    fitted = fit_tile(image, tile_width=width - 24, tile_height=height - 52)
    canvas.paste(fitted, (x + 12, y + 40))


def render_animation_frame(
    *,
    context_id: str,
    query_index: int,
    step_index: int,
    total_steps: int,
    input_head: Image.Image,
    input_wrist: Image.Image,
    generated_head: Image.Image,
    generated_wrist: Image.Image,
    future_head: Image.Image,
    future_wrist: Image.Image,
) -> Image.Image:
    margin = 24
    top_margin = 22
    panel_gap = 18
    title_font = load_font(24)
    subtitle_font = load_font(15)
    panel_font = load_font(17)

    panel_width = 320
    panel_height = 220
    canvas_width = margin * 2 + panel_width * 2 + panel_gap
    canvas_height = top_margin + 86 + panel_height * 3 + panel_gap * 2 + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, top_margin), "GE-Sim vs Logged LIBERO: Longer Trajectory", fill=TEXT, font=title_font)
    draw.text(
        (margin, top_margin + 34),
        f"context={context_id} | query={query_index} | future step {step_index + 1}/{total_steps}",
        fill=MUTED_TEXT,
        font=subtitle_font,
    )
    draw.text(
        (margin, top_margin + 56),
        "Blue: current LIBERO input frame | Green: GE-Sim generated future | Orange: logged LIBERO future",
        fill=MUTED_TEXT,
        font=subtitle_font,
    )

    y0 = top_margin + 86
    x0 = margin
    x1 = margin + panel_width + panel_gap
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=x0,
        y=y0,
        width=panel_width,
        height=panel_height,
        title="Current input: head",
        accent=ACCENT_INPUT,
        image=input_head,
        title_font=panel_font,
    )
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=x1,
        y=y0,
        width=panel_width,
        height=panel_height,
        title="Current input: wrist",
        accent=ACCENT_INPUT,
        image=input_wrist,
        title_font=panel_font,
    )

    y1 = y0 + panel_height + panel_gap
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=x0,
        y=y1,
        width=panel_width,
        height=panel_height,
        title="GE-Sim head",
        accent=ACCENT_GESIM,
        image=generated_head,
        title_font=panel_font,
    )
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=x1,
        y=y1,
        width=panel_width,
        height=panel_height,
        title="LIBERO head",
        accent=ACCENT_GT,
        image=future_head,
        title_font=panel_font,
    )

    y2 = y1 + panel_height + panel_gap
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=x0,
        y=y2,
        width=panel_width,
        height=panel_height,
        title="GE-Sim wrist",
        accent=ACCENT_GESIM,
        image=generated_wrist,
        title_font=panel_font,
    )
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=x1,
        y=y2,
        width=panel_width,
        height=panel_height,
        title="LIBERO wrist",
        accent=ACCENT_GT,
        image=future_wrist,
        title_font=panel_font,
    )
    return canvas


def write_mp4_from_gif(*, gif_path: Path, mp4_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not available on PATH")
    ensure_parent(mp4_path)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(gif_path),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(mp4_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace_path).resolve()
    rerun_root = Path(args.rerun_root).resolve()
    gif_output_path = Path(args.gif_output_path).resolve()
    mp4_output_path = Path(args.mp4_output_path).resolve() if args.mp4_output_path else None
    query_indices = parse_query_indices(args.query_indices)

    frames: list[Image.Image] = []
    for query_index in query_indices:
        record = load_selected_trace_record(
            trace_path,
            context_id=args.context_id,
            policy_query_index=query_index,
        )
        metadata_path = Path(record["selected_provider_aux"]["metadata_path"]).resolve()
        bundle_dir = metadata_path.parent
        gesim_output_dir = rerun_root / f"query_{query_index:03d}" / "gesim_rerun"
        if not gesim_output_dir.exists():
            raise FileNotFoundError(f"missing rerun output dir: {gesim_output_dir}")

        runtime_config = read_yaml(gesim_output_dir / "gesim_runtime.yaml")
        frame_height, frame_width = [int(value) for value in runtime_config["data"]["train"]["sample_size"]]
        valid_cam = [str(value) for value in runtime_config["data"]["train"]["valid_cam"]]
        if "head" not in valid_cam or "hand_right" not in valid_cam:
            raise ValueError(f"expected GE-Sim valid_cam to include head and hand_right, got {valid_cam}")

        generated_frames = load_generated_frames(gesim_output_dir)
        generated_head = crop_generated_view(
            generated_frames,
            frame_height=frame_height,
            frame_width=frame_width,
            view_index=valid_cam.index("head"),
        )
        generated_wrist = crop_generated_view(
            generated_frames,
            frame_height=frame_height,
            frame_width=frame_width,
            view_index=valid_cam.index("hand_right"),
        )
        future_head = [array_to_image(step["image"]) for step in record["next_obs_sequence"]]
        future_wrist = [array_to_image(step["wrist_image"]) for step in record["next_obs_sequence"]]
        frame_count = min(len(generated_head), len(generated_wrist), len(future_head), len(future_wrist))
        if frame_count == 0:
            continue

        input_head = load_bundle_history(bundle_dir, "head")[-1]
        input_wrist = load_bundle_history(bundle_dir, "hand_right")[-1]
        for step_index in range(frame_count):
            frames.append(
                render_animation_frame(
                    context_id=args.context_id,
                    query_index=query_index,
                    step_index=step_index,
                    total_steps=frame_count,
                    input_head=input_head,
                    input_wrist=input_wrist,
                    generated_head=generated_head[step_index],
                    generated_wrist=generated_wrist[step_index],
                    future_head=future_head[step_index],
                    future_wrist=future_wrist[step_index],
                )
            )

    if not frames:
        raise RuntimeError("no animation frames were produced")

    ensure_parent(gif_output_path)
    frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"wrote gif: {gif_output_path}")

    if mp4_output_path is not None:
        write_mp4_from_gif(gif_path=gif_output_path, mp4_path=mp4_output_path)
        print(f"wrote mp4: {mp4_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
