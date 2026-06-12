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
            "Render an animated GE-Sim-vs-LIBERO comparison for a selected Stage-E chunk trace."
        )
    )
    parser.add_argument("--trace-path", required=True, help="Stage-E chunk trace JSONL path.")
    parser.add_argument("--context-id", required=True, help="Trace context id.")
    parser.add_argument("--policy-query-index", type=int, required=True, help="1-based policy query index.")
    parser.add_argument("--gesim-output-dir", required=True, help="GE-Sim output directory with video.pt or video.gif.")
    parser.add_argument("--gif-output-path", required=True, help="Animated GIF output path.")
    parser.add_argument("--mp4-output-path", default=None, help="Optional MP4 output path.")
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=700,
        help="Per-frame duration for the GIF animation.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return payload


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def load_selected_trace_record(trace_path: Path, *, context_id: str, policy_query_index: int) -> dict:
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("context_id") == context_id and int(record.get("policy_query_index", -1)) == policy_query_index:
                return record
    raise FileNotFoundError(
        f"no trace record found in {trace_path} for context_id={context_id} policy_query_index={policy_query_index}"
    )


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


def fit_tile(image: Image.Image, *, tile_width: int, tile_height: int) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), (tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height), PANEL_BACKGROUND)
    tile.paste(fitted, ((tile_width - fitted.width) // 2, (tile_height - fitted.height) // 2))
    return tile


def build_history_strip(
    frames: list[Image.Image],
    *,
    tile_width: int,
    tile_height: int,
    gap: int,
    label_prefix: str,
) -> Image.Image:
    label_font = load_font(13)
    width = len(frames) * tile_width + max(0, len(frames) - 1) * gap
    strip = Image.new("RGB", (width, tile_height + 18), BACKGROUND)
    draw = ImageDraw.Draw(strip)
    cursor_x = 0
    for index, frame in enumerate(frames):
        tile = fit_tile(frame, tile_width=tile_width, tile_height=tile_height)
        strip.paste(tile, (cursor_x, 18))
        draw.rectangle((cursor_x, 18, cursor_x + tile_width - 1, 18 + tile_height - 1), outline=BORDER, width=1)
        draw.text((cursor_x, 0), f"{label_prefix}{index}", fill=MUTED_TEXT, font=label_font)
        cursor_x += tile_width + gap
    return strip


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
    history_head_strip: Image.Image,
    history_wrist_strip: Image.Image,
    generated_head_frame: Image.Image,
    future_head_frame: Image.Image,
    generated_wrist_frame: Image.Image,
    future_wrist_frame: Image.Image,
    record: dict,
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    margin = 24
    top_margin = 24
    history_gap = 18
    panel_gap = 20
    title_font = load_font(26)
    subtitle_font = load_font(16)
    panel_font = load_font(18)
    step_font = load_font(20)

    panel_width = 320
    panel_height = 260
    canvas_width = margin * 2 + panel_width * 2 + panel_gap
    history_width = max(history_head_strip.width, history_wrist_strip.width)
    history_x_right = canvas_width - margin - history_width
    canvas_height = top_margin + 100 + history_head_strip.height + history_gap + panel_height * 2 + panel_gap + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, top_margin), "GE-Sim vs Logged LIBERO Future", fill=TEXT, font=title_font)
    subtitle = (
        f"context={record['context_id']} | query={record['policy_query_index']} | "
        f"candidate={record['selector']['selected_candidate_index']} | frame {frame_index + 1}/{total_frames}"
    )
    draw.text((margin, top_margin + 36), subtitle, fill=MUTED_TEXT, font=subtitle_font)
    draw.text(
        (margin, top_margin + 60),
        "Blue: LIBERO input history | Green: GE-Sim generated future | Orange: logged LIBERO future",
        fill=MUTED_TEXT,
        font=subtitle_font,
    )

    history_y = top_margin + 92
    draw.text((margin, history_y - 6), "Input history: head", fill=TEXT, font=step_font)
    canvas.paste(history_head_strip, (margin, history_y + 20))
    draw.text((history_x_right, history_y - 6), "Input history: wrist", fill=TEXT, font=step_font)
    canvas.paste(history_wrist_strip, (history_x_right, history_y + 20))

    panel_y = history_y + history_head_strip.height + history_gap + 28
    left_x = margin
    right_x = margin + panel_width + panel_gap

    draw_labeled_panel(
        canvas,
        draw=draw,
        x=left_x,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        title="GE-Sim head",
        accent=ACCENT_GESIM,
        image=generated_head_frame,
        title_font=panel_font,
    )
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=right_x,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        title="LIBERO head",
        accent=ACCENT_GT,
        image=future_head_frame,
        title_font=panel_font,
    )
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=left_x,
        y=panel_y + panel_height + panel_gap,
        width=panel_width,
        height=panel_height,
        title="GE-Sim wrist",
        accent=ACCENT_GESIM,
        image=generated_wrist_frame,
        title_font=panel_font,
    )
    draw_labeled_panel(
        canvas,
        draw=draw,
        x=right_x,
        y=panel_y + panel_height + panel_gap,
        width=panel_width,
        height=panel_height,
        title="LIBERO wrist",
        accent=ACCENT_GT,
        image=future_wrist_frame,
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
    gesim_output_dir = Path(args.gesim_output_dir).resolve()
    gif_output_path = Path(args.gif_output_path).resolve()
    mp4_output_path = Path(args.mp4_output_path).resolve() if args.mp4_output_path else None

    record = load_selected_trace_record(
        trace_path,
        context_id=args.context_id,
        policy_query_index=args.policy_query_index,
    )
    selected_provider_aux = dict(record["selected_provider_aux"])
    bundle_dir = Path(selected_provider_aux["metadata_path"]).resolve().parent
    runtime_config = read_yaml(gesim_output_dir / "gesim_runtime.yaml")
    sample_height, sample_width = [int(value) for value in runtime_config["data"]["train"]["sample_size"]]
    valid_cam = [str(value) for value in runtime_config["data"]["train"]["valid_cam"]]
    if "head" not in valid_cam or "hand_right" not in valid_cam:
        raise ValueError(f"expected GE-Sim valid_cam to include head and hand_right, got {valid_cam}")

    generated_frames = load_generated_frames(gesim_output_dir)
    generated_head = crop_generated_view(
        generated_frames,
        frame_height=sample_height,
        frame_width=sample_width,
        view_index=valid_cam.index("head"),
    )
    generated_wrist = crop_generated_view(
        generated_frames,
        frame_height=sample_height,
        frame_width=sample_width,
        view_index=valid_cam.index("hand_right"),
    )
    future_head = [array_to_image(step["image"]) for step in record["next_obs_sequence"]]
    future_wrist = [array_to_image(step["wrist_image"]) for step in record["next_obs_sequence"]]
    frame_count = min(len(generated_head), len(generated_wrist), len(future_head), len(future_wrist))
    if frame_count <= 0:
        raise ValueError("no overlapping future frames available for comparison")
    generated_head = generated_head[:frame_count]
    generated_wrist = generated_wrist[:frame_count]
    future_head = future_head[:frame_count]
    future_wrist = future_wrist[:frame_count]

    history_head_strip = build_history_strip(
        load_bundle_history(bundle_dir, "head"),
        tile_width=80,
        tile_height=80,
        gap=6,
        label_prefix="h",
    )
    history_wrist_strip = build_history_strip(
        load_bundle_history(bundle_dir, "hand_right"),
        tile_width=80,
        tile_height=80,
        gap=6,
        label_prefix="h",
    )

    frames = [
        render_animation_frame(
            history_head_strip=history_head_strip,
            history_wrist_strip=history_wrist_strip,
            generated_head_frame=generated_head[index],
            future_head_frame=future_head[index],
            generated_wrist_frame=generated_wrist[index],
            future_wrist_frame=future_wrist[index],
            record=record,
            frame_index=index,
            total_frames=frame_count,
        )
        for index in range(frame_count)
    ]

    ensure_parent(gif_output_path)
    frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(args.frame_duration_ms), 1),
        loop=0,
    )
    print(f"gif_output_path={gif_output_path}")

    if mp4_output_path is not None:
        write_mp4_from_gif(gif_path=gif_output_path, mp4_path=mp4_output_path)
        print(f"mp4_output_path={mp4_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
