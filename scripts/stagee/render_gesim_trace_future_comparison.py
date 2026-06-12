#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
            "Render a side-by-side comparison between GE-Sim generated future frames "
            "and the logged LIBERO future frames from a Stage-E chunk trace."
        )
    )
    parser.add_argument("--trace-path", required=True, help="Stage-E chunk trace JSONL path.")
    parser.add_argument("--context-id", required=True, help="Trace context id.")
    parser.add_argument("--policy-query-index", type=int, required=True, help="1-based policy query index.")
    parser.add_argument("--gesim-output-dir", required=True, help="GE-Sim output directory with video.pt or video.gif.")
    parser.add_argument("--output-path", required=True, help="PNG path for the rendered overview.")
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


def fit_tile(image: Image.Image, *, tile_width: int, tile_height: int) -> Image.Image:
    rgb = image.convert("RGB")
    fitted = ImageOps.contain(rgb, (tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height), PANEL_BACKGROUND)
    tile.paste(fitted, ((tile_width - fitted.width) // 2, (tile_height - fitted.height) // 2))
    return tile


def build_strip(
    frames: list[Image.Image],
    *,
    tile_width: int,
    tile_height: int,
    gap: int,
    label_prefix: str,
) -> Image.Image:
    label_font = load_font(14)
    width = len(frames) * tile_width + max(0, len(frames) - 1) * gap
    strip = Image.new("RGB", (width, tile_height + 20), BACKGROUND)
    draw = ImageDraw.Draw(strip)
    cursor_x = 0
    for index, frame in enumerate(frames):
        tile = fit_tile(frame, tile_width=tile_width, tile_height=tile_height)
        strip.paste(tile, (cursor_x, 20))
        draw.rectangle((cursor_x, 20, cursor_x + tile_width - 1, 20 + tile_height - 1), outline=BORDER, width=1)
        draw.text((cursor_x, 0), f"{label_prefix}{index}", fill=MUTED_TEXT, font=label_font)
        cursor_x += tile_width + gap
    return strip


def draw_row(
    canvas: Image.Image,
    *,
    draw: ImageDraw.ImageDraw,
    title: str,
    accent: tuple[int, int, int],
    strip: Image.Image,
    x: int,
    y: int,
    title_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> int:
    draw.text((x, y), title, fill=TEXT, font=title_font)
    draw.rounded_rectangle((x, y + 26, x + 10, y + 36), radius=4, fill=accent)
    canvas.paste(strip, (x + 18, y + 18))
    return y + max(48, strip.height + 26)


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
        # PIL can construct from nested lists only after converting through torch or numpy,
        # so keep the dependency requirement explicit.
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


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace_path).resolve()
    gesim_output_dir = Path(args.gesim_output_dir).resolve()
    output_path = Path(args.output_path).resolve()

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
    head_index = valid_cam.index("head")
    hand_right_index = valid_cam.index("hand_right")
    generated_head = crop_generated_view(
        generated_frames,
        frame_height=sample_height,
        frame_width=sample_width,
        view_index=head_index,
    )
    generated_wrist = crop_generated_view(
        generated_frames,
        frame_height=sample_height,
        frame_width=sample_width,
        view_index=hand_right_index,
    )

    history_head = load_bundle_history(bundle_dir, "head")
    history_wrist = load_bundle_history(bundle_dir, "hand_right")
    future_head = [array_to_image(step["image"]) for step in record["next_obs_sequence"]]
    future_wrist = [array_to_image(step["wrist_image"]) for step in record["next_obs_sequence"]]

    if len(generated_head) != len(future_head):
        frame_count = min(len(generated_head), len(future_head))
        generated_head = generated_head[:frame_count]
        generated_wrist = generated_wrist[:frame_count]
        future_head = future_head[:frame_count]
        future_wrist = future_wrist[:frame_count]

    tile_width = 160
    tile_height = 120
    gap = 8
    margin = 24
    top_margin = 26
    row_gap = 18
    section_gap = 30

    rows = [
        ("LIBERO input history: frontview/head", ACCENT_INPUT, build_strip(history_head, tile_width=tile_width, tile_height=tile_height, gap=gap, label_prefix="h")),
        ("LIBERO input history: wrist / hand_right", ACCENT_INPUT, build_strip(history_wrist, tile_width=tile_width, tile_height=tile_height, gap=gap, label_prefix="h")),
        ("GE-Sim generated future: head", ACCENT_GESIM, build_strip(generated_head, tile_width=tile_width, tile_height=tile_height, gap=gap, label_prefix="g")),
        ("LIBERO logged future: frontview/head", ACCENT_GT, build_strip(future_head, tile_width=tile_width, tile_height=tile_height, gap=gap, label_prefix="t")),
        ("GE-Sim generated future: wrist / hand_right", ACCENT_GESIM, build_strip(generated_wrist, tile_width=tile_width, tile_height=tile_height, gap=gap, label_prefix="g")),
        ("LIBERO logged future: wrist", ACCENT_GT, build_strip(future_wrist, tile_width=tile_width, tile_height=tile_height, gap=gap, label_prefix="t")),
    ]

    max_strip_width = max(strip.width for _, _, strip in rows)
    header_font = load_font(28)
    subheader_font = load_font(16)
    row_font = load_font(18)
    header_height = 108
    content_height = sum(max(48, strip.height + 26) + row_gap for _, _, strip in rows) - row_gap
    content_height += section_gap * 2
    canvas_width = margin * 2 + 18 + max_strip_width
    canvas_height = top_margin + header_height + content_height + 24
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, top_margin), "GE-Sim vs Logged LIBERO Future", fill=TEXT, font=header_font)
    draw.text(
        (margin, top_margin + 40),
        (
            f"context={record['context_id']} | query={record['policy_query_index']} | "
            f"selected_candidate={record['selector']['selected_candidate_index']} | "
            f"task_id={record['task_id']} | init_state_index={record['init_state_index']}"
        ),
        fill=MUTED_TEXT,
        font=subheader_font,
    )
    draw.text(
        (margin, top_margin + 64),
        "Ground-truth future comes from the logged selected chunk trace. hand_left is omitted because the trace stores frontview and wrist only.",
        fill=MUTED_TEXT,
        font=subheader_font,
    )

    cursor_y = top_margin + header_height
    for row_index, (title, accent, strip) in enumerate(rows):
        cursor_y = draw_row(
            canvas,
            draw=draw,
            title=title,
            accent=accent,
            strip=strip,
            x=margin,
            y=cursor_y,
            title_font=row_font,
        )
        cursor_y += row_gap
        if row_index in (1, 3):
            draw.line((margin, cursor_y - 8, canvas_width - margin, cursor_y - 8), fill=BORDER, width=2)
            cursor_y += section_gap - 8

    ensure_parent(output_path)
    canvas.save(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
