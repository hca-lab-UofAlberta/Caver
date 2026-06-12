#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence
import yaml

try:
    import torch
except ImportError:  # pragma: no cover - allowed when only GIF previews are available
    torch = None


BACKGROUND = (248, 249, 251)
PANEL_BACKGROUND = (255, 255, 255)
TEXT = (22, 24, 29)
MUTED_TEXT = (90, 97, 106)
BORDER = (214, 219, 226)
ACCENT_LIBERO = (51, 102, 204)
ACCENT_GESIM = (0, 128, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a static side-by-side overview from a saved LIBERO->GE-Sim bundle "
            "and a GE-Sim output directory."
        )
    )
    parser.add_argument("--bundle-dir", required=True, help="Saved provider bundle directory.")
    parser.add_argument("--gesim-output-dir", required=True, help="GE-Sim output directory with video.pt or video.gif.")
    parser.add_argument("--output-path", required=True, help="PNG path for the rendered overview image.")
    parser.add_argument(
        "--max-generated-frames",
        type=int,
        default=6,
        help="Maximum number of generated frames to show per camera strip.",
    )
    parser.add_argument(
        "--bundle-label",
        default=None,
        help="Optional short label describing the bundle, shown in the figure subtitle.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object in {path}")
    return payload


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def sample_indices(length: int, limit: int) -> list[int]:
    if length <= 0:
        return []
    if limit <= 0 or length <= limit:
        return list(range(length))
    if limit == 1:
        return [length - 1]
    step = (length - 1) / float(limit - 1)
    values = [int(round(index * step)) for index in range(limit)]
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    if deduped[-1] != length - 1:
        deduped[-1] = length - 1
    return deduped


def fit_tile(image: Image.Image, *, tile_width: int, tile_height: int) -> Image.Image:
    rgb = image.convert("RGB")
    fitted = ImageOps.contain(rgb, (tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height), PANEL_BACKGROUND)
    offset_x = (tile_width - fitted.width) // 2
    offset_y = (tile_height - fitted.height) // 2
    tile.paste(fitted, (offset_x, offset_y))
    return tile


def build_strip(
    frames: Iterable[Image.Image],
    *,
    tile_width: int,
    tile_height: int,
    gap: int,
    label_prefix: str | None = None,
) -> Image.Image:
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("cannot build strip from an empty frame list")

    label_font = load_font(14)
    label_height = 20 if label_prefix is not None else 0
    width = len(frame_list) * tile_width + max(0, len(frame_list) - 1) * gap
    strip = Image.new("RGB", (width, tile_height + label_height), BACKGROUND)
    draw = ImageDraw.Draw(strip)

    cursor_x = 0
    for index, frame in enumerate(frame_list):
        tile = fit_tile(frame, tile_width=tile_width, tile_height=tile_height)
        strip.paste(tile, (cursor_x, label_height))
        draw.rectangle(
            (
                cursor_x,
                label_height,
                cursor_x + tile_width - 1,
                label_height + tile_height - 1,
            ),
            outline=BORDER,
            width=1,
        )
        if label_prefix is not None:
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


def load_bundle_frames(bundle_dir: Path, camera_name: str) -> list[Image.Image]:
    image_dir = bundle_dir / f"{camera_name}_color"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"bundle camera directory not found: {image_dir}")
    frames: list[Image.Image] = []
    for image_path in sorted(image_dir.glob("*.png")):
        with Image.open(image_path) as image:
            frames.append(image.convert("RGB"))
    if not frames:
        raise FileNotFoundError(f"no PNG frames found under {image_dir}")
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

    raise FileNotFoundError(f"expected {tensor_path} or {gif_path}")


def crop_generated_view(
    frames: list[Image.Image],
    *,
    frame_height: int,
    frame_width: int,
    view_index: int,
) -> list[Image.Image]:
    cropped: list[Image.Image] = []
    left = view_index * frame_width
    top = 0
    right = left + frame_width
    bottom = frame_height
    for frame in frames:
        cropped.append(frame.crop((left, top, right, bottom)))
    return cropped


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    output_dir = Path(args.gesim_output_dir).resolve()
    output_path = Path(args.output_path).resolve()

    bundle_metadata = read_json(bundle_dir / "bundle_metadata.json")
    runtime_config = read_yaml(output_dir / "gesim_runtime.yaml")
    valid_cam = [str(value) for value in runtime_config["data"]["train"]["valid_cam"]]
    sample_height, sample_width = [int(value) for value in runtime_config["data"]["train"]["sample_size"]]

    generated_frames = load_generated_frames(output_dir)
    generated_indices = sample_indices(len(generated_frames), args.max_generated_frames)
    selected_generated = [generated_frames[index] for index in generated_indices]

    tile_width = 160
    tile_height = 120
    strip_gap = 8
    side_margin = 24
    top_margin = 26
    row_gap = 18
    section_gap = 28

    source_names = bundle_metadata.get("source_camera_names", {})
    bundle_label = args.bundle_label or str(bundle_metadata.get("context_id") or bundle_dir.name)

    bundle_rows: list[tuple[str, Image.Image]] = []
    generated_rows: list[tuple[str, Image.Image]] = []
    for view_index, camera_name in enumerate(valid_cam):
        bundle_frames = load_bundle_frames(bundle_dir, camera_name)
        bundle_strip = build_strip(
            bundle_frames,
            tile_width=tile_width,
            tile_height=tile_height,
            gap=strip_gap,
            label_prefix="t",
        )
        source_name = source_names.get(camera_name, camera_name)
        bundle_rows.append((f"LIBERO history: {camera_name} ({source_name})", bundle_strip))

        cropped = crop_generated_view(
            selected_generated,
            frame_height=sample_height,
            frame_width=sample_width,
            view_index=view_index,
        )
        generated_strip = build_strip(
            cropped,
            tile_width=tile_width,
            tile_height=tile_height,
            gap=strip_gap,
            label_prefix="g",
        )
        generated_rows.append((f"GE-Sim generated future: {camera_name}", generated_strip))

    max_strip_width = max(
        max(strip.width for _, strip in bundle_rows),
        max(strip.width for _, strip in generated_rows),
    )

    header_font = load_font(28)
    subheader_font = load_font(16)
    row_font = load_font(18)

    header_height = 92
    content_height = 0
    for _title, strip in bundle_rows:
        content_height += max(48, strip.height + 26) + row_gap
    content_height += section_gap
    for _title, strip in generated_rows:
        content_height += max(48, strip.height + 26) + row_gap
    content_height -= row_gap

    canvas_width = side_margin * 2 + 18 + max_strip_width
    canvas_height = top_margin + header_height + content_height + 24
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((side_margin, top_margin), "GE-Sim vs LIBERO Visual Comparison", fill=TEXT, font=header_font)
    subtitle = (
        f"Bundle: {bundle_label} | "
        f"history_frames={bundle_metadata.get('memory_frame_count', 'unknown')} | "
        f"generated_frames={len(generated_frames)} | views={len(valid_cam)}"
    )
    draw.text((side_margin, top_margin + 40), subtitle, fill=MUTED_TEXT, font=subheader_font)
    draw.text(
        (side_margin, top_margin + 62),
        "This compares LIBERO observation history saved in the provider bundle against GE-Sim's generated future frames.",
        fill=MUTED_TEXT,
        font=subheader_font,
    )

    cursor_y = top_margin + header_height
    for title, strip in bundle_rows:
        cursor_y = draw_row(
            canvas,
            draw=draw,
            title=title,
            accent=ACCENT_LIBERO,
            strip=strip,
            x=side_margin,
            y=cursor_y,
            title_font=row_font,
        )
        cursor_y += row_gap

    draw.line(
        (side_margin, cursor_y - 8, canvas_width - side_margin, cursor_y - 8),
        fill=BORDER,
        width=2,
    )
    cursor_y += section_gap - 8

    for title, strip in generated_rows:
        cursor_y = draw_row(
            canvas,
            draw=draw,
            title=title,
            accent=ACCENT_GESIM,
            strip=strip,
            x=side_margin,
            y=cursor_y,
            title_font=row_font,
        )
        cursor_y += row_gap

    ensure_parent(output_path)
    canvas.save(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
