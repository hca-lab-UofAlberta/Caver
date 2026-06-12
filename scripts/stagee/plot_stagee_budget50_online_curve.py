#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


METHOD_ORDER = ("caver", "real_only")
METHOD_LABELS = {
    "caver": "CAVER",
    "real_only": "Real-only",
}
METHOD_COLORS = {
    "caver": "#0b6e4f",
    "real_only": "#9c2f2f",
}
CONTEXT_LOG_NAMES = {
    "caver": "caver_online_contexts.jsonl",
    "real_only": "real_only_online_contexts.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the strict Stage-E budget-50 online success trajectory as line graphs. "
            "This uses the completed context logs behind stagee_mainline_budget50_summary.json."
        )
    )
    parser.add_argument(
        "--summary-json",
        default="figures/stagee_mainline_budget50_summary.json",
        help="Strict budget-50 summary JSON produced by plot_stagee_strict_mainline_summary.py.",
    )
    parser.add_argument(
        "--output-json",
        default="figures/stagee_mainline_budget50_online_curve.json",
        help="Output source-data JSON path.",
    )
    parser.add_argument(
        "--output-png",
        default="figures/stagee_mainline_budget50_summary.png",
        help="Output PNG path. Defaults to replacing the paper figure with the line-graph version.",
    )
    parser.add_argument(
        "--output-pdf",
        default="figures/stagee_mainline_budget50_summary.pdf",
        help="Output PDF path. Defaults to replacing the paper figure with the line-graph version.",
    )
    parser.add_argument("--rolling-window", type=int, default=10)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def read_successes(path: Path) -> list[int]:
    successes: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            successes.append(1 if payload.get("success") else 0)
    if not successes:
        raise ValueError(f"no context records found in {path}")
    return successes


def cumulative_rate(successes: list[int]) -> list[float]:
    total = 0
    rates: list[float] = []
    for index, success in enumerate(successes, start=1):
        total += success
        rates.append(float(total / index))
    return rates


def rolling_rate(successes: list[int], window: int) -> list[float]:
    rates: list[float] = []
    for index in range(len(successes)):
        start = max(0, index + 1 - window)
        values = successes[start : index + 1]
        rates.append(float(sum(values) / len(values)))
    return rates


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values) / math.sqrt(len(values)))


def collect_curves(summary: dict[str, Any], rolling_window: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "budget": summary.get("budget"),
        "rolling_window": rolling_window,
        "methods": {},
        "aggregate": {},
        "comparison": summary.get("comparison", {}),
        "note": (
            "These are online executed-context trajectories from the strict budget-50 run. "
            "They are not held-out checkpoint learning curves; held-out validation/test were measured at the final checkpoint."
        ),
    }

    for method in METHOD_ORDER:
        rows = summary["methods"][method]
        method_payload: list[dict[str, Any]] = []
        for row in rows:
            context_path = Path(row["run_dir"]) / "results" / CONTEXT_LOG_NAMES[method]
            successes = read_successes(context_path)
            method_payload.append(
                {
                    "seed": int(row["seed"]),
                    "context_log_path": str(context_path.resolve()),
                    "successes": successes,
                    "cumulative_success_rate": cumulative_rate(successes),
                    "rolling_success_rate": rolling_rate(successes, rolling_window),
                    "final_online_success_rate": float(row["online_success_rate"]),
                    "final_heldout_validation_success_rate": float(row["validation_success_rate"]),
                    "final_heldout_test_success_rate": float(row["test_success_rate"]),
                    "demo_items_written": int(row["demo_items_written"]),
                    "primitive_steps_total": int(row["primitive_steps_total"]),
                    "contexts_covered": int(row["contexts_covered"]),
                }
            )
        payload["methods"][method] = method_payload

        horizon = min(len(row["successes"]) for row in method_payload)
        aggregate_rows = []
        for index in range(horizon):
            cumulative_values = [row["cumulative_success_rate"][index] for row in method_payload]
            rolling_values = [row["rolling_success_rate"][index] for row in method_payload]
            aggregate_rows.append(
                {
                    "context_index": index + 1,
                    "cumulative_mean": mean(cumulative_values),
                    "cumulative_sem": sem(cumulative_values),
                    "rolling_mean": mean(rolling_values),
                    "rolling_sem": sem(rolling_values),
                }
            )
        payload["aggregate"][method] = aggregate_rows
    return payload


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float, *, fill: str, width: int = 2) -> None:
    y = y0
    dash = 8
    gap = 6
    while y < y1:
        draw.line((x, y, x, min(y + dash, y1)), fill=fill, width=width)
        y += dash + gap


class Panel:
    def __init__(self, left: int, top: int, width: int, height: int, *, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.plot_left = left + 74
        self.plot_top = top + 54
        self.plot_width = width - 112
        self.plot_height = height - 118

    def x(self, value: float) -> float:
        return self.plot_left + ((value - self.x_min) / (self.x_max - self.x_min)) * self.plot_width

    def y(self, value: float) -> float:
        return self.plot_top + ((self.y_max - value) / (self.y_max - self.y_min)) * self.plot_height


def draw_panel_base(
    draw: ImageDraw.ImageDraw,
    panel: Panel,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    title_font: ImageFont.ImageFont,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.text((panel.left, panel.top), title, fill="#111111", font=title_font)
    draw.line(
        (
            panel.plot_left,
            panel.plot_top + panel.plot_height,
            panel.plot_left + panel.plot_width,
            panel.plot_top + panel.plot_height,
        ),
        fill="#111111",
        width=2,
    )
    draw.line((panel.plot_left, panel.plot_top, panel.plot_left, panel.plot_top + panel.plot_height), fill="#111111", width=2)

    for y_value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = panel.y(y_value)
        draw.line((panel.plot_left, y, panel.plot_left + panel.plot_width, y), fill="#e6e6e6", width=1)
        draw.text((panel.plot_left - 44, y - 9), f"{y_value:.1f}", fill="#333333", font=small_font)

    for x_value in (1, 10, 20, 30, 40, 50):
        x = panel.x(float(x_value))
        draw.line((x, panel.plot_top + panel.plot_height, x, panel.plot_top + panel.plot_height + 6), fill="#111111", width=1)
        draw.text((x - 10, panel.plot_top + panel.plot_height + 12), str(x_value), fill="#333333", font=small_font)

    draw.text(
        (panel.plot_left + (panel.plot_width // 2) - 82, panel.top + panel.height - 34),
        xlabel,
        fill="#111111",
        font=font,
    )
    draw.text((panel.left + 2, panel.plot_top - 28), ylabel, fill="#111111", font=font)


def draw_mean_curve(
    draw: ImageDraw.ImageDraw,
    panel: Panel,
    rows: list[dict[str, Any]],
    *,
    metric_prefix: str,
    color: str,
) -> None:
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    mean_points: list[tuple[float, float]] = []
    for row in rows:
        x_value = float(row["context_index"])
        y_mean = float(row[f"{metric_prefix}_mean"])
        y_sem = float(row[f"{metric_prefix}_sem"])
        upper.append((panel.x(x_value), panel.y(min(1.0, y_mean + y_sem))))
        lower.append((panel.x(x_value), panel.y(max(0.0, y_mean - y_sem))))
        mean_points.append((panel.x(x_value), panel.y(y_mean)))
    if len(upper) > 1:
        draw.polygon(upper + list(reversed(lower)), fill=color + "33")
        draw.line(mean_points, fill=color, width=4)
    for point in mean_points[::10] + mean_points[-1:]:
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="#111111", width=1)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot(payload: dict[str, Any], output_png: Path, output_pdf: Path) -> None:
    ensure_parent(output_png)
    ensure_parent(output_pdf)

    width, height = 1700, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(34, bold=True)
    panel_title_font = load_font(23, bold=True)
    font = load_font(18)
    small_font = load_font(15)
    tiny_font = load_font(13)

    draw.text((56, 36), "Stage E Budget-50 Online Success Trajectory", fill="#111111", font=title_font)
    draw.text(
        (56, 82),
        "Mean across seeds 7, 13, and 29; shaded bands show SEM. Final held-out scores are checkpoint-level, not per-context measurements.",
        fill="#333333",
        font=font,
    )

    left = Panel(56, 145, 760, 600, x_min=1.0, x_max=50.0, y_min=0.0, y_max=1.02)
    right = Panel(884, 145, 760, 600, x_min=1.0, x_max=50.0, y_min=0.0, y_max=1.02)
    draw_panel_base(
        draw,
        left,
        title="Cumulative Online Success",
        xlabel="Executed context index",
        ylabel="Success rate",
        title_font=panel_title_font,
        font=font,
        small_font=small_font,
    )
    draw_panel_base(
        draw,
        right,
        title=f"Rolling Online Success, Last {payload['rolling_window']} Contexts",
        xlabel="Executed context index",
        ylabel="Success rate",
        title_font=panel_title_font,
        font=font,
        small_font=small_font,
    )

    for panel in (left, right):
        x_round = panel.x(25.0)
        draw_dashed_vertical(draw, x_round, panel.plot_top, panel.plot_top + panel.plot_height, fill="#777777", width=2)
        draw.text((x_round + 8, panel.plot_top + 8), "round boundary", fill="#555555", font=tiny_font)

    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        rows = payload["aggregate"][method]
        draw_mean_curve(draw, left, rows, metric_prefix="cumulative", color=color)
        draw_mean_curve(draw, right, rows, metric_prefix="rolling", color=color)

    legend_x, legend_y = 62, 776
    for offset, method in enumerate(METHOD_ORDER):
        y = legend_y + offset * 32
        color = METHOD_COLORS[method]
        draw.line((legend_x, y + 10, legend_x + 42, y + 10), fill=color, width=5)
        draw.ellipse((legend_x + 17, y + 5, legend_x + 27, y + 15), fill=color, outline="#111111", width=1)
        draw.text((legend_x + 56, y), METHOD_LABELS[method], fill="#111111", font=font)

    comparison = payload["comparison"]
    caver_final = payload["aggregate"]["caver"][-1]["cumulative_mean"]
    real_final = payload["aggregate"]["real_only"][-1]["cumulative_mean"]
    summary_text = (
        f"Final online success: CAVER {caver_final:.3f}, real-only {real_final:.3f}\n"
        f"Final held-out test delta: {comparison['test_success_delta']:+.3f}\n"
        f"CAVER uses {comparison['demo_item_reduction_ratio']:.1f}x fewer admitted demo items at N=50."
    )
    draw.rounded_rectangle((884, 760, 1644, 848), radius=6, fill="#f7f7f2", outline="#c9c9c0", width=1)
    draw.text((908, 779), summary_text, fill="#111111", font=font, spacing=6)

    draw.text(
        (56, 852),
        "Interpretation: the x-axis is online executed context order. Because contexts are task-family ordered, this curve is not a monotone held-out learning curve.",
        fill="#444444",
        font=tiny_font,
    )

    image.save(output_png)
    if output_pdf:
        image.save(output_pdf, "PDF", resolution=180.0)


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    output_json = Path(args.output_json).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf = Path(args.output_pdf).resolve() if args.output_pdf else Path()

    summary = load_json(summary_path)
    payload = collect_curves(summary, args.rolling_window)
    write_json(output_json, payload)
    plot(payload, output_png, output_pdf)
    print(json.dumps({"json": str(output_json), "png": str(output_png), "pdf": str(output_pdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
