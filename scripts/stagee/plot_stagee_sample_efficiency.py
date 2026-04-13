#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


ONLINE_BUDGETS = (25, 50, 100, 200)
HELDOUT_SEEDS = (7, 13, 29)
METHOD_ORDER = ("caver", "real_only")
METHOD_LABELS = {
    "caver": "CAVER",
    "real_only": "Real-only",
}
METHOD_COLORS = {
    "caver": "#0b6e4f",
    "real_only": "#9c2f2f",
}
POSTTRAIN_DIR_PATTERNS = {
    "caver": "stagee__caver-round__manifest-t_train_s0-all__seed{seed}__budget100__*",
    "real_only": "stagee__real-only-round__manifest-t_train_s0-all__seed{seed}__budget100__*",
}
ONLINE_DEMO_KEYS = {
    "caver": "caver_round_demo.summary.json",
    "real_only": "real_only_round_demo.summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Stage-E sample-efficiency figures from the authoritative online grid "
            "and the held-out N=100 post-train summaries."
        )
    )
    parser.add_argument(
        "--grid-snapshot",
        default="logs/runtime/stagee_native_horizon_grid_snapshot.json",
        help="Path to summarize_stagee_grid JSON output.",
    )
    parser.add_argument(
        "--posttrain-root",
        default="/rdss/p57098/euijin1/caver/stagee_posttrain",
        help="Root directory containing held-out post-train summaries.",
    )
    parser.add_argument(
        "--output-png",
        default="logs/runtime/stagee_sample_efficiency.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-pdf",
        default="logs/runtime/stagee_sample_efficiency.pdf",
        help="Optional PDF output path.",
    )
    parser.add_argument(
        "--output-json",
        default="logs/runtime/stagee_sample_efficiency.json",
        help="Output source-data JSON path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty list")
    return float(sum(values) / len(values))


def collect_online_budget_curve(snapshot_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, float]]]]:
    snapshot = load_json(snapshot_path)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for cell in snapshot["cells"]:
        method = str(cell["method"])
        budget = int(cell["budget"])
        if method not in METHOD_ORDER:
            continue
        if budget not in ONLINE_BUDGETS:
            continue
        if str(cell.get("status")) != "completed":
            continue
        demo = cell.get("demo", {})
        online = cell.get("online", {})
        if demo.get("demo_items_written") is None or online.get("success_rate") is None:
            continue
        grouped[(method, budget)].append(cell)

    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[int, dict[str, float]]] = {method: {} for method in METHOD_ORDER}
    for method in METHOD_ORDER:
        for budget in ONLINE_BUDGETS:
            cells = grouped.get((method, budget), [])
            if not cells:
                continue
            demo_items = [float(cell["demo"]["demo_items_written"]) for cell in cells]
            success_rates = [float(cell["online"]["success_rate"]) for cell in cells]
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "seeds": sorted(int(cell["seed"]) for cell in cells),
                    "mean_demo_items": mean(demo_items),
                    "mean_online_success_rate": mean(success_rates),
                    "demo_items": demo_items,
                    "online_success_rates": success_rates,
                }
            )
            curves[method][budget] = {
                "mean_demo_items": mean(demo_items),
                "mean_online_success_rate": mean(success_rates),
            }
    return rows, curves


def pick_latest_matching_dir(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no post-train directory matched {pattern!r} under {root}")
    return matches[-1]


def collect_heldout_n100(posttrain_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for seed in HELDOUT_SEEDS:
            run_dir = pick_latest_matching_dir(posttrain_root, POSTTRAIN_DIR_PATTERNS[method].format(seed=seed))
            summary_path = run_dir / "posttrain_holdout_summary.json"
            summary = load_json(summary_path)
            demo_summary_path = Path(summary["demo_manifest"]).resolve().with_name(ONLINE_DEMO_KEYS[method])
            demo_summary = load_json(demo_summary_path)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "summary_path": str(summary_path.resolve()),
                    "demo_items_written": int(demo_summary["demo_items_written"]),
                    "contexts_covered": int(demo_summary["contexts_covered"]),
                    "validation_success_rate": float(summary["heldout"]["validation"]["success_rate"]),
                    "test_success_rate": float(summary["heldout"]["test"]["success_rate"]),
                    "validation_successes": int(summary["heldout"]["validation"]["successes"]),
                    "test_successes": int(summary["heldout"]["test"]["successes"]),
                }
            )

    aggregate: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "mean_demo_items": mean([float(row["demo_items_written"]) for row in method_rows]),
            "mean_validation_success_rate": mean([float(row["validation_success_rate"]) for row in method_rows]),
            "mean_test_success_rate": mean([float(row["test_success_rate"]) for row in method_rows]),
            "total_demo_items": float(sum(int(row["demo_items_written"]) for row in method_rows)),
            "total_validation_successes": float(sum(int(row["validation_successes"]) for row in method_rows)),
            "total_test_successes": float(sum(int(row["test_successes"]) for row in method_rows)),
        }
    return rows, aggregate


def annotate_budget_points(ax: Any, budgets_to_points: dict[int, dict[str, float]], *, color: str) -> None:
    for budget, payload in sorted(budgets_to_points.items()):
        ax.annotate(
            f"N={budget}",
            (payload["mean_demo_items"], payload["mean_online_success_rate"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=color,
        )


def try_plot_with_matplotlib(
    sample_efficiency_path_png: Path,
    sample_efficiency_path_pdf: Path,
    *,
    online_curves: dict[str, dict[int, dict[str, float]]],
    heldout_rows: list[dict[str, Any]],
    heldout_aggregate: dict[str, dict[str, float]],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    ax = axes[0]
    for method in METHOD_ORDER:
        budgets = [budget for budget in ONLINE_BUDGETS if budget in online_curves[method]]
        xs = [online_curves[method][budget]["mean_demo_items"] for budget in budgets]
        ys = [online_curves[method][budget]["mean_online_success_rate"] for budget in budgets]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        annotate_budget_points(ax, {budget: online_curves[method][budget] for budget in budgets}, color=METHOD_COLORS[method])
    ax.set_title("Stage E Online Success vs Admitted Data")
    ax.set_xlabel("Mean Admitted Demo Items")
    ax.set_ylabel("Mean Online Success Rate")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    ax = axes[1]
    for method in METHOD_ORDER:
        method_rows = [row for row in heldout_rows if row["method"] == method]
        ax.scatter(
            [row["demo_items_written"] for row in method_rows],
            [row["test_success_rate"] for row in method_rows],
            s=70,
            alpha=0.85,
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} seeds",
        )
        for row in method_rows:
            ax.annotate(
                f"seed {row['seed']}",
                (row["demo_items_written"], row["test_success_rate"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=METHOD_COLORS[method],
            )
        ax.scatter(
            [heldout_aggregate[method]["mean_demo_items"]],
            [heldout_aggregate[method]["mean_test_success_rate"]],
            s=220,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} mean",
        )
    ax.set_title("Held-out Test Success at N=100")
    ax.set_xlabel("Admitted Demo Items")
    ax.set_ylabel("Held-out Test Success Rate")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("CAVER Stage E Sample-Efficiency Summary", fontsize=14)
    fig.savefig(sample_efficiency_path_png, dpi=220, bbox_inches="tight")
    if sample_efficiency_path_pdf:
        fig.savefig(sample_efficiency_path_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_axes(draw: ImageDraw.ImageDraw, *, left: int, top: int, width: int, height: int, xlabel: str, ylabel: str, title: str, font: ImageFont.ImageFont, small_font: ImageFont.ImageFont) -> None:
    draw.rectangle((left, top, left + width, top + height), outline="#d0d0d0", width=1)
    draw.line((left + 54, top + height - 40, left + width - 18, top + height - 40), fill="black", width=2)
    draw.line((left + 54, top + 18, left + 54, top + height - 40), fill="black", width=2)
    draw.text((left + 10, top - 2), title, fill="black", font=font)
    draw.text((left + width // 2 - 55, top + height - 28), xlabel, fill="black", font=small_font)
    draw.text((left + 2, top + height // 2 - 20), ylabel, fill="black", font=small_font)


def _map_x(value: float, *, x_min: float, x_max: float, left: int, width: int) -> float:
    usable = width - 72
    return left + 54 + ((value - x_min) / (x_max - x_min)) * usable


def _map_y(value: float, *, y_min: float, y_max: float, top: int, height: int) -> float:
    usable = height - 58
    return top + 18 + ((y_max - value) / (y_max - y_min)) * usable


def _draw_ticks(draw: ImageDraw.ImageDraw, *, left: int, top: int, width: int, height: int, x_ticks: list[tuple[float, str]], y_ticks: list[tuple[float, str]], x_min: float, x_max: float, y_min: float, y_max: float, small_font: ImageFont.ImageFont) -> None:
    for tick_value, tick_label in x_ticks:
        x = _map_x(tick_value, x_min=x_min, x_max=x_max, left=left, width=width)
        y = top + height - 40
        draw.line((x, y, x, y + 6), fill="black", width=1)
        draw.text((x - 16, y + 8), tick_label, fill="black", font=small_font)
    for tick_value, tick_label in y_ticks:
        x = left + 54
        y = _map_y(tick_value, y_min=y_min, y_max=y_max, top=top, height=height)
        draw.line((x - 6, y, x, y), fill="black", width=1)
        draw.text((left + 8, y - 7), tick_label, fill="black", font=small_font)


def _draw_circle(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: int, *, fill: str, outline: str | None = None, width: int = 1) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(outline or fill), width=width)


def _draw_star(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: int, *, fill: str, outline: str = "black", width: int = 2) -> None:
    x, y = center
    draw.line((x - radius, y, x + radius, y), fill=fill, width=width)
    draw.line((x, y - radius, x, y + radius), fill=fill, width=width)
    draw.line((x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2), fill=fill, width=width)
    draw.line((x - radius + 2, y + radius - 2, x + radius - 2, y - radius + 2), fill=fill, width=width)
    _draw_circle(draw, center, 2, fill=outline, outline=outline, width=1)


def plot_with_pil(
    sample_efficiency_path_png: Path,
    sample_efficiency_path_pdf: Path,
    *,
    online_curves: dict[str, dict[int, dict[str, float]]],
    heldout_rows: list[dict[str, Any]],
    heldout_aggregate: dict[str, dict[str, float]],
) -> None:
    width, height = 1400, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw.text((20, 14), "CAVER Stage E Sample-Efficiency Summary", fill="black", font=font)

    left_panel = (30, 52, 640, 520)
    right_panel = (720, 52, 640, 520)
    _draw_axes(
        draw,
        left=left_panel[0],
        top=left_panel[1],
        width=left_panel[2],
        height=left_panel[3],
        xlabel="Mean admitted demo items",
        ylabel="Mean online success rate",
        title="Stage E Online Success vs Admitted Data",
        font=font,
        small_font=small_font,
    )
    _draw_axes(
        draw,
        left=right_panel[0],
        top=right_panel[1],
        width=right_panel[2],
        height=right_panel[3],
        xlabel="Admitted demo items",
        ylabel="Held-out test success rate",
        title="Held-out Test Success at N=100",
        font=font,
        small_font=small_font,
    )

    online_points = [
        payload
        for method in METHOD_ORDER
        for payload in online_curves[method].values()
    ]
    x_min = min(point["mean_demo_items"] for point in online_points) * 0.9
    x_max = max(point["mean_demo_items"] for point in online_points) * 1.05
    y_min = 0.0
    y_max = max(point["mean_online_success_rate"] for point in online_points) + 0.06
    _draw_ticks(
        draw,
        left=left_panel[0],
        top=left_panel[1],
        width=left_panel[2],
        height=left_panel[3],
        x_ticks=[(500.0, "500"), (2000.0, "2000"), (4000.0, "4000"), (6000.0, "6000")],
        y_ticks=[(0.0, "0.00"), (0.1, "0.10"), (0.2, "0.20"), (0.3, "0.30"), (0.4, "0.40")],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        small_font=small_font,
    )
    for method in METHOD_ORDER:
        budgets = [budget for budget in ONLINE_BUDGETS if budget in online_curves[method]]
        coords = []
        for budget in budgets:
            payload = online_curves[method][budget]
            coords.append(
                (
                    _map_x(payload["mean_demo_items"], x_min=x_min, x_max=x_max, left=left_panel[0], width=left_panel[2]),
                    _map_y(payload["mean_online_success_rate"], y_min=y_min, y_max=y_max, top=left_panel[1], height=left_panel[3]),
                )
            )
        if len(coords) >= 2:
            draw.line(coords, fill=METHOD_COLORS[method], width=3)
        for budget, coord in zip(budgets, coords):
            _draw_circle(draw, coord, 5, fill=METHOD_COLORS[method], outline="black", width=1)
            draw.text((coord[0] - 12, coord[1] - 20), f"N={budget}", fill=METHOD_COLORS[method], font=small_font)

    heldout_x_min = min(float(row["demo_items_written"]) for row in heldout_rows) * 0.9
    heldout_x_max = max(float(row["demo_items_written"]) for row in heldout_rows) * 1.05
    heldout_y_min = 0.18
    heldout_y_max = 0.25
    _draw_ticks(
        draw,
        left=right_panel[0],
        top=right_panel[1],
        width=right_panel[2],
        height=right_panel[3],
        x_ticks=[(1500.0, "1500"), (3000.0, "3000"), (4500.0, "4500"), (6000.0, "6000")],
        y_ticks=[(0.18, "0.18"), (0.20, "0.20"), (0.22, "0.22"), (0.24, "0.24")],
        x_min=heldout_x_min,
        x_max=heldout_x_max,
        y_min=heldout_y_min,
        y_max=heldout_y_max,
        small_font=small_font,
    )
    for method in METHOD_ORDER:
        method_rows = [row for row in heldout_rows if row["method"] == method]
        for row in method_rows:
            coord = (
                _map_x(float(row["demo_items_written"]), x_min=heldout_x_min, x_max=heldout_x_max, left=right_panel[0], width=right_panel[2]),
                _map_y(float(row["test_success_rate"]), y_min=heldout_y_min, y_max=heldout_y_max, top=right_panel[1], height=right_panel[3]),
            )
            _draw_circle(draw, coord, 6, fill=METHOD_COLORS[method], outline="black", width=1)
            draw.text((coord[0] - 18, coord[1] - 20), f"seed {row['seed']}", fill=METHOD_COLORS[method], font=small_font)
        mean_coord = (
            _map_x(float(heldout_aggregate[method]["mean_demo_items"]), x_min=heldout_x_min, x_max=heldout_x_max, left=right_panel[0], width=right_panel[2]),
            _map_y(float(heldout_aggregate[method]["mean_test_success_rate"]), y_min=heldout_y_min, y_max=heldout_y_max, top=right_panel[1], height=right_panel[3]),
        )
        _draw_star(draw, mean_coord, 10, fill=METHOD_COLORS[method])

    legend_x, legend_y = 1060, 72
    for index, method in enumerate(METHOD_ORDER):
        y = legend_y + (index * 22)
        _draw_circle(draw, (legend_x, y + 6), 5, fill=METHOD_COLORS[method], outline="black", width=1)
        draw.text((legend_x + 14, y), METHOD_LABELS[method], fill="black", font=small_font)
    draw.text((740, 560), "Mean star markers summarize the three N=100 held-out seeds.", fill="black", font=small_font)

    image.save(sample_efficiency_path_png)
    if sample_efficiency_path_pdf:
        image.save(sample_efficiency_path_pdf, "PDF", resolution=150.0)


def plot(sample_efficiency_path_png: Path, sample_efficiency_path_pdf: Path, source_data_path: Path, *, online_rows: list[dict[str, Any]], online_curves: dict[str, dict[int, dict[str, float]]], heldout_rows: list[dict[str, Any]], heldout_aggregate: dict[str, dict[str, float]]) -> None:
    ensure_parent(sample_efficiency_path_png)
    ensure_parent(source_data_path)
    if sample_efficiency_path_pdf:
        ensure_parent(sample_efficiency_path_pdf)

    payload = {
        "online_budget_curve": online_rows,
        "heldout_n100": heldout_rows,
        "heldout_n100_aggregate": heldout_aggregate,
    }
    with source_data_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if not try_plot_with_matplotlib(
        sample_efficiency_path_png,
        sample_efficiency_path_pdf,
        online_curves=online_curves,
        heldout_rows=heldout_rows,
        heldout_aggregate=heldout_aggregate,
    ):
        plot_with_pil(
            sample_efficiency_path_png,
            sample_efficiency_path_pdf,
            online_curves=online_curves,
            heldout_rows=heldout_rows,
            heldout_aggregate=heldout_aggregate,
        )


def main() -> int:
    args = parse_args()
    snapshot_path = Path(args.grid_snapshot).resolve()
    posttrain_root = Path(args.posttrain_root).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf = Path(args.output_pdf).resolve() if args.output_pdf else Path()
    output_json = Path(args.output_json).resolve()

    online_rows, online_curves = collect_online_budget_curve(snapshot_path)
    heldout_rows, heldout_aggregate = collect_heldout_n100(posttrain_root)
    plot(
        output_png,
        output_pdf,
        output_json,
        online_rows=online_rows,
        online_curves=online_curves,
        heldout_rows=heldout_rows,
        heldout_aggregate=heldout_aggregate,
    )

    print(json.dumps({"png": str(output_png), "pdf": str(output_pdf), "json": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
