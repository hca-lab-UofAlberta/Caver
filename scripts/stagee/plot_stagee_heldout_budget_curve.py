#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_ORDER = (
    "caver",
    "real_only",
    "real_only_k1",
    "uniform_k4",
    "selection_only",
    "admission_only",
    "success_only",
    "k1_guarded_success_only",
    "top_m_success",
    "hard_rescue_fulltrace",
    "fasr_progress_ranked",
    "k1_fasr",
    "uniform_k4_fasr",
    "caver_lvd",
    "caver_lvd_fasr",
    "caver_lvd_no_provider",
    "caver_lvd_no_dr",
    "no_dr",
    "no_provider",
)
METHOD_LABELS = {
    "caver": "CAVER",
    "real_only": "Real-only K=4 uniform",
    "real_only_k1": "Vanilla real-only K=1",
    "uniform_k4": "Uniform K=4",
    "selection_only": "Selection-only",
    "admission_only": "Admission-only",
    "success_only": "Success-only admission",
    "k1_guarded_success_only": "K=1-guarded CAVER + success-only",
    "top_m_success": "Top-M15 success admission",
    "hard_rescue_fulltrace": "Hard-family rescue CAVER",
    "fasr_progress_ranked": "Progress-ranked FASR",
    "k1_fasr": "Vanilla K=1 + FASR",
    "uniform_k4_fasr": "Uniform K=4 + FASR",
    "caver_lvd": "CAVER-LVD",
    "caver_lvd_fasr": "CAVER-LVD + FASR",
    "caver_lvd_no_provider": "CAVER-LVD no-provider",
    "caver_lvd_no_dr": "CAVER-LVD no-DR",
    "no_dr": "No-DR CAVER",
    "no_provider": "No-provider CAVER",
}
METHOD_COLORS = {
    "caver": "#0b6e4f",
    "real_only": "#9c2f2f",
    "real_only_k1": "#2f5597",
    "uniform_k4": "#7b7b7b",
    "selection_only": "#c77800",
    "admission_only": "#6f4aa8",
    "success_only": "#7f7f00",
    "k1_guarded_success_only": "#d55e00",
    "top_m_success": "#4b8bbe",
    "hard_rescue_fulltrace": "#009e73",
    "fasr_progress_ranked": "#cc79a7",
    "k1_fasr": "#1f78b4",
    "uniform_k4_fasr": "#33a02c",
    "caver_lvd": "#005f73",
    "caver_lvd_fasr": "#9b2226",
    "caver_lvd_no_provider": "#bc6c25",
    "caver_lvd_no_dr": "#6a4c93",
    "no_dr": "#0072b2",
    "no_provider": "#8a5a44",
}
METHOD_PATTERNS = {
    "caver": "stagee__caver-lagged__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
    "real_only": "stagee__real-only-round__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
    "real_only_k1": "stagee__real-only-round__manifest-t_train_s0-all-vanilla-k1__seed{seed}__budget{budget}__*",
    "uniform_k4": "stagee__caver-lagged__manifest-t_train_s0-all-uniform-k4__seed{seed}__budget{budget}__*",
    "selection_only": "stagee__caver-lagged__manifest-t_train_s0-all-selection-only__seed{seed}__budget{budget}__*",
    "admission_only": "stagee__caver-lagged__manifest-t_train_s0-all-admission-only__seed{seed}__budget{budget}__*",
    "success_only": "stagee__caver-lagged__manifest-t_train_s0-all-success-only__seed{seed}__budget{budget}__*",
    "k1_guarded_success_only": "k1guarded_seed{seed}_budget{budget}_*",
    "top_m_success": "topm15_seed{seed}_budget{budget}_*",
    "hard_rescue_fulltrace": "hardrescue_fulltrace_seed{seed}_budget{budget}_*",
    "fasr_progress_ranked": "seed{seed}_budget{budget}_*",
    "k1_fasr": "stagee__caver-lagged__manifest-t_train_s0-all-k1-fasr-n{budget}__seed{seed}__budget{budget}__*",
    "uniform_k4_fasr": "stagee__caver-lagged__manifest-t_train_s0-all-uniform-k4-fasr-n{budget}__seed{seed}__budget{budget}__*",
    "caver_lvd": "stagee__caver-lagged__manifest-t_train_s0-all-caver-lvd-n{budget}__seed{seed}__budget{budget}__*",
    "caver_lvd_fasr": "stagee__caver-lagged__manifest-t_train_s0-all-caver-lvd-fasr-n{budget}__seed{seed}__budget{budget}__*",
    "caver_lvd_no_provider": "stagee__caver-lagged__manifest-t_train_s0-all-caver-lvd-no-provider-n{budget}__seed{seed}__budget{budget}__*",
    "caver_lvd_no_dr": "stagee__caver-lagged__manifest-t_train_s0-all-caver-lvd-no-dr-n{budget}__seed{seed}__budget{budget}__*",
    "no_dr": "stagee__caver-lagged__manifest-t_train_s0-all-no-dr__seed{seed}__budget{budget}__*",
    "no_provider": "stagee__caver-lagged__manifest-t_train_s0-all-no-provider__seed{seed}__budget{budget}__*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot proposal-mainline Stage-E held-out validation/test learning curves "
            "over trusted execution budgets."
        )
    )
    parser.add_argument(
        "--posttrain-root",
        default="/rdss/p57098/euijin1/caver/stagee_posttrain",
        help="Comma-separated posttrain roots to search, newest matching run wins.",
    )
    parser.add_argument("--budgets", default="25,50,100")
    parser.add_argument("--seeds", default="7,13,29")
    parser.add_argument("--output-json", default="figures/stagee_heldout_budget_curve.json")
    parser.add_argument("--output-png", default="figures/stagee_heldout_budget_curve.png")
    parser.add_argument("--output-pdf", default="figures/stagee_heldout_budget_curve.pdf")
    parser.add_argument(
        "--methods",
        default="caver,real_only",
        help=(
            "Comma-separated method keys to include. Available: "
            "caver,real_only,real_only_k1,uniform_k4,selection_only,admission_only,success_only,"
            "k1_guarded_success_only,top_m_success,hard_rescue_fulltrace,fasr_progress_ranked,"
            "k1_fasr,uniform_k4_fasr,caver_lvd,caver_lvd_fasr,caver_lvd_no_provider,"
            "caver_lvd_no_dr,no_dr,no_provider."
        ),
    )
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def pick_latest_summary(roots: list[Path], *, method: str, budget: int, seed: int) -> Path | None:
    pattern = METHOD_PATTERNS[method].format(seed=seed, budget=budget)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.glob(pattern))
    for candidate in reversed(sorted(candidates)):
        summary = candidate / "posttrain_holdout_summary.json"
        if summary.exists():
            return summary.resolve()
    return None


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values) / math.sqrt(len(values)))


def collect_rows(
    posttrain_roots: list[Path],
    budgets: list[int],
    seeds: list[int],
    methods: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for method in methods:
        for budget in budgets:
            for seed in seeds:
                summary_path = pick_latest_summary(posttrain_roots, method=method, budget=budget, seed=seed)
                if summary_path is None:
                    missing.append({"method": method, "budget": budget, "seed": seed})
                    continue
                summary = load_json(summary_path)
                heldout = summary.get("heldout", {})
                exact = summary.get("exact_rollout_batch") or {}
                online = summary.get("base_round_online") or {}
                try:
                    validation = heldout["validation"]
                    test = heldout["test"]
                except KeyError:
                    missing.append(
                        {
                            "method": method,
                            "budget": budget,
                            "seed": seed,
                            "summary_path": str(summary_path),
                            "reason": "missing heldout validation/test",
                        }
                    )
                    continue
                rows.append(
                    {
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "summary_path": str(summary_path),
                        "run_dir": str(summary_path.parent),
                        "online_success_rate": online.get("success_rate"),
                        "online_successes": online.get("successes"),
                        "trusted_contexts": online.get("episodes_run", budget),
                        "contexts_covered": exact.get("contexts_covered"),
                        "admitted_demo_items": exact.get("trace_records_read"),
                        "primitive_steps_total": exact.get("primitive_steps_total"),
                        "validation_success_rate": float(validation["success_rate"]),
                        "validation_successes": int(validation["successes"]),
                        "validation_episodes": int(validation["episodes_run"]),
                        "test_success_rate": float(test["success_rate"]),
                        "test_successes": int(test["successes"]),
                        "test_episodes": int(test["episodes_run"]),
                    }
                )
    return rows, missing


def aggregate_rows(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["budget"]))].append(row)

    aggregate: list[dict[str, Any]] = []
    for method in methods:
        for (group_method, budget), group_rows in sorted(grouped.items(), key=lambda item: item[0][1]):
            if group_method != method:
                continue
            validation = [float(row["validation_success_rate"]) for row in group_rows]
            test = [float(row["test_success_rate"]) for row in group_rows]
            demo_items = [
                float(row["admitted_demo_items"])
                for row in group_rows
                if row.get("admitted_demo_items") is not None
            ]
            primitive_steps = [
                float(row["primitive_steps_total"])
                for row in group_rows
                if row.get("primitive_steps_total") is not None
            ]
            aggregate.append(
                {
                    "method": method,
                    "budget": budget,
                    "seeds": sorted(int(row["seed"]) for row in group_rows),
                    "n_seeds": len(group_rows),
                    "validation_mean": float(sum(validation) / len(validation)),
                    "validation_sem": sem(validation),
                    "test_mean": float(sum(test) / len(test)),
                    "test_sem": sem(test),
                    "admitted_demo_items_mean": float(sum(demo_items) / len(demo_items)) if demo_items else None,
                    "primitive_steps_mean": float(sum(primitive_steps) / len(primitive_steps)) if primitive_steps else None,
                }
            )
    return aggregate


def metric_bounds(aggregate: list[dict[str, Any]], mean_key: str) -> tuple[float, float]:
    values: list[float] = []
    for row in aggregate:
        if row.get(mean_key) is None:
            continue
        mean = float(row[mean_key])
        sem_key = mean_key.replace("_mean", "_sem")
        err = float(row.get(sem_key, 0.0) or 0.0)
        values.extend([mean - err, mean + err])
    if not values:
        return 0.0, 1.0
    low = max(0.0, min(values))
    high = min(1.0, max(values))
    span = high - low
    padding = max(0.02, span * 0.18)
    low = max(0.0, low - padding)
    high = min(1.0, high + padding)
    if high - low < 0.08:
        midpoint = (low + high) / 2.0
        low = max(0.0, midpoint - 0.04)
        high = min(1.0, midpoint + 0.04)
    return low, high


def write_source(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    methods: list[str],
) -> None:
    ensure_parent(path)
    payload = {
        "workflow": "stagee_heldout_budget_curve_v1",
        "x_axis": "trusted execution contexts N",
        "methods": {method: METHOD_LABELS[method] for method in methods},
        "rows": rows,
        "aggregate": aggregate,
        "missing_cells": missing,
        "note": (
            "This is the held-out checkpoint curve needed for sample-efficiency claims. "
            "It is distinct from the within-run online trajectory diagnostic."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_with_matplotlib(
    path_png: Path,
    path_pdf: Path | None,
    *,
    aggregate: list[dict[str, Any]],
    budgets: list[int],
    missing: list[dict[str, Any]],
    methods: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(path_png)
    if path_pdf is not None:
        ensure_parent(path_pdf)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True, constrained_layout=True)
    metric_specs = (
        ("validation_mean", "validation_sem", "Held-out Validation Success"),
        ("test_mean", "test_sem", "Held-out Test Success"),
    )
    for ax, (mean_key, sem_key, title) in zip(axes, metric_specs):
        y_low, y_high = metric_bounds(aggregate, mean_key)
        for method in methods:
            points = [row for row in aggregate if row["method"] == method]
            points = sorted(points, key=lambda row: int(row["budget"]))
            if not points:
                continue
            xs = [int(row["budget"]) for row in points]
            ys = [float(row[mean_key]) for row in points]
            errors = [float(row[sem_key]) for row in points]
            ax.errorbar(
                xs,
                ys,
                yerr=errors,
                marker="o",
                linewidth=2.2,
                markersize=7,
                capsize=4,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            for row, x, y in zip(points, xs, ys):
                ax.annotate(
                    f"n={row['n_seeds']}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=METHOD_COLORS[method],
                )
        ax.set_title(title)
        ax.set_xlabel("Trusted execution budget N")
        ax.set_ylabel("Success rate")
        ax.set_xticks(budgets)
        ax.set_ylim(y_low, y_high)
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend(frameon=False)

    missing_count = len(missing)
    note = (
        "All requested cells complete."
        if missing_count == 0
        else "Mainline curves complete; selected ablations are shown only at completed budget points."
    )
    fig.suptitle("Stage-E Proposal-Mainline Held-out Sample-Efficiency Curve", fontsize=14)
    fig.text(0.5, -0.02, note, ha="center", fontsize=9)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    if path_pdf is not None:
        fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_with_pil(
    path_png: Path,
    path_pdf: Path | None,
    *,
    aggregate: list[dict[str, Any]],
    budgets: list[int],
    missing: list[dict[str, Any]],
    methods: list[str],
) -> None:
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    ensure_parent(path_png)
    if path_pdf is not None:
        ensure_parent(path_pdf)

    width, height = 1300, 540
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    panels = [
        (60, 78, 540, 360, "validation_mean", "validation_sem", "Held-out Validation Success"),
        (710, 78, 540, 360, "test_mean", "test_sem", "Held-out Test Success"),
    ]
    x_min = min(budgets)
    x_max = max(budgets)
    x_span = x_max - x_min
    y_min = min(metric_bounds(aggregate, "validation_mean")[0], metric_bounds(aggregate, "test_mean")[0])
    y_max = max(metric_bounds(aggregate, "validation_mean")[1], metric_bounds(aggregate, "test_mean")[1])

    def sx(left: int, panel_width: int, value: float) -> float:
        if x_span == 0:
            return left + 58 + (panel_width - 86) / 2
        return left + 58 + ((value - x_min) / x_span) * (panel_width - 86)

    def sy(top: int, panel_height: int, value: float) -> float:
        return top + 20 + ((y_max - value) / (y_max - y_min)) * (panel_height - 64)

    draw.text((60, 24), "Stage-E Proposal-Mainline Held-out Sample-Efficiency Curve", fill="#111111", font=font)
    for left, top, panel_width, panel_height, mean_key, sem_key, title in panels:
        draw.text((left, top - 28), title, fill="#111111", font=font)
        x0 = left + 58
        y0 = top + panel_height - 44
        draw.line((x0, y0, left + panel_width - 28, y0), fill="#111111", width=2)
        draw.line((x0, top + 20, x0, y0), fill="#111111", width=2)
        for tick in budgets:
            x = sx(left, panel_width, tick)
            draw.line((x, y0, x, y0 + 6), fill="#111111", width=1)
            draw.text((x - 10, y0 + 12), str(tick), fill="#333333", font=font)
        tick_count = 6
        tick_values = [y_min + (y_max - y_min) * index / (tick_count - 1) for index in range(tick_count)]
        for tick in tick_values:
            y = sy(top, panel_height, tick)
            draw.line((x0, y, left + panel_width - 28, y), fill="#e8e8e8", width=1)
            draw.text((left + 12, y - 6), f"{tick:.2f}", fill="#333333", font=font)
        draw.text((left + 190, top + panel_height - 18), "Trusted execution budget N", fill="#111111", font=font)
        draw.text((left + 2, top + 6), "Success rate", fill="#111111", font=font)

        for method in methods:
            points = [row for row in aggregate if row["method"] == method]
            points = sorted(points, key=lambda row: int(row["budget"]))
            coords: list[tuple[float, float]] = []
            for row in points:
                x = sx(left, panel_width, float(row["budget"]))
                y = sy(top, panel_height, float(row[mean_key]))
                coords.append((x, y))
                err = float(row[sem_key])
                if err:
                    y_hi = sy(top, panel_height, float(row[mean_key]) + err)
                    y_lo = sy(top, panel_height, float(row[mean_key]) - err)
                    draw.line((x, y_hi, x, y_lo), fill=METHOD_COLORS[method], width=2)
                    draw.line((x - 5, y_hi, x + 5, y_hi), fill=METHOD_COLORS[method], width=2)
                    draw.line((x - 5, y_lo, x + 5, y_lo), fill=METHOD_COLORS[method], width=2)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=METHOD_COLORS[method], outline="#111111")
                draw.text((x - 10, y - 20), f"n={row['n_seeds']}", fill=METHOD_COLORS[method], font=font)
            if len(coords) >= 2:
                draw.line(coords, fill=METHOD_COLORS[method], width=3)

    legend_x, legend_y = 1060, 26
    for index, method in enumerate(methods):
        y = legend_y + index * 22
        draw.ellipse((legend_x, y, legend_x + 10, y + 10), fill=METHOD_COLORS[method], outline="#111111")
        draw.text((legend_x + 18, y - 1), METHOD_LABELS[method], fill="#111111", font=font)

    note = (
        "All requested cells complete."
        if not missing
        else "Mainline curves complete; selected ablations are shown only at completed budget points."
    )
    draw.text((60, 500), note, fill="#333333", font=font)
    image.save(path_png)
    if path_pdf is not None:
        image.save(path_pdf, "PDF", resolution=150.0)


def main() -> int:
    args = parse_args()
    posttrain_roots = [Path(token).resolve() for token in parse_csv_strings(args.posttrain_root)]
    if not posttrain_roots:
        raise SystemExit("error: --posttrain-root must include at least one path")
    budgets = parse_csv_ints(args.budgets)
    seeds = parse_csv_ints(args.seeds)
    output_json = Path(args.output_json).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf = Path(args.output_pdf).resolve() if args.output_pdf else None
    methods = parse_csv_strings(args.methods)
    bad_methods = sorted(set(methods).difference(METHOD_ORDER))
    if bad_methods:
        raise SystemExit(f"error: unsupported methods: {bad_methods}")

    rows, missing = collect_rows(posttrain_roots, budgets, seeds, methods)
    aggregate = aggregate_rows(rows, methods)
    write_source(output_json, rows=rows, aggregate=aggregate, missing=missing, methods=methods)
    try:
        plot_with_matplotlib(output_png, output_pdf, aggregate=aggregate, budgets=budgets, missing=missing, methods=methods)
    except ModuleNotFoundError:
        plot_with_pil(output_png, output_pdf, aggregate=aggregate, budgets=budgets, missing=missing, methods=methods)

    print(
        json.dumps(
            {
                "json": str(output_json),
                "png": str(output_png),
                "pdf": str(output_pdf) if output_pdf is not None else "",
                "complete_cells": len(rows),
                "missing_cells": len(missing),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
