#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (7, 13, 29)
METHODS = {
    "caver": {
        "label": "CAVER",
        "color": "#0b6e4f",
        "runs_pattern": "stagee__caver-lagged__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "posttrain_pattern": "stagee__caver-lagged__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "round_summary_name": "caver_round_summary.json",
    },
    "real_only": {
        "label": "Real-only",
        "color": "#9c2f2f",
        "runs_pattern": "stagee__real-only-round__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "posttrain_pattern": "stagee__real-only-round__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "round_summary_name": "real_only_round_summary.json",
    },
}
METRICS = ("online_success_rate", "validation_success_rate", "test_success_rate")
METRIC_LABELS = {
    "online_success_rate": "Online",
    "validation_success_rate": "Held-out val",
    "test_success_rate": "Held-out test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot the strict Stage-E proposal-mainline budget-50 comparison."
    )
    parser.add_argument(
        "--runs-root",
        default="/rdss/p57098/euijin1/caver/runs",
        help="Root containing Stage-E run directories.",
    )
    parser.add_argument(
        "--posttrain-root",
        default="/rdss/p57098/euijin1/caver/stagee_posttrain",
        help="Root containing Stage-E held-out post-train summaries.",
    )
    parser.add_argument("--budget", type=int, default=50, help="Budget point to summarize.")
    parser.add_argument(
        "--output-json",
        default="figures/stagee_mainline_budget50_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-png",
        default="figures/stagee_mainline_budget50_summary.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-pdf",
        default="figures/stagee_mainline_budget50_summary.pdf",
        help="Output PDF figure path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def pick_latest_matching_dir(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no directory matched {pattern!r} under {root}")
    return matches[-1]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty list")
    return float(sum(values) / len(values))


def stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def collect_method_rows(
    *,
    method: str,
    budget: int,
    runs_root: Path,
    posttrain_root: Path,
) -> list[dict[str, Any]]:
    method_spec = METHODS[method]
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = pick_latest_matching_dir(
            runs_root,
            method_spec["runs_pattern"].format(seed=seed, budget=budget),
        )
        posttrain_dir = pick_latest_matching_dir(
            posttrain_root,
            method_spec["posttrain_pattern"].format(seed=seed, budget=budget),
        )
        round_summary = load_json(run_dir / "results" / method_spec["round_summary_name"])
        posttrain_summary = load_json(posttrain_dir / "posttrain_holdout_summary.json")

        row = {
            "method": method,
            "label": method_spec["label"],
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "posttrain_dir": str(posttrain_dir.resolve()),
            "online_success_rate": float(round_summary["online"]["success_rate"]),
            "validation_success_rate": float(posttrain_summary["heldout"]["validation"]["success_rate"]),
            "test_success_rate": float(posttrain_summary["heldout"]["test"]["success_rate"]),
            "demo_items_written": int(round_summary["demo"]["demo_items_written"]),
            "primitive_steps_total": int(round_summary["demo"]["primitive_steps_total"]),
            "contexts_covered": int(round_summary["demo"]["contexts_covered"]),
        }
        if method == "caver":
            row["contexts_admitted"] = int(round_summary["admission"]["contexts_admitted"])
            row["contexts_rejected"] = int(round_summary["admission"]["contexts_rejected"])
            row["selector_mode"] = str(round_summary["selector"]["selector_mode"])
            row["dr_calibrator_model_id"] = str(round_summary["online"]["dr_calibrator_model_id"])
            row["value_proxy_model_id"] = str(round_summary["online"]["value_proxy_model_id"])
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in (
        "online_success_rate",
        "validation_success_rate",
        "test_success_rate",
        "demo_items_written",
        "primitive_steps_total",
        "contexts_covered",
    ):
        values = [float(row[key]) for row in rows]
        aggregate[key] = {"mean": mean(values), "stdev": stdev(values)}
    if "contexts_admitted" in rows[0]:
        for key in ("contexts_admitted", "contexts_rejected"):
            values = [float(row[key]) for row in rows]
            aggregate[key] = {"mean": mean(values), "stdev": stdev(values)}
        aggregate["selector_mode"] = rows[0]["selector_mode"]
        aggregate["dr_calibrator_model_id"] = rows[0]["dr_calibrator_model_id"]
        aggregate["value_proxy_model_id"] = rows[0]["value_proxy_model_id"]
    return aggregate


def build_summary(*, budget: int, runs_root: Path, posttrain_root: Path) -> dict[str, Any]:
    per_method: dict[str, list[dict[str, Any]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = collect_method_rows(
            method=method,
            budget=budget,
            runs_root=runs_root,
            posttrain_root=posttrain_root,
        )
        per_method[method] = rows
        aggregate[method] = aggregate_rows(rows)

    caver = aggregate["caver"]
    real_only = aggregate["real_only"]
    comparison = {
        "budget": budget,
        "demo_item_reduction_ratio": real_only["demo_items_written"]["mean"] / caver["demo_items_written"]["mean"],
        "primitive_step_reduction_ratio": real_only["primitive_steps_total"]["mean"] / caver["primitive_steps_total"]["mean"],
        "test_success_delta": caver["test_success_rate"]["mean"] - real_only["test_success_rate"]["mean"],
        "validation_success_delta": caver["validation_success_rate"]["mean"] - real_only["validation_success_rate"]["mean"],
        "online_success_delta": caver["online_success_rate"]["mean"] - real_only["online_success_rate"]["mean"],
    }
    return {
        "stage": "Stage E",
        "budget": budget,
        "methods": per_method,
        "aggregate": aggregate,
        "comparison": comparison,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot_summary(summary: dict[str, Any], *, output_png: Path, output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent(output_png)
    ensure_parent(output_pdf)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    scatter_ax = axes[0]
    for method in METHODS:
        rows = summary["methods"][method]
        aggregate = summary["aggregate"][method]
        color = METHODS[method]["color"]
        label = METHODS[method]["label"]

        scatter_ax.scatter(
            [row["demo_items_written"] for row in rows],
            [row["test_success_rate"] for row in rows],
            s=72,
            alpha=0.9,
            color=color,
            label=f"{label} seeds",
        )
        for row in rows:
            scatter_ax.annotate(
                f"seed {row['seed']}",
                (row["demo_items_written"], row["test_success_rate"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=color,
            )
        scatter_ax.scatter(
            [aggregate["demo_items_written"]["mean"]],
            [aggregate["test_success_rate"]["mean"]],
            s=260,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            color=color,
            label=f"{label} mean",
            zorder=3,
        )

    scatter_ax.set_xscale("log")
    scatter_ax.set_title("Held-out Test Success vs Executed Data")
    scatter_ax.set_xlabel("Admitted demo items (log scale)")
    scatter_ax.set_ylabel("Held-out test success rate")
    scatter_ax.grid(alpha=0.25, linestyle="--")
    scatter_ax.legend(frameon=False, loc="lower right")

    bars_ax = axes[1]
    x_positions = list(range(len(METRICS)))
    width = 0.34
    for offset, method in ((-width / 2.0, "caver"), (width / 2.0, "real_only")):
        aggregate = summary["aggregate"][method]
        values = [aggregate[metric]["mean"] for metric in METRICS]
        bars = bars_ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=METHODS[method]["color"],
            label=METHODS[method]["label"],
        )
        for bar, value in zip(bars, values):
            bars_ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    comparison = summary["comparison"]
    bars_ax.text(
        0.98,
        0.98,
        (
            f"CAVER uses {comparison['demo_item_reduction_ratio']:.1f}x fewer demo items\n"
            f"and {comparison['primitive_step_reduction_ratio']:.1f}x fewer primitive steps\n"
            f"than real-only at N={summary['budget']}."
        ),
        transform=bars_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
    )
    bars_ax.set_xticks(x_positions)
    bars_ax.set_xticklabels([METRIC_LABELS[metric] for metric in METRICS])
    bars_ax.set_ylim(0.0, 0.5)
    bars_ax.set_ylabel("Success rate")
    bars_ax.set_title("Strict Proposal-Mainline Budget-50 Comparison")
    bars_ax.grid(axis="y", alpha=0.25, linestyle="--")
    bars_ax.legend(frameon=False, loc="lower left")

    fig.suptitle("Stage E: Strict Mainline Comparison on LIBERO", fontsize=14)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    posttrain_root = Path(args.posttrain_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf = Path(args.output_pdf).resolve()

    summary = build_summary(
        budget=args.budget,
        runs_root=runs_root,
        posttrain_root=posttrain_root,
    )
    write_json(output_json, summary)
    plot_summary(summary, output_png=output_png, output_pdf=output_pdf)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))
    print(f"wrote {output_json}")
    print(f"wrote {output_png}")
    print(f"wrote {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
