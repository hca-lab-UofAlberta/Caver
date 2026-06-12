#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SEEDS = (7, 13, 29)
PIPER_RELEVANT_FAMILIES = (
    "block_to_tray_proxy",
    "container_insertion_proxy",
    "two_object_stack_proxy",
)
METHODS = {
    "caver": {
        "label": "CAVER",
        "runs_pattern": "stagee__caver-lagged__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "posttrain_pattern": "stagee__caver-lagged__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "round_summary_name": "caver_round_summary.json",
        "demo_summary_name": "caver_round_demo.summary.json",
        "admission_summary_name": "caver_admission_summary.json",
    },
    "real_only": {
        "label": "Real-only",
        "runs_pattern": "stagee__real-only-round__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "posttrain_pattern": "stagee__real-only-round__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
        "round_summary_name": "real_only_round_summary.json",
        "demo_summary_name": "real_only_round_demo.summary.json",
        "admission_summary_name": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize family-level Stage-E diagnostics from the strict proposal-mainline artifacts."
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
        default="logs/runtime/stagee_mainline_family_diagnostics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default="logs/runtime/stagee_mainline_family_diagnostics.md",
        help="Output Markdown path.",
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


def sum_counts(mapping: dict[str, Any] | None, keys: tuple[str, ...]) -> int:
    if not mapping:
        return 0
    total = 0
    for key in keys:
        total += int(mapping.get(key, 0))
    return total


def int_mapping(mapping: dict[str, Any] | None) -> dict[str, int]:
    if not mapping:
        return {}
    return {str(key): int(value) for key, value in mapping.items()}


def collect_method_rows(
    *,
    method: str,
    budget: int,
    runs_root: Path,
    posttrain_root: Path,
) -> list[dict[str, Any]]:
    spec = METHODS[method]
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = pick_latest_matching_dir(
            runs_root,
            spec["runs_pattern"].format(seed=seed, budget=budget),
        )
        posttrain_dir = pick_latest_matching_dir(
            posttrain_root,
            spec["posttrain_pattern"].format(seed=seed, budget=budget),
        )
        round_summary = load_json(run_dir / "results" / spec["round_summary_name"])
        demo_summary = load_json(run_dir / "results" / spec["demo_summary_name"])
        posttrain_summary = load_json(posttrain_dir / "posttrain_holdout_summary.json")

        row: dict[str, Any] = {
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "posttrain_dir": str(posttrain_dir.resolve()),
            "online_success_rate": float(round_summary["online"]["success_rate"]),
            "validation_success_rate": float(posttrain_summary["heldout"]["validation"]["success_rate"]),
            "test_success_rate": float(posttrain_summary["heldout"]["test"]["success_rate"]),
            "demo_family_counts": int_mapping(demo_summary.get("family_counts")),
            "demo_items_written": int(round_summary["demo"]["demo_items_written"]),
            "primitive_steps_total": int(round_summary["demo"]["primitive_steps_total"]),
            "contexts_covered": int(round_summary["demo"]["contexts_covered"]),
        }
        if spec["admission_summary_name"]:
            admission_summary = load_json(run_dir / "results" / spec["admission_summary_name"])
            row["admitted_family_counts"] = int_mapping(admission_summary.get("admitted_family_counts"))
            row["lcb_acceptance_counts"] = int_mapping(admission_summary.get("lcb_acceptance_counts"))
            row["contexts_admitted"] = int(admission_summary["contexts_admitted"])
            row["contexts_rejected"] = int(admission_summary["contexts_rejected"])
        rows.append(row)
    return rows


def aggregate_method(rows: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    demo_family_totals: dict[str, int] = {}
    admitted_family_totals: dict[str, int] = {}
    lcb_reason_totals: dict[str, int] = {}
    for row in rows:
        for family, value in row.get("demo_family_counts", {}).items():
            demo_family_totals[family] = demo_family_totals.get(family, 0) + int(value)
        for family, value in row.get("admitted_family_counts", {}).items():
            admitted_family_totals[family] = admitted_family_totals.get(family, 0) + int(value)
        for reason, value in row.get("lcb_acceptance_counts", {}).items():
            lcb_reason_totals[reason] = lcb_reason_totals.get(reason, 0) + int(value)

    aggregate = {
        "seed_count": len(rows),
        "demo_family_totals": dict(sorted(demo_family_totals.items())),
        "demo_piper_relevant_total": sum_counts(demo_family_totals, PIPER_RELEVANT_FAMILIES),
        "demo_total": sum(sum(row.get("demo_family_counts", {}).values()) for row in rows),
        "online_success_mean": sum(float(row["online_success_rate"]) for row in rows) / float(len(rows)),
        "validation_success_mean": sum(float(row["validation_success_rate"]) for row in rows) / float(len(rows)),
        "test_success_mean": sum(float(row["test_success_rate"]) for row in rows) / float(len(rows)),
    }
    if method == "caver":
        aggregate.update(
            {
                "admitted_family_totals": dict(sorted(admitted_family_totals.items())),
                "admitted_piper_relevant_total": sum_counts(admitted_family_totals, PIPER_RELEVANT_FAMILIES),
                "admitted_total": sum(int(row["contexts_admitted"]) for row in rows),
                "rejected_total": sum(int(row["contexts_rejected"]) for row in rows),
                "lcb_acceptance_totals": dict(sorted(lcb_reason_totals.items())),
            }
        )
    return aggregate


def build_summary(*, budget: int, runs_root: Path, posttrain_root: Path) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        rows = collect_method_rows(
            method=method,
            budget=budget,
            runs_root=runs_root,
            posttrain_root=posttrain_root,
        )
        methods[method] = {
            "rows": rows,
            "aggregate": aggregate_method(rows, method=method),
        }

    caver_agg = methods["caver"]["aggregate"]
    concentration_warning = (
        list(caver_agg.get("admitted_family_totals", {}).keys()) == ["container_insertion_proxy"]
    )
    readiness_note = (
        "Current strict N=50 CAVER admissions are concentrated in container_insertion_proxy; "
        "treat PiPER transition as gated readiness + pilot work, not the full Stage-1 study."
        if concentration_warning
        else "Current strict N=50 CAVER admissions span more than one family."
    )
    return {
        "stage": "Stage E",
        "budget": budget,
        "piper_relevant_families": list(PIPER_RELEVANT_FAMILIES),
        "methods": methods,
        "findings": {
            "caver_admission_family_concentration_warning": concentration_warning,
            "readiness_note": readiness_note,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_markdown(summary: dict[str, Any]) -> str:
    caver = summary["methods"]["caver"]["aggregate"]
    real_only = summary["methods"]["real_only"]["aggregate"]
    lines = [
        f"# Stage-E Mainline Family Diagnostics (N={summary['budget']})",
        "",
        summary["findings"]["readiness_note"],
        "",
        "## CAVER aggregate",
        "",
        f"- online success mean: `{caver['online_success_mean']:.3f}`",
        f"- held-out validation mean: `{caver['validation_success_mean']:.3f}`",
        f"- held-out test mean: `{caver['test_success_mean']:.3f}`",
        f"- admitted total across seeds: `{caver['admitted_total']}`",
        f"- rejected total across seeds: `{caver['rejected_total']}`",
        f"- admitted family totals: `{json.dumps(caver['admitted_family_totals'], sort_keys=True)}`",
        f"- LCB acceptance totals: `{json.dumps(caver['lcb_acceptance_totals'], sort_keys=True)}`",
        "",
        "## Real-only aggregate",
        "",
        f"- online success mean: `{real_only['online_success_mean']:.3f}`",
        f"- held-out validation mean: `{real_only['validation_success_mean']:.3f}`",
        f"- held-out test mean: `{real_only['test_success_mean']:.3f}`",
        f"- demo family totals: `{json.dumps(real_only['demo_family_totals'], sort_keys=True)}`",
        "",
        "## Seed-level CAVER admissions",
        "",
    ]
    for row in summary["methods"]["caver"]["rows"]:
        lines.extend(
            [
                f"- seed `{row['seed']}`:",
                f"  admitted families `{json.dumps(row['admitted_family_counts'], sort_keys=True)}`",
                f"  LCB reasons `{json.dumps(row['lcb_acceptance_counts'], sort_keys=True)}`",
                f"  contexts admitted/rejected `{row['contexts_admitted']}/{row['contexts_rejected']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    posttrain_root = Path(args.posttrain_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()

    summary = build_summary(
        budget=args.budget,
        runs_root=runs_root,
        posttrain_root=posttrain_root,
    )
    write_json(output_json, summary)
    write_text(output_md, build_markdown(summary))
    print(json.dumps(summary["findings"], indent=2, sort_keys=True))
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
