#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (7, 13, 29)
DEFAULT_BUDGET = 50
PIPER_RELEVANT_FAMILIES = (
    "block_to_tray_proxy",
    "container_insertion_proxy",
    "two_object_stack_proxy",
)

METHODS: dict[str, dict[str, str]] = {
    "caver": {
        "label": "Mainline CAVER",
        "root": "/rdss/p57098/euijin1/caver/stagee_posttrain",
        "pattern": "stagee__caver-lagged__manifest-t_train_s0-all__seed{seed}__budget{budget}__*",
    },
    "real_only_k1": {
        "label": "Vanilla real-only K=1",
        "root": "/rdss/p57098/euijin1/caver/stagee_posttrain",
        "pattern": "stagee__real-only-round__manifest-t_train_s0-all-vanilla-k1__seed{seed}__budget{budget}__*",
    },
    "success_only": {
        "label": "Success-only admission",
        "root": "/projects/p57098/euijin1/caver_stagee_ablation_exactpayload_posttrain",
        "pattern": "stagee__caver-lagged__manifest-t_train_s0-all-success-only__seed{seed}__budget{budget}__*",
    },
    "k1_guarded_success_only": {
        "label": "K=1-guarded CAVER + success-only",
        "root": "/projects/p57098/euijin1/caver_stagee_k1_guarded_posttrain",
        "pattern": "k1guarded_seed{seed}_budget{budget}_*",
    },
    "top_m_success": {
        "label": "Top-M15 success admission",
        "root": "/projects/p57098/euijin1/caver_stagee_plus_posttrain",
        "pattern": "topm15_seed{seed}_budget{budget}_*",
    },
    "hard_rescue_fulltrace": {
        "label": "Hard-family rescue CAVER",
        "root": "/projects/p57098/euijin1/caver_stagee_hard_rescue_fulltrace_posttrain",
        "pattern": "hardrescue_fulltrace_seed{seed}_budget{budget}_*",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage-E methods by LIBERO proxy family using held-out context logs "
            "and exact-backend training summaries."
        )
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--methods",
        default="caver,real_only_k1,success_only,k1_guarded_success_only,top_m_success",
        help=f"Comma-separated method keys. Available: {','.join(METHODS)}",
    )
    parser.add_argument(
        "--output-json",
        default="logs/runtime/stagee_family_comparison_n50.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/runtime/stagee_family_comparison_n50.md",
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


def pick_latest_summary(method: str, *, budget: int, seed: int) -> Path:
    spec = METHODS[method]
    root = Path(spec["root"])
    pattern = spec["pattern"].format(seed=seed, budget=budget)
    matches = sorted(root.glob(pattern))
    for candidate in reversed(matches):
        summary_path = candidate / "posttrain_holdout_summary.json"
        if summary_path.exists():
            return summary_path.resolve()
    raise FileNotFoundError(f"no posttrain summary for method={method} seed={seed} under {root}")


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def read_eval_contexts(results_path: Path) -> list[dict[str, Any]]:
    payload = load_json(results_path)
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
        raise ValueError(f"expected contexts list in {results_path}")
    return [context for context in contexts if isinstance(context, dict)]


def aggregate_contexts_by_family(contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        family = str(context.get("proxy_family_id") or "unknown")
        grouped[family].append(context)

    family_rows: dict[str, dict[str, Any]] = {}
    for family, family_contexts in sorted(grouped.items()):
        successes = sum(1 for context in family_contexts if bool(context.get("success")))
        episodes = len(family_contexts)
        policy_steps = [
            float(context.get("policy_steps", 0.0))
            for context in family_contexts
            if context.get("policy_steps") is not None
        ]
        family_rows[family] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "mean_policy_steps": safe_mean(policy_steps),
        }
    return family_rows


def merge_family_eval(seed_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    per_seed_rates: dict[str, list[float]] = defaultdict(list)
    for row in seed_rows:
        for family, family_row in row["heldout_by_family"][split].items():
            target = totals.setdefault(
                family,
                {"episodes": 0, "successes": 0, "mean_policy_steps_values": []},
            )
            target["episodes"] += int(family_row["episodes"])
            target["successes"] += int(family_row["successes"])
            if family_row.get("mean_policy_steps") is not None:
                target["mean_policy_steps_values"].append(float(family_row["mean_policy_steps"]))
            per_seed_rates[family].append(float(family_row["success_rate"]))

    merged: dict[str, dict[str, Any]] = {}
    for family, target in sorted(totals.items()):
        episodes = int(target["episodes"])
        successes = int(target["successes"])
        rates = per_seed_rates[family]
        merged[family] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "seed_mean_success_rate": safe_mean(rates),
            "seed_sem_success_rate": sem(rates),
            "mean_policy_steps": safe_mean(target["mean_policy_steps_values"]),
        }
    return merged


def sum_mapping(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for family, value in (row.get(key) or {}).items():
            total[family] = total.get(family, 0) + int(value)
    return dict(sorted(total.items()))


def summarize_method(method: str, *, budget: int, seeds: list[int]) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        summary_path = pick_latest_summary(method, budget=budget, seed=seed)
        summary = load_json(summary_path)
        exact = summary.get("exact_rollout_batch") or {}
        heldout = summary.get("heldout") or {}

        seed_row: dict[str, Any] = {
            "seed": seed,
            "summary_path": str(summary_path),
            "run_dir": str(summary_path.parent),
            "online_success_rate": (summary.get("base_round_online") or {}).get("success_rate"),
            "validation_success_rate": heldout.get("validation", {}).get("success_rate"),
            "test_success_rate": heldout.get("test", {}).get("success_rate"),
            "contexts_covered": exact.get("contexts_covered"),
            "admitted_demo_items": exact.get("trace_records_read"),
            "primitive_steps_total": exact.get("primitive_steps_total"),
            "admitted_context_family_counts": exact.get("family_counts") or {},
            "admitted_record_family_counts": exact.get("raw_family_record_counts") or {},
            "heldout_by_family": {},
        }
        for split in ("validation", "test"):
            split_info = heldout.get(split) or {}
            results_path_raw = split_info.get("results_path")
            if results_path_raw is None:
                raise ValueError(f"missing {split} results_path in {summary_path}")
            results_path = Path(results_path_raw)
            contexts = read_eval_contexts(results_path)
            seed_row["heldout_by_family"][split] = aggregate_contexts_by_family(contexts)
        seed_rows.append(seed_row)

    aggregate = {
        "label": METHODS[method]["label"],
        "online_success_mean": safe_mean(
            [float(row["online_success_rate"]) for row in seed_rows if row["online_success_rate"] is not None]
        ),
        "validation_success_mean": safe_mean([float(row["validation_success_rate"]) for row in seed_rows]),
        "validation_success_sem": sem([float(row["validation_success_rate"]) for row in seed_rows]),
        "test_success_mean": safe_mean([float(row["test_success_rate"]) for row in seed_rows]),
        "test_success_sem": sem([float(row["test_success_rate"]) for row in seed_rows]),
        "contexts_covered_mean": safe_mean(
            [float(row["contexts_covered"]) for row in seed_rows if row["contexts_covered"] is not None]
        ),
        "admitted_demo_items_mean": safe_mean(
            [float(row["admitted_demo_items"]) for row in seed_rows if row["admitted_demo_items"] is not None]
        ),
        "primitive_steps_total_mean": safe_mean(
            [float(row["primitive_steps_total"]) for row in seed_rows if row["primitive_steps_total"] is not None]
        ),
        "admitted_context_family_totals": sum_mapping(seed_rows, "admitted_context_family_counts"),
        "admitted_record_family_totals": sum_mapping(seed_rows, "admitted_record_family_counts"),
        "validation_by_family": merge_family_eval(seed_rows, "validation"),
        "test_by_family": merge_family_eval(seed_rows, "test"),
    }
    return {"rows": seed_rows, "aggregate": aggregate}


def build_summary(*, budget: int, seeds: list[int], methods: list[str]) -> dict[str, Any]:
    method_payloads = {
        method: summarize_method(method, budget=budget, seeds=seeds)
        for method in methods
    }
    findings = build_findings(method_payloads)
    return {
        "workflow": "stagee_family_comparison_v1",
        "budget": budget,
        "seeds": seeds,
        "piper_relevant_families": list(PIPER_RELEVANT_FAMILIES),
        "methods": method_payloads,
        "findings": findings,
    }


def build_findings(method_payloads: dict[str, Any]) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    success_only = method_payloads.get("success_only", {}).get("aggregate")
    top_m = method_payloads.get("top_m_success", {}).get("aggregate")
    hard_rescue = method_payloads.get("hard_rescue_fulltrace", {}).get("aggregate")
    k1 = method_payloads.get("real_only_k1", {}).get("aggregate")
    if success_only and k1:
        findings["success_only_minus_k1_test"] = (
            float(success_only["test_success_mean"]) - float(k1["test_success_mean"])
        )
    if top_m and success_only:
        findings["top_m_minus_success_only_test"] = (
            float(top_m["test_success_mean"]) - float(success_only["test_success_mean"])
        )
    if hard_rescue and k1:
        findings["hard_rescue_minus_k1_test"] = (
            float(hard_rescue["test_success_mean"]) - float(k1["test_success_mean"])
        )
    if hard_rescue and success_only:
        findings["hard_rescue_minus_success_only_test"] = (
            float(hard_rescue["test_success_mean"]) - float(success_only["test_success_mean"])
        )
    if success_only:
        context_totals = success_only.get("admitted_context_family_totals") or {}
        admitted_total = sum(int(value) for value in context_totals.values())
        piper_total = sum(int(context_totals.get(family, 0)) for family in PIPER_RELEVANT_FAMILIES)
        findings["success_only_piper_relevant_admission_fraction"] = (
            piper_total / admitted_total if admitted_total else None
        )
        findings["success_only_admitted_context_family_count"] = len(
            [family for family, value in context_totals.items() if int(value) > 0]
        )
    return findings


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Stage-E Family Comparison (N={summary['budget']})",
        "",
        "This diagnostic compares held-out success by LIBERO proxy family and admitted backend data by family.",
        "",
        "## Aggregate",
        "",
    ]
    aggregate_rows: list[list[Any]] = []
    for method, payload in summary["methods"].items():
        agg = payload["aggregate"]
        aggregate_rows.append(
            [
                agg["label"],
                fmt(agg["online_success_mean"]),
                fmt(agg["validation_success_mean"]),
                fmt(agg["test_success_mean"]),
                fmt(agg["contexts_covered_mean"], 1),
                fmt(agg["admitted_demo_items_mean"], 1),
                fmt(agg["primitive_steps_total_mean"], 1),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Method",
                "Online",
                "Val",
                "Test",
                "Admitted ctx",
                "Demo items",
                "Primitive steps",
            ],
            aggregate_rows,
        )
    )

    lines.extend(["", "## Held-Out Test By Family", ""])
    families = sorted(
        {
            family
            for payload in summary["methods"].values()
            for family in payload["aggregate"]["test_by_family"]
        }
    )
    family_rows: list[list[Any]] = []
    for family in families:
        row: list[Any] = [family]
        for method in summary["methods"]:
            family_row = summary["methods"][method]["aggregate"]["test_by_family"].get(family)
            row.append(fmt(family_row["success_rate"]) if family_row else "-")
        family_rows.append(row)
    lines.extend(markdown_table(["Family", *[summary["methods"][m]["aggregate"]["label"] for m in summary["methods"]]], family_rows))

    lines.extend(["", "## Admitted Contexts By Family", ""])
    admitted_rows: list[list[Any]] = []
    for family in sorted(
        {
            family
            for payload in summary["methods"].values()
            for family in payload["aggregate"]["admitted_context_family_totals"]
        }
    ):
        row = [family]
        for method in summary["methods"]:
            row.append(summary["methods"][method]["aggregate"]["admitted_context_family_totals"].get(family, 0))
        admitted_rows.append(row)
    lines.extend(markdown_table(["Family", *[summary["methods"][m]["aggregate"]["label"] for m in summary["methods"]]], admitted_rows))

    lines.extend(["", "## Findings", ""])
    findings = summary["findings"]
    for key, value in sorted(findings.items()):
        lines.append(f"- `{key}`: `{fmt(value)}`")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    methods = parse_csv_strings(args.methods)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")
    summary = build_summary(budget=args.budget, seeds=seeds, methods=methods)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    write_json(output_json, summary)
    write_text(output_md, build_markdown(summary))
    print(json.dumps({"json": str(output_json.resolve()), "md": str(output_md.resolve())}, indent=2))


if __name__ == "__main__":
    main()
