#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from build_caver_round_artifacts import iter_trace_records
from caver_heuristic import compute_candidate_metrics
from stagee_dr_calibration import STAGEE_DR_DATASET_FORMAT
from stagee_dr_calibration import clamp_unit_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a candidate-level Stage-E DR pseudo-outcome dataset from online results and chunk traces."
    )
    parser.add_argument("--online-results", required=True, help="Path to the online rollout summary JSON.")
    parser.add_argument("--trace-path", required=True, help="Path to the chunk-trace JSONL emitted by the bridge.")
    parser.add_argument("--output-path", required=True, help="Output JSONL path for the candidate-level DR dataset.")
    parser.add_argument("--summary-path", required=True, help="Output JSON path for the DR dataset summary.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def resolve_lagged_nuisance_mean(metric: dict[str, Any]) -> float:
    for key in (
        "lagged_nuisance_mean",
        "lagged_dr_utility_mean",
        "admission_value_proxy",
        "raw_value_proxy",
    ):
        if key in metric and metric[key] is not None:
            return float(metric[key])
    return 0.0


def main() -> int:
    args = parse_args()

    online_results_path = Path(args.online_results).resolve()
    trace_path = Path(args.trace_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve()

    with online_results_path.open("r", encoding="utf-8") as handle:
        online_results = json.load(handle)

    context_records = list(online_results["contexts"])
    context_records_by_id = {str(record["context_id"]): record for record in context_records}
    if len(context_records_by_id) != len(context_records):
        raise ValueError("online results contain duplicate context ids")

    records_total = 0
    selected_records = 0
    safe_candidate_counts: collections.Counter[int] = collections.Counter()
    partition_counts: collections.Counter[str] = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    selector_mode_counts: collections.Counter[str] = collections.Counter()
    contexts_seen: set[str] = set()
    pseudo_min = float("inf")
    pseudo_max = float("-inf")

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for trace_record in iter_trace_records(trace_path):
            context_id = str(trace_record["context_id"])
            context = context_records_by_id.get(context_id)
            if context is None:
                raise ValueError(f"trace contains unknown context id not present in online results: {context_id}")
            contexts_seen.add(context_id)

            selector = dict(trace_record.get("selector", {}))
            safe_candidate_indices = [int(index) for index in selector.get("safe_candidate_indices", [])]
            if not safe_candidate_indices:
                continue
            selected_candidate_index = int(selector["selected_candidate_index"])
            candidate_probabilities = [float(value) for value in selector["candidate_probabilities"]]
            candidate_chunks = trace_record.get("candidate_chunks")
            if candidate_chunks is None:
                raise ValueError(f"trace record for context {context_id} is missing candidate chunks")
            recomputed_metrics = compute_candidate_metrics(
                candidate_chunks,
                candidate_provider_aux=trace_record.get("candidate_provider_aux"),
                history_vectors=None,
            )
            logged_metrics = selector.get("candidate_metric_table") or [{} for _ in recomputed_metrics]
            if len(logged_metrics) != len(recomputed_metrics):
                raise ValueError(
                    f"candidate metric table length mismatch for context {context_id}: "
                    f"{len(logged_metrics)} != {len(recomputed_metrics)}"
                )

            executed_outcome = 1.0 if bool(context.get("success")) else 0.0
            selector_mode = str(selector.get("selector_mode") or trace_record.get("selection_policy") or "unknown")
            selector_mode_counts[selector_mode] += 1

            for candidate_index in safe_candidate_indices:
                propensity = float(candidate_probabilities[candidate_index])
                if propensity <= 0.0:
                    raise ValueError(
                        f"safe candidate {candidate_index} in context {context_id} has non-positive propensity"
                    )
                recomputed_metric = dict(recomputed_metrics[candidate_index])
                logged_metric = dict(logged_metrics[candidate_index])
                nuisance_mean = resolve_lagged_nuisance_mean(logged_metric)
                candidate_selected = candidate_index == selected_candidate_index
                importance_weight = (1.0 / propensity) if candidate_selected else 0.0
                dr_pseudo_outcome = (
                    nuisance_mean + ((executed_outcome - nuisance_mean) / propensity)
                    if candidate_selected
                    else nuisance_mean
                )
                dr_pseudo_outcome_clipped = clamp_unit_interval(dr_pseudo_outcome)

                record = {
                    "dataset_format": STAGEE_DR_DATASET_FORMAT,
                    "context_id": context_id,
                    "policy_query_index": int(trace_record["policy_query_index"]),
                    "candidate_index": int(candidate_index),
                    "selected_candidate_index": selected_candidate_index,
                    "candidate_selected": bool(candidate_selected),
                    "candidate_count": int(selector.get("candidate_count") or len(candidate_chunks)),
                    "safe_candidate_count": len(safe_candidate_indices),
                    "logged_propensity": propensity,
                    "importance_weight": importance_weight,
                    "executed_outcome": executed_outcome,
                    "lagged_nuisance_mean": nuisance_mean,
                    "dr_pseudo_outcome": float(dr_pseudo_outcome),
                    "dr_pseudo_outcome_clipped": float(dr_pseudo_outcome_clipped),
                    "base_feature_vector": list(recomputed_metric["base_feature_vector"]),
                    "raw_uncertainty_proxy": float(recomputed_metric["raw_uncertainty_proxy"]),
                    "raw_diversity_proxy": float(recomputed_metric["raw_diversity_proxy"]),
                    "raw_novelty_proxy": float(logged_metric.get("raw_novelty_proxy", 0.0)),
                    "partition_name": context.get("partition_name"),
                    "proxy_family_id": context.get("proxy_family_id"),
                    "proposal_task": context.get("proposal_task"),
                    "round_index": int(context.get("budget", {}).get("round_index", 0)),
                    "round_context_index": int(context.get("budget", {}).get("round_context_index", 0)),
                    "selector_mode": selector_mode,
                    "selection_policy": selector.get("selection_policy"),
                    "utility_source": logged_metric.get("value_proxy_source"),
                    "completed_reason": trace_record.get("completed_reason"),
                }
                json.dump(record, handle, sort_keys=True, default=json_default)
                handle.write("\n")

                records_total += 1
                selected_records += int(candidate_selected)
                safe_candidate_counts[len(safe_candidate_indices)] += 1
                if context.get("partition_name") is not None:
                    partition_counts[str(context["partition_name"])] += 1
                if context.get("proxy_family_id") is not None:
                    family_counts[str(context["proxy_family_id"])] += 1
                pseudo_min = min(pseudo_min, float(dr_pseudo_outcome))
                pseudo_max = max(pseudo_max, float(dr_pseudo_outcome))

    if records_total <= 0:
        raise ValueError(f"DR dataset build produced zero records from {trace_path}")

    summary = {
        "workflow": "build_stagee_dr_dataset_v1",
        "online_results_path": str(online_results_path),
        "trace_path": str(trace_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "contexts_total": len(contexts_seen),
        "records_total": records_total,
        "selected_records": selected_records,
        "selected_record_fraction": (selected_records / float(records_total)),
        "safe_candidate_count_distribution": {str(key): value for key, value in safe_candidate_counts.items()},
        "partition_counts": dict(partition_counts),
        "family_counts": dict(family_counts),
        "selector_mode_counts": dict(selector_mode_counts),
        "dr_pseudo_outcome_min": float(pseudo_min),
        "dr_pseudo_outcome_max": float(pseudo_max),
    }
    write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
