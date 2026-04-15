#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from stagee_dr_calibration import STAGEE_DR_DATASET_FORMAT
from stagee_dr_calibration import clamp_unit_interval


SEED_PROXY_DATASET_FORMAT = "stage0_value_proxy_dataset_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Stage-0 executed seed-tuple dataset into the Stage-E calibrator-training format "
            "so round 1 can start from an explicit seed-fitted calibrator artifact."
        )
    )
    parser.add_argument("--input-path", required=True, help="Stage-0 seed tuple JSONL dataset.")
    parser.add_argument("--output-path", required=True, help="Output JSONL path for the seed calibrator dataset.")
    parser.add_argument("--summary-path", required=True, help="Output JSON summary path.")
    parser.add_argument(
        "--label-key",
        default="context_success_label",
        choices=("context_success_label", "chunk_success_label"),
        help="Verified seed label to copy into the calibrator target (default: context_success_label).",
    )
    parser.add_argument(
        "--nuisance-key",
        default="raw_value_proxy",
        choices=("raw_value_proxy", "zero"),
        help="Cheap seed nuisance value stored alongside the target (default: raw_value_proxy).",
    )
    parser.add_argument(
        "--weight-mode",
        default="inverse_context_trace_count",
        choices=("uniform", "inverse_context_trace_count"),
        help=(
            "uniform gives every tuple equal weight; inverse_context_trace_count gives each seed episode "
            "roughly equal total weight (default: inverse_context_trace_count)."
        ),
    )
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to decode {path} line {line_number}") from exc
    return records


def resolve_weight(record: dict[str, Any], *, weight_mode: str) -> float:
    if weight_mode == "uniform":
        return 1.0
    if weight_mode == "inverse_context_trace_count":
        return 1.0 / max(1.0, float(record.get("context_trace_count") or 1))
    raise ValueError(f"unsupported weight mode {weight_mode!r}")


def resolve_nuisance(record: dict[str, Any], *, nuisance_key: str) -> float:
    if nuisance_key == "zero":
        return 0.0
    return float(record.get(nuisance_key) or 0.0)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve()

    source_records = read_jsonl(input_path)
    if not source_records:
        raise ValueError(f"dataset contains zero records: {input_path}")

    family_counts: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[int] = collections.Counter()
    context_ids: set[str] = set()
    max_policy_query_index = 0
    sample_weight_total = 0.0
    context_trace_count_total = 0

    first_format = source_records[0].get("dataset_format")
    if first_format is not None and first_format != SEED_PROXY_DATASET_FORMAT:
        raise ValueError(f"unexpected input dataset format {first_format!r} in {input_path}")

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record_index, source in enumerate(source_records):
            target = clamp_unit_interval(float(source[args.label_key]))
            context_id = str(source["context_id"])
            proxy_family_id = str(source.get("proxy_family_id") or "unknown")
            policy_query_index = int(source.get("policy_query_index") or 0)
            context_trace_count = int(source.get("context_trace_count") or 1)
            nuisance_mean = resolve_nuisance(source, nuisance_key=args.nuisance_key)
            sample_weight = resolve_weight(source, weight_mode=args.weight_mode)

            output_record = {
                "dataset_format": STAGEE_DR_DATASET_FORMAT,
                "context_id": context_id,
                "policy_query_index": policy_query_index,
                "candidate_index": 0,
                "selected_candidate_index": 0,
                "candidate_selected": True,
                "candidate_count": 1,
                "safe_candidate_count": 1,
                "logged_propensity": 1.0,
                "importance_weight": 1.0,
                "sample_weight": float(sample_weight),
                "executed_outcome": float(target),
                "lagged_nuisance_mean": float(nuisance_mean),
                "dr_pseudo_outcome": float(target),
                "dr_pseudo_outcome_clipped": float(target),
                "base_feature_vector": list(source["base_feature_vector"]),
                "raw_uncertainty_proxy": float(source.get("raw_uncertainty_proxy") or 0.0),
                "raw_diversity_proxy": float(source.get("raw_diversity_proxy") or 0.0),
                "raw_novelty_proxy": 0.0,
                "partition_name": source.get("partition_name"),
                "proxy_family_id": proxy_family_id,
                "proposal_task": source.get("proposal_task"),
                "round_index": 0,
                "round_context_index": policy_query_index + 1,
                "selector_mode": "seed_calibrator_bootstrap_v1",
                "selection_policy": "seed_verified_tuple",
                "utility_source": "stage0_seed_verified_tuple",
                "completed_reason": "stage0_seed",
                "source_record_index": record_index,
                "source_context_trace_count": context_trace_count,
                "source_label_key": args.label_key,
                "source_nuisance_key": args.nuisance_key,
                "source_weight_mode": args.weight_mode,
            }
            json.dump(output_record, handle, sort_keys=True, default=json_default)
            handle.write("\n")

            family_counts[proxy_family_id] += 1
            label_counts[int(round(target))] += 1
            context_ids.add(context_id)
            max_policy_query_index = max(max_policy_query_index, policy_query_index)
            sample_weight_total += float(sample_weight)
            context_trace_count_total += context_trace_count

    records_total = len(source_records)
    summary = {
        "workflow": "build_stage0_seed_calibrator_dataset_v1",
        "input_path": input_path,
        "output_path": output_path,
        "summary_path": summary_path,
        "records_total": records_total,
        "contexts_total": len(context_ids),
        "family_counts": dict(family_counts),
        "label_key": args.label_key,
        "label_counts": {str(key): value for key, value in label_counts.items()},
        "label_positive_rate": (label_counts.get(1, 0) / float(records_total)),
        "nuisance_key": args.nuisance_key,
        "weight_mode": args.weight_mode,
        "sample_weight_total": float(sample_weight_total),
        "sample_weight_mean": float(sample_weight_total / float(records_total)),
        "context_trace_count_mean": float(context_trace_count_total / float(records_total)),
        "policy_query_index_max": max_policy_query_index,
    }
    write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
