#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from caver_heuristic import compute_candidate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact Stage-0 proxy-value dataset from the sharded warm-start demo and "
            "the Stage-0 context ledger without rereading the raw trace JSONL."
        )
    )
    parser.add_argument("--demo-manifest", required=True, help="Sharded Stage-0 warm-start demo manifest JSON.")
    parser.add_argument("--context-log", required=True, help="Stage-0 seed context log JSONL.")
    parser.add_argument("--output-path", required=True, help="Output JSONL dataset path.")
    parser.add_argument("--summary-path", required=True, help="Output JSON summary path.")
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension used to reshape flattened chunk actions (default: 7).",
    )
    return parser.parse_args()


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    path_type = Path
    if isinstance(value, path_type):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def main() -> int:
    args = parse_args()

    demo_manifest_path = Path(args.demo_manifest).resolve()
    context_log_path = Path(args.context_log).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "error: torch is required to read the Stage-0 demo shards; run this with the pi-StepNFT venv"
        ) from exc

    with demo_manifest_path.open("r", encoding="utf-8") as handle:
        demo_manifest = json.load(handle)

    context_records = read_jsonl(context_log_path)
    if not context_records:
        raise ValueError(f"context log contains zero records: {context_log_path}")

    total_expected_demo_items = sum(int(record["chunk_traces_written"]) for record in context_records)
    manifest_demo_items = int(demo_manifest["total_items"])
    if total_expected_demo_items != manifest_demo_items:
        raise ValueError(
            "context log chunk count does not match demo manifest item count: "
            f"{total_expected_demo_items} != {manifest_demo_items}"
        )

    dataset_records: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    positive_context_counts: dict[str, int] = {}
    policy_query_max = 0

    context_index = 0
    context = dict(context_records[context_index])
    remaining_in_context = int(context["chunk_traces_written"])
    policy_query_index = 0

    for shard in demo_manifest["shards"]:
        shard_path = (demo_manifest_path.parent / shard["path"]).resolve()
        items = torch.load(shard_path, map_location="cpu")
        for item in items:
            while remaining_in_context == 0:
                context_index += 1
                if context_index >= len(context_records):
                    raise ValueError("demo items exceed the available Stage-0 context records")
                context = dict(context_records[context_index])
                remaining_in_context = int(context["chunk_traces_written"])
                policy_query_index = 0

            action_tensor = item["action"].detach().cpu()
            horizon = int(item["rewards"].numel())
            expected_size = horizon * int(args.action_dim)
            if int(action_tensor.numel()) != expected_size:
                raise ValueError(
                    "unexpected flattened action size in demo shard: "
                    f"{action_tensor.numel()} != {expected_size}"
                )
            action_chunk = action_tensor.reshape(horizon, int(args.action_dim)).tolist()
            metric = compute_candidate_metrics([action_chunk], history_vectors=None)[0]

            reward_sum = float(item["rewards"].sum().item())
            chunk_success_label = int(bool(item["success_once"].any().item()))
            context_success_label = int(bool(context["success"]))
            context_trace_count = int(context["chunk_traces_written"])
            proxy_family_id = str(context.get("proxy_family_id") or "unknown")

            dataset_records.append(
                {
                    "context_id": str(context["context_id"]),
                    "proposal_task": context.get("proposal_task"),
                    "proxy_family_id": proxy_family_id,
                    "partition_name": context.get("partition_name"),
                    "policy_query_index": policy_query_index,
                    "context_trace_count": context_trace_count,
                    "context_success_label": context_success_label,
                    "chunk_success_label": chunk_success_label,
                    "reward_sum": reward_sum,
                    "chunk_action_horizon": horizon,
                    "raw_value_proxy": float(metric["raw_value_proxy"]),
                    "raw_uncertainty_proxy": float(metric["raw_uncertainty_proxy"]),
                    "raw_diversity_proxy": float(metric["raw_diversity_proxy"]),
                    "base_feature_vector": list(metric["base_feature_vector"]),
                }
            )

            family_counts[proxy_family_id] = family_counts.get(proxy_family_id, 0) + 1
            positive_context_counts[proxy_family_id] = positive_context_counts.get(proxy_family_id, 0) + context_success_label
            policy_query_max = max(policy_query_max, policy_query_index)
            remaining_in_context -= 1
            policy_query_index += 1

    if len(dataset_records) != manifest_demo_items:
        raise ValueError(f"dataset record count mismatch: {len(dataset_records)} != {manifest_demo_items}")

    if context_index != len(context_records) - 1 or remaining_in_context != 0:
        raise ValueError("did not consume every Stage-0 context while reconstructing the proxy dataset")

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in dataset_records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    summary = {
        "dataset_format": "stage0_value_proxy_dataset_v1",
        "demo_manifest_path": demo_manifest_path,
        "context_log_path": context_log_path,
        "output_path": output_path,
        "summary_path": summary_path,
        "records_total": len(dataset_records),
        "contexts_total": len(context_records),
        "chunk_action_horizon": int(dataset_records[0]["chunk_action_horizon"]) if dataset_records else 0,
        "action_dim": int(args.action_dim),
        "policy_query_index_max": policy_query_max,
        "context_success_positive_records": sum(record["context_success_label"] for record in dataset_records),
        "chunk_success_positive_records": sum(record["chunk_success_label"] for record in dataset_records),
        "context_success_positive_rate": (
            sum(record["context_success_label"] for record in dataset_records) / float(len(dataset_records))
            if dataset_records
            else 0.0
        ),
        "chunk_success_positive_rate": (
            sum(record["chunk_success_label"] for record in dataset_records) / float(len(dataset_records))
            if dataset_records
            else 0.0
        ),
        "family_counts": family_counts,
        "positive_context_counts_by_family": positive_context_counts,
    }
    write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
