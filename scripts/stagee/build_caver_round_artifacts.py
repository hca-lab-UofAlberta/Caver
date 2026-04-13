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

from caver_heuristic import CAVER_ADMISSION_IMPLEMENTATION_PHASE
from caver_heuristic import CAVER_ADMISSION_POLICY
from caver_heuristic import CAVER_SELECTOR_IMPLEMENTATION_PHASE
from caver_heuristic import CAVER_SELECTOR_MODE
from caver_heuristic import selected_metric_from_record
from caver_heuristic import summarize_admission_context
from caver_heuristic import summarize_admission_metrics

TRACE_INPUT_FORMAT = "stagee_trace_source_manifest_v1"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage-E CAVER selector and admission artifacts from a LIBERO online rollout, "
            "and write an admitted-trace subset for backend conversion."
        )
    )
    parser.add_argument("--online-results", required=True, help="Path to the online rollout summary JSON.")
    parser.add_argument("--trace-path", required=True, help="Path to the chunk-trace JSONL emitted by the bridge.")
    parser.add_argument(
        "--selector-context-path",
        required=True,
        help="Output JSONL path for per-context selector records.",
    )
    parser.add_argument(
        "--selector-summary-path",
        required=True,
        help="Output JSON path for the selector summary.",
    )
    parser.add_argument(
        "--admission-context-path",
        required=True,
        help="Output JSONL path for per-context admission decisions.",
    )
    parser.add_argument(
        "--admission-summary-path",
        required=True,
        help="Output JSON path for the admission summary.",
    )
    parser.add_argument(
        "--admitted-trace-path",
        required=True,
        help="Output JSONL path for the admitted trace subset.",
    )
    parser.add_argument(
        "--selector-mode",
        default=CAVER_SELECTOR_MODE,
        help="Selector implementation label written into the artifacts.",
    )
    parser.add_argument(
        "--admission-policy",
        default=CAVER_ADMISSION_POLICY,
        choices=("all_executed_nonerror", CAVER_ADMISSION_POLICY),
        help="Admission policy label written into the artifacts.",
    )
    parser.add_argument(
        "--require-candidate-bank",
        action="store_true",
        help="Require every trace record to include the full candidate bank.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True, default=json_default)
            handle.write("\n")


def load_trace_source_manifest(trace_path: Path) -> dict[str, Any] | None:
    first_nonempty: str | None = None
    with trace_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            first_nonempty = line
            break
    if first_nonempty is None:
        return None
    try:
        payload = json.loads(first_nonempty)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("trace_input_format") == TRACE_INPUT_FORMAT:
        return payload
    return None


def iter_trace_records_from_source(
    trace_path: Path,
    *,
    completed_prefix_contexts: int | None = None,
) -> Any:
    with trace_path.open("r", encoding="utf-8") as handle:
        contexts_seen: set[str] = set()
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if completed_prefix_contexts is not None and len(contexts_seen) >= completed_prefix_contexts:
                    break
                continue
            context_id = str(record["context_id"])
            if completed_prefix_contexts is not None and context_id not in contexts_seen:
                if len(contexts_seen) >= completed_prefix_contexts:
                    break
                contexts_seen.add(context_id)
            yield record


def iter_trace_records(trace_path: Path) -> Any:
    manifest = load_trace_source_manifest(trace_path)
    if manifest is None:
        yield from iter_trace_records_from_source(trace_path)
        return
    for source in manifest.get("sources", []):
        source_path = Path(source["path"]).resolve()
        completed_prefix_contexts = source.get("completed_prefix_contexts")
        yield from iter_trace_records_from_source(
            source_path,
            completed_prefix_contexts=(
                None if completed_prefix_contexts is None else int(completed_prefix_contexts)
            ),
        )


def make_context_trace_state() -> dict[str, Any]:
    return {
        "trace_record_count": 0,
        "trace_formats": set(),
        "trace_chunk_horizons": set(),
        "candidate_bank_complete": True,
        "selector_modes": [],
        "selected_candidate_indices": [],
        "selected_candidate_probabilities": [],
        "candidate_probability_vectors": [],
        "safe_candidate_counts": [],
        "candidate_action_lengths": [],
        "selected_metric_table": [],
    }


def build_context_trace_index(
    *,
    trace_path: Path,
    context_records_by_id: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    collections.Counter[str],
    collections.Counter[str],
    collections.Counter[str],
    int,
]:
    trace_states: dict[str, dict[str, Any]] = {}
    trace_formats: collections.Counter[str] = collections.Counter()
    trace_completed_reasons: collections.Counter[str] = collections.Counter()
    selector_mode_counts: collections.Counter[str] = collections.Counter()
    total_records = 0

    for record in iter_trace_records(trace_path):
        context_id = str(record["context_id"])
        if context_id not in context_records_by_id:
            raise ValueError(f"trace contains unknown context id not present in online results: {context_id}")
        state = trace_states.setdefault(context_id, make_context_trace_state())
        state["trace_record_count"] += 1
        total_records += 1

        trace_format = str(record.get("trace_format") or "unknown")
        state["trace_formats"].add(trace_format)
        trace_formats[trace_format] += 1
        trace_completed_reasons[str(record.get("completed_reason") or "unknown")] += 1

        if record.get("chunk_action_horizon") is not None:
            state["trace_chunk_horizons"].add(int(record["chunk_action_horizon"]))
        if "candidate_chunks" not in record:
            state["candidate_bank_complete"] = False

        selector_payload = record.get("selector", {})
        selector_mode = str(selector_payload.get("selector_mode") or "unknown")
        state["selector_modes"].append(selector_mode)
        selector_mode_counts[selector_mode] += 1
        state["selected_candidate_indices"].append(selector_payload.get("selected_candidate_index"))
        state["selected_candidate_probabilities"].append(selector_payload.get("selected_candidate_probability"))
        state["candidate_probability_vectors"].append(selector_payload.get("candidate_probabilities"))
        state["safe_candidate_counts"].append(selector_payload.get("safe_candidate_count"))
        state["candidate_action_lengths"].append(selector_payload.get("candidate_action_lengths"))
        state["selected_metric_table"].append(selected_metric_from_record(record))

    if total_records <= 0:
        raise ValueError(f"trace file contains zero records: {trace_path}")

    return trace_states, trace_formats, trace_completed_reasons, selector_mode_counts, total_records


def write_admitted_trace_subset(
    *,
    trace_path: Path,
    admitted_trace_path: Path,
    admitted_context_ids: set[str],
) -> int:
    ensure_parent(admitted_trace_path)
    written = 0
    with admitted_trace_path.open("w", encoding="utf-8") as handle:
        for record in iter_trace_records(trace_path):
            if str(record["context_id"]) not in admitted_context_ids:
                continue
            json.dump(record, handle, sort_keys=True, default=json_default)
            handle.write("\n")
            written += 1
    return written


def main() -> int:
    args = parse_args()

    online_results_path = Path(args.online_results).resolve()
    trace_path = Path(args.trace_path).resolve()
    selector_context_path = Path(args.selector_context_path).resolve()
    selector_summary_path = Path(args.selector_summary_path).resolve()
    admission_context_path = Path(args.admission_context_path).resolve()
    admission_summary_path = Path(args.admission_summary_path).resolve()
    admitted_trace_path = Path(args.admitted_trace_path).resolve()

    with online_results_path.open("r", encoding="utf-8") as handle:
        online_results = json.load(handle)

    context_records: list[dict[str, Any]] = list(online_results["contexts"])
    context_records_by_id = {str(record["context_id"]): record for record in context_records}
    if len(context_records_by_id) != len(context_records):
        raise ValueError("online results contain duplicate context ids")

    trace_states, trace_formats, trace_completed_reasons, selector_mode_counts, total_trace_records = build_context_trace_index(
        trace_path=trace_path,
        context_records_by_id=context_records_by_id,
    )

    candidate_count = int(online_results["config"]["candidate_count"])
    selection_policy = str(online_results["config"]["selection_policy"])
    selector_seed = int(online_results["config"]["selector_seed"])

    selector_context_records: list[dict[str, Any]] = []
    admission_context_records: list[dict[str, Any]] = []
    admitted_context_ids: set[str] = set()
    admitted_family_counts: collections.Counter[str] = collections.Counter()
    admitted_partition_counts: collections.Counter[str] = collections.Counter()
    contexts_with_candidate_bank = 0
    contexts_missing_candidate_bank = 0
    policy_queries_total = 0
    admitted_trace_records = 0

    for context in context_records:
        context_id = str(context["context_id"])
        trace_state = trace_states.get(context_id, make_context_trace_state())
        context_trace_formats = sorted(trace_state["trace_formats"])
        trace_record_count = int(trace_state["trace_record_count"])
        policy_queries_total += trace_record_count
        trace_chunk_horizons = sorted(trace_state["trace_chunk_horizons"])
        candidate_bank_logged = trace_record_count > 0 and bool(trace_state["candidate_bank_complete"])
        if candidate_bank_logged:
            contexts_with_candidate_bank += 1
        else:
            contexts_missing_candidate_bank += 1
        if args.require_candidate_bank and not candidate_bank_logged:
            raise ValueError(
                f"context {context_id} is missing candidate-bank logging required for CAVER scaffold artifacts"
            )

        selected_candidate_indices = list(trace_state["selected_candidate_indices"])
        selected_candidate_probabilities = list(trace_state["selected_candidate_probabilities"])
        candidate_probability_vectors = list(trace_state["candidate_probability_vectors"])
        safe_candidate_counts = list(trace_state["safe_candidate_counts"])
        candidate_action_lengths = list(trace_state["candidate_action_lengths"])
        per_query_selected_metrics = list(trace_state["selected_metric_table"])
        context_selector_modes = sorted(set(trace_state["selector_modes"]))
        if not context_selector_modes:
            context_selector_mode = args.selector_mode
        elif len(context_selector_modes) == 1:
            context_selector_mode = context_selector_modes[0]
        else:
            context_selector_mode = "mixed"

        selector_context_record = {
            "implementation_phase": CAVER_SELECTOR_IMPLEMENTATION_PHASE,
            "selector_mode": context_selector_mode,
            "selector_mode_trace_values": context_selector_modes,
            "context_id": context_id,
            "suite_name": context.get("suite_name"),
            "partition_name": context.get("partition_name"),
            "proxy_family_id": context.get("proxy_family_id"),
            "proposal_task": context.get("proposal_task"),
            "task_id": context.get("task_id"),
            "task_name": context.get("task_name"),
            "episode_idx": context.get("episode_idx"),
            "init_state_index": context.get("init_state_index"),
            "candidate_count": candidate_count,
            "selection_policy": selection_policy,
            "selector_seed": selector_seed,
            "trace_record_count": trace_record_count,
            "trace_formats": context_trace_formats,
            "trace_chunk_horizons": trace_chunk_horizons,
            "candidate_bank_logged": candidate_bank_logged,
            "selected_candidate_indices": selected_candidate_indices,
            "selected_candidate_probabilities": selected_candidate_probabilities,
            "candidate_probability_vectors": candidate_probability_vectors,
            "safe_candidate_counts": safe_candidate_counts,
            "candidate_action_lengths": candidate_action_lengths,
            "selected_metric_table": per_query_selected_metrics,
        }
        selector_context_records.append(selector_context_record)

        has_trace = trace_record_count > 0
        has_error = context.get("error") is not None
        safety_abort = bool(context.get("budget", {}).get("safety_abort", False))
        admission_metrics = summarize_admission_metrics(context=context, selected_metrics=per_query_selected_metrics)
        admit_for_training = bool(admission_metrics["admit_for_training"])
        if args.admission_policy == "all_executed_nonerror":
            admit_for_training = has_trace and not has_error and not safety_abort
        if admit_for_training:
            admitted_context_ids.add(context_id)
            admitted_trace_records += trace_record_count
            if context.get("proxy_family_id") is not None:
                admitted_family_counts[str(context["proxy_family_id"])] += 1
            if context.get("partition_name") is not None:
                admitted_partition_counts[str(context["partition_name"])] += 1

        if args.admission_policy == "all_executed_nonerror":
            if not has_trace:
                admission_reason = "missing_trace_records"
            elif has_error:
                admission_reason = "context_error"
            elif safety_abort:
                admission_reason = "safety_abort"
            else:
                admission_reason = "admit_executed_nonerror"
            admission_confidence = 1.0 if admit_for_training else 0.0
        else:
            admission_reason = str(admission_metrics["admission_reason"])
            admission_confidence = float(admission_metrics["admission_confidence"])

        admission_context_record = {
            "implementation_phase": (
                "caver_scaffold_v1"
                if args.admission_policy == "all_executed_nonerror"
                else CAVER_ADMISSION_IMPLEMENTATION_PHASE
            ),
            "admission_policy": args.admission_policy,
            "context_id": context_id,
            "suite_name": context.get("suite_name"),
            "partition_name": context.get("partition_name"),
            "proxy_family_id": context.get("proxy_family_id"),
            "proposal_task": context.get("proposal_task"),
            "task_id": context.get("task_id"),
            "task_name": context.get("task_name"),
            "episode_idx": context.get("episode_idx"),
            "init_state_index": context.get("init_state_index"),
            "success": context.get("success"),
            "error": context.get("error"),
            "policy_steps": context.get("policy_steps"),
            "chunk_traces_written": context.get("chunk_traces_written"),
            "budget": context.get("budget"),
            "trace_record_count": trace_record_count,
            "candidate_bank_logged": candidate_bank_logged,
            "admit_for_training": admit_for_training,
            "admission_reason": admission_reason,
            "admission_confidence": admission_confidence,
            "executed_value_mean": admission_metrics["executed_value_mean"],
            "executed_uncertainty_mean": admission_metrics["executed_uncertainty_mean"],
            "executed_diversity_mean": admission_metrics["executed_diversity_mean"],
            "executed_novelty_mean": admission_metrics["executed_novelty_mean"],
            "executed_lcb": admission_metrics["executed_lcb"],
            "selected_query_count": admission_metrics["selected_query_count"],
            "acceptance_threshold": admission_metrics["acceptance_threshold"],
            "kappa": admission_metrics["kappa"],
        }
        admission_context_records.append(admission_context_record)

    if not selector_mode_counts:
        resolved_selector_mode = args.selector_mode
    elif len(selector_mode_counts) == 1:
        resolved_selector_mode = next(iter(selector_mode_counts))
    else:
        resolved_selector_mode = "mixed"

    written_admitted_trace_records = write_admitted_trace_subset(
        trace_path=trace_path,
        admitted_trace_path=admitted_trace_path,
        admitted_context_ids=admitted_context_ids,
    )
    if written_admitted_trace_records != admitted_trace_records:
        raise ValueError(
            "admitted trace record count mismatch between summary pass "
            f"({admitted_trace_records}) and trace subset write ({written_admitted_trace_records})"
        )
    write_jsonl(selector_context_path, selector_context_records)
    write_jsonl(admission_context_path, admission_context_records)

    selector_summary = {
        "implementation_phase": (
            "caver_scaffold_v1"
            if resolved_selector_mode == "logged_policy_passthrough"
            else CAVER_SELECTOR_IMPLEMENTATION_PHASE
        ),
        "selector_mode": resolved_selector_mode,
        "selector_mode_counts": dict(selector_mode_counts),
        "online_results_path": online_results_path,
        "trace_path": trace_path,
        "selector_context_path": selector_context_path,
        "contexts_total": len(context_records),
        "contexts_with_candidate_bank": contexts_with_candidate_bank,
        "contexts_missing_candidate_bank": contexts_missing_candidate_bank,
        "policy_queries_total": policy_queries_total,
        "trace_records_total": total_trace_records,
        "candidate_count": candidate_count,
        "selection_policy": selection_policy,
        "selector_seed": selector_seed,
        "value_proxy_model_path": online_results["config"].get("value_proxy_model_path"),
        "value_proxy_model_id": online_results["config"].get("value_proxy_model_id"),
        "trace_formats": dict(trace_formats),
        "trace_completed_reasons": dict(trace_completed_reasons),
    }
    write_json(selector_summary_path, selector_summary)

    admission_summary = {
        "implementation_phase": (
            "caver_scaffold_v1"
            if args.admission_policy == "all_executed_nonerror"
            else CAVER_ADMISSION_IMPLEMENTATION_PHASE
        ),
        "admission_policy": args.admission_policy,
        "online_results_path": online_results_path,
        "trace_path": trace_path,
        "admission_context_path": admission_context_path,
        "admitted_trace_path": admitted_trace_path,
        "contexts_total": len(context_records),
        "contexts_admitted": len(admitted_context_ids),
        "contexts_rejected": len(context_records) - len(admitted_context_ids),
        "admitted_trace_records": admitted_trace_records,
        "rejected_context_ids": sorted(
            record["context_id"] for record in admission_context_records if not record["admit_for_training"]
        ),
        "admitted_family_counts": dict(admitted_family_counts),
        "admitted_partition_counts": dict(admitted_partition_counts),
        "lcb_acceptance_counts": dict(
            collections.Counter(record["admission_reason"] for record in admission_context_records)
        ),
    }
    write_json(admission_summary_path, admission_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
