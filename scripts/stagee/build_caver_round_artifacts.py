#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import gzip
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
from caver_heuristic import segment_repair_eligible
from caver_heuristic import selected_metric_from_record
from caver_heuristic import summarize_admission_context
from caver_heuristic import summarize_admission_metrics
from stage0_value_proxy import load_value_proxy_model
from stagee_dr_calibration import load_stagee_dr_calibrator_model

TRACE_INPUT_FORMAT = "stagee_trace_source_manifest_v1"
ADMISSION_POLICY_CHOICES = (
    "all_executed_nonerror",
    "success_only",
    "caver_success_preserving",
    "caver_relaxed_lcb",
    "caver_top_m_success",
    "caver_family_balanced_success",
    "caver_failure_diagnostic_success",
    "caver_hard_family_rescue",
    "caver_family_segment_repair",
    CAVER_ADMISSION_POLICY,
)


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
        "--demo-trace-path",
        default=None,
        help=(
            "Optional full observation-bearing trace used when writing admitted backend demos. "
            "If omitted, --trace-path is used for both selector/admission and demo filtering."
        ),
    )
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
        "--skip-admitted-trace-write",
        action="store_true",
        help=(
            "Write selector/admission summaries only and skip the admitted trace subset. "
            "This is for fast admission-rule smoke tests; backend training still requires the trace."
        ),
    )
    parser.add_argument(
        "--selector-mode",
        default=CAVER_SELECTOR_MODE,
        help="Selector implementation label written into the artifacts.",
    )
    parser.add_argument(
        "--admission-policy",
        default=CAVER_ADMISSION_POLICY,
        choices=ADMISSION_POLICY_CHOICES,
        help="Admission policy label written into the artifacts.",
    )
    parser.add_argument(
        "--admission-kappa",
        type=float,
        default=None,
        help="Optional kappa override for LCB-style admission policies.",
    )
    parser.add_argument(
        "--admission-threshold",
        type=float,
        default=None,
        help="Optional acceptance-threshold override for LCB-style admission policies.",
    )
    parser.add_argument(
        "--top-m-success-count",
        type=int,
        default=0,
        help=(
            "For caver_top_m_success, admit the top M successful contexts by executed LCB. "
            "If omitted or <=0, all successful contexts are admitted."
        ),
    )
    parser.add_argument(
        "--family-min-success-count",
        type=int,
        default=0,
        help=(
            "For caver_family_balanced_success, admit at least this many successful contexts "
            "per proxy family by executed LCB when available. For caver_hard_family_rescue "
            "and caver_failure_diagnostic_success, this is the per-family minimum used to "
            "select hard-family near misses."
        ),
    )
    parser.add_argument(
        "--rescue-family-ids",
        default="",
        help=(
            "Comma-separated proxy families eligible for hard-family rescue or failure diagnostics. "
            "If omitted, families below --family-min-success-count are rescued automatically."
        ),
    )
    parser.add_argument(
        "--rescue-per-family-count",
        type=int,
        default=0,
        help=(
            "For hard-family rescue or failure diagnostics, select up to this many failed near-miss "
            "contexts per rescued family. If omitted, --family-min-success-count is used."
        ),
    )
    parser.add_argument(
        "--repair-min-trace-records",
        type=int,
        default=1,
        help=(
            "For caver_family_segment_repair, minimum chunk records kept after a verified "
            "progress segment is selected."
        ),
    )
    parser.add_argument(
        "--repair-max-trace-records",
        type=int,
        default=12,
        help=(
            "For caver_family_segment_repair, maximum chunk-prefix records to keep from a "
            "repaired failed context after verified progress selection."
        ),
    )
    parser.add_argument(
        "--repair-min-progress",
        type=float,
        default=0.03,
        help="Minimum verified task-progress gain required for caver_family_segment_repair.",
    )
    parser.add_argument(
        "--repair-min-primitive-steps",
        type=int,
        default=4,
        help="Minimum primitive steps before a failed prefix can be repaired.",
    )
    parser.add_argument(
        "--repair-max-regression",
        type=float,
        default=0.10,
        help="Maximum allowed progress regression before the selected repair endpoint.",
    )
    parser.add_argument(
        "--require-candidate-bank",
        action="store_true",
        help="Require every trace record to include the full candidate bank.",
    )
    parser.add_argument(
        "--value-proxy-model-path",
        default=None,
        help="Optional fitted Stage-0 value-proxy JSON used when trace metrics must be recomputed.",
    )
    parser.add_argument(
        "--dr-calibrator-model-path",
        default=None,
        help="Optional lagged Stage-E DR calibrator JSON used when trace metrics must be recomputed.",
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


def open_text_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_trace_source_manifest(trace_path: Path) -> dict[str, Any] | None:
    first_nonempty: str | None = None
    with open_text_maybe_gzip(trace_path, "rt") as handle:
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
    with open_text_maybe_gzip(trace_path, "rt") as handle:
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


def iter_trace_records(
    trace_path: Path,
    *,
    completed_prefix_contexts: int | None = None,
    _depth: int = 0,
) -> Any:
    if _depth > 8:
        raise ValueError(f"trace-source manifest nesting is too deep at {trace_path}")
    manifest = load_trace_source_manifest(trace_path)
    if manifest is None:
        yield from iter_trace_records_from_source(
            trace_path,
            completed_prefix_contexts=completed_prefix_contexts,
        )
        return
    contexts_seen: set[str] = set()
    for source in manifest.get("sources", []):
        source_path = Path(source["path"]).resolve()
        source_completed_prefix_contexts = source.get("completed_prefix_contexts")
        for record in iter_trace_records(
            source_path,
            completed_prefix_contexts=(
                None
                if source_completed_prefix_contexts is None
                else int(source_completed_prefix_contexts)
            ),
            _depth=_depth + 1,
        ):
            if completed_prefix_contexts is not None:
                context_id = str(record["context_id"])
                if context_id not in contexts_seen:
                    if len(contexts_seen) >= completed_prefix_contexts:
                        return
                    contexts_seen.add(context_id)
            yield record


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
        "repair_trace_records": [],
    }


def build_context_trace_index(
    *,
    trace_path: Path,
    context_records_by_id: dict[str, dict[str, Any]],
    value_proxy_model: dict[str, Any] | None,
    dr_calibrator_model: dict[str, Any] | None,
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
        state["selected_metric_table"].append(
            selected_metric_from_record(
                record,
                value_proxy_model=value_proxy_model,
                dr_calibrator_model=dr_calibrator_model,
            )
        )
        state["repair_trace_records"].append(
            {
                "policy_query_index": int(record.get("policy_query_index") or 0),
                "steps_executed": int(record.get("steps_executed") or len(record.get("actions") or [])),
                "stage0_progress_start": record.get("stage0_progress_start"),
                "stage0_progress_sequence": record.get("stage0_progress_sequence"),
            }
        )

    if total_records <= 0:
        raise ValueError(f"trace file contains zero records: {trace_path}")

    return trace_states, trace_formats, trace_completed_reasons, selector_mode_counts, total_records


def write_admitted_trace_subset(
    *,
    trace_path: Path,
    admitted_trace_path: Path,
    admission_context_records: list[dict[str, Any]],
) -> int:
    ensure_parent(admitted_trace_path)
    admitted_specs = {
        str(record["context_id"]): record
        for record in admission_context_records
        if bool(record.get("admit_for_training", False))
    }
    expected_records = sum(effective_admitted_trace_record_count(record) for record in admitted_specs.values())
    written = 0
    with open_text_maybe_gzip(admitted_trace_path, "wt") as handle:
        for record in iter_trace_records(trace_path):
            context_id = str(record["context_id"])
            spec = admitted_specs.get(context_id)
            if spec is None:
                continue
            if bool(spec.get("segment_repair_for_training", False)):
                policy_query_index = int(record.get("policy_query_index") or 0)
                start_index = int(spec.get("repair_trace_record_start") or 1)
                end_index = int(spec.get("repair_trace_record_end") or 0)
                if policy_query_index < start_index or policy_query_index > end_index:
                    continue
                record = dict(record)
                record["caver_policy_imitation_source"] = "verified_failed_segment_repair"
                record["repair_trace_record_start"] = start_index
                record["repair_trace_record_end"] = end_index
                record["repair_policy"] = spec.get("repair_policy")
                record["repair_score"] = spec.get("repair_score")
                record["repair_start_step"] = spec.get("repair_start_step")
                record["repair_end_step"] = spec.get("repair_end_step")
                record["repair_progress_gain"] = spec.get("repair_progress_gain")
                record["repair_family_id"] = spec.get("repair_family_id")
                record["repair_progress_source"] = spec.get("repair_progress_source")
                record["full_failed_trace_admitted"] = False
            else:
                record = dict(record)
                record["caver_policy_imitation_source"] = "verified_success"
            json.dump(record, handle, sort_keys=True, default=json_default)
            handle.write("\n")
            written += 1
            if expected_records > 0 and written >= expected_records:
                break
    return written


def is_scaffold_admission_policy(policy: str) -> bool:
    return policy in {"all_executed_nonerror", "success_only", "caver_success_preserving"}


def success_eligible(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("success"))
        and int(record.get("trace_record_count") or 0) > 0
        and record.get("error") is None
        and not bool((record.get("budget") or {}).get("safety_abort", False))
    )


def rescue_eligible(record: dict[str, Any]) -> bool:
    return (
        not bool(record.get("success"))
        and int(record.get("trace_record_count") or 0) > 0
        and record.get("error") is None
        and not bool((record.get("budget") or {}).get("safety_abort", False))
    )


def parse_family_ids(raw: str) -> set[str]:
    return {token.strip() for token in raw.split(",") if token.strip()}


def recompute_admitted_counts(
    admission_context_records: list[dict[str, Any]],
) -> tuple[set[str], collections.Counter[str], collections.Counter[str], int]:
    admitted_context_ids: set[str] = set()
    admitted_family_counts: collections.Counter[str] = collections.Counter()
    admitted_partition_counts: collections.Counter[str] = collections.Counter()
    admitted_trace_records = 0
    for record in admission_context_records:
        if not bool(record["admit_for_training"]):
            continue
        context_id = str(record["context_id"])
        admitted_context_ids.add(context_id)
        admitted_trace_records += effective_admitted_trace_record_count(record)
        if record.get("proxy_family_id") is not None:
            admitted_family_counts[str(record["proxy_family_id"])] += 1
        if record.get("partition_name") is not None:
            admitted_partition_counts[str(record["partition_name"])] += 1
    return admitted_context_ids, admitted_family_counts, admitted_partition_counts, admitted_trace_records


def effective_admitted_trace_record_count(record: dict[str, Any]) -> int:
    if not bool(record.get("admit_for_training", False)):
        return 0
    if bool(record.get("segment_repair_for_training", False)):
        start_index = int(record.get("repair_trace_record_start") or 1)
        end_index = int(record.get("repair_trace_record_end") or 0)
        if end_index < start_index:
            return 0
        return end_index - start_index + 1
    return int(record.get("trace_record_count") or 0)


def apply_global_admission_policy(
    *,
    admission_context_records: list[dict[str, Any]],
    repair_trace_records_by_context: dict[str, list[dict[str, Any]]],
    admission_policy: str,
    top_m_success_count: int,
    family_min_success_count: int,
    rescue_family_ids: set[str],
    rescue_per_family_count: int,
    repair_min_trace_records: int,
    repair_max_trace_records: int,
    repair_min_progress: float,
    repair_min_primitive_steps: int,
    repair_max_regression: float,
) -> None:
    if admission_policy not in {
        "caver_top_m_success",
        "caver_family_balanced_success",
        "caver_failure_diagnostic_success",
        "caver_hard_family_rescue",
        "caver_family_segment_repair",
    }:
        return

    for record in admission_context_records:
        record["admit_for_training"] = False
        record["failure_diagnostic_for_calibration"] = False
        record["segment_repair_for_training"] = False
        record["repair_trace_record_start"] = None
        record["repair_trace_record_end"] = None
        record["repair_trace_records"] = 0
        record["repair_score"] = None
        record["repair_policy"] = None
        record["repair_start_step"] = None
        record["repair_end_step"] = None
        record["repair_progress_gain"] = None
        record["repair_family_id"] = None
        record["repair_progress_source"] = None
        record["repair_progress_point_count"] = int(record.get("repair_progress_point_count") or 0)
        record["full_failed_trace_admitted"] = False
        if success_eligible(record):
            record["admission_reason"] = "success_not_selected_by_global_policy"
        else:
            if int(record.get("trace_record_count") or 0) <= 0:
                record["admission_reason"] = "missing_trace_records"
            elif record.get("error") is not None:
                record["admission_reason"] = "context_error"
            elif bool((record.get("budget") or {}).get("safety_abort", False)):
                record["admission_reason"] = "safety_abort"
            elif rescue_eligible(record):
                record["admission_reason"] = "failed_execution_candidate_rescue"
            else:
                record["admission_reason"] = "failed_execution"
        record["admission_confidence"] = 0.0

    eligible = [record for record in admission_context_records if success_eligible(record)]
    eligible.sort(key=lambda record: (float(record.get("executed_lcb") or 0.0), str(record["context_id"])), reverse=True)

    selected_context_ids: set[str] = set()
    if admission_policy == "caver_top_m_success":
        limit = int(top_m_success_count)
        if limit <= 0:
            limit = len(eligible)
        selected_context_ids = {str(record["context_id"]) for record in eligible[:limit]}
    elif admission_policy == "caver_family_balanced_success":
        family_min = int(family_min_success_count)
        if family_min <= 0:
            family_min = 1
        by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in eligible:
            by_family[str(record.get("proxy_family_id") or "unknown")].append(record)
        for family_records in by_family.values():
            for record in family_records[:family_min]:
                selected_context_ids.add(str(record["context_id"]))
    elif admission_policy in {
        "caver_failure_diagnostic_success",
        "caver_hard_family_rescue",
        "caver_family_segment_repair",
    }:
        # Preserve all verified successes, then add high-ranked near misses for families
        # whose successful admissions are starved. The hard-rescue policy admits full
        # near misses for training; the segment-repair policy admits only a bounded
        # verified-progress prefix; the failure-diagnostic policy only marks them for
        # calibration/analysis and keeps them out of the backend imitation trace.
        selected_context_ids = {str(record["context_id"]) for record in eligible}
        family_min = int(family_min_success_count)
        if family_min <= 0:
            family_min = 2
        rescue_limit = int(rescue_per_family_count)
        if rescue_limit <= 0:
            rescue_limit = family_min

        success_counts: collections.Counter[str] = collections.Counter(
            str(record.get("proxy_family_id") or "unknown") for record in eligible
        )
        all_families = {
            str(record.get("proxy_family_id") or "unknown")
            for record in admission_context_records
        }
        target_families = set(rescue_family_ids)
        if not target_families:
            target_families = {
                family for family in all_families if int(success_counts.get(family, 0)) < family_min
            }

        rescue_by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in admission_context_records:
            family = str(record.get("proxy_family_id") or "unknown")
            if family not in target_families or not rescue_eligible(record):
                continue
            rescue_by_family[family].append(record)
        for family, family_records in rescue_by_family.items():
            family_records.sort(
                key=lambda record: (
                    float(record.get("executed_lcb") or 0.0),
                    float(record.get("executed_value_mean") or 0.0),
                    -float(record.get("policy_steps") or 0.0),
                    str(record["context_id"]),
                ),
                reverse=True,
            )
            needed = max(0, family_min - int(success_counts.get(family, 0)))
            limit = min(rescue_limit, needed)
            if admission_policy == "caver_family_segment_repair":
                repair_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for record in family_records:
                    context_id = str(record["context_id"])
                    trace_records = repair_trace_records_by_context.get(context_id, [])
                    repair_cap = int(repair_max_trace_records)
                    if repair_cap > 0:
                        trace_records = trace_records[:repair_cap]
                    repair = segment_repair_eligible(
                        record,
                        trace_records,
                        min_progress_gain=float(repair_min_progress),
                        min_steps=int(repair_min_primitive_steps),
                        max_regression=float(repair_max_regression),
                    )
                    if repair is None:
                        record["admission_reason"] = "failed_no_verified_progress_repair"
                        continue
                    repair_candidates.append((record, repair))

                repair_candidates.sort(
                    key=lambda item: (
                        float(item[1].get("repair_progress_gain") or 0.0),
                        float(item[0].get("executed_lcb") or 0.0),
                        float(item[0].get("executed_value_mean") or 0.0),
                        -float(item[0].get("policy_steps") or 0.0),
                        str(item[0]["context_id"]),
                    ),
                    reverse=True,
                )
                for record, repair in repair_candidates[:limit]:
                    context_id = str(record["context_id"])
                    repair_end_query = int(repair["repair_end_query_index"])
                    if repair_end_query < max(1, int(repair_min_trace_records)):
                        record["admission_reason"] = "failed_repair_shorter_than_min_trace_records"
                        continue
                    selected_context_ids.add(context_id)
                    record.update(repair)
                    record["repair_trace_record_start"] = 1
                    record["repair_trace_record_end"] = repair_end_query
                    record["repair_trace_records"] = repair_end_query
                    record["repair_family_id"] = family
                    record["repair_score"] = float(repair["repair_progress_gain"])
                    record["full_failed_trace_admitted"] = False
                    record["segment_repair_for_training"] = bool(
                        int(record.get("repair_trace_records") or 0) > 0
                    )
                for record, repair in repair_candidates[limit:]:
                    if str(record["context_id"]) in selected_context_ids:
                        continue
                    record.update(repair)
                    record["repair_score"] = float(repair["repair_progress_gain"])
                    record["repair_family_id"] = family
                    record["admission_reason"] = "failed_repair_not_selected"
                    record["full_failed_trace_admitted"] = False
                    record["segment_repair_for_training"] = False
                continue

            for record in family_records[:limit]:
                if admission_policy == "caver_hard_family_rescue":
                    selected_context_ids.add(str(record["context_id"]))
                else:
                    record["failure_diagnostic_for_calibration"] = True
                    record["admission_reason"] = "failure_diagnostic_hard_family_near_miss"
                    record["admission_confidence"] = float(
                        max(0.0, min(1.0, float(record.get("executed_lcb") or 0.0)))
                    )

    for record in admission_context_records:
        context_id = str(record["context_id"])
        if context_id not in selected_context_ids:
            continue
        record["admit_for_training"] = True
        if admission_policy == "caver_top_m_success":
            record["admission_reason"] = "admit_top_m_success"
        elif admission_policy == "caver_family_balanced_success":
            record["admission_reason"] = "admit_family_balanced_success"
        elif admission_policy == "caver_failure_diagnostic_success":
            record["admission_reason"] = "admit_success_failure_diagnostic"
        elif admission_policy == "caver_family_segment_repair" and bool(
            record.get("segment_repair_for_training", False)
        ):
            record["admission_reason"] = "admit_hard_family_segment_repair"
        elif admission_policy == "caver_family_segment_repair":
            record["admission_reason"] = "admit_success_with_segment_repair"
        elif success_eligible(record):
            record["admission_reason"] = "admit_success_with_hard_family_rescue"
        else:
            record["admission_reason"] = "admit_hard_family_near_miss"
        record["admission_confidence"] = float(max(0.0, min(1.0, float(record.get("executed_lcb") or 0.0))))


def main() -> int:
    args = parse_args()

    online_results_path = Path(args.online_results).resolve()
    trace_path = Path(args.trace_path).resolve()
    demo_trace_path = Path(args.demo_trace_path).resolve() if args.demo_trace_path is not None else trace_path
    selector_context_path = Path(args.selector_context_path).resolve()
    selector_summary_path = Path(args.selector_summary_path).resolve()
    admission_context_path = Path(args.admission_context_path).resolve()
    admission_summary_path = Path(args.admission_summary_path).resolve()
    admitted_trace_path = Path(args.admitted_trace_path).resolve()
    value_proxy_model_path = (
        Path(args.value_proxy_model_path).resolve() if args.value_proxy_model_path is not None else None
    )
    dr_calibrator_model_path = (
        Path(args.dr_calibrator_model_path).resolve() if args.dr_calibrator_model_path is not None else None
    )
    value_proxy_model = load_value_proxy_model(value_proxy_model_path) if value_proxy_model_path is not None else None
    dr_calibrator_model = (
        load_stagee_dr_calibrator_model(dr_calibrator_model_path) if dr_calibrator_model_path is not None else None
    )

    with online_results_path.open("r", encoding="utf-8") as handle:
        online_results = json.load(handle)

    context_records: list[dict[str, Any]] = list(online_results["contexts"])
    context_records_by_id = {str(record["context_id"]): record for record in context_records}
    if len(context_records_by_id) != len(context_records):
        raise ValueError("online results contain duplicate context ids")

    trace_states, trace_formats, trace_completed_reasons, selector_mode_counts, total_trace_records = build_context_trace_index(
        trace_path=trace_path,
        context_records_by_id=context_records_by_id,
        value_proxy_model=value_proxy_model,
        dr_calibrator_model=dr_calibrator_model,
    )

    candidate_count = int(online_results["config"]["candidate_count"])
    selection_policy = str(online_results["config"]["selection_policy"])
    selector_seed = int(online_results["config"]["selector_seed"])

    selector_context_records: list[dict[str, Any]] = []
    admission_context_records: list[dict[str, Any]] = []
    repair_trace_records_by_context: dict[str, list[dict[str, Any]]] = {}
    contexts_with_candidate_bank = 0
    contexts_missing_candidate_bank = 0
    policy_queries_total = 0

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
        repair_trace_records = list(trace_state["repair_trace_records"])
        repair_trace_records_by_context[context_id] = repair_trace_records
        repair_progress_point_count = sum(
            len(record.get("stage0_progress_sequence") or [])
            + (1 if isinstance(record.get("stage0_progress_start"), dict) else 0)
            for record in repair_trace_records
        )
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
        admission_metrics = summarize_admission_metrics(
            context=context,
            selected_metrics=per_query_selected_metrics,
            kappa=args.admission_kappa,
            acceptance_threshold=args.admission_threshold,
        )
        admit_for_training = bool(admission_metrics["admit_for_training"])
        if args.admission_policy == "all_executed_nonerror":
            admit_for_training = has_trace and not has_error and not safety_abort
        elif args.admission_policy in {"success_only", "caver_success_preserving"}:
            admit_for_training = has_trace and not has_error and not safety_abort and bool(context.get("success"))
        elif args.admission_policy in {
            "caver_top_m_success",
            "caver_family_balanced_success",
            "caver_failure_diagnostic_success",
            "caver_hard_family_rescue",
            "caver_family_segment_repair",
        }:
            admit_for_training = False

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
        elif args.admission_policy in {"success_only", "caver_success_preserving"}:
            if not has_trace:
                admission_reason = "missing_trace_records"
            elif has_error:
                admission_reason = "context_error"
            elif safety_abort:
                admission_reason = "safety_abort"
            elif context.get("success"):
                admission_reason = (
                    "admit_success_preserving"
                    if args.admission_policy == "caver_success_preserving"
                    else "admit_success"
                )
            else:
                admission_reason = "failed_execution"
            admission_confidence = 1.0 if admit_for_training else 0.0
        elif args.admission_policy in {
            "caver_top_m_success",
            "caver_family_balanced_success",
            "caver_failure_diagnostic_success",
            "caver_hard_family_rescue",
            "caver_family_segment_repair",
        }:
            if not has_trace:
                admission_reason = "missing_trace_records"
            elif has_error:
                admission_reason = "context_error"
            elif safety_abort:
                admission_reason = "safety_abort"
            elif context.get("success"):
                admission_reason = "pending_global_success_selection"
            else:
                if args.admission_policy == "caver_hard_family_rescue":
                    admission_reason = "pending_hard_family_rescue_selection"
                elif args.admission_policy == "caver_family_segment_repair":
                    admission_reason = "pending_hard_family_segment_repair_selection"
                elif args.admission_policy == "caver_failure_diagnostic_success":
                    admission_reason = "pending_failure_diagnostic_selection"
                else:
                    admission_reason = "failed_execution"
            admission_confidence = 0.0
        else:
            admission_reason = str(admission_metrics["admission_reason"])
            admission_confidence = float(admission_metrics["admission_confidence"])

        admission_context_record = {
            "implementation_phase": (
                "caver_scaffold_v1"
                if is_scaffold_admission_policy(args.admission_policy)
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
            "failure_diagnostic_for_calibration": False,
            "segment_repair_for_training": False,
            "repair_progress_trace_available": repair_progress_point_count > 0,
            "repair_progress_point_count": repair_progress_point_count,
            "repair_trace_record_start": None,
            "repair_trace_record_end": None,
            "repair_trace_records": 0,
            "repair_score": None,
            "repair_policy": None,
            "repair_start_step": None,
            "repair_end_step": None,
            "repair_progress_gain": None,
            "repair_family_id": None,
            "repair_progress_source": None,
            "full_failed_trace_admitted": False,
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

    apply_global_admission_policy(
        admission_context_records=admission_context_records,
        repair_trace_records_by_context=repair_trace_records_by_context,
        admission_policy=args.admission_policy,
        top_m_success_count=args.top_m_success_count,
        family_min_success_count=args.family_min_success_count,
        rescue_family_ids=parse_family_ids(args.rescue_family_ids),
        rescue_per_family_count=args.rescue_per_family_count,
        repair_min_trace_records=args.repair_min_trace_records,
        repair_max_trace_records=args.repair_max_trace_records,
        repair_min_progress=args.repair_min_progress,
        repair_min_primitive_steps=args.repair_min_primitive_steps,
        repair_max_regression=args.repair_max_regression,
    )
    (
        admitted_context_ids,
        admitted_family_counts,
        admitted_partition_counts,
        admitted_trace_records,
    ) = recompute_admitted_counts(admission_context_records)
    failure_diagnostic_records = [
        record
        for record in admission_context_records
        if bool(record.get("failure_diagnostic_for_calibration", False))
    ]
    failure_diagnostic_family_counts = collections.Counter(
        str(record.get("proxy_family_id") or "unknown") for record in failure_diagnostic_records
    )
    segment_repair_records = [
        record
        for record in admission_context_records
        if bool(record.get("segment_repair_for_training", False))
    ]
    segment_repair_family_counts = collections.Counter(
        str(record.get("proxy_family_id") or "unknown") for record in segment_repair_records
    )
    segment_repair_trace_records = sum(
        int(record.get("repair_trace_records") or 0) for record in segment_repair_records
    )
    full_failed_contexts_admitted = [
        record["context_id"]
        for record in admission_context_records
        if bool(record.get("admit_for_training", False))
        and not bool(record.get("success", False))
        and not bool(record.get("segment_repair_for_training", False))
    ]

    if not selector_mode_counts:
        resolved_selector_mode = args.selector_mode
    elif len(selector_mode_counts) == 1:
        resolved_selector_mode = next(iter(selector_mode_counts))
    else:
        resolved_selector_mode = "mixed"

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
        "demo_trace_path": demo_trace_path,
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

    admission_summary = {
        "implementation_phase": (
            "caver_scaffold_v1"
            if is_scaffold_admission_policy(args.admission_policy)
            else CAVER_ADMISSION_IMPLEMENTATION_PHASE
        ),
        "admission_policy": args.admission_policy,
        "admission_kappa": args.admission_kappa,
        "admission_threshold": args.admission_threshold,
        "top_m_success_count": args.top_m_success_count,
        "family_min_success_count": args.family_min_success_count,
        "rescue_family_ids": sorted(parse_family_ids(args.rescue_family_ids)),
        "rescue_per_family_count": args.rescue_per_family_count,
        "repair_min_trace_records": args.repair_min_trace_records,
        "repair_max_trace_records": args.repair_max_trace_records,
        "repair_min_progress": args.repair_min_progress,
        "repair_min_primitive_steps": args.repair_min_primitive_steps,
        "repair_max_regression": args.repair_max_regression,
        "online_results_path": online_results_path,
        "trace_path": trace_path,
        "demo_trace_path": demo_trace_path,
        "admission_context_path": admission_context_path,
        "admitted_trace_path": admitted_trace_path,
        "contexts_total": len(context_records),
        "contexts_admitted": len(admitted_context_ids),
        "contexts_rejected": len(context_records) - len(admitted_context_ids),
        "admitted_trace_records": admitted_trace_records,
        "admitted_trace_write_skipped": bool(args.skip_admitted_trace_write),
        "written_admitted_trace_records": None,
        "admitted_trace_write_status": (
            "skipped" if args.skip_admitted_trace_write else "pending"
        ),
        "rejected_context_ids": sorted(
            record["context_id"] for record in admission_context_records if not record["admit_for_training"]
        ),
        "failure_diagnostic_context_ids": sorted(
            record["context_id"] for record in failure_diagnostic_records
        ),
        "failure_diagnostic_family_counts": dict(failure_diagnostic_family_counts),
        "segment_repair_context_ids": sorted(
            record["context_id"] for record in segment_repair_records
        ),
        "segment_repair_family_counts": dict(segment_repair_family_counts),
        "segment_repair_trace_records": int(segment_repair_trace_records),
        "segment_repair_progress_gains": {
            record["context_id"]: record.get("repair_progress_gain")
            for record in segment_repair_records
        },
        "full_failed_context_ids_admitted": sorted(full_failed_contexts_admitted),
        "full_failed_contexts_admitted": len(full_failed_contexts_admitted),
        "admitted_family_counts": dict(admitted_family_counts),
        "admitted_partition_counts": dict(admitted_partition_counts),
        "lcb_acceptance_counts": dict(
            collections.Counter(record["admission_reason"] for record in admission_context_records)
        ),
    }
    # Write summaries before materializing the full observation-bearing trace. Some
    # rescued near-miss trajectories are large, and this preserves diagnostics if a
    # Slurm walltime limit interrupts the slow trace write.
    write_json(selector_summary_path, selector_summary)
    write_json(admission_summary_path, admission_summary)

    if args.skip_admitted_trace_write:
        written_admitted_trace_records = None
    else:
        written_admitted_trace_records = write_admitted_trace_subset(
            trace_path=demo_trace_path,
            admitted_trace_path=admitted_trace_path,
            admission_context_records=admission_context_records,
        )
        if written_admitted_trace_records != admitted_trace_records:
            admission_summary["written_admitted_trace_records"] = written_admitted_trace_records
            admission_summary["admitted_trace_write_status"] = "mismatch"
            write_json(admission_summary_path, admission_summary)
            raise ValueError(
                "admitted trace record count mismatch between summary pass "
                f"({admitted_trace_records}) and trace subset write ({written_admitted_trace_records})"
            )
        admission_summary["written_admitted_trace_records"] = written_admitted_trace_records
        admission_summary["admitted_trace_write_status"] = "complete"
        write_json(admission_summary_path, admission_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
