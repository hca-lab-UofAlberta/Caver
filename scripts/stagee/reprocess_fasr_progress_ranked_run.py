#!/usr/bin/env python3
"""Rebuild Stage-E FASR final artifacts with progress-ranked repairs.

This is a post-hoc artifact repair for completed online runs. It does not rerun
LIBERO or GE-Sim; it reuses the recorded online/demo traces and writes a new
results directory with corrected selector/admission artifacts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--repo-root", default=pathlib.Path.cwd(), type=pathlib.Path)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--run-prefix", default="stagee__caver-lagged-reprocess-progress-ranked")
    parser.add_argument("--selector-mode", default="k1_guarded_verified_progress_fasr_v1")
    parser.add_argument("--admission-policy", default="caver_family_segment_repair")
    parser.add_argument(
        "--value-proxy-model-path",
        default="metadata/stage0/value_proxy/stage0_context_success_progress_sq_mlp3head_v2.json",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--dr-calibrator-model-path",
        default="metadata/stage0/calibrator/stage0_seed_dr_calibrator_mlp_v2.json",
        type=pathlib.Path,
    )
    parser.add_argument("--family-min-success-count", default=2, type=int)
    parser.add_argument(
        "--rescue-family-ids",
        default="drawer_open_proxy,relocate_to_region_proxy,two_object_stack_proxy",
    )
    parser.add_argument("--rescue-per-family-count", default=2, type=int)
    parser.add_argument("--repair-min-trace-records", default=1, type=int)
    parser.add_argument("--repair-max-trace-records", default=12, type=int)
    parser.add_argument("--repair-min-progress", default=0.03, type=float)
    parser.add_argument("--repair-min-primitive-steps", default=4, type=int)
    parser.add_argument("--repair-max-regression", default=0.10, type=float)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def infer_seed_budget(source_run_dir: pathlib.Path) -> tuple[str, str]:
    parts = source_run_dir.name.split("__")
    seed = "unknown"
    budget = "unknown"
    for part in parts:
        if part.startswith("seed"):
            seed = part.removeprefix("seed")
        elif part.startswith("budget"):
            budget = part.removeprefix("budget")
    return seed, budget


def resolve_model_path(repo_root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_run_dir = args.source_run_dir.resolve()
    source_results = source_run_dir / "results"
    output_root = args.output_root.resolve()
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed, budget = infer_seed_budget(source_run_dir)
    run_dir = (
        output_root
        / f"{args.run_prefix}__verified-progress-fasr-n{budget}__seed{seed}__budget{budget}__{stamp}"
    )
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    required = [
        source_results / "caver_online_eval.json",
        source_results / "caver_online_chunks.jsonl",
        source_results / "caver_online_demo_chunks.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")

    value_proxy_model_path = resolve_model_path(repo_root, args.value_proxy_model_path)
    dr_calibrator_model_path = resolve_model_path(repo_root, args.dr_calibrator_model_path)

    command = [
        "python3",
        str(repo_root / "scripts/stagee/build_caver_round_artifacts.py"),
        "--online-results",
        str(source_results / "caver_online_eval.json"),
        "--trace-path",
        str(source_results / "caver_online_chunks.jsonl"),
        "--demo-trace-path",
        str(source_results / "caver_online_demo_chunks.jsonl"),
        "--selector-context-path",
        str(results_dir / "caver_selector_contexts.jsonl"),
        "--selector-summary-path",
        str(results_dir / "caver_selector_summary.json"),
        "--admission-context-path",
        str(results_dir / "caver_admission_contexts.jsonl"),
        "--admission-summary-path",
        str(results_dir / "caver_admission_summary.json"),
        "--admitted-trace-path",
        str(results_dir / "caver_admitted_chunks.jsonl.gz"),
        "--selector-mode",
        args.selector_mode,
        "--admission-policy",
        args.admission_policy,
        "--value-proxy-model-path",
        str(value_proxy_model_path),
        "--dr-calibrator-model-path",
        str(dr_calibrator_model_path),
        "--family-min-success-count",
        str(args.family_min_success_count),
        "--rescue-family-ids",
        args.rescue_family_ids,
        "--rescue-per-family-count",
        str(args.rescue_per_family_count),
        "--repair-min-trace-records",
        str(args.repair_min_trace_records),
        "--repair-max-trace-records",
        str(args.repair_max_trace_records),
        "--repair-min-progress",
        str(args.repair_min_progress),
        "--repair-min-primitive-steps",
        str(args.repair_min_primitive_steps),
        "--repair-max-regression",
        str(args.repair_max_regression),
    ]
    subprocess.run(command, cwd=str(repo_root), check=True)

    for name in [
        "caver_online_eval.json",
        "caver_online_contexts.jsonl",
        "caver_online_chunks.jsonl",
        "caver_online_demo_chunks.jsonl",
        "lagged_round_chain.summary.json",
    ]:
        source_path = source_results / name
        if source_path.exists():
            shutil.copy2(source_path, results_dir / name)

    original_summary_path = source_results / "caver_round_summary.json"
    summary = (
        load_json(original_summary_path)
        if original_summary_path.exists()
        else {"workflow": "stage0_caver_round_v1"}
    )
    admission = load_json(results_dir / "caver_admission_summary.json")
    selector = load_json(results_dir / "caver_selector_summary.json")
    online = load_json(source_results / "caver_online_eval.json")
    summary["online"] = online.get("summary", summary.get("online", {}))
    summary["selector"] = {
        key: selector.get(key)
        for key in [
            "implementation_phase",
            "selector_mode",
            "contexts_total",
            "contexts_with_candidate_bank",
            "policy_queries_total",
        ]
        if key in selector
    }
    summary["admission"] = {
        key: admission.get(key)
        for key in [
            "implementation_phase",
            "admission_policy",
            "contexts_admitted",
            "contexts_rejected",
            "admitted_trace_records",
            "segment_repair_trace_records",
            "segment_repair_family_counts",
            "full_failed_contexts_admitted",
        ]
        if key in admission
    }
    summary["admission_summary_path"] = str(results_dir / "caver_admission_summary.json")
    summary["selector_summary_path"] = str(results_dir / "caver_selector_summary.json")
    summary["reprocess"] = {
        "source_run_dir": str(source_run_dir),
        "created_utc": stamp,
        "patch": "segment_repair_candidates_ranked_by_verified_progress_gain_first",
    }
    write_json(results_dir / "caver_round_summary.json", summary)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "results_dir": str(results_dir),
                "seed": seed,
                "budget": budget,
                "contexts_admitted": admission.get("contexts_admitted"),
                "admitted_trace_records": admission.get("admitted_trace_records"),
                "segment_repair_trace_records": admission.get("segment_repair_trace_records"),
                "segment_repair_family_counts": admission.get("segment_repair_family_counts"),
                "full_failed_contexts_admitted": admission.get("full_failed_contexts_admitted"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
