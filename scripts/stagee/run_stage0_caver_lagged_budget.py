#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from finalize_stagee_budget200_cell import build_online_results
from finalize_stagee_budget200_cell import load_manifest_selection
from finalize_stagee_budget200_cell import load_value_proxy_metadata
from finalize_stagee_budget200_cell import resolve_path
from finalize_stagee_budget200_cell import write_json
from finalize_stagee_budget200_cell import write_jsonl


FLAG_OPTIONS = {
    "--count-legacy-contexts-as-online-budget",
    "--no-count-legacy-contexts-as-online-budget",
    "--exact-rollout-payload",
    "--exact-no-nft-loss",
    "--exact-add-value-head",
    "--exact-value-after-vlm",
    "--no-require-candidate-bank",
    "--skip-online",
    "--skip-backend-train",
    "--dry-run",
}
TRACE_INPUT_FORMAT = "stagee_trace_source_manifest_v1"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage-E CAVER in explicit lagged rounds: each subround collects online data, "
            "fits the next DR calibrator, and the merged results are finalized once with --skip-online."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--trace-reference-mode",
        default="manifest",
        choices=("manifest", "materialize"),
        help="manifest writes a lightweight trace-source manifest; materialize concatenates all round traces.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned subround and finalizer commands without running them.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse completed lagged subrounds already present under --results-dir instead of "
            "requiring an empty destination directory."
        ),
    )
    parser.add_argument(
        "--cleanup-incomplete-rounds",
        action="store_true",
        help=(
            "With --resume-existing, delete round directories that exist but do not contain a "
            "completed caver_round_summary.json before rerunning that round."
        ),
    )
    parsed, remaining = parser.parse_known_args(argv)
    return parsed, remaining


def parse_runner_options(tokens: list[str]) -> collections.OrderedDict[str, Any]:
    options: collections.OrderedDict[str, Any] = collections.OrderedDict()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-h", "--help"}:
            raise ValueError("pass --help to this wrapper only; subrunner help passthrough is not supported here")
        if token in FLAG_OPTIONS:
            options[token] = True
            index += 1
            continue
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional token: {token}")
        if index + 1 >= len(tokens):
            raise ValueError(f"missing value for option: {token}")
        options[token] = tokens[index + 1]
        index += 2
    return options


def remove_option(
    options: collections.OrderedDict[str, Any],
    key: str,
) -> collections.OrderedDict[str, Any]:
    updated: collections.OrderedDict[str, Any] = collections.OrderedDict()
    for existing_key, value in options.items():
        if existing_key == key:
            continue
        updated[existing_key] = value
    return updated


def set_option(
    options: collections.OrderedDict[str, Any],
    key: str,
    value: Any,
) -> collections.OrderedDict[str, Any]:
    updated = remove_option(options, key)
    updated[key] = value
    return updated


def options_to_tokens(options: collections.OrderedDict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in options.items():
        tokens.append(key)
        if value is not True:
            tokens.append(str(value))
    return tokens


def ensure_empty_or_create(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"directory already exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reindex_context_budgets(contexts: list[dict[str, Any]], *, round_size: int) -> list[dict[str, Any]]:
    partition_counters: collections.Counter[str] = collections.Counter()
    reindexed: list[dict[str, Any]] = []
    for context in contexts:
        updated_context = dict(context)
        budget = dict(updated_context.get("budget") or {})
        partition_label = str(updated_context.get("partition_name") or "legacy")
        partition_counters[partition_label] += 1
        partition_context_index = partition_counters[partition_label]
        counts_against_online_budget = bool(budget.get("counts_against_online_budget", partition_label == "T_train_S0"))
        online_budget_index = partition_context_index if counts_against_online_budget else None
        round_index = ((partition_context_index - 1) // round_size + 1) if counts_against_online_budget else 0
        round_context_index = ((partition_context_index - 1) % round_size + 1) if counts_against_online_budget else None
        budget.update(
            {
                "partition_context_index": partition_context_index,
                "online_budget_index": online_budget_index,
                "round_index": round_index,
                "round_context_index": round_context_index,
                "round_size": round_size,
            }
        )
        updated_context["budget"] = budget
        reindexed.append(updated_context)
    return reindexed


def write_trace_source_manifest_from_paths(*, manifest_path: Path, trace_paths: list[Path]) -> None:
    payload = {
        "trace_input_format": TRACE_INPUT_FORMAT,
        "sources": [{"path": str(path.resolve())} for path in trace_paths],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def append_trace(dst: Path, src: Path) -> int:
    written = 0
    with dst.open("a", encoding="utf-8") as out_handle, src.open("r", encoding="utf-8") as in_handle:
        for line_number, raw_line in enumerate(in_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to decode {src} line {line_number}") from exc
            out_handle.write(line)
            out_handle.write("\n")
            written += 1
    return written


def summarize_round(round_results_dir: Path) -> dict[str, Any]:
    online_path = round_results_dir / "caver_online_eval.json"
    summary_path = round_results_dir / "caver_round_summary.json"
    calibrator_path = round_results_dir / "caver_lagged_dr_calibrator.json"
    calibrator_summary_path = round_results_dir / "caver_lagged_dr_calibrator.summary.json"
    if not online_path.exists():
        raise ValueError(f"missing round online results: {online_path}")
    if not summary_path.exists():
        raise ValueError(f"missing round summary: {summary_path}")
    online = json.loads(online_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibrator_summary = (
        json.loads(calibrator_summary_path.read_text(encoding="utf-8"))
        if calibrator_summary_path.exists()
        else None
    )
    return {
        "results_dir": str(round_results_dir),
        "online": online,
        "summary": summary,
        "calibrator_path": (str(calibrator_path) if calibrator_path.exists() else None),
        "calibrator_summary_path": (
            str(calibrator_summary_path) if calibrator_summary_path.exists() else None
        ),
        "calibrator_summary": calibrator_summary,
    }


def main(argv: list[str]) -> int:
    wrapper_args, runner_tokens = parse_args(argv)
    runner_options = parse_runner_options(runner_tokens)
    if "--skip-online" in runner_options:
        raise SystemExit("error: do not pass --skip-online to the lagged wrapper")
    if "--skip-backend-train" in runner_options:
        raise SystemExit("error: do not pass --skip-backend-train to the lagged wrapper")
    if "--results-dir" not in runner_options:
        raise SystemExit("error: --results-dir is required")
    if "--manifest-path" not in runner_options:
        raise SystemExit("error: --manifest-path is required for lagged Stage-E execution")
    if "--partition-name" not in runner_options:
        raise SystemExit("error: --partition-name is required for lagged Stage-E execution")
    if "--max-contexts" not in runner_options:
        raise SystemExit("error: --max-contexts is required and defines the total budget for the lagged run")

    workdir = Path.cwd().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "scripts" / "stagee" / "run_stage0_caver_round.sh"

    total_contexts = int(runner_options["--max-contexts"])
    round_size = int(runner_options.get("--round-size", "25"))
    context_offset = int(runner_options.get("--context-offset", "0"))
    seed = int(runner_options.get("--seed", "7"))
    if total_contexts < 1:
        raise SystemExit("error: --max-contexts must be positive")
    if round_size < 1:
        raise SystemExit("error: --round-size must be positive")

    overall_results_dir = resolve_path(workdir, runner_options["--results-dir"])
    assert overall_results_dir is not None
    lagged_round_root = overall_results_dir / "lagged_rounds"

    if not wrapper_args.dry_run:
        if wrapper_args.resume_existing:
            ensure_directory(overall_results_dir)
            ensure_directory(lagged_round_root)
        else:
            ensure_empty_or_create(overall_results_dir)
            ensure_empty_or_create(lagged_round_root)

    manifest_path = resolve_path(workdir, runner_options.get("--manifest-path"))
    value_proxy_model_path = resolve_path(workdir, runner_options.get("--value-proxy-model-path"))
    initial_dr_calibrator_model_path = resolve_path(workdir, runner_options.get("--dr-calibrator-model-path"))
    value_proxy_model_path_str, value_proxy_model_id = load_value_proxy_metadata(value_proxy_model_path)
    (
        initial_dr_calibrator_model_path_str,
        initial_dr_calibrator_model_id,
    ) = load_value_proxy_metadata(initial_dr_calibrator_model_path)

    lagged_round_reports: list[dict[str, Any]] = []
    round_trace_paths: list[Path] = []
    merged_contexts: list[dict[str, Any]] = []
    previous_calibrator_path = initial_dr_calibrator_model_path
    round_count = (total_contexts + round_size - 1) // round_size

    base_subround_options = remove_option(runner_options, "--dr-calibrator-model-path")
    for round_zero_index in range(round_count):
        round_index = round_zero_index + 1
        round_context_offset = context_offset + (round_zero_index * round_size)
        round_context_count = min(round_size, total_contexts - (round_zero_index * round_size))
        round_results_dir = lagged_round_root / f"round_{round_index:03d}"

        subround_options = collections.OrderedDict(base_subround_options)
        subround_options = set_option(subround_options, "--results-dir", str(round_results_dir))
        subround_options = set_option(subround_options, "--context-offset", str(round_context_offset))
        subround_options = set_option(subround_options, "--max-contexts", str(round_context_count))
        subround_options["--skip-backend-train"] = True
        if previous_calibrator_path is not None:
            subround_options = set_option(
                subround_options,
                "--dr-calibrator-model-path",
                str(previous_calibrator_path),
            )

        subround_cmd = [str(runner_path), *options_to_tokens(subround_options)]
        log(
            f"[stagee-lagged] round {round_index}/{round_count} "
            f"offset={round_context_offset} count={round_context_count}"
        )
        log(f"[stagee-lagged] subround command: {' '.join(shlex.quote(part) for part in subround_cmd)}")
        if not wrapper_args.dry_run:
            round_summary_path = round_results_dir / "caver_round_summary.json"
            if round_summary_path.exists():
                log(f"[stagee-lagged] reusing completed round at {round_results_dir}")
                round_report = summarize_round(round_results_dir)
                round_report["round_index"] = round_index
                round_report["context_offset"] = round_context_offset
                round_report["context_count"] = round_context_count
                round_report["input_dr_calibrator_path"] = (
                    None if previous_calibrator_path is None else str(previous_calibrator_path)
                )
                next_calibrator_path = round_results_dir / "caver_lagged_dr_calibrator.json"
                if next_calibrator_path.exists():
                    previous_calibrator_path = next_calibrator_path.resolve()
                    round_report["output_dr_calibrator_path"] = str(previous_calibrator_path)
                else:
                    round_report["output_dr_calibrator_path"] = None
                lagged_round_reports.append(round_report)
                round_trace_paths.append(round_results_dir / "caver_online_chunks.jsonl")
                merged_contexts.extend(list(round_report["online"]["contexts"]))
                continue

            if round_results_dir.exists():
                if not round_results_dir.is_dir():
                    raise ValueError(f"path exists and is not a directory: {round_results_dir}")
                if any(round_results_dir.iterdir()):
                    if wrapper_args.resume_existing and wrapper_args.cleanup_incomplete_rounds:
                        log(f"[stagee-lagged] removing incomplete round directory {round_results_dir}")
                        shutil.rmtree(round_results_dir)
                    elif wrapper_args.resume_existing:
                        raise ValueError(
                            f"incomplete round directory exists: {round_results_dir}; "
                            "rerun with --cleanup-incomplete-rounds to replace it"
                        )
            ensure_empty_or_create(round_results_dir)
            subprocess.run(subround_cmd, check=True, cwd=str(repo_root))
            round_report = summarize_round(round_results_dir)
            round_report["round_index"] = round_index
            round_report["context_offset"] = round_context_offset
            round_report["context_count"] = round_context_count
            round_report["input_dr_calibrator_path"] = (
                None if previous_calibrator_path is None else str(previous_calibrator_path)
            )
            next_calibrator_path = round_results_dir / "caver_lagged_dr_calibrator.json"
            if next_calibrator_path.exists():
                previous_calibrator_path = next_calibrator_path.resolve()
                round_report["output_dr_calibrator_path"] = str(previous_calibrator_path)
            else:
                round_report["output_dr_calibrator_path"] = None
            lagged_round_reports.append(round_report)
            round_trace_paths.append(round_results_dir / "caver_online_chunks.jsonl")
            merged_contexts.extend(list(round_report["online"]["contexts"]))

    merged_online_path = overall_results_dir / "caver_online_eval.json"
    merged_context_path = overall_results_dir / "caver_online_contexts.jsonl"
    merged_trace_path = overall_results_dir / "caver_online_chunks.jsonl"
    merge_report_path = overall_results_dir / "lagged_round_chain.summary.json"

    if not wrapper_args.dry_run:
        reindexed_contexts = reindex_context_budgets(merged_contexts, round_size=round_size)
        selection = load_manifest_selection(manifest_path, runner_options)
        config_template = {
            "num_trials_per_task": int(runner_options.get("--num-trials-per-task", "50")),
            "count_legacy_contexts_as_online_budget": bool(
                runner_options.get("--count-legacy-contexts-as-online-budget", False)
            ),
            "candidate_count": int(runner_options.get("--candidate-count", "1")),
            "selection_policy": str(runner_options.get("--selection-policy", "first")),
            "selector_seed": int(runner_options.get("--selector-seed") or runner_options.get("--seed", "7")),
            "value_proxy_model_path": value_proxy_model_path_str,
            "value_proxy_model_id": value_proxy_model_id,
            "dr_calibrator_model_path": initial_dr_calibrator_model_path_str,
            "dr_calibrator_model_id": initial_dr_calibrator_model_id,
            "num_steps_wait": int(runner_options.get("--num-steps-wait", "10")),
            "replan_steps": int(runner_options.get("--replan-steps", "4")),
            "resize_size": int(runner_options.get("--resize-size", "224")),
            "resolution": int(runner_options.get("--resolution", "256")),
            "max_steps_override": (
                None if runner_options.get("--max-env-steps") is None else int(runner_options["--max-env-steps"])
            ),
            "video_dir": None,
            "save_failures_only": False,
            "lagged_round_driver": "stage0_caver_lagged_budget_v1",
            "lagged_round_count": round_count,
            "lagged_round_results_root": str(lagged_round_root),
            "lagged_round_result_dirs": [record["results_dir"] for record in lagged_round_reports],
            "final_round_dr_calibrator_path": (
                None if previous_calibrator_path is None else str(previous_calibrator_path)
            ),
        }
        merged_online = build_online_results(
            contexts=reindexed_contexts,
            seed=seed,
            selection=selection,
            config_template=config_template,
            context_log_path=merged_context_path,
            trace_path=merged_trace_path,
        )
        if lagged_round_reports:
            merged_online["server"] = lagged_round_reports[0]["online"].get("server", merged_online["server"])
        write_json(merged_online_path, merged_online)
        write_jsonl(merged_context_path, reindexed_contexts)

        if wrapper_args.trace_reference_mode == "manifest":
            write_trace_source_manifest_from_paths(manifest_path=merged_trace_path, trace_paths=round_trace_paths)
        else:
            merged_trace_path.unlink(missing_ok=True)
            total_trace_records = 0
            for trace_path in round_trace_paths:
                total_trace_records += append_trace(merged_trace_path, trace_path)
            log(
                f"[stagee-lagged] materialized merged trace at {merged_trace_path} "
                f"(records={total_trace_records})"
            )

        merge_report = {
            "workflow": "stage0_caver_lagged_budget_v1",
            "overall_results_dir": str(overall_results_dir),
            "lagged_round_root": str(lagged_round_root),
            "trace_reference_mode": wrapper_args.trace_reference_mode,
            "seed": seed,
            "context_offset": context_offset,
            "total_contexts": total_contexts,
            "round_size": round_size,
            "round_count": round_count,
            "manifest_path": (None if manifest_path is None else str(manifest_path)),
            "partition_name": runner_options.get("--partition-name"),
            "value_proxy_model_path": value_proxy_model_path_str,
            "value_proxy_model_id": value_proxy_model_id,
            "initial_dr_calibrator_model_path": initial_dr_calibrator_model_path_str,
            "initial_dr_calibrator_model_id": initial_dr_calibrator_model_id,
            "final_round_dr_calibrator_path": (
                None if previous_calibrator_path is None else str(previous_calibrator_path)
            ),
            "rounds": [
                {
                    "round_index": record["round_index"],
                    "context_offset": record["context_offset"],
                    "context_count": record["context_count"],
                    "results_dir": record["results_dir"],
                    "input_dr_calibrator_path": record["input_dr_calibrator_path"],
                    "output_dr_calibrator_path": record["output_dr_calibrator_path"],
                    "online_successes": record["online"]["summary"]["successes"],
                    "online_episodes_run": record["online"]["summary"]["episodes_run"],
                    "contexts_admitted": record["summary"]["admission"]["contexts_admitted"],
                    "admitted_trace_records": record["summary"]["admission"]["admitted_trace_records"],
                    "selector_mode": record["summary"]["selector"]["selector_mode"],
                }
                for record in lagged_round_reports
            ],
        }
        write_json(merge_report_path, merge_report)

    finalizer_options = collections.OrderedDict(runner_options)
    finalizer_options = remove_option(finalizer_options, "--dr-calibrator-model-path")
    finalizer_options = remove_option(finalizer_options, "--skip-online")
    finalizer_options = remove_option(finalizer_options, "--skip-backend-train")
    finalizer_options = remove_option(finalizer_options, "--dry-run")
    finalizer_options = set_option(finalizer_options, "--results-dir", str(overall_results_dir))
    finalizer_options["--skip-online"] = True
    finalizer_cmd = [str(runner_path), *options_to_tokens(finalizer_options)]
    log(f"[stagee-lagged] finalizer command: {' '.join(shlex.quote(part) for part in finalizer_cmd)}")

    if wrapper_args.dry_run:
        dry_run_payload = {
            "workflow": "stage0_caver_lagged_budget_v1",
            "overall_results_dir": str(overall_results_dir),
            "round_count": round_count,
            "trace_reference_mode": wrapper_args.trace_reference_mode,
            "subround_commands": [],
            "finalizer_command": finalizer_cmd,
        }
        previous_dry_run_calibrator = (
            None if initial_dr_calibrator_model_path is None else str(initial_dr_calibrator_model_path)
        )
        for round_zero_index in range(round_count):
            round_index = round_zero_index + 1
            round_context_offset = context_offset + (round_zero_index * round_size)
            round_context_count = min(round_size, total_contexts - (round_zero_index * round_size))
            round_results_dir = lagged_round_root / f"round_{round_index:03d}"
            subround_options = collections.OrderedDict(base_subround_options)
            subround_options = set_option(subround_options, "--results-dir", str(round_results_dir))
            subround_options = set_option(subround_options, "--context-offset", str(round_context_offset))
            subround_options = set_option(subround_options, "--max-contexts", str(round_context_count))
            subround_options["--skip-backend-train"] = True
            if previous_dry_run_calibrator is not None:
                subround_options = set_option(
                    subround_options,
                    "--dr-calibrator-model-path",
                    previous_dry_run_calibrator,
                )
            dry_run_payload["subround_commands"].append(
                {
                    "round_index": round_index,
                    "context_offset": round_context_offset,
                    "context_count": round_context_count,
                    "command": [str(runner_path), *options_to_tokens(subround_options)],
                }
            )
            previous_dry_run_calibrator = str(round_results_dir / "caver_lagged_dr_calibrator.json")
        print(json.dumps(dry_run_payload, indent=2))
        return 0

    subprocess.run(finalizer_cmd, check=True, cwd=str(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
