#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

FLAG_OPTIONS = {
    "--count-legacy-contexts-as-online-budget",
    "--no-count-legacy-contexts-as-online-budget",
    "--no-require-candidate-bank",
    "--skip-online",
    "--dry-run",
}
TRACE_INPUT_FORMAT = "stagee_trace_source_manifest_v1"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a failed Stage-E budget200 prefix with a completed continuation "
            "tail and finalize with --skip-online."
        )
    )
    parser.add_argument("--original-job-script", required=True)
    parser.add_argument("--original-trace-path", required=True)
    parser.add_argument("--completed-prefix", required=True, type=int)
    parser.add_argument("--tail-results-dir", required=True)
    parser.add_argument("--merged-results-dir", required=True)
    parser.add_argument("--runtime-log-root", default="/rdss/p57098/euijin1/caver/runtime_logs")
    parser.add_argument(
        "--trace-reference-mode",
        default="manifest",
        choices=("manifest", "materialize"),
        help="manifest writes a lightweight trace-source manifest; materialize copies a merged JSONL trace.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def write_compact_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True, default=json_default)
            handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to decode {path} line {line_number}") from exc
    return records


def parse_job_script(job_script: Path) -> tuple[Path, Path, dict[str, Any]]:
    workdir: Path | None = None
    command_line: str | None = None
    with job_script.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("cd "):
                parts = shlex.split(line)
                if len(parts) == 2:
                    workdir = Path(parts[1]).resolve()
                continue
            if "run_stage0_real_only_round.sh" in line or "run_stage0_caver_round.sh" in line:
                command_line = line
    if workdir is None or command_line is None:
        raise ValueError(f"failed to parse job script: {job_script}")

    tokens = shlex.split(command_line)
    runner_path = Path(tokens[0]).resolve()
    options: dict[str, Any] = collections.OrderedDict()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in FLAG_OPTIONS:
            options[token] = True
            index += 1
            continue
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional token in job command: {token}")
        if index + 1 >= len(tokens):
            raise ValueError(f"missing value for option in job command: {token}")
        options[token] = tokens[index + 1]
        index += 2
    return workdir, runner_path, options


def resolve_path(workdir: Path, raw: str | None) -> Path | None:
    if raw is None or raw == "":
        return None
    path = Path(raw)
    return (workdir / path).resolve() if not path.is_absolute() else path.resolve()


def parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def load_value_proxy_metadata(model_path: Path | None) -> tuple[str | None, str | None]:
    if model_path is None or not model_path.is_file():
        return None, None
    try:
        with model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return str(model_path), None
    return str(model_path), payload.get("model_id")


def load_manifest_selection(manifest_path: Path | None, options: dict[str, Any]) -> dict[str, Any]:
    selection = {
        "mode": "legacy",
        "context_offset": int(options.get("--context-offset", "0")),
        "max_contexts": parse_optional_int(options.get("--max-contexts")),
        "round_size": int(options.get("--round-size", "25")),
        "requested_family_ids": [part for part in str(options.get("--family-ids", "")).split(",") if part],
        "selected_family_ids": [],
        "partition_name": options.get("--partition-name"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_global_partition_counts": {},
    }
    if manifest_path is None or not manifest_path.is_file():
        selection["task_suite_name"] = options.get("--task-suite")
        selection["task_ids"] = [int(part) for part in str(options.get("--task-ids", "")).split(",") if part]
        selection["num_trials_per_task"] = int(options.get("--num-trials-per-task", "50"))
        return selection

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_selection = manifest.get("selection", {})
    selection["mode"] = "manifest"
    selection["manifest_global_partition_counts"] = dict(manifest.get("global_partition_counts", {}))
    if not selection["selected_family_ids"]:
        selection["selected_family_ids"] = list(manifest_selection.get("selected_family_ids", []))
    return selection


def build_contexts_from_prefix_trace(
    *,
    original_trace_path: Path,
    merged_trace_path: Path | None,
    completed_prefix: int,
    candidate_count: int,
    selection_policy: str,
    selector_seed: int,
) -> tuple[list[dict[str, Any]], int]:
    states: dict[str, dict[str, Any]] = collections.OrderedDict()
    total_trace_records = 0
    if merged_trace_path is not None:
        ensure_parent(merged_trace_path)
    with original_trace_path.open("r", encoding="utf-8") as source:
        target = (
            merged_trace_path.open("w", encoding="utf-8")
            if merged_trace_path is not None
            else None
        )
        try:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context_id = str(record["context_id"])
                if context_id not in states:
                    if len(states) >= completed_prefix:
                        break
                    states[context_id] = {
                        "context_id": context_id,
                        "source_mode": record.get("source_mode"),
                        "suite_name": record.get("suite_name"),
                        "partition_name": record.get("partition_name"),
                        "proxy_family_id": record.get("proxy_family_id"),
                        "proposal_task": record.get("proposal_task"),
                        "task_id": record.get("task_id"),
                        "task_name": record.get("task_name"),
                        "episode_idx": record.get("episode_idx"),
                        "init_state_index": record.get("init_state_index"),
                        "prompt": record.get("prompt"),
                        "budget": dict(record.get("budget") or {}),
                        "success": False,
                        "policy_steps": 0,
                        "error": None,
                        "chunk_traces_written": 0,
                        "selector": {
                            "candidate_count": candidate_count,
                            "selection_policy": selection_policy,
                            "selector_seed": selector_seed,
                            "policy_queries": 0,
                            "selected_candidate_indices": [],
                            "selected_candidate_probabilities": [],
                            "candidate_probability_vectors": [],
                            "safe_candidate_counts": [],
                        },
                    }
                    if len(states) == 1 or len(states) % 10 == 0 or len(states) == completed_prefix:
                        log(
                            f"[stagee-finalize] recovered prefix context {len(states)}/{completed_prefix} "
                            f"from {original_trace_path}"
                        )
                state = states[context_id]
                if target is not None:
                    target.write(line)
                    target.write("\n")
                total_trace_records += 1
                selector = record.get("selector", {})
                state["policy_steps"] += int(record.get("steps_executed") or 0)
                state["chunk_traces_written"] += 1
                state["selector"]["policy_queries"] += 1
                state["selector"]["selected_candidate_indices"].append(selector.get("selected_candidate_index"))
                state["selector"]["selected_candidate_probabilities"].append(
                    selector.get("selected_candidate_probability")
                )
                state["selector"]["candidate_probability_vectors"].append(selector.get("candidate_probabilities"))
                state["selector"]["safe_candidate_counts"].append(selector.get("safe_candidate_count"))
                success_once = record.get("success_once") or []
                if any(bool(value) for value in success_once) or record.get("completed_reason") == "terminated":
                    state["success"] = True
                if record.get("error") is not None:
                    state["error"] = str(record.get("error"))
        finally:
            if target is not None:
                target.close()
    if len(states) != completed_prefix:
        raise ValueError(
            f"expected {completed_prefix} completed contexts but recovered {len(states)} from {original_trace_path}"
        )

    contexts: list[dict[str, Any]] = []
    for state in states.values():
        contexts.append(
            {
                "context_id": state["context_id"],
                "source_mode": state["source_mode"],
                "suite_name": state["suite_name"],
                "partition_name": state["partition_name"],
                "proxy_family_id": state["proxy_family_id"],
                "proposal_task": state["proposal_task"],
                "task_id": state["task_id"],
                "task_name": state["task_name"],
                "episode_idx": state["episode_idx"],
                "init_state_index": state["init_state_index"],
                "success": state["success"],
                "policy_steps": state["policy_steps"],
                "duration_sec": None,
                "video_path": None,
                "error": state["error"],
                "chunk_traces_written": state["chunk_traces_written"],
                "budget": state["budget"],
                "selector": state["selector"],
            }
        )
    return contexts, total_trace_records


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


def write_trace_input_manifest(
    *,
    manifest_path: Path,
    original_trace_path: Path,
    completed_prefix: int,
    tail_trace_path: Path,
) -> None:
    payload = {
        "trace_input_format": TRACE_INPUT_FORMAT,
        "sources": [
            {
                "path": str(original_trace_path),
                "completed_prefix_contexts": int(completed_prefix),
            },
            {
                "path": str(tail_trace_path),
            },
        ],
    }
    write_compact_json(manifest_path, payload)


def build_task_summaries(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], dict[str, Any]] = collections.OrderedDict()
    for context in contexts:
        key = (context.get("suite_name"), context.get("task_id"))
        task_state = grouped.setdefault(
            key,
            {
                "suite_name": context.get("suite_name"),
                "task_id": context.get("task_id"),
                "task_name": context.get("task_name"),
                "task_description": None,
                "available_initial_states": None,
                "episodes": [],
            },
        )
        task_state["episodes"].append(context)

    tasks = []
    for task_state in grouped.values():
        successes = sum(int(bool(context["success"])) for context in task_state["episodes"])
        tasks.append(
            {
                "suite_name": task_state["suite_name"],
                "task_id": task_state["task_id"],
                "task_name": task_state["task_name"],
                "task_description": task_state["task_description"],
                "available_initial_states": task_state["available_initial_states"],
                "episodes": task_state["episodes"],
                "episodes_run": len(task_state["episodes"]),
                "successes": successes,
                "success_rate": successes / len(task_state["episodes"]) if task_state["episodes"] else 0.0,
            }
        )
    return tasks


def build_online_results(
    *,
    contexts: list[dict[str, Any]],
    seed: int,
    selection: dict[str, Any],
    config_template: dict[str, Any],
    context_log_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    partition_counters: collections.Counter[str] = collections.Counter()
    task_ids: list[int] = []
    seen_task_ids: set[int] = set()
    for context in contexts:
        partition_name = context.get("partition_name")
        if partition_name is not None:
            partition_counters[str(partition_name)] += 1
        task_id = context.get("task_id")
        if task_id is not None and int(task_id) not in seen_task_ids:
            seen_task_ids.add(int(task_id))
            task_ids.append(int(task_id))

    successes = sum(int(bool(context["success"])) for context in contexts)
    config = dict(config_template)
    config["context_log_path"] = str(context_log_path)
    config["transition_trace_path"] = str(trace_path)
    return {
        "task_suite_name": None if selection.get("mode") == "manifest" else selection.get("task_suite_name"),
        "task_ids": task_ids,
        "seed": seed,
        "server": {
            "host": None,
            "port": None,
            "metadata": {"reconstructed_online_results": True},
        },
        "selection": selection,
        "config": config,
        "budget": {
            "context_units_total": len(contexts),
            "online_training_budget_units": sum(
                int(context.get("budget", {}).get("online_budget_units", 0)) for context in contexts
            ),
            "safety_abort_units": sum(
                int(bool(context.get("budget", {}).get("safety_abort", False))) for context in contexts
            ),
            "per_partition_contexts": dict(partition_counters),
            "round_size": selection.get("round_size"),
        },
        "summary": {
            "episodes_run": len(contexts),
            "successes": successes,
            "success_rate": successes / len(contexts) if contexts else 0.0,
            "duration_sec": None,
            "chunk_traces_written": sum(int(context["chunk_traces_written"]) for context in contexts),
        },
        "contexts": contexts,
        "tasks": build_task_summaries(contexts),
    }


def build_finalizer_command(runner_path: Path, options: dict[str, Any], merged_results_dir: Path) -> list[str]:
    rewritten = collections.OrderedDict()
    for key, value in options.items():
        if key in {"--dry-run", "--skip-online", "--results-dir"}:
            continue
        rewritten[key] = value
    rewritten["--results-dir"] = str(merged_results_dir)
    rewritten["--skip-online"] = True

    command = [str(runner_path)]
    for key, value in rewritten.items():
        command.append(key)
        if value is not True:
            command.append(str(value))
    return command


def main() -> int:
    args = parse_args()
    original_job_script = Path(args.original_job_script).resolve()
    original_trace_path = Path(args.original_trace_path).resolve()
    tail_results_dir = Path(args.tail_results_dir).resolve()
    merged_results_dir = Path(args.merged_results_dir).resolve()
    runtime_log_root = Path(args.runtime_log_root).resolve()
    merged_results_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_root.mkdir(parents=True, exist_ok=True)

    log(
        f"[stagee-finalize] start mode={args.trace_reference_mode} "
        f"original_job_script={original_job_script} merged_results_dir={merged_results_dir}"
    )

    workdir, runner_path, options = parse_job_script(original_job_script)
    manifest_path = resolve_path(workdir, options.get("--manifest-path"))
    value_proxy_model_path = resolve_path(workdir, options.get("--value-proxy-model-path"))
    value_proxy_model_path_str, value_proxy_model_id = load_value_proxy_metadata(value_proxy_model_path)
    method = "caver" if "caver" in runner_path.name else "real_only"
    prefix_trace_name = "caver_online_chunks.jsonl" if method == "caver" else "real_only_online_chunks.jsonl"
    merged_trace_path = merged_results_dir / prefix_trace_name
    merged_online_path = merged_results_dir / (
        "caver_online_eval.json" if method == "caver" else "real_only_online_eval.json"
    )
    merged_context_path = merged_results_dir / (
        "caver_online_contexts.jsonl" if method == "caver" else "real_only_online_contexts.jsonl"
    )
    tail_online_path = tail_results_dir / merged_online_path.name
    tail_context_path = tail_results_dir / merged_context_path.name
    tail_trace_path = tail_results_dir / merged_trace_path.name

    candidate_count = int(options.get("--candidate-count", "1"))
    selection_policy = str(options.get("--selection-policy", "first"))
    selector_seed = int(options.get("--selector-seed") or options["--seed"])
    seed = int(options["--seed"])
    selection = load_manifest_selection(manifest_path, options)
    config_template = {
        "num_trials_per_task": int(options.get("--num-trials-per-task", "50")),
        "count_legacy_contexts_as_online_budget": bool(
            options.get("--count-legacy-contexts-as-online-budget", False)
        ),
        "candidate_count": candidate_count,
        "selection_policy": selection_policy,
        "selector_seed": selector_seed,
        "value_proxy_model_path": value_proxy_model_path_str,
        "value_proxy_model_id": value_proxy_model_id,
        "num_steps_wait": int(options.get("--num-steps-wait", "10")),
        "replan_steps": int(options.get("--replan-steps", "5")),
        "resize_size": int(options.get("--resize-size", "224")),
        "resolution": int(options.get("--resolution", "256")),
        "max_steps_override": parse_optional_int(options.get("--max-env-steps")),
        "video_dir": None,
        "save_failures_only": False,
    }

    prefix_contexts, prefix_trace_records = build_contexts_from_prefix_trace(
        original_trace_path=original_trace_path,
        merged_trace_path=(merged_trace_path if args.trace_reference_mode == "materialize" else None),
        completed_prefix=args.completed_prefix,
        candidate_count=candidate_count,
        selection_policy=selection_policy,
        selector_seed=selector_seed,
    )
    log(
        f"[stagee-finalize] recovered {len(prefix_contexts)} prefix contexts "
        f"and {prefix_trace_records} trace records"
    )
    tail_contexts = load_jsonl(tail_context_path)
    tail_online = load_json(tail_online_path)
    overlap = sorted(
        {str(record["context_id"]) for record in prefix_contexts}
        & {str(record["context_id"]) for record in tail_contexts}
    )
    if overlap:
        raise ValueError(f"duplicate context ids between prefix and tail: {overlap[:5]}")

    if args.trace_reference_mode == "materialize":
        tail_trace_records = append_trace(merged_trace_path, tail_trace_path)
        log(
            f"[stagee-finalize] materialized merged trace at {merged_trace_path} "
            f"(tail trace records appended={tail_trace_records})"
        )
    else:
        write_trace_input_manifest(
            manifest_path=merged_trace_path,
            original_trace_path=original_trace_path,
            completed_prefix=args.completed_prefix,
            tail_trace_path=tail_trace_path,
        )
        tail_trace_records = int(tail_online["summary"]["chunk_traces_written"])
        log(
            f"[stagee-finalize] wrote trace-source manifest at {merged_trace_path} "
            f"with tail trace records={tail_trace_records}"
        )
    merged_contexts = prefix_contexts + tail_contexts
    merged_online = build_online_results(
        contexts=merged_contexts,
        seed=seed,
        selection=selection,
        config_template=config_template,
        context_log_path=merged_context_path,
        trace_path=merged_trace_path,
    )
    merged_online["server"] = tail_online.get("server", merged_online["server"])
    merged_online["summary"]["chunk_traces_written"] = prefix_trace_records + tail_trace_records
    write_json(merged_online_path, merged_online)
    write_jsonl(merged_context_path, merged_contexts)

    merge_report = {
        "original_job_script": str(original_job_script),
        "original_trace_path": str(original_trace_path),
        "completed_prefix": args.completed_prefix,
        "tail_results_dir": str(tail_results_dir),
        "merged_results_dir": str(merged_results_dir),
        "trace_reference_mode": args.trace_reference_mode,
        "prefix_trace_records": prefix_trace_records,
        "tail_trace_records": tail_trace_records,
        "merged_context_count": len(merged_contexts),
    }
    write_json(merged_results_dir / "merge_report.json", merge_report)
    log(
        f"[stagee-finalize] merged context count={len(merged_contexts)} "
        f"trace_reference_mode={args.trace_reference_mode}"
    )

    finalizer_cmd = build_finalizer_command(runner_path, options, merged_results_dir)
    if args.dry_run:
        print(json.dumps({"merge_report": merge_report, "finalizer_cmd": finalizer_cmd}, indent=2))
        return 0

    env = dict(os.environ)
    env["CAVER_DEFAULT_RUNTIME_LOG_ROOT"] = str(runtime_log_root)
    log(f"[stagee-finalize] launching finalizer: {' '.join(shlex.quote(part) for part in finalizer_cmd)}")
    subprocess.run(finalizer_cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
