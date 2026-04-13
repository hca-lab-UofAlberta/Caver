#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


RUN_DIR_RE = re.compile(
    r"^stagee__(?P<method>real-only-round|caver-round)__"
    r"(?P<target>.+)__seed(?P<seed>\d+)__budget(?P<budget>\d+)__"
    r"(?P<stamp>\d{8}T\d{6}Z)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the latest Stage-E run per (method, seed, budget) cell so "
            "native-horizon reruns can be tracked without hand-merging directories."
        )
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Directory containing stagee__* run folders (default: runs).",
    )
    parser.add_argument(
        "--target-token",
        default="manifest-t_train_s0-all",
        help="Only include run directories whose target token matches this value.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional Markdown table output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_run(run_dir: Path, method_key: str, target_token: str, seed: int, budget: int, stamp: str) -> dict[str, Any]:
    results_dir = run_dir / "results"
    if method_key == "real-only-round":
        summary_path = results_dir / "real_only_round_summary.json"
        online_path = results_dir / "real_only_online_eval.json"
        context_log_path = results_dir / "real_only_online_contexts.jsonl"
    else:
        summary_path = results_dir / "caver_round_summary.json"
        online_path = results_dir / "caver_online_eval.json"
        context_log_path = results_dir / "caver_online_contexts.jsonl"

    record: dict[str, Any] = {
        "method": ("real_only" if method_key == "real-only-round" else "caver"),
        "method_token": method_key,
        "target_token": target_token,
        "seed": seed,
        "budget": budget,
        "run_dir": str(run_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "stamp": stamp,
        "summary_path": str(summary_path.resolve()),
        "online_path": str(online_path.resolve()),
        "context_log_path": str(context_log_path.resolve()),
        "summary_present": summary_path.exists(),
        "online_present": online_path.exists(),
        "context_log_present": context_log_path.exists(),
    }

    if not summary_path.exists():
        record["status"] = "running_or_incomplete"
        return record

    summary = load_json(summary_path)
    record["status"] = "completed"
    record["workflow"] = summary.get("workflow")
    training = dict(summary.get("training", {}))
    training_log_dir_value = summary.get("training_log_dir")
    if training_log_dir_value:
        training_log_dir = Path(training_log_dir_value)
        replay_buffer_snapshot = Path(training.get("replay_buffer_snapshot", training_log_dir / "replay_buffer_0.pkl"))
        training_completed_marker = Path(training.get("training_completed_marker", training_log_dir / "training_completed.marker"))
        training_completed = replay_buffer_snapshot.exists() or training_completed_marker.exists()
        training["replay_buffer_snapshot"] = str(replay_buffer_snapshot.resolve())
        training["training_completed_marker"] = str(training_completed_marker.resolve())
        if training_completed:
            training["training_completed"] = True
            if "training_skipped" in training and training.get("training_skipped"):
                training["training_skipped"] = False
    record["training"] = training
    record["demo"] = summary.get("demo", {})
    online = summary.get("online", {})
    record["online"] = online
    record["episodes_run"] = online.get("episodes_run")
    record["successes"] = online.get("successes")
    record["success_rate"] = online.get("success_rate")
    record["chunk_traces_written"] = online.get("chunk_traces_written")

    if method_key == "caver-round":
        selector = summary.get("selector", {})
        admission = summary.get("admission", {})
        record["selector"] = selector
        record["admission"] = admission
        record["contexts_admitted"] = admission.get("contexts_admitted")
        record["contexts_rejected"] = admission.get("contexts_rejected")
        record["admitted_trace_records"] = admission.get("admitted_trace_records")

    return record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, cells: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    header = [
        "| Method | Seed | Budget | Status | Successes | Episodes | Admitted | Run Dir |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    lines = []
    for cell in sorted(cells, key=lambda item: (item["method"], item["seed"], item["budget"])):
        admitted = cell.get("contexts_admitted")
        lines.append(
            "| {method} | {seed} | {budget} | {status} | {successes} | {episodes} | {admitted} | `{run_dir}` |".format(
                method=cell["method"],
                seed=cell["seed"],
                budget=cell["budget"],
                status=cell["status"],
                successes=("-" if cell.get("successes") is None else cell["successes"]),
                episodes=("-" if cell.get("episodes_run") is None else cell["episodes_run"]),
                admitted=("-" if admitted is None else admitted),
                run_dir=cell["run_dir"],
            )
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header + lines) + "\n")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"error: runs root not found: {runs_root}")

    latest_by_cell: dict[tuple[str, int, int], tuple[str, Path, str]] = {}
    for candidate in runs_root.iterdir():
        if not candidate.is_dir():
            continue
        match = RUN_DIR_RE.match(candidate.name)
        if not match:
            continue
        target_token = match.group("target")
        if target_token != args.target_token:
            continue
        method_key = match.group("method")
        seed = int(match.group("seed"))
        budget = int(match.group("budget"))
        stamp = match.group("stamp")
        key = (method_key, seed, budget)
        previous = latest_by_cell.get(key)
        if previous is None or stamp > previous[0]:
            latest_by_cell[key] = (stamp, candidate, target_token)

    cells: list[dict[str, Any]] = []
    for (method_key, seed, budget), (stamp, run_dir, target_token) in sorted(latest_by_cell.items()):
        cells.append(summarize_run(run_dir, method_key, target_token, seed, budget, stamp))

    payload = {
        "target_token": args.target_token,
        "runs_root": str(runs_root),
        "cells": cells,
    }

    if args.output_json:
        write_json(Path(args.output_json).resolve(), payload)
    if args.output_md:
        write_markdown(Path(args.output_md).resolve(), cells)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
