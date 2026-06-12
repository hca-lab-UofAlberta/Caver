#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import gzip
import json
import logging
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Any, TextIO

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

_STAGEE_DIR = Path(__file__).resolve().parents[1] / "stagee"
if str(_STAGEE_DIR) not in sys.path:
    sys.path.append(str(_STAGEE_DIR))

from caver_heuristic import append_selector_history
from caver_heuristic import compute_progress_value_from_semantic_state
from caver_heuristic import compute_selector_decision
from caver_heuristic import make_selector_history

K1_GUARDED_CAVER_MIXTURE_MASS = 0.50
from libero_gesim_provider import extract_libero_gesim_provider_observation
from libero_gesim_provider import LIBERO_GESIM_HISTORY_LENGTH
from libero_gesim_provider import write_libero_gesim_bundle
from stage0_value_proxy import load_value_proxy_model
from stagee_dr_calibration import load_stagee_dr_calibrator_model
from stagee_lvd_selector import load_lvd_selector_model

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
STAGE0_PARTITIONS = ("T_seed_S0", "T_train_S0", "T_val_S0", "T_test_S0")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GESIM_RUNNER = _REPO_ROOT / "scripts" / "stagee" / "run_gesim_inference.sh"
_GESIM_PREPARE_RUNTIME_ONCE = _REPO_ROOT / "scripts" / "stagee" / "prepare_gesim_runtime_once.sh"
_GESIM_PERSISTENT_WORKER = _REPO_ROOT / "scripts" / "stagee" / "gesim_persistent_worker.py"
_GESIM_ENV_WRAPPER = _REPO_ROOT / "scripts" / "env" / "with_gesim_infer.sh"


@dataclass(frozen=True)
class EvalContext:
    suite_name: str
    task_id: int
    init_state_index: int
    context_id: str
    source_mode: str
    partition_name: str | None = None
    proxy_family_id: str | None = None
    proposal_task: str | None = None
    task_name: str | None = None


def partition_budget_domain(partition_name: str | None) -> str:
    if partition_name == "T_seed_S0":
        return "seed_warm_start"
    if partition_name == "T_train_S0":
        return "online_train"
    if partition_name == "T_val_S0":
        return "validation"
    if partition_name == "T_test_S0":
        return "audit_test"
    return "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO evaluation against an OpenPI websocket policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite-name", default="libero_spatial", choices=LIBERO_TASK_SUITES)
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional Stage-0 partition manifest. When set, the evaluator runs explicit contexts from the manifest.",
    )
    parser.add_argument(
        "--partition-name",
        default=None,
        choices=STAGE0_PARTITIONS,
        help="Required with --manifest-path. Selects T_seed_S0, T_train_S0, T_val_S0, or T_test_S0.",
    )
    parser.add_argument(
        "--family-ids",
        default="",
        help="Optional comma-separated subset of Stage-0 proxy family ids when --manifest-path is used.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated subset of task ids. Example: 0,3,4",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional limit on the number of tasks to run.")
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Optional limit on the number of selected contexts after manifest filtering.",
    )
    parser.add_argument(
        "--context-offset",
        type=int,
        default=0,
        help="Optional offset into the selected manifest contexts before applying --max-contexts.",
    )
    parser.add_argument(
        "--round-size",
        type=int,
        default=25,
        help="Stage-0 online round size used for budget-ledger annotation in manifest mode.",
    )
    parser.add_argument(
        "--count-legacy-contexts-as-online-budget",
        action="store_true",
        help=(
            "When set in legacy mode, treat each executed context as one online-budget unit. "
            "This is useful for Stage-E real-only rounds before the manifest-driven CAVER loop exists."
        ),
    )
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1,
        help="Number of policy chunks to sample at each replan step before selection.",
    )
    parser.add_argument(
        "--selection-policy",
        default="first",
        choices=("first", "uniform", "caver_heuristic", "caver_k1_guarded", "caver_lvd"),
        help="Chunk selector applied across the sampled candidates at each policy query.",
    )
    parser.add_argument(
        "--selector-seed",
        type=int,
        default=None,
        help="Optional RNG seed for stochastic candidate selection. Defaults to --seed when needed.",
    )
    parser.add_argument(
        "--value-proxy-model-path",
        default=None,
        help="Optional fitted Stage-0 value-proxy JSON used by the caver_heuristic selector.",
    )
    parser.add_argument(
        "--dr-calibrator-model-path",
        default=None,
        help="Optional lagged Stage-E DR calibrator JSON used to refresh selector utility inputs.",
    )
    parser.add_argument(
        "--lvd-selector-model-path",
        default=None,
        help="Optional CAVER-LVD selector JSON used by selection_policy=caver_lvd.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override suite-specific rollout horizon. Defaults to the upstream OpenPI values per suite.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-dir", default=None, help="Optional directory for replay videos.")
    parser.add_argument(
        "--save-failures-only",
        action="store_true",
        help="When set with --video-dir, save only failed episodes.",
    )
    parser.add_argument(
        "--provider-mode",
        default="none",
        choices=("none", "gesim_bundle", "gesim_live_summary"),
        help=(
            "Optional provider-side export mode. "
            "'gesim_bundle' materializes GE-Sim-compatible candidate bundles for each policy query. "
            "'gesim_live_summary' also runs GE-Sim immediately and attaches a provider summary."
        ),
    )
    parser.add_argument(
        "--provider-bundle-root",
        default=None,
        help="Optional output root for provider bundles. Required when --provider-mode uses GE-Sim bundles.",
    )
    parser.add_argument(
        "--provider-gesim-timeout-sec",
        type=int,
        default=900,
        help="Timeout for one live GE-Sim candidate inference when --provider-mode=gesim_live_summary.",
    )
    parser.add_argument(
        "--provider-gesim-prompt",
        default="best quality, consistent and smooth motion, realistic, clear and distinct.",
        help="Prompt string passed through to GE-Sim when live provider inference is enabled.",
    )
    parser.add_argument(
        "--provider-gesim-execution-mode",
        default="persistent_worker",
        choices=("persistent_worker", "subprocess"),
        help=(
            "Execution strategy for live GE-Sim summaries. "
            "'persistent_worker' loads GE-Sim once per evaluation process; "
            "'subprocess' preserves the legacy per-candidate shell-out path."
        ),
    )
    parser.add_argument("--results-path", default=None, help="Optional JSON path for evaluation summary output.")
    parser.add_argument(
        "--context-log-path",
        default=None,
        help="Optional JSONL path for per-context execution and budget ledger records.",
    )
    parser.add_argument(
        "--transition-trace-path",
        default=None,
        help=(
            "Optional JSONL path for chunk-level step traces. "
            "Each record captures one policy query and the executed primitive actions beneath it."
        ),
    )
    parser.add_argument(
        "--demo-trace-path",
        default=None,
        help="Optional full observation-bearing JSONL/JSONL.GZ trace for backend demo conversion.",
    )
    parser.add_argument(
        "--demo-trace-write-policy",
        default="all",
        choices=("all", "success_only"),
        help="Which full traces to write to --demo-trace-path. Selector traces are unaffected.",
    )
    parser.add_argument(
        "--trace-policy-aux-mode",
        default="full",
        choices=("full", "summary", "none"),
        help="Amount of policy auxiliary payload stored in traces.",
    )
    parser.add_argument(
        "--demo-trace-policy-aux-mode",
        default="full",
        choices=("full", "summary", "none"),
        help=(
            "Amount of policy auxiliary payload stored in the full demo trace. "
            "Keep this at 'full' for exact_offline_nft because it needs NFT tensors."
        ),
    )
    parser.add_argument(
        "--trace-next-obs-mode",
        default="all",
        choices=("all", "last"),
        help="Store every next observation or only the final next observation per chunk.",
    )
    parser.add_argument(
        "--trace-stage0-progress",
        action="store_true",
        help=(
            "Attach compact LIBERO semantic state and verified Stage-0 progress values "
            "to chunk traces for progress-based segment repair."
        ),
    )
    parser.add_argument(
        "--trace-observation-mode",
        default="full",
        choices=("full", "none"),
        help="Use 'none' for compact selector traces that omit image/state observations.",
    )
    return parser.parse_args()


def tail_text_file(path: Path, *, max_lines: int = 40) -> str | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    if not lines:
        return None
    return "".join(lines[-max_lines:]).strip() or None


class GesimPersistentWorkerClient:
    def __init__(
        self,
        *,
        provider_bundle_root: Path,
        startup_timeout_sec: int = 1800,
    ) -> None:
        self.provider_bundle_root = provider_bundle_root
        self.worker_root = provider_bundle_root / "_gesim_worker"
        self.runtime_config_path = self.worker_root / "gesim_runtime.yaml"
        self.runtime_status_path = self.worker_root / "gesim_runtime_status.json"
        self.stderr_log_path = self.worker_root / "worker.stderr.log"
        self.startup_timeout_sec = startup_timeout_sec
        self.process: subprocess.Popen[str] | None = None
        self.stderr_handle: TextIO | None = None
        self.request_counter = 0

    def ensure_started(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close(force=True)
        self.worker_root.mkdir(parents=True, exist_ok=True)
        self._prepare_runtime_once()
        self.stderr_handle = self.stderr_log_path.open("a", encoding="utf-8")
        cmd = [
            str(_GESIM_ENV_WRAPPER),
            "--",
            "python",
            "-u",
            str(_GESIM_PERSISTENT_WORKER),
            "--config-path",
            str(self.runtime_config_path),
            "--runtime-status-path",
            str(self.runtime_status_path),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
        )
        ready_payload = self._read_json_line(timeout_sec=self.startup_timeout_sec)
        if ready_payload.get("event") != "ready":
            self.close(force=True)
            raise RuntimeError(f"GE-Sim worker failed to become ready: {ready_payload}")

    def infer(
        self,
        *,
        bundle_dir: Path,
        timeout_sec: int,
        prompt: str,
    ) -> dict[str, Any]:
        output_dir = bundle_dir / "inference"
        try:
            self.ensure_started()
            process = self._require_process()
            if process.stdin is None:
                raise RuntimeError("GE-Sim worker stdin is unavailable")
            self.request_counter += 1
            request_id = f"gesim-{self.request_counter:08d}"
            payload = {
                "action": "infer",
                "request_id": request_id,
                "bundle_dir": str(bundle_dir),
                "output_dir": str(output_dir),
                "prompt": prompt,
            }
            process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            process.stdin.flush()
            response = self._read_json_line(timeout_sec=timeout_sec)
            if response.get("request_id") != request_id:
                raise RuntimeError(
                    f"GE-Sim worker returned mismatched request_id={response.get('request_id')} expected={request_id}"
                )
            return response
        except TimeoutError as exc:
            self.close(force=True)
            return {
                "inference_status": "timeout",
                "inference_output_dir": str(output_dir),
                "provider_summary_path": str(output_dir / "provider_summary.json"),
                "provider_summary": None,
                "provider_error": (
                    f"TimeoutExpired: {exc}\n"
                    f"{tail_text_file(self.stderr_log_path) or ''}".strip()
                ),
            }
        except Exception as exc:  # noqa: BLE001
            self.close(force=True)
            return {
                "inference_status": "error",
                "inference_output_dir": str(output_dir),
                "provider_summary_path": str(output_dir / "provider_summary.json"),
                "provider_summary": None,
                "provider_error": (
                    f"{type(exc).__name__}: {exc}\n"
                    f"{tail_text_file(self.stderr_log_path) or ''}".strip()
                ),
            }

    def close(self, *, force: bool = False) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                if not force and process.stdin is not None:
                    process.stdin.write(json.dumps({"action": "shutdown", "request_id": "shutdown"}) + "\n")
                    process.stdin.flush()
                    self._read_json_line(timeout_sec=10.0, process=process)
                else:
                    process.terminate()
                    process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                process.kill()
                process.wait(timeout=10)
        if self.stderr_handle is not None:
            self.stderr_handle.close()
            self.stderr_handle = None

    def _prepare_runtime_once(self) -> None:
        cmd = [
            "bash",
            str(_GESIM_PREPARE_RUNTIME_ONCE),
            "--output-dir",
            str(self.worker_root),
        ]
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-30:]) if completed.stderr.strip() else ""
            stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-30:]) if completed.stdout.strip() else ""
            raise RuntimeError(
                "GE-Sim runtime preparation failed\n"
                f"stdout:\n{stdout_tail}\n"
                f"stderr:\n{stderr_tail}".strip()
            )

    def _require_process(self) -> subprocess.Popen[str]:
        process = self.process
        if process is None:
            raise RuntimeError("GE-Sim worker process is not running")
        return process

    def _read_json_line(
        self,
        *,
        timeout_sec: float,
        process: subprocess.Popen[str] | None = None,
    ) -> dict[str, Any]:
        line = self._readline_with_timeout(timeout_sec=timeout_sec, process=process)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"failed to decode GE-Sim worker JSON response: {line!r}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"expected GE-Sim worker JSON object, got {type(payload).__name__}")
        return payload

    def _readline_with_timeout(
        self,
        *,
        timeout_sec: float,
        process: subprocess.Popen[str] | None = None,
    ) -> str:
        process = process or self._require_process()
        if process.stdout is None:
            raise RuntimeError("GE-Sim worker stdout is unavailable")
        deadline = time.time() + timeout_sec
        while True:
            if process.poll() is not None:
                remaining = process.stdout.readline()
                if remaining:
                    return remaining.strip()
                stderr_tail = tail_text_file(self.stderr_log_path)
                raise RuntimeError(
                    "GE-Sim worker exited before replying"
                    + (f"\n{stderr_tail}" if stderr_tail else "")
                )
            remaining_sec = deadline - time.time()
            if remaining_sec <= 0:
                raise TimeoutError(f"GE-Sim worker response timed out after {timeout_sec:.1f}s")
            ready, _, _ = select.select([process.stdout], [], [], min(remaining_sec, 1.0))
            if not ready:
                continue
            line = process.stdout.readline()
            if not line:
                continue
            return line.strip()


def run_gesim_live_summary(
    *,
    bundle_dir: Path,
    timeout_sec: int,
    prompt: str,
    worker_client: GesimPersistentWorkerClient | None = None,
) -> dict[str, Any]:
    if worker_client is not None:
        return worker_client.infer(bundle_dir=bundle_dir, timeout_sec=timeout_sec, prompt=prompt)

    output_dir = bundle_dir / "inference"
    cmd = [
        str(_GESIM_RUNNER),
        "--bundle-dir",
        str(bundle_dir),
        "--output-dir",
        str(output_dir),
        "--prompt",
        str(prompt),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "inference_status": "timeout",
            "inference_output_dir": str(output_dir),
            "provider_summary_path": str(output_dir / "provider_summary.json"),
            "provider_summary": None,
            "provider_error": f"TimeoutExpired: {exc}",
        }

    summary_path = output_dir / "provider_summary.json"
    summary_payload = None
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                raw_payload = json.load(handle)
            if isinstance(raw_payload, dict):
                summary_payload = raw_payload
        except json.JSONDecodeError:
            summary_payload = None

    stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-20:]) if completed.stderr.strip() else None
    stdout_tail = "\n".join(completed.stdout.strip().splitlines()[-20:]) if completed.stdout.strip() else None
    return {
        "inference_status": "completed" if completed.returncode == 0 else "error",
        "inference_returncode": int(completed.returncode),
        "inference_output_dir": str(output_dir),
        "provider_summary_path": str(summary_path),
        "provider_summary": summary_payload,
        "provider_error": None if completed.returncode == 0 else stderr_tail or stdout_tail,
    }


def resolve_task_ids(args: argparse.Namespace, num_tasks_in_suite: int) -> list[int]:
    if args.task_ids:
        task_ids = []
        for raw_task_id in args.task_ids.split(","):
            value = raw_task_id.strip()
            if not value:
                continue
            task_id = int(value)
            if task_id < 0 or task_id >= num_tasks_in_suite:
                raise ValueError(f"task id {task_id} is out of range for suite with {num_tasks_in_suite} tasks")
            task_ids.append(task_id)
        return task_ids

    task_ids = list(range(num_tasks_in_suite))
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    return task_ids


def get_max_steps(args: argparse.Namespace, task_suite_name: str) -> int:
    if args.max_steps is not None:
        return args.max_steps
    return LIBERO_MAX_STEPS[task_suite_name]


def parse_family_ids(raw_value: str) -> list[str]:
    family_ids = []
    for raw_family_id in raw_value.split(","):
        family_id = raw_family_id.strip()
        if family_id:
            family_ids.append(family_id)
    return family_ids


def load_manifest_contexts(args: argparse.Namespace) -> tuple[list[EvalContext], dict[str, Any]]:
    manifest_path = Path(args.manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    requested_family_ids = parse_family_ids(args.family_ids)
    requested_family_id_set = set(requested_family_ids)
    contexts: list[EvalContext] = []
    selected_family_ids: list[str] = []

    for family in manifest["families"]:
        family_id = family["family_id"]
        if requested_family_id_set and family_id not in requested_family_id_set:
            continue

        selected_family_ids.append(family_id)
        for context in family["partitions"][args.partition_name]:
            contexts.append(
                EvalContext(
                    suite_name=context["suite"],
                    task_id=int(context["task_index"]),
                    init_state_index=int(context["init_state_index"]),
                    context_id=context["context_id"],
                    source_mode="manifest",
                    partition_name=args.partition_name,
                    proxy_family_id=family_id,
                    proposal_task=family.get("proposal_task"),
                    task_name=context.get("task_name"),
                )
            )

    if requested_family_id_set:
        missing_family_ids = sorted(requested_family_id_set.difference(selected_family_ids))
        if missing_family_ids:
            raise ValueError(f"requested family ids were not found in manifest: {missing_family_ids}")

    if args.context_offset < 0:
        raise ValueError("--context-offset must be non-negative")
    if args.context_offset:
        contexts = contexts[args.context_offset :]
    if args.max_contexts is not None:
        if args.max_contexts < 1:
            raise ValueError("--max-contexts must be positive")
        contexts = contexts[: args.max_contexts]

    if not contexts:
        raise ValueError("manifest selection produced zero contexts")

    metadata = {
        "mode": "manifest",
        "manifest_path": str(manifest_path),
        "partition_name": args.partition_name,
        "requested_family_ids": requested_family_ids,
        "selected_family_ids": selected_family_ids,
        "context_offset": args.context_offset,
        "max_contexts": args.max_contexts,
        "round_size": args.round_size,
        "manifest_global_partition_counts": manifest.get("global_partition_counts", {}),
    }
    return contexts, metadata


def build_legacy_contexts(args: argparse.Namespace) -> tuple[list[EvalContext], dict[str, Any]]:
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = resolve_task_ids(args, task_suite.n_tasks)
    contexts: list[EvalContext] = []

    for task_id in task_ids:
        initial_states = task_suite.get_task_init_states(task_id)
        episodes_to_run = min(args.num_trials_per_task, len(initial_states))
        for episode_idx in range(episodes_to_run):
            contexts.append(
                EvalContext(
                    suite_name=args.task_suite_name,
                    task_id=task_id,
                    init_state_index=episode_idx,
                    context_id=f"{args.task_suite_name}__task{task_id:02d}__episode{episode_idx:03d}",
                    source_mode="legacy",
                )
            )

    metadata = {
        "mode": "legacy",
        "task_suite_name": args.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
    }
    return contexts, metadata


def build_eval_contexts(args: argparse.Namespace) -> tuple[list[EvalContext], dict[str, Any]]:
    if args.manifest_path:
        if args.partition_name is None:
            raise ValueError("--partition-name is required with --manifest-path")
        return load_manifest_contexts(args)
    return build_legacy_contexts(args)


def build_budget_record(
    *,
    context: EvalContext,
    partition_context_index: int,
    round_size: int,
    count_legacy_contexts_as_online_budget: bool,
) -> dict[str, Any]:
    counts_against_online_budget = (
        context.partition_name == "T_train_S0"
        or (context.partition_name is None and count_legacy_contexts_as_online_budget)
    )
    if context.partition_name is None and count_legacy_contexts_as_online_budget:
        budget_domain = "legacy_online"
    else:
        budget_domain = partition_budget_domain(context.partition_name)
    online_budget_index = partition_context_index if counts_against_online_budget else None
    round_index = ((partition_context_index - 1) // round_size + 1) if counts_against_online_budget else 0
    round_context_index = ((partition_context_index - 1) % round_size + 1) if counts_against_online_budget else None

    return {
        "budget_domain": budget_domain,
        "partition_name": context.partition_name,
        "counts_against_online_budget": counts_against_online_budget,
        "context_cost_units": 1,
        "online_budget_units": 1 if counts_against_online_budget else 0,
        "partition_context_index": partition_context_index,
        "online_budget_index": online_budget_index,
        "round_index": round_index,
        "round_context_index": round_context_index,
        "round_size": round_size,
        "safety_abort": False,
        "budget_reason": "executed_context",
    }


def get_libero_env(
    task,
    resolution: int,
    seed: int,
    *,
    include_frontview: bool = False,
) -> tuple[OffScreenRenderEnv, str]:
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    camera_names = ["agentview", "robot0_eye_in_hand"]
    if include_frontview:
        camera_names.insert(0, "frontview")
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
    }
    render_gpu_device_id = os.environ.get("LIBERO_RENDER_GPU_DEVICE_ID")
    if render_gpu_device_id is not None and render_gpu_device_id != "":
        env_args["render_gpu_device_id"] = int(render_gpu_device_id)
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def maybe_save_video(
    video_dir: Path | None,
    task_description: str,
    task_id: int,
    episode_idx: int,
    success: bool,
    replay_images: list[np.ndarray],
) -> str | None:
    if video_dir is None or not replay_images:
        return None

    import imageio

    suffix = "success" if success else "failure"
    task_segment = task_description.replace(" ", "_")
    output_path = video_dir / f"task{task_id:02d}_episode{episode_idx:03d}_{task_segment}_{suffix}.mp4"
    imageio.mimwrite(output_path, [np.asarray(frame) for frame in replay_images], fps=10)
    return str(output_path)


def extract_policy_observation(
    obs: dict[str, Any],
    *,
    resize_size: int,
) -> dict[str, np.ndarray]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat_to_axisangle(obs["robot0_eef_quat"].copy()),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return {
        "image": image,
        "wrist_image": wrist_image,
        "state": state,
    }


def _serializable_position(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    if array.size < 3:
        return None
    return [float(item) for item in array[:3]]


def extract_stage0_semantic_state(
    obs: dict[str, Any],
    *,
    env: OffScreenRenderEnv,
    context: EvalContext,
    task_description: str,
) -> dict[str, Any]:
    object_positions: dict[str, list[float]] = {}
    for key, value in obs.items():
        if key.startswith("robot0"):
            continue
        if not key.endswith("_pos") or "_to_robot0" in key:
            continue
        position = _serializable_position(value)
        if position is not None:
            object_positions[key[: -len("_pos")]] = position

    site_positions: dict[str, list[float]] = {}
    for site_index in range(env.sim.model.nsite):
        site_name = env.sim.model.site_id2name(site_index)
        if not site_name:
            continue
        lowered = site_name.lower()
        if not any(token in lowered for token in ("basket", "tray", "caddy", "contain", "region", "cabinet")):
            continue
        site_positions[site_name] = [float(item) for item in env.sim.data.site_xpos[site_index].tolist()]

    body_positions: dict[str, list[float]] = {}
    for body_index in range(env.sim.model.nbody):
        body_name = env.sim.model.body_id2name(body_index)
        if not body_name:
            continue
        lowered = body_name.lower()
        if not any(token in lowered for token in ("basket", "tray", "caddy", "cabinet")):
            continue
        body_positions[body_name] = [float(item) for item in env.sim.data.body_xpos[body_index].tolist()]

    joint_qpos: dict[str, float] = {}
    for joint_index in range(env.sim.model.njnt):
        joint_name = env.sim.model.joint_id2name(joint_index)
        if not joint_name:
            continue
        lowered = joint_name.lower()
        if not any(token in lowered for token in ("drawer", "cabinet", "level")):
            continue
        try:
            qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if qpos.size == 1:
            joint_qpos[joint_name] = float(qpos[0])

    semantic_state = {
        "semantic_state_schema": "libero_stage0_semantic_state_v1",
        "context_id": context.context_id,
        "proxy_family_id": context.proxy_family_id,
        "proposal_task": context.proposal_task,
        "task_description": str(task_description),
        "task_id": context.task_id,
        "object_positions": object_positions,
        "site_positions": site_positions,
        "body_positions": body_positions,
        "joint_qpos": joint_qpos,
    }
    semantic_state["progress"] = compute_progress_value_from_semantic_state(semantic_state)
    return semantic_state


def write_jsonl_record(handle: TextIO, payload: dict[str, Any]) -> None:
    json.dump(payload, handle, sort_keys=True, default=json_default)
    handle.write("\n")
    handle.flush()


def compact_policy_aux_payload(payload: dict[str, Any] | None, *, mode: str) -> dict[str, Any] | None:
    if payload is None or mode == "none":
        return None
    if mode == "full":
        return payload
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, np.ndarray):
            compact[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "mean": float(np.mean(value)) if value.size else 0.0,
                "std": float(np.std(value)) if value.size else 0.0,
            }
        elif isinstance(value, (list, tuple)) and len(value) <= 16:
            compact[key] = value
        else:
            compact[key] = {
                "type": type(value).__name__,
                "length": len(value) if hasattr(value, "__len__") else None,
            }
    return compact or None


def prepare_trace_record_for_write(record: dict[str, Any], args: argparse.Namespace, *, full_demo: bool) -> dict[str, Any]:
    prepared = dict(record)
    policy_aux_mode = args.demo_trace_policy_aux_mode if full_demo else args.trace_policy_aux_mode
    if policy_aux_mode != "full":
        prepared["candidate_policy_aux"] = [
            compact_policy_aux_payload(payload, mode=policy_aux_mode)
            for payload in prepared.get("candidate_policy_aux", [])
        ]
        prepared["selected_policy_aux"] = compact_policy_aux_payload(
            prepared.get("selected_policy_aux"),
            mode=policy_aux_mode,
        )
    if args.trace_next_obs_mode == "last":
        next_obs = prepared.get("next_obs_sequence") or []
        prepared["next_obs_sequence"] = next_obs[-1:] if next_obs else []
    if not full_demo and args.trace_observation_mode == "none":
        prepared.pop("obs", None)
        prepared.pop("next_obs_sequence", None)
        prepared["observation_payload_mode"] = "omitted"
    return prepared


def open_text_maybe_gzip(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def run_episode(
    *,
    env: OffScreenRenderEnv,
    client: websocket_client_policy.WebsocketClientPolicy,
    initial_state: np.ndarray,
    task_description: str,
    task_id: int,
    episode_idx: int,
    args: argparse.Namespace,
    max_steps: int,
    video_dir: Path | None,
    context: EvalContext,
    budget_record: dict[str, Any],
    transition_trace_handle: TextIO | None,
    demo_trace_handle: TextIO | None,
    selection_rng: np.random.Generator | None,
    selector_seed: int | None,
    selector_history: collections.deque[np.ndarray],
    value_proxy_model: dict[str, Any] | None,
    dr_calibrator_model: dict[str, Any] | None,
    lvd_selector_model: dict[str, Any] | None,
    gesim_worker_client: GesimPersistentWorkerClient | None,
) -> dict[str, Any]:
    env.reset()
    obs = env.set_init_state(initial_state)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    replay_images: list[np.ndarray] = []
    success = False
    error = None
    policy_steps = 0
    episode_started = time.time()
    policy_query_index = 0
    chunk_traces_written = 0
    active_chunk_trace: dict[str, Any] | None = None
    context_demo_chunk_traces: list[dict[str, Any]] = []
    selected_candidate_probabilities: list[float] = []
    selected_candidate_indices: list[int] = []
    candidate_probability_vectors: list[list[float]] = []
    safe_candidate_counts: list[int] = []
    provider_observation_history = (
        collections.deque(maxlen=LIBERO_GESIM_HISTORY_LENGTH)
        if args.provider_mode in ("gesim_bundle", "gesim_live_summary")
        else None
    )

    def flush_chunk_trace(reason: str, *, trace_error: str | None = None) -> None:
        nonlocal active_chunk_trace, chunk_traces_written
        if active_chunk_trace is None:
            return
        if transition_trace_handle is None and demo_trace_handle is None:
            active_chunk_trace = None
            return
        if not active_chunk_trace["actions"]:
            active_chunk_trace = None
            return
        active_chunk_trace["completed_reason"] = reason
        active_chunk_trace["error"] = trace_error
        active_chunk_trace["steps_executed"] = len(active_chunk_trace["actions"])
        if transition_trace_handle is not None:
            write_jsonl_record(
                transition_trace_handle,
                prepare_trace_record_for_write(active_chunk_trace, args, full_demo=False),
            )
        if demo_trace_handle is not None:
            demo_record = prepare_trace_record_for_write(active_chunk_trace, args, full_demo=True)
            if args.demo_trace_write_policy == "all":
                write_jsonl_record(demo_trace_handle, demo_record)
            elif args.demo_trace_write_policy == "success_only":
                context_demo_chunk_traces.append(demo_record)
        chunk_traces_written += 1
        active_chunk_trace = None

    for sim_step in range(max_steps + args.num_steps_wait):
        try:
            if sim_step < args.num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                continue

            policy_obs = extract_policy_observation(obs, resize_size=args.resize_size)
            stage0_semantic_state = (
                extract_stage0_semantic_state(
                    obs,
                    env=env,
                    context=context,
                    task_description=str(task_description),
                )
                if args.trace_stage0_progress
                else None
            )
            if provider_observation_history is not None:
                provider_observation_history.append(
                    extract_libero_gesim_provider_observation(obs, env=env)
                )

            if video_dir is not None:
                replay_images.append(policy_obs["image"])

            if not action_plan:
                policy_query_index += 1
                element = {
                    "observation/image": policy_obs["image"],
                    "observation/wrist_image": policy_obs["wrist_image"],
                    "observation/state": policy_obs["state"],
                    "prompt": str(task_description),
                }
                candidate_chunks: list[np.ndarray] = []
                candidate_action_lengths: list[int] = []
                candidate_policy_aux_payloads: list[dict[str, Any] | None] = []
                candidate_provider_aux_payloads: list[dict[str, Any] | None] = []
                for _candidate_index in range(args.candidate_count):
                    policy_result = dict(client.infer(element))
                    action_chunk = np.asarray(policy_result["actions"])
                    candidate_action_lengths.append(len(action_chunk))
                    policy_aux_payload = {
                        key: value for key, value in policy_result.items() if key != "actions"
                    }
                    candidate_policy_aux_payloads.append(policy_aux_payload or None)
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"policy returned {len(action_chunk)} actions, which is shorter than replan_steps={args.replan_steps}"
                    )
                    candidate_chunks.append(np.asarray(action_chunk[: args.replan_steps], dtype=np.float32))
                if args.provider_mode in ("gesim_bundle", "gesim_live_summary"):
                    if provider_observation_history is None or not provider_observation_history:
                        raise RuntimeError("provider bundle export requested without any provider observation history")
                    if args.provider_bundle_root is None:
                        raise RuntimeError("provider bundle export requested without a provider bundle root")
                    provider_history_snapshot = list(provider_observation_history)
                    for candidate_index, candidate_chunk in enumerate(candidate_chunks):
                        bundle_dir = (
                            Path(args.provider_bundle_root)
                            / str(context.context_id)
                            / f"query_{policy_query_index:03d}"
                            / f"candidate_{candidate_index:02d}"
                        )
                        provider_payload = write_libero_gesim_bundle(
                            provider_observation_history=provider_history_snapshot,
                            candidate_chunk=candidate_chunk,
                            output_dir=bundle_dir,
                            context_id=context.context_id,
                            policy_query_index=policy_query_index,
                            candidate_index=candidate_index,
                        )
                        if args.provider_mode == "gesim_live_summary":
                            provider_payload.update(
                                run_gesim_live_summary(
                                    bundle_dir=bundle_dir,
                                    timeout_sec=args.provider_gesim_timeout_sec,
                                    prompt=args.provider_gesim_prompt,
                                    worker_client=gesim_worker_client,
                                )
                            )
                        candidate_provider_aux_payloads.append(provider_payload)
                else:
                    candidate_provider_aux_payloads = [None] * len(candidate_chunks)

                safe_candidate_mask = [True] * len(candidate_chunks)
                candidate_safety_reason_codes: list[str | None] = [None] * len(candidate_chunks)
                safe_candidate_indices = [idx for idx, is_safe in enumerate(safe_candidate_mask) if is_safe]
                safe_candidate_count = len(safe_candidate_indices)
                selected_candidate_index = 0
                selected_candidate_probability = 1.0
                candidate_probabilities = [0.0] * len(candidate_chunks)
                candidate_metric_table: list[dict[str, Any]] = []
                selector_score_table: list[float] = []
                selector_mode = args.selection_policy
                selector_temperature: float | None = None
                selector_exploration_floor: float | None = None
                selector_weights: dict[str, float] | None = None
                selector_value_proxy_source: str | None = None
                selector_value_proxy_model_id: str | None = None
                selector_utility_scale_source: str | None = None
                selector_utility_scale_model_id: str | None = None
                if args.selection_policy == "uniform":
                    if selection_rng is None:
                        raise RuntimeError("selection RNG is unavailable for uniform candidate selection")
                    selected_candidate_index = int(selection_rng.integers(0, len(candidate_chunks)))
                    selected_candidate_probability = 1.0 / float(len(candidate_chunks))
                    candidate_probabilities = [selected_candidate_probability] * len(candidate_chunks)
                    if value_proxy_model is not None or dr_calibrator_model is not None or args.provider_mode != "none":
                        selector_diagnostics = compute_selector_decision(
                            candidate_chunks,
                            safe_candidate_mask=safe_candidate_mask,
                            candidate_provider_aux=candidate_provider_aux_payloads,
                            history_vectors=selector_history,
                            rng=None,
                            value_proxy_model=value_proxy_model,
                            dr_calibrator_model=dr_calibrator_model,
                            lvd_selector_model=None,
                            proxy_family_id=context.proxy_family_id,
                            policy_query_index=policy_query_index,
                        )
                        candidate_metric_table = [
                            {
                                key: value
                                for key, value in metric.items()
                                if key != "base_feature_vector"
                            }
                            for metric in selector_diagnostics["candidate_metrics"]
                        ]
                        selector_score_table = list(selector_diagnostics["candidate_scores"])
                        selector_value_proxy_source = selector_diagnostics.get("value_proxy_source")
                        selector_value_proxy_model_id = selector_diagnostics.get("value_proxy_model_id")
                        selector_utility_scale_source = selector_diagnostics.get("utility_scale_source")
                        selector_utility_scale_model_id = selector_diagnostics.get("utility_scale_model_id")
                elif args.selection_policy in {"caver_heuristic", "caver_k1_guarded", "caver_lvd"}:
                    if selection_rng is None:
                        raise RuntimeError(f"selection RNG is unavailable for {args.selection_policy} candidate selection")
                    selector_decision = compute_selector_decision(
                        candidate_chunks,
                        safe_candidate_mask=safe_candidate_mask,
                        candidate_provider_aux=candidate_provider_aux_payloads,
                        history_vectors=selector_history,
                        rng=selection_rng,
                        value_proxy_model=value_proxy_model,
                        dr_calibrator_model=dr_calibrator_model,
                        lvd_selector_model=lvd_selector_model,
                        proxy_family_id=context.proxy_family_id,
                        policy_query_index=policy_query_index,
                    )
                    candidate_probabilities = list(selector_decision["candidate_probabilities"])
                    if args.selection_policy == "caver_k1_guarded":
                        if safe_candidate_mask[0]:
                            guarded_mass = float(K1_GUARDED_CAVER_MIXTURE_MASS)
                            candidate_probabilities = [
                                float((1.0 - guarded_mass) * probability)
                                for probability in candidate_probabilities
                            ]
                            candidate_probabilities[0] += guarded_mass
                            safe_total = sum(
                                probability
                                for probability, is_safe in zip(candidate_probabilities, safe_candidate_mask)
                                if is_safe
                            )
                            candidate_probabilities = [
                                float(probability / safe_total) if is_safe else 0.0
                                for probability, is_safe in zip(candidate_probabilities, safe_candidate_mask)
                            ]
                        safe_probability_vector = [candidate_probabilities[index] for index in safe_candidate_indices]
                        safe_probability_total = sum(safe_probability_vector)
                        safe_probability_vector = [value / safe_probability_total for value in safe_probability_vector]
                        draw = float(selection_rng.random())
                        cumulative = 0.0
                        selected_candidate_index = safe_candidate_indices[-1]
                        for candidate_index, probability in zip(safe_candidate_indices, safe_probability_vector):
                            cumulative += probability
                            if draw <= cumulative:
                                selected_candidate_index = candidate_index
                                break
                    else:
                        selected_candidate_index = int(selector_decision["selected_candidate_index"])
                    selected_candidate_probability = float(candidate_probabilities[selected_candidate_index])
                    candidate_metric_table = []
                    for metric in selector_decision["candidate_metrics"]:
                        metric_record = {
                            key: value
                            for key, value in metric.items()
                            if key != "base_feature_vector"
                        }
                        candidate_metric_table.append(metric_record)
                    selector_score_table = list(selector_decision["candidate_scores"])
                    selector_mode = str(selector_decision["selector_mode"])
                    selector_temperature = float(selector_decision["selector_temperature"])
                    selector_exploration_floor = float(selector_decision["selector_exploration_floor"])
                    selector_weights = dict(selector_decision["selector_weights"])
                    selector_value_proxy_source = selector_decision.get("value_proxy_source")
                    selector_value_proxy_model_id = selector_decision.get("value_proxy_model_id")
                    selector_utility_scale_source = selector_decision.get("utility_scale_source")
                    selector_utility_scale_model_id = selector_decision.get("utility_scale_model_id")
                    if args.selection_policy == "caver_k1_guarded":
                        selector_mode = f"{selector_mode}__k1_guarded_mixture_v1"
                        selector_weights["k1_guarded_caver_mixture_mass"] = float(K1_GUARDED_CAVER_MIXTURE_MASS)
                        selected_base_feature_vector = selector_decision["candidate_metrics"][selected_candidate_index][
                            "base_feature_vector"
                        ]
                    else:
                        selected_base_feature_vector = selector_decision["selected_base_feature_vector"]
                    append_selector_history(selector_history, selected_base_feature_vector)
                else:
                    candidate_probabilities[0] = 1.0

                planned_actions = candidate_chunks[selected_candidate_index]
                action_plan.extend(planned_actions)
                selected_candidate_probabilities.append(selected_candidate_probability)
                selected_candidate_indices.append(selected_candidate_index)
                candidate_probability_vectors.append(list(candidate_probabilities))
                safe_candidate_counts.append(safe_candidate_count)
                active_chunk_trace = {
                    "trace_format": "caver_stage0_chunk_trace_v2",
                    "context_id": context.context_id,
                    "source_mode": context.source_mode,
                    "suite_name": context.suite_name,
                    "partition_name": context.partition_name,
                    "proxy_family_id": context.proxy_family_id,
                    "proposal_task": context.proposal_task,
                    "task_id": task_id,
                    "task_name": context.task_name,
                    "episode_idx": episode_idx,
                    "init_state_index": context.init_state_index,
                    "policy_query_index": policy_query_index,
                    "chunk_action_horizon": args.replan_steps,
                    "resize_size": args.resize_size,
                    "prompt": str(task_description),
                    "obs": policy_obs,
                    "candidate_chunks": candidate_chunks,
                    "candidate_policy_aux": candidate_policy_aux_payloads,
                    "candidate_provider_aux": candidate_provider_aux_payloads,
                    "selected_policy_aux": candidate_policy_aux_payloads[selected_candidate_index],
                    "selected_provider_aux": candidate_provider_aux_payloads[selected_candidate_index],
                    "stage0_progress_logging": bool(args.trace_stage0_progress),
                    "stage0_semantic_state_start": stage0_semantic_state,
                    "stage0_progress_start": (
                        None
                        if stage0_semantic_state is None
                        else stage0_semantic_state.get("progress")
                    ),
                    "stage0_semantic_state_sequence": [],
                    "stage0_progress_sequence": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "terminations": [],
                    "truncations": [],
                    "success_once": [],
                    "next_obs_sequence": [],
                    "budget": budget_record,
                    "selector": {
                        "candidate_count": args.candidate_count,
                        "selection_policy": args.selection_policy,
                        "selector_seed": selector_seed,
                        "selected_candidate_index": selected_candidate_index,
                        "selected_candidate_probability": selected_candidate_probability,
                        "candidate_probabilities": candidate_probabilities,
                        "safe_candidate_mask": safe_candidate_mask,
                        "safe_candidate_indices": safe_candidate_indices,
                        "safe_candidate_count": safe_candidate_count,
                        "candidate_safety_reason_codes": candidate_safety_reason_codes,
                        "candidate_action_lengths": candidate_action_lengths,
                        "selector_mode": selector_mode,
                        "selector_temperature": selector_temperature,
                        "selector_exploration_floor": selector_exploration_floor,
                        "value_proxy_source": selector_value_proxy_source,
                        "value_proxy_model_id": selector_value_proxy_model_id,
                        "utility_scale_source": selector_utility_scale_source,
                        "utility_scale_model_id": selector_utility_scale_model_id,
                        "selector_weights": selector_weights,
                        "candidate_scores": selector_score_table,
                        "candidate_metric_table": candidate_metric_table,
                        "history_size": len(selector_history),
                        "provider_mode": args.provider_mode,
                    },
                }

            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, reward, terminated, _ = env.step(action.tolist())
            policy_steps += 1
            truncated = policy_steps >= max_steps and not terminated
            done = bool(terminated or truncated)
            next_policy_obs = extract_policy_observation(obs, resize_size=args.resize_size)
            next_stage0_semantic_state = (
                extract_stage0_semantic_state(
                    obs,
                    env=env,
                    context=context,
                    task_description=str(task_description),
                )
                if args.trace_stage0_progress
                else None
            )
            if active_chunk_trace is not None:
                active_chunk_trace["actions"].append(action)
                active_chunk_trace["rewards"].append(float(reward))
                active_chunk_trace["dones"].append(done)
                active_chunk_trace["terminations"].append(bool(terminated))
                active_chunk_trace["truncations"].append(bool(truncated))
                active_chunk_trace["success_once"].append(bool(terminated))
                active_chunk_trace["next_obs_sequence"].append(next_policy_obs)
                if next_stage0_semantic_state is not None:
                    active_chunk_trace["stage0_semantic_state_sequence"].append(next_stage0_semantic_state)
                    active_chunk_trace["stage0_progress_sequence"].append(
                        next_stage0_semantic_state.get("progress")
                    )
            if done:
                success = bool(terminated)
                flush_chunk_trace("terminated" if terminated else "truncated_horizon")
                break
            if not action_plan:
                flush_chunk_trace("chunk_exhausted")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logging.exception("episode failed on task %s episode %s", task_id, episode_idx)
            flush_chunk_trace("error", trace_error=error)
            break

    if error is None and active_chunk_trace is not None:
        flush_chunk_trace("episode_end")

    if demo_trace_handle is not None and args.demo_trace_write_policy == "success_only" and success:
        for demo_record in context_demo_chunk_traces:
            write_jsonl_record(demo_trace_handle, demo_record)

    video_path = maybe_save_video(video_dir, task_description, task_id, episode_idx, success, replay_images)
    return {
        "context_id": context.context_id,
        "source_mode": context.source_mode,
        "suite_name": context.suite_name,
        "partition_name": context.partition_name,
        "proxy_family_id": context.proxy_family_id,
        "proposal_task": context.proposal_task,
        "task_id": task_id,
        "task_name": context.task_name,
        "episode_idx": episode_idx,
        "init_state_index": context.init_state_index,
        "success": success,
        "policy_steps": policy_steps,
        "duration_sec": time.time() - episode_started,
        "video_path": video_path,
        "error": error,
        "chunk_traces_written": chunk_traces_written,
        "budget": budget_record,
        "selector": {
            "candidate_count": args.candidate_count,
            "selection_policy": args.selection_policy,
            "selector_seed": selector_seed,
            "policy_queries": policy_query_index,
            "selected_candidate_indices": selected_candidate_indices,
            "selected_candidate_probabilities": selected_candidate_probabilities,
            "candidate_probability_vectors": candidate_probability_vectors,
            "safe_candidate_counts": safe_candidate_counts,
        },
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.max_tasks is not None and args.max_tasks < 1:
        raise SystemExit("error: --max-tasks must be positive")
    if args.context_offset < 0:
        raise SystemExit("error: --context-offset must be non-negative")
    if args.max_contexts is not None and args.max_contexts < 1:
        raise SystemExit("error: --max-contexts must be positive")
    if args.num_trials_per_task < 1:
        raise SystemExit("error: --num-trials-per-task must be positive")
    if args.candidate_count < 1:
        raise SystemExit("error: --candidate-count must be positive")
    if args.replan_steps < 1:
        raise SystemExit("error: --replan-steps must be positive")
    if args.round_size < 1:
        raise SystemExit("error: --round-size must be positive")
    if args.provider_mode in ("gesim_bundle", "gesim_live_summary") and not args.provider_bundle_root:
        raise SystemExit("error: --provider-bundle-root is required when --provider-mode uses GE-Sim bundles")

    video_dir = Path(args.video_dir).resolve() if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results_path).resolve() if args.results_path else None
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)

    context_log_path = Path(args.context_log_path).resolve() if args.context_log_path else None
    if context_log_path is not None:
        context_log_path.parent.mkdir(parents=True, exist_ok=True)

    transition_trace_path = Path(args.transition_trace_path).resolve() if args.transition_trace_path else None
    if transition_trace_path is not None:
        transition_trace_path.parent.mkdir(parents=True, exist_ok=True)
    demo_trace_path = Path(args.demo_trace_path).resolve() if args.demo_trace_path else None
    if demo_trace_path is not None:
        demo_trace_path.parent.mkdir(parents=True, exist_ok=True)
    provider_bundle_root = Path(args.provider_bundle_root).resolve() if args.provider_bundle_root else None
    if provider_bundle_root is not None:
        provider_bundle_root.mkdir(parents=True, exist_ok=True)
    args.provider_bundle_root = provider_bundle_root
    value_proxy_model_path = Path(args.value_proxy_model_path).resolve() if args.value_proxy_model_path else None
    value_proxy_model = load_value_proxy_model(value_proxy_model_path) if value_proxy_model_path is not None else None
    dr_calibrator_model_path = Path(args.dr_calibrator_model_path).resolve() if args.dr_calibrator_model_path else None
    dr_calibrator_model = (
        load_stagee_dr_calibrator_model(dr_calibrator_model_path) if dr_calibrator_model_path is not None else None
    )
    lvd_selector_model_path = Path(args.lvd_selector_model_path).resolve() if args.lvd_selector_model_path else None
    if args.selection_policy == "caver_lvd" and lvd_selector_model_path is None:
        raise ValueError("--lvd-selector-model-path is required when --selection-policy=caver_lvd")
    lvd_selector_model = (
        load_lvd_selector_model(lvd_selector_model_path) if lvd_selector_model_path is not None else None
    )

    eval_contexts, eval_plan = build_eval_contexts(args)
    selector_seed = args.selector_seed if args.selector_seed is not None else args.seed
    selection_rng = (
        np.random.default_rng(selector_seed)
        if args.selection_policy in {"uniform", "caver_heuristic", "caver_k1_guarded", "caver_lvd"}
        else None
    )
    selector_history = make_selector_history()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()

    logging.info("connected to policy server at %s:%s", args.host, args.port)
    logging.info("server metadata: %s", server_metadata)
    logging.info(
        "starting LIBERO evaluation: mode=%s contexts=%s",
        eval_plan["mode"],
        len(eval_contexts),
    )

    overall_started = time.time()
    total_episodes = 0
    total_successes = 0
    suite_cache: dict[str, Any] = {}
    task_state_cache: dict[tuple[str, int], dict[str, Any]] = {}
    task_episode_counters: collections.Counter[tuple[str, int]] = collections.Counter()
    partition_counters: collections.Counter[str] = collections.Counter()
    context_summaries: list[dict[str, Any]] = []
    grouped_task_summaries: collections.OrderedDict[tuple[str, int], dict[str, Any]] = collections.OrderedDict()
    current_env_key: tuple[str, int] | None = None
    current_env: OffScreenRenderEnv | None = None
    transition_trace_handle: TextIO | None = None
    demo_trace_handle: TextIO | None = None
    gesim_worker_client: GesimPersistentWorkerClient | None = None

    try:
        if (
            args.provider_mode == "gesim_live_summary"
            and args.provider_gesim_execution_mode == "persistent_worker"
            and provider_bundle_root is not None
        ):
            gesim_worker_client = GesimPersistentWorkerClient(provider_bundle_root=provider_bundle_root)
        if transition_trace_path is not None:
            transition_trace_handle = transition_trace_path.open("w", encoding="utf-8")
        if demo_trace_path is not None:
            demo_trace_handle = open_text_maybe_gzip(demo_trace_path, "wt")

        for context_position, context in enumerate(eval_contexts, start=1):
            if context.suite_name not in suite_cache:
                suite_cache[context.suite_name] = benchmark.get_benchmark_dict()[context.suite_name]()
            task_suite = suite_cache[context.suite_name]

            task_key = (context.suite_name, context.task_id)
            if task_key not in task_state_cache:
                task = task_suite.get_task(context.task_id)
                initial_states = task_suite.get_task_init_states(context.task_id)
                task_state_cache[task_key] = {
                    "task": task,
                    "initial_states": initial_states,
                }
            task_state = task_state_cache[task_key]
            task = task_state["task"]
            initial_states = task_state["initial_states"]

            if context.init_state_index < 0 or context.init_state_index >= len(initial_states):
                raise ValueError(
                    f"context {context.context_id} requested init_state_index={context.init_state_index}, "
                    f"but task only has {len(initial_states)} initial states"
                )

            if current_env_key != task_key:
                if current_env is not None:
                    current_env.close()
                current_env, task_description = get_libero_env(
                    task,
                    args.resolution,
                    args.seed,
                    include_frontview=(args.provider_mode in ("gesim_bundle", "gesim_live_summary")),
                )
                current_env_key = task_key
            else:
                task_description = task.language

            partition_label = context.partition_name or "legacy"
            partition_counters[partition_label] += 1
            budget_record = build_budget_record(
                context=context,
                partition_context_index=partition_counters[partition_label],
                round_size=args.round_size,
                count_legacy_contexts_as_online_budget=args.count_legacy_contexts_as_online_budget,
            )
            max_steps = get_max_steps(args, context.suite_name)
            task_episode_counters[task_key] += 1
            episode_idx = task_episode_counters[task_key] - 1

            logging.info(
                "context %s/%s: suite=%s task=%s init_state=%s partition=%s family=%s max_steps=%s",
                context_position,
                len(eval_contexts),
                context.suite_name,
                context.task_id,
                context.init_state_index,
                context.partition_name,
                context.proxy_family_id,
                max_steps,
            )

            summary = run_episode(
                env=current_env,
                client=client,
                initial_state=initial_states[context.init_state_index],
                task_description=task_description,
                task_id=context.task_id,
                episode_idx=episode_idx,
                args=args,
                max_steps=max_steps,
                video_dir=video_dir,
                context=context,
                budget_record=budget_record,
                transition_trace_handle=transition_trace_handle,
                demo_trace_handle=demo_trace_handle,
                selection_rng=selection_rng,
                selector_seed=selector_seed,
                selector_history=selector_history,
            value_proxy_model=value_proxy_model,
            dr_calibrator_model=dr_calibrator_model,
            lvd_selector_model=lvd_selector_model,
            gesim_worker_client=gesim_worker_client,
        )
            context_summaries.append(summary)
            total_episodes += 1
            total_successes += int(summary["success"])

            if task_key not in grouped_task_summaries:
                grouped_task_summaries[task_key] = {
                    "suite_name": context.suite_name,
                    "task_id": context.task_id,
                    "task_name": task.name,
                    "task_description": task_description,
                    "available_initial_states": len(initial_states),
                    "episodes": [],
                }
            grouped_task_summaries[task_key]["episodes"].append(summary)

            logging.info(
                "context %s complete: success=%s policy_steps=%s error=%s online_budget_units=%s",
                context.context_id,
                summary["success"],
                summary["policy_steps"],
                summary["error"],
                summary["budget"]["online_budget_units"],
            )
    finally:
        if gesim_worker_client is not None:
            gesim_worker_client.close()
        if transition_trace_handle is not None:
            transition_trace_handle.close()
        if demo_trace_handle is not None:
            demo_trace_handle.close()
        if current_env is not None:
            current_env.close()

    task_summaries: list[dict[str, Any]] = []
    for task_summary in grouped_task_summaries.values():
        episodes = task_summary["episodes"]
        task_successes = sum(int(summary["success"]) for summary in episodes)
        task_summaries.append(
            {
                "suite_name": task_summary["suite_name"],
                "task_id": task_summary["task_id"],
                "task_name": task_summary["task_name"],
                "task_description": task_summary["task_description"],
                "episodes_run": len(episodes),
                "successes": task_successes,
                "success_rate": (task_successes / len(episodes)) if episodes else 0.0,
                "available_initial_states": task_summary["available_initial_states"],
                "episodes": episodes,
            }
        )

    budget_summary = {
        "context_units_total": len(context_summaries),
        "online_training_budget_units": sum(
            summary["budget"]["online_budget_units"] for summary in context_summaries
        ),
        "safety_abort_units": sum(int(summary["budget"]["safety_abort"]) for summary in context_summaries),
        "per_partition_contexts": dict(partition_counters),
        "round_size": args.round_size,
    }

    results = {
        "task_suite_name": args.task_suite_name if eval_plan["mode"] == "legacy" else None,
        "task_ids": eval_plan.get("task_ids", []),
        "seed": args.seed,
        "server": {
            "host": args.host,
            "port": args.port,
            "metadata": server_metadata,
        },
        "selection": eval_plan,
        "config": {
            "num_trials_per_task": args.num_trials_per_task,
            "count_legacy_contexts_as_online_budget": args.count_legacy_contexts_as_online_budget,
            "candidate_count": args.candidate_count,
            "selection_policy": args.selection_policy,
            "selector_seed": selector_seed,
            "value_proxy_model_path": value_proxy_model_path,
            "value_proxy_model_id": (value_proxy_model.get("model_id") if value_proxy_model is not None else None),
            "dr_calibrator_model_path": dr_calibrator_model_path,
            "dr_calibrator_model_id": (
                dr_calibrator_model.get("model_id") if dr_calibrator_model is not None else None
            ),
            "lvd_selector_model_path": lvd_selector_model_path,
            "lvd_selector_model_id": (
                lvd_selector_model.get("model_id") if lvd_selector_model is not None else None
            ),
            "num_steps_wait": args.num_steps_wait,
            "replan_steps": args.replan_steps,
            "resize_size": args.resize_size,
            "resolution": args.resolution,
            "max_steps_override": args.max_steps,
            "provider_mode": args.provider_mode,
            "provider_bundle_root": provider_bundle_root,
            "provider_gesim_execution_mode": args.provider_gesim_execution_mode,
            "video_dir": video_dir,
            "save_failures_only": args.save_failures_only,
            "context_log_path": context_log_path,
            "transition_trace_path": transition_trace_path,
            "demo_trace_path": demo_trace_path,
            "demo_trace_write_policy": args.demo_trace_write_policy,
            "trace_policy_aux_mode": args.trace_policy_aux_mode,
            "demo_trace_policy_aux_mode": args.demo_trace_policy_aux_mode,
            "trace_next_obs_mode": args.trace_next_obs_mode,
            "trace_observation_mode": args.trace_observation_mode,
        },
        "budget": budget_summary,
        "summary": {
            "episodes_run": total_episodes,
            "successes": total_successes,
            "success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
            "duration_sec": time.time() - overall_started,
            "chunk_traces_written": sum(summary["chunk_traces_written"] for summary in context_summaries),
        },
        "contexts": context_summaries,
        "tasks": task_summaries,
    }

    if context_log_path is not None:
        with context_log_path.open("w", encoding="utf-8") as handle:
            for summary in context_summaries:
                json.dump(summary, handle, sort_keys=True, default=json_default)
                handle.write("\n")
        logging.info("wrote context log to %s", context_log_path)

    if results_path is not None:
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        logging.info("wrote results to %s", results_path)

    if transition_trace_path is not None:
        logging.info("wrote transition traces to %s", transition_trace_path)
    if demo_trace_path is not None:
        logging.info("wrote demo traces to %s", demo_trace_path)

    logging.info(
        "evaluation complete: successes=%s/%s success_rate=%.3f",
        total_successes,
        total_episodes,
        results["summary"]["success_rate"],
    )


if __name__ == "__main__":
    main()
