#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
USER = os.environ.get("USER", "euijin1")
ACCOUNT = "p57098"
RDSS_ROOT = Path(f"/rdss/{ACCOUNT}/{USER}")
RUN_ROOT = Path(os.environ.get("CAVER_STAGEE_RUN_ROOT", str(RDSS_ROOT / "caver" / "runs"))).resolve()
POSTTRAIN_ROOT = Path(
    os.environ.get("CAVER_STAGEE_POSTTRAIN_ROOT", str(RDSS_ROOT / "caver" / "stagee_posttrain"))
).resolve()
SLURM_LOG_ROOT = Path(
    os.environ.get("CAVER_STAGEE_SLURM_LOG_ROOT", str(RDSS_ROOT / "caver" / "logs" / "slurm"))
).resolve()
RUNTIME_LOG_ROOT = Path(
    os.environ.get("CAVER_STAGEE_RUNTIME_LOG_ROOT", str(RDSS_ROOT / "caver" / "runtime_logs"))
).resolve()
DEFAULT_MANIFEST = REPO_ROOT / "metadata" / "stage0" / "libero_stage0_partitions.json"
DEFAULT_VALUE_PROXY = (
    REPO_ROOT / "metadata" / "stage0" / "value_proxy" / "stage0_context_success_progress_sq_mlp3head_v2.json"
)
DEFAULT_DR_CALIBRATOR = (
    REPO_ROOT / "metadata" / "stage0" / "calibrator" / "stage0_seed_dr_calibrator_mlp_v2.json"
)
DEFAULT_LVD_SELECTOR = (
    REPO_ROOT / "metadata" / "stage0" / "lvd_selector" / "stage0_seed_lvd_selector_dr_clipped_mlp_v1.json"
)
DEFAULT_LVD_NO_DR_SELECTOR = (
    REPO_ROOT
    / "metadata"
    / "stage0"
    / "lvd_selector"
    / "stage0_seed_lvd_selector_observed_selected_else_nuisance_mlp_v1.json"
)
TIMEZONE = ZoneInfo("America/Edmonton")
SEEDS = (7, 13, 29)


@dataclass(frozen=True)
class Cell:
    method: str
    budget: int
    seed: int

    @property
    def label(self) -> str:
        labels = {
            "caver": "caver-mainline",
            "real_only": "real-only-mainline",
            "real_only_k1": "vanilla-real-only-k1",
            "selection_only": "selection-only",
            "admission_only": "admission-only",
            "success_only": "success-only",
            "no_dr": "no-dr",
            "no_provider": "no-provider",
            "fasr_progress_ranked": "verified-progress-fasr",
            "uniform_k4": "uniform-real-only-k4",
            "k1_fasr": "k1-fasr",
            "uniform_k4_fasr": "uniform-k4-fasr",
            "caver_lvd": "caver-lvd",
            "caver_lvd_fasr": "caver-lvd-fasr",
            "caver_lvd_no_provider": "caver-lvd-no-provider",
            "caver_lvd_no_dr": "caver-lvd-no-dr",
        }
        return labels[self.method]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit missing Stage-E held-out budget-curve cells as self-contained jobs. "
            "Each job runs the online round and then post-train held-out evaluation inline, "
            "so failed round jobs do not leave stale afterok dependencies in the queue."
        )
    )
    parser.add_argument("--budgets", default="25,100")
    parser.add_argument("--seeds", default="7,13,29")
    parser.add_argument("--methods", default="caver")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--partition-name", default="T_train_S0")
    parser.add_argument("--round-size", type=int, default=25)
    parser.add_argument("--family-offset", type=int, default=0)
    parser.add_argument("--partition", default="gpu-h200")
    parser.add_argument("--gpu-type", default="h200")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="128G")
    parser.add_argument(
        "--total-time-limit",
        default="",
        help="Override the inline round+posttrain Slurm walltime for each submitted cell.",
    )
    parser.add_argument("--force", action="store_true", help="Submit even if a held-out summary already exists.")
    parser.add_argument(
        "--chain-afterany",
        action="store_true",
        help="Submit cells serially with afterany dependencies to avoid concurrent exact-policy startup pressure.",
    )
    parser.add_argument(
        "--initial-dependency",
        default="",
        help="Optional dependency for the first submitted cell, for example afterany:6907.",
    )
    parser.add_argument(
        "--exclusive-node",
        action="store_true",
        help="Request exclusive node allocation for each cell. Use only when serial execution is not enough.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--lvd-selector-model-path",
        default=str(DEFAULT_LVD_SELECTOR),
        help="Seed/frozen LVD selector for caver_lvd, caver_lvd_fasr, and caver_lvd_no_provider.",
    )
    parser.add_argument(
        "--lvd-no-dr-selector-model-path",
        default=str(DEFAULT_LVD_NO_DR_SELECTOR),
        help="Seed/frozen LVD selector for caver_lvd_no_dr.",
    )
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_scaffold_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    required = {"run_id", "run_dir", "manifest", "job_script", "stdout", "stderr"}
    missing = sorted(required.difference(parsed))
    if missing:
        raise ValueError(f"failed to parse dry-run scaffolding output; missing keys: {missing}")
    return parsed


def parse_job_id(output: str) -> int:
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise ValueError(f"failed to parse Slurm job id from output: {output!r}")
    return int(match.group(1))


def build_selection_manifest(
    *,
    manifest_path: Path,
    partition_name: str,
    budget: int,
    family_offset: int,
    round_size: int,
    label: str,
    seed: int,
    stamp: str,
) -> Path:
    output_path = (
        RUNTIME_LOG_ROOT
        / "stagee_manifests"
        / f"stagee_curve__{label}__{partition_name.lower()}__budget{budget}__offset{family_offset}__seed{seed}__{stamp}.json"
    )
    ensure_parent(output_path)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "stagee" / "build_stage0_balanced_manifest.py"),
        "--input-manifest",
        str(manifest_path),
        "--output-manifest",
        str(output_path),
        "--partition-name",
        partition_name,
        "--budget",
        str(budget),
        "--family-offset",
        str(family_offset),
        "--round-size",
        str(round_size),
    ]
    run_command(command)
    return output_path.resolve()


def read_backend_info(selection_manifest: Path) -> dict[str, str]:
    payload = json.loads(selection_manifest.read_text(encoding="utf-8"))
    backend = payload["backend"]
    suites = backend.get("task_suite_names", [])
    if len(suites) != 1:
        raise ValueError(f"expected exactly one backend task suite in {selection_manifest}, found {suites}")
    return {
        "backend_task_suite": str(suites[0]),
        "backend_task_ids": ",".join(str(value) for value in backend.get("task_ids", [])),
    }


def round_time_limit(method: str, budget: int) -> str:
    if method in {"real_only", "real_only_k1", "uniform_k4", "k1_fasr", "uniform_k4_fasr"}:
        if budget <= 25:
            return "06:00:00"
        if budget <= 50:
            return "08:00:00"
        if budget <= 100:
            return "14:00:00"
        return "3-00:00:00"
    if budget <= 25:
        return "1-12:00:00"
    if budget <= 50:
        return "3-00:00:00"
    if budget <= 100:
        return "5-18:00:00"
    return "5-08:00:00"


def posttrain_time_limit(budget: int) -> str:
    if budget <= 25:
        return "10:00:00"
    if budget <= 100:
        return "14:00:00"
    return "1-12:00:00"


def total_time_limit(method: str, budget: int) -> str:
    total = parse_time_limit(round_time_limit(method, budget)) + parse_time_limit(posttrain_time_limit(budget)) + timedelta(hours=1)
    return format_time_limit(total)


def parse_time_limit(raw: str) -> timedelta:
    days = 0
    time_part = raw
    if "-" in raw:
        day_part, time_part = raw.split("-", 1)
        days = int(day_part)
    hours, minutes, seconds = (int(token) for token in time_part.split(":"))
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def format_time_limit(value: timedelta) -> str:
    total_seconds = int(value.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_local_time(value: datetime) -> str:
    return value.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


def has_complete_posttrain(cell: Cell) -> Path | None:
    if cell.method == "caver":
        pattern = f"stagee__caver-lagged__manifest-t_train_s0-all__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "selection_only":
        pattern = f"stagee__caver-lagged__manifest-t_train_s0-all-selection-only__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "admission_only":
        pattern = f"stagee__caver-lagged__manifest-t_train_s0-all-admission-only__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "success_only":
        pattern = f"stagee__caver-lagged__manifest-t_train_s0-all-success-only__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "no_dr":
        pattern = f"stagee__caver-lagged__manifest-t_train_s0-all-no-dr__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "no_provider":
        pattern = f"stagee__caver-lagged__manifest-t_train_s0-all-no-provider__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "fasr_progress_ranked":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-verified-progress-fasr-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "uniform_k4":
        pattern = (
            "stagee__real-only-round__"
            f"manifest-t_train_s0-all-uniform-k4__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "k1_fasr":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-k1-fasr-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "uniform_k4_fasr":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-uniform-k4-fasr-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "caver_lvd":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-caver-lvd-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "caver_lvd_fasr":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-caver-lvd-fasr-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "caver_lvd_no_provider":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-caver-lvd-no-provider-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "caver_lvd_no_dr":
        pattern = (
            "stagee__caver-lagged__"
            f"manifest-t_train_s0-all-caver-lvd-no-dr-n{cell.budget}__"
            f"seed{cell.seed}__budget{cell.budget}__*"
        )
    elif cell.method == "real_only":
        pattern = f"stagee__real-only-round__manifest-t_train_s0-all__seed{cell.seed}__budget{cell.budget}__*"
    elif cell.method == "real_only_k1":
        pattern = f"stagee__real-only-round__manifest-t_train_s0-all-vanilla-k1__seed{cell.seed}__budget{cell.budget}__*"
    else:
        raise ValueError(f"unsupported method: {cell.method}")
    matches = sorted(POSTTRAIN_ROOT.glob(pattern))
    for candidate in reversed(matches):
        summary = candidate / "posttrain_holdout_summary.json"
        if not summary.exists():
            continue
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        heldout = payload.get("heldout", {})
        if "validation" in heldout and "test" in heldout:
            return summary.resolve()
    return None


def prepare_round_scaffolding(cell: Cell, selection_manifest: Path, backend: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    if cell.method in {"real_only", "real_only_k1", "uniform_k4"}:
        candidate_count = "1" if cell.method == "real_only_k1" else "4"
        selection_policy = "first" if cell.method == "real_only_k1" else "uniform"
        experiment_name = "stage0_real_only_budget"
        run_label_suffix = ""
        if cell.method == "real_only_k1":
            experiment_name = "stage0_vanilla_real_only_k1_budget"
            run_label_suffix = "vanilla-k1"
        elif cell.method == "uniform_k4":
            experiment_name = "stage0_uniform_k4_budget"
            run_label_suffix = "uniform-k4"
        command = [
            str(REPO_ROOT / "scripts" / "slurm" / "submit_stage0_real_only_round.sh"),
            "--partition",
            args.partition,
            "--qos",
            args.qos,
            "--gpu-type",
            args.gpu_type,
            "--gpus",
            str(args.gpus),
            "--run-root",
            str(RUN_ROOT),
            "--log-root",
            str(SLURM_LOG_ROOT),
            "--time",
            round_time_limit(cell.method, cell.budget),
            "--manifest-path",
            str(selection_manifest),
            "--partition-name",
            args.partition_name,
            "--max-contexts",
            str(cell.budget),
            "--round-size",
            str(args.round_size),
            "--seed",
            str(cell.seed),
            "--candidate-count",
            candidate_count,
            "--selection-policy",
            selection_policy,
            "--libero-gl-backend",
            "osmesa",
            "--server-mode",
            "openpi-exact",
            "--exact-rollout-payload",
            "--backend-task-suite",
            backend["backend_task_suite"],
            "--backend-task-ids",
            backend["backend_task_ids"],
            "--experiment-name",
            experiment_name,
            "--dry-run",
        ]
        if run_label_suffix:
            command[-1:-1] = ["--run-label-suffix", run_label_suffix]
    else:
        candidate_count = "4"
        demo_trace_write_policy = "success_only"
        use_value_proxy = True
        use_fasr_repair_flags = False
        lvd_selector_model_path: Path | None = None
        lvd_target_source = "dr_clipped"
        if cell.method == "selection_only":
            selection_policy = "caver_heuristic"
            admission_policy = "all_executed_nonerror"
            selector_mode = "selection_only_caver_selector_v1"
            run_label_suffix = "selection-only"
            experiment_name = "stage0_selection_only_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
        elif cell.method == "admission_only":
            selection_policy = "uniform"
            admission_policy = "success_lcb_v1"
            selector_mode = "admission_only_uniform_safe_v1"
            run_label_suffix = "admission-only"
            experiment_name = "stage0_admission_only_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
        elif cell.method == "success_only":
            selection_policy = "caver_heuristic"
            admission_policy = "success_only"
            selector_mode = "success_only_caver_selector_v1"
            run_label_suffix = "success-only"
            experiment_name = "stage0_success_only_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
        elif cell.method == "no_dr":
            selection_policy = "caver_heuristic"
            admission_policy = "success_lcb_v1"
            selector_mode = "no_dr_value_proxy_softmax_v1"
            run_label_suffix = "no-dr"
            experiment_name = "stage0_no_dr_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = False
            disable_lagged_dr = True
        elif cell.method == "no_provider":
            selection_policy = "caver_heuristic"
            admission_policy = "success_lcb_v1"
            selector_mode = "no_provider_caver_selector_v1"
            run_label_suffix = "no-provider"
            experiment_name = "stage0_no_provider_budget"
            provider_mode = "none"
            use_dr_calibrator = True
            disable_lagged_dr = False
        elif cell.method == "k1_fasr":
            candidate_count = "1"
            selection_policy = "first"
            admission_policy = "caver_family_segment_repair"
            selector_mode = "k1_fasr_first_progress_repair_v1"
            run_label_suffix = f"k1-fasr-n{cell.budget}"
            experiment_name = "stage0_k1_fasr_budget"
            provider_mode = "none"
            use_dr_calibrator = False
            disable_lagged_dr = True
            use_value_proxy = False
            demo_trace_write_policy = "all"
            use_fasr_repair_flags = True
        elif cell.method == "uniform_k4_fasr":
            candidate_count = "4"
            selection_policy = "uniform"
            admission_policy = "caver_family_segment_repair"
            selector_mode = "uniform_k4_fasr_progress_repair_v1"
            run_label_suffix = f"uniform-k4-fasr-n{cell.budget}"
            experiment_name = "stage0_uniform_k4_fasr_budget"
            provider_mode = "none"
            use_dr_calibrator = False
            disable_lagged_dr = True
            use_value_proxy = False
            demo_trace_write_policy = "all"
            use_fasr_repair_flags = True
        elif cell.method == "caver_lvd":
            selection_policy = "caver_lvd"
            admission_policy = "success_only"
            selector_mode = "lvd_listwise_softmax_v1"
            run_label_suffix = f"caver-lvd-n{cell.budget}"
            experiment_name = "stage0_caver_lvd_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
            lvd_selector_model_path = Path(args.lvd_selector_model_path).resolve()
        elif cell.method == "caver_lvd_fasr":
            selection_policy = "caver_lvd"
            admission_policy = "caver_family_segment_repair"
            selector_mode = "lvd_listwise_fasr_v1"
            run_label_suffix = f"caver-lvd-fasr-n{cell.budget}"
            experiment_name = "stage0_caver_lvd_fasr_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
            lvd_selector_model_path = Path(args.lvd_selector_model_path).resolve()
            demo_trace_write_policy = "all"
            use_fasr_repair_flags = True
        elif cell.method == "caver_lvd_no_provider":
            selection_policy = "caver_lvd"
            admission_policy = "success_only"
            selector_mode = "lvd_no_provider_listwise_v1"
            run_label_suffix = f"caver-lvd-no-provider-n{cell.budget}"
            experiment_name = "stage0_caver_lvd_no_provider_budget"
            provider_mode = "none"
            use_dr_calibrator = True
            disable_lagged_dr = False
            lvd_selector_model_path = Path(args.lvd_selector_model_path).resolve()
        elif cell.method == "caver_lvd_no_dr":
            selection_policy = "caver_lvd"
            admission_policy = "success_only"
            selector_mode = "lvd_no_dr_listwise_v1"
            run_label_suffix = f"caver-lvd-no-dr-n{cell.budget}"
            experiment_name = "stage0_caver_lvd_no_dr_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = False
            disable_lagged_dr = True
            lvd_selector_model_path = Path(args.lvd_no_dr_selector_model_path).resolve()
            lvd_target_source = "observed_selected_else_nuisance"
        elif cell.method == "fasr_progress_ranked":
            selection_policy = "caver_heuristic"
            admission_policy = "caver_family_segment_repair"
            selector_mode = "k1_guarded_verified_progress_fasr_v1"
            run_label_suffix = f"verified-progress-fasr-n{cell.budget}"
            experiment_name = "stage0_verified_progress_fasr_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
            use_fasr_repair_flags = True
        else:
            selection_policy = "caver_heuristic"
            admission_policy = "success_lcb_v1"
            selector_mode = "frozen_actionspace_softmax_v1"
            run_label_suffix = ""
            experiment_name = "stage0_caver_lagged_budget"
            provider_mode = "gesim_live_summary"
            use_dr_calibrator = True
            disable_lagged_dr = False
        command = [
            str(REPO_ROOT / "scripts" / "slurm" / "submit_stage0_caver_lagged_budget.sh"),
            "--partition",
            args.partition,
            "--qos",
            args.qos,
            "--gpu-type",
            args.gpu_type,
            "--gpus",
            str(args.gpus),
            "--run-root",
            str(RUN_ROOT),
            "--log-root",
            str(SLURM_LOG_ROOT),
            "--time",
            round_time_limit(cell.method, cell.budget),
            "--trace-reference-mode",
            "manifest",
            "--manifest-path",
            str(selection_manifest),
            "--partition-name",
            args.partition_name,
            "--max-contexts",
            str(cell.budget),
            "--round-size",
            str(args.round_size),
            "--seed",
            str(cell.seed),
            "--candidate-count",
            candidate_count,
            "--selection-policy",
            selection_policy,
            "--num-steps-wait",
            "10",
            "--replan-steps",
            "4",
            "--resize-size",
            "224",
            "--resolution",
            "256",
            "--libero-gl-backend",
            "osmesa",
            "--selector-mode",
            selector_mode,
            "--admission-policy",
            admission_policy,
            "--provider-mode",
            provider_mode,
            "--server-mode",
            "openpi-exact",
            "--exact-rollout-payload",
            "--backend-task-suite",
            backend["backend_task_suite"],
            "--backend-task-ids",
            backend["backend_task_ids"],
            "--experiment-name",
            experiment_name,
            "--finalizer-skip-backend-update",
            "--train-envs",
            "1",
            "--eval-envs",
            "1",
            "--dry-run",
        ]
        if use_value_proxy:
            command[command.index("--provider-mode"):command.index("--provider-mode")] = [
                "--value-proxy-model-path",
                str(DEFAULT_VALUE_PROXY),
            ]
        if use_dr_calibrator:
            command[command.index("--provider-mode"):command.index("--provider-mode")] = [
                "--dr-calibrator-model-path",
                str(DEFAULT_DR_CALIBRATOR),
            ]
        if lvd_selector_model_path is not None:
            command[command.index("--provider-mode"):command.index("--provider-mode")] = [
                "--lvd-selector-model-path",
                str(lvd_selector_model_path),
            ]
            command[-1:-1] = ["--lvd-target-source", lvd_target_source]
        if disable_lagged_dr:
            command[-5:-5] = ["--disable-lagged-dr", "--skip-dr-calibrator-fit"]
        if run_label_suffix:
            command[-5:-5] = ["--run-label-suffix", run_label_suffix]
        if demo_trace_write_policy != "success_only":
            command[-5:-5] = ["--demo-trace-write-policy", demo_trace_write_policy]
        if use_fasr_repair_flags:
            command[-1:-1] = [
                "--trace-stage0-progress",
                "--family-min-success-count",
                "2",
                "--rescue-family-ids",
                "drawer_open_proxy,relocate_to_region_proxy,two_object_stack_proxy",
                "--rescue-per-family-count",
                "2",
                "--repair-min-trace-records",
                "1",
                "--repair-max-trace-records",
                "12",
                "--repair-min-progress",
                "0.03",
                "--repair-min-primitive-steps",
                "4",
                "--repair-max-regression",
                "0.10",
            ]
    completed = run_command(command)
    parsed = parse_scaffold_output(completed.stdout)
    parsed["dry_run_command"] = " ".join(shlex.quote(part) for part in command)
    return parsed


def make_inline_job_script(cell: Cell, scaffolding: dict[str, str], args: argparse.Namespace) -> Path:
    run_dir = Path(scaffolding["run_dir"]).resolve()
    script_path = run_dir / "heldout_budget_curve_inline.sbatch"
    stdout = SLURM_LOG_ROOT / f"{run_dir.name}__heldout-curve-%j.out"
    stderr = SLURM_LOG_ROOT / f"{run_dir.name}__heldout-curve-%j.err"
    total_time = args.total_time_limit or total_time_limit(cell.method, cell.budget)
    round_script = Path(scaffolding["job_script"]).resolve()
    posttrain_method = "real_only" if cell.method in {"real_only", "real_only_k1", "uniform_k4"} else "caver"
    summary_name = (
        "real_only_round_summary.json"
        if cell.method in {"real_only", "real_only_k1", "uniform_k4"}
        else "caver_round_summary.json"
    )
    posttrain_line = (
        f"{shlex.quote(str(REPO_ROOT / 'scripts' / 'stagee' / 'run_stage0_posttrain_from_round.sh'))} "
        '--round-results-dir "${CAVER_LOCAL_RUN_DIR}/results" '
        '--artifact-root "${CAVER_LOCAL_POSTTRAIN_DIR}" '
        f"--method {shlex.quote(posttrain_method)} "
        "--train-backend exact_offline_nft "
        "--libero-gl-backend osmesa "
        '--training-log-root "${TMPDIR}/posttrain_training" '
        "--keep-export-node-local "
        "--cleanup-training-log-dir"
    )
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --account={ACCOUNT}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --qos={args.qos}",
        f"#SBATCH --job-name=stagee-curve-{cell.method[:4]}-s{cell.seed}-b{cell.budget}",
        f"#SBATCH --gres=gpu:{args.gpu_type}:{args.gpus}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={total_time}",
        f"#SBATCH --output={stdout}",
        f"#SBATCH --error={stderr}",
    ]
    if args.exclusive_node:
        lines.append("#SBATCH --exclusive")
    lines += [
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
        "export TORCHINDUCTOR_COMPILE_THREADS=1",
        "export TOKENIZERS_PARALLELISM=false",
        f"export CAVER_DEFAULT_RUNTIME_LOG_ROOT={shlex.quote(str(RUNTIME_LOG_ROOT))}",
        'export TMPDIR="${CAVER_NODE_LOCAL_TMP_ROOT:-/tmp/${USER}/caver_stagee_curve}/${SLURM_JOB_ID:-manual}"',
        'export CAVER_OUTPUT_STAGING_ROOT="${TMPDIR}/output_staging"',
        'export CAVER_STAGEE_HEAVY_TRACE_ROOT="${TMPDIR}/heavy_traces"',
        f"export CAVER_DURABLE_RUN_DIR={shlex.quote(str(run_dir))}",
        f"export CAVER_DURABLE_POSTTRAIN_DIR={shlex.quote(str(POSTTRAIN_ROOT / run_dir.name))}",
        'export CAVER_LOCAL_RUN_DIR="${TMPDIR}/cell_run"',
        'export CAVER_LOCAL_POSTTRAIN_DIR="${TMPDIR}/cell_posttrain"',
        'export CAVER_LOCAL_ROUND_SCRIPT="${TMPDIR}/round_job.sbatch"',
        (
            "mkdir -p "
            "\"$CAVER_DEFAULT_RUNTIME_LOG_ROOT\" "
            "\"$CAVER_OUTPUT_STAGING_ROOT\" "
            "\"$CAVER_STAGEE_HEAVY_TRACE_ROOT\" "
            "\"$CAVER_LOCAL_RUN_DIR\" "
            "\"$CAVER_LOCAL_POSTTRAIN_DIR\" "
            "\"$CAVER_DURABLE_RUN_DIR\" "
            "\"$CAVER_DURABLE_POSTTRAIN_DIR\" "
            "\"$TMPDIR\""
        ),
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "echo \"[stagee-curve] started $(date --iso-8601=seconds)\"",
        f"echo \"[stagee-curve] cell method={cell.method} seed={cell.seed} budget={cell.budget}\"",
        f"echo \"[stagee-curve] durable round script {round_script}\"",
        'echo "[stagee-curve] local run dir ${CAVER_LOCAL_RUN_DIR}"',
        f"python3 - {shlex.quote(str(round_script))} \"$CAVER_LOCAL_ROUND_SCRIPT\" \"$CAVER_DURABLE_RUN_DIR\" \"$CAVER_LOCAL_RUN_DIR\" <<'PY'",
        "import pathlib",
        "import sys",
        "",
        "src = pathlib.Path(sys.argv[1])",
        "dst = pathlib.Path(sys.argv[2])",
        "durable = sys.argv[3]",
        "local = sys.argv[4]",
        "text = src.read_text(encoding='utf-8').replace(durable, local)",
        "dst.write_text(text, encoding='utf-8')",
        "PY",
        'chmod +x "$CAVER_LOCAL_ROUND_SCRIPT"',
        'bash "$CAVER_LOCAL_ROUND_SCRIPT"',
        "echo \"[stagee-curve] round complete $(date --iso-8601=seconds)\"",
        f'[ -f "${{CAVER_LOCAL_RUN_DIR}}/results/{summary_name}" ]',
        "echo \"[stagee-curve] posttrain start $(date --iso-8601=seconds)\"",
        posttrain_line,
        "echo \"[stagee-curve] copying compact artifacts $(date --iso-8601=seconds)\"",
        "python3 - \"$CAVER_LOCAL_RUN_DIR\" \"$CAVER_DURABLE_RUN_DIR\" \"$CAVER_LOCAL_POSTTRAIN_DIR\" \"$CAVER_DURABLE_POSTTRAIN_DIR\" <<'PY'",
        "import os",
        "import pathlib",
        "import shutil",
        "import sys",
        "import time",
        "",
        "local_run = pathlib.Path(sys.argv[1]).resolve()",
        "durable_run = pathlib.Path(sys.argv[2]).resolve()",
        "local_post = pathlib.Path(sys.argv[3]).resolve()",
        "durable_post = pathlib.Path(sys.argv[4]).resolve()",
        "",
        "run_names = {",
        "    'caver_admission_summary.json',",
        "    'caver_dr_candidate_dataset.summary.json',",
        "    'caver_lagged_dr_calibrator.summary.json',",
        "    'caver_online_eval.json',",
        "    'caver_round_demo.summary.json',",
        "    'caver_round_summary.json',",
        "    'caver_selector_summary.json',",
        "    'lagged_round_chain.summary.json',",
        "    'real_only_online_eval.json',",
        "    'real_only_round_demo.summary.json',",
        "    'real_only_round_summary.json',",
        "}",
        "post_names = {",
        "    'posttrain_checkpoint_export.summary.json',",
        "    'posttrain_eval_T_test_S0.json',",
        "    'posttrain_eval_T_val_S0.json',",
        "    'posttrain_exact_rollout_batch.summary.json',",
        "    'posttrain_holdout_summary.json',",
        "}",
        "",
        "def byte_copy(src: pathlib.Path, dst: pathlib.Path) -> None:",
        "    dst.parent.mkdir(parents=True, exist_ok=True)",
        "    tmp = dst.with_name(f'{dst.name}.tmp.{os.getpid()}')",
        "    for attempt in range(1, 4):",
        "        try:",
        "            try:",
        "                tmp.unlink()",
        "            except FileNotFoundError:",
        "                pass",
        "            with src.open('rb') as fsrc, tmp.open('wb') as fdst:",
        "                shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)",
        "            tmp.replace(dst)",
        "            return",
        "        except OSError as exc:",
        "            try:",
        "                tmp.unlink()",
        "            except OSError:",
        "                pass",
        "            if attempt == 3:",
        "                raise",
        "            print(f'copy retry {attempt}/3 for {dst}: {exc}', file=sys.stderr)",
        "            time.sleep(5 * attempt)",
        "",
        "def copy_selected(src_root: pathlib.Path, dst_root: pathlib.Path, names: set[str]) -> int:",
        "    if not src_root.exists():",
        "        return 0",
        "    copied = 0",
        "    for src in src_root.rglob('*'):",
        "        if not src.is_file() or src.name not in names:",
        "            continue",
        "        rel = src.relative_to(src_root)",
        "        dst = dst_root / rel",
        "        byte_copy(src, dst)",
        "        copied += 1",
        "    return copied",
        "",
        "copied_run = copy_selected(local_run / 'results', durable_run / 'results', run_names)",
        "copied_post = copy_selected(local_post, durable_post, post_names)",
        "print(f'copied compact artifacts: run={copied_run} posttrain={copied_post}')",
        "PY",
        "echo \"[stagee-curve] complete $(date --iso-8601=seconds)\"",
        "",
    ]
    script_path.write_text("\n".join(lines), encoding="utf-8")
    return script_path.resolve()


def submit_script(script_path: Path, dry_run: bool, dependency: str = "") -> int | None:
    if dry_run:
        return None
    command = ["sbatch"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(script_path))
    completed = run_command(command)
    return parse_job_id(completed.stdout)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-E Held-out Budget Curve Submission",
        "",
        f"- submitted_at: `{payload['submitted_at_local']}`",
        f"- dry_run: `{payload['dry_run']}`",
        f"- dependency_policy: `{payload['dependency_policy']}`",
        f"- chain_afterany: `{payload['chain_afterany']}`",
        f"- initial_dependency: `{payload['initial_dependency']}`",
        f"- note: `{payload['note']}`",
        "",
        "| Method | Budget | Seed | Status | Job | Dependency | Time limit | Run dir |",
        "|---|---:|---:|---|---:|---|---|---|",
    ]
    for record in payload["cells"]:
        lines.append(
            "| {method} | {budget} | {seed} | {status} | {job_id} | {dependency} | {time_limit} | `{run_dir}` |".format(
                method=record["method"],
                budget=record["budget"],
                seed=record["seed"],
                status=record["status"],
                job_id=record.get("job_id") or "-",
                dependency=record.get("dependency") or "-",
                time_limit=record.get("time_limit") or "-",
                run_dir=record.get("run_dir") or "-",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    budgets = parse_csv_ints(args.budgets)
    seeds = parse_csv_ints(args.seeds)
    methods = parse_csv_strings(args.methods)
    supported_methods = {
        "caver",
        "real_only",
        "real_only_k1",
        "selection_only",
        "admission_only",
        "success_only",
        "no_dr",
        "no_provider",
        "fasr_progress_ranked",
        "uniform_k4",
        "k1_fasr",
        "uniform_k4_fasr",
        "caver_lvd",
        "caver_lvd_fasr",
        "caver_lvd_no_provider",
        "caver_lvd_no_dr",
    }
    bad_methods = sorted(set(methods).difference(supported_methods))
    if bad_methods:
        raise SystemExit(f"error: unsupported methods: {bad_methods}")

    manifest_path = Path(args.manifest_path).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"error: manifest not found: {manifest_path}")
    if not DEFAULT_VALUE_PROXY.exists():
        raise SystemExit(f"error: value proxy not found: {DEFAULT_VALUE_PROXY}")
    if not DEFAULT_DR_CALIBRATOR.exists():
        raise SystemExit(f"error: DR calibrator not found: {DEFAULT_DR_CALIBRATOR}")
    needs_lvd = bool({"caver_lvd", "caver_lvd_fasr", "caver_lvd_no_provider"}.intersection(methods))
    needs_lvd_no_dr = "caver_lvd_no_dr" in methods
    lvd_selector_model_path = Path(args.lvd_selector_model_path).resolve()
    lvd_no_dr_selector_model_path = Path(args.lvd_no_dr_selector_model_path).resolve()
    if needs_lvd and not lvd_selector_model_path.exists():
        raise SystemExit(f"error: seed LVD selector not found: {lvd_selector_model_path}")
    if needs_lvd_no_dr and not lvd_no_dr_selector_model_path.exists():
        raise SystemExit(f"error: seed no-DR LVD selector not found: {lvd_no_dr_selector_model_path}")

    stamp = datetime.now(tz=TIMEZONE).strftime("%Y%m%dT%H%M%S%z")
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else (REPO_ROOT / "logs" / "runtime" / f"stagee_heldout_budget_curve_{stamp}.json").resolve()
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else (REPO_ROOT / "logs" / "runtime" / f"stagee_heldout_budget_curve_{stamp}.md").resolve()
    )
    ensure_parent(output_json)
    ensure_parent(output_md)

    submitted_at = datetime.now(tz=TIMEZONE)
    records: list[dict[str, Any]] = []
    next_dependency = args.initial_dependency.strip()
    for budget in budgets:
        for method in methods:
            for seed in seeds:
                cell = Cell(method=method, budget=budget, seed=seed)
                existing = has_complete_posttrain(cell)
                if existing is not None and not args.force:
                    records.append(
                        {
                            **asdict(cell),
                            "status": "skipped_existing",
                            "existing_summary": str(existing),
                        }
                    )
                    continue
                selection_manifest = build_selection_manifest(
                    manifest_path=manifest_path,
                    partition_name=args.partition_name,
                    budget=budget,
                    family_offset=args.family_offset,
                    round_size=args.round_size,
                    label=cell.label,
                    seed=seed,
                    stamp=stamp,
                )
                backend = read_backend_info(selection_manifest)
                scaffolding = prepare_round_scaffolding(cell, selection_manifest, backend, args)
                inline_script = make_inline_job_script(cell, scaffolding, args)
                dependency = next_dependency if args.chain_afterany and next_dependency else ""
                job_id = submit_script(inline_script, args.dry_run, dependency)
                if args.chain_afterany and job_id is not None:
                    next_dependency = f"afterany:{job_id}"
                records.append(
                    {
                        **asdict(cell),
                        "status": "submitted" if job_id is not None else "dry_run",
                        "job_id": job_id,
                        "dependency": dependency,
                        "time_limit": args.total_time_limit or total_time_limit(method, budget),
                        "selection_manifest": str(selection_manifest),
                        "run_dir": scaffolding["run_dir"],
                        "round_job_script": scaffolding["job_script"],
                        "inline_job_script": str(inline_script),
                        "round_dry_run_command": scaffolding["dry_run_command"],
                    }
                )

    payload = {
        "workflow": "stagee_heldout_budget_curve_submission_v1",
        "submitted_at_local": format_local_time(submitted_at),
        "submitted_at_epoch": int(submitted_at.timestamp()),
        "dry_run": bool(args.dry_run),
        "dependency_policy": (
            "inline_round_then_posttrain_afterany_serial"
            if args.chain_afterany
            else "inline_round_then_posttrain_no_afterok_children"
        ),
        "chain_afterany": bool(args.chain_afterany),
        "initial_dependency": args.initial_dependency.strip(),
        "exclusive_node": bool(args.exclusive_node),
        "note": (
            "Each submitted job runs the round and post-train held-out evaluation inline. "
            "When --chain-afterany is used, cells run serially without stale afterok dependencies."
        ),
        "partition": args.partition,
        "gpu_type": args.gpu_type,
        "gpus": args.gpus,
        "cells": records,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
