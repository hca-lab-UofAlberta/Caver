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
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
COMMON_SH = REPO_ROOT / "scripts" / "common.sh"
USER = os.environ.get("USER", "euijin1")
ACCOUNT = "p57098"
RDSS_ROOT = Path(f"/rdss/{ACCOUNT}/{USER}")
RUN_ROOT = RDSS_ROOT / "caver" / "runs"
SLURM_LOG_ROOT = RDSS_ROOT / "caver" / "logs" / "slurm"
RUNTIME_LOG_ROOT = RDSS_ROOT / "caver" / "runtime_logs"
DEFAULT_MANIFEST = REPO_ROOT / "metadata" / "stage0" / "libero_stage0_partitions.json"
DEFAULT_VALUE_PROXY = (
    REPO_ROOT / "metadata" / "stage0" / "value_proxy" / "stage0_context_success_progress_sq_mlp3head_v2.json"
)
DEFAULT_DR_CALIBRATOR = (
    REPO_ROOT / "metadata" / "stage0" / "calibrator" / "stage0_seed_dr_calibrator_mlp_v2.json"
)
TIMEZONE = ZoneInfo("America/Edmonton")


@dataclass(frozen=True)
class JobSpec:
    label: str
    round_method: str
    budget: int
    seed: int
    selection_policy: str
    admission_policy: str | None
    selector_mode: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the Stage-E parallel SDRE follow-on pack: mainline budget-curve reruns and "
            "minimal attribution ablations, each with dependent held-out post-train evaluation."
        )
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--partition-name", default="T_train_S0")
    parser.add_argument("--round-size", type=int, default=25)
    parser.add_argument("--family-offset", type=int, default=0)
    parser.add_argument("--seeds", default="7,13,29")
    parser.add_argument("--mainline-budgets", default="25,100")
    parser.add_argument("--ablation-budget", type=int, default=50)
    parser.add_argument("--caver-partition", default="gpu-h200")
    parser.add_argument("--caver-gpu-type", default="h200")
    parser.add_argument("--caver-gpus", type=int, default=2)
    parser.add_argument("--real-partition", default="gpu-l40s")
    parser.add_argument("--real-gpu-type", default="l40s")
    parser.add_argument("--real-gpus", type=int, default=1)
    parser.add_argument("--posttrain-partition", default="gpu-h200")
    parser.add_argument("--posttrain-gpu-type", default="h200")
    parser.add_argument("--posttrain-gpus", type=int, default=1)
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--skip-mainline", action="store_true")
    parser.add_argument("--skip-selection-only", action="store_true")
    parser.add_argument("--skip-admission-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional explicit JSON path for the submission log.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional explicit Markdown path for the submission log.",
    )
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    return values


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )


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


def read_manifest_backend_info(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = payload["backend"]
    task_suite_names = backend.get("task_suite_names", [])
    if len(task_suite_names) != 1:
        raise ValueError(f"expected exactly one backend task suite in {path}, found {task_suite_names}")
    task_ids = ",".join(str(value) for value in backend.get("task_ids", []))
    return {
        "backend_task_suite": str(task_suite_names[0]),
        "backend_task_ids": task_ids,
    }


def time_limit_for_round(spec: JobSpec) -> str:
    if spec.round_method == "real_only":
        if spec.budget <= 25:
            return "06:00:00"
        if spec.budget <= 50:
            return "08:00:00"
        return "12:00:00"
    if spec.budget <= 25:
        return "08:00:00"
    if spec.budget <= 50:
        return "12:00:00"
    return "16:00:00"


def time_limit_for_posttrain(spec: JobSpec) -> str:
    if spec.budget <= 25:
        return "08:00:00"
    if spec.budget <= 50:
        return "10:00:00"
    return "12:00:00"


def parse_time_limit(raw: str) -> timedelta:
    day_part = 0
    time_part = raw
    if "-" in raw:
        day_token, time_part = raw.split("-", 1)
        day_part = int(day_token)
    hour_token, minute_token, second_token = time_part.split(":")
    return timedelta(
        days=day_part,
        hours=int(hour_token),
        minutes=int(minute_token),
        seconds=int(second_token),
    )


def format_local_time(value: datetime) -> str:
    return value.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


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
        / f"stagee_followon__{label}__{partition_name.lower()}__budget{budget}__offset{family_offset}__seed{seed}__{stamp}.json"
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


def prepare_round_scaffolding(
    *,
    spec: JobSpec,
    selection_manifest: Path,
    backend_task_suite: str,
    backend_task_ids: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    time_limit = time_limit_for_round(spec)
    if spec.round_method == "real_only":
        command = [
            str(REPO_ROOT / "scripts" / "slurm" / "submit_stage0_real_only_round.sh"),
            "--partition",
            args.real_partition,
            "--qos",
            args.qos,
            "--gpu-type",
            args.real_gpu_type,
            "--gpus",
            str(args.real_gpus),
            "--time",
            time_limit,
            "--manifest-path",
            str(selection_manifest),
            "--partition-name",
            args.partition_name,
            "--max-contexts",
            str(spec.budget),
            "--round-size",
            str(args.round_size),
            "--seed",
            str(spec.seed),
            "--candidate-count",
            "4",
            "--selection-policy",
            spec.selection_policy,
            "--libero-gl-backend",
            "osmesa",
            "--server-mode",
            "openpi-exact",
            "--exact-rollout-payload",
            "--backend-task-suite",
            backend_task_suite,
            "--backend-task-ids",
            backend_task_ids,
            "--dry-run",
        ]
    else:
        command = [
            str(REPO_ROOT / "scripts" / "slurm" / "submit_stage0_caver_lagged_budget.sh"),
            "--partition",
            args.caver_partition,
            "--qos",
            args.qos,
            "--gpu-type",
            args.caver_gpu_type,
            "--gpus",
            str(args.caver_gpus),
            "--time",
            time_limit,
            "--trace-reference-mode",
            "manifest",
            "--manifest-path",
            str(selection_manifest),
            "--partition-name",
            args.partition_name,
            "--max-contexts",
            str(spec.budget),
            "--round-size",
            str(args.round_size),
            "--seed",
            str(spec.seed),
            "--candidate-count",
            "4",
            "--selection-policy",
            spec.selection_policy,
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
            str(spec.selector_mode or "frozen_actionspace_softmax_v1"),
            "--admission-policy",
            str(spec.admission_policy or "success_lcb_v1"),
            "--value-proxy-model-path",
            str(DEFAULT_VALUE_PROXY),
            "--dr-calibrator-model-path",
            str(DEFAULT_DR_CALIBRATOR),
            "--provider-mode",
            "gesim_live_summary",
            "--server-mode",
            "openpi-exact",
            "--exact-rollout-payload",
            "--backend-task-suite",
            backend_task_suite,
            "--backend-task-ids",
            backend_task_ids,
            "--dry-run",
        ]
    completed = run_command(command)
    scaffolding = parse_scaffold_output(completed.stdout)
    scaffolding["time_limit"] = time_limit
    scaffolding["dry_run_command"] = " ".join(command)
    return scaffolding


def submit_job_script(job_script: Path, *, dry_run: bool) -> int | None:
    if dry_run:
        return None
    completed = run_command(["sbatch", str(job_script)])
    return parse_job_id(completed.stdout)


def submit_posttrain(
    *,
    round_results_dir: Path,
    method: str,
    dependency_job_id: int | None,
    time_limit: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> dict[str, Any]:
    command = [
        str(REPO_ROOT / "scripts" / "slurm" / "submit_stage0_posttrain_from_round.sh"),
        "--round-results-dir",
        str(round_results_dir),
        "--partition",
        args.posttrain_partition,
        "--qos",
        args.qos,
        "--gpu-type",
        args.posttrain_gpu_type,
        "--gpus",
        str(args.posttrain_gpus),
        "--time",
        time_limit,
        "--method",
        method,
        "--train-backend",
        "exact_offline_nft",
    ]
    if dependency_job_id is not None:
        command.extend(["--dependency", f"afterok:{dependency_job_id}"])
    if dry_run:
        return {
            "job_id": None,
            "submit_command": " ".join(command),
            "stdout": None,
            "stderr": None,
        }
    completed = run_command(command)
    job_id = parse_job_id(completed.stdout)
    return {
        "job_id": job_id,
        "submit_command": " ".join(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_job_specs(args: argparse.Namespace) -> list[JobSpec]:
    seeds = parse_csv_ints(args.seeds)
    mainline_budgets = parse_csv_ints(args.mainline_budgets)
    specs: list[JobSpec] = []
    if not args.skip_mainline:
        for budget in mainline_budgets:
            for seed in seeds:
                specs.append(
                    JobSpec(
                        label="caver-mainline",
                        round_method="caver",
                        budget=budget,
                        seed=seed,
                        selection_policy="caver_heuristic",
                        admission_policy="success_lcb_v1",
                        selector_mode="frozen_actionspace_softmax_v1",
                    )
                )
                specs.append(
                    JobSpec(
                        label="real-only-mainline",
                        round_method="real_only",
                        budget=budget,
                        seed=seed,
                        selection_policy="uniform",
                        admission_policy=None,
                        selector_mode=None,
                    )
                )
    if not args.skip_selection_only:
        for seed in seeds:
            specs.append(
                JobSpec(
                    label="selection-only",
                    round_method="caver",
                    budget=args.ablation_budget,
                    seed=seed,
                    selection_policy="caver_heuristic",
                    admission_policy="all_executed_nonerror",
                    selector_mode="selection_only_caver_selector_v1",
                )
            )
    if not args.skip_admission_only:
        for seed in seeds:
            specs.append(
                JobSpec(
                    label="admission-only",
                    round_method="caver",
                    budget=args.ablation_budget,
                    seed=seed,
                    selection_policy="uniform",
                    admission_policy="success_lcb_v1",
                    selector_mode="admission_only_uniform_safe_v1",
                )
            )
    return specs


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-E Parallel SDRE Follow-on Pack",
        "",
        f"- submitted_at: `{payload['submitted_at_local']}`",
        f"- dry_run: `{payload['dry_run']}`",
        f"- diagnostics_artifacts:",
        f"  - `logs/runtime/stagee_mainline_family_diagnostics.json`",
        f"  - `logs/runtime/stagee_mainline_family_diagnostics.md`",
        "",
    ]
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in payload["jobs"]:
        by_label.setdefault(record["label"], []).append(record)
    for label in sorted(by_label):
        lines.append(f"## {label}")
        lines.append("")
        for record in sorted(by_label[label], key=lambda item: (item["budget"], item["seed"])):
            lines.append(
                f"- seed `{record['seed']}`, budget `{record['budget']}`:"
            )
            lines.append(
                f"  - round method `{record['round_method']}`"
            )
            lines.append(
                f"  - selection manifest `{record['selection_manifest']}`"
            )
            lines.append(
                f"  - run dir `{record['run_dir']}`"
            )
            lines.append(
                f"  - round job `{record['round_job_id']}` on `{record['round_partition']}` with limit `{record['round_time_limit']}`"
            )
            lines.append(
                f"  - posttrain job `{record['posttrain_job_id']}` on `{record['posttrain_partition']}` with limit `{record['posttrain_time_limit']}`"
            )
            lines.append(
                f"  - upper-bound finish window `{record['round_finish_upper_bound_local']}` -> `{record['posttrain_finish_upper_bound_local']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    specs = build_job_specs(args)
    manifest_path = Path(args.manifest_path).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"error: manifest not found: {manifest_path}")
    if not DEFAULT_VALUE_PROXY.exists():
        raise SystemExit(f"error: missing value proxy artifact: {DEFAULT_VALUE_PROXY}")
    if not DEFAULT_DR_CALIBRATOR.exists():
        raise SystemExit(f"error: missing DR calibrator artifact: {DEFAULT_DR_CALIBRATOR}")

    stamp = datetime.now(tz=TIMEZONE).strftime("%Y%m%dT%H%M%S%z")
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else (REPO_ROOT / "logs" / "runtime" / f"stagee_parallel_followon_pack_{stamp}.json").resolve()
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md is not None
        else (REPO_ROOT / "logs" / "runtime" / f"stagee_parallel_followon_pack_{stamp}.md").resolve()
    )
    ensure_parent(output_json)
    ensure_parent(output_md)

    submitted_at = datetime.now(tz=TIMEZONE)
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        selection_manifest = build_selection_manifest(
            manifest_path=manifest_path,
            partition_name=args.partition_name,
            budget=spec.budget,
            family_offset=args.family_offset,
            round_size=args.round_size,
            label=spec.label,
            seed=spec.seed,
            stamp=stamp,
        )
        backend_info = read_manifest_backend_info(selection_manifest)
        scaffolding = prepare_round_scaffolding(
            spec=spec,
            selection_manifest=selection_manifest,
            backend_task_suite=backend_info["backend_task_suite"],
            backend_task_ids=backend_info["backend_task_ids"],
            args=args,
        )
        job_script = Path(scaffolding["job_script"]).resolve()
        run_dir = Path(scaffolding["run_dir"]).resolve()
        round_job_id = submit_job_script(job_script, dry_run=args.dry_run)
        posttrain_time_limit = time_limit_for_posttrain(spec)
        posttrain_submission = submit_posttrain(
            round_results_dir=run_dir / "results",
            method=("real_only" if spec.round_method == "real_only" else "caver"),
            dependency_job_id=round_job_id,
            time_limit=posttrain_time_limit,
            args=args,
            dry_run=args.dry_run,
        )

        round_delta = parse_time_limit(scaffolding["time_limit"])
        posttrain_delta = parse_time_limit(posttrain_time_limit)
        round_finish_upper_bound = submitted_at + round_delta
        posttrain_finish_upper_bound = round_finish_upper_bound + posttrain_delta

        jobs.append(
            {
                **asdict(spec),
                "selection_manifest": str(selection_manifest),
                "backend_task_suite": backend_info["backend_task_suite"],
                "backend_task_ids": backend_info["backend_task_ids"],
                "run_id": scaffolding["run_id"],
                "run_dir": str(run_dir),
                "job_script": str(job_script),
                "round_job_id": round_job_id,
                "round_partition": args.real_partition if spec.round_method == "real_only" else args.caver_partition,
                "round_gpu_type": args.real_gpu_type if spec.round_method == "real_only" else args.caver_gpu_type,
                "round_gpus": args.real_gpus if spec.round_method == "real_only" else args.caver_gpus,
                "round_time_limit": scaffolding["time_limit"],
                "round_stdout": scaffolding["stdout"],
                "round_stderr": scaffolding["stderr"],
                "posttrain_job_id": posttrain_submission["job_id"],
                "posttrain_partition": args.posttrain_partition,
                "posttrain_gpu_type": args.posttrain_gpu_type,
                "posttrain_gpus": args.posttrain_gpus,
                "posttrain_time_limit": posttrain_time_limit,
                "posttrain_finish_upper_bound_local": format_local_time(posttrain_finish_upper_bound),
                "round_finish_upper_bound_local": format_local_time(round_finish_upper_bound),
                "posttrain_submit_command": posttrain_submission["submit_command"],
            }
        )

    payload = {
        "workflow": "stagee_parallel_followon_pack_v1",
        "submitted_at_local": format_local_time(submitted_at),
        "submitted_at_epoch": int(submitted_at.timestamp()),
        "dry_run": bool(args.dry_run),
        "manifest_path": str(manifest_path),
        "partition_name": args.partition_name,
        "round_size": int(args.round_size),
        "family_offset": int(args.family_offset),
        "jobs": jobs,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(str(output_json))
    print(str(output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
