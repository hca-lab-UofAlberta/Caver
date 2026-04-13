#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import socket
import subprocess
from datetime import datetime, timezone


def run_git(repo_root: pathlib.Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def load_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a CAVER run manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--qos", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--gpus", required=True, type=int)
    parser.add_argument("--cpus-per-task", required=True, type=int)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--time-limit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-script", required=True)
    parser.add_argument("--slurm-stdout", required=True)
    parser.add_argument("--slurm-stderr", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args()

    template_path = pathlib.Path(args.template)
    output_path = pathlib.Path(args.output)
    repo_root = template_path.parent.parent.resolve()
    manifest = load_json(template_path)

    manifest["run"]["id"] = args.run_id
    manifest["run"]["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["run"]["status"] = "created"

    manifest["study"]["stage"] = args.stage
    manifest["study"]["method"] = args.method
    manifest["study"]["task"] = args.task
    manifest["study"]["seed"] = args.seed
    manifest["study"]["budget"] = args.budget

    manifest["cluster"]["account"] = args.account
    manifest["cluster"]["partition"] = args.partition
    manifest["cluster"]["qos"] = args.qos
    manifest["cluster"]["gpu_type"] = args.gpu_type
    manifest["cluster"]["gpus"] = args.gpus
    manifest["cluster"]["cpus_per_task"] = args.cpus_per_task
    manifest["cluster"]["memory"] = args.memory
    manifest["cluster"]["time_limit"] = args.time_limit

    manifest["code"]["repo_root"] = str(repo_root)
    manifest["code"]["git_commit"] = run_git(repo_root, "rev-parse", "HEAD")
    manifest["code"]["git_branch"] = run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")

    manifest["paths"]["run_dir"] = args.run_dir
    manifest["paths"]["job_script"] = args.job_script
    manifest["paths"]["slurm_stdout"] = args.slurm_stdout
    manifest["paths"]["slurm_stderr"] = args.slurm_stderr

    manifest["slurm"]["job_name"] = args.job_name
    manifest["slurm"]["job_id"] = os.environ.get("SLURM_JOB_ID", "")
    manifest["slurm"]["submission_host"] = socket.gethostname()
    manifest["slurm"]["submission_user"] = os.environ.get("USER", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
      json.dump(manifest, handle, indent=2, sort_keys=False)
      handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

