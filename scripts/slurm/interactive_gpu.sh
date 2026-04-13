#!/usr/bin/env bash
set -euo pipefail

_CAVER_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  interactive_gpu.sh [options] -- [command ...]

Options:
  --account ACCOUNT     Slurm account (default: p57098)
  --partition NAME      Partition (default: gpu-l40s)
  --qos NAME            QoS (default: interactive)
  --gpu-type TYPE       GPU type name (default: l40s)
  --gpus COUNT          GPU count (default: 1)
  --cpus COUNT          CPU count (default: 8)
  --mem SIZE            Memory request (default: 32G)
  --time LIMIT          Time limit (default: 02:00:00)
  --job-name NAME       Job name (default: caver-dev)
  --dry-run             Print the command without running it
  -h, --help            Show this message

Any arguments after -- are used as the interactive command.
EOF
}

account="${CAVER_DEFAULT_ACCOUNT}"
partition="gpu-l40s"
qos="interactive"
gpu_type="l40s"
gpus="1"
cpus="8"
mem="32G"
time_limit="02:00:00"
job_name="caver-dev"
dry_run=0

while (($# > 0)); do
  case "$1" in
    --account)
      account="${2:?missing value for --account}"
      shift 2
      ;;
    --partition)
      partition="${2:?missing value for --partition}"
      shift 2
      ;;
    --qos)
      qos="${2:?missing value for --qos}"
      shift 2
      ;;
    --gpu-type)
      gpu_type="${2:?missing value for --gpu-type}"
      shift 2
      ;;
    --gpus)
      gpus="${2:?missing value for --gpus}"
      shift 2
      ;;
    --cpus)
      cpus="${2:?missing value for --cpus}"
      shift 2
      ;;
    --mem)
      mem="${2:?missing value for --mem}"
      shift 2
      ;;
    --time)
      time_limit="${2:?missing value for --time}"
      shift 2
      ;;
    --job-name)
      job_name="${2:?missing value for --job-name}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command srun

if (($# == 0)); then
  command_args=(bash -l)
else
  command_args=("$@")
fi

srun_args=(
  --account="${account}"
  --partition="${partition}"
  --qos="${qos}"
  --job-name="${job_name}"
  --gres="gpu:${gpu_type}:${gpus}"
  --cpus-per-task="${cpus}"
  --mem="${mem}"
  --time="${time_limit}"
  --pty
)

if [ "${dry_run}" = "1" ]; then
  printf "srun"
  printf " %q" "${srun_args[@]}"
  printf " %q" "${command_args[@]}"
  printf "\n"
  exit 0
fi

exec srun "${srun_args[@]}" "${command_args[@]}"

