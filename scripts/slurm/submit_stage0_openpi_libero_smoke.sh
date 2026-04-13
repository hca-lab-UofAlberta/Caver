#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_openpi_libero_smoke.sh [options]

Options:
  --task-suite-name NAME   LIBERO task suite (default: libero_spatial)
  --task-ids IDS           Optional comma-separated task ids
  --max-tasks N            Optional maximum number of tasks
  --trials N               Episodes per task (default: 1)
  --max-steps N            Rollout horizon override (default: 20)
  --replan-steps N         Replan steps for the client (default: 5)
  --libero-gl-backend NAME Simulator GL backend: egl or osmesa (default: egl)
  --partition NAME         Slurm partition (default: gpu-l40s)
  --qos NAME               Slurm QoS (default: normal)
  --gpu-type TYPE          GPU type (default: l40s)
  --time LIMIT             Slurm time limit (default: 02:00:00)
  --cpus COUNT             CPU request (default: 16)
  --mem SIZE               Memory request (default: 96G)
  --seed INTEGER           Evaluation seed (default: 7)
  --config-name NAME       Optional custom OpenPI config
  --pretrained-path PATH   Optional custom checkpoint path
  --dry-run                Generate run scaffolding but do not submit
  -h, --help               Show this message
EOF
}

task_suite_name="libero_spatial"
task_ids=""
max_tasks="1"
trials="1"
max_steps="20"
replan_steps="5"
libero_gl_backend="egl"
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
time_limit="02:00:00"
cpus="16"
mem="96G"
seed="7"
config_name=""
pretrained_path=""
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --task-suite-name)
      task_suite_name="${2:?missing value for --task-suite-name}"
      shift 2
      ;;
    --task-ids)
      task_ids="${2:?missing value for --task-ids}"
      shift 2
      ;;
    --max-tasks)
      max_tasks="${2:?missing value for --max-tasks}"
      shift 2
      ;;
    --trials)
      trials="${2:?missing value for --trials}"
      shift 2
      ;;
    --max-steps)
      max_steps="${2:?missing value for --max-steps}"
      shift 2
      ;;
    --replan-steps)
      replan_steps="${2:?missing value for --replan-steps}"
      shift 2
      ;;
    --libero-gl-backend)
      libero_gl_backend="${2:?missing value for --libero-gl-backend}"
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
    --time)
      time_limit="${2:?missing value for --time}"
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
    --seed)
      seed="${2:?missing value for --seed}"
      shift 2
      ;;
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --pretrained-path)
      pretrained_path="${2:?missing value for --pretrained-path}"
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
    *)
      echo "error: unknown option: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

client_args=(
  --task-suite-name "${task_suite_name}"
  --trials "${trials}"
  --replan-steps "${replan_steps}"
  --max-steps "${max_steps}"
  --libero-gl-backend "${libero_gl_backend}"
  --seed "${seed}"
)

if [ -n "${task_ids}" ]; then
  client_args+=(--task-ids "${task_ids}")
elif [ -n "${max_tasks}" ]; then
  client_args+=(--max-tasks "${max_tasks}")
fi

runtime_args=()
method_name="openpi-native-smoke"
if [ -n "${config_name}" ] || [ -n "${pretrained_path}" ]; then
  if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
    echo "error: custom checkpoint mode requires both --config-name and --pretrained-path" >&2
    exit 1
  fi
  runtime_args+=(--config-name "${config_name}" --pretrained-path "${pretrained_path}")
  method_name="openpi-native-custom"
fi

submit_args=(
  --stage stage0
  --method "${method_name}"
  --task "${task_suite_name}"
  --seed "${seed}"
  --budget 0
  --partition "${partition}"
  --qos "${qos}"
  --gpu-type "${gpu_type}"
  --cpus "${cpus}"
  --mem "${mem}"
  --time "${time_limit}"
)

if ((dry_run)); then
  submit_args+=(--dry-run)
fi

"${CAVER_REPO_ROOT}/scripts/slurm/submit_experiment.sh" \
  "${submit_args[@]}" \
  -- \
  "${CAVER_REPO_ROOT}/scripts/bridge/run_stage0_openpi_libero_smoke.sh" \
  "${client_args[@]}" \
  "${runtime_args[@]}"
