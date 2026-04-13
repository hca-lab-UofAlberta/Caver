#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_BRIDGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_BRIDGE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_openpi_libero_smoke.sh [options]

Options:
  --task-suite-name NAME   LIBERO task suite (default: libero_spatial)
  --task-ids IDS           Optional comma-separated task ids
  --max-tasks N            Optional maximum number of tasks (default: 1)
  --trials N               Episodes per task (default: 1)
  --max-steps N            Rollout horizon override (default: 20)
  --replan-steps N         Replan steps for the client (default: 5)
  --libero-gl-backend NAME Simulator GL backend: egl or osmesa (default: egl)
  --seed INTEGER           Evaluation seed (default: 7)
  --config-name NAME       Optional custom OpenPI config
  --pretrained-path PATH   Optional custom checkpoint path
  --results-path PATH      Optional explicit results path
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
seed="7"
config_name=""
pretrained_path=""
results_path=""

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
    --results-path)
      results_path="${2:?missing value for --results-path}"
      shift 2
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

case "${libero_gl_backend}" in
  egl|osmesa)
    ;;
  *)
    echo "error: unsupported --libero-gl-backend: ${libero_gl_backend}" >&2
    exit 1
    ;;
esac

if [ -z "${results_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    results_path="${CAVER_RUN_DIR}/results/libero_eval.json"
  else
    results_path="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stage0_openpi_libero_smoke.json"
  fi
fi
ensure_directory "$(dirname -- "${results_path}")"
export MUJOCO_GL="${libero_gl_backend}"

server_args=(--openpi-native)
if [ -n "${config_name}" ] || [ -n "${pretrained_path}" ]; then
  if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
    echo "error: custom checkpoint mode requires both --config-name and --pretrained-path" >&2
    exit 1
  fi
  server_args+=(--config-name "${config_name}" --pretrained-path "${pretrained_path}")
fi

client_args=(
  --task-suite-name "${task_suite_name}"
  --num-trials-per-task "${trials}"
  --replan-steps "${replan_steps}"
  --max-steps "${max_steps}"
  --seed "${seed}"
  --results-path "${results_path}"
)

if [ -n "${task_ids}" ]; then
  client_args+=(--task-ids "${task_ids}")
else
  client_args+=(--max-tasks "${max_tasks}")
fi

exec "${CAVER_REPO_ROOT}/scripts/bridge/run_libero_remote_eval.sh" \
  "${server_args[@]}" \
  -- \
  "${client_args[@]}"
