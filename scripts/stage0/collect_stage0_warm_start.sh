#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  collect_stage0_warm_start.sh [options]

This is a thin wrapper over run_stage0_partition_eval.sh for the fixed
Stage-0 seed partition T_seed_S0.

Options:
  --manifest-path PATH       Stage-0 partition manifest
                             (default: metadata/stage0/libero_stage0_partitions.json)
  --family-ids IDS           Optional comma-separated family subset
  --max-contexts N           Optional context cap (default: all 120 seed contexts)
  --context-offset N         Optional manifest offset (default: 0)
  --libero-gl-backend NAME   egl or osmesa (default: osmesa)
  --seed INTEGER             Evaluation seed (default: 7)
  --dummy-server             Use the dummy websocket policy server
  --openpi-native            Use the native OpenPI policy server (default)
  --config-name NAME         Optional custom OpenPI config
  --pretrained-path PATH     Optional custom OpenPI checkpoint path
  --max-steps N              Optional rollout horizon override
  --replan-steps N           Replan steps per policy query (default: 5)
  --num-steps-wait N         Initial wait steps before policy rollout (default: 10)
  --results-path PATH        Optional warm-start summary JSON path
  --context-log-path PATH    Optional warm-start JSONL ledger path
  --transition-trace-path PATH
                             Optional warm-start chunk-trace JSONL path
  -h, --help                 Show this message
EOF
}

manifest_path="${CAVER_REPO_ROOT}/metadata/stage0/libero_stage0_partitions.json"
family_ids=""
max_contexts=""
context_offset="0"
libero_gl_backend="osmesa"
seed="7"
server_mode="openpi-native"
config_name=""
pretrained_path=""
max_steps=""
replan_steps="5"
num_steps_wait="10"
results_path=""
context_log_path=""
transition_trace_path=""

while (($# > 0)); do
  case "${1}" in
    --manifest-path)
      manifest_path="${2:?missing value for --manifest-path}"
      shift 2
      ;;
    --family-ids)
      family_ids="${2:?missing value for --family-ids}"
      shift 2
      ;;
    --max-contexts)
      max_contexts="${2:?missing value for --max-contexts}"
      shift 2
      ;;
    --context-offset)
      context_offset="${2:?missing value for --context-offset}"
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
    --dummy-server)
      server_mode="dummy"
      shift
      ;;
    --openpi-native)
      server_mode="openpi-native"
      shift
      ;;
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --pretrained-path)
      pretrained_path="${2:?missing value for --pretrained-path}"
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
    --num-steps-wait)
      num_steps_wait="${2:?missing value for --num-steps-wait}"
      shift 2
      ;;
    --results-path)
      results_path="${2:?missing value for --results-path}"
      shift 2
      ;;
    --context-log-path)
      context_log_path="${2:?missing value for --context-log-path}"
      shift 2
      ;;
    --transition-trace-path)
      transition_trace_path="${2:?missing value for --transition-trace-path}"
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

if [ -z "${results_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    results_path="${CAVER_RUN_DIR}/results/stage0_seed_warm_start.json"
  else
    results_path="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stage0_seed_warm_start.json"
  fi
fi

if [ -z "${context_log_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    context_log_path="${CAVER_RUN_DIR}/results/stage0_seed_warm_start_contexts.jsonl"
  else
    context_log_path="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stage0_seed_warm_start_contexts.jsonl"
  fi
fi

if [ -z "${transition_trace_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    transition_trace_path="${CAVER_RUN_DIR}/results/stage0_seed_warm_start_chunks.jsonl"
  fi
fi

run_args=(
  --manifest-path "${manifest_path}"
  --partition-name T_seed_S0
  --context-offset "${context_offset}"
  --libero-gl-backend "${libero_gl_backend}"
  --seed "${seed}"
  --replan-steps "${replan_steps}"
  --num-steps-wait "${num_steps_wait}"
  --results-path "${results_path}"
  --context-log-path "${context_log_path}"
)

if [ -n "${transition_trace_path}" ]; then
  run_args+=(--transition-trace-path "${transition_trace_path}")
fi
if [ -n "${family_ids}" ]; then
  run_args+=(--family-ids "${family_ids}")
fi
if [ -n "${max_contexts}" ]; then
  run_args+=(--max-contexts "${max_contexts}")
fi
if [ -n "${max_steps}" ]; then
  run_args+=(--max-steps "${max_steps}")
fi

case "${server_mode}" in
  dummy)
    run_args+=(--dummy-server)
    ;;
  openpi-native)
    run_args+=(--openpi-native)
    if [ -n "${config_name}" ] || [ -n "${pretrained_path}" ]; then
      if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
        echo "error: custom OpenPI native mode requires both --config-name and --pretrained-path" >&2
        exit 1
      fi
      run_args+=(--config-name "${config_name}" --pretrained-path "${pretrained_path}")
    fi
    ;;
  *)
    echo "error: unsupported server mode: ${server_mode}" >&2
    exit 1
    ;;
esac

exec "${_CAVER_STAGE0_DIR}/run_stage0_partition_eval.sh" "${run_args[@]}"
