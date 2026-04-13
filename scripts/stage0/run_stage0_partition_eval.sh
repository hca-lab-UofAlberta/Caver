#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_partition_eval.sh [options]

Options:
  --manifest-path PATH       Stage-0 partition manifest
                             (default: metadata/stage0/libero_stage0_partitions.json)
  --partition-name NAME      One of T_seed_S0, T_train_S0, T_val_S0, T_test_S0
                             (default: T_seed_S0)
  --family-ids IDS           Optional comma-separated proxy family ids
  --max-contexts N           Optional maximum selected contexts
  --context-offset N         Optional manifest context offset (default: 0)
  --round-size N             Online round size for budget annotation (default: 25)
  --libero-gl-backend NAME   Simulator GL backend: egl or osmesa (default: osmesa)
  --seed INTEGER             Evaluation seed (default: 7)
  --host HOST                Policy server host (default: 127.0.0.1)
  --port PORT                Policy server port (default: Slurm-derived loopback port, else 8000)
  --dummy-server             Use the local dummy websocket policy server
  --openpi-native            Use the native OpenPI websocket policy server (default)
  --action-horizon N         Dummy server action horizon (default: 5)
  --action-dim N             Dummy server action dimension (default: 7)
  --num-steps-wait N         Initial wait steps before policy rollout (default: 10)
  --replan-steps N           Replan steps per policy query (default: 5)
  --resize-size N            Image resize edge length (default: 224)
  --resolution N             Simulator camera resolution (default: 256)
  --max-steps N              Optional rollout horizon override
  --config-name NAME         Optional custom OpenPI config
  --pretrained-path PATH     Optional custom OpenPI checkpoint path
  --results-path PATH        Optional summary JSON output path
  --context-log-path PATH    Optional JSONL context ledger path
  --transition-trace-path PATH
                             Optional JSONL chunk-trace path for warm-start conversion
  -h, --help                 Show this message
EOF
}

manifest_path="${CAVER_REPO_ROOT}/metadata/stage0/libero_stage0_partitions.json"
partition_name="T_seed_S0"
family_ids=""
max_contexts=""
context_offset="0"
round_size="25"
libero_gl_backend="osmesa"
seed="7"
host="127.0.0.1"
port=""
server_mode="openpi-native"
action_horizon="5"
action_dim="7"
num_steps_wait="10"
replan_steps="5"
resize_size="224"
resolution="256"
max_steps=""
config_name=""
pretrained_path=""
results_path=""
context_log_path=""
transition_trace_path=""

while (($# > 0)); do
  case "${1}" in
    --manifest-path)
      manifest_path="${2:?missing value for --manifest-path}"
      shift 2
      ;;
    --partition-name)
      partition_name="${2:?missing value for --partition-name}"
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
    --round-size)
      round_size="${2:?missing value for --round-size}"
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
    --host)
      host="${2:?missing value for --host}"
      shift 2
      ;;
    --port)
      port="${2:?missing value for --port}"
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
    --action-horizon)
      action_horizon="${2:?missing value for --action-horizon}"
      shift 2
      ;;
    --action-dim)
      action_dim="${2:?missing value for --action-dim}"
      shift 2
      ;;
    --num-steps-wait)
      num_steps_wait="${2:?missing value for --num-steps-wait}"
      shift 2
      ;;
    --replan-steps)
      replan_steps="${2:?missing value for --replan-steps}"
      shift 2
      ;;
    --resize-size)
      resize_size="${2:?missing value for --resize-size}"
      shift 2
      ;;
    --resolution)
      resolution="${2:?missing value for --resolution}"
      shift 2
      ;;
    --max-steps)
      max_steps="${2:?missing value for --max-steps}"
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

case "${partition_name}" in
  T_seed_S0|T_train_S0|T_val_S0|T_test_S0)
    ;;
  *)
    echo "error: unsupported --partition-name: ${partition_name}" >&2
    exit 1
    ;;
esac

case "${libero_gl_backend}" in
  egl|osmesa)
    ;;
  *)
    echo "error: unsupported --libero-gl-backend: ${libero_gl_backend}" >&2
    exit 1
    ;;
esac

if [ -z "${port}" ]; then
  port="$(caver_default_loopback_port)"
fi

if [ -z "${results_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    results_path="${CAVER_RUN_DIR}/results/stage0_${partition_name}.json"
  else
    results_path="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stage0_${partition_name}.json"
  fi
fi

if [ -z "${context_log_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    context_log_path="${CAVER_RUN_DIR}/results/stage0_${partition_name}_contexts.jsonl"
  else
    context_log_path="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stage0_${partition_name}_contexts.jsonl"
  fi
fi

if [ -z "${transition_trace_path}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    transition_trace_path="${CAVER_RUN_DIR}/results/stage0_${partition_name}_chunks.jsonl"
  fi
fi

ensure_directory "$(dirname -- "${results_path}")"
ensure_directory "$(dirname -- "${context_log_path}")"
if [ -n "${transition_trace_path}" ]; then
  ensure_directory "$(dirname -- "${transition_trace_path}")"
fi
export MUJOCO_GL="${libero_gl_backend}"

server_args=(--host "${host}" --port "${port}")
case "${server_mode}" in
  dummy)
    server_args+=(--dummy-server --action-horizon "${action_horizon}" --action-dim "${action_dim}")
    ;;
  openpi-native)
    server_args+=(--openpi-native)
    if [ -n "${config_name}" ] || [ -n "${pretrained_path}" ]; then
      if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
        echo "error: custom OpenPI native mode requires both --config-name and --pretrained-path" >&2
        exit 1
      fi
      server_args+=(--config-name "${config_name}" --pretrained-path "${pretrained_path}")
    fi
    ;;
  *)
    echo "error: unsupported server mode: ${server_mode}" >&2
    exit 1
    ;;
esac

client_args=(
  --host "${host}"
  --port "${port}"
  --manifest-path "${manifest_path}"
  --partition-name "${partition_name}"
  --context-offset "${context_offset}"
  --round-size "${round_size}"
  --num-steps-wait "${num_steps_wait}"
  --replan-steps "${replan_steps}"
  --resize-size "${resize_size}"
  --resolution "${resolution}"
  --seed "${seed}"
  --results-path "${results_path}"
  --context-log-path "${context_log_path}"
)

if [ -n "${transition_trace_path}" ]; then
  client_args+=(--transition-trace-path "${transition_trace_path}")
fi
if [ -n "${family_ids}" ]; then
  client_args+=(--family-ids "${family_ids}")
fi
if [ -n "${max_contexts}" ]; then
  client_args+=(--max-contexts "${max_contexts}")
fi
if [ -n "${max_steps}" ]; then
  client_args+=(--max-steps "${max_steps}")
fi

exec "${CAVER_REPO_ROOT}/scripts/bridge/run_libero_remote_eval.sh" \
  "${server_args[@]}" \
  -- \
  "${client_args[@]}"
