#!/usr/bin/env bash
set -euo pipefail

_CAVER_BRIDGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_BRIDGE_DIR}/../common.sh"

wait_for_server() {
  local host="$1"
  local port="$2"
  local pid="$3"
  local log_path="$4"
  local attempts="${5:-300}"

  local i
  for ((i = 0; i < attempts; i++)); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "error: policy server exited before accepting connections" >&2
      if [ -f "${log_path}" ]; then
        echo "server log follows:" >&2
        sed -n '1,240p' "${log_path}" >&2 || true
      fi
      return 1
    fi

    if python3 - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    status = sock.connect_ex((host, port))
raise SystemExit(0 if status == 0 else 1)
PY
    then
      return 0
    fi
    sleep 2
  done

  echo "error: timed out waiting for policy server on ${host}:${port}" >&2
  if [ -f "${log_path}" ]; then
    echo "server log follows:" >&2
    sed -n '1,240p' "${log_path}" >&2 || true
  fi
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  run_libero_remote_eval.sh [server options] -- [client options]

Server options:
  --host HOST
  --port PORT
  --dummy-server
  --openpi-native
  --exact-rollout-payload
  --rlinf-config-name NAME
  --exact-action-chunk N
  --exact-no-nft-loss
  --exact-add-value-head
  --exact-value-after-vlm
  --exact-solver-type NAME
  --exact-infer-mode MODE
  --action-horizon N
  --action-dim N
  --config-name NAME
  --pretrained-path PATH
  --num-steps N

Examples:
  scripts/bridge/run_libero_remote_eval.sh \
    --dummy-server \
    -- \
    --task-suite-name libero_spatial \
    --max-tasks 1 \
    --num-trials-per-task 1 \
    --max-steps 20

  scripts/bridge/run_libero_remote_eval.sh \
    --openpi-native \
    -- \
    --task-suite-name libero_spatial \
    --max-tasks 1 \
    --num-trials-per-task 1

  scripts/bridge/run_libero_remote_eval.sh \
    --openpi-native \
    --config-name pi0_libero \
    --pretrained-path /path/to/checkpoint \
    -- \
    --task-suite-name libero_spatial \
    --task-ids 0,1,2
EOF
}

host="127.0.0.1"
port=""
dummy_server=0
openpi_native=0
action_horizon="5"
action_dim="7"
config_name=""
pretrained_path=""
num_steps="5"
exact_rollout_payload=0
rlinf_config_name=""
exact_action_chunk=""
exact_no_nft_loss=0
exact_add_value_head=0
exact_value_after_vlm=0
exact_solver_type="flow_sde"
exact_infer_mode="train"

while (($# > 0)); do
  case "${1}" in
    -h|--help)
      usage
      exit 0
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
      dummy_server=1
      shift
      ;;
    --openpi-native)
      openpi_native=1
      shift
      ;;
    --exact-rollout-payload)
      exact_rollout_payload=1
      shift
      ;;
    --rlinf-config-name)
      rlinf_config_name="${2:?missing value for --rlinf-config-name}"
      shift 2
      ;;
    --exact-action-chunk)
      exact_action_chunk="${2:?missing value for --exact-action-chunk}"
      shift 2
      ;;
    --exact-no-nft-loss)
      exact_no_nft_loss=1
      shift
      ;;
    --exact-add-value-head)
      exact_add_value_head=1
      shift
      ;;
    --exact-value-after-vlm)
      exact_value_after_vlm=1
      shift
      ;;
    --exact-solver-type)
      exact_solver_type="${2:?missing value for --exact-solver-type}"
      shift 2
      ;;
    --exact-infer-mode)
      exact_infer_mode="${2:?missing value for --exact-infer-mode}"
      shift 2
      ;;
    --action-horizon)
      action_horizon="${2:?missing value for --action-horizon}"
      shift 2
      ;;
    --action-dim)
      action_dim="${2:?missing value for --action-dim}"
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
    --num-steps)
      num_steps="${2:?missing value for --num-steps}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "error: unknown option: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ((dummy_server == 0 && openpi_native == 0)); then
  if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
    echo "error: real policy mode requires --config-name and --pretrained-path" >&2
    exit 1
  fi
fi

if [ -z "${port}" ]; then
  port="$(caver_default_loopback_port)"
fi

ensure_directory "${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/bridge"
stamp="$(timestamp_utc)"
server_log="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/bridge/policy_server__${stamp}.log"

cleanup() {
  if [ -n "${server_pid:-}" ] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

server_cmd=(
  python
  "${CAVER_REPO_ROOT}/scripts/bridge/openpi_policy_server.py"
  --host "${host}"
  --port "${port}"
)

if ((dummy_server)); then
  server_cmd+=(
    --dummy
    --action-horizon "${action_horizon}"
    --action-dim "${action_dim}"
  )
elif ((openpi_native)); then
  server_cmd=(
    python
    "${CAVER_REPO_ROOT}/third_party/src/openpi/scripts/serve_policy.py"
    --env LIBERO
    --port "${port}"
  )
  if [ -n "${config_name}" ] || [ -n "${pretrained_path}" ]; then
    if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
      echo "error: custom OpenPI native mode requires both --config-name and --pretrained-path" >&2
      exit 1
    fi
    server_cmd+=(
      policy:checkpoint
      --policy.config "${config_name}"
      --policy.dir "${pretrained_path}"
    )
  fi
else
  server_cmd+=(
    --config-name "${config_name}"
    --pretrained-path "${pretrained_path}"
    --num-steps "${num_steps}"
  )
  if ((exact_rollout_payload)); then
    server_cmd+=(--exact-rollout-payload)
  fi
  if [ -n "${rlinf_config_name}" ]; then
    server_cmd+=(--rlinf-config-name "${rlinf_config_name}")
  fi
  if [ -n "${exact_action_chunk}" ]; then
    server_cmd+=(--exact-action-chunk "${exact_action_chunk}")
  fi
  if ((exact_no_nft_loss)); then
    server_cmd+=(--exact-no-nft-loss)
  fi
  if ((exact_add_value_head)); then
    server_cmd+=(--exact-add-value-head)
  fi
  if ((exact_value_after_vlm)); then
    server_cmd+=(--exact-value-after-vlm)
  fi
  if [ -n "${exact_solver_type}" ]; then
    server_cmd+=(--exact-solver-type "${exact_solver_type}")
  fi
  if [ -n "${exact_infer_mode}" ]; then
    server_cmd+=(--exact-infer-mode "${exact_infer_mode}")
  fi
fi

"${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- "${server_cmd[@]}" >"${server_log}" 2>&1 &
server_pid=$!

echo "policy server log: ${server_log}" >&2

wait_for_server "${host}" "${port}" "${server_pid}" "${server_log}"

"${CAVER_REPO_ROOT}/scripts/env/with_libero_eval.sh" -- \
  python "${CAVER_REPO_ROOT}/scripts/bridge/libero_remote_eval.py" \
  --host "${host}" \
  --port "${port}" \
  "$@"
