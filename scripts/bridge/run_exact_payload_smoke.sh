#!/usr/bin/env bash
set -euo pipefail

_CAVER_BRIDGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_BRIDGE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_exact_payload_smoke.sh [options]

Options:
  --host HOST
  --port PORT
  --config-name NAME
  --pretrained-path PATH
  --rlinf-config-name NAME
  --action-dim N
  --num-steps N
  --exact-action-chunk N
  --exact-solver-type NAME
  --exact-infer-mode MODE
  --exact-no-nft-loss
  --exact-add-value-head
  --exact-value-after-vlm
  -h, --help
EOF
}

host="127.0.0.1"
port="${CAVER_EXACT_SMOKE_PORT:-24322}"
config_name="pi05_libero"
pretrained_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
rlinf_config_name="libero_goal_ppo_openpi_pi05"
action_dim="7"
num_steps="5"
exact_action_chunk="5"
exact_solver_type="flow_sde"
exact_infer_mode="train"
exact_no_nft_loss=0
exact_add_value_head=0
exact_value_after_vlm=0

while (($# > 0)); do
  case "${1}" in
    --host) host="${2:?missing value for --host}"; shift 2 ;;
    --port) port="${2:?missing value for --port}"; shift 2 ;;
    --config-name) config_name="${2:?missing value for --config-name}"; shift 2 ;;
    --pretrained-path) pretrained_path="${2:?missing value for --pretrained-path}"; shift 2 ;;
    --rlinf-config-name) rlinf_config_name="${2:?missing value for --rlinf-config-name}"; shift 2 ;;
    --action-dim) action_dim="${2:?missing value for --action-dim}"; shift 2 ;;
    --num-steps) num_steps="${2:?missing value for --num-steps}"; shift 2 ;;
    --exact-action-chunk) exact_action_chunk="${2:?missing value for --exact-action-chunk}"; shift 2 ;;
    --exact-solver-type) exact_solver_type="${2:?missing value for --exact-solver-type}"; shift 2 ;;
    --exact-infer-mode) exact_infer_mode="${2:?missing value for --exact-infer-mode}"; shift 2 ;;
    --exact-no-nft-loss) exact_no_nft_loss=1; shift ;;
    --exact-add-value-head) exact_add_value_head=1; shift ;;
    --exact-value-after-vlm) exact_value_after_vlm=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: ${1}" >&2; usage >&2; exit 1 ;;
  esac
done

server_log="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/bridge/exact_payload_smoke_server.log"
ensure_directory "$(dirname -- "${server_log}")"
rm -f -- "${server_log}"

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
  --config-name "${config_name}"
  --pretrained-path "${pretrained_path}"
  --num-steps "${num_steps}"
  --action-dim "${action_dim}"
  --exact-rollout-payload
  --rlinf-config-name "${rlinf_config_name}"
  --exact-action-chunk "${exact_action_chunk}"
  --exact-solver-type "${exact_solver_type}"
  --exact-infer-mode "${exact_infer_mode}"
)

if ((exact_no_nft_loss)); then
  server_cmd+=(--exact-no-nft-loss)
fi
if ((exact_add_value_head)); then
  server_cmd+=(--exact-add-value-head)
fi
if ((exact_value_after_vlm)); then
  server_cmd+=(--exact-value-after-vlm)
fi

export PYTHONUNBUFFERED=1
"${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- "${server_cmd[@]}" >"${server_log}" 2>&1 &
server_pid=$!

for _attempt in $(seq 1 180); do
  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    cat "${server_log}" >&2 || true
    echo "error: exact payload smoke server exited before opening ${host}:${port}" >&2
    exit 1
  fi
  if python3 - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
  then
    break
  fi
  sleep 2
done

"${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- \
  python "${CAVER_REPO_ROOT}/scripts/bridge/openpi_client_smoke.py" \
  --host "${host}" \
  --port "${port}"

tail -n 20 "${server_log}" || true
