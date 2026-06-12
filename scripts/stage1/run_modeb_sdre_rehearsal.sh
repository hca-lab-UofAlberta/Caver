#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE1_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE1_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_modeb_sdre_rehearsal.sh [options]

Options:
  --host HOST
  --port PORT
  --output-root DIR
  --config-name NAME
  --pretrained-path PATH
  --rlinf-config-name NAME
  --num-steps N
  --exact-action-chunk N
  --task-id NAME
  --warmup N
  --requests N
  --observed-gripper-m FLOAT
  --record-policy-arrays
  -h, --help
EOF
}

host="127.0.0.1"
port="$(caver_default_loopback_port)"
output_root="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stage1_modeb_rehearsal/${SLURM_JOB_ID:-manual}"
config_name="pi05_libero"
pretrained_path="${CAVER_REPO_ROOT}/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
rlinf_config_name="libero_goal_ppo_openpi_pi05"
num_steps="4"
exact_action_chunk="4"
task_id="block_to_tray"
warmup="5"
requests="20"
observed_gripper_m="0.0"
record_policy_arrays=0

while (($# > 0)); do
  case "${1}" in
    --host) host="${2:?missing value for --host}"; shift 2 ;;
    --port) port="${2:?missing value for --port}"; shift 2 ;;
    --output-root) output_root="${2:?missing value for --output-root}"; shift 2 ;;
    --config-name) config_name="${2:?missing value for --config-name}"; shift 2 ;;
    --pretrained-path) pretrained_path="${2:?missing value for --pretrained-path}"; shift 2 ;;
    --rlinf-config-name) rlinf_config_name="${2:?missing value for --rlinf-config-name}"; shift 2 ;;
    --num-steps) num_steps="${2:?missing value for --num-steps}"; shift 2 ;;
    --exact-action-chunk) exact_action_chunk="${2:?missing value for --exact-action-chunk}"; shift 2 ;;
    --task-id) task_id="${2:?missing value for --task-id}"; shift 2 ;;
    --warmup) warmup="${2:?missing value for --warmup}"; shift 2 ;;
    --requests) requests="${2:?missing value for --requests}"; shift 2 ;;
    --observed-gripper-m) observed_gripper_m="${2:?missing value for --observed-gripper-m}"; shift 2 ;;
    --record-policy-arrays) record_policy_arrays=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: ${1}" >&2; usage >&2; exit 1 ;;
  esac
done

ensure_directory "${output_root}"
server_log="${output_root}/exact_server.log"
latency_json="${output_root}/latency.json"
summary_json="${output_root}/summary.json"
shadow_dir="${output_root}/shadow_context"

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
  --action-dim 7
  --exact-rollout-payload
  --rlinf-config-name "${rlinf_config_name}"
  --exact-action-chunk "${exact_action_chunk}"
  --exact-solver-type flow_sde
  --exact-infer-mode train
)

export PYTHONUNBUFFERED=1
"${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- "${server_cmd[@]}" >"${server_log}" 2>&1 &
server_pid=$!

for _attempt in $(seq 1 180); do
  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    cat "${server_log}" >&2 || true
    echo "error: exact service exited before opening ${host}:${port}" >&2
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

"${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- \
  python "${CAVER_REPO_ROOT}/scripts/bridge/measure_policy_latency.py" \
    --host "${host}" \
    --port "${port}" \
    --warmup "${warmup}" \
    --requests "${requests}" \
    --output-json "${latency_json}"

shadow_args=(
  python
  "${CAVER_REPO_ROOT}/scripts/stage1/shadow_mode_dry_run.py"
  --output-dir "${shadow_dir}"
  --task-id "${task_id}"
  --policy-host "${host}"
  --policy-port "${port}"
  --synthetic-left-pose
  --observed-gripper-m "${observed_gripper_m}"
)
if ((record_policy_arrays)); then
  shadow_args+=(--record-policy-arrays)
fi

"${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- "${shadow_args[@]}"

python3 - "${summary_json}" "${latency_json}" "${shadow_dir}/shadow_context.json" "${server_log}" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
latency_path = pathlib.Path(sys.argv[2])
shadow_path = pathlib.Path(sys.argv[3])
server_log = pathlib.Path(sys.argv[4])

payload = {
    "status": "modeb_sdre_rehearsal_complete",
    "latency_json": str(latency_path),
    "shadow_context_json": str(shadow_path),
    "server_log": str(server_log),
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"OK summary={summary_path}")
PY

tail -n 40 "${server_log}" || true
