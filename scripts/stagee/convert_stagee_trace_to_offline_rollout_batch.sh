#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_CONVERT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_CONVERT_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  convert_stagee_trace_to_offline_rollout_batch.sh [options]

Required:
  --trace-path PATH     Admitted Stage-E exact trace JSONL or trace-source manifest
  --output-path PATH    Output .pt path for the offline rollout batch

Optional:
  --summary-path PATH
  --openpi-config-name NAME
  --max-token-len N
  --discrete-state-input
  --max-records N
  --max-contexts N
  --include-prev-logprobs
  --dry-run
  -h, --help
EOF
}

trace_path=""
output_path=""
summary_path=""
openpi_config_name="pi05_libero"
max_token_len=""
discrete_state_input=0
max_records=""
max_contexts=""
include_prev_logprobs=0
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --trace-path) trace_path="${2:?missing value for --trace-path}"; shift 2 ;;
    --output-path) output_path="${2:?missing value for --output-path}"; shift 2 ;;
    --summary-path) summary_path="${2:?missing value for --summary-path}"; shift 2 ;;
    --openpi-config-name) openpi_config_name="${2:?missing value for --openpi-config-name}"; shift 2 ;;
    --max-token-len) max_token_len="${2:?missing value for --max-token-len}"; shift 2 ;;
    --discrete-state-input) discrete_state_input=1; shift ;;
    --max-records) max_records="${2:?missing value for --max-records}"; shift 2 ;;
    --max-contexts) max_contexts="${2:?missing value for --max-contexts}"; shift 2 ;;
    --include-prev-logprobs) include_prev_logprobs=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: ${1}" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "${trace_path}" ] || [ -z "${output_path}" ]; then
  echo "error: --trace-path and --output-path are required" >&2
  usage >&2
  exit 1
fi

trace_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${trace_path}")"
output_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${output_path}")"
if [ -n "${summary_path}" ]; then
  summary_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${summary_path}")"
fi

cmd=(
  "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh"
  --
  python
  "${CAVER_REPO_ROOT}/scripts/stagee/convert_stagee_trace_to_offline_rollout_batch.py"
  --trace-path "${trace_path}"
  --output-path "${output_path}"
  --openpi-config-name "${openpi_config_name}"
)

if [ -n "${summary_path}" ]; then
  cmd+=(--summary-path "${summary_path}")
fi
if [ -n "${max_token_len}" ]; then
  cmd+=(--max-token-len "${max_token_len}")
fi
if ((discrete_state_input)); then
  cmd+=(--discrete-state-input)
fi
if [ -n "${max_records}" ]; then
  cmd+=(--max-records "${max_records}")
fi
if [ -n "${max_contexts}" ]; then
  cmd+=(--max-contexts "${max_contexts}")
fi
if ((include_prev_logprobs)); then
  cmd+=(--include-prev-logprobs)
fi

printf 'convert command:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'artifacts:\n'
printf '  trace_path: %s\n' "${trace_path}"
printf '  output_path: %s\n' "${output_path}"
if [ -n "${summary_path}" ]; then
  printf '  summary_path: %s\n' "${summary_path}"
fi

if ((dry_run)); then
  exit 0
fi

exec "${cmd[@]}"
