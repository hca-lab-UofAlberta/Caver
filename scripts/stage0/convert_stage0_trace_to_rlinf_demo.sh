#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  convert_stage0_trace_to_rlinf_demo.sh [options]

Options:
  --trace-path PATH         Required Stage-0 chunk-trace JSONL path
  --output-path PATH        Optional output path (.pt for single_pt, .json for sharded_manifest)
  --summary-path PATH       Optional JSON summary path
  --output-mode NAME        single_pt or sharded_manifest (default: single_pt)
  --max-items-per-shard N   Max items per shard for sharded_manifest output
  --demo-format NAME        chunk_step or primitive_step (default: chunk_step)
  --openpi-config-name NAME OpenPI config name (default: pi05_libero)
  --max-token-len N         Optional prompt tokenizer length override
  --discrete-state-input    Tokenize prompts with discretized state text
  --max-records N           Optional limit on chunk records read from the trace
  -h, --help                Show this message
EOF
}

trace_path=""
output_path=""
summary_path=""
output_mode="single_pt"
max_items_per_shard=""
demo_format="chunk_step"
openpi_config_name="pi05_libero"
max_token_len=""
discrete_state_input=0
max_records=""

while (($# > 0)); do
  case "${1}" in
    --trace-path)
      trace_path="${2:?missing value for --trace-path}"
      shift 2
      ;;
    --output-path)
      output_path="${2:?missing value for --output-path}"
      shift 2
      ;;
    --summary-path)
      summary_path="${2:?missing value for --summary-path}"
      shift 2
      ;;
    --output-mode)
      output_mode="${2:?missing value for --output-mode}"
      shift 2
      ;;
    --max-items-per-shard)
      max_items_per_shard="${2:?missing value for --max-items-per-shard}"
      shift 2
      ;;
    --demo-format)
      demo_format="${2:?missing value for --demo-format}"
      shift 2
      ;;
    --openpi-config-name)
      openpi_config_name="${2:?missing value for --openpi-config-name}"
      shift 2
      ;;
    --max-token-len)
      max_token_len="${2:?missing value for --max-token-len}"
      shift 2
      ;;
    --discrete-state-input)
      discrete_state_input=1
      shift
      ;;
    --max-records)
      max_records="${2:?missing value for --max-records}"
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

if [ -z "${trace_path}" ]; then
  echo "error: --trace-path is required" >&2
  usage >&2
  exit 1
fi

trace_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${trace_path}")"

if [ -z "${output_path}" ]; then
  if [ "${output_mode}" = "sharded_manifest" ]; then
    if [ -n "${CAVER_RUN_DIR:-}" ]; then
      output_path="${CAVER_RUN_DIR}/results/stage0_seed_warm_start_demo.manifest.json"
    else
      output_path="${trace_path%.jsonl}.demo.manifest.json"
    fi
  else
    if [ -n "${CAVER_RUN_DIR:-}" ]; then
      output_path="${CAVER_RUN_DIR}/results/stage0_seed_warm_start_demo.pt"
    else
      output_path="${trace_path%.jsonl}.demo.pt"
    fi
  fi
fi

if [ -z "${summary_path}" ]; then
  summary_path="${output_path%.pt}.summary.json"
  if [ "${output_mode}" = "sharded_manifest" ]; then
    summary_path="${output_path%.json}.summary.json"
  fi
fi

cmd=(
  python
  "${CAVER_REPO_ROOT}/scripts/stage0/convert_stage0_trace_to_rlinf_demo.py"
  --trace-path "${trace_path}"
  --output-path "${output_path}"
  --summary-path "${summary_path}"
  --output-mode "${output_mode}"
  --demo-format "${demo_format}"
  --openpi-config-name "${openpi_config_name}"
)

if [ -n "${max_token_len}" ]; then
  cmd+=(--max-token-len "${max_token_len}")
fi
if ((discrete_state_input)); then
  cmd+=(--discrete-state-input)
fi
if [ -n "${max_records}" ]; then
  cmd+=(--max-records "${max_records}")
fi
if [ -n "${max_items_per_shard}" ]; then
  cmd+=(--max-items-per-shard "${max_items_per_shard}")
fi

exec "${CAVER_REPO_ROOT}/scripts/env/with_openpi_libero_eval.sh" -- "${cmd[@]}"
