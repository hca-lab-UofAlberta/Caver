#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_fit_stage0_seed_calibrator_mlp.sh [options] [-- extra_fit_args...]

Build a Stage-0 seed calibrator dataset from the executed seed tuples, then fit
the proposal-side Stage-E MLP calibrator so round 1 can start from an explicit
seed-fitted artifact.

Options:
  --input-path PATH          Seed tuple dataset (default: metadata/stage0/value_proxy/stage0_seed_context_success_dataset.jsonl)
  --dataset-path PATH        Output seed-calibrator dataset JSONL
  --dataset-summary-path PATH
                             Output seed-calibrator dataset summary JSON
  --output-path PATH         Output calibrator artifact JSON
  --summary-path PATH        Output calibrator training summary JSON
  --label-key NAME           context_success_label or chunk_success_label (default: context_success_label)
  --nuisance-key NAME        raw_value_proxy or zero (default: raw_value_proxy)
  --weight-mode NAME         uniform or inverse_context_trace_count (default: inverse_context_trace_count)
  -h, --help                 Show this message

Any arguments after `--` are passed directly to fit_stagee_dr_calibrator_mlp.py.
EOF
}

input_path="${CAVER_REPO_ROOT}/metadata/stage0/value_proxy/stage0_seed_context_success_dataset.jsonl"
dataset_path="${CAVER_REPO_ROOT}/metadata/stage0/calibrator/stage0_seed_dr_candidate_dataset.jsonl"
dataset_summary_path="${CAVER_REPO_ROOT}/metadata/stage0/calibrator/stage0_seed_dr_candidate_dataset.summary.json"
output_path="${CAVER_REPO_ROOT}/metadata/stage0/calibrator/stage0_seed_dr_calibrator_mlp_v2.json"
summary_path="${CAVER_REPO_ROOT}/metadata/stage0/calibrator/stage0_seed_dr_calibrator_mlp_v2.summary.json"
label_key="context_success_label"
nuisance_key="raw_value_proxy"
weight_mode="inverse_context_trace_count"
fit_args=()

while (($# > 0)); do
  case "${1}" in
    --input-path)
      input_path="${2:?missing value for --input-path}"
      shift 2
      ;;
    --dataset-path)
      dataset_path="${2:?missing value for --dataset-path}"
      shift 2
      ;;
    --dataset-summary-path)
      dataset_summary_path="${2:?missing value for --dataset-summary-path}"
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
    --label-key)
      label_key="${2:?missing value for --label-key}"
      shift 2
      ;;
    --nuisance-key)
      nuisance_key="${2:?missing value for --nuisance-key}"
      shift 2
      ;;
    --weight-mode)
      weight_mode="${2:?missing value for --weight-mode}"
      shift 2
      ;;
    --)
      shift
      fit_args=("$@")
      break
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

python3 "${CAVER_REPO_ROOT}/scripts/stagee/build_stage0_seed_calibrator_dataset.py" \
  --input-path "${input_path}" \
  --output-path "${dataset_path}" \
  --summary-path "${dataset_summary_path}" \
  --label-key "${label_key}" \
  --nuisance-key "${nuisance_key}" \
  --weight-mode "${weight_mode}"

"${CAVER_REPO_ROOT}/scripts/stagee/run_fit_stagee_dr_calibrator_mlp.sh" \
  --dataset-path "${dataset_path}" \
  --output-path "${output_path}" \
  --summary-path "${summary_path}" \
  --target-key dr_pseudo_outcome \
  --model-id stage0_seed_dr_calibrator_mlp_v2 \
  "${fit_args[@]}"
