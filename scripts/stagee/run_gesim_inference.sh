#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_gesim_inference.sh --bundle-dir PATH [options]

Options:
  --bundle-dir PATH            GE-Sim-compatible bundle directory from Stage-E LIBERO export
  --output-dir PATH            Output directory for config, status, and generated video
  --config-template-path PATH  Template YAML config (default: Genie-Envisioner acwm_cosmos.yaml)
  --cosmos-assets-root PATH    Local gated Cosmos asset snapshot directory
  --gesim-checkpoint-path PATH Local ge_sim_cosmos_v0.1.safetensors path
  --prompt TEXT                Prompt string passed to GE-Sim inference
  --download-cosmos-assets     Attempt gated HF download before running
  --verify-checkpoint-sha256   Verify the full GE-Sim checkpoint hash before running
  --dry-run                    Print commands without executing them
  -h, --help                   Show this message

Example:
  scripts/stagee/run_gesim_inference.sh \
    --bundle-dir .tmp/gesim_bundle_smoke \
    --output-dir .tmp/gesim_runtime_smoke
EOF
}

bundle_dir=""
output_dir=""
config_template_path="${CAVER_REPO_ROOT}/third_party/src/Genie-Envisioner/configs/cosmos_model/acwm_cosmos.yaml"
cosmos_assets_root="${CAVER_REPO_ROOT}/third_party/model-cache/gesim/Cosmos-Predict2-2B-Video2World"
gesim_checkpoint_path="${CAVER_REPO_ROOT}/third_party/model-cache/gesim/ge_sim_cosmos_v0.1.safetensors"
prompt="best quality, consistent and smooth motion, realistic, clear and distinct."
download_cosmos_assets=0
verify_checkpoint_sha256=0
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --bundle-dir)
      bundle_dir="${2:?missing value for --bundle-dir}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --config-template-path)
      config_template_path="${2:?missing value for --config-template-path}"
      shift 2
      ;;
    --cosmos-assets-root)
      cosmos_assets_root="${2:?missing value for --cosmos-assets-root}"
      shift 2
      ;;
    --gesim-checkpoint-path)
      gesim_checkpoint_path="${2:?missing value for --gesim-checkpoint-path}"
      shift 2
      ;;
    --prompt)
      prompt="${2:?missing value for --prompt}"
      shift 2
      ;;
    --download-cosmos-assets)
      download_cosmos_assets=1
      shift
      ;;
    --verify-checkpoint-sha256)
      verify_checkpoint_sha256=1
      shift
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
      echo "error: unknown argument: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${bundle_dir}" ]; then
  echo "error: --bundle-dir is required" >&2
  usage >&2
  exit 1
fi

bundle_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${bundle_dir}")"
if [ ! -d "${bundle_dir}" ]; then
  echo "error: bundle directory not found: ${bundle_dir}" >&2
  exit 1
fi

for required_path in \
  "${bundle_dir}/actions.npy" \
  "${bundle_dir}/extrinsic_head.npy" \
  "${bundle_dir}/intrinsic_head.npy" \
  "${bundle_dir}/head_color/0.png"; do
  if [ ! -e "${required_path}" ]; then
    echo "error: bundle is missing required GE-Sim input: ${required_path}" >&2
    exit 1
  fi
done

if [ -z "${output_dir}" ]; then
  output_dir="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/gesim_infer__$(timestamp_utc)"
fi
ensure_directory "${output_dir}"
output_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${output_dir}")"

runtime_status_path="${output_dir}/gesim_runtime_status.json"
runtime_config_path="${output_dir}/gesim_runtime.yaml"
provider_summary_path="${output_dir}/provider_summary.json"

prepare_cmd=(
  "${CAVER_REPO_ROOT}/scripts/env/with_gesim_infer.sh"
  --
  python
  "${CAVER_REPO_ROOT}/scripts/stagee/prepare_gesim_runtime.py"
  --config-template-path "${config_template_path}"
  --output-config-path "${runtime_config_path}"
  --status-json-path "${runtime_status_path}"
  --cosmos-assets-root "${cosmos_assets_root}"
  --gesim-checkpoint-path "${gesim_checkpoint_path}"
  --require-ready
)

if ((download_cosmos_assets)); then
  prepare_cmd+=(--download-cosmos-assets)
fi
if ((verify_checkpoint_sha256)); then
  prepare_cmd+=(--verify-checkpoint-sha256)
fi

infer_cmd=(
  "${CAVER_REPO_ROOT}/scripts/env/with_gesim_infer.sh"
  --
  python
  "${CAVER_REPO_ROOT}/third_party/src/Genie-Envisioner/gesim_video_gen_examples/infer_gesim.py"
  --config_file "${runtime_config_path}"
  --image_root "${bundle_dir}"
  --extrinsic_root "${bundle_dir}"
  --intrinsic_root "${bundle_dir}"
  --action_path "${bundle_dir}/actions.npy"
  --output_path "${output_dir}"
  --prompt "${prompt}"
)

summary_cmd=(
  "${CAVER_REPO_ROOT}/scripts/env/with_gesim_infer.sh"
  --
  python
  "${CAVER_REPO_ROOT}/scripts/stagee/summarize_gesim_output.py"
  --output-dir "${output_dir}"
  --config-path "${runtime_config_path}"
  --summary-path "${provider_summary_path}"
)

printf 'prepare command:'
printf ' %q' "${prepare_cmd[@]}"
printf '\n'
printf 'infer command:'
printf ' %q' "${infer_cmd[@]}"
printf '\n'
printf 'summary command:'
printf ' %q' "${summary_cmd[@]}"
printf '\n'

if ((dry_run)); then
  exit 0
fi

"${prepare_cmd[@]}"
"${infer_cmd[@]}"
"${summary_cmd[@]}"

printf 'GE-Sim output directory: %s\n' "${output_dir}"
