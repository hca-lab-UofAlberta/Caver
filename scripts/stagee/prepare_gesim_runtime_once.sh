#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  prepare_gesim_runtime_once.sh --output-dir PATH [options]

Options:
  --output-dir PATH            Output directory for rendered runtime YAML and status JSON
  --config-template-path PATH  Template YAML config (default: Genie-Envisioner acwm_cosmos.yaml)
  --cosmos-assets-root PATH    Local gated Cosmos asset snapshot directory
  --gesim-checkpoint-path PATH Local ge_sim_cosmos_v0.1.safetensors path
  --download-cosmos-assets     Attempt gated HF download before running
  --verify-checkpoint-sha256   Verify the full GE-Sim checkpoint hash before running
  --dry-run                    Print command without executing it
  -h, --help                   Show this message
EOF
}

output_dir=""
config_template_path="${CAVER_REPO_ROOT}/third_party/src/Genie-Envisioner/configs/cosmos_model/acwm_cosmos.yaml"
cosmos_assets_root="${CAVER_REPO_ROOT}/third_party/model-cache/gesim/Cosmos-Predict2-2B-Video2World"
gesim_checkpoint_path="${CAVER_REPO_ROOT}/third_party/model-cache/gesim/ge_sim_cosmos_v0.1.safetensors"
download_cosmos_assets=0
verify_checkpoint_sha256=0
dry_run=0

while (($# > 0)); do
  case "${1}" in
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

if [ -z "${output_dir}" ]; then
  echo "error: --output-dir is required" >&2
  usage >&2
  exit 1
fi

ensure_directory "${output_dir}"
output_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${output_dir}")"
runtime_status_path="${output_dir}/gesim_runtime_status.json"
runtime_config_path="${output_dir}/gesim_runtime.yaml"

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

printf 'prepare command:'
printf ' %q' "${prepare_cmd[@]}"
printf '\n'

if ((dry_run)); then
  exit 0
fi

"${prepare_cmd[@]}"
printf 'GE-Sim prepared runtime directory: %s\n' "${output_dir}"
