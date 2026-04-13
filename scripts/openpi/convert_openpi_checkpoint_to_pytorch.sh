#!/usr/bin/env bash
set -euo pipefail

_CAVER_OPENPI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_OPENPI_SCRIPT_DIR}/../common.sh"
# shellcheck disable=SC1091
source "${_CAVER_OPENPI_SCRIPT_DIR}/../env/module_stacks.sh"

usage() {
  cat <<'EOF'
Usage:
  convert_openpi_checkpoint_to_pytorch.sh [options]

Options:
  --checkpoint-dir PATH    Source OpenPI JAX checkpoint directory
  --config-name NAME       OpenPI config name (default: pi05_libero)
  --output-path PATH       Output directory (default: <checkpoint-dir>_pytorch)
  --precision NAME         Output precision: float32 or bfloat16 (default: bfloat16)
  --inspect-only           Only inspect the source checkpoint metadata
  -h, --help               Show this message
EOF
}

checkpoint_dir=""
config_name="pi05_libero"
output_path=""
precision="bfloat16"
inspect_only=0

while (($# > 0)); do
  case "${1}" in
    --checkpoint-dir)
      checkpoint_dir="${2:?missing value for --checkpoint-dir}"
      shift 2
      ;;
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --output-path)
      output_path="${2:?missing value for --output-path}"
      shift 2
      ;;
    --precision)
      precision="${2:?missing value for --precision}"
      shift 2
      ;;
    --inspect-only)
      inspect_only=1
      shift
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

if [ -z "${checkpoint_dir}" ]; then
  echo "error: --checkpoint-dir is required" >&2
  usage >&2
  exit 1
fi

case "${precision}" in
  float32|bfloat16)
    ;;
  *)
    echo "error: unsupported --precision value: ${precision}" >&2
    exit 1
    ;;
esac

require_command python

checkpoint_dir="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${checkpoint_dir}")"
if [ ! -d "${checkpoint_dir}" ]; then
  echo "error: checkpoint directory not found: ${checkpoint_dir}" >&2
  exit 1
fi

if [ -z "${output_path}" ]; then
  output_path="${checkpoint_dir}_pytorch"
fi
output_path="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${output_path}")"

activate_train_stack

venv_dir="${CAVER_DEFAULT_VENV_ROOT}/openpi"
if [ ! -f "${venv_dir}/bin/activate" ]; then
  echo "error: openpi venv not found at ${venv_dir}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"

export PIP_CONFIG_FILE=/dev/null
ensure_directory "${OPENPI_DATA_HOME:-${CAVER_DEFAULT_OPENPI_CACHE_ROOT}}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CAVER_DEFAULT_OPENPI_CACHE_ROOT}}"
"${CAVER_REPO_ROOT}/scripts/openpi/install_transformers_replace.sh" --quiet

cmd=(
  python
  "${CAVER_DEFAULT_SOURCE_ROOT}/openpi/examples/convert_jax_model_to_pytorch.py"
  --checkpoint_dir "${checkpoint_dir}"
  --config_name "${config_name}"
)

if ((inspect_only)); then
  cmd+=(--inspect-only)
else
  ensure_directory "${output_path}"
  cmd+=(--output_path "${output_path}" --precision "${precision}")
fi

printf 'running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

if ((inspect_only)); then
  exit 0
fi

assets_src="${checkpoint_dir}/assets"
assets_dst="${output_path}/assets"
if [ -d "${assets_src}" ]; then
  ensure_directory "${assets_dst}"
  cp -a "${assets_src}/." "${assets_dst}/"
  echo "copied assets directory into ${assets_dst}"
fi
