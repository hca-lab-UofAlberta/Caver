#!/usr/bin/env bash
set -euo pipefail

_CAVER_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_ENV_DIR}/../common.sh"
# shellcheck disable=SC1091
source "${_CAVER_ENV_DIR}/module_stacks.sh"

usage() {
  cat <<'EOF'
Usage:
  with_openpi_libero_eval.sh -- <command> [args...]

Examples:
  scripts/env/with_openpi_libero_eval.sh -- bash -l
  scripts/env/with_openpi_libero_eval.sh -- python -c 'import toolkits.eval_scripts_openpi'
EOF
}

if (($# == 0)); then
  usage >&2
  exit 1
fi

if [ "${1}" = "-h" ] || [ "${1}" = "--help" ]; then
  usage
  exit 0
fi

if [ "${1}" = "--" ]; then
  shift
fi

if (($# == 0)); then
  usage >&2
  exit 1
fi

activate_train_stack

venv_dir="${CAVER_DEFAULT_VENV_ROOT}/openpi"
if [ ! -f "${venv_dir}/bin/activate" ]; then
  echo "error: openpi venv not found at ${venv_dir}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"

if [ -n "${CAVER_OPENPI_CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CAVER_OPENPI_CUDA_VISIBLE_DEVICES}"
fi

export PIP_CONFIG_FILE=/dev/null
export LIBERO_CONFIG_PATH="${CAVER_REPO_ROOT}/third_party/config/libero"
ensure_directory "${OPENPI_DATA_HOME:-${CAVER_DEFAULT_OPENPI_CACHE_ROOT}}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CAVER_DEFAULT_OPENPI_CACHE_ROOT}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CAVER_DEFAULT_SOURCE_ROOT}/pi-StepNFT:${CAVER_DEFAULT_SOURCE_ROOT}/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
