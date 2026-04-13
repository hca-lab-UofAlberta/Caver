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
  with_libero_eval.sh -- <command> [args...]

Examples:
  scripts/env/with_libero_eval.sh -- bash -l
  scripts/env/with_libero_eval.sh -- python -c 'from libero.libero import benchmark'
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

activate_libero_stack

venv_dir="${CAVER_DEFAULT_VENV_ROOT}/libero"
if [ ! -f "${venv_dir}/bin/activate" ]; then
  echo "error: libero venv not found at ${venv_dir}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"

export PIP_CONFIG_FILE=/dev/null
export LIBERO_CONFIG_PATH="${CAVER_REPO_ROOT}/third_party/config/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
case "${MUJOCO_GL}" in
  egl)
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
    export LIBERO_RENDER_GPU_DEVICE_ID="${LIBERO_RENDER_GPU_DEVICE_ID:-0}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    ;;
  osmesa)
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
    ;;
  *)
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
    ;;
esac
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CAVER_DEFAULT_SOURCE_ROOT}/LIBERO:${CAVER_DEFAULT_SOURCE_ROOT}/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
