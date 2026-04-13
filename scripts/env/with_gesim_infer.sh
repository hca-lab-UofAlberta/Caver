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
  with_gesim_infer.sh -- <command> [args...]

Examples:
  scripts/env/with_gesim_infer.sh -- bash -l
  scripts/env/with_gesim_infer.sh -- python -c 'import gesim_video_gen_examples.infer_gesim'
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

activate_gesim_stack

venv_dir="${CAVER_DEFAULT_VENV_ROOT}/gesim"
if [ ! -f "${venv_dir}/bin/activate" ]; then
  echo "error: gesim venv not found at ${venv_dir}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"

if [ -n "${CAVER_GESIM_CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CAVER_GESIM_CUDA_VISIBLE_DEVICES}"
fi

export PIP_CONFIG_FILE=/dev/null
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CAVER_DEFAULT_SOURCE_ROOT}/Genie-Envisioner${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${CAVER_REPO_ROOT}/third_party/model-cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-${CAVER_DEFAULT_RDSS_ROOT}/tmp/${SLURM_JOB_ID:-manual}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
ensure_directory "${HF_HOME}"
ensure_directory "${HUGGINGFACE_HUB_CACHE}"
ensure_directory "${TMPDIR}"

exec "$@"
