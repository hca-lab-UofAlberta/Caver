#!/usr/bin/env bash
set -euo pipefail

_CAVER_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${_CAVER_SLURM_DIR}/interactive_gpu.sh" --partition gpu-h200 --qos interactive --gpu-type h200 "$@"

