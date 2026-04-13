#!/usr/bin/env bash
set -euo pipefail

_CAVER_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${_CAVER_SLURM_DIR}/interactive_gpu.sh" --partition gpu-l40s --qos interactive --gpu-type l40s "$@"

