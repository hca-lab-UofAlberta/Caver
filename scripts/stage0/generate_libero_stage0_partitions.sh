#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

exec "${_CAVER_STAGE0_DIR}/../env/with_libero_eval.sh" -- \
  python "${_CAVER_STAGE0_DIR}/generate_stage0_partitions.py" "$@"
