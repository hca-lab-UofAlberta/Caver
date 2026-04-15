#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_train_stage0_value_proxy_mlp.sh [args...]

This wrapper runs the proposal-side Stage-0 value-proxy fitter inside the
existing OpenPI/pi-StepNFT training environment.
EOF
}

if (($# == 0)); then
  usage >&2
  exit 1
fi

if [ "${1}" = "-h" ] || [ "${1}" = "--help" ]; then
  usage
  "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh" -- \
    python "${CAVER_REPO_ROOT}/scripts/stagee/train_stage0_value_proxy_mlp.py" --help
  exit 0
fi

exec "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh" -- \
  python "${CAVER_REPO_ROOT}/scripts/stagee/train_stage0_value_proxy_mlp.py" "$@"
