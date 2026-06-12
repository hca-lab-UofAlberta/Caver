#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_fit_stagee_lvd_selector.sh [args...]

This wrapper runs the Stage-E CAVER-LVD selector fitter inside the existing
OpenPI/pi-StepNFT training environment.
EOF
}

if (($# == 0)); then
  usage >&2
  exit 1
fi

if [ "${1}" = "-h" ] || [ "${1}" = "--help" ]; then
  usage
  "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh" -- \
    python "${CAVER_REPO_ROOT}/scripts/stagee/fit_stagee_lvd_selector.py" --help
  exit 0
fi

exec "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh" -- \
  python "${CAVER_REPO_ROOT}/scripts/stagee/fit_stagee_lvd_selector.py" "$@"
