#!/usr/bin/env bash
set -euo pipefail

_CAVER_OPENPI_EXPORT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_OPENPI_EXPORT_DIR}/../common.sh"

exec "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh" -- \
  python "${CAVER_REPO_ROOT}/scripts/openpi/export_rlinf_actor_checkpoint_to_pytorch.py" "$@"
