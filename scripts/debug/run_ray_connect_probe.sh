#!/usr/bin/env bash
set -euo pipefail

_CAVER_DEBUG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_DEBUG_DIR}/../common.sh"

short_ray_tmpdir="${RAY_TMPDIR:-$(caver_default_ray_tmpdir)}"
ensure_directory "${short_ray_tmpdir}"

export RAY_TMPDIR="${short_ray_tmpdir}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

exec "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh" \
  -- python -u "${CAVER_REPO_ROOT}/scripts/debug/ray_init_probe.py" "$@"
