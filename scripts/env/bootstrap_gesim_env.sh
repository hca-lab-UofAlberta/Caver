#!/usr/bin/env bash
set -euo pipefail

_CAVER_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${_CAVER_ENV_DIR}/bootstrap_component.sh" gesim "$@"

