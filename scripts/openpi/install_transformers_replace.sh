#!/usr/bin/env bash
set -euo pipefail

_CAVER_OPENPI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_OPENPI_SCRIPT_DIR}/../common.sh"

quiet=0

while (($# > 0)); do
  case "${1}" in
    --quiet)
      quiet=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  install_transformers_replace.sh [--quiet]

Repairs the OpenPI PyTorch `transformers_replace` overlay inside the active Python environment.
Run this after activating the target venv.
EOF
      exit 0
      ;;
    *)
      echo "error: unknown option: ${1}" >&2
      exit 1
      ;;
  esac
done

require_command python

overlay_dir="${CAVER_DEFAULT_SOURCE_ROOT}/openpi/src/openpi/models_pytorch/transformers_replace"
if [ ! -d "${overlay_dir}" ]; then
  echo "error: transformers_replace overlay not found: ${overlay_dir}" >&2
  exit 1
fi

if python - <<'PY' >/dev/null 2>&1
from transformers.models.siglip import check
raise SystemExit(0 if check.check_whether_transformers_replace_is_installed_correctly() else 1)
PY
then
  if ((!quiet)); then
    echo "transformers_replace already installed correctly"
  fi
  exit 0
fi

transformers_dir="$(python - <<'PY'
import pathlib
import transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)"

if [ ! -d "${transformers_dir}" ]; then
  echo "error: transformers package directory not found: ${transformers_dir}" >&2
  exit 1
fi

cp -r "${overlay_dir}/"* "${transformers_dir}/"

python - <<'PY' >/dev/null
from transformers.models.siglip import check
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit(1)
PY

if ((!quiet)); then
  echo "installed transformers_replace into ${transformers_dir}"
fi
