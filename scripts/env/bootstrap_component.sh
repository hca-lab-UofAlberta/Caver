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
  bootstrap_component.sh <component> [options]

Components:
  openpi
  libero
  pistepnft
  gesim

Options:
  --src-root PATH     Source checkout root (default: third_party/src)
  --env-root PATH     Virtualenv root (default: third_party/venvs)
  --ref REF           Override the default git ref
  --profile NAME      Optional install profile override
  --skip-clone        Reuse an existing checkout
  --skip-install      Prepare the venv but do not install the package
  --dry-run           Print the actions without modifying anything
  -h, --help          Show this message
EOF
}

clone_or_update_repo() {
  local repo_url="${1:?missing repo url}"
  local repo_ref="${2:?missing repo ref}"
  local checkout_dir="${3:?missing checkout dir}"
  local use_submodules="${4:?missing submodule flag}"
  local remote_head=""

  if [ ! -d "${checkout_dir}/.git" ]; then
    ensure_directory "$(dirname -- "${checkout_dir}")"
    if [ "${use_submodules}" = "1" ]; then
      GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --recurse-submodules "${repo_url}" "${checkout_dir}"
    else
      git clone --filter=blob:none "${repo_url}" "${checkout_dir}"
    fi
  fi

  git -C "${checkout_dir}" fetch --tags --force origin
  git -C "${checkout_dir}" fetch --force origin "${repo_ref}" >/dev/null 2>&1 || true

  if git -C "${checkout_dir}" rev-parse --verify --quiet "${repo_ref}^{commit}" >/dev/null; then
    git -C "${checkout_dir}" checkout --detach "${repo_ref}"
  elif git -C "${checkout_dir}" rev-parse --verify --quiet "origin/${repo_ref}^{commit}" >/dev/null; then
    git -C "${checkout_dir}" checkout -B "${repo_ref}" "origin/${repo_ref}"
  else
    remote_head="$(git -C "${checkout_dir}" symbolic-ref --quiet refs/remotes/origin/HEAD 2>/dev/null || true)"
    if [ -n "${remote_head}" ]; then
      git -C "${checkout_dir}" checkout -B "${repo_ref}" "${remote_head}"
    else
      echo "error: unable to resolve git ref '${repo_ref}' for ${checkout_dir}" >&2
      return 1
    fi
  fi

  if [ "${use_submodules}" = "1" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "${checkout_dir}" submodule update --init --recursive
  fi
}

component=""
src_root="${CAVER_DEFAULT_SOURCE_ROOT}"
env_root="${CAVER_DEFAULT_VENV_ROOT}"
repo_ref_override=""
install_profile=""
skip_clone=0
skip_install=0
dry_run=0

while (($# > 0)); do
  case "$1" in
    openpi|libero|pistepnft|gesim)
      component="$1"
      shift
      ;;
    --src-root)
      src_root="${2:?missing value for --src-root}"
      shift 2
      ;;
    --env-root)
      env_root="${2:?missing value for --env-root}"
      shift 2
      ;;
    --ref)
      repo_ref_override="${2:?missing value for --ref}"
      shift 2
      ;;
    --profile)
      install_profile="${2:?missing value for --profile}"
      shift 2
      ;;
    --skip-clone)
      skip_clone=1
      shift
      ;;
    --skip-install)
      skip_install=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${component}" ]; then
  echo "error: component is required" >&2
  usage >&2
  exit 1
fi

repo_url=""
repo_ref=""
checkout_name=""
venv_name=""
use_submodules="0"
stack_function=""

case "${component}" in
  openpi)
    repo_url="https://github.com/Physical-Intelligence/openpi.git"
    repo_ref="fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc"
    checkout_name="openpi"
    venv_name="openpi"
    use_submodules="1"
    stack_function="activate_train_stack"
    ;;
  libero)
    repo_url="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    repo_ref="master"
    checkout_name="LIBERO"
    venv_name="libero"
    stack_function="activate_libero_stack"
    ;;
  pistepnft)
    repo_url="https://github.com/wangst0181/pi-StepNFT.git"
    repo_ref="140eece2ef7e8574c34c69a221fd4e3f56c2423c"
    checkout_name="pi-StepNFT"
    venv_name="pi-stepnft"
    stack_function="activate_train_stack"
    ;;
  gesim)
    repo_url="https://github.com/AgibotTech/Genie-Envisioner.git"
    repo_ref="0e0de7e2ed8ea04e6ba41d47c93964ae65e5fbd2"
    checkout_name="Genie-Envisioner"
    venv_name="gesim"
    stack_function="activate_gesim_stack"
    ;;
esac

if [ -n "${repo_ref_override}" ]; then
  repo_ref="${repo_ref_override}"
fi

checkout_dir="${src_root}/${checkout_name}"
venv_dir="${env_root}/${venv_name}"

if [ "${dry_run}" = "1" ]; then
  cat <<EOF
component: ${component}
repo_url: ${repo_url}
repo_ref: ${repo_ref}
checkout_dir: ${checkout_dir}
venv_dir: ${venv_dir}
stack_function: ${stack_function}
install_profile: ${install_profile:-default}
skip_clone: ${skip_clone}
skip_install: ${skip_install}
EOF
  exit 0
fi

"${stack_function}"
python --version
print_loaded_modules

if [ "${skip_clone}" = "0" ]; then
  clone_or_update_repo "${repo_url}" "${repo_ref}" "${checkout_dir}" "${use_submodules}"
fi

ensure_directory "${env_root}"
if [ ! -d "${venv_dir}" ]; then
  python -m venv "${venv_dir}"
fi

# shellcheck disable=SC1090
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [ "${skip_install}" = "1" ]; then
  echo "prepared ${component} environment at ${venv_dir}"
  exit 0
fi

case "${component}" in
  openpi)
    python -m pip install -e "${checkout_dir}"
    ;;
  libero)
    if [ -f "${checkout_dir}/requirements.txt" ]; then
      python -m pip install -r "${checkout_dir}/requirements.txt"
    fi
    python -m pip install -e "${checkout_dir}"
    ;;
  pistepnft)
    if [ -f "${checkout_dir}/requirements.txt" ]; then
      python -m pip install -r "${checkout_dir}/requirements.txt"
    fi
    python -m pip install -e "${checkout_dir}"
    ;;
  gesim)
    if [ -n "${install_profile}" ] && [ "${install_profile}" != "default" ]; then
      if [ "${install_profile}" = "sdre-infer" ]; then
        profile_requirements="${_CAVER_ENV_DIR}/requirements/gesim_sdre_infer.txt"
        if [ ! -f "${profile_requirements}" ]; then
          echo "error: missing GE-Sim profile requirements: ${profile_requirements}" >&2
          exit 1
        fi
        python -m pip install -r "${profile_requirements}"
      else
        echo "error: unsupported GE-Sim install profile: ${install_profile}" >&2
        exit 1
      fi
    elif [ -f "${checkout_dir}/requirements.txt" ]; then
      python -m pip install -r "${checkout_dir}/requirements.txt"
    fi
    if [ -f "${checkout_dir}/pyproject.toml" ] || [ -f "${checkout_dir}/setup.py" ]; then
      python -m pip install -e "${checkout_dir}"
    fi
    ;;
esac

echo "bootstrap complete for ${component}"
git -C "${checkout_dir}" rev-parse HEAD
