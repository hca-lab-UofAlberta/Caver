#!/usr/bin/env bash

_CAVER_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CAVER_REPO_ROOT="${CAVER_REPO_ROOT:-$(cd -- "${_CAVER_COMMON_DIR}/.." && pwd -P)}"
export CAVER_DEFAULT_ACCOUNT="${CAVER_DEFAULT_ACCOUNT:-p57098}"
export CAVER_DEFAULT_RDSS_ROOT="${CAVER_DEFAULT_RDSS_ROOT:-/rdss/${CAVER_DEFAULT_ACCOUNT}/${USER:-euijin1}}"
export CAVER_DEFAULT_RUN_ROOT="${CAVER_DEFAULT_RUN_ROOT:-${CAVER_DEFAULT_RDSS_ROOT}/caver/runs}"
export CAVER_DEFAULT_SLURM_LOG_ROOT="${CAVER_DEFAULT_SLURM_LOG_ROOT:-${CAVER_DEFAULT_RDSS_ROOT}/caver/logs/slurm}"
export CAVER_DEFAULT_RUNTIME_LOG_ROOT="${CAVER_DEFAULT_RUNTIME_LOG_ROOT:-${CAVER_DEFAULT_RDSS_ROOT}/caver/runtime_logs}"
export CAVER_DEFAULT_TEMPLATE="${CAVER_DEFAULT_TEMPLATE:-${CAVER_REPO_ROOT}/metadata/manifest.template.json}"
export CAVER_DEFAULT_SCHEMA="${CAVER_DEFAULT_SCHEMA:-${CAVER_REPO_ROOT}/metadata/manifest.schema.json}"
export CAVER_DEFAULT_SOURCE_ROOT="${CAVER_DEFAULT_SOURCE_ROOT:-${CAVER_REPO_ROOT}/third_party/src}"
export CAVER_DEFAULT_VENV_ROOT="${CAVER_DEFAULT_VENV_ROOT:-${CAVER_REPO_ROOT}/third_party/venvs}"
export CAVER_DEFAULT_OPENPI_CACHE_ROOT="${CAVER_DEFAULT_OPENPI_CACHE_ROOT:-${CAVER_REPO_ROOT}/third_party/openpi-cache}"
export CAVER_DEFAULT_RAY_TMP_ROOT="${CAVER_DEFAULT_RAY_TMP_ROOT:-${CAVER_DEFAULT_RDSS_ROOT}/ray}"

ensure_module_command() {
  if type module >/dev/null 2>&1; then
    return 0
  fi

  local lmod_init="/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/bash"
  if [ -r "${lmod_init}" ]; then
    # shellcheck disable=SC1090
    source "${lmod_init}"
  fi

  if ! type module >/dev/null 2>&1; then
    echo "error: module command is unavailable in this shell" >&2
    return 1
  fi
}

require_command() {
  local command_name="${1:?missing command name}"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "error: required command not found: ${command_name}" >&2
    return 1
  fi
}

ensure_directory() {
  local directory="${1:?missing directory path}"
  mkdir -p -- "${directory}"
}

timestamp_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

sanitize_token() {
  local value="${1:-unset}"
  value="$(printf "%s" "${value}" | tr '[:upper:]' '[:lower:]')"
  printf "%s" "${value}" | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

make_run_id() {
  local stage="${1:?missing stage}"
  local method="${2:?missing method}"
  local task="${3:?missing task}"
  local seed="${4:?missing seed}"
  local budget="${5:?missing budget}"
  local stamp="${6:-$(timestamp_utc)}"

  printf "%s__%s__%s__seed%s__budget%s__%s" \
    "$(sanitize_token "${stage}")" \
    "$(sanitize_token "${method}")" \
    "$(sanitize_token "${task}")" \
    "$(sanitize_token "${seed}")" \
    "$(sanitize_token "${budget}")" \
    "${stamp}"
}

git_commit_or_empty() {
  git -C "${CAVER_REPO_ROOT}" rev-parse HEAD 2>/dev/null || true
}

git_branch_or_empty() {
  git -C "${CAVER_REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

caver_default_loopback_port() {
  local fallback_port="${CAVER_DEFAULT_LOOPBACK_PORT_FALLBACK:-8000}"
  local base_port="${CAVER_DEFAULT_LOOPBACK_PORT_BASE:-18000}"
  local port_span="${CAVER_DEFAULT_LOOPBACK_PORT_SPAN:-20000}"

  if [ -z "${SLURM_JOB_ID:-}" ]; then
    printf "%s" "${fallback_port}"
    return 0
  fi

  printf "%s" "$((base_port + (SLURM_JOB_ID % port_span)))"
}

caver_default_ray_tmpdir() {
  printf "%s/%s" "${CAVER_DEFAULT_RAY_TMP_ROOT}" "${SLURM_JOB_ID:-manual}"
}

caver_parse_slurm_count() {
  local raw_value="${1:-}"
  local value="${raw_value//[[:space:]]/}"
  if [ -z "${value}" ]; then
    return 1
  fi
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${value}"
    return 0
  fi
  if [[ "${value}" =~ ^([0-9]+)\(x[0-9]+\)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${value}" =~ ^([^:]+:)*([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

caver_allocated_gpu_count() {
  local env_name=""
  local parsed=""
  for env_name in SLURM_GPUS_ON_NODE SLURM_GPUS_PER_NODE SLURM_GPUS; do
    if parsed="$(caver_parse_slurm_count "${!env_name:-}")"; then
      printf '%s\n' "${parsed}"
      return 0
    fi
  done

  local visible="${CUDA_VISIBLE_DEVICES:-}"
  if [ -z "${visible}" ] || [ "${visible}" = "NoDevFiles" ] || [ "${visible}" = "void" ]; then
    return 1
  fi

  local count=0
  local device=""
  local -a devices=()
  IFS=',' read -r -a devices <<< "${visible}"
  for device in "${devices[@]}"; do
    device="${device//[[:space:]]/}"
    if [ -n "${device}" ]; then
      count=$((count + 1))
    fi
  done
  if [ "${count}" -gt 0 ]; then
    printf '%s\n' "${count}"
    return 0
  fi
  return 1
}
