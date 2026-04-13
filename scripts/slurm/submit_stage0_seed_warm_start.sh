#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_seed_warm_start.sh [options]

Options:
  --manifest-path PATH       Stage-0 partition manifest
                             (default: metadata/stage0/libero_stage0_partitions.json)
  --family-ids IDS           Optional comma-separated proxy family ids
  --max-contexts N           Optional maximum selected contexts
  --context-offset N         Optional manifest offset (default: 0)
  --libero-gl-backend NAME   Simulator GL backend: egl or osmesa (default: osmesa)
  --partition NAME           Slurm partition (default: gpu-l40s)
  --qos NAME                 Slurm QoS (default: normal)
  --gpu-type TYPE            GPU type (default: l40s)
  --time LIMIT               Slurm time limit (default: 08:00:00)
  --cpus COUNT               CPU request (default: 16)
  --mem SIZE                 Memory request (default: 128G)
  --seed INTEGER             Evaluation seed (default: 7)
  --config-name NAME         Optional custom OpenPI config
  --pretrained-path PATH     Optional custom OpenPI checkpoint path
  --transition-trace-path PATH
                             Optional run-local JSONL chunk-trace path
  --dry-run                  Generate run scaffolding but do not submit
  -h, --help                 Show this message
EOF
}

manifest_path="${CAVER_REPO_ROOT}/metadata/stage0/libero_stage0_partitions.json"
family_ids=""
max_contexts=""
context_offset="0"
libero_gl_backend="osmesa"
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
time_limit="08:00:00"
cpus="16"
mem="128G"
seed="7"
config_name=""
pretrained_path=""
transition_trace_path=""
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --manifest-path)
      manifest_path="${2:?missing value for --manifest-path}"
      shift 2
      ;;
    --family-ids)
      family_ids="${2:?missing value for --family-ids}"
      shift 2
      ;;
    --max-contexts)
      max_contexts="${2:?missing value for --max-contexts}"
      shift 2
      ;;
    --context-offset)
      context_offset="${2:?missing value for --context-offset}"
      shift 2
      ;;
    --libero-gl-backend)
      libero_gl_backend="${2:?missing value for --libero-gl-backend}"
      shift 2
      ;;
    --partition)
      partition="${2:?missing value for --partition}"
      shift 2
      ;;
    --qos)
      qos="${2:?missing value for --qos}"
      shift 2
      ;;
    --gpu-type)
      gpu_type="${2:?missing value for --gpu-type}"
      shift 2
      ;;
    --time)
      time_limit="${2:?missing value for --time}"
      shift 2
      ;;
    --cpus)
      cpus="${2:?missing value for --cpus}"
      shift 2
      ;;
    --mem)
      mem="${2:?missing value for --mem}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing value for --seed}"
      shift 2
      ;;
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --pretrained-path)
      pretrained_path="${2:?missing value for --pretrained-path}"
      shift 2
      ;;
    --transition-trace-path)
      transition_trace_path="${2:?missing value for --transition-trace-path}"
      shift 2
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
      echo "error: unknown option: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

runtime_args=(
  --manifest-path "${manifest_path}"
  --context-offset "${context_offset}"
  --libero-gl-backend "${libero_gl_backend}"
  --seed "${seed}"
)

if [ -n "${family_ids}" ]; then
  runtime_args+=(--family-ids "${family_ids}")
fi
if [ -n "${max_contexts}" ]; then
  runtime_args+=(--max-contexts "${max_contexts}")
fi
if [ -n "${transition_trace_path}" ]; then
  runtime_args+=(--transition-trace-path "${transition_trace_path}")
fi

method_name="seed-warm-start"
if [ -n "${config_name}" ] || [ -n "${pretrained_path}" ]; then
  if [ -z "${config_name}" ] || [ -z "${pretrained_path}" ]; then
    echo "error: custom OpenPI native mode requires both --config-name and --pretrained-path" >&2
    exit 1
  fi
  runtime_args+=(--config-name "${config_name}" --pretrained-path "${pretrained_path}")
  method_name="seed-warm-start-custom"
fi

submit_args=(
  --stage stage0
  --method "${method_name}"
  --task libero-stage0-seed
  --seed "${seed}"
  --budget 0
  --partition "${partition}"
  --qos "${qos}"
  --gpu-type "${gpu_type}"
  --cpus "${cpus}"
  --mem "${mem}"
  --time "${time_limit}"
)

if ((dry_run)); then
  submit_args+=(--dry-run)
fi

"${CAVER_REPO_ROOT}/scripts/slurm/submit_experiment.sh" \
  "${submit_args[@]}" \
  -- \
  "${CAVER_REPO_ROOT}/scripts/stage0/collect_stage0_warm_start.sh" \
  --openpi-native \
  "${runtime_args[@]}"
