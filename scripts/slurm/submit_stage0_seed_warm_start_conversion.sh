#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_CONVERT_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_CONVERT_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_seed_warm_start_conversion.sh [options]

Options:
  --trace-path PATH         Stage-0 chunk-trace JSONL path to convert
  --partition NAME          Slurm partition (default: cpu)
  --qos NAME                Slurm QoS (default: normal)
  --time LIMIT              Slurm time limit (default: 08:00:00)
  --cpus COUNT              CPU request (default: 8)
  --mem SIZE                Memory request (default: 128G)
  --seed INTEGER            Logical seed recorded in the manifest (default: 7)
  --max-items-per-shard N   Demo items per output shard (default: 128)
  --run-root PATH           Run directory root (default: runs/)
  --log-root PATH           Slurm log root (default: logs/slurm/)
  --dry-run                 Generate run scaffolding but do not submit
  -h, --help                Show this message
EOF
}

trace_path=""
partition="cpu"
qos="normal"
time_limit="08:00:00"
cpus="8"
mem="128G"
seed="7"
max_items_per_shard="128"
run_root="${CAVER_DEFAULT_RUN_ROOT}"
log_root="${CAVER_DEFAULT_SLURM_LOG_ROOT}"
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --trace-path)
      trace_path="${2:?missing value for --trace-path}"
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
    --max-items-per-shard)
      max_items_per_shard="${2:?missing value for --max-items-per-shard}"
      shift 2
      ;;
    --run-root)
      run_root="${2:?missing value for --run-root}"
      shift 2
      ;;
    --log-root)
      log_root="${2:?missing value for --log-root}"
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

if [ -z "${trace_path}" ]; then
  echo "error: --trace-path is required" >&2
  usage >&2
  exit 1
fi

require_command sbatch
require_command python3

trace_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${trace_path}")"
stamp="$(timestamp_utc)"
run_id="$(make_run_id "stage0" "seed-demo-convert" "libero-stage0-seed" "${seed}" "0" "${stamp}")"
job_name="caver-stage0-seed-convert"
run_dir="${run_root}/${run_id}"
results_dir="${run_dir}/results"
job_script="${run_dir}/job.sbatch"
manifest_path="${run_dir}/manifest.json"
slurm_stdout="${log_root}/${run_id}-%j.out"
slurm_stderr="${log_root}/${run_id}-%j.err"
output_manifest="${results_dir}/stage0_seed_warm_start_demo.manifest.json"
summary_path="${results_dir}/stage0_seed_warm_start_demo.summary.json"

ensure_directory "${run_dir}"
ensure_directory "${results_dir}"
ensure_directory "${log_root}"

python3 "${CAVER_REPO_ROOT}/scripts/manifest/create_manifest.py" \
  --output "${manifest_path}" \
  --run-id "${run_id}" \
  --stage stage0 \
  --method seed-demo-convert \
  --task libero-stage0-seed \
  --seed "${seed}" \
  --budget 0 \
  --account "${CAVER_DEFAULT_ACCOUNT}" \
  --partition "${partition}" \
  --qos "${qos}" \
  --gpu-type none \
  --gpus 0 \
  --cpus-per-task "${cpus}" \
  --memory "${mem}" \
  --time-limit "${time_limit}" \
  --run-dir "${run_dir}" \
  --job-script "${job_script}" \
  --slurm-stdout "${slurm_stdout}" \
  --slurm-stderr "${slurm_stderr}" \
  --job-name "${job_name}" \
  --template "${CAVER_DEFAULT_TEMPLATE}"

cat >"${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${CAVER_DEFAULT_ACCOUNT}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}
#SBATCH --output=${slurm_stdout}
#SBATCH --error=${slurm_stderr}

set -euo pipefail

cd ${CAVER_REPO_ROOT}
export CAVER_RUN_ID=${run_id}
export CAVER_RUN_DIR=${run_dir}
export CAVER_MANIFEST_PATH=${manifest_path}

${CAVER_REPO_ROOT}/scripts/stage0/convert_stage0_trace_to_rlinf_demo.sh \
  --trace-path ${trace_path} \
  --output-mode sharded_manifest \
  --max-items-per-shard ${max_items_per_shard} \
  --output-path ${output_manifest} \
  --summary-path ${summary_path}
EOF
chmod +x "${job_script}"

if ((dry_run)); then
  echo "created run scaffolding:"
  echo "  run_id: ${run_id}"
  echo "  run_dir: ${run_dir}"
  echo "  manifest: ${manifest_path}"
  echo "  job_script: ${job_script}"
  echo "  stdout: ${slurm_stdout}"
  echo "  stderr: ${slurm_stderr}"
  exit 0
fi

submission_output="$(sbatch "${job_script}")"
echo "${submission_output}"
