#!/usr/bin/env bash
set -euo pipefail

_CAVER_OPENPI_CONVERT_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_OPENPI_CONVERT_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_openpi_checkpoint_conversion.sh [options]

Options:
  --checkpoint-dir PATH    Source OpenPI JAX checkpoint directory
  --config-name NAME       OpenPI config name (default: pi05_libero)
  --output-path PATH       Output directory (default: <checkpoint-dir>_pytorch)
  --precision NAME         Output precision: float32 or bfloat16 (default: bfloat16)
  --partition NAME         Slurm partition (default: cpu)
  --qos NAME               Slurm QoS (default: normal)
  --time LIMIT             Slurm time limit (default: 08:00:00)
  --cpus COUNT             CPU request (default: 8)
  --mem SIZE               Memory request (default: 192G)
  --run-root PATH          Run directory root (default: runs/)
  --log-root PATH          Slurm log root (default: logs/slurm/)
  --dry-run                Generate run scaffolding but do not submit
  -h, --help               Show this message
EOF
}

checkpoint_dir=""
config_name="pi05_libero"
output_path=""
precision="bfloat16"
partition="cpu"
qos="normal"
time_limit="08:00:00"
cpus="8"
mem="192G"
run_root="${CAVER_DEFAULT_RUN_ROOT}"
log_root="${CAVER_DEFAULT_SLURM_LOG_ROOT}"
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --checkpoint-dir)
      checkpoint_dir="${2:?missing value for --checkpoint-dir}"
      shift 2
      ;;
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --output-path)
      output_path="${2:?missing value for --output-path}"
      shift 2
      ;;
    --precision)
      precision="${2:?missing value for --precision}"
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

if [ -z "${checkpoint_dir}" ]; then
  echo "error: --checkpoint-dir is required" >&2
  usage >&2
  exit 1
fi

case "${precision}" in
  float32|bfloat16)
    ;;
  *)
    echo "error: unsupported --precision value: ${precision}" >&2
    exit 1
    ;;
esac

require_command sbatch
require_command python3

checkpoint_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${checkpoint_dir}")"
if [ ! -d "${checkpoint_dir}" ]; then
  echo "error: checkpoint directory not found: ${checkpoint_dir}" >&2
  exit 1
fi

if [ -z "${output_path}" ]; then
  output_path="${checkpoint_dir}_pytorch"
fi
output_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${output_path}")"

checkpoint_name="$(basename "${checkpoint_dir}")"
stamp="$(timestamp_utc)"
run_id="$(make_run_id "stageD" "openpi-pytorch-convert" "${checkpoint_name}" "0" "0" "${stamp}")"
job_name="caver-openpi-convert"
run_dir="${run_root}/${run_id}"
results_dir="${run_dir}/results"
job_script="${run_dir}/job.sbatch"
manifest_path="${run_dir}/manifest.json"
slurm_stdout="${log_root}/${run_id}-%j.out"
slurm_stderr="${log_root}/${run_id}-%j.err"

ensure_directory "${run_dir}"
ensure_directory "${results_dir}"
ensure_directory "${log_root}"

python3 "${CAVER_REPO_ROOT}/scripts/manifest/create_manifest.py" \
  --output "${manifest_path}" \
  --run-id "${run_id}" \
  --stage stageD \
  --method openpi-pytorch-convert \
  --task "${checkpoint_name}" \
  --seed 0 \
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

${CAVER_REPO_ROOT}/scripts/openpi/convert_openpi_checkpoint_to_pytorch.sh \
  --checkpoint-dir ${checkpoint_dir} \
  --config-name ${config_name} \
  --output-path ${output_path} \
  --precision ${precision}
EOF
chmod +x "${job_script}"

if ((dry_run)); then
  echo "created run scaffolding:"
  echo "  run_id: ${run_id}"
  echo "  run_dir: ${run_dir}"
  echo "  manifest: ${manifest_path}"
  echo "  job_script: ${job_script}"
  echo "  output_path: ${output_path}"
  echo "  stdout: ${slurm_stdout}"
  echo "  stderr: ${slurm_stderr}"
  exit 0
fi

submission_output="$(sbatch "${job_script}")"
echo "${submission_output}"
