#!/usr/bin/env bash
set -euo pipefail

_CAVER_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_experiment.sh [options] -- <command ...>

Required options:
  --stage NAME
  --method NAME
  --task NAME
  --seed INTEGER
  --budget INTEGER

Optional resource options:
  --account ACCOUNT     Slurm account (default: p57098)
  --partition NAME      Partition (default: gpu-l40s)
  --qos NAME            QoS (default: normal)
  --gpu-type TYPE       GPU type name (default: l40s)
  --gpus COUNT          GPU count (default: 1)
  --cpus COUNT          CPU count (default: 16)
  --mem SIZE            Memory request (default: 64G)
  --time LIMIT          Time limit (default: 1-00:00:00)
  --job-name NAME       Override the generated job name
  --run-root PATH       Run directory root (default: runs/)
  --log-root PATH       Slurm log root (default: logs/slurm/)
  --dry-run             Create files but do not submit with sbatch
  -h, --help            Show this message
EOF
}

stage=""
method=""
task=""
seed=""
budget=""
account="${CAVER_DEFAULT_ACCOUNT}"
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
cpus="16"
mem="64G"
time_limit="1-00:00:00"
job_name=""
run_root="${CAVER_DEFAULT_RUN_ROOT}"
log_root="${CAVER_DEFAULT_SLURM_LOG_ROOT}"
dry_run=0

while (($# > 0)); do
  case "$1" in
    --stage)
      stage="${2:?missing value for --stage}"
      shift 2
      ;;
    --method)
      method="${2:?missing value for --method}"
      shift 2
      ;;
    --task)
      task="${2:?missing value for --task}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing value for --seed}"
      shift 2
      ;;
    --budget)
      budget="${2:?missing value for --budget}"
      shift 2
      ;;
    --account)
      account="${2:?missing value for --account}"
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
    --gpus)
      gpus="${2:?missing value for --gpus}"
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
    --time)
      time_limit="${2:?missing value for --time}"
      shift 2
      ;;
    --job-name)
      job_name="${2:?missing value for --job-name}"
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
    --)
      shift
      break
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${stage}" ] || [ -z "${method}" ] || [ -z "${task}" ] || [ -z "${seed}" ] || [ -z "${budget}" ]; then
  echo "error: stage, method, task, seed, and budget are required" >&2
  usage >&2
  exit 1
fi

if (($# == 0)); then
  echo "error: command is required after --" >&2
  usage >&2
  exit 1
fi

require_command sbatch
require_command python3

stamp="$(timestamp_utc)"
run_id="$(make_run_id "${stage}" "${method}" "${task}" "${seed}" "${budget}" "${stamp}")"
if [ -z "${job_name}" ]; then
  job_name="caver-$(sanitize_token "${stage}")-$(sanitize_token "${method}")-$(sanitize_token "${task}")"
fi
if [ "${#job_name}" -gt 40 ]; then
  job_name="$(printf "%s" "${job_name}" | cut -c1-40)"
  job_name="${job_name%-}"
fi

run_dir="${run_root}/${run_id}"
job_script="${run_dir}/job.sbatch"
manifest_path="${run_dir}/manifest.json"
slurm_stdout="${log_root}/${run_id}-%j.out"
slurm_stderr="${log_root}/${run_id}-%j.err"

ensure_directory "${run_dir}"
ensure_directory "${log_root}"

python3 "${CAVER_REPO_ROOT}/scripts/manifest/create_manifest.py" \
  --output "${manifest_path}" \
  --run-id "${run_id}" \
  --stage "${stage}" \
  --method "${method}" \
  --task "${task}" \
  --seed "${seed}" \
  --budget "${budget}" \
  --account "${account}" \
  --partition "${partition}" \
  --qos "${qos}" \
  --gpu-type "${gpu_type}" \
  --gpus "${gpus}" \
  --cpus-per-task "${cpus}" \
  --memory "${mem}" \
  --time-limit "${time_limit}" \
  --run-dir "${run_dir}" \
  --job-script "${job_script}" \
  --slurm-stdout "${slurm_stdout}" \
  --slurm-stderr "${slurm_stderr}" \
  --job-name "${job_name}" \
  --template "${CAVER_DEFAULT_TEMPLATE}"

command_args=("$@")
printf -v command_string "%q " "${command_args[@]}"
command_string="${command_string% }"

cat >"${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${account}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --job-name=${job_name}
#SBATCH --gres=gpu:${gpu_type}:${gpus}
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

${command_string}
EOF
if ! chmod +x "${job_script}" 2>/dev/null; then
  echo "warning: unable to chmod job script; continuing because sbatch only needs read access: ${job_script}" >&2
fi

if [ "${dry_run}" = "1" ]; then
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
