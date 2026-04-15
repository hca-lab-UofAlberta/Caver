#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_caver_round_reprocess.sh --source-run-dir PATH [options]

Required options:
  --source-run-dir PATH
      Completed Stage-E CAVER round directory containing `job.sbatch` and `results/`.

Slurm options:
  --dependency SPEC
  --partition NAME
  --qos NAME
  --gpu-type TYPE
  --gpus COUNT
  --time LIMIT
  --cpus COUNT
  --mem SIZE
  --run-root PATH
  --log-root PATH

Reprocess options:
  --skip-backend-train
      Rebuild selector/admission/DR artifacts only; do not run replay conversion or backend training.
  --dry-run
  -h, --help
EOF
}

dependency=""
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="02:00:00"
cpus="8"
mem="96G"
run_root="${CAVER_DEFAULT_RDSS_ROOT}/caver/runs"
log_root="${CAVER_DEFAULT_RDSS_ROOT}/caver/logs/slurm"

source_run_dir=""
skip_backend_train=0
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --source-run-dir)
      source_run_dir="${2:?missing value for --source-run-dir}"
      shift 2
      ;;
    --dependency)
      dependency="${2:?missing value for --dependency}"
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
    --skip-backend-train)
      skip_backend_train=1
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
      echo "error: unknown option: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${source_run_dir}" ]; then
  echo "error: --source-run-dir is required" >&2
  usage >&2
  exit 1
fi

require_command sbatch
require_command python3

source_run_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${source_run_dir}")"
source_results_dir="${source_run_dir}/results"
source_job_script="${source_run_dir}/job.sbatch"
source_manifest="${source_run_dir}/manifest.json"

for required_path in \
  "${source_job_script}" \
  "${source_results_dir}/caver_online_eval.json" \
  "${source_results_dir}/caver_online_contexts.jsonl" \
  "${source_results_dir}/caver_online_chunks.jsonl"; do
  if [ ! -e "${required_path}" ]; then
    echo "error: required source artifact missing: ${required_path}" >&2
    exit 1
  fi
done

source_run_id="$(basename -- "${source_run_dir}")"
seed_token="$(printf '%s' "${source_run_id}" | sed -n 's/.*__seed\([^_]*\)__.*/\1/p')"
budget_token="$(printf '%s' "${source_run_id}" | sed -n 's/.*__budget\([^_]*\)__.*/\1/p')"
if [ -z "${seed_token}" ]; then
  seed_token="reuse"
fi
if [ -z "${budget_token}" ]; then
  budget_token="reuse"
fi

stamp="$(timestamp_utc)"
run_id="$(make_run_id "stageE" "caver-reprocess" "${source_run_id}" "${seed_token}" "${budget_token}" "${stamp}")"
job_name="caver-stageE-reprocess"
run_dir="${run_root}/${run_id}"
results_dir="${run_dir}/results"
job_script="${run_dir}/job.sbatch"
manifest_out="${run_dir}/manifest.json"
slurm_stdout="${log_root}/${run_id}-%j.out"
slurm_stderr="${log_root}/${run_id}-%j.err"

ensure_directory "${run_dir}"
ensure_directory "${results_dir}"
ensure_directory "${log_root}"

python3 "${CAVER_REPO_ROOT}/scripts/manifest/create_manifest.py" \
  --output "${manifest_out}" \
  --run-id "${run_id}" \
  --stage stageE \
  --method caver-reprocess \
  --task "source-${source_run_id}" \
  --seed "${seed_token}" \
  --budget "${budget_token}" \
  --account "${CAVER_DEFAULT_ACCOUNT}" \
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

dependency_directive=""
if [ -n "${dependency}" ]; then
  dependency_directive="#SBATCH --dependency=${dependency}"
fi

job_cmd_str="$(python3 - "${source_job_script}" "${results_dir}" "${skip_backend_train}" <<'PY'
import pathlib
import shlex
import sys

source_job_script = pathlib.Path(sys.argv[1]).resolve()
results_dir = pathlib.Path(sys.argv[2]).resolve()
skip_backend_train = bool(int(sys.argv[3]))

command_line = None
with source_job_script.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        if "run_stage0_caver_round.sh" in raw_line:
            command_line = raw_line.strip()
if not command_line:
    raise SystemExit(f"could not find run_stage0_caver_round.sh command in {source_job_script}")

args = shlex.split(command_line)
for index, value in enumerate(args[:-1]):
    if value == "--results-dir":
        args[index + 1] = str(results_dir)
        break
else:
    raise SystemExit("source command does not contain --results-dir")

if "--skip-online" not in args:
    args.append("--skip-online")
if skip_backend_train and "--skip-backend-train" not in args:
    args.append("--skip-backend-train")

print(shlex.join(args))
PY
)"

cat >"${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${CAVER_DEFAULT_ACCOUNT}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --job-name=${job_name}
#SBATCH --gres=gpu:${gpu_type}:${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}
#SBATCH --output=${slurm_stdout}
#SBATCH --error=${slurm_stderr}
${dependency_directive}

set -euo pipefail

cd ${CAVER_REPO_ROOT}
export CAVER_RUN_ID=${run_id}
export CAVER_RUN_DIR=${run_dir}
export CAVER_MANIFEST_PATH=${manifest_out}

mkdir -p ${results_dir}
ln -sfn ${source_results_dir}/caver_online_eval.json ${results_dir}/caver_online_eval.json
ln -sfn ${source_results_dir}/caver_online_contexts.jsonl ${results_dir}/caver_online_contexts.jsonl
ln -sfn ${source_results_dir}/caver_online_chunks.jsonl ${results_dir}/caver_online_chunks.jsonl

${job_cmd_str}
EOF
if ! chmod +x "${job_script}" 2>/dev/null; then
  printf 'warning: could not chmod +x %s; continuing because sbatch only needs a readable script\n' "${job_script}" >&2
fi

if ((dry_run)); then
  echo "created run scaffolding:"
  echo "  source_run_dir: ${source_run_dir}"
  echo "  source_manifest: ${source_manifest}"
  echo "  run_id: ${run_id}"
  echo "  run_dir: ${run_dir}"
  echo "  results_dir: ${results_dir}"
  echo "  manifest: ${manifest_out}"
  echo "  job_script: ${job_script}"
  echo "  stdout: ${slurm_stdout}"
  echo "  stderr: ${slurm_stderr}"
  exit 0
fi

submission_output="$(sbatch "${job_script}")"
echo "${submission_output}"
