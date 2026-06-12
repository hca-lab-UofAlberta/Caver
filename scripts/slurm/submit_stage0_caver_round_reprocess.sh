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
  --gpus COUNT              Use 0 to omit --gres and run as CPU-only reprocessing.
  --time LIMIT
  --cpus COUNT
  --mem SIZE
  --run-root PATH
  --log-root PATH

Reprocess options:
  --skip-backend-train
      Rebuild selector/admission/DR artifacts only; do not run replay conversion or backend training.
  --admission-policy NAME
      Override the source admission policy for the reprocessed final artifacts.
  --admission-kappa VALUE
      Optional kappa override for LCB-style admission policies.
  --admission-threshold VALUE
      Optional acceptance-threshold override for LCB-style admission policies.
  --top-m-success-count COUNT
      For caver_top_m_success, admit top M successful contexts by executed LCB.
  --family-min-success-count COUNT
      For caver_family_balanced_success, admit this many successful contexts per proxy family.
  --rescue-family-ids IDS
      For caver_hard_family_rescue, comma-separated proxy families eligible for near-miss rescue.
  --rescue-per-family-count COUNT
      For caver_hard_family_rescue, failed near misses to admit per rescued family.
  --repair-min-trace-records COUNT
      For caver_family_segment_repair, minimum repaired prefix chunks.
  --repair-max-trace-records COUNT
      For caver_family_segment_repair, maximum repaired prefix chunks.
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
admission_policy=""
admission_kappa=""
admission_threshold=""
top_m_success_count=""
family_min_success_count=""
rescue_family_ids=""
rescue_per_family_count=""
repair_min_trace_records=""
repair_max_trace_records=""
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
    --admission-policy)
      admission_policy="${2:?missing value for --admission-policy}"
      shift 2
      ;;
    --admission-kappa)
      admission_kappa="${2:?missing value for --admission-kappa}"
      shift 2
      ;;
    --admission-threshold)
      admission_threshold="${2:?missing value for --admission-threshold}"
      shift 2
      ;;
    --top-m-success-count)
      top_m_success_count="${2:?missing value for --top-m-success-count}"
      shift 2
      ;;
    --family-min-success-count)
      family_min_success_count="${2:?missing value for --family-min-success-count}"
      shift 2
      ;;
    --rescue-family-ids)
      rescue_family_ids="${2:?missing value for --rescue-family-ids}"
      shift 2
      ;;
    --rescue-per-family-count)
      rescue_per_family_count="${2:?missing value for --rescue-per-family-count}"
      shift 2
      ;;
    --repair-min-trace-records)
      repair_min_trace_records="${2:?missing value for --repair-min-trace-records}"
      shift 2
      ;;
    --repair-max-trace-records)
      repair_max_trace_records="${2:?missing value for --repair-max-trace-records}"
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
  "${source_results_dir}"; do
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
gres_directive=""
if [ "${gpus}" != "0" ]; then
  gres_directive="#SBATCH --gres=gpu:${gpu_type}:${gpus}"
fi

job_cmd_payload="$(python3 - \
  "${source_job_script}" \
  "${results_dir}" \
  "${skip_backend_train}" \
  "${admission_policy}" \
  "${admission_kappa}" \
  "${admission_threshold}" \
  "${top_m_success_count}" \
  "${family_min_success_count}" \
  "${rescue_family_ids}" \
  "${rescue_per_family_count}" \
  "${repair_min_trace_records}" \
  "${repair_max_trace_records}" <<'PY'
import pathlib
import shlex
import sys

source_job_script = pathlib.Path(sys.argv[1]).resolve()
results_dir = pathlib.Path(sys.argv[2]).resolve()
skip_backend_train = bool(int(sys.argv[3]))
admission_policy = sys.argv[4]
admission_kappa = sys.argv[5]
admission_threshold = sys.argv[6]
top_m_success_count = sys.argv[7]
family_min_success_count = sys.argv[8]
rescue_family_ids = sys.argv[9]
rescue_per_family_count = sys.argv[10]
repair_min_trace_records = sys.argv[11]
repair_max_trace_records = sys.argv[12]

command_line = None
command_kind = None
with source_job_script.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        if "run_stage0_caver_lagged_budget.py" in raw_line:
            command_line = raw_line.strip()
            command_kind = "lagged"
        elif command_line is None and "run_stage0_caver_round.sh" in raw_line:
            command_line = raw_line.strip()
            command_kind = "round"
if not command_line:
    raise SystemExit(f"could not find Stage-E CAVER runner command in {source_job_script}")

args = shlex.split(command_line)

def remove_option(tokens: list[str], option: str, *, has_value: bool = True) -> list[str]:
    updated: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == option:
            index += 2 if has_value else 1
            continue
        updated.append(tokens[index])
        index += 1
    return updated

def set_option(tokens: list[str], option: str, value: str) -> list[str]:
    tokens = remove_option(tokens, option, has_value=True)
    tokens.extend([option, value])
    return tokens

def ensure_flag(tokens: list[str], option: str) -> list[str]:
    if option not in tokens:
        tokens.append(option)
    return tokens

for index, value in enumerate(args[:-1]):
    if value == "--results-dir":
        args[index + 1] = str(results_dir)
        break
else:
    raise SystemExit("source command does not contain --results-dir")

for option in [
    "--admission-policy",
    "--admission-kappa",
    "--admission-threshold",
    "--top-m-success-count",
    "--family-min-success-count",
    "--rescue-family-ids",
    "--rescue-per-family-count",
    "--repair-min-trace-records",
    "--repair-max-trace-records",
]:
    args = remove_option(args, option, has_value=True)

if admission_policy:
    args.extend(["--admission-policy", admission_policy])
if admission_kappa:
    args.extend(["--admission-kappa", admission_kappa])
if admission_threshold:
    args.extend(["--admission-threshold", admission_threshold])
if top_m_success_count:
    args.extend(["--top-m-success-count", top_m_success_count])
if family_min_success_count:
    args.extend(["--family-min-success-count", family_min_success_count])
if rescue_family_ids:
    args.extend(["--rescue-family-ids", rescue_family_ids])
if rescue_per_family_count:
    args.extend(["--rescue-per-family-count", rescue_per_family_count])
if repair_min_trace_records:
    args.extend(["--repair-min-trace-records", repair_min_trace_records])
if repair_max_trace_records:
    args.extend(["--repair-max-trace-records", repair_max_trace_records])

if command_kind == "lagged":
    args = ensure_flag(args, "--resume-existing")
    args = remove_option(args, "--dry-run", has_value=False)
    args = remove_option(args, "--finalizer-skip-backend-train", has_value=False)
    args = remove_option(args, "--finalizer-skip-backend-update", has_value=False)
    if skip_backend_train:
        args = ensure_flag(args, "--finalizer-skip-backend-train")
else:
    args = ensure_flag(args, "--skip-online")
    if skip_backend_train:
        args = ensure_flag(args, "--skip-backend-train")

print(command_kind)
print(shlex.join(args))
PY
)"
command_kind="$(printf '%s\n' "${job_cmd_payload}" | sed -n '1p')"
job_cmd_str="$(printf '%s\n' "${job_cmd_payload}" | sed -n '2,$p')"

cat >"${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${CAVER_DEFAULT_ACCOUNT}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --job-name=${job_name}
${gres_directive}
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
if [ "${command_kind}" = "lagged" ]; then
  ln -sfn ${source_results_dir}/lagged_rounds ${results_dir}/lagged_rounds
else
  ln -sfn ${source_results_dir}/caver_online_eval.json ${results_dir}/caver_online_eval.json
  ln -sfn ${source_results_dir}/caver_online_contexts.jsonl ${results_dir}/caver_online_contexts.jsonl
  ln -sfn ${source_results_dir}/caver_online_chunks.jsonl ${results_dir}/caver_online_chunks.jsonl
  if [ -f ${source_results_dir}/caver_online_demo_chunks.jsonl.gz ]; then
    ln -sfn ${source_results_dir}/caver_online_demo_chunks.jsonl.gz ${results_dir}/caver_online_demo_chunks.jsonl.gz
  elif [ -f ${source_results_dir}/caver_online_demo_chunks.jsonl ]; then
    ln -sfn ${source_results_dir}/caver_online_demo_chunks.jsonl ${results_dir}/caver_online_demo_chunks.jsonl.gz
  fi
fi

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
echo "run_dir: ${run_dir}"
echo "results_dir: ${results_dir}"
echo "stdout: ${slurm_stdout}"
echo "stderr: ${slurm_stderr}"
