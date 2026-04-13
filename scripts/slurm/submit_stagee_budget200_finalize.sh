#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_FINALIZE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_FINALIZE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stagee_budget200_finalize.sh [options]

Required:
  --original-job-script PATH   Original failed Stage-E job.sbatch path
  --original-trace-path PATH   Original large online chunk trace path
  --completed-prefix COUNT     Number of fully completed contexts in the failed prefix
  --tail-results-dir PATH      Results directory from the successful continuation tail

Slurm options:
  --partition NAME             Slurm partition (default: gpu-l40s)
  --qos NAME                   Slurm QoS (default: normal)
  --gpu-type TYPE              GPU type specifier (default: l40s)
  --gpus COUNT                 GPU count (default: 1)
  --time LIMIT                 Slurm time limit (default: 04:00:00)
  --cpus COUNT                 CPU request (default: 8)
  --mem SIZE                   Memory request (default: 128G)
  --exclude NODES              Optional Slurm node exclusion list

Output options:
  --final-run-root PATH        Actual finalized run root (default: /rdss/.../caver/stagee_budget200_finalized)
  --link-run-root PATH         Symlinked run root visible to Stage-E summarizer (default: CAVER runs/)
  --slurm-log-root PATH        Slurm log root (default: /rdss/.../caver/stagee_budget200_finalized/logs)
  --runtime-log-root PATH      Runtime log root for policy server / helper (default: /rdss/.../caver/runtime_logs)

Other:
  --dry-run                    Print derived paths and sbatch command without submitting
  -h, --help                   Show this message
EOF
}

original_job_script=""
original_trace_path=""
completed_prefix=""
tail_results_dir=""

partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="04:00:00"
cpus="8"
mem="128G"
exclude_nodes=""

final_run_root="${CAVER_DEFAULT_RDSS_ROOT}/caver/stagee_budget200_finalized"
link_run_root="${CAVER_DEFAULT_RUN_ROOT}"
slurm_log_root="${final_run_root}/logs"
runtime_log_root="${CAVER_DEFAULT_RDSS_ROOT}/caver/runtime_logs"
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --original-job-script)
      original_job_script="${2:?missing value for --original-job-script}"
      shift 2
      ;;
    --original-trace-path)
      original_trace_path="${2:?missing value for --original-trace-path}"
      shift 2
      ;;
    --completed-prefix)
      completed_prefix="${2:?missing value for --completed-prefix}"
      shift 2
      ;;
    --tail-results-dir)
      tail_results_dir="${2:?missing value for --tail-results-dir}"
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
    --exclude)
      exclude_nodes="${2:?missing value for --exclude}"
      shift 2
      ;;
    --final-run-root)
      final_run_root="${2:?missing value for --final-run-root}"
      shift 2
      ;;
    --link-run-root)
      link_run_root="${2:?missing value for --link-run-root}"
      shift 2
      ;;
    --slurm-log-root)
      slurm_log_root="${2:?missing value for --slurm-log-root}"
      shift 2
      ;;
    --runtime-log-root)
      runtime_log_root="${2:?missing value for --runtime-log-root}"
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

require_command python3
require_command sbatch

for required_value in \
  "${original_job_script}" \
  "${original_trace_path}" \
  "${completed_prefix}" \
  "${tail_results_dir}"
do
  if [ -z "${required_value}" ]; then
    echo "error: missing required arguments" >&2
    usage >&2
    exit 1
  fi
done

original_job_script="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${original_job_script}")"
original_trace_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${original_trace_path}")"
tail_results_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${tail_results_dir}")"
final_run_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${final_run_root}")"
link_run_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${link_run_root}")"
slurm_log_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${slurm_log_root}")"
runtime_log_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${runtime_log_root}")"

if [ ! -f "${original_job_script}" ]; then
  echo "error: original job script not found: ${original_job_script}" >&2
  exit 1
fi
if [ ! -f "${original_trace_path}" ]; then
  echo "error: original trace not found: ${original_trace_path}" >&2
  exit 1
fi
if [ ! -d "${tail_results_dir}" ]; then
  echo "error: tail results dir not found: ${tail_results_dir}" >&2
  exit 1
fi

run_info="$(
  python3 - "${original_job_script}" <<'PY'
import json
import pathlib
import re
import sys

job_script = pathlib.Path(sys.argv[1]).resolve()
run_dir = job_script.parent
pattern = re.compile(
    r"^stagee__(?P<method>real-only-round|caver-round)__"
    r"(?P<target>.+)__seed(?P<seed>\d+)__budget(?P<budget>\d+)__"
    r"(?P<stamp>\d{8}T\d{6}Z)$"
)
match = pattern.match(run_dir.name)
if match is None:
    raise SystemExit(f"error: could not parse Stage-E run dir: {run_dir.name}")
payload = {
    "method": match.group("method"),
    "target": match.group("target"),
    "seed": int(match.group("seed")),
    "budget": int(match.group("budget")),
    "original_run_dir": str(run_dir),
}
print(json.dumps(payload, sort_keys=True))
PY
)"

run_stamp="$(timestamp_utc)"
run_name="$(
  python3 - "${run_info}" "${run_stamp}" <<'PY'
import json
import sys
info = json.loads(sys.argv[1])
stamp = sys.argv[2]
print(f"stagee__{info['method']}__{info['target']}__seed{info['seed']}__budget{info['budget']}__{stamp}")
PY
)"

actual_run_dir="${final_run_root}/${run_name}"
results_dir="${actual_run_dir}/results"
link_run_dir="${link_run_root}/${run_name}"
job_script_path="${actual_run_dir}/job.sbatch"
merge_manifest_path="${actual_run_dir}/merge_manifest.json"

ensure_directory "${actual_run_dir}"
ensure_directory "${results_dir}"
ensure_directory "${slurm_log_root}"
ensure_directory "${runtime_log_root}"
ensure_directory "$(dirname -- "${link_run_dir}")"

ln -sfn "${actual_run_dir}" "${link_run_dir}"

python3 - "${merge_manifest_path}" "${run_info}" "${original_job_script}" "${original_trace_path}" "${completed_prefix}" "${tail_results_dir}" "${results_dir}" "${runtime_log_root}" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
run_info = json.loads(sys.argv[2])
payload = {
    "stage": "stageE_budget200_finalize",
    "run_info": run_info,
    "original_job_script": sys.argv[3],
    "original_trace_path": sys.argv[4],
    "completed_prefix": int(sys.argv[5]),
    "tail_results_dir": sys.argv[6],
    "merged_results_dir": sys.argv[7],
    "runtime_log_root": sys.argv[8],
}
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

job_name="$(
  python3 - "${run_info}" <<'PY'
import json
import sys
info = json.loads(sys.argv[1])
print("caver-stageE-finalize-real" if info["method"] == "real-only-round" else "caver-stageE-finalize-caver")
PY
)"

cat > "${job_script_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${CAVER_DEFAULT_ACCOUNT}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --job-name=${job_name}
#SBATCH --gres=gpu:${gpu_type}:${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}
#SBATCH --output=${slurm_log_root}/${run_name}-%j.out
#SBATCH --error=${slurm_log_root}/${run_name}-%j.err
$(if [ -n "${exclude_nodes}" ]; then printf '#SBATCH --exclude=%s\n' "${exclude_nodes}"; fi)

set -euo pipefail

cd ${CAVER_REPO_ROOT}
export CAVER_RUN_ID=${run_name}
export CAVER_RUN_DIR=${actual_run_dir}
export CAVER_MANIFEST_PATH=${merge_manifest_path}
export CAVER_DEFAULT_RUNTIME_LOG_ROOT=${runtime_log_root}

python3 ${CAVER_REPO_ROOT}/scripts/stagee/finalize_stagee_budget200_cell.py \\
  --original-job-script ${original_job_script} \\
  --original-trace-path ${original_trace_path} \\
  --completed-prefix ${completed_prefix} \\
  --tail-results-dir ${tail_results_dir} \\
  --merged-results-dir ${results_dir} \\
  --runtime-log-root ${runtime_log_root}
EOF

printf 'run info: %s\n' "${run_info}"
printf 'actual run dir: %s\n' "${actual_run_dir}"
printf 'symlinked run dir: %s\n' "${link_run_dir}"
printf 'merge manifest: %s\n' "${merge_manifest_path}"
printf 'job script: %s\n' "${job_script_path}"
printf 'runtime log root: %s\n' "${runtime_log_root}"

submit_cmd=(sbatch "${job_script_path}")
printf 'submit command:'
printf ' %q' "${submit_cmd[@]}"
printf '\n'

if (( dry_run )); then
  exit 0
fi

"${submit_cmd[@]}"
