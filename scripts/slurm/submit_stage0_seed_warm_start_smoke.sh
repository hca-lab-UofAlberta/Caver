#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGE0_SMOKE_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGE0_SMOKE_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_seed_warm_start_smoke.sh [options]

Options:
  --config-name NAME         RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --model-path PATH          Converted OpenPI PyTorch checkpoint directory
  --demo-manifest PATH       Stage-0 warm-start demo manifest (default: 10-item smoke manifest)
  --dependency SPEC          Slurm dependency spec, for example afterok:5393
  --partition NAME           Slurm partition (default: gpu-l40s)
  --qos NAME                 Slurm QoS (default: normal)
  --gpu-type TYPE            GPU type specifier (default: l40s)
  --gpus COUNT               GPU count (default: 1)
  --time LIMIT               Slurm time limit (default: 02:00:00)
  --cpus COUNT               CPU request (default: 8)
  --mem SIZE                 Memory request (default: 128G)
  --task-suite NAME          LIBERO suite name (default: libero_goal)
  --task-ids IDS             Comma-separated task ids (default: 0)
  --train-envs COUNT         Train env count (default: 4)
  --eval-envs COUNT          Eval env count (default: 4)
  --max-steps COUNT          Runner max_steps (default: 1)
  --max-epochs COUNT         Runner max_epochs cap (default: 1)
  --rollout-steps COUNT      Train/eval max rollout steps (default: 16)
  --micro-batch COUNT        Actor micro batch size (default: 2)
  --global-batch COUNT       Actor global batch size (default: 4)
  --replay-capacity COUNT    Replay buffer capacity (default: 512)
  --min-buffer-size COUNT    Minimum replay size before update (default: 1)
  --train-actor-steps COUNT  Minimum replay size before actor updates (default: 1)
  --run-root PATH            Run directory root (default: runs/)
  --log-root PATH            Slurm log root (default: logs/slurm/)
  --dry-run                  Generate run scaffolding but do not submit
  -h, --help                 Show this message
EOF
}

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
demo_manifest="/uhome/euijin1/projects/p57098/euijin1/Caver/logs/runtime/stage0_seed_warm_start_demo_smoke.manifest.json"
dependency=""
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="02:00:00"
cpus="8"
mem="128G"
task_suite="libero_goal"
task_ids="0"
train_envs="1"
eval_envs="1"
max_steps="1"
max_epochs="1"
rollout_steps="5"
micro_batch="1"
global_batch="2"
replay_capacity="512"
min_buffer_size="1"
train_actor_steps="1"
run_root="${CAVER_DEFAULT_RUN_ROOT}"
log_root="${CAVER_DEFAULT_SLURM_LOG_ROOT}"
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --model-path)
      model_path="${2:?missing value for --model-path}"
      shift 2
      ;;
    --demo-manifest)
      demo_manifest="${2:?missing value for --demo-manifest}"
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
    --task-suite)
      task_suite="${2:?missing value for --task-suite}"
      shift 2
      ;;
    --task-ids)
      task_ids="${2:?missing value for --task-ids}"
      shift 2
      ;;
    --train-envs)
      train_envs="${2:?missing value for --train-envs}"
      shift 2
      ;;
    --eval-envs)
      eval_envs="${2:?missing value for --eval-envs}"
      shift 2
      ;;
    --max-steps)
      max_steps="${2:?missing value for --max-steps}"
      shift 2
      ;;
    --max-epochs)
      max_epochs="${2:?missing value for --max-epochs}"
      shift 2
      ;;
    --rollout-steps)
      rollout_steps="${2:?missing value for --rollout-steps}"
      shift 2
      ;;
    --micro-batch)
      micro_batch="${2:?missing value for --micro-batch}"
      shift 2
      ;;
    --global-batch)
      global_batch="${2:?missing value for --global-batch}"
      shift 2
      ;;
    --replay-capacity)
      replay_capacity="${2:?missing value for --replay-capacity}"
      shift 2
      ;;
    --min-buffer-size)
      min_buffer_size="${2:?missing value for --min-buffer-size}"
      shift 2
      ;;
    --train-actor-steps)
      train_actor_steps="${2:?missing value for --train-actor-steps}"
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

require_command sbatch
require_command python3

model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${model_path}")"
demo_manifest="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${demo_manifest}")"
if [ ! -f "${demo_manifest}" ]; then
  echo "error: demo manifest not found: ${demo_manifest}" >&2
  exit 1
fi

stamp="$(timestamp_utc)"
task_token="$(printf "%s" "${task_suite}-task-${task_ids}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
run_id="$(make_run_id "stageD" "seed-sac-smoke" "${task_token}" "7" "0" "${stamp}")"
job_name="caver-stageD-sac-smoke"
run_dir="${run_root}/${run_id}"
results_dir="${run_dir}/results"
job_script="${run_dir}/job.sbatch"
manifest_path="${run_dir}/manifest.json"
slurm_stdout="${log_root}/${run_id}-%j.out"
slurm_stderr="${log_root}/${run_id}-%j.err"
rlinf_log_dir="${results_dir}/rlinf_logs"

ensure_directory "${run_dir}"
ensure_directory "${results_dir}"
ensure_directory "${log_root}"

python3 "${CAVER_REPO_ROOT}/scripts/manifest/create_manifest.py" \
  --output "${manifest_path}" \
  --run-id "${run_id}" \
  --stage stageD \
  --method seed-sac-smoke \
  --task "${task_token}" \
  --seed 7 \
  --budget 0 \
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
export CAVER_MANIFEST_PATH=${manifest_path}

${CAVER_REPO_ROOT}/scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh \
  --config-name ${config_name} \
  --model-path ${model_path} \
  --demo-manifest ${demo_manifest} \
  --log-dir ${rlinf_log_dir} \
  --task-suite ${task_suite} \
  --task-ids ${task_ids} \
  --train-envs ${train_envs} \
  --eval-envs ${eval_envs} \
  --max-steps ${max_steps} \
  --max-epochs ${max_epochs} \
  --rollout-steps ${rollout_steps} \
  --micro-batch ${micro_batch} \
  --global-batch ${global_batch} \
  --replay-capacity ${replay_capacity} \
  --min-buffer-size ${min_buffer_size} \
  --train-actor-steps ${train_actor_steps}
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
  if [ -n "${dependency}" ]; then
    echo "  dependency: ${dependency}"
  fi
  exit 0
fi

submission_output="$(sbatch "${job_script}")"
echo "${submission_output}"
