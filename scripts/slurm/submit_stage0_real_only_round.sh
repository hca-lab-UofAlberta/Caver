#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_real_only_round.sh [options]

Slurm options:
  --dependency SPEC          Slurm dependency spec, for example afterok:5395
  --partition NAME           Slurm partition (default: gpu-l40s)
  --qos NAME                 Slurm QoS (default: normal)
  --gpu-type TYPE            GPU type specifier (default: l40s)
  --gpus COUNT               GPU count (default: 1)
  --time LIMIT               Slurm time limit (default: 02:00:00)
  --cpus COUNT               CPU request (default: 8)
  --mem SIZE                 Memory request (default: 128G)
  --run-root PATH            Run directory root (default: runs/)
  --log-root PATH            Slurm log root (default: logs/slurm/)

Online execution options:
  --task-suite NAME
  --task-ids IDS
  --num-trials-per-task COUNT
  --count-legacy-contexts-as-online-budget
  --no-count-legacy-contexts-as-online-budget
  --manifest-path PATH
  --partition-name NAME
  --family-ids IDS
  --context-offset COUNT
  --max-contexts COUNT
  --candidate-count COUNT
  --selection-policy NAME
  --selector-seed COUNT
  --round-size COUNT
  --num-steps-wait COUNT
  --replan-steps COUNT
  --resize-size COUNT
  --resolution COUNT
  --max-env-steps COUNT
  --libero-gl-backend NAME
  --seed COUNT
  --server-mode NAME
  --policy-config-name NAME
  --policy-pretrained-path PATH
  --exact-rollout-payload
  --exact-rlinf-config-name NAME
  --exact-action-chunk COUNT
  --exact-no-nft-loss
  --exact-add-value-head
  --exact-value-after-vlm
  --exact-solver-type NAME
  --exact-infer-mode MODE

Backend update options:
  --config-name NAME
  --model-path PATH
  --backend-task-suite NAME
  --backend-task-ids IDS
  --experiment-name NAME
  --train-envs COUNT
  --eval-envs COUNT
  --runner-max-steps COUNT
  --runner-max-epochs COUNT
  --rollout-steps COUNT
  --micro-batch COUNT
  --global-batch COUNT
  --replay-capacity COUNT
  --min-buffer-size COUNT
  --train-actor-steps COUNT

Artifact options:
  --demo-output-mode NAME
  --max-items-per-shard COUNT
  --demo-format NAME
  --dry-run                  Generate run scaffolding but do not submit
  -h, --help                 Show this message
EOF
}

dependency=""
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="02:00:00"
cpus="8"
mem="128G"
run_root="${CAVER_DEFAULT_RUN_ROOT}"
log_root="${CAVER_DEFAULT_SLURM_LOG_ROOT}"

task_suite="libero_goal"
task_ids="0"
num_trials_per_task="1"
count_legacy_contexts_as_online_budget=1
manifest_path=""
partition_name=""
family_ids=""
context_offset="0"
max_contexts=""
candidate_count="4"
selection_policy="uniform"
selector_seed=""
round_size="25"
num_steps_wait="10"
replan_steps="4"
resize_size="224"
resolution="256"
max_env_steps=""
libero_gl_backend="osmesa"
seed="7"
server_mode="openpi-native"
policy_config_name=""
policy_pretrained_path=""
exact_rollout_payload=0
exact_rlinf_config_name=""
exact_action_chunk=""
exact_no_nft_loss=0
exact_add_value_head=0
exact_value_after_vlm=0
exact_solver_type="flow_sde"
exact_infer_mode="train"

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
backend_task_suite=""
backend_task_ids=""
experiment_name="stage0_real_only_round"
train_envs="1"
eval_envs="1"
runner_max_steps="1"
runner_max_epochs="1"
rollout_steps="4"
micro_batch="1"
global_batch="2"
replay_capacity="512"
min_buffer_size="1"
train_actor_steps="1"

demo_output_mode="sharded_manifest"
max_items_per_shard="128"
demo_format="chunk_step"
dry_run=0

while (($# > 0)); do
  case "${1}" in
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
    --task-suite)
      task_suite="${2:?missing value for --task-suite}"
      shift 2
      ;;
    --task-ids)
      task_ids="${2:?missing value for --task-ids}"
      shift 2
      ;;
    --num-trials-per-task)
      num_trials_per_task="${2:?missing value for --num-trials-per-task}"
      shift 2
      ;;
    --count-legacy-contexts-as-online-budget)
      count_legacy_contexts_as_online_budget=1
      shift
      ;;
    --no-count-legacy-contexts-as-online-budget)
      count_legacy_contexts_as_online_budget=0
      shift
      ;;
    --manifest-path)
      manifest_path="${2:?missing value for --manifest-path}"
      shift 2
      ;;
    --partition-name)
      partition_name="${2:?missing value for --partition-name}"
      shift 2
      ;;
    --family-ids)
      family_ids="${2:?missing value for --family-ids}"
      shift 2
      ;;
    --context-offset)
      context_offset="${2:?missing value for --context-offset}"
      shift 2
      ;;
    --max-contexts)
      max_contexts="${2:?missing value for --max-contexts}"
      shift 2
      ;;
    --candidate-count)
      candidate_count="${2:?missing value for --candidate-count}"
      shift 2
      ;;
    --selection-policy)
      selection_policy="${2:?missing value for --selection-policy}"
      shift 2
      ;;
    --selector-seed)
      selector_seed="${2:?missing value for --selector-seed}"
      shift 2
      ;;
    --round-size)
      round_size="${2:?missing value for --round-size}"
      shift 2
      ;;
    --num-steps-wait)
      num_steps_wait="${2:?missing value for --num-steps-wait}"
      shift 2
      ;;
    --replan-steps)
      replan_steps="${2:?missing value for --replan-steps}"
      shift 2
      ;;
    --resize-size)
      resize_size="${2:?missing value for --resize-size}"
      shift 2
      ;;
    --resolution)
      resolution="${2:?missing value for --resolution}"
      shift 2
      ;;
    --max-env-steps)
      max_env_steps="${2:?missing value for --max-env-steps}"
      shift 2
      ;;
    --libero-gl-backend)
      libero_gl_backend="${2:?missing value for --libero-gl-backend}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing value for --seed}"
      shift 2
      ;;
    --server-mode)
      server_mode="${2:?missing value for --server-mode}"
      shift 2
      ;;
    --policy-config-name)
      policy_config_name="${2:?missing value for --policy-config-name}"
      shift 2
      ;;
    --policy-pretrained-path)
      policy_pretrained_path="${2:?missing value for --policy-pretrained-path}"
      shift 2
      ;;
    --exact-rollout-payload)
      exact_rollout_payload=1
      shift
      ;;
    --exact-rlinf-config-name)
      exact_rlinf_config_name="${2:?missing value for --exact-rlinf-config-name}"
      shift 2
      ;;
    --exact-action-chunk)
      exact_action_chunk="${2:?missing value for --exact-action-chunk}"
      shift 2
      ;;
    --exact-no-nft-loss)
      exact_no_nft_loss=1
      shift
      ;;
    --exact-add-value-head)
      exact_add_value_head=1
      shift
      ;;
    --exact-value-after-vlm)
      exact_value_after_vlm=1
      shift
      ;;
    --exact-solver-type)
      exact_solver_type="${2:?missing value for --exact-solver-type}"
      shift 2
      ;;
    --exact-infer-mode)
      exact_infer_mode="${2:?missing value for --exact-infer-mode}"
      shift 2
      ;;
    --config-name)
      config_name="${2:?missing value for --config-name}"
      shift 2
      ;;
    --model-path)
      model_path="${2:?missing value for --model-path}"
      shift 2
      ;;
    --backend-task-suite)
      backend_task_suite="${2:?missing value for --backend-task-suite}"
      shift 2
      ;;
    --backend-task-ids)
      backend_task_ids="${2:?missing value for --backend-task-ids}"
      shift 2
      ;;
    --experiment-name)
      experiment_name="${2:?missing value for --experiment-name}"
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
    --runner-max-steps)
      runner_max_steps="${2:?missing value for --runner-max-steps}"
      shift 2
      ;;
    --runner-max-epochs)
      runner_max_epochs="${2:?missing value for --runner-max-epochs}"
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
    --demo-output-mode)
      demo_output_mode="${2:?missing value for --demo-output-mode}"
      shift 2
      ;;
    --max-items-per-shard)
      max_items_per_shard="${2:?missing value for --max-items-per-shard}"
      shift 2
      ;;
    --demo-format)
      demo_format="${2:?missing value for --demo-format}"
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

if [ -n "${manifest_path}" ]; then
  task_token="$(printf "%s" "manifest-${partition_name:-unset}-${family_ids:-all}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
  budget_token="${max_contexts:-${round_size}}"
else
  task_token="$(printf "%s" "${task_suite}-task-${task_ids}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
  budget_token="${num_trials_per_task}"
fi

stamp="$(timestamp_utc)"
run_id="$(make_run_id "stageE" "real-only-round" "${task_token}" "${seed}" "${budget_token}" "${stamp}")"
job_name="caver-stageE-real-only"
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
  --method real-only-round \
  --task "${task_token}" \
  --seed "${seed}" \
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

job_cmd=(
  "${CAVER_REPO_ROOT}/scripts/stagee/run_stage0_real_only_round.sh"
  --results-dir "${results_dir}"
  --task-suite "${task_suite}"
  --task-ids "${task_ids}"
  --num-trials-per-task "${num_trials_per_task}"
  --candidate-count "${candidate_count}"
  --selection-policy "${selection_policy}"
  --round-size "${round_size}"
  --num-steps-wait "${num_steps_wait}"
  --replan-steps "${replan_steps}"
  --resize-size "${resize_size}"
  --resolution "${resolution}"
  --libero-gl-backend "${libero_gl_backend}"
  --seed "${seed}"
  --server-mode "${server_mode}"
  --config-name "${config_name}"
  --model-path "${model_path}"
  --experiment-name "${experiment_name}"
  --train-envs "${train_envs}"
  --eval-envs "${eval_envs}"
  --runner-max-steps "${runner_max_steps}"
  --runner-max-epochs "${runner_max_epochs}"
  --rollout-steps "${rollout_steps}"
  --micro-batch "${micro_batch}"
  --global-batch "${global_batch}"
  --replay-capacity "${replay_capacity}"
  --min-buffer-size "${min_buffer_size}"
  --train-actor-steps "${train_actor_steps}"
  --demo-output-mode "${demo_output_mode}"
  --max-items-per-shard "${max_items_per_shard}"
  --demo-format "${demo_format}"
)

if [ -n "${manifest_path}" ]; then
  job_cmd+=(--manifest-path "${manifest_path}" --partition-name "${partition_name}" --context-offset "${context_offset}")
else
  if ((count_legacy_contexts_as_online_budget)); then
    job_cmd+=(--count-legacy-contexts-as-online-budget)
  fi
fi
if [ -n "${family_ids}" ]; then
  job_cmd+=(--family-ids "${family_ids}")
fi
if [ -n "${max_contexts}" ]; then
  job_cmd+=(--max-contexts "${max_contexts}")
fi
if [ -n "${selector_seed}" ]; then
  job_cmd+=(--selector-seed "${selector_seed}")
fi
if [ -n "${max_env_steps}" ]; then
  job_cmd+=(--max-env-steps "${max_env_steps}")
fi
if [ -n "${policy_config_name}" ]; then
  job_cmd+=(--policy-config-name "${policy_config_name}" --policy-pretrained-path "${policy_pretrained_path}")
fi
if ((exact_rollout_payload)); then
  job_cmd+=(--exact-rollout-payload)
fi
if [ -n "${exact_rlinf_config_name}" ]; then
  job_cmd+=(--exact-rlinf-config-name "${exact_rlinf_config_name}")
fi
if [ -n "${exact_action_chunk}" ]; then
  job_cmd+=(--exact-action-chunk "${exact_action_chunk}")
fi
if ((exact_no_nft_loss)); then
  job_cmd+=(--exact-no-nft-loss)
fi
if ((exact_add_value_head)); then
  job_cmd+=(--exact-add-value-head)
fi
if ((exact_value_after_vlm)); then
  job_cmd+=(--exact-value-after-vlm)
fi
if [ -n "${exact_solver_type}" ]; then
  job_cmd+=(--exact-solver-type "${exact_solver_type}")
fi
if [ -n "${exact_infer_mode}" ]; then
  job_cmd+=(--exact-infer-mode "${exact_infer_mode}")
fi
if [ -n "${backend_task_suite}" ]; then
  job_cmd+=(--backend-task-suite "${backend_task_suite}")
fi
if [ -n "${backend_task_ids}" ]; then
  job_cmd+=(--backend-task-ids "${backend_task_ids}")
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
export CAVER_MANIFEST_PATH=${manifest_out}

$(printf '%q ' "${job_cmd[@]}")
EOF
if ! chmod +x "${job_script}" 2>/dev/null; then
  printf 'warning: could not chmod +x %s; continuing because sbatch only needs a readable script\n' "${job_script}" >&2
fi

if ((dry_run)); then
  echo "created run scaffolding:"
  echo "  run_id: ${run_id}"
  echo "  run_dir: ${run_dir}"
  echo "  manifest: ${manifest_out}"
  echo "  job_script: ${job_script}"
  echo "  stdout: ${slurm_stdout}"
  echo "  stderr: ${slurm_stderr}"
  exit 0
fi

submission_output="$(sbatch "${job_script}")"
echo "${submission_output}"
