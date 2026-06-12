#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_caver_lagged_budget.sh [options]

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

Lagged driver options:
  --trace-reference-mode NAME

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

CAVER scaffold options:
  --selector-mode NAME
  --admission-policy NAME
  --admission-kappa VALUE
  --admission-threshold VALUE
  --top-m-success-count COUNT
  --family-min-success-count COUNT
  --value-proxy-model-path PATH
  --dr-calibrator-model-path PATH
  --rescue-family-ids IDS
  --rescue-per-family-count COUNT
  --repair-min-trace-records COUNT
  --repair-max-trace-records COUNT
  --repair-min-progress VALUE
  --repair-min-primitive-steps COUNT
  --repair-max-regression VALUE
  --provider-mode NAME
  --provider-bundle-root PATH
  --provider-gesim-timeout-sec COUNT
  --provider-gesim-prompt TEXT
  --demo-trace-write-policy NAME
  --trace-stage0-progress
  --no-require-candidate-bank

Backend update options:
  --config-name NAME
  --model-path PATH
  --backend-task-suite NAME
  --backend-task-ids IDS
  --experiment-name NAME
  --run-label-suffix TOKEN
  --finalizer-skip-backend-update
  --disable-lagged-dr
  --skip-dr-calibrator-fit
  --disable-lagged-lvd
  --skip-lvd-selector-fit
  --lvd-target-source NAME
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
  --dry-run
  -h, --help
EOF
}

dependency=""
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="08:30:00"
cpus="8"
mem="128G"
run_root="${CAVER_DEFAULT_RUN_ROOT}"
log_root="${CAVER_DEFAULT_SLURM_LOG_ROOT}"

trace_reference_mode="manifest"

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
selection_policy="caver_heuristic"
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

selector_mode="frozen_actionspace_softmax_v1"
admission_policy="success_lcb_v1"
admission_kappa=""
admission_threshold=""
top_m_success_count=""
family_min_success_count=""
value_proxy_model_path=""
dr_calibrator_model_path=""
lvd_selector_model_path=""
rescue_family_ids=""
rescue_per_family_count=""
repair_min_trace_records=""
repair_max_trace_records=""
repair_min_progress=""
repair_min_primitive_steps=""
repair_max_regression=""
provider_mode="none"
provider_bundle_root=""
provider_gesim_timeout_sec="900"
provider_gesim_prompt="best quality, consistent and smooth motion, realistic, clear and distinct."
demo_trace_write_policy="success_only"
trace_stage0_progress=0
require_candidate_bank=1

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
backend_task_suite=""
backend_task_ids=""
experiment_name="stage0_caver_lagged_budget"
run_label_suffix=""
finalizer_skip_backend_update=0
disable_lagged_dr=0
skip_dr_calibrator_fit=0
disable_lagged_lvd=0
skip_lvd_selector_fit=0
lvd_target_source="dr_clipped"
train_envs="1"
eval_envs="1"
train_envs_explicit=0
eval_envs_explicit=0
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
    --trace-reference-mode)
      trace_reference_mode="${2:?missing value for --trace-reference-mode}"
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
    --selector-mode)
      selector_mode="${2:?missing value for --selector-mode}"
      shift 2
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
    --value-proxy-model-path)
      value_proxy_model_path="${2:?missing value for --value-proxy-model-path}"
      shift 2
      ;;
    --dr-calibrator-model-path)
      dr_calibrator_model_path="${2:?missing value for --dr-calibrator-model-path}"
      shift 2
      ;;
    --lvd-selector-model-path)
      lvd_selector_model_path="${2:?missing value for --lvd-selector-model-path}"
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
    --repair-min-progress)
      repair_min_progress="${2:?missing value for --repair-min-progress}"
      shift 2
      ;;
    --repair-min-primitive-steps)
      repair_min_primitive_steps="${2:?missing value for --repair-min-primitive-steps}"
      shift 2
      ;;
    --repair-max-regression)
      repair_max_regression="${2:?missing value for --repair-max-regression}"
      shift 2
      ;;
    --provider-mode)
      provider_mode="${2:?missing value for --provider-mode}"
      shift 2
      ;;
    --provider-bundle-root)
      provider_bundle_root="${2:?missing value for --provider-bundle-root}"
      shift 2
      ;;
    --provider-gesim-timeout-sec)
      provider_gesim_timeout_sec="${2:?missing value for --provider-gesim-timeout-sec}"
      shift 2
      ;;
    --provider-gesim-prompt)
      provider_gesim_prompt="${2:?missing value for --provider-gesim-prompt}"
      shift 2
      ;;
    --demo-trace-write-policy)
      demo_trace_write_policy="${2:?missing value for --demo-trace-write-policy}"
      shift 2
      ;;
    --trace-stage0-progress)
      trace_stage0_progress=1
      shift
      ;;
    --no-require-candidate-bank)
      require_candidate_bank=0
      shift
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
    --run-label-suffix)
      run_label_suffix="${2:?missing value for --run-label-suffix}"
      shift 2
      ;;
    --finalizer-skip-backend-update)
      finalizer_skip_backend_update=1
      shift
      ;;
    --disable-lagged-dr)
      disable_lagged_dr=1
      shift
      ;;
    --skip-dr-calibrator-fit)
      skip_dr_calibrator_fit=1
      shift
      ;;
    --disable-lagged-lvd)
      disable_lagged_lvd=1
      shift
      ;;
    --skip-lvd-selector-fit)
      skip_lvd_selector_fit=1
      shift
      ;;
    --lvd-target-source)
      lvd_target_source="${2:?missing value for --lvd-target-source}"
      shift 2
      ;;
    --train-envs)
      train_envs="${2:?missing value for --train-envs}"
      train_envs_explicit=1
      shift 2
      ;;
    --eval-envs)
      eval_envs="${2:?missing value for --eval-envs}"
      eval_envs_explicit=1
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

if (( ! train_envs_explicit )) && [ "${gpus}" -gt 1 ]; then
  train_envs="${gpus}"
fi
if (( ! eval_envs_explicit )) && [ "${gpus}" -gt 1 ]; then
  eval_envs="${gpus}"
fi

require_command sbatch
require_command python3

case "${trace_reference_mode}" in
  manifest|materialize)
    ;;
  *)
    echo "error: unsupported --trace-reference-mode ${trace_reference_mode}" >&2
    exit 1
    ;;
esac

case "${demo_trace_write_policy}" in
  all|success_only)
    ;;
  *)
    echo "error: unsupported --demo-trace-write-policy ${demo_trace_write_policy}" >&2
    exit 1
    ;;
esac

if [ -n "${manifest_path}" ]; then
  task_token="$(printf "%s" "manifest-${partition_name:-unset}-${family_ids:-all}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
  budget_token="${max_contexts:-${round_size}}"
else
  task_token="$(printf "%s" "${task_suite}-task-${task_ids}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
  budget_token="${num_trials_per_task}"
fi
if [ -n "${run_label_suffix}" ]; then
  run_label_suffix="$(printf "%s" "${run_label_suffix}" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+//; s/-+$//')"
  task_token="${task_token}-${run_label_suffix}"
fi

stamp="$(timestamp_utc)"
run_id="$(make_run_id "stageE" "caver-lagged" "${task_token}" "${seed}" "${budget_token}" "${stamp}")"
job_name="caver-stageE-lagged"
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
  --method caver-lagged \
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
  python3
  "${CAVER_REPO_ROOT}/scripts/stagee/run_stage0_caver_lagged_budget.py"
  --results-dir "${results_dir}"
  --trace-reference-mode "${trace_reference_mode}"
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
  --selector-mode "${selector_mode}"
  --admission-policy "${admission_policy}"
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
if [ -n "${value_proxy_model_path}" ]; then
  job_cmd+=(--value-proxy-model-path "${value_proxy_model_path}")
fi
if [ -n "${dr_calibrator_model_path}" ]; then
  job_cmd+=(--dr-calibrator-model-path "${dr_calibrator_model_path}")
fi
if [ -n "${lvd_selector_model_path}" ]; then
  job_cmd+=(--lvd-selector-model-path "${lvd_selector_model_path}")
fi
if [ -n "${admission_kappa}" ]; then
  job_cmd+=(--admission-kappa "${admission_kappa}")
fi
if [ -n "${admission_threshold}" ]; then
  job_cmd+=(--admission-threshold "${admission_threshold}")
fi
if [ -n "${top_m_success_count}" ]; then
  job_cmd+=(--top-m-success-count "${top_m_success_count}")
fi
if [ -n "${family_min_success_count}" ]; then
  job_cmd+=(--family-min-success-count "${family_min_success_count}")
fi
if [ -n "${rescue_family_ids}" ]; then
  job_cmd+=(--rescue-family-ids "${rescue_family_ids}")
fi
if [ -n "${rescue_per_family_count}" ]; then
  job_cmd+=(--rescue-per-family-count "${rescue_per_family_count}")
fi
if [ -n "${repair_min_trace_records}" ]; then
  job_cmd+=(--repair-min-trace-records "${repair_min_trace_records}")
fi
if [ -n "${repair_max_trace_records}" ]; then
  job_cmd+=(--repair-max-trace-records "${repair_max_trace_records}")
fi
if [ -n "${repair_min_progress}" ]; then
  job_cmd+=(--repair-min-progress "${repair_min_progress}")
fi
if [ -n "${repair_min_primitive_steps}" ]; then
  job_cmd+=(--repair-min-primitive-steps "${repair_min_primitive_steps}")
fi
if [ -n "${repair_max_regression}" ]; then
  job_cmd+=(--repair-max-regression "${repair_max_regression}")
fi
if [ -n "${provider_mode}" ]; then
  job_cmd+=(--provider-mode "${provider_mode}")
fi
if [ -n "${provider_bundle_root}" ]; then
  job_cmd+=(--provider-bundle-root "${provider_bundle_root}")
fi
if [ -n "${provider_gesim_timeout_sec}" ]; then
  job_cmd+=(--provider-gesim-timeout-sec "${provider_gesim_timeout_sec}")
fi
if [ -n "${provider_gesim_prompt}" ]; then
  job_cmd+=(--provider-gesim-prompt "${provider_gesim_prompt}")
fi
if [ -n "${demo_trace_write_policy}" ]; then
  job_cmd+=(--demo-trace-write-policy "${demo_trace_write_policy}")
fi
if ((trace_stage0_progress)); then
  job_cmd+=(--trace-stage0-progress)
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
if (( ! require_candidate_bank )); then
  job_cmd+=(--no-require-candidate-bank)
fi
if (( finalizer_skip_backend_update )); then
  job_cmd+=(--finalizer-skip-backend-update)
fi
if (( disable_lagged_dr )); then
  job_cmd+=(--disable-lagged-dr)
fi
if (( skip_dr_calibrator_fit )); then
  job_cmd+=(--skip-dr-calibrator-fit)
fi
if (( disable_lagged_lvd )); then
  job_cmd+=(--disable-lagged-lvd)
fi
if (( skip_lvd_selector_fit )); then
  job_cmd+=(--skip-lvd-selector-fit)
fi
if [ -n "${lvd_target_source}" ]; then
  job_cmd+=(--lvd-target-source "${lvd_target_source}")
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
export TMPDIR="\${CAVER_NODE_LOCAL_TMP_ROOT:-/tmp/\${USER}/caver_stagee_lagged}/\${SLURM_JOB_ID:-manual}"
export CAVER_STAGEE_HEAVY_TRACE_ROOT="\${TMPDIR}/heavy_traces"
mkdir -p "\${CAVER_STAGEE_HEAVY_TRACE_ROOT}" "\${TMPDIR}"

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
