#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_caver_round.sh [options]

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
  --exact-rollout-payload      Use RLinf-backed exact OpenPI serving and log exact rollout payloads
  --exact-rlinf-config-name NAME
  --exact-action-chunk COUNT
  --exact-no-nft-loss
  --exact-add-value-head
  --exact-value-after-vlm
  --exact-solver-type NAME
  --exact-infer-mode MODE

CAVER scaffold options:
  --selector-mode NAME       Selector implementation label (default: frozen_actionspace_softmax_v1)
  --admission-policy NAME    Admission policy label (default: success_lcb_v1)
  --value-proxy-model-path PATH
                             Optional fitted Stage-0 value-proxy JSON for the selector
  --dr-calibrator-model-path PATH
                             Optional lagged Stage-E DR calibrator JSON for the selector
  --provider-mode NAME
                             Optional provider mode: none, gesim_bundle, gesim_live_summary
  --provider-bundle-root PATH
                             Optional provider bundle / inference root
  --provider-gesim-timeout-sec COUNT
  --provider-gesim-prompt TEXT
  --no-require-candidate-bank
                             Allow pre-v2 traces that omit the full candidate bank

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
  --results-dir PATH
  --demo-output-mode NAME
  --max-items-per-shard COUNT
  --demo-format NAME
  --skip-backend-train
                             Build online/selector/DR artifacts only; skip demo conversion and backend training
  --skip-online
  --dry-run
  -h, --help
EOF
}

task_suite="libero_goal"
task_ids="0"
num_trials_per_task="25"
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
replan_steps="5"
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
value_proxy_model_path=""
dr_calibrator_model_path=""
provider_mode="none"
provider_bundle_root=""
provider_gesim_timeout_sec="900"
provider_gesim_prompt="best quality, consistent and smooth motion, realistic, clear and distinct."
require_candidate_bank=1

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
backend_task_suite=""
backend_task_ids=""
experiment_name="stage0_caver_round"
train_envs="1"
eval_envs="1"
runner_max_steps="1"
runner_max_epochs="1"
rollout_steps="5"
micro_batch="1"
global_batch="2"
replay_capacity="512"
min_buffer_size="1"
train_actor_steps="1"

results_dir=""
demo_output_mode="sharded_manifest"
max_items_per_shard="128"
demo_format="chunk_step"
skip_backend_train=0
skip_online=0
dry_run=0

while (($# > 0)); do
  case "${1}" in
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
    --value-proxy-model-path)
      value_proxy_model_path="${2:?missing value for --value-proxy-model-path}"
      shift 2
      ;;
    --dr-calibrator-model-path)
      dr_calibrator_model_path="${2:?missing value for --dr-calibrator-model-path}"
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
    --results-dir)
      results_dir="${2:?missing value for --results-dir}"
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
    --skip-backend-train)
      skip_backend_train=1
      shift
      ;;
    --skip-online)
      skip_online=1
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

require_command python3

case "${server_mode}" in
  openpi-native|openpi-exact|dummy)
    ;;
  *)
    echo "error: unsupported --server-mode ${server_mode}" >&2
    exit 1
    ;;
esac

case "${selection_policy}" in
  first|uniform|caver_heuristic)
    ;;
  *)
    echo "error: unsupported --selection-policy ${selection_policy}" >&2
    exit 1
    ;;
esac

case "${provider_mode}" in
  none|gesim_bundle|gesim_live_summary)
    ;;
  *)
    echo "error: unsupported --provider-mode ${provider_mode}" >&2
    exit 1
    ;;
esac

case "${demo_output_mode}" in
  single_pt|sharded_manifest)
    ;;
  *)
    echo "error: unsupported --demo-output-mode ${demo_output_mode}" >&2
    exit 1
    ;;
esac

case "${libero_gl_backend}" in
  egl|osmesa)
    ;;
  *)
    echo "error: unsupported --libero-gl-backend ${libero_gl_backend}" >&2
    exit 1
    ;;
esac

if [ -n "${manifest_path}" ]; then
  if [ -z "${partition_name}" ]; then
    echo "error: --partition-name is required with --manifest-path" >&2
    exit 1
  fi
  manifest_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${manifest_path}")"
  if [ ! -f "${manifest_path}" ]; then
    echo "error: manifest not found: ${manifest_path}" >&2
    exit 1
  fi
fi

if [ -n "${policy_config_name}" ] || [ -n "${policy_pretrained_path}" ]; then
  if [ -z "${policy_config_name}" ] || [ -z "${policy_pretrained_path}" ]; then
    echo "error: --policy-config-name and --policy-pretrained-path must be provided together" >&2
    exit 1
  fi
  policy_pretrained_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${policy_pretrained_path}")"
fi

if [ -n "${value_proxy_model_path}" ]; then
  value_proxy_model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${value_proxy_model_path}")"
  if [ ! -f "${value_proxy_model_path}" ]; then
    echo "error: value proxy model not found: ${value_proxy_model_path}" >&2
    exit 1
  fi
fi

if [ -n "${dr_calibrator_model_path}" ]; then
  dr_calibrator_model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${dr_calibrator_model_path}")"
  if [ ! -f "${dr_calibrator_model_path}" ]; then
    echo "error: DR calibrator model not found: ${dr_calibrator_model_path}" >&2
    exit 1
  fi
fi

model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${model_path}")"
selector_seed_effective="${selector_seed:-${seed}}"
backend_task_suite="${backend_task_suite:-${task_suite}}"
backend_task_ids="${backend_task_ids:-${task_ids}}"
policy_config_name_effective="${policy_config_name}"
policy_pretrained_path_effective="${policy_pretrained_path}"
if [ "${server_mode}" = "openpi-exact" ]; then
  if [ -z "${policy_config_name_effective}" ]; then
    policy_config_name_effective="pi05_libero"
  fi
  if [ -z "${policy_pretrained_path_effective}" ]; then
    policy_pretrained_path_effective="${model_path}"
  fi
fi
exact_rlinf_config_name_effective="${exact_rlinf_config_name:-${config_name}}"

if [ -z "${results_dir}" ]; then
  if [ -n "${CAVER_RUN_DIR:-}" ]; then
    results_dir="${CAVER_RUN_DIR}/results"
  else
    results_dir="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stagee_caver_round__$(timestamp_utc)"
  fi
fi
results_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${results_dir}")"
ensure_directory "${results_dir}"
if [ -z "${provider_bundle_root}" ] && [ "${provider_mode}" != "none" ]; then
  provider_bundle_root="${results_dir}/provider_bundles"
fi
if [ -n "${provider_bundle_root}" ]; then
  provider_bundle_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${provider_bundle_root}")"
  ensure_directory "${provider_bundle_root}"
fi

visible_gpu_count=0
visible_gpu_ids=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  for visible_gpu_id in "${visible_gpu_ids[@]}"; do
    trimmed_visible_gpu_id="${visible_gpu_id//[[:space:]]/}"
    if [ -n "${trimmed_visible_gpu_id}" ]; then
      visible_gpu_count=$((visible_gpu_count + 1))
    fi
  done
fi
if [ "${provider_mode}" != "none" ] && [ "${visible_gpu_count}" -ge 2 ]; then
  export CAVER_OPENPI_CUDA_VISIBLE_DEVICES="${CAVER_OPENPI_CUDA_VISIBLE_DEVICES:-${visible_gpu_ids[0]}}"
  export CAVER_GESIM_CUDA_VISIBLE_DEVICES="${CAVER_GESIM_CUDA_VISIBLE_DEVICES:-${visible_gpu_ids[1]}}"
fi

online_results_path="${results_dir}/caver_online_eval.json"
online_context_log_path="${results_dir}/caver_online_contexts.jsonl"
trace_path="${results_dir}/caver_online_chunks.jsonl"
selector_context_path="${results_dir}/caver_selector_contexts.jsonl"
selector_summary_path="${results_dir}/caver_selector_summary.json"
admission_context_path="${results_dir}/caver_admission_contexts.jsonl"
admission_summary_path="${results_dir}/caver_admission_summary.json"
admitted_trace_path="${results_dir}/caver_admitted_chunks.jsonl"
dr_dataset_path="${results_dir}/caver_dr_candidate_dataset.jsonl"
dr_summary_path="${results_dir}/caver_dr_candidate_dataset.summary.json"
next_dr_calibrator_path="${results_dir}/caver_lagged_dr_calibrator.json"
next_dr_calibrator_summary_path="${results_dir}/caver_lagged_dr_calibrator.summary.json"
demo_summary_path="${results_dir}/caver_round_demo.summary.json"
training_log_dir="${results_dir}/rlinf_logs"
round_summary_path="${results_dir}/caver_round_summary.json"
if [ "${demo_output_mode}" = "sharded_manifest" ]; then
  demo_output_path="${results_dir}/caver_round_demo.manifest.json"
else
  demo_output_path="${results_dir}/caver_round_demo.pt"
fi

online_cmd=(
  "${CAVER_REPO_ROOT}/scripts/bridge/run_libero_remote_eval.sh"
)
if [ "${server_mode}" = "dummy" ]; then
  online_cmd+=(--dummy-server)
elif [ "${server_mode}" = "openpi-native" ]; then
  online_cmd+=(--openpi-native)
fi
if [ -n "${policy_config_name_effective}" ]; then
  online_cmd+=(--config-name "${policy_config_name_effective}" --pretrained-path "${policy_pretrained_path_effective}")
fi
if ((exact_rollout_payload)) || [ "${server_mode}" = "openpi-exact" ]; then
  online_cmd+=(--exact-rollout-payload)
  online_cmd+=(--rlinf-config-name "${exact_rlinf_config_name_effective}")
  if [ -n "${exact_action_chunk}" ]; then
    online_cmd+=(--exact-action-chunk "${exact_action_chunk}")
  fi
  if ((exact_no_nft_loss)); then
    online_cmd+=(--exact-no-nft-loss)
  fi
  if ((exact_add_value_head)); then
    online_cmd+=(--exact-add-value-head)
  fi
  if ((exact_value_after_vlm)); then
    online_cmd+=(--exact-value-after-vlm)
  fi
  if [ -n "${exact_solver_type}" ]; then
    online_cmd+=(--exact-solver-type "${exact_solver_type}")
  fi
  if [ -n "${exact_infer_mode}" ]; then
    online_cmd+=(--exact-infer-mode "${exact_infer_mode}")
  fi
fi
online_cmd+=(
  --
  --seed "${seed}"
  --candidate-count "${candidate_count}"
  --selection-policy "${selection_policy}"
  --selector-seed "${selector_seed_effective}"
  --round-size "${round_size}"
  --num-steps-wait "${num_steps_wait}"
  --replan-steps "${replan_steps}"
  --resize-size "${resize_size}"
  --resolution "${resolution}"
  --results-path "${online_results_path}"
  --context-log-path "${online_context_log_path}"
  --transition-trace-path "${trace_path}"
)

if [ -n "${value_proxy_model_path}" ]; then
  online_cmd+=(--value-proxy-model-path "${value_proxy_model_path}")
fi
if [ -n "${dr_calibrator_model_path}" ]; then
  online_cmd+=(--dr-calibrator-model-path "${dr_calibrator_model_path}")
fi
if [ "${provider_mode}" != "none" ]; then
  online_cmd+=(
    --provider-mode "${provider_mode}"
    --provider-bundle-root "${provider_bundle_root}"
    --provider-gesim-timeout-sec "${provider_gesim_timeout_sec}"
    --provider-gesim-prompt "${provider_gesim_prompt}"
  )
fi

if [ -n "${max_env_steps}" ]; then
  online_cmd+=(--max-steps "${max_env_steps}")
fi

if [ -n "${manifest_path}" ]; then
  online_cmd+=(
    --manifest-path "${manifest_path}"
    --partition-name "${partition_name}"
    --context-offset "${context_offset}"
  )
  if [ -n "${family_ids}" ]; then
    online_cmd+=(--family-ids "${family_ids}")
  fi
  if [ -n "${max_contexts}" ]; then
    online_cmd+=(--max-contexts "${max_contexts}")
  fi
else
  online_cmd+=(
    --task-suite-name "${task_suite}"
    --task-ids "${task_ids}"
    --num-trials-per-task "${num_trials_per_task}"
  )
  if ((count_legacy_contexts_as_online_budget)); then
    online_cmd+=(--count-legacy-contexts-as-online-budget)
  fi
fi

artifact_cmd=(
  python3
  "${CAVER_REPO_ROOT}/scripts/stagee/build_caver_round_artifacts.py"
  --online-results "${online_results_path}"
  --trace-path "${trace_path}"
  --selector-context-path "${selector_context_path}"
  --selector-summary-path "${selector_summary_path}"
  --admission-context-path "${admission_context_path}"
  --admission-summary-path "${admission_summary_path}"
  --admitted-trace-path "${admitted_trace_path}"
  --selector-mode "${selector_mode}"
  --admission-policy "${admission_policy}"
)
if ((require_candidate_bank)); then
  artifact_cmd+=(--require-candidate-bank)
fi

dr_dataset_cmd=(
  python3
  "${CAVER_REPO_ROOT}/scripts/stagee/build_stagee_dr_dataset.py"
  --online-results "${online_results_path}"
  --trace-path "${trace_path}"
  --output-path "${dr_dataset_path}"
  --summary-path "${dr_summary_path}"
)

dr_calibrator_cmd=(
  python3
  "${CAVER_REPO_ROOT}/scripts/stagee/fit_stagee_dr_calibrator.py"
  --dataset-path "${dr_dataset_path}"
  --output-path "${next_dr_calibrator_path}"
  --summary-path "${next_dr_calibrator_summary_path}"
)

convert_cmd=(
  "${CAVER_REPO_ROOT}/scripts/stage0/convert_stage0_trace_to_rlinf_demo.sh"
  --trace-path "${admitted_trace_path}"
  --output-path "${demo_output_path}"
  --summary-path "${demo_summary_path}"
  --output-mode "${demo_output_mode}"
  --demo-format "${demo_format}"
)
if [ "${demo_output_mode}" = "sharded_manifest" ]; then
  convert_cmd+=(--max-items-per-shard "${max_items_per_shard}")
fi

train_cmd=(
  "${CAVER_REPO_ROOT}/scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh"
  --config-name "${config_name}"
  --model-path "${model_path}"
  --demo-manifest "${demo_output_path}"
  --experiment-name "${experiment_name}"
  --log-dir "${training_log_dir}"
  --task-suite "${backend_task_suite}"
  --task-ids "${backend_task_ids}"
  --train-envs "${train_envs}"
  --eval-envs "${eval_envs}"
  --max-steps "${runner_max_steps}"
  --max-epochs "${runner_max_epochs}"
  --rollout-steps "${rollout_steps}"
  --micro-batch "${micro_batch}"
  --global-batch "${global_batch}"
  --replay-capacity "${replay_capacity}"
  --min-buffer-size "${min_buffer_size}"
  --train-actor-steps "${train_actor_steps}"
)

printf 'online command:'
printf ' %q' "${online_cmd[@]}"
printf '\n'
printf 'artifact command:'
printf ' %q' "${artifact_cmd[@]}"
printf '\n'
printf 'convert command:'
printf ' %q' "${convert_cmd[@]}"
printf '\n'
printf 'dr dataset command:'
printf ' %q' "${dr_dataset_cmd[@]}"
printf '\n'
printf 'dr calibrator command:'
printf ' %q' "${dr_calibrator_cmd[@]}"
printf '\n'
printf 'train command:'
printf ' %q' "${train_cmd[@]}"
printf '\n'
printf 'environment:\n'
printf '  MUJOCO_GL: %s\n' "${libero_gl_backend}"
printf '  server_mode: %s\n' "${server_mode}"
printf '  policy_config_name_effective: %s\n' "${policy_config_name_effective:-none}"
printf '  policy_pretrained_path_effective: %s\n' "${policy_pretrained_path_effective:-none}"
printf '  exact_rollout_payload: %s\n' "$([ "${server_mode}" = "openpi-exact" ] || ((exact_rollout_payload)) && printf true || printf false)"
if [ "${server_mode}" = "openpi-exact" ] || ((exact_rollout_payload)); then
  printf '  exact_rlinf_config_name_effective: %s\n' "${exact_rlinf_config_name_effective}"
  printf '  exact_action_chunk: %s\n' "${exact_action_chunk:-default}"
  printf '  exact_solver_type: %s\n' "${exact_solver_type}"
  printf '  exact_infer_mode: %s\n' "${exact_infer_mode}"
fi
printf 'artifacts:\n'
printf '  results_dir: %s\n' "${results_dir}"
printf '  online_results: %s\n' "${online_results_path}"
printf '  online_context_log: %s\n' "${online_context_log_path}"
printf '  trace_path: %s\n' "${trace_path}"
printf '  selector_summary: %s\n' "${selector_summary_path}"
printf '  admission_summary: %s\n' "${admission_summary_path}"
printf '  admitted_trace: %s\n' "${admitted_trace_path}"
printf '  demo_output: %s\n' "${demo_output_path}"
printf '  demo_summary: %s\n' "${demo_summary_path}"
printf '  training_log_dir: %s\n' "${training_log_dir}"
printf '  round_summary: %s\n' "${round_summary_path}"
printf '  value_proxy_model_path: %s\n' "${value_proxy_model_path:-none}"
printf '  dr_calibrator_model_path: %s\n' "${dr_calibrator_model_path:-none}"
printf '  provider_mode: %s\n' "${provider_mode}"
printf '  provider_bundle_root: %s\n' "${provider_bundle_root:-none}"
if [ "${provider_mode}" != "none" ]; then
  printf '  provider_gesim_timeout_sec: %s\n' "${provider_gesim_timeout_sec}"
  printf '  openpi_cuda_visible_devices: %s\n' "${CAVER_OPENPI_CUDA_VISIBLE_DEVICES:-inherit}"
  printf '  gesim_cuda_visible_devices: %s\n' "${CAVER_GESIM_CUDA_VISIBLE_DEVICES:-inherit}"
fi
printf '  dr_dataset_path: %s\n' "${dr_dataset_path}"
printf '  dr_summary_path: %s\n' "${dr_summary_path}"
printf '  next_dr_calibrator_path: %s\n' "${next_dr_calibrator_path}"
printf '  next_dr_calibrator_summary_path: %s\n' "${next_dr_calibrator_summary_path}"
skip_online_label="false"
if ((skip_online)); then
  skip_online_label="true"
fi
skip_backend_train_label="false"
if ((skip_backend_train)); then
  skip_backend_train_label="true"
fi
printf '  skip_online: %s\n' "${skip_online_label}"
printf '  skip_backend_train: %s\n' "${skip_backend_train_label}"

if ((dry_run)); then
  exit 0
fi

export MUJOCO_GL="${libero_gl_backend}"
if ((skip_online)); then
  for required_path in "${online_results_path}" "${online_context_log_path}" "${trace_path}"; do
    if [ ! -f "${required_path}" ]; then
      echo "error: --skip-online requires existing artifact: ${required_path}" >&2
      exit 1
    fi
  done
else
  "${online_cmd[@]}"
fi
reuse_selector_artifacts=0
if ((skip_online)) \
  && [ -f "${selector_summary_path}" ] \
  && [ -f "${admission_summary_path}" ] \
  && [ -f "${admitted_trace_path}" ]; then
  reuse_selector_artifacts=1
  printf 'reusing existing selector/admission artifacts under %s\n' "${results_dir}"
fi

if ((reuse_selector_artifacts)); then
  :
else
  "${artifact_cmd[@]}"
fi

"${dr_dataset_cmd[@]}"

dr_records_total="$(python3 - "${dr_summary_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload["records_total"]))
PY
)"

if [ "${dr_records_total}" -gt 0 ]; then
  "${dr_calibrator_cmd[@]}"
fi

admitted_trace_records="$(python3 - "${admission_summary_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload["admitted_trace_records"]))
PY
)"
contexts_admitted="$(python3 - "${admission_summary_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload["contexts_admitted"]))
PY
)"

if ((skip_backend_train)); then
  ensure_directory "${training_log_dir}"
  python3 - "${demo_output_path}" "${demo_summary_path}" "${admitted_trace_path}" "${admission_summary_path}" <<'PY'
import json
import pathlib
import sys

demo_output_path = pathlib.Path(sys.argv[1]).resolve()
demo_summary_path = pathlib.Path(sys.argv[2]).resolve()
admitted_trace_path = pathlib.Path(sys.argv[3]).resolve()
admission_summary_path = pathlib.Path(sys.argv[4]).resolve()

with admission_summary_path.open("r", encoding="utf-8") as handle:
    admission = json.load(handle)

payload = {
    "output_path": str(demo_output_path),
    "output_mode": "skipped_backend_train",
    "trace_path": str(admitted_trace_path),
    "summary_path": str(demo_summary_path),
    "demo_items_written": 0,
    "contexts_covered": int(admission["contexts_admitted"]),
    "primitive_steps_total": 0,
    "chunk_action_horizons": [],
    "completed_reason_counts": {},
    "family_counts": {},
    "partition_counts": {},
    "notes": [
        "Replay conversion and backend training were skipped by --skip-backend-train.",
    ],
    "contexts_admitted": int(admission["contexts_admitted"]),
    "admitted_trace_records": int(admission["admitted_trace_records"]),
}
demo_summary_path.parent.mkdir(parents=True, exist_ok=True)
with demo_summary_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
elif [ "${admitted_trace_records}" -gt 0 ] && [ "${contexts_admitted}" -gt 0 ]; then
  reuse_demo_artifacts=0
  if ((skip_online)) && [ -f "${demo_output_path}" ] && [ -f "${demo_summary_path}" ]; then
    reuse_demo_artifacts=1
    printf 'reusing existing admitted-demo artifacts under %s\n' "${results_dir}"
  fi

  if ((reuse_demo_artifacts)); then
    :
  else
    "${convert_cmd[@]}"
  fi

  if [ -d "${training_log_dir}" ] \
    && [ ! -f "${training_log_dir}/replay_buffer_0.pkl" ] \
    && [ ! -f "${training_log_dir}/training_completed.marker" ]; then
    printf 'removing stale training log directory: %s\n' "${training_log_dir}"
    rm -rf -- "${training_log_dir}"
  fi
  "${train_cmd[@]}"
else
  ensure_directory "${training_log_dir}"
  python3 - "${demo_output_path}" "${demo_summary_path}" "${admitted_trace_path}" "${admission_summary_path}" <<'PY'
import json
import pathlib
import sys

demo_output_path = pathlib.Path(sys.argv[1]).resolve()
demo_summary_path = pathlib.Path(sys.argv[2]).resolve()
admitted_trace_path = pathlib.Path(sys.argv[3]).resolve()
admission_summary_path = pathlib.Path(sys.argv[4]).resolve()

with admission_summary_path.open("r", encoding="utf-8") as handle:
    admission = json.load(handle)

payload = {
    "output_path": str(demo_output_path),
    "output_mode": "skipped_no_admitted_contexts",
    "trace_path": str(admitted_trace_path),
    "summary_path": str(demo_summary_path),
    "demo_items_written": 0,
    "contexts_covered": 0,
    "primitive_steps_total": 0,
    "chunk_action_horizons": [],
    "completed_reason_counts": {},
    "family_counts": {},
    "partition_counts": {},
    "notes": [
        "Replay conversion was skipped because admission rejected every executed context.",
    ],
    "contexts_admitted": int(admission["contexts_admitted"]),
    "admitted_trace_records": int(admission["admitted_trace_records"]),
}
demo_summary_path.parent.mkdir(parents=True, exist_ok=True)
with demo_summary_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
fi

python3 - "${online_results_path}" "${selector_summary_path}" "${admission_summary_path}" "${demo_summary_path}" "${dr_summary_path}" "${next_dr_calibrator_summary_path}" "${training_log_dir}" "${round_summary_path}" "${skip_backend_train}" <<'PY'
import json
import pathlib
import sys

online_results_path = pathlib.Path(sys.argv[1]).resolve()
selector_summary_path = pathlib.Path(sys.argv[2]).resolve()
admission_summary_path = pathlib.Path(sys.argv[3]).resolve()
demo_summary_path = pathlib.Path(sys.argv[4]).resolve()
dr_summary_path = pathlib.Path(sys.argv[5]).resolve()
next_dr_calibrator_summary_path = pathlib.Path(sys.argv[6]).resolve()
training_log_dir = pathlib.Path(sys.argv[7]).resolve()
round_summary_path = pathlib.Path(sys.argv[8]).resolve()
skip_backend_train = bool(int(sys.argv[9]))

with online_results_path.open("r", encoding="utf-8") as handle:
    online = json.load(handle)
with selector_summary_path.open("r", encoding="utf-8") as handle:
    selector = json.load(handle)
with admission_summary_path.open("r", encoding="utf-8") as handle:
    admission = json.load(handle)
with demo_summary_path.open("r", encoding="utf-8") as handle:
    demo = json.load(handle)
with dr_summary_path.open("r", encoding="utf-8") as handle:
    dr_dataset = json.load(handle)

if next_dr_calibrator_summary_path.exists():
    with next_dr_calibrator_summary_path.open("r", encoding="utf-8") as handle:
        next_dr_calibrator = json.load(handle)
else:
    next_dr_calibrator = None

replay_buffer_snapshot = training_log_dir / "replay_buffer_0.pkl"
training_completed_marker = training_log_dir / "training_completed.marker"
training_completed = replay_buffer_snapshot.exists() or training_completed_marker.exists()
training_skipped = (
    skip_backend_train
    or int(admission["contexts_admitted"]) <= 0
    or int(admission["admitted_trace_records"]) <= 0
)

summary = {
    "workflow": "stage0_caver_round_v2",
    "online_results_path": str(online_results_path),
    "selector_summary_path": str(selector_summary_path),
    "admission_summary_path": str(admission_summary_path),
    "demo_summary_path": str(demo_summary_path),
    "training_log_dir": str(training_log_dir),
    "online": {
        "episodes_run": online["summary"]["episodes_run"],
        "successes": online["summary"]["successes"],
        "success_rate": online["summary"]["success_rate"],
        "chunk_traces_written": online["summary"]["chunk_traces_written"],
        "candidate_count": online["config"]["candidate_count"],
        "selection_policy": online["config"]["selection_policy"],
        "selector_seed": online["config"]["selector_seed"],
        "value_proxy_model_path": online["config"].get("value_proxy_model_path"),
        "value_proxy_model_id": online["config"].get("value_proxy_model_id"),
        "dr_calibrator_model_path": online["config"].get("dr_calibrator_model_path"),
        "dr_calibrator_model_id": online["config"].get("dr_calibrator_model_id"),
    },
    "selector": {
        "implementation_phase": selector["implementation_phase"],
        "selector_mode": selector["selector_mode"],
        "contexts_total": selector["contexts_total"],
        "contexts_with_candidate_bank": selector["contexts_with_candidate_bank"],
        "policy_queries_total": selector["policy_queries_total"],
    },
    "admission": {
        "implementation_phase": admission["implementation_phase"],
        "admission_policy": admission["admission_policy"],
        "contexts_admitted": admission["contexts_admitted"],
        "contexts_rejected": admission["contexts_rejected"],
        "admitted_trace_records": admission["admitted_trace_records"],
    },
    "demo": {
        "output_path": demo["output_path"],
        "output_mode": demo["output_mode"],
        "demo_items_written": demo["demo_items_written"],
        "contexts_covered": demo["contexts_covered"],
        "primitive_steps_total": demo["primitive_steps_total"],
    },
    "dr_dataset": {
        "summary_path": str(dr_summary_path),
        "records_total": dr_dataset["records_total"],
        "contexts_total": dr_dataset["contexts_total"],
        "selected_records": dr_dataset["selected_records"],
    },
    "next_dr_calibrator": {
        "summary_path": str(next_dr_calibrator_summary_path),
        "fit_completed": bool(next_dr_calibrator is not None),
        "best_epoch": (next_dr_calibrator or {}).get("best_epoch"),
        "best_val_metrics": (next_dr_calibrator or {}).get("best_val_metrics"),
    },
    "training": {
        "replay_buffer_snapshot": str(replay_buffer_snapshot),
        "training_completed_marker": str(training_completed_marker),
        "training_completed": training_completed,
        "training_skipped": training_skipped,
        "skip_backend_train": skip_backend_train,
    },
}

round_summary_path.parent.mkdir(parents=True, exist_ok=True)
with round_summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
