#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_CAVER_BUDGET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_CAVER_BUDGET_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_caver_budget.sh [options]

Slurm options:
  --dependency SPEC          Slurm dependency spec, for example afterok:5480
  --partition NAME           Slurm partition (default: gpu-l40s)
  --qos NAME                 Slurm QoS (default: normal)
  --gpu-type TYPE            GPU type specifier (default: l40s)
  --gpus COUNT               GPU count (default: 1)
  --time LIMIT               Slurm time limit (default: 03:00:00)
  --cpus COUNT               CPU request (default: 8)
  --mem SIZE                 Memory request (default: 128G)
  --run-root PATH            Run directory root passed to the round submitter
  --log-root PATH            Slurm log root passed to the round submitter

Selection options:
  --manifest-path PATH       Source Stage-0 manifest (default: metadata/stage0/libero_stage0_partitions.json)
  --partition-name NAME      Source partition to slice (default: T_train_S0)
  --family-ids IDS           Optional comma-separated family ids (default: all families)
  --budget COUNT             Total contexts across selected families (default: 25)
  --family-offset COUNT      Starting offset within each family partition (default: 0)
  --round-size COUNT         Stage-E round size metadata (default: 25)
  --seed COUNT               Run seed (default: 7)

Execution options:
  --candidate-count COUNT    Number of chunk candidates per query (default: 4)
  --selection-policy NAME    first, uniform, or caver_heuristic (default: caver_heuristic)
  --selector-seed COUNT      Optional selector RNG seed
  --num-steps-wait COUNT     Bridge wait interval between policy queries (default: 10)
  --replan-steps COUNT       Chunk horizon in action steps (default: 5)
  --resize-size COUNT        Image resize sent to the policy (default: 224)
  --resolution COUNT         Simulator capture resolution (default: 256)
  --libero-gl-backend NAME   egl or osmesa (default: osmesa)
  --max-env-steps COUNT      Optional LIBERO rollout horizon override (default: suite-native horizon)
  --server-mode NAME         openpi-native, openpi-exact, or dummy (default: openpi-native)
  --policy-config-name NAME  Optional custom OpenPI config
  --policy-pretrained-path PATH
                             Optional custom OpenPI checkpoint dir
  --exact-rollout-payload    Use RLinf-backed exact OpenPI serving and log exact rollout payloads
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
  --provider-mode NAME       none, gesim_bundle, or gesim_live_summary (default: none)
  --provider-bundle-root PATH
                             Optional provider bundle / inference root for the round runner
  --provider-gesim-timeout-sec COUNT
  --provider-gesim-prompt TEXT
  --no-require-candidate-bank
                             Allow traces without full candidate-bank logging

Backend options:
  --config-name NAME         RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --model-path PATH          Converted OpenPI PyTorch checkpoint dir
  --experiment-name NAME     RLinf logger experiment name (default: stage0_caver_budget)
  --train-envs COUNT         Train env count (default: 1)
  --eval-envs COUNT          Eval env count (default: 1)
  --runner-max-steps COUNT   RLinf runner max_steps (default: 1)
  --runner-max-epochs COUNT  RLinf runner max_epochs (default: 1)
  --rollout-steps COUNT      RLinf env rollout-step setting (default: 5)
  --micro-batch COUNT        RLinf actor micro batch size (default: 1)
  --global-batch COUNT       RLinf actor global batch size (default: 2)
  --replay-capacity COUNT    Replay buffer capacity (default: 512)
  --min-buffer-size COUNT    Replay minimum size before update (default: 1)
  --train-actor-steps COUNT  Actor update steps gate (default: 1)

Artifact options:
  --demo-output-mode NAME    single_pt or sharded_manifest (default: sharded_manifest)
  --max-items-per-shard COUNT
                             Max shard size for sharded manifests (default: 128)
  --demo-format NAME         chunk_step or primitive_step (default: chunk_step)
  --dry-run                  Print the derived manifest path and resolved submit command
  -h, --help                 Show this message
EOF
}

dependency=""
partition="gpu-l40s"
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="03:00:00"
cpus="8"
mem="128G"
run_root=""
log_root=""

manifest_path="${CAVER_REPO_ROOT}/metadata/stage0/libero_stage0_partitions.json"
partition_name="T_train_S0"
family_ids=""
budget="25"
family_offset="0"
round_size="25"
seed="7"

candidate_count="4"
selection_policy="caver_heuristic"
selector_seed=""
num_steps_wait="10"
replan_steps="5"
resize_size="224"
resolution="256"
libero_gl_backend="osmesa"
max_env_steps=""
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
experiment_name="stage0_caver_budget"
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
    --budget)
      budget="${2:?missing value for --budget}"
      shift 2
      ;;
    --family-offset)
      family_offset="${2:?missing value for --family-offset}"
      shift 2
      ;;
    --round-size)
      round_size="${2:?missing value for --round-size}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing value for --seed}"
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
    --libero-gl-backend)
      libero_gl_backend="${2:?missing value for --libero-gl-backend}"
      shift 2
      ;;
    --max-env-steps)
      max_env_steps="${2:?missing value for --max-env-steps}"
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

require_command python3

manifest_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${manifest_path}")"
model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${model_path}")"

if [ ! -f "${manifest_path}" ]; then
  echo "error: manifest not found: ${manifest_path}" >&2
  exit 1
fi

if [ -n "${policy_config_name}" ] || [ -n "${policy_pretrained_path}" ]; then
  if [ -z "${policy_config_name}" ] || [ -z "${policy_pretrained_path}" ]; then
    echo "error: --policy-config-name and --policy-pretrained-path must be provided together" >&2
    exit 1
  fi
  policy_pretrained_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${policy_pretrained_path}")"
fi

ensure_directory "${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stagee_manifests"
selection_stamp="$(timestamp_utc)"
family_token="$(sanitize_token "${family_ids:-all}")"
selection_manifest="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stagee_manifests/stagee_caver__$(sanitize_token "${partition_name}")__budget${budget}__offset${family_offset}__seed${seed}__${family_token}__${selection_stamp}.json"
selection_summary="${selection_manifest%.json}.summary.json"

python3 "${CAVER_REPO_ROOT}/scripts/stagee/build_stage0_balanced_manifest.py" \
  --input-manifest "${manifest_path}" \
  --output-manifest "${selection_manifest}" \
  --partition-name "${partition_name}" \
  --budget "${budget}" \
  --family-offset "${family_offset}" \
  --round-size "${round_size}" \
  ${family_ids:+--family-ids "${family_ids}"}

selection_info="$(
  python3 - "${selection_manifest}" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
backend = manifest["backend"]
task_suite_names = backend.get("task_suite_names", [])
if len(task_suite_names) != 1:
    raise SystemExit(f"error: expected exactly one backend task suite, found {task_suite_names}")

print(json.dumps(
    {
        "selected_family_ids": manifest["selection"]["selected_family_ids"],
        "contexts_per_family": manifest["selection"]["contexts_per_family"],
        "backend_task_suite": task_suite_names[0],
        "backend_task_ids": ",".join(str(value) for value in backend.get("task_ids", [])),
    },
    sort_keys=True,
))
PY
)"

backend_task_suite="$(
  python3 - "${selection_info}" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["backend_task_suite"])
PY
)"
backend_task_ids="$(
  python3 - "${selection_info}" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["backend_task_ids"])
PY
)"

submit_cmd=(
  "${CAVER_REPO_ROOT}/scripts/slurm/submit_stage0_caver_round.sh"
  --partition "${partition}"
  --qos "${qos}"
  --gpu-type "${gpu_type}"
  --gpus "${gpus}"
  --time "${time_limit}"
  --cpus "${cpus}"
  --mem "${mem}"
  --manifest-path "${selection_manifest}"
  --partition-name "${partition_name}"
  --max-contexts "${budget}"
  --round-size "${round_size}"
  --seed "${seed}"
  --candidate-count "${candidate_count}"
  --selection-policy "${selection_policy}"
  --num-steps-wait "${num_steps_wait}"
  --replan-steps "${replan_steps}"
  --resize-size "${resize_size}"
  --resolution "${resolution}"
  --libero-gl-backend "${libero_gl_backend}"
  --selector-mode "${selector_mode}"
  --admission-policy "${admission_policy}"
  --config-name "${config_name}"
  --model-path "${model_path}"
  --backend-task-suite "${backend_task_suite}"
  --backend-task-ids "${backend_task_ids}"
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

if [ -n "${dependency}" ]; then
  submit_cmd+=(--dependency "${dependency}")
fi
if [ -n "${run_root}" ]; then
  submit_cmd+=(--run-root "${run_root}")
fi
if [ -n "${log_root}" ]; then
  submit_cmd+=(--log-root "${log_root}")
fi
if [ -n "${family_ids}" ]; then
  submit_cmd+=(--family-ids "${family_ids}")
fi
if [ -n "${selector_seed}" ]; then
  submit_cmd+=(--selector-seed "${selector_seed}")
fi
if [ -n "${value_proxy_model_path}" ]; then
  submit_cmd+=(--value-proxy-model-path "${value_proxy_model_path}")
fi
if [ -n "${dr_calibrator_model_path}" ]; then
  submit_cmd+=(--dr-calibrator-model-path "${dr_calibrator_model_path}")
fi
if [ -n "${provider_mode}" ]; then
  submit_cmd+=(--provider-mode "${provider_mode}")
fi
if [ -n "${provider_bundle_root}" ]; then
  submit_cmd+=(--provider-bundle-root "${provider_bundle_root}")
fi
if [ -n "${provider_gesim_timeout_sec}" ]; then
  submit_cmd+=(--provider-gesim-timeout-sec "${provider_gesim_timeout_sec}")
fi
if [ -n "${provider_gesim_prompt}" ]; then
  submit_cmd+=(--provider-gesim-prompt "${provider_gesim_prompt}")
fi
if [ -n "${max_env_steps}" ]; then
  submit_cmd+=(--max-env-steps "${max_env_steps}")
fi
if [ -n "${policy_config_name}" ]; then
  submit_cmd+=(--policy-config-name "${policy_config_name}" --policy-pretrained-path "${policy_pretrained_path}")
fi
if [ -n "${server_mode}" ]; then
  submit_cmd+=(--server-mode "${server_mode}")
fi
if ((exact_rollout_payload)); then
  submit_cmd+=(--exact-rollout-payload)
fi
if [ -n "${exact_rlinf_config_name}" ]; then
  submit_cmd+=(--exact-rlinf-config-name "${exact_rlinf_config_name}")
fi
if [ -n "${exact_action_chunk}" ]; then
  submit_cmd+=(--exact-action-chunk "${exact_action_chunk}")
fi
if ((exact_no_nft_loss)); then
  submit_cmd+=(--exact-no-nft-loss)
fi
if ((exact_add_value_head)); then
  submit_cmd+=(--exact-add-value-head)
fi
if ((exact_value_after_vlm)); then
  submit_cmd+=(--exact-value-after-vlm)
fi
if [ -n "${exact_solver_type}" ]; then
  submit_cmd+=(--exact-solver-type "${exact_solver_type}")
fi
if [ -n "${exact_infer_mode}" ]; then
  submit_cmd+=(--exact-infer-mode "${exact_infer_mode}")
fi
if (( ! require_candidate_bank )); then
  submit_cmd+=(--no-require-candidate-bank)
fi

printf 'selection manifest: %s\n' "${selection_manifest}"
printf 'selection summary: %s\n' "${selection_summary}"
printf 'selection info: %s\n' "${selection_info}"
printf 'submit command:'
printf ' %q' "${submit_cmd[@]}"
printf '\n'

python3 - "${selection_manifest}" "${selection_summary}" "${selection_info}" "${budget}" "${seed}" "${partition_name}" "${family_offset}" "${selector_mode}" "${admission_policy}" "${require_candidate_bank}" "${provider_mode}" "${provider_bundle_root}" "${provider_gesim_timeout_sec}" "${provider_gesim_prompt}" <<'PY'
import json
import pathlib
import sys

selection_manifest = pathlib.Path(sys.argv[1]).resolve()
selection_summary = pathlib.Path(sys.argv[2]).resolve()
selection_info = json.loads(sys.argv[3])
summary = {
    "selection_manifest": str(selection_manifest),
    "budget": int(sys.argv[4]),
    "seed": int(sys.argv[5]),
    "partition_name": sys.argv[6],
    "family_offset": int(sys.argv[7]),
    "selector_mode": sys.argv[8],
    "admission_policy": sys.argv[9],
    "require_candidate_bank": bool(int(sys.argv[10])),
    "provider_mode": sys.argv[11],
    "provider_bundle_root": sys.argv[12] or None,
    "provider_gesim_timeout_sec": int(sys.argv[13]),
    "provider_gesim_prompt": sys.argv[14],
    "selected_family_ids": selection_info["selected_family_ids"],
    "contexts_per_family": selection_info["contexts_per_family"],
    "backend_task_suite": selection_info["backend_task_suite"],
    "backend_task_ids": selection_info["backend_task_ids"],
}
selection_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if ((dry_run)); then
  exit 0
fi

submission_output="$("${submit_cmd[@]}")"
echo "${submission_output}"
