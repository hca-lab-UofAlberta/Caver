#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_real_only_round.sh [options]

Online execution options:
  --task-suite NAME            LIBERO suite name for legacy task selection (default: libero_goal)
  --task-ids IDS               Comma-separated task ids for legacy selection (default: 0)
  --num-trials-per-task COUNT  Episode count per legacy task (default: 25)
  --count-legacy-contexts-as-online-budget
                               Count each legacy Stage-E context against the online budget ledger (default: enabled)
  --no-count-legacy-contexts-as-online-budget
                               Leave legacy contexts as non-budgeted evaluation records
  --manifest-path PATH         Optional Stage-0 manifest for manifest-mode selection
  --partition-name NAME        Required with --manifest-path
  --family-ids IDS             Optional manifest family filter
  --context-offset COUNT       Manifest context offset (default: 0)
  --max-contexts COUNT         Manifest context limit
  --candidate-count COUNT      Number of chunk candidates to sample per query (default: 4)
  --selection-policy NAME      first or uniform (default: uniform)
  --selector-seed COUNT        Optional selector RNG seed (defaults to --seed)
  --round-size COUNT           Budget round size annotation (default: 25)
  --num-steps-wait COUNT       Warmup dummy-action steps (default: 10)
  --replan-steps COUNT         Executed action chunk length (default: 5)
  --resize-size COUNT          Policy image size (default: 224)
  --resolution COUNT           LIBERO render resolution (default: 256)
  --max-env-steps COUNT        Optional LIBERO horizon override
  --libero-gl-backend NAME     LIBERO render backend: egl or osmesa (default: osmesa)
  --seed COUNT                 Execution seed (default: 7)
  --server-mode NAME           openpi-native, openpi-exact, or dummy (default: openpi-native)
  --policy-config-name NAME    Optional custom OpenPI server config
  --policy-pretrained-path PATH
                               Optional custom OpenPI checkpoint dir
  --exact-rollout-payload      Use RLinf-backed exact OpenPI serving and log exact rollout payloads
  --exact-rlinf-config-name NAME
                               Optional RLinf config override for exact serving
  --exact-action-chunk COUNT   Optional exact action-chunk override
  --exact-no-nft-loss          Disable NFT traces in exact serving
  --exact-add-value-head       Force a value head in exact serving
  --exact-value-after-vlm      Force value_after_vlm in exact serving
  --exact-solver-type NAME     Exact OpenPI solver type (default: flow_sde)
  --exact-infer-mode MODE      Exact infer mode: train or eval (default: train)

Backend update options:
  --config-name NAME           RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --model-path PATH            Converted OpenPI PyTorch checkpoint directory
  --backend-task-suite NAME    RLinf env suite override for backend config
  --backend-task-ids IDS       RLinf env task ids override for backend config
  --experiment-name NAME       RLinf logger experiment name (default: stage0_real_only_round)
  --train-envs COUNT           Train env count (default: 1)
  --eval-envs COUNT            Eval env count (default: 1)
  --runner-max-steps COUNT     RLinf runner max_steps (default: 1)
  --runner-max-epochs COUNT    RLinf runner max_epochs (default: 1)
  --rollout-steps COUNT        RLinf env rollout-step setting (default: 5)
  --micro-batch COUNT          RLinf actor micro batch size (default: 1)
  --global-batch COUNT         RLinf actor global batch size (default: 2)
  --replay-capacity COUNT      Replay buffer capacity (default: 512)
  --min-buffer-size COUNT      Replay minimum size before update (default: 1)
  --train-actor-steps COUNT    Actor update steps gate (default: 1)

Artifact options:
  --results-dir PATH           Output directory (default: \$CAVER_RUN_DIR/results or a runtime log directory)
  --demo-output-mode NAME      single_pt or sharded_manifest (default: sharded_manifest)
  --max-items-per-shard COUNT  Max shard size for sharded manifests (default: 128)
  --demo-format NAME           chunk_step or primitive_step (default: chunk_step)
  --skip-online                Reuse existing online rollout artifacts under --results-dir
  --dry-run                    Print resolved commands without executing them
  -h, --help                   Show this message
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
selection_policy="uniform"
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

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
backend_task_suite=""
backend_task_ids=""
experiment_name="stage0_real_only_round"
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
    echo "error: unsupported --server-mode ${server_mode}; expected openpi-native, openpi-exact, or dummy" >&2
    exit 1
    ;;
esac

case "${selection_policy}" in
  first|uniform)
    ;;
  *)
    echo "error: unsupported --selection-policy ${selection_policy}; expected first or uniform" >&2
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
    echo "error: unsupported --libero-gl-backend ${libero_gl_backend}; expected egl or osmesa" >&2
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
    results_dir="${CAVER_DEFAULT_RUNTIME_LOG_ROOT}/stagee_real_only_round__$(timestamp_utc)"
  fi
fi
results_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${results_dir}")"
ensure_directory "${results_dir}"

task_token="$(printf "%s" "${backend_task_suite}-task-${backend_task_ids}" | sed -E 's/[^A-Za-z0-9._-]+/-/g')"
online_results_path="${results_dir}/real_only_online_eval.json"
online_context_log_path="${results_dir}/real_only_online_contexts.jsonl"
trace_path="${results_dir}/real_only_online_chunks.jsonl"
demo_summary_path="${results_dir}/real_only_round_demo.summary.json"
training_log_dir="${results_dir}/rlinf_logs"
round_summary_path="${results_dir}/real_only_round_summary.json"
if [ "${demo_output_mode}" = "sharded_manifest" ]; then
  demo_output_path="${results_dir}/real_only_round_demo.manifest.json"
else
  demo_output_path="${results_dir}/real_only_round_demo.pt"
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
  online_cmd+=(
    --config-name "${policy_config_name_effective}"
    --pretrained-path "${policy_pretrained_path_effective}"
  )
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

convert_cmd=(
  "${CAVER_REPO_ROOT}/scripts/stage0/convert_stage0_trace_to_rlinf_demo.sh"
  --trace-path "${trace_path}"
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
printf 'convert command:'
printf ' %q' "${convert_cmd[@]}"
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
printf '  demo_output: %s\n' "${demo_output_path}"
printf '  demo_summary: %s\n' "${demo_summary_path}"
printf '  training_log_dir: %s\n' "${training_log_dir}"
printf '  round_summary: %s\n' "${round_summary_path}"
skip_online_label="false"
if ((skip_online)); then
  skip_online_label="true"
fi
printf '  skip_online: %s\n' "${skip_online_label}"

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
reuse_demo_artifacts=0
if ((skip_online)) && [ -f "${demo_output_path}" ] && [ -f "${demo_summary_path}" ]; then
  reuse_demo_artifacts=1
  printf 'reusing existing demo artifacts under %s\n' "${results_dir}"
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

python3 - "${online_results_path}" "${demo_summary_path}" "${training_log_dir}" "${round_summary_path}" <<'PY'
import json
import pathlib
import sys

online_results_path = pathlib.Path(sys.argv[1]).resolve()
demo_summary_path = pathlib.Path(sys.argv[2]).resolve()
training_log_dir = pathlib.Path(sys.argv[3]).resolve()
round_summary_path = pathlib.Path(sys.argv[4]).resolve()

with online_results_path.open("r", encoding="utf-8") as handle:
    online = json.load(handle)
with demo_summary_path.open("r", encoding="utf-8") as handle:
    demo = json.load(handle)

replay_buffer_snapshot = training_log_dir / "replay_buffer_0.pkl"
training_completed_marker = training_log_dir / "training_completed.marker"
training_completed = replay_buffer_snapshot.exists() or training_completed_marker.exists()

summary = {
    "workflow": "stage0_real_only_round_v1",
    "online_results_path": str(online_results_path),
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
    },
    "demo": {
        "output_path": demo["output_path"],
        "output_mode": demo["output_mode"],
        "demo_items_written": demo["demo_items_written"],
        "contexts_covered": demo["contexts_covered"],
        "primitive_steps_total": demo["primitive_steps_total"],
    },
    "training": {
        "replay_buffer_snapshot": str(replay_buffer_snapshot),
        "training_completed_marker": str(training_completed_marker),
        "training_completed": training_completed,
    },
}

round_summary_path.parent.mkdir(parents=True, exist_ok=True)
with round_summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
