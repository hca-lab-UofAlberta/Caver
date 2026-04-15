#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_POSTTRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_POSTTRAIN_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_posttrain_from_round.sh [options]

Required:
  --round-results-dir PATH   Existing Stage-E round results directory

Training options:
  --method NAME              real_only or caver (default: auto-detect from results dir)
  --train-backend NAME       sac_demo or exact_offline_nft (default: exact_offline_nft)
  --config-name NAME         RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --policy-config NAME       OpenPI serve config name (default: pi05_libero)
  --base-model-path PATH     Base OpenPI PyTorch checkpoint dir
  --artifact-root PATH       Output root for heavy post-train artifacts
  --task-suite NAME          Backend LIBERO suite name (default: libero_90)
  --task-ids IDS             Comma-separated backend task ids
  --train-max-steps COUNT    RLinf runner max_steps (default: 20)
  --train-max-epochs COUNT   RLinf runner max_epochs (default: 20)
  --save-interval COUNT      RLinf save interval (default: 20)
  --micro-batch COUNT        RLinf micro batch size (default: 4)
  --global-batch COUNT       RLinf global batch size (default: 16)
  --min-buffer-size COUNT    Demo buffer gate (default: 256)
  --train-actor-steps COUNT  Demo buffer actor gate (default: 256)
  --update-epoch COUNT       RLinf update_epoch (default: 1)
  --rollout-steps COUNT      RLinf train/eval rollout horizon (default: 4)
  --exact-trace-path PATH    Optional exact admitted-trace source for exact_offline_nft
  --exact-rollout-batch-path PATH
                             Reuse an existing exact rollout batch .pt and skip trace conversion

Held-out evaluation options:
  --manifest-path PATH       Stage-0 manifest (default: metadata/stage0/libero_stage0_partitions.json)
  --val-partition NAME       Validation partition (default: T_val_S0)
  --test-partition NAME      Test partition (default: T_test_S0)
  --family-ids IDS           Optional comma-separated subset of proxy family ids
  --eval-seed COUNT          Evaluation seed (default: 7)
  --eval-max-contexts COUNT  Optional cap for each held-out partition
  --eval-partitions NAMES    Comma-separated subset of eval phases: val,test (default: val,test)
  --max-env-steps COUNT      Optional rollout horizon override for held-out eval
  --replan-steps COUNT       Held-out executed chunk horizon (default: 4)
  --libero-gl-backend NAME   egl or osmesa (default: osmesa)

Resume / debug options:
  --skip-train               Reuse existing training outputs
  --skip-export              Reuse existing exported checkpoint
  --skip-eval                Skip held-out evaluation and only train/export
  --dry-run                  Print resolved commands without running them
  -h, --help                 Show this message
EOF
}

round_results_dir=""
method=""
train_backend="exact_offline_nft"
config_name="libero_goal_ppo_openpi_pi05"
policy_config="pi05_libero"
base_model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
artifact_root=""
task_suite="libero_90"
task_ids="6,7,11,16,17,46,47,48,57,58,59,63,73,74,75"
train_max_steps="20"
train_max_epochs="20"
save_interval="20"
micro_batch="4"
global_batch="16"
min_buffer_size="256"
train_actor_steps="256"
update_epoch="1"
rollout_steps="4"
exact_trace_path=""
exact_rollout_batch_path_override=""

manifest_path="${CAVER_REPO_ROOT}/metadata/stage0/libero_stage0_partitions.json"
val_partition="T_val_S0"
test_partition="T_test_S0"
family_ids=""
eval_seed="7"
eval_max_contexts=""
eval_partitions="val,test"
max_env_steps=""
replan_steps="4"
libero_gl_backend="osmesa"

skip_train=0
skip_export=0
skip_eval=0
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --round-results-dir) round_results_dir="${2:?missing value for --round-results-dir}"; shift 2 ;;
    --method) method="${2:?missing value for --method}"; shift 2 ;;
    --train-backend) train_backend="${2:?missing value for --train-backend}"; shift 2 ;;
    --config-name) config_name="${2:?missing value for --config-name}"; shift 2 ;;
    --policy-config) policy_config="${2:?missing value for --policy-config}"; shift 2 ;;
    --base-model-path) base_model_path="${2:?missing value for --base-model-path}"; shift 2 ;;
    --artifact-root) artifact_root="${2:?missing value for --artifact-root}"; shift 2 ;;
    --task-suite) task_suite="${2:?missing value for --task-suite}"; shift 2 ;;
    --task-ids) task_ids="${2:?missing value for --task-ids}"; shift 2 ;;
    --train-max-steps) train_max_steps="${2:?missing value for --train-max-steps}"; shift 2 ;;
    --train-max-epochs) train_max_epochs="${2:?missing value for --train-max-epochs}"; shift 2 ;;
    --save-interval) save_interval="${2:?missing value for --save-interval}"; shift 2 ;;
    --micro-batch) micro_batch="${2:?missing value for --micro-batch}"; shift 2 ;;
    --global-batch) global_batch="${2:?missing value for --global-batch}"; shift 2 ;;
    --min-buffer-size) min_buffer_size="${2:?missing value for --min-buffer-size}"; shift 2 ;;
    --train-actor-steps) train_actor_steps="${2:?missing value for --train-actor-steps}"; shift 2 ;;
    --update-epoch) update_epoch="${2:?missing value for --update-epoch}"; shift 2 ;;
    --rollout-steps) rollout_steps="${2:?missing value for --rollout-steps}"; shift 2 ;;
    --exact-trace-path) exact_trace_path="${2:?missing value for --exact-trace-path}"; shift 2 ;;
    --exact-rollout-batch-path) exact_rollout_batch_path_override="${2:?missing value for --exact-rollout-batch-path}"; shift 2 ;;
    --manifest-path) manifest_path="${2:?missing value for --manifest-path}"; shift 2 ;;
    --val-partition) val_partition="${2:?missing value for --val-partition}"; shift 2 ;;
    --test-partition) test_partition="${2:?missing value for --test-partition}"; shift 2 ;;
    --family-ids) family_ids="${2:?missing value for --family-ids}"; shift 2 ;;
    --eval-seed) eval_seed="${2:?missing value for --eval-seed}"; shift 2 ;;
    --eval-max-contexts) eval_max_contexts="${2:?missing value for --eval-max-contexts}"; shift 2 ;;
    --eval-partitions) eval_partitions="${2:?missing value for --eval-partitions}"; shift 2 ;;
    --max-env-steps) max_env_steps="${2:?missing value for --max-env-steps}"; shift 2 ;;
    --replan-steps) replan_steps="${2:?missing value for --replan-steps}"; shift 2 ;;
    --libero-gl-backend) libero_gl_backend="${2:?missing value for --libero-gl-backend}"; shift 2 ;;
    --skip-train) skip_train=1; shift ;;
    --skip-export) skip_export=1; shift ;;
    --skip-eval) skip_eval=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: ${1}" >&2; usage >&2; exit 1 ;;
  esac
done

require_command python3

if [ -z "${round_results_dir}" ]; then
  echo "error: --round-results-dir is required" >&2
  exit 1
fi

round_results_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${round_results_dir}")"
manifest_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${manifest_path}")"
base_model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${base_model_path}")"

if [ ! -d "${round_results_dir}" ]; then
  echo "error: round results dir not found: ${round_results_dir}" >&2
  exit 1
fi
if [ ! -f "${manifest_path}" ]; then
  echo "error: manifest not found: ${manifest_path}" >&2
  exit 1
fi

if [ -z "${method}" ]; then
  if [ -f "${round_results_dir}/real_only_round_summary.json" ]; then
    method="real_only"
  elif [ -f "${round_results_dir}/caver_round_summary.json" ]; then
    method="caver"
  else
    echo "error: could not infer method from ${round_results_dir}" >&2
    exit 1
  fi
fi
case "${train_backend}" in
  sac_demo|exact_offline_nft) ;;
  *)
    echo "error: unsupported --train-backend ${train_backend}" >&2
    exit 1
    ;;
esac

case "${method}" in
  real_only)
    demo_manifest="${round_results_dir}/real_only_round_demo.manifest.json"
    round_summary_source="${round_results_dir}/real_only_round_summary.json"
    default_exact_trace_path="${round_results_dir}/real_only_online_chunks.jsonl"
    ;;
  caver)
    demo_manifest="${round_results_dir}/caver_round_demo.manifest.json"
    round_summary_source="${round_results_dir}/caver_round_summary.json"
    default_exact_trace_path="${round_results_dir}/caver_admitted_chunks.jsonl"
    ;;
  *)
    echo "error: unsupported method ${method}" >&2
    exit 1
    ;;
esac

if [ "${train_backend}" = "sac_demo" ] && [ ! -f "${demo_manifest}" ]; then
  echo "error: expected demo manifest missing: ${demo_manifest}" >&2
  exit 1
fi
if [ "${train_backend}" = "exact_offline_nft" ]; then
  if [ -z "${exact_trace_path}" ]; then
    exact_trace_path="${default_exact_trace_path}"
  fi
  exact_trace_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${exact_trace_path}")"
  if [ ! -f "${exact_trace_path}" ]; then
    echo "error: exact trace path not found: ${exact_trace_path}" >&2
    exit 1
  fi
  python3 - "${exact_trace_path}" <<'PY'
import json
import pathlib
import sys

trace_path = pathlib.Path(sys.argv[1]).resolve()
with trace_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        selected_policy_aux = record.get("selected_policy_aux")
        if not isinstance(selected_policy_aux, dict) or not isinstance(
            selected_policy_aux.get("forward_inputs"), dict
        ):
            context_id = record.get("context_id", "<unknown>")
            raise SystemExit(
                "error: exact_offline_nft requires exact-payload traces, but "
                f"{trace_path} is missing selected_policy_aux.forward_inputs at "
                f"line {line_number} context {context_id}. "
                "This round was likely collected with openpi-native. "
                "Re-run the online round with --server-mode openpi-exact or "
                "--exact-rollout-payload, or use --train-backend sac_demo."
            )
        break
PY
  if [ "${config_name}" = "libero_goal_ppo_openpi_pi05" ]; then
    config_name="libero_goal_nft_actor_openpi_pi05"
  fi
fi

experiment_name="stage0_posttrain_${method}"
round_run_dir="$(dirname -- "${round_results_dir}")"
round_run_name="$(basename -- "${round_run_dir}")"
if [ -z "${artifact_root}" ]; then
  artifact_root="${CAVER_DEFAULT_RDSS_ROOT}/caver/stagee_posttrain/${round_run_name}"
else
  artifact_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${artifact_root}")"
fi
ensure_directory "${artifact_root}"
training_log_dir="${artifact_root}/posttrain_train"
export_dir="${artifact_root}/posttrain_checkpoint"
export_summary_path="${artifact_root}/posttrain_checkpoint_export.summary.json"
exact_rollout_batch_path="${artifact_root}/posttrain_exact_rollout_batch.pt"
exact_rollout_batch_summary_path="${artifact_root}/posttrain_exact_rollout_batch.summary.json"
val_results_path="${artifact_root}/posttrain_eval_${val_partition}.json"
val_context_log_path="${artifact_root}/posttrain_eval_${val_partition}.contexts.jsonl"
test_results_path="${artifact_root}/posttrain_eval_${test_partition}.json"
test_context_log_path="${artifact_root}/posttrain_eval_${test_partition}.contexts.jsonl"
posttrain_summary_path="${artifact_root}/posttrain_holdout_summary.json"

train_data_path="${demo_manifest}"
train_data_type="robot_demo"
train_loss_type="embodied_sac"
convert_cmd=()
if [ "${train_backend}" = "exact_offline_nft" ]; then
  if [ -n "${exact_rollout_batch_path_override}" ]; then
    train_data_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${exact_rollout_batch_path_override}")"
    if [ ! -f "${train_data_path}" ]; then
      echo "error: exact rollout batch path not found: ${train_data_path}" >&2
      exit 1
    fi
    exact_rollout_batch_path="${train_data_path}"
  else
    train_data_path="${exact_rollout_batch_path}"
  fi
  train_data_type="embodied_rollout_batch"
  train_loss_type=""
  if [ -z "${exact_rollout_batch_path_override}" ]; then
    convert_cmd=(
      "${CAVER_REPO_ROOT}/scripts/stagee/convert_stagee_trace_to_offline_rollout_batch.sh"
      --trace-path "${exact_trace_path}"
      --output-path "${exact_rollout_batch_path}"
      --summary-path "${exact_rollout_batch_summary_path}"
      --openpi-config-name "${policy_config}"
    )
  fi
fi

train_cmd=(
  "${CAVER_REPO_ROOT}/scripts/pistepnft/run_stage0_demo_training.sh"
  --config-name "${config_name}"
  --model-path "${base_model_path}"
  --data-path "${train_data_path}"
  --data-type "${train_data_type}"
  --experiment-name "${experiment_name}"
  --log-dir "${training_log_dir}"
  --task-suite "${task_suite}"
  --task-ids "${task_ids}"
  --max-steps "${train_max_steps}"
  --max-epochs "${train_max_epochs}"
  --save-interval "${save_interval}"
  --micro-batch "${micro_batch}"
  --global-batch "${global_batch}"
  --min-buffer-size "${min_buffer_size}"
  --train-actor-steps "${train_actor_steps}"
  --update-epoch "${update_epoch}"
  --rollout-steps "${rollout_steps}"
  --action-chunk "${rollout_steps}"
)
if [ -n "${train_loss_type}" ]; then
  train_cmd+=(--algorithm-loss-type "${train_loss_type}")
fi

if [ -n "${eval_max_contexts}" ]; then
  eval_max_contexts_args=(--max-contexts "${eval_max_contexts}")
else
  eval_max_contexts_args=()
fi
if [ -n "${family_ids}" ]; then
  family_args=(--family-ids "${family_ids}")
else
  family_args=()
fi
if [ -n "${max_env_steps}" ]; then
  max_env_steps_args=(--max-steps "${max_env_steps}")
else
  max_env_steps_args=()
fi

eval_partitions_csv=",${eval_partitions},"
run_val_partition=0
run_test_partition=0
if [[ "${eval_partitions_csv}" == *",val,"* ]]; then
  run_val_partition=1
fi
if [[ "${eval_partitions_csv}" == *",test,"* ]]; then
  run_test_partition=1
fi
if ((skip_eval == 0)) && ((run_val_partition == 0 && run_test_partition == 0)); then
  echo "error: --eval-partitions must include at least one of val,test when eval is enabled" >&2
  exit 1
fi

export_cmd=(
  "${CAVER_REPO_ROOT}/scripts/openpi/export_rlinf_actor_checkpoint_to_pytorch.sh"
  --actor-checkpoint-dir "__ACTOR_CHECKPOINT_DIR__"
  --base-model-path "${base_model_path}"
  --output-path "${export_dir}"
  --summary-path "${export_summary_path}"
)

make_eval_cmd() {
  local partition_name="$1"
  local results_path="$2"
  local context_log_path="$3"
  local -a cmd=(
    "${CAVER_REPO_ROOT}/scripts/bridge/run_libero_remote_eval.sh"
    --openpi-native
    --config-name "${policy_config}"
    --pretrained-path "${export_dir}"
    --
    --manifest-path "${manifest_path}"
    --partition-name "${partition_name}"
    --seed "${eval_seed}"
    --candidate-count "1"
    --selection-policy "first"
    --num-steps-wait "10"
    --replan-steps "${replan_steps}"
    --resize-size "224"
    --resolution "256"
    --results-path "${results_path}"
    --context-log-path "${context_log_path}"
  )
  if [ "${#family_args[@]}" -gt 0 ]; then
    cmd+=("${family_args[@]}")
  fi
  if [ "${#eval_max_contexts_args[@]}" -gt 0 ]; then
    cmd+=("${eval_max_contexts_args[@]}")
  fi
  if [ "${#max_env_steps_args[@]}" -gt 0 ]; then
    cmd+=("${max_env_steps_args[@]}")
  fi
  printf '%q ' "${cmd[@]}"
}

printf 'train command:'
printf ' %q' "${train_cmd[@]}"
printf '\n'
if [ "${#convert_cmd[@]}" -gt 0 ]; then
  printf 'convert command:'
  printf ' %q' "${convert_cmd[@]}"
  printf '\n'
fi
printf 'export command template:'
printf ' %q' "${export_cmd[@]}"
printf '\n'
if ((run_val_partition)); then
  printf 'val eval command: %s\n' "$(make_eval_cmd "${val_partition}" "${val_results_path}" "${val_context_log_path}")"
fi
if ((run_test_partition)); then
  printf 'test eval command: %s\n' "$(make_eval_cmd "${test_partition}" "${test_results_path}" "${test_context_log_path}")"
fi
printf 'artifacts:\n'
  printf '  round_results_dir: %s\n' "${round_results_dir}"
  printf '  round_summary_source: %s\n' "${round_summary_source}"
printf '  artifact_root: %s\n' "${artifact_root}"
printf '  train_backend: %s\n' "${train_backend}"
printf '  train_data_path: %s\n' "${train_data_path}"
printf '  train_data_type: %s\n' "${train_data_type}"
if [ "${train_backend}" = "exact_offline_nft" ]; then
  printf '  exact_trace_path: %s\n' "${exact_trace_path}"
  printf '  exact_rollout_batch_path: %s\n' "${exact_rollout_batch_path}"
fi
printf '  training_log_dir: %s\n' "${training_log_dir}"
printf '  export_dir: %s\n' "${export_dir}"
printf '  posttrain_summary: %s\n' "${posttrain_summary_path}"

if ((dry_run)); then
  exit 0
fi

export MUJOCO_GL="${libero_gl_backend}"

if ((skip_train == 0)); then
  if [ -d "${training_log_dir}" ]; then
    rm -rf -- "${training_log_dir}"
  fi
  if [ "${#convert_cmd[@]}" -gt 0 ]; then
    rm -f -- "${exact_rollout_batch_path}" "${exact_rollout_batch_summary_path}"
    "${convert_cmd[@]}"
  fi
  "${train_cmd[@]}"
elif [ "${train_backend}" = "exact_offline_nft" ] && [ ! -f "${exact_rollout_batch_path}" ]; then
  echo "error: --skip-train requested but exact rollout batch is missing: ${exact_rollout_batch_path}" >&2
  exit 1
fi

checkpoint_root="${training_log_dir}/${experiment_name}/checkpoints"
latest_actor_checkpoint="$(
  python3 - "${checkpoint_root}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve()
paths = []
if root.is_dir():
    pattern = re.compile(r"global_step_(\d+)$")
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        match = pattern.match(candidate.name)
        if match:
            paths.append((int(match.group(1)), candidate / "actor"))
if not paths:
    raise SystemExit("")
paths.sort()
print(paths[-1][1])
PY
)"
if [ -z "${latest_actor_checkpoint}" ]; then
  echo "error: no RLinf actor checkpoint found under ${checkpoint_root}" >&2
  exit 1
fi

if ((skip_export == 0)); then
  rm -rf -- "${export_dir}"
  export_cmd_resolved=("${export_cmd[@]/__ACTOR_CHECKPOINT_DIR__/${latest_actor_checkpoint}}")
  "${export_cmd_resolved[@]}"
fi

if ((skip_eval == 0)); then
  val_eval_cmd=(
    "${CAVER_REPO_ROOT}/scripts/bridge/run_libero_remote_eval.sh"
    --openpi-native
    --config-name "${policy_config}"
    --pretrained-path "${export_dir}"
    --
    --manifest-path "${manifest_path}"
    --partition-name "${val_partition}"
    --seed "${eval_seed}"
    --candidate-count "1"
    --selection-policy "first"
    --num-steps-wait "10"
    --replan-steps "${replan_steps}"
    --resize-size "224"
    --resolution "256"
    --results-path "${val_results_path}"
    --context-log-path "${val_context_log_path}"
  )
  test_eval_cmd=(
    "${CAVER_REPO_ROOT}/scripts/bridge/run_libero_remote_eval.sh"
    --openpi-native
    --config-name "${policy_config}"
    --pretrained-path "${export_dir}"
    --
    --manifest-path "${manifest_path}"
    --partition-name "${test_partition}"
    --seed "${eval_seed}"
    --candidate-count "1"
    --selection-policy "first"
    --num-steps-wait "10"
    --replan-steps "${replan_steps}"
    --resize-size "224"
    --resolution "256"
    --results-path "${test_results_path}"
    --context-log-path "${test_context_log_path}"
  )
  if [ "${#family_args[@]}" -gt 0 ]; then
    val_eval_cmd+=("${family_args[@]}")
    test_eval_cmd+=("${family_args[@]}")
  fi
  if [ "${#eval_max_contexts_args[@]}" -gt 0 ]; then
    val_eval_cmd+=("${eval_max_contexts_args[@]}")
    test_eval_cmd+=("${eval_max_contexts_args[@]}")
  fi
  if [ "${#max_env_steps_args[@]}" -gt 0 ]; then
    val_eval_cmd+=("${max_env_steps_args[@]}")
    test_eval_cmd+=("${max_env_steps_args[@]}")
  fi

  if ((run_val_partition)); then
    "${val_eval_cmd[@]}"
  fi
  if ((run_test_partition)); then
    "${test_eval_cmd[@]}"
  fi
fi

python3 - \
  "${method}" \
  "${train_backend}" \
  "${round_summary_source}" \
  "${train_data_path}" \
  "${training_log_dir}" \
  "${latest_actor_checkpoint}" \
  "${export_dir}" \
  "${export_summary_path}" \
  "${exact_trace_path}" \
  "${exact_rollout_batch_summary_path}" \
  "${val_results_path}" \
  "${test_results_path}" \
  "${posttrain_summary_path}" <<'PY'
import json
import pathlib
import sys

(
    method,
    train_backend,
    round_summary_source,
    train_data_path,
    training_log_dir,
    latest_actor_checkpoint,
    export_dir,
    export_summary_path,
    exact_trace_path,
    exact_rollout_batch_summary_path,
    val_results_path,
    test_results_path,
    posttrain_summary_path,
) = sys.argv[1:]

round_summary_source = pathlib.Path(round_summary_source).resolve()
train_data_path = pathlib.Path(train_data_path).resolve()
training_log_dir = pathlib.Path(training_log_dir).resolve()
latest_actor_checkpoint = pathlib.Path(latest_actor_checkpoint).resolve()
export_dir = pathlib.Path(export_dir).resolve()
export_summary_path = pathlib.Path(export_summary_path).resolve()
exact_trace_path = pathlib.Path(exact_trace_path).resolve() if exact_trace_path else None
exact_rollout_batch_summary_path = (
    pathlib.Path(exact_rollout_batch_summary_path).resolve()
    if exact_rollout_batch_summary_path
    else None
)
val_results_path = pathlib.Path(val_results_path).resolve()
test_results_path = pathlib.Path(test_results_path).resolve()
posttrain_summary_path = pathlib.Path(posttrain_summary_path).resolve()

def load_json_if_exists(path: pathlib.Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

round_summary = load_json_if_exists(round_summary_source)
export_summary = load_json_if_exists(export_summary_path)
exact_rollout_batch_summary = load_json_if_exists(exact_rollout_batch_summary_path)
val_results = load_json_if_exists(val_results_path)
test_results = load_json_if_exists(test_results_path)

summary = {
    "workflow": "stage0_posttrain_from_round_v1",
    "method": method,
    "train_backend": train_backend,
    "round_summary_source": str(round_summary_source),
    "train_data_path": str(train_data_path),
    "training_log_dir": str(training_log_dir),
    "latest_actor_checkpoint": str(latest_actor_checkpoint),
    "export_dir": str(export_dir),
    "export_summary_path": str(export_summary_path),
    "base_round_online": (round_summary or {}).get("online"),
    "heldout": {},
}
if export_summary is not None:
    summary["export"] = export_summary
if exact_trace_path is not None:
    summary["exact_trace_path"] = str(exact_trace_path)
if exact_rollout_batch_summary is not None:
    summary["exact_rollout_batch"] = exact_rollout_batch_summary
if val_results is not None:
    summary["heldout"]["validation"] = {
        "results_path": str(val_results_path),
        "episodes_run": val_results["summary"]["episodes_run"],
        "successes": val_results["summary"]["successes"],
        "success_rate": val_results["summary"]["success_rate"],
    }
if test_results is not None:
    summary["heldout"]["test"] = {
        "results_path": str(test_results_path),
        "episodes_run": test_results["summary"]["episodes_run"],
        "successes": test_results["summary"]["successes"],
        "success_rate": test_results["summary"]["success_rate"],
    }

posttrain_summary_path.parent.mkdir(parents=True, exist_ok=True)
posttrain_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
