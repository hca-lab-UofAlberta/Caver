#!/usr/bin/env bash
set -euo pipefail

_CAVER_STAGEE_POSTTRAIN_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_STAGEE_POSTTRAIN_SLURM_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  submit_stage0_posttrain_from_round.sh [options]

Required:
  --round-results-dir PATH   Existing Stage-E results directory

Slurm options:
  --dependency SPEC          Slurm dependency spec
  --partition NAME           Slurm partition (default: gpu-l40s)
  --exclude NODES            Comma-separated node exclude list
  --qos NAME                 Slurm QoS (default: normal)
  --gpu-type TYPE            GPU type specifier (default: l40s)
  --gpus COUNT               GPU count (default: 1)
  --time LIMIT               Slurm time limit (default: 08:00:00)
  --cpus COUNT               CPU request (default: 8)
  --mem SIZE                 Memory request (default: 128G)

Pipeline options:
  --method NAME              real_only or caver (default: auto)
  --train-backend NAME       sac_demo or exact_offline_nft (default: exact_offline_nft)
  --config-name NAME         RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --policy-config NAME       OpenPI serve config name (default: pi05_libero)
  --base-model-path PATH     Base OpenPI PyTorch checkpoint dir
  --artifact-root PATH       Output root for post-train artifacts
  --training-log-root PATH   Optional RLinf training log/checkpoint root
  --node-local-training-log-root
                             Use /tmp/<submission-run-id> for RLinf checkpoints
  --cleanup-training-log-dir Remove node-local RLinf logs after successful export
  --task-suite NAME          Backend LIBERO suite (default: libero_90)
  --task-ids IDS             Comma-separated task ids
  --train-max-steps COUNT    RLinf max_steps (default: 20)
  --train-max-epochs COUNT   RLinf max_epochs (default: 20)
  --save-interval COUNT      RLinf save interval (default: 20)
  --micro-batch COUNT        RLinf micro batch size (default: 4)
  --global-batch COUNT       RLinf global batch size (default: 16)
  --min-buffer-size COUNT    Demo buffer gate (default: 256)
  --train-actor-steps COUNT  Actor gate (default: 256)
  --update-epoch COUNT       RLinf update_epoch (default: 1)
  --rollout-steps COUNT      RLinf train/eval rollout horizon (default: 4)
  --exact-rollout-batch-path PATH
                             Reuse an existing exact rollout batch .pt and skip trace conversion
  --manifest-path PATH       Stage-0 manifest (default: metadata/stage0/libero_stage0_partitions.json)
  --val-partition NAME       Held-out validation partition (default: T_val_S0)
  --test-partition NAME      Held-out test partition (default: T_test_S0)
  --eval-partitions NAMES    Comma-separated subset of eval phases: val,test (default: val,test)
  --family-ids IDS           Optional family subset
  --eval-seed COUNT          Held-out eval seed (default: 7)
  --eval-max-contexts COUNT  Optional cap for each held-out partition
  --max-env-steps COUNT      Optional rollout horizon override
  --replan-steps COUNT       Held-out executed chunk horizon (default: 4)
  --libero-gl-backend NAME   egl or osmesa (default: osmesa)
  --skip-train              Reuse existing training outputs
  --skip-export             Reuse existing exported checkpoint
  --skip-eval               Skip held-out evaluation after export
  --dry-run                  Print the resolved sbatch command
  -h, --help                 Show this message
EOF
}

round_results_dir=""
dependency=""
partition="gpu-l40s"
exclude_nodes=""
qos="normal"
gpu_type="l40s"
gpus="1"
time_limit="08:00:00"
cpus="8"
mem="128G"

method=""
train_backend="exact_offline_nft"
config_name="libero_goal_ppo_openpi_pi05"
policy_config="pi05_libero"
base_model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
artifact_root=""
training_log_root=""
node_local_training_log_root=0
cleanup_training_log_dir=0
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
exact_rollout_batch_path=""
manifest_path="${CAVER_REPO_ROOT}/metadata/stage0/libero_stage0_partitions.json"
val_partition="T_val_S0"
test_partition="T_test_S0"
eval_partitions="val,test"
family_ids=""
eval_seed="7"
eval_max_contexts=""
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
    --dependency) dependency="${2:?missing value for --dependency}"; shift 2 ;;
    --partition) partition="${2:?missing value for --partition}"; shift 2 ;;
    --exclude) exclude_nodes="${2:?missing value for --exclude}"; shift 2 ;;
    --qos) qos="${2:?missing value for --qos}"; shift 2 ;;
    --gpu-type) gpu_type="${2:?missing value for --gpu-type}"; shift 2 ;;
    --gpus) gpus="${2:?missing value for --gpus}"; shift 2 ;;
    --time) time_limit="${2:?missing value for --time}"; shift 2 ;;
    --cpus) cpus="${2:?missing value for --cpus}"; shift 2 ;;
    --mem) mem="${2:?missing value for --mem}"; shift 2 ;;
    --method) method="${2:?missing value for --method}"; shift 2 ;;
    --train-backend) train_backend="${2:?missing value for --train-backend}"; shift 2 ;;
    --config-name) config_name="${2:?missing value for --config-name}"; shift 2 ;;
    --policy-config) policy_config="${2:?missing value for --policy-config}"; shift 2 ;;
    --base-model-path) base_model_path="${2:?missing value for --base-model-path}"; shift 2 ;;
    --artifact-root) artifact_root="${2:?missing value for --artifact-root}"; shift 2 ;;
    --training-log-root) training_log_root="${2:?missing value for --training-log-root}"; shift 2 ;;
    --node-local-training-log-root) node_local_training_log_root=1; shift ;;
    --cleanup-training-log-dir) cleanup_training_log_dir=1; shift ;;
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
    --exact-rollout-batch-path) exact_rollout_batch_path="${2:?missing value for --exact-rollout-batch-path}"; shift 2 ;;
    --manifest-path) manifest_path="${2:?missing value for --manifest-path}"; shift 2 ;;
    --val-partition) val_partition="${2:?missing value for --val-partition}"; shift 2 ;;
    --test-partition) test_partition="${2:?missing value for --test-partition}"; shift 2 ;;
    --eval-partitions) eval_partitions="${2:?missing value for --eval-partitions}"; shift 2 ;;
    --family-ids) family_ids="${2:?missing value for --family-ids}"; shift 2 ;;
    --eval-seed) eval_seed="${2:?missing value for --eval-seed}"; shift 2 ;;
    --eval-max-contexts) eval_max_contexts="${2:?missing value for --eval-max-contexts}"; shift 2 ;;
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
base_model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${base_model_path}")"
manifest_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${manifest_path}")"
if [ -n "${artifact_root}" ]; then
  artifact_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${artifact_root}")"
fi
if [ -n "${training_log_root}" ]; then
  training_log_root="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${training_log_root}")"
fi
if [ -n "${exact_rollout_batch_path}" ]; then
  exact_rollout_batch_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${exact_rollout_batch_path}")"
fi

if [ ! -d "${round_results_dir}" ]; then
  echo "error: round results dir not found: ${round_results_dir}" >&2
  exit 1
fi

seed="$(
  python3 - "${round_results_dir}" <<'PY'
import pathlib
import re
import sys

name = pathlib.Path(sys.argv[1]).resolve().parent.name
match = re.search(r"__seed(\d+)__", name)
if not match:
    raise SystemExit("error: could not infer seed from run directory name")
print(match.group(1))
PY
)"
budget="$(
  python3 - "${round_results_dir}" <<'PY'
import pathlib
import re
import sys

name = pathlib.Path(sys.argv[1]).resolve().parent.name
match = re.search(r"__budget(\d+)__", name)
if not match:
    raise SystemExit("error: could not infer budget from run directory name")
print(match.group(1))
PY
)"
method_token="${method:-auto}"
run_id="$(make_run_id "stagee-posttrain" "${method_token}" "heldout-n100" "${seed}" "${budget}")"
job_name="s0pt_$(sanitize_token "${method_token}")_s${seed}_b${budget}"
if ((node_local_training_log_root)) && [ -z "${training_log_root}" ]; then
  training_log_root="/tmp/${run_id}_training_logs"
fi

ensure_directory "${CAVER_DEFAULT_SLURM_LOG_ROOT}"
slurm_stdout="${CAVER_DEFAULT_SLURM_LOG_ROOT}/${run_id}-%j.out"
slurm_stderr="${CAVER_DEFAULT_SLURM_LOG_ROOT}/${run_id}-%j.err"

cmd=(
  sbatch
  "--account=${CAVER_DEFAULT_ACCOUNT}"
  "--partition=${partition}"
  "--qos=${qos}"
  "--gres=gpu:${gpu_type}:${gpus}"
  "--time=${time_limit}"
  "--cpus-per-task=${cpus}"
  "--mem=${mem}"
  "--job-name=${job_name}"
  "--output=${slurm_stdout}"
  "--error=${slurm_stderr}"
)
if [ -n "${dependency}" ]; then
  cmd+=("--dependency=${dependency}")
fi
if [ -n "${exclude_nodes}" ]; then
  cmd+=("--exclude=${exclude_nodes}")
fi
resume_args=()
if ((skip_train)); then
  resume_args+=(--skip-train)
fi
if ((skip_export)); then
  resume_args+=(--skip-export)
fi
if ((skip_eval)); then
  resume_args+=(--skip-eval)
fi
cleanup_args=()
if ((cleanup_training_log_dir)); then
  cleanup_args+=(--cleanup-training-log-dir)
fi
cmd+=(
  --wrap
  "$(printf "%q " \
    env "CAVER_DEFAULT_RUNTIME_LOG_ROOT=/rdss/${CAVER_DEFAULT_ACCOUNT}/${USER}/caver/runtime_logs" \
    "OMP_NUM_THREADS=1" \
    "OPENBLAS_NUM_THREADS=1" \
    "MKL_NUM_THREADS=1" \
    "NUMEXPR_NUM_THREADS=1" \
    "VECLIB_MAXIMUM_THREADS=1" \
    "TORCHINDUCTOR_COMPILE_THREADS=1" \
    "TOKENIZERS_PARALLELISM=false" \
    "${CAVER_REPO_ROOT}/scripts/stagee/run_stage0_posttrain_from_round.sh" \
    --round-results-dir "${round_results_dir}" \
    ${method:+--method "${method}"} \
    --train-backend "${train_backend}" \
    --config-name "${config_name}" \
    --policy-config "${policy_config}" \
    --base-model-path "${base_model_path}" \
    ${artifact_root:+--artifact-root "${artifact_root}"} \
    ${training_log_root:+--training-log-root "${training_log_root}"} \
    "${cleanup_args[@]}" \
    --task-suite "${task_suite}" \
    --task-ids "${task_ids}" \
    --train-max-steps "${train_max_steps}" \
    --train-max-epochs "${train_max_epochs}" \
    --save-interval "${save_interval}" \
    --micro-batch "${micro_batch}" \
    --global-batch "${global_batch}" \
    --min-buffer-size "${min_buffer_size}" \
    --train-actor-steps "${train_actor_steps}" \
    --update-epoch "${update_epoch}" \
    --rollout-steps "${rollout_steps}" \
    ${exact_rollout_batch_path:+--exact-rollout-batch-path "${exact_rollout_batch_path}"} \
    --manifest-path "${manifest_path}" \
    --val-partition "${val_partition}" \
    --test-partition "${test_partition}" \
    --eval-partitions "${eval_partitions}" \
    ${family_ids:+--family-ids "${family_ids}"} \
    --eval-seed "${eval_seed}" \
    ${eval_max_contexts:+--eval-max-contexts "${eval_max_contexts}"} \
    ${max_env_steps:+--max-env-steps "${max_env_steps}"} \
    --replan-steps "${replan_steps}" \
    "${resume_args[@]}" \
    --libero-gl-backend "${libero_gl_backend}")"
)

printf 'submit command:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'slurm stdout: %s\n' "${slurm_stdout}"
printf 'slurm stderr: %s\n' "${slurm_stderr}"

if ((dry_run)); then
  exit 0
fi

"${cmd[@]}"
