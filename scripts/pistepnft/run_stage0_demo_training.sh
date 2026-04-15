#!/usr/bin/env bash
set -euo pipefail

_CAVER_PISTEPNFT_TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_PISTEPNFT_TRAIN_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_demo_training.sh [options]

Options:
  --config-name NAME         RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --model-path PATH          Converted OpenPI PyTorch checkpoint directory
  --demo-manifest PATH       Backward-compatible alias for --data-path
  --data-path PATH           Offline dataset path for RLinf training
  --data-type NAME           robot_demo or embodied_rollout_batch (default: robot_demo)
  --algorithm-loss-type NAME Optional algorithm.loss_type override. Defaults to embodied_sac for robot_demo and preserves the config value otherwise
  --experiment-name NAME     Logger experiment name (default: stage0_posttrain_demo)
  --log-dir PATH             Training log directory
  --task-suite NAME          LIBERO suite name (default: libero_90)
  --task-ids IDS             Comma-separated task ids for config alignment
  --train-envs COUNT         Train env count override (default: 1)
  --eval-envs COUNT          Eval env count override (default: 1)
  --max-steps COUNT          Runner max_steps (default: 20)
  --max-epochs COUNT         Runner max_epochs cap (default: 20)
  --save-interval COUNT      Runner save interval (default: 20)
  --val-check-interval COUNT Runner validation interval (default: -1)
  --rollout-steps COUNT      Train/eval max rollout steps (default: 5)
  --action-chunk COUNT       Override actor.model.num_action_chunks / openpi.action_chunk
  --micro-batch COUNT        Actor micro batch size (default: 4)
  --global-batch COUNT       Actor global batch size (default: 16)
  --replay-capacity COUNT    Replay buffer capacity (default: 512)
  --min-buffer-size COUNT    Minimum replay size before update (default: 256)
  --train-actor-steps COUNT  Minimum demo/replay size before actor updates (default: 256)
  --update-epoch COUNT       SAC update_epoch override (default: 1)
  --offline-demo-only        Train only from the demo manifest (default)
  --online-rollout           Use RLinf live env/rollout path instead of demo-only
  --resume-dir PATH          Optional RLinf resume_dir
  --dry-run                  Print the resolved command without running it
  -h, --help                 Show this message
EOF
}

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
data_path=""
data_type="robot_demo"
algorithm_loss_type=""
experiment_name="stage0_posttrain_demo"
log_dir="${CAVER_REPO_ROOT}/logs/runtime/pistepnft_stage0_posttrain_demo"
task_suite="libero_90"
task_ids="6,7,11,16,17,46,47,48,57,58,59,63,73,74,75"
train_envs="1"
eval_envs="1"
max_steps="20"
max_epochs="20"
save_interval="20"
val_check_interval="-1"
rollout_steps="5"
action_chunk=""
micro_batch="4"
global_batch="16"
replay_capacity="512"
min_buffer_size="256"
train_actor_steps="256"
update_epoch="1"
offline_demo_only=1
resume_dir=""
dry_run=0

while (($# > 0)); do
  case "${1}" in
    --config-name) config_name="${2:?missing value for --config-name}"; shift 2 ;;
    --model-path) model_path="${2:?missing value for --model-path}"; shift 2 ;;
    --demo-manifest) data_path="${2:?missing value for --demo-manifest}"; shift 2 ;;
    --data-path) data_path="${2:?missing value for --data-path}"; shift 2 ;;
    --data-type) data_type="${2:?missing value for --data-type}"; shift 2 ;;
    --algorithm-loss-type) algorithm_loss_type="${2:?missing value for --algorithm-loss-type}"; shift 2 ;;
    --experiment-name) experiment_name="${2:?missing value for --experiment-name}"; shift 2 ;;
    --log-dir) log_dir="${2:?missing value for --log-dir}"; shift 2 ;;
    --task-suite) task_suite="${2:?missing value for --task-suite}"; shift 2 ;;
    --task-ids) task_ids="${2:?missing value for --task-ids}"; shift 2 ;;
    --train-envs) train_envs="${2:?missing value for --train-envs}"; shift 2 ;;
    --eval-envs) eval_envs="${2:?missing value for --eval-envs}"; shift 2 ;;
    --max-steps) max_steps="${2:?missing value for --max-steps}"; shift 2 ;;
    --max-epochs) max_epochs="${2:?missing value for --max-epochs}"; shift 2 ;;
    --save-interval) save_interval="${2:?missing value for --save-interval}"; shift 2 ;;
    --val-check-interval) val_check_interval="${2:?missing value for --val-check-interval}"; shift 2 ;;
    --rollout-steps) rollout_steps="${2:?missing value for --rollout-steps}"; shift 2 ;;
    --action-chunk) action_chunk="${2:?missing value for --action-chunk}"; shift 2 ;;
    --micro-batch) micro_batch="${2:?missing value for --micro-batch}"; shift 2 ;;
    --global-batch) global_batch="${2:?missing value for --global-batch}"; shift 2 ;;
    --replay-capacity) replay_capacity="${2:?missing value for --replay-capacity}"; shift 2 ;;
    --min-buffer-size) min_buffer_size="${2:?missing value for --min-buffer-size}"; shift 2 ;;
    --train-actor-steps) train_actor_steps="${2:?missing value for --train-actor-steps}"; shift 2 ;;
    --update-epoch) update_epoch="${2:?missing value for --update-epoch}"; shift 2 ;;
    --offline-demo-only) offline_demo_only=1; shift ;;
    --online-rollout) offline_demo_only=0; shift ;;
    --resume-dir) resume_dir="${2:?missing value for --resume-dir}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: ${1}" >&2; usage >&2; exit 1 ;;
  esac
done

require_command python3

if [ -z "${data_path}" ]; then
  echo "error: --data-path is required" >&2
  exit 1
fi
case "${data_type}" in
  robot_demo|embodied_rollout_batch) ;;
  *)
    echo "error: unsupported --data-type ${data_type}; expected robot_demo or embodied_rollout_batch" >&2
    exit 1
    ;;
esac

model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${model_path}")"
data_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${data_path}")"
log_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${log_dir}")"
config_path="${CAVER_DEFAULT_SOURCE_ROOT}/pi-StepNFT/examples/embodiment/config/${config_name}.yaml"

if [ ! -f "${data_path}" ]; then
  echo "error: data path not found: ${data_path}" >&2
  exit 1
fi
if [ ! -f "${model_path}/model.safetensors" ]; then
  echo "error: converted OpenPI checkpoint missing model.safetensors under ${model_path}" >&2
  exit 1
fi
if [ ! -f "${config_path}" ]; then
  echo "error: RLinf config not found: ${config_path}" >&2
  exit 1
fi

config_num_action_chunks="$(python3 - "${config_path}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r"^\s*num_action_chunks:\s*(\d+)\s*$", text, flags=re.MULTILINE)
print(match.group(1) if match else "")
PY
)"
effective_action_chunk="${action_chunk:-${config_num_action_chunks}}"
if [ -n "${effective_action_chunk}" ] && [ $(( rollout_steps % effective_action_chunk )) -ne 0 ]; then
  echo "error: rollout-steps ${rollout_steps} must be divisible by actor.model.num_action_chunks=${effective_action_chunk} for ${config_name}" >&2
  exit 1
fi

ensure_directory "${log_dir}"
training_completed_marker="${log_dir}/training_completed.marker"
training_completed_metadata="${log_dir}/training_completed.json"
task_ids_override="[${task_ids}]"
offline_demo_only_override="false"
allow_demo_only_training_override="false"
export CAVER_RLINF_FORCE_LOCAL_RAY="${CAVER_RLINF_FORCE_LOCAL_RAY:-1}"
if ((offline_demo_only)); then
  offline_demo_only_override="true"
  allow_demo_only_training_override="true"
fi
resolved_algorithm_loss_type="${algorithm_loss_type}"
if [ -z "${resolved_algorithm_loss_type}" ] && [ "${data_type}" = "robot_demo" ]; then
  resolved_algorithm_loss_type="embodied_sac"
fi

cmd=(
  "${CAVER_REPO_ROOT}/scripts/env/with_openpi_pistepnft_libero_train.sh"
  --
  python
  "${CAVER_DEFAULT_SOURCE_ROOT}/pi-StepNFT/examples/embodiment/train_embodied_agent.py"
  --config-path "${CAVER_DEFAULT_SOURCE_ROOT}/pi-StepNFT/examples/embodiment/config"
  --config-name "${config_name}"
  "runner.logger.log_path=${log_dir}"
  "runner.logger.project_name=caver"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.max_steps=${max_steps}"
  "runner.max_epochs=${max_epochs}"
  "runner.val_check_interval=${val_check_interval}"
  "runner.save_interval=${save_interval}"
  "+runner.offline_demo_only=${offline_demo_only_override}"
  "algorithm.update_epoch=${update_epoch}"
  "+algorithm.allow_demo_only_training=${allow_demo_only_training_override}"
  "actor.global_batch_size=${global_batch}"
  "actor.micro_batch_size=${micro_batch}"
  "actor.model.model_path=${model_path}"
  "actor.model.num_action_chunks=${effective_action_chunk}"
  "actor.model.openpi.action_chunk=${effective_action_chunk}"
  "actor.model.openpi.solver_type=flow_sde"
  "+actor.model.openpi.pytorch_compile_mode=null"
  "rollout.model.model_path=${model_path}"
  "env.train.task_suite_name=${task_suite}"
  "env.eval.task_suite_name=${task_suite}"
  "env.train.task_ids=${task_ids_override}"
  "env.eval.task_ids=${task_ids_override}"
  "env.train.total_num_envs=${train_envs}"
  "env.eval.total_num_envs=${eval_envs}"
  "env.train.max_steps_per_rollout_epoch=${rollout_steps}"
  "env.eval.max_steps_per_rollout_epoch=${rollout_steps}"
  "env.train.max_episode_steps=${rollout_steps}"
  "env.eval.max_episode_steps=${rollout_steps}"
  "+data.type=${data_type}"
  "+data.path=${data_path}"
)

if [ -n "${resolved_algorithm_loss_type}" ]; then
  cmd+=("algorithm.loss_type=${resolved_algorithm_loss_type}")
fi

if [ "${data_type}" = "robot_demo" ]; then
  cmd+=(
    "algorithm.rollout_epoch=1"
    "algorithm.eval_rollout_epoch=0"
    "+algorithm.replay_buffer_capacity=${replay_capacity}"
    "+algorithm.min_buffer_size=${min_buffer_size}"
    "+algorithm.train_actor_steps=${train_actor_steps}"
    "+algorithm.tau=0.005"
    "+algorithm.initial_alpha=0.01"
    "+algorithm.alpha_lr=3e-4"
    "+algorithm.auto_entropy_tuning=true"
    "+algorithm.target_update_freq=1"
    "+algorithm.critic_actor_ratio=1"
    "+algorithm.backup_entropy=true"
    "+algorithm.agg_q=min"
    "+algorithm.bootstrap_type=standard"
    "actor.model.add_value_head=true"
    "+actor.model.openpi.use_nft_loss=false"
  )
fi

if [ -n "${resume_dir}" ]; then
  cmd+=("runner.resume_dir=${resume_dir}")
fi

printf 'train command:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'artifacts:\n'
printf '  log_dir: %s\n' "${log_dir}"
printf '  training_completed_marker: %s\n' "${training_completed_marker}"
printf '  experiment_name: %s\n' "${experiment_name}"
printf '  data_path: %s\n' "${data_path}"
printf '  data_type: %s\n' "${data_type}"
printf '  algorithm_loss_type: %s\n' "${resolved_algorithm_loss_type:-<config-default>}"

if ((dry_run)); then
  exit 0
fi

rm -f -- "${training_completed_marker}" "${training_completed_metadata}"
"${cmd[@]}"

python3 - "${training_completed_marker}" "${training_completed_metadata}" "${experiment_name}" "${data_path}" "${data_type}" "${resolved_algorithm_loss_type}" "${model_path}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

marker_path = pathlib.Path(sys.argv[1]).resolve()
metadata_path = pathlib.Path(sys.argv[2]).resolve()
experiment_name = sys.argv[3]
data_path = pathlib.Path(sys.argv[4]).resolve()
data_type = sys.argv[5]
algorithm_loss_type = sys.argv[6]
model_path = pathlib.Path(sys.argv[7]).resolve()

marker_path.parent.mkdir(parents=True, exist_ok=True)
marker_path.write_text("", encoding="utf-8")
payload = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_name": experiment_name,
    "data_path": str(data_path),
    "data_type": data_type,
    "algorithm_loss_type": algorithm_loss_type or None,
    "model_path": str(model_path),
}
metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
