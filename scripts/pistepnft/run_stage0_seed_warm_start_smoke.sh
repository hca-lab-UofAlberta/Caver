#!/usr/bin/env bash
set -euo pipefail

_CAVER_PISTEPNFT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${_CAVER_PISTEPNFT_SCRIPT_DIR}/../common.sh"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_seed_warm_start_smoke.sh [options]

Options:
  --config-name NAME         RLinf config name (default: libero_goal_ppo_openpi_pi05)
  --model-path PATH          Converted OpenPI PyTorch checkpoint directory
  --demo-manifest PATH       Stage-0 warm-start demo manifest (default: 10-item smoke manifest)
  --experiment-name NAME     Logger experiment name (default: stage0_seed_warm_start_smoke)
  --log-dir PATH             Training log directory
  --task-suite NAME          LIBERO suite name (default: libero_goal)
  --task-ids IDS             Comma-separated task ids for train/eval (default: 0)
  --train-envs COUNT         Train env count (default: 4)
  --eval-envs COUNT          Eval env count (default: 4)
  --max-steps COUNT          Runner max_steps (default: 1)
  --max-epochs COUNT         Runner max_epochs cap (default: 1)
  --rollout-steps COUNT      Train/eval max rollout steps (default: 16)
  --action-chunk COUNT       Override actor.model.num_action_chunks / openpi.action_chunk
  --micro-batch COUNT        Actor micro batch size (default: 2)
  --global-batch COUNT       Actor global batch size (default: 4)
  --replay-capacity COUNT    Replay buffer capacity (default: 512)
  --min-buffer-size COUNT    Minimum replay size before update (default: 1)
  --train-actor-steps COUNT  Minimum replay size before actor updates (default: 1)
  --online-rollout           Use RLinf's live rollout/env path instead of the demo-only warm-start smoke
  --dry-run                  Print the resolved command without running it
  -h, --help                 Show this message
EOF
}

config_name="libero_goal_ppo_openpi_pi05"
model_path="/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch"
demo_manifest="/uhome/euijin1/projects/p57098/euijin1/Caver/logs/runtime/stage0_seed_warm_start_demo_smoke.manifest.json"
experiment_name="stage0_seed_warm_start_smoke"
log_dir="${CAVER_REPO_ROOT}/logs/runtime/pistepnft_stage0_seed_warm_start_smoke"
task_suite="libero_goal"
task_ids="0"
train_envs="1"
eval_envs="1"
max_steps="1"
max_epochs="1"
rollout_steps="5"
action_chunk=""
micro_batch="1"
global_batch="2"
replay_capacity="512"
min_buffer_size="1"
train_actor_steps="1"
offline_demo_only=1
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
    --experiment-name)
      experiment_name="${2:?missing value for --experiment-name}"
      shift 2
      ;;
    --log-dir)
      log_dir="${2:?missing value for --log-dir}"
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
    --action-chunk)
      action_chunk="${2:?missing value for --action-chunk}"
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
    --online-rollout)
      offline_demo_only=0
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

model_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${model_path}")"
demo_manifest="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${demo_manifest}")"
log_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${log_dir}")"
config_path="${CAVER_DEFAULT_SOURCE_ROOT}/pi-StepNFT/examples/embodiment/config/${config_name}.yaml"

if [ ! -f "${demo_manifest}" ]; then
  echo "error: demo manifest not found: ${demo_manifest}" >&2
  exit 1
fi

if ((!dry_run)) && [ ! -f "${model_path}/model.safetensors" ]; then
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
  "runner.val_check_interval=-1"
  "runner.save_interval=-1"
  "+runner.offline_demo_only=${offline_demo_only_override}"
  "algorithm.loss_type=embodied_sac"
  "algorithm.update_epoch=1"
  "algorithm.rollout_epoch=1"
  "algorithm.eval_rollout_epoch=0"
  "+algorithm.allow_demo_only_training=${allow_demo_only_training_override}"
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
  "actor.global_batch_size=${global_batch}"
  "actor.micro_batch_size=${micro_batch}"
  "actor.model.model_path=${model_path}"
  "actor.model.add_value_head=true"
  "actor.model.num_action_chunks=${effective_action_chunk}"
  "actor.model.openpi.action_chunk=${effective_action_chunk}"
  "+actor.model.openpi.use_nft_loss=false"
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
  "env.train.video_cfg.save_video=false"
  "env.eval.video_cfg.save_video=false"
  "+data.type=robot_demo"
  "+data.path=${demo_manifest}"
)

printf 'resolved command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if ((dry_run)); then
  exit 0
fi

export PYTHONUNBUFFERED=1
export CAVER_STAGE_DEBUG=1
rm -f -- "${training_completed_marker}" "${training_completed_metadata}"
"${cmd[@]:0:2}" python -u "${cmd[@]:3}"
python3 - "${training_completed_marker}" "${training_completed_metadata}" "${experiment_name}" "${demo_manifest}" "${model_path}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

marker_path = pathlib.Path(sys.argv[1]).resolve()
metadata_path = pathlib.Path(sys.argv[2]).resolve()
payload = {
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "experiment_name": sys.argv[3],
    "demo_manifest": str(pathlib.Path(sys.argv[4]).resolve()),
    "model_path": str(pathlib.Path(sys.argv[5]).resolve()),
}
marker_path.touch()
metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
