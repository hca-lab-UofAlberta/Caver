#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

PARTITION="${PARTITION:-gpu-l40s}"
QOS="${QOS:-normal}"
GPU_TYPE="${GPU_TYPE:-l40s}"
GPUS="${GPUS:-1}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-128G}"

RUN_ROOT="${RUN_ROOT:-/projects/p57098/euijin1/caver_stagee_k1_guarded_runs}"
LOG_ROOT="${LOG_ROOT:-/projects/p57098/euijin1/caver_stagee_k1_guarded_logs/slurm}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/projects/p57098/euijin1/caver_stagee_ablation_exactpayload_runtime_logs/stagee_manifests}"

VALUE_PROXY_MODEL_PATH="${VALUE_PROXY_MODEL_PATH:-${ROOT_DIR}/metadata/stage0/value_proxy/stage0_context_success_progress_sq_mlp3head_v2.json}"
DR_CALIBRATOR_MODEL_PATH="${DR_CALIBRATOR_MODEL_PATH:-${ROOT_DIR}/metadata/stage0/calibrator/stage0_seed_dr_calibrator_mlp_v2.json}"

SEEDS=("${@:-7 13 29}")
if [[ "${#SEEDS[@]}" -eq 1 && "${SEEDS[0]}" == "7 13 29" ]]; then
  SEEDS=(7 13 29)
fi

for seed in "${SEEDS[@]}"; do
  manifest="${MANIFEST_ROOT}/stagee_curve__success-only__t_train_s0__budget50__offset0__seed${seed}__20260529T110854-0600.json"
  if [[ ! -f "${manifest}" ]]; then
    echo "missing manifest for seed ${seed}: ${manifest}" >&2
    exit 1
  fi

  "${ROOT_DIR}/scripts/slurm/submit_stage0_caver_lagged_budget.sh" \
    --partition "${PARTITION}" \
    --qos "${QOS}" \
    --gpu-type "${GPU_TYPE}" \
    --gpus "${GPUS}" \
    --time "${TIME_LIMIT}" \
    --cpus "${CPUS}" \
    --mem "${MEM}" \
    --run-root "${RUN_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --trace-reference-mode manifest \
    --manifest-path "${manifest}" \
    --partition-name T_train_S0 \
    --max-contexts 50 \
    --round-size 25 \
    --seed "${seed}" \
    --candidate-count 4 \
    --selection-policy caver_k1_guarded \
    --num-steps-wait 10 \
    --replan-steps 4 \
    --resize-size 224 \
    --resolution 256 \
    --libero-gl-backend osmesa \
    --selector-mode k1_guarded_success_only_caver_selector_v1 \
    --admission-policy success_only \
    --value-proxy-model-path "${VALUE_PROXY_MODEL_PATH}" \
    --dr-calibrator-model-path "${DR_CALIBRATOR_MODEL_PATH}" \
    --provider-mode gesim_live_summary \
    --provider-gesim-timeout-sec 900 \
    --server-mode openpi-exact \
    --exact-rollout-payload \
    --exact-solver-type flow_sde \
    --exact-infer-mode train \
    --backend-task-suite libero_90 \
    --backend-task-ids 6,7,11,16,17,46,47,48,57,58,59,63,73,74,75 \
    --experiment-name stage0_k1_guarded_success_only \
    --run-label-suffix k1-guarded-success-only-n50 \
    --finalizer-skip-backend-update
done
