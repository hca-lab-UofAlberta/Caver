# CAVER

This repository is the SDRE-oriented implementation of the CAVER proposal in `caver_proposal_positioned.tex`.

CAVER is not a new RL optimizer. The method here is a selective-verification layer around a fixed `pi-StepNFT` executed-trajectory backend:

1. sample `K=4` candidate chunks from `pi0.5`
2. score them with cheap features and frozen-provider summaries
3. execute one safe candidate with exact logged propensity
4. convert the trusted outcome into candidate-level DR pseudo-outcomes
5. fit a lagged calibrator
6. admit only confident executed trajectories into the unchanged backend

The proposal is staged:

- Stage 0: simulation proxy stage on LIBERO
- Stage 1: transfer to PiPER

This repository is currently focused on Stage 0. PiPER is not the active path yet.

## Status

As of `2026-04-13`:

- SDRE-native setup for `openpi`, `LIBERO`, `pi-StepNFT`, and `GE-Sim` is in place.
- Real-only and single-round provider-aware CAVER runs complete end to end on `gpu-l40s`.
- The GE-Sim live-provider path is stable with split-GPU routing and RDSS-backed provider bundles.
- The proposal-required round-1 seed calibrator path is now also validated:
  - seed artifact: `metadata/stage0/calibrator/stage0_seed_dr_calibrator_mlp_v2.json`
  - validation smoke: job `5887`
  - result: `COMPLETED`, `1/5` successes, selector mode `lagged_dr_calibrated_softmax_v1`, next-round calibrator fit succeeded
- The remaining method gate before PiPER is the full lagged Stage-E path:
  - provider -> DR dataset -> lagged calibrator -> refreshed next-round selector
- The first explicit lagged validation run is job `5873`.

If you want the detailed chronology, read:

- `plans/overall_plan.md`
- `plans/Stage E.md`

## Repository Layout

Tracked source and documentation:

- `README.md`: this file
- `caver_proposal_positioned.tex`, `caver_proposal_positioned.pdf`, `caver_proposal_positioned.bib`: proposal source and rendered draft
- `scripts/`: SDRE setup, Slurm wrappers, Stage 0 / Stage E runners, summarizers, and helpers
- `metadata/`: manifest schema, Stage 0 task-family definitions, partition manifests
- `plans/`: operational plan and running notes by stage
- `docs/`: bootstrap notes
- `third_party/README.md`: expected local third-party layout

Generated or local-only state is intentionally not tracked:

- `third_party/src/`
- `third_party/venvs/`
- `third_party/openpi-cache/`
- `third_party/model-cache/`
- `logs/`
- `runs/`
- `.tmp/`
- `raytmp/`

That split is important for collaboration. Git should carry code, manifests, plans, and the proposal, not cached models or experiment artifacts.

## SDRE Assumptions

This repository was brought up on the University of Alberta SDRE cluster with:

- account: `p57098`
- partitions: `cpu`, `gpu-l40s`, `gpu-h200`
- explicit QoS required: `normal` or `interactive`
- active code root: `/projects/p57098/euijin1/Caver`
- large artifacts: `/rdss/p57098/euijin1/caver`

Operational defaults are encoded in `scripts/common.sh`.

Important current conventions:

- use `/projects/.../Caver` for code and small local runtime files
- use `/rdss/.../caver` for large Slurm logs, provider bundles, and long-lived experiment outputs
- request `2 x L40S` for GE-Sim live-provider runs so OpenPI and GE-Sim do not contend for one GPU
- pass `--qos` explicitly in Slurm submissions

## Quick Setup

### 1. Clone the repo

```bash
git clone https://github.com/qordmlwls/Caver.git
cd Caver
```

### 2. Bootstrap the environments

These wrappers clone public upstream repos into `third_party/src/` and create venvs under `third_party/venvs/`.

```bash
scripts/env/bootstrap_openpi_env.sh
scripts/env/bootstrap_libero_env.sh
scripts/env/bootstrap_pistepnft_env.sh
scripts/env/bootstrap_gesim_env.sh --profile sdre-infer
```

Dry-run versions are available if you want to inspect first:

```bash
scripts/env/bootstrap_openpi_env.sh --dry-run
scripts/env/bootstrap_libero_env.sh --dry-run
scripts/env/bootstrap_pistepnft_env.sh --dry-run
scripts/env/bootstrap_gesim_env.sh --profile sdre-infer --dry-run
```

The module stacks used by those scripts are:

- `openpi` / `pi-StepNFT`: `StdEnv/2023 + gcc/12.3 + cuda/12.6 + python/3.11.5 + cudnn/9.10.0.56`
- `LIBERO`: `StdEnv/2020 + python/3.8.10`
- `GE-Sim`: `StdEnv/2023 + gcc/12.3 + cuda/12.6 + python/3.10.13 + cudnn/9.10.0.56`

### 3. Use the wrapper shells

These wrappers are the supported way to enter the right interpreter + module stack.

```bash
scripts/env/with_openpi_libero_eval.sh -- bash -l
scripts/env/with_libero_eval.sh -- bash -l
scripts/env/with_openpi_pistepnft_libero_train.sh -- bash -l
scripts/env/with_gesim_infer.sh -- bash -l
```

### 4. OpenPI checkpoint

The current experiment scripts expect the converted PyTorch checkpoint at:

```text
third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch
```

The source JAX checkpoint is expected at:

```text
third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero
```

Once the source checkpoint exists, convert it with:

```bash
scripts/openpi/convert_openpi_checkpoint_to_pytorch.sh \
  --checkpoint-dir /projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero \
  --output-path /projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch
```

Notes:

- the SDRE bring-up path already assumes `OPENPI_DATA_HOME=/projects/p57098/euijin1/Caver/third_party/openpi-cache`
- the cache location is local-only and should not be committed

### 5. GE-Sim assets

GE-Sim requires two asset groups:

- public checkpoint:
  - `third_party/model-cache/gesim/ge_sim_cosmos_v0.1.safetensors`
- gated Cosmos assets:
  - `third_party/model-cache/gesim/Cosmos-Predict2-2B-Video2World/`

The runtime checker is:

```bash
scripts/env/with_gesim_infer.sh -- \
  python scripts/stagee/prepare_gesim_runtime.py --require-ready
```

To download the gated Cosmos assets into the expected cache root, use an authorized Hugging Face token:

```bash
scripts/env/with_gesim_infer.sh -- \
  python scripts/stagee/prepare_gesim_runtime.py \
    --download-cosmos-assets \
    --hf-token-file ~/.config/huggingface_token_nvidia \
    --require-ready
```

Notes:

- `prepare_gesim_runtime.py` can read `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or a token file
- the public `ge_sim_cosmos_v0.1.safetensors` checkpoint must still exist locally at the expected path
- none of these model-cache directories should be committed

## Recommended Experiment Flow

### 1. Smoke tests

Interactive shell:

```bash
scripts/slurm/interactive_l40s.sh --time 01:00:00 -- bash -l
```

OpenPI + LIBERO smoke:

```bash
scripts/bridge/run_stage0_openpi_libero_smoke.sh \
  --task-suite-name libero_spatial \
  --max-tasks 1 \
  --trials 1 \
  --max-steps 1 \
  --replan-steps 1 \
  --libero-gl-backend osmesa
```

### 2. Stage 0 manifest and seed data

The Stage 0 partition manifest used by the current work is:

```text
metadata/stage0/libero_stage0_partitions.json
```

The corresponding semantic task-family spec is:

```text
metadata/stage0/libero_stage0_task_families.json
```

If you need to regenerate Stage 0 partitions:

```bash
scripts/stage0/generate_libero_stage0_partitions.sh
```

### 3. Real-only Stage E baseline

Matched-budget real-only baseline:

```bash
scripts/slurm/submit_stage0_real_only_budget.sh \
  --budget 25 \
  --seed 7 \
  --partition gpu-l40s \
  --qos normal \
  --gpu-type l40s
```

### 4. Single-round provider-aware CAVER

The current single-round provider-aware launcher is:

```bash
scripts/slurm/submit_stage0_caver_budget.sh \
  --budget 25 \
  --seed 7 \
  --partition gpu-l40s \
  --qos normal \
  --gpu-type l40s \
  --gpus 2 \
  --time 05:30:00 \
  --num-steps-wait 1 \
  --replan-steps 4 \
  --run-root /rdss/p57098/${USER}/caver/runs \
  --log-root /rdss/p57098/${USER}/caver/logs/slurm \
  --provider-mode gesim_live_summary \
  --provider-bundle-root /rdss/p57098/${USER}/caver/provider_bundles/stagee_seed7_budget25
```

That is the right entrypoint for a clean single-round provider-aware Stage-E run.

If you want the proposal-aligned round-1 seed-calibrator version, add:

```bash
  --value-proxy-model-path metadata/stage0/value_proxy/stage0_context_success_progress_sq_mlp3head_v2.json \
  --dr-calibrator-model-path metadata/stage0/calibrator/stage0_seed_dr_calibrator_mlp_v2.json
```

The reference validation for that path is job `5887`, which completed end to end on `gpu-l40s` with the seed calibrator loaded from round start.

### 5. Explicit lagged Stage-E run

The lagged driver is:

```text
scripts/stagee/run_stage0_caver_lagged_budget.py
```

At the time of writing there is no dedicated Slurm submit wrapper for it yet, so the current pattern is to submit it via `scripts/slurm/submit_experiment.sh`.

The active reference run is job `5873`, which launches the lagged driver with:

- total budget `50`
- round size `25`
- initial calibrator from the completed provider-aware `budget=25` run
- live GE-Sim provider mode
- merged finalization after the two subrounds

If you need to reproduce that path, use the same structure as the live `5873` job script under:

```text
/rdss/p57098/euijin1/caver/runs/stagee__caver-lagged__manifest-t_train_s0-all__seed7__budget50__20260413T092720Z/job.sbatch
```

### 6. Summaries and plots

Grid summarization:

```bash
python scripts/stagee/summarize_stagee_grid.py
```

Sample-efficiency plot:

```bash
python scripts/stagee/plot_stagee_sample_efficiency.py
```

## Files Worth Reading First

If someone new joins the project, this is the fastest path through the repo:

1. `README.md`
2. `caver_proposal_positioned.tex`
3. `plans/overall_plan.md`
4. `plans/Stage E.md`
5. `scripts/stagee/caver_heuristic.py`
6. `scripts/stagee/build_stagee_dr_dataset.py`
7. `scripts/stagee/fit_stagee_dr_calibrator.py`
8. `scripts/stagee/run_stage0_caver_round.sh`
9. `scripts/stagee/run_stage0_caver_lagged_budget.py`

Proposal-side neural replacements for the old linear surrogate path:

- `scripts/stagee/run_train_stage0_value_proxy_mlp.sh`
- `scripts/stagee/run_fit_stagee_dr_calibrator_mlp.sh`
- `scripts/stagee/train_stage0_value_proxy_mlp.py`
- `scripts/stagee/fit_stagee_dr_calibrator_mlp.py`
- `scripts/stagee/tiny_mlp_artifact.py`

Those scripts fit the width-256 GELU proxy/calibrator described in the proposal, but save plain JSON artifacts so runtime selection still works without importing `torch`.

## Collaboration and Git

This repo should be pushed with code and manifests only.

Do commit:

- `scripts/`
- `metadata/`
- `plans/`
- `docs/`
- `README.md`
- `caver_proposal_positioned.tex`
- `caver_proposal_positioned.bib`
- `caver_proposal_positioned.pdf`
- `third_party/README.md`

Do not commit:

- `runs/`
- `logs/`
- `third_party/src/`
- `third_party/venvs/`
- `third_party/openpi-cache/`
- `third_party/model-cache/`
- any Hugging Face token files

Recommended workflow:

```bash
git checkout -b <your-branch>
git status --short
git add README.md .gitignore docs metadata plans scripts tools third_party/README.md caver_proposal_positioned.tex caver_proposal_positioned.bib caver_proposal_positioned.pdf
git commit -m "Initial SDRE CAVER scaffold"
git push -u origin <your-branch>
```

If you want to publish directly from `main` instead:

```bash
git add README.md .gitignore docs metadata plans scripts tools third_party/README.md caver_proposal_positioned.tex caver_proposal_positioned.bib caver_proposal_positioned.pdf
git commit -m "Initial SDRE CAVER scaffold"
git push -u origin main
```

Before pushing, always check that `git status --short` does not include:

- experiment outputs
- model caches
- venvs
- third-party source checkouts

## Known Gaps

- Stage 1 PiPER execution is not complete yet.
- WoVR / RLinf baseline support is still conditional on the SDRE deployment path.
- A dedicated Slurm submit wrapper for `run_stage0_caver_lagged_budget.py` would still be useful.
- The decisive Stage-E question is still whether the lagged refresh path produces nonzero admissions and better sample efficiency than the baselines under matched budgets.
