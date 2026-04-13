# Stage B: Pinned Software Stack Bring-up on SDRE

## Goal

Establish working SDRE-native environments for the components we actually need, without assuming Conda or Docker.

## Core adaptation

The proposal's install snippets use `conda`, `uv`, and Docker. On SDRE, the current visible path is:

- modules
- `python -m venv`
- pinned repo checkouts

That should be treated as the default unless a sanctioned cluster alternative appears later.

## Environment matrix

### `env-libero`

- Target use: LIBERO Stage-0 harness
- Candidate stack: `StdEnv/2020 + python/3.8.10`
- Rationale: closest match to the proposal's LIBERO-era Python expectation

### `env-caver-train`

- Target use: `openpi`, CAVER code, `pi-StepNFT`
- Candidate stack: `StdEnv/2023 + gcc/12.3 + cuda/12.6 + cudnn/9.10.0.56 + python/3.11.5`
- Status: verified to load cleanly; `venv` creation works; `openpi` requires Python `>=3.11`

### `env-caver-py310`

- Target use: compatibility fallback
- Candidate stack: `StdEnv/2023 + gcc/12.3 + cuda/12.6 + cudnn/9.10.0.56 + python/3.10.13`
- Status: verified to load cleanly

### `env-gesim`

- Target use: GE-Sim inference
- Initial guess: `StdEnv/2023 + gcc/12.3 + cuda/12.6 + cudnn/9.10.0.56 + python/3.10.13`
- Decision rule: only fork this environment if GE-Sim forces a version mismatch

### `env-rlinf`

- Target use: WoVR baseline
- Status: unresolved
- Blocker: proposal assumes Docker, but no Docker or Apptainer path is visible in this shell

## Pinned upstream components from the proposal

- `openpi`
- `Genie-Envisioner / GE-Sim`
- `pi-StepNFT`
- `RLinf / WoVR`
- `LIBERO`

## Current status on SDRE

- `openpi` imports cleanly in the Python 3.11 training environment.
- `LIBERO` imports and `OffScreenRenderEnv` creation work in the Python 3.8 environment.
- `pi-StepNFT` base RLinf imports work, and the OpenPI-adjacent imports are sufficient for websocket policy serving from the training environment.
- A single mixed `openpi + RLinf + LIBERO` interpreter still segfaults at simulator startup on SDRE.
- The stable Stage-0 execution path is therefore split:
  - policy server in `env-caver-train`
  - LIBERO simulator client in `env-libero`
  - websocket transport between them
- Public OpenPI checkpoints now cache under `/projects/p57098/euijin1/Caver/third_party/openpi-cache`, and the SDRE fallback path for `gs://openpi-assets/...` works without `gsutil`.
- Native OpenPI serving with `pi05_libero` was killed during restore on the login node, but restores and serves correctly on `gpu-l40s`.
- LIBERO rendering on `gpu-l40s` is currently viable with `MUJOCO_GL=osmesa`; the `egl` path still fails on these nodes.
- Stage B is effectively complete for the Stage-0 split-bridge path:
  - `openpi` policy serving works from `env-caver-train`
  - `LIBERO` simulation works from `env-libero`
  - `pi-StepNFT` imports are sufficient for the shared backend path and OpenPI-adjacent code
- The main unresolved Stage-B-adjacent items are no longer base bring-up blockers:
  - GE-Sim integration remains deferred work for Stage D/E
  - WoVR/RLinf remains conditional on a viable non-Docker SDRE deployment path

## Tasks

- Clone the pinned public repositories under a controlled `third_party` location.
- Record exact SHAs in `manifest.lock`.
- Build and smoke-test:
  - `openpi`
  - `LIBERO`
  - `pi-StepNFT`
  - GE-Sim, if dependency friction is acceptable
- Identify which components can share one environment and which require separate ones.
- Determine whether RLinf/WoVR can be built from source on SDRE without Docker.

## Required smoke tests

- `python -c "import torch"` inside the chosen training environment
- `openpi` import and minimal model/config load
- `LIBERO` import and environment creation
- `pi-StepNFT` import and minimal trainer entrypoint load
- GE-Sim inference sanity check on a tiny local sample
- RLinf import or build viability check

## Deliverables

- environment bootstrap scripts
- pinned repo checkout layout
- import-smoke logs per component
- a short note stating whether WoVR is feasible now, deferred, or blocked

## Exit criteria

- We can activate the training environment and the LIBERO environment reliably inside Slurm jobs.
- `openpi`, `LIBERO`, and `pi-StepNFT` all import successfully.
- GE-Sim is either running or explicitly marked as the current blocker.
- We know whether WoVR can join the Stage-0 study on SDRE or must be deferred.

## Risks and watchpoints

- `openpi` may expect tooling that is easier under `uv` than under plain `pip`.
- GE-Sim may force a dependency split from the main training environment.
- RLinf may remain blocked until a non-Docker deployment path is found.
