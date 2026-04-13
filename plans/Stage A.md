# Stage A: SDRE Bring-up and Reproducibility Scaffold

## Goal

Translate the proposal's paper assumptions into SDRE-operable conventions before touching the learning stack.

## Why this stage exists

The proposal assumes a generic Ubuntu-plus-Conda workflow. SDRE is Slurm-based, module-driven, and currently does not expose the same packaging and container tools. If this translation is not done first, every later stage becomes harder to debug.

## Scope

- Lock cluster facts that affect every run: account, QoS, partitions, GPU type names, walltimes, and storage paths.
- Decide where active outputs live and where archived outputs live.
- Define the run directory layout, naming scheme, and `manifest.lock` schema.
- Create submission patterns for short `interactive` bring-up jobs and long `normal` experiment jobs.
- Define the environment matrix we will use in later stages.

## Concrete SDRE decisions already made

- Active repo and runs: `/projects/p57098/euijin1/Caver`
- Active logs:
  - `/projects/p57098/euijin1/Caver/logs/slurm`
  - `/projects/p57098/euijin1/Caver/logs/runtime`
- Archived logs and artifacts:
  - `/rdss/p57098/euijin1/caver/logs/slurm`
  - `/rdss/p57098/euijin1/caver/logs/runtime`
  - `/rdss/p57098/euijin1/caver/artifacts`
- Slurm account: `p57098`
- Allowed QoS values: `interactive`, `normal`
- Default QoS: none pinned by partition or user association, so every job should pass `--qos=` explicitly
- Preferred typed GPU requests:
  - `--gres=gpu:l40s:1`
  - `--gres=gpu:h200:1`

## Recommended job patterns

- Short bring-up on L40S:
  - use `--partition=gpu-l40s --qos=interactive --gres=gpu:l40s:1`
- Long single-run training:
  - use `--partition=gpu-l40s --qos=normal --gres=gpu:l40s:1`
- Heavy or memory-hungry training:
  - use `--partition=gpu-h200 --qos=normal --gres=gpu:h200:1`

## Tasks

- Write reusable Slurm launcher templates for:
  - interactive debug
  - batch single-run experiment
  - batch sweep driver
- Define a run ID convention that encodes:
  - stage
  - method
  - task
  - seed
  - budget
  - timestamp
- Define `manifest.lock` fields for:
  - public repo SHAs
  - local config checksums
  - module stack
  - Python version
  - Slurm submission metadata
  - seed values
  - dataset partition membership
- Decide what gets archived automatically to `/rdss` and what stays live in `/projects`.

## Deliverables

- Slurm helper scripts or templates
- directory layout for runs and logs
- initial `manifest.lock` schema
- one SDRE note capturing the final cluster conventions

## Exit criteria

- A user can launch one L40S interactive job and one H200 batch job with the intended account/QoS/GPU flags.
- Logs land in predictable per-run locations.
- The stage-to-run naming scheme is fixed.
- Later stages no longer need to guess where to store outputs or which Slurm flags to use.

## Risks and watchpoints

- `interactive` is capped at `08:00:00`; it is for debugging only.
- `normal` is capped at `7-00:00:00`; long sweeps must be chunked around that cap.
- `/scratch` is not currently usable despite the environment variable.
- One L40S node has only three GPUs, so request counts on that partition should stay conservative.

## Status on 2026-04-01

- Stage A exit criteria are met for the current SDRE path.
- The Slurm helper scripts, manifest schema, run-directory layout, and active-versus-archive storage decisions are all in repo and already exercised by Stage-0 smoke submissions.
- Later stages should treat Stage A as frozen unless SDRE policy or storage layout changes.
