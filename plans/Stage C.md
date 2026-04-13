# Stage C: LIBERO Stage-0 Harness and Data Protocol

## Goal

Create the exact simulation proxy stage that the proposal uses for debugging and controlled comparison.

## What this stage must honor from the proposal

- five LIBERO tasks
- `K=4` candidates per context
- `H=4` executed commands per chunk
- `B_round=25` counted training contexts
- budgets `N in {25, 50, 100, 200}`
- three random seeds per Stage-0 experiment
- disjoint `seed`, `train`, `val`, and `test` template partitions
- one 120-context simulated seed warm-start set, balanced as 24 contexts per task

## The five Stage-0 tasks

- pick block to tray
- place can into bowl
- stack two blocks
- relocate object to a marked goal region
- open a simple drawer or cabinet handle

## Tasks

- Map the proposal's five tasks to concrete LIBERO tasks or nearest reproducible equivalents.
- Define the Stage-0 partition files:
  - `T_seed_S0`
  - `T_train_S0`
  - `T_val_S0`
  - `T_test_S0`
- Implement round accounting so that:
  - one executed context counts as one budget unit
  - one `safety_abort` with no executable candidate also counts as one budget unit
- Implement the shared Stage-0 seed warm-start pass:
  - collect 120 fully verified contexts
  - run one uniform-selector warm-start with 500 backend update steps
  - fit the initial calibrator before round 1
- Lock the logging schema for:
  - context IDs
  - candidate IDs
  - propensities
  - outcomes
  - safety flags
  - per-round checkpoints

## Current progress on SDRE

- Repo-local Stage-0 runners now exist for:
  - native OpenPI server plus LIBERO client split bridge
  - Slurm submission of one-task smoke jobs
- Verified smoke path:
  - job `5329` on `gpu-l40s`
  - task suite `libero_spatial`
  - tasks `[0]`
  - trials per task `1`
  - policy steps `1`
  - renderer backend `osmesa`
  - results written to `/projects/p57098/euijin1/Caver/runs/stage0__openpi-native-smoke__libero_spatial__seed7__budget0__20260402T014201Z/results/libero_eval.json`
- The same smoke path fails with compute-node `egl`, so Stage C should currently treat `osmesa` as the SDRE-compatible renderer until proven otherwise.
- The Stage-0 semantic task map is now frozen under `/projects/p57098/euijin1/Caver/metadata/stage0/libero_stage0_task_families.json`.
- The first deterministic Stage-0 partition manifest is now generated under `/projects/p57098/euijin1/Caver/metadata/stage0/libero_stage0_partitions.json`.
- A non-obvious LIBERO constraint is now documented and handled:
  - stock concrete tasks expose only `50` init-state templates each on this installation
  - the proposal's disjoint `24` seed plus `20` val plus `20` test split is therefore impossible per single task
  - the SDRE workaround is to define five semantic proxy families, each backed by three concrete `libero_90` tasks
- Current frozen family-level counts are:
  - `block_to_tray_proxy`: `24` seed, `20` val, `20` test, `86` train
  - `container_insertion_proxy`: `24` seed, `20` val, `20` test, `86` train
  - `two_object_stack_proxy`: `24` seed, `20` val, `20` test, `86` train
  - `relocate_to_region_proxy`: `24` seed, `20` val, `20` test, `86` train
  - `drawer_open_proxy`: `24` seed, `20` val, `20` test, `86` train
- Global Stage-0 counts now available from the manifest are:
  - `T_seed_S0 = 120`
  - `T_val_S0 = 100`
  - `T_test_S0 = 100`
  - `T_train_S0 = 430`
- Stage-0 family-aware execution now exists:
  - partition runner: `/projects/p57098/euijin1/Caver/scripts/stage0/run_stage0_partition_eval.sh`
  - warm-start wrapper: `/projects/p57098/euijin1/Caver/scripts/stage0/collect_stage0_warm_start.sh`
  - manifest-aware evaluator path: `/projects/p57098/euijin1/Caver/scripts/bridge/libero_remote_eval.py`
- The evaluator now emits a per-context budget ledger with:
  - `budget_domain`
  - `partition_name`
  - `counts_against_online_budget`
  - `online_budget_units`
  - `round_index`
  - `round_context_index`
  - `budget_reason`
- Current limitation of that ledger:
  - `safety_abort=false` on the current split-bridge path because the Stage-D safety mask has not been inserted yet
  - the schema is ready for `safety_abort` accounting once the selector and shield are live
- The new runner has been exercised locally with the dummy websocket server:
  - manifest-mode smoke artifact: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_partition_dummy.json`
  - manifest-mode context ledger: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_partition_dummy.jsonl`
  - warm-start wrapper smoke artifact: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_seed_wrapper_dummy.json`
  - warm-start wrapper ledger: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_seed_wrapper_dummy.jsonl`
  - legacy regression artifact after the same evaluator changes: `/projects/p57098/euijin1/Caver/logs/runtime/libero_legacy_dummy.json`
- Full seed-collection submission is now live on SDRE:
  - submission wrapper: `/projects/p57098/euijin1/Caver/scripts/slurm/submit_stage0_seed_warm_start.sh`
  - first corrected live job: `5332`
  - node: `l40s-01`
  - run dir: `/projects/p57098/euijin1/Caver/runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T033646Z`
  - current observed progress in Slurm log: `starting LIBERO evaluation: mode=manifest contexts=120` and `context 1/120`
- One launch-time bug was resolved during this step:
  - failed job `5331` showed that `collect_stage0_warm_start.sh` lacked `--manifest-path`
  - the wrapper was patched and resubmitted successfully as job `5332`
- Stage-0 trace capture now exists for actual warm-start conversion:
  - evaluator trace flag: `/projects/p57098/euijin1/Caver/scripts/bridge/libero_remote_eval.py` via `--transition-trace-path`
  - partition wrapper pass-through: `/projects/p57098/euijin1/Caver/scripts/stage0/run_stage0_partition_eval.sh`
  - seed wrapper pass-through: `/projects/p57098/euijin1/Caver/scripts/stage0/collect_stage0_warm_start.sh`
- Replay-buffer conversion now exists and is smoke-validated:
  - converter: `/projects/p57098/euijin1/Caver/scripts/stage0/convert_stage0_trace_to_rlinf_demo.py`
  - wrapper: `/projects/p57098/euijin1/Caver/scripts/stage0/convert_stage0_trace_to_rlinf_demo.sh`
  - trace smoke artifact: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_trace_smoke_chunks.jsonl`
  - chunk-step demo artifact: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_trace_smoke_demo.pt`
  - chunk-step demo summary: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_trace_smoke_demo.summary.json`
  - primitive-step fallback demo artifact: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_trace_smoke_primitive_demo.pt`
- The current SDRE-compatible target is now explicit:
  - the public `pi-StepNFT` `robot_demo` loader feeds `SACReplayBuffer`
  - the public NFT actor path still lacks a native offline demo ingest hook
  - current Stage-0 conversion therefore targets SAC-compatible replay items rather than direct NFT warm-start tuples
- Validation status of the converter path:
  - chunk-trace smoke run completed locally with `--dummy-server`, `--max-contexts 1`, `--max-steps 2`
  - the chunk-step artifact loads structurally into `SACReplayBuffer.create_from_demo(...)`
  - direct import of `rlinf.data.replay_buffer` still pulls `ray` transitively in the current `pi-StepNFT` venv, so the loader check needed a local scheduler stub during validation
- The trace-authoritative rerun is now also live on SDRE:
  - job id: `5333`
  - node: `l40s-01`
  - run dir: `/projects/p57098/euijin1/Caver/runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T043351Z`
  - Slurm log now shows: `starting LIBERO evaluation: mode=manifest contexts=120` and `context 1/120`
  - the live trace file already exists in the run directory:
    `/projects/p57098/euijin1/Caver/runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T043351Z/results/stage0_seed_warm_start_chunks.jsonl`
- Wrapper-default trace placement is now fixed for future runs:
  - `collect_stage0_warm_start.sh` now defaults to `${CAVER_RUN_DIR}/results/stage0_seed_warm_start_chunks.jsonl`
  - `run_stage0_partition_eval.sh` now defaults to `${CAVER_RUN_DIR}/results/stage0_${partition_name}_chunks.jsonl`
- The `5333` rerun is now diagnosed as invalid for final warm-start use:
  - it completed with exit code `0`, but its server log shows `OSError: [Errno 98] ... address already in use`
  - because `5332` was still running on the same node and also using port `8000`, `5333`'s client attached to `5332`'s server instead of its own
  - once `5332` finished, `5333` degraded into `ConnectionClosedError` for the remaining contexts
  - summary fallout: `7/120` successes, `103` error contexts, and traces covering only `18` contexts
- Port-collision prevention is now fixed in the runtime wrappers:
  - `scripts/common.sh` provides a Slurm-derived loopback port helper
  - `scripts/stage0/run_stage0_partition_eval.sh` and `scripts/bridge/run_libero_remote_eval.sh` now default to that derived port instead of always using `8000`
- The clean trace-authoritative rerun is now complete:
  - job id: `5334`
  - run dir: `/projects/p57098/euijin1/Caver/runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T053122Z`
  - derived port confirmed in the Slurm log: `127.0.0.1:23334`
  - final state on `2026-04-02`: `COMPLETED`, exit code `0`, elapsed `05:36:10`
  - authoritative summary: `/projects/p57098/euijin1/Caver/runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T053122Z/results/stage0_seed_warm_start.json`
  - authoritative trace file: `/projects/p57098/euijin1/Caver/runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T053122Z/results/stage0_seed_warm_start_chunks.jsonl`
  - final counts: `120/120` contexts, `28` successes, `0` errors, and `8456` chunk traces across all `120` contexts
- The full `5334` trace made the converter scaling requirement explicit:
  - trace size: `71G`
  - the original monolithic conversion path does not fit comfortably in login-node memory at this scale
- The conversion path is now upgraded for full Stage-0 traces:
  - `convert_stage0_trace_to_rlinf_demo.py` supports `--output-mode sharded_manifest`
  - `rlinf.data.replay_buffer.SACReplayBuffer.create_from_demo(...)` now accepts the emitted sharded manifest and ingests shards sequentially
  - sharded smoke artifact from the real `5334` trace: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_seed_warm_start_demo_smoke.manifest.json`
  - sharded smoke summary: `/projects/p57098/euijin1/Caver/logs/runtime/stage0_seed_warm_start_demo_smoke.summary.json`
  - the sharded smoke manifest loads into `SACReplayBuffer` successfully with the same local scheduler stub used for earlier replay-buffer validation
- The first full sharded conversion is now complete on Slurm:
  - submission wrapper: `/projects/p57098/euijin1/Caver/scripts/slurm/submit_stage0_seed_warm_start_conversion.sh`
  - job id: `5392`
  - node: `cpu-01`
  - run dir: `/projects/p57098/euijin1/Caver/runs/stage0__seed-demo-convert__libero-stage0-seed__seed7__budget0__20260402T203345Z`
  - output manifest target: `/projects/p57098/euijin1/Caver/runs/stage0__seed-demo-convert__libero-stage0-seed__seed7__budget0__20260402T203345Z/results/stage0_seed_warm_start_demo.manifest.json`
  - summary target: `/projects/p57098/euijin1/Caver/runs/stage0__seed-demo-convert__libero-stage0-seed__seed7__budget0__20260402T203345Z/results/stage0_seed_warm_start_demo.summary.json`
  - final state on `2026-04-02`: `COMPLETED`, exit code `0`, elapsed `01:31:44`
  - final artifact shape: `8456` demo items, `120` contexts covered, `67` shards, `max_items_per_shard=128`
- What is still missing from Stage C is now narrower:
  - final confirmation that the emitted full manifest is the artifact we will point `pi-StepNFT` at for warm-start SAC replay loading

## Deliverables

- LIBERO task mapping file
  - `/projects/p57098/euijin1/Caver/metadata/stage0/libero_stage0_task_families.json`
- Stage-0 partition manifests
  - `/projects/p57098/euijin1/Caver/metadata/stage0/libero_stage0_partitions.json`
- seed warm-start dataset
- Stage-0 environment wrapper with proposal-consistent budget counting
- one end-to-end uniform-selector smoke run on a single task

## Exit criteria

- We can run a single Stage-0 task end to end with the proposal's round and budget semantics.
- The 120-context simulated seed workflow is reproducible.
- Validation and audit test data are disjoint from training data.
- All later methods can share the same harness without rewriting the accounting rules.

## Risks and watchpoints

- Exact proposal tasks may not have one-to-one LIBERO names; if a near-equivalent task is used, the mapping must be documented and frozen.
- Logging discipline matters here because later DR correction and calibration comparisons depend on it.
