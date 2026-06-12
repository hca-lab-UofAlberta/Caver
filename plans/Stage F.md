# Stage F: PiPER Transfer Readiness and Real-Interface Bring-up

## Goal

Port the Stage-E-stabilized method to the real PiPER interfaces without changing the scientific core of CAVER.

## What changes here

Only robot-specific and provider-specific interface modules should change:

- execution adapter
- provider adapter
- safety shield
- verified labeler
- camera and calibration configs

Selector logic, DR correction, calibrator fitting, and the `pi-StepNFT` backend should remain structurally the same.

## PiPER-specific work items

- implement the `H x 7` PiPER execution chunk semantics
- implement the `H x 16` PiPER-to-GE-Sim compatibility writer
- implement the URDF-based hard safety shield and runtime abort logging
- define the camera bundle and calibration files
- implement the marker-based geometric labeler
- reproduce the proposal's week-1 interface sanity test

## Seed real data requirement

Collect the shared 120-context real seed set only after F0 readiness and the F1 one-task pilot pass:

- 40 contexts for block-to-tray
- 40 contexts for can-to-bowl
- 40 contexts for two-block stack

Allowed uses:

- common `pi-StepNFT` warm-start for all methods
- proxy-head and initial-calibrator fit for CAVER
- normalization/stat initialization for WoVR only if the same seed set is reused

## Deliverables

- PiPER interface modules
- safety-reason logging codes
- camera calibration config
- seed real dataset plan
- successful interface sanity-test log

## Exit criteria

- PiPER action chunks, safety checks, and GE-Sim conditioning files are consistent.
- The seed real collection plan is frozen.
- The real stack can execute the proposal's interface test before any Stage-1 budgeted run begins.

## Risks and watchpoints

- This stage depends on hardware access outside pure SDRE compute-node work.
- Real-robot safety logic must be auditable before any budgeted learning run begins.
- If GE-Sim integration was only partially stable in Stage E, this stage becomes higher risk and should stay conservative.

## Current gating note

- updated on `2026-04-12`:
  - Stage F stays planned, but it is not the primary active implementation stage right now
  - gating condition before PiPER becomes the main line again:
    - Stage E must first land and rerun the proposal-critical method pieces:
      - candidate-level DR pseudo-outcomes
      - lagged calibrator fitting
      - selector refresh from the lagged calibrator
  - reason:
    - Stage F should transfer a stabilized scientific core, not become the place where the missing Stage-E method logic is first debugged

## Updated gate on 2026-04-18

- Stage F is now allowed to proceed, but only in a gated form:
  - `F0` readiness and shadow-mode validation
  - `F1` small PiPER pilot
  - not the full Stage-1 study yet
- reason:
  - the strict Stage-E mainline result is strong enough to justify hardware integration work
  - but the current CAVER admitted-training signal is still concentrated in `container_insertion_proxy` at `N=50`
  - that concentration means the simulation result is not yet equally reassuring for `block-to-tray` and `two-block stack`
- the parallel SDRE follow-on pack is now actively running in parallel:
  - launch log: `logs/runtime/stagee_parallel_followon_pack_20260418T015945-0600.md`
  - this is the active evidence-producing queue that should inform whether Stage F stops at readiness / pilot or expands to the full Stage-1 study

## Stage F execution shape

### F0: readiness only

- PiPER shadow mode with no learned execution
- GE-Sim latency measurement on real observation bundles
- safety-shield scripted and invalid-chunk validation
- marker / geometric labeler validation against manual inspection
- camera and calibration bring-up

### F1: small pilot only

- one-task pilot before the shared real seed set is consumed
- recommended ordering:
  - `block-to-tray` for control and safety bring-up
  - `can-to-bowl` as the first method pilot because the current Stage-E admissions concentrate there
- target shape:
  - one seed
  - `N=25`
  - small validation set
  - no significance claim

### F2: full Stage-1 study

- only after F0 and F1 are stable
- only after the parallel SDRE follow-on pack from Stage E has produced:
  - per-task / per-family diagnostics
  - budget-curve evidence at `N in {25,50,100}`
  - minimal attribution on selection-only and admission-only
  - a clean plan for the no-DR ablation

## Seed real data freeze rule

- Do not consume the shared `120`-context real seed set until F0 passes.
- The shared seed set should be collected only after:
  - shadow mode works
  - safety reason logging is auditable
  - the labeler is trusted
  - the current Stage-E admission concentration has been reviewed through the parallel SDRE diagnostics

## Pre-hardware audit on 2026-04-18

- Goal:
  - check everything that can be checked on SDRE before the physical PiPER stack exists
  - separate software-ready pieces from hardware-blocked pieces

- Verified working on SDRE:
  - `scripts/env/with_openpi_libero_eval.sh` imports `openpi` and `toolkits.eval_scripts_openpi`
  - `scripts/env/with_openpi_pistepnft_libero_train.sh` resolves `rlinf`, `openpi`, and `hydra`
  - `scripts/env/with_gesim_infer.sh -- python scripts/stagee/prepare_gesim_runtime.py` reports `runtime_ready=true`
  - the repo bridge path works in dummy mode:
    - `scripts/bridge/openpi_policy_server.py --dummy`
    - `scripts/bridge/openpi_client_smoke.py`
    - returned `actions` with shape `(4, 7)` and metadata over websockets
  - the synthetic loopback latency probe works against the dummy websocket path:
    - `scripts/bridge/measure_policy_latency.py`
    - baseline on the current SDRE loopback path:
      - client p50 about `0.90 ms`
      - client p95 about `1.35 ms`
      - client p99 about `2.20 ms`
    - interpretation:
      - transport overhead is small in dummy mode
      - this is only a bridge baseline, not a real model-service latency claim
  - the exact rollout-payload service path now passes on `gpu-h200`:
    - job `6105`
    - node `h200-01`
    - wall time about `00:01:28`
    - returned `actions`, `forward_inputs`, `prev_logprobs`, `prev_values`, and `server_timing`

- Verified missing or not yet committed in the repo:
  - no committed `scripts/stage1/...` entrypoints yet
  - no committed `configs/pi05_piper_h4.py`
  - no committed `configs/wovr_piper_h4.yaml`
  - no committed `configs/camera/piper_multiview.yaml`
  - no committed `configs/piper_tasks.yaml`
  - no committed root `manifest.lock`
  - the current manifest schema is still generic Stage-0/cluster metadata, not the full Stage-1 proposal lockfile

- Verified not yet healthy on the current CPU-only SDRE path:
  - `scripts/bridge/run_stage0_openpi_libero_smoke.sh` failed before accepting connections
  - failure symptom:
    - the OpenPI native server process was killed during checkpoint restore on the current path
  - log:
    - `/rdss/p57098/euijin1/caver/runtime_logs/bridge/policy_server__20260418T093741Z__manual__p24064__2538163.log`
  - `scripts/bridge/run_exact_payload_smoke.sh --num-steps 4 --exact-action-chunk 4` did not open its websocket service within `90 s` on the current CPU-only path
  - outcome:
    - this path is not yet validated as a practical pre-hardware shadow-mode service on CPU alone
  - the tiny Stage-0 remote-eval smoke on `gpu-h200` failed under `egl`:
    - job `6106`
    - node `h200-01`
    - `ImportError: Cannot initialize a EGL device display`
  - the same tiny Stage-0 remote-eval smoke on `gpu-h200` passed under `osmesa`:
    - job `6107`
    - node `h200-01`
    - wall time about `00:01:10`
    - wrote results to `/rdss/p57098/euijin1/caver/runtime_logs/stage0_openpi_libero_smoke.json`
  - interpretation:
    - the current H200 bridge-plus-LIBERO smoke looks backend-sensitive rather than fundamentally broken
    - for SDRE-only rehearsals on the current nodes, prefer `osmesa` if `egl` device initialization fails

- Interpretation:
  - GE-Sim assets, repo wrappers, and the bridge skeleton are present
  - the Stage-F-specific software contract is still only partially materialized in code
  - the most likely next no-hardware wins are:
    - commit the missing Stage-1 config and lockfile artifacts
    - scaffold the PiPER execution-adapter / provider-adapter / safety / labeler modules behind local interfaces
    - repeat the exact-payload service smoke on an interactive GPU allocation rather than the current CPU-only path

## Stage-1 scaffold landed on 2026-04-18

- Newly committed local artifacts:
  - `configs/pi05_piper_h4.py`
  - `configs/wovr_piper_h4.yaml`
  - `configs/camera/piper_multiview.yaml`
  - `configs/piper_tasks.yaml`
  - root `manifest.lock` scaffold
  - `metadata/manifest.stage1.schema.json`
  - `scripts/stage1/` local interface modules for execution, provider adaptation, safety, and labeling
  - `scripts/stage1/validate_stage1_scaffold.py`

- Purpose:
  - close the proposal-to-repo gap at the contract level before hardware exists
  - make Stage F auditable in the repository even though the real robot path is not executable yet

- Expected limitation:
  - these files are still scaffold-level and do not replace the eventual hardware integration work
  - the local policy config is a study-local contract, not yet an upstream OpenPI registry entry
  - the joint limits still need to be filled from the actual PiPER ROS / URDF stack before the first real run

- Validator result:
  - `scripts/env/with_openpi_libero_eval.sh -- python scripts/stage1/validate_stage1_scaffold.py`
  - passed on `2026-04-18`

- Next no-hardware runner:
  - `scripts/stage1/shadow_mode_dry_run.py`
  - purpose:
    - synthesize one Stage-1 shadow context
    - optionally query a websocket policy server
    - write `policy_raw_actions.npy`, `exec_actions.npy`, `actions.npy`, and `shadow_context.json`
    - exercise the execution-adapter, safety, and provider-adapter path without commanding hardware
  - checked on `2026-04-18` against the dummy websocket server:
    - unsafe dry run:
      - output dir `/rdss/p57098/euijin1/caver/runtime_logs/stage1_shadow_dry_run_dummy`
      - safety rejected the chunk for `velocity` and `acceleration`
    - safe dry run:
      - output dir `/rdss/p57098/euijin1/caver/runtime_logs/stage1_shadow_dry_run_dummy_safe`
      - wrote `policy_raw_actions.npy`, `exec_actions.npy`, `actions.npy`, and `shadow_context.json`
      - safety passed with no reasons

## Mode B later-path gate

- Mode B remains a later path, not the default `F0/F1` path.
- It is only worth revisiting after:
  - admin confirmation for the SDRE tunnel / service pattern
  - one successful exact-payload or provider-backed service smoke on `gpu-h200`
  - one measured shadow-mode latency rehearsal with no robot motion
  - proof that the local bridge remains safety-authoritative
- Until then:
  - keep Stage F framed around Mode A readiness and a local-authority shadow / pilot path

## Mode B SDRE-side verification on 2026-04-18

- Goal:
  - finish everything that can be verified on SDRE before the physical PiPER stack exists
  - separate SDRE-compute readiness from hardware and policy approvals outside the repo

- Exact policy service rehearsal:
  - job `6108` on `h200-01` proved that the exact rollout-payload service, websocket smoke, and latency probe all worked
  - `6108` failed only in the dry-run artifact writer because nested payload keys like `observation/image` were used directly as file names
  - local fix applied:
    - `scripts/stage1/shadow_mode_dry_run.py` now sanitizes payload-derived artifact names before writing `.npy` files
  - clean rerun:
    - job `6109`
    - partition `gpu-h200`
    - node `h200-01`
    - wall time about `00:01:23`
    - status `COMPLETED`

- What `6109` verified end to end:
  - exact websocket policy service opened successfully on GPU
  - exact payload client smoke passed
  - latency probe completed and wrote:
    - `/rdss/p57098/euijin1/caver/runtime_logs/stage1_modeb_rehearsal/6109/latency.json`
  - shadow-mode dry run completed and wrote:
    - `/rdss/p57098/euijin1/caver/runtime_logs/stage1_modeb_rehearsal/6109/shadow_context/shadow_context.json`
  - exact-payload tensors were archived successfully under:
    - `/rdss/p57098/euijin1/caver/runtime_logs/stage1_modeb_rehearsal/6109/shadow_context/policy_payload/`

- Measured exact-service latency on `6109`:
  - client end-to-end:
    - p50 about `174.8 ms`
    - p95 about `182.3 ms`
    - p99 about `182.4 ms`
  - server-side inference:
    - p50 about `173.2 ms`
    - p95 about `180.7 ms`
    - p99 about `180.7 ms`
  - interpretation:
    - on-node transport overhead is small relative to model inference time
    - the compute-side service is stable enough for later shadow-mode work

- Safety-authority rehearsal result:
  - the exact-model dry run produced a chunk that the local scaffold safety layer rejected for:
    - `velocity`
    - `acceleration`
  - interpretation:
    - this is acceptable for Mode-B readiness
    - the SDRE-side requirement is that the local bridge remains safety-authoritative and can reject unsafe chunks without commanding hardware
    - this is not evidence that the real robot is ready; it is evidence that the software path preserves the intended safety boundary

- SDRE-side conclusion:
  - everything materially verifiable on the SDRE compute side is now done:
    - wrappers and runtime assets
    - exact GPU-backed policy service
    - websocket bridge
    - timing capture
    - Stage-1 scaffold contracts
    - no-hardware shadow-mode dry run
  - remaining blockers are external to SDRE compute:
    - SDRE admin confirmation that the intended tunnel / service pattern is allowed
    - the local robot-control PC with ROS / CAN authority
    - real network-path latency measurement from that local machine
    - physical PiPER hardware, camera calibration, and labeler validation

- Practical next move:
  - Mode B is now sufficiently verified on the SDRE side to justify moving on to local PiPER environment setup
  - do not treat this as approval to skip the local safety-authoritative bridge or the `F0` shadow-mode staircase

## Stage-F readiness note after Stage-E review on 2026-05-11

- Stage F should not start as a full three-task PiPER study yet.
- Allowed next Stage-F work:
  - local robot-control machine setup
  - camera/URDF/safety bridge validation
  - no-motion shadow-mode logging
  - one-task pilot preparation
- Still required before full PiPER study:
  - complete or recover the Stage-E CAVER `N=25/100` held-out budget cells
  - regenerate the held-out budget curve
  - inspect per-family admission and performance, especially the current `container_insertion_proxy` admission skew
  - run minimal attribution diagnostics or explicitly defer them from the first hardware pilot
- Source alignment patched on 2026-05-11:
  - raw DR target, raw calibrator mean, novelty-aware candidate features, deterministic Stage-0 validity mask, and backend-finalizer recovery path

## Baseline policy for PiPER planning on 2026-05-12

- Use the same budget-accounting language as Stage E:
  - one trusted executed candidate costs one unit of `N`
  - round size remains `B_round=25`
  - backend updates happen between rounds, not after every decision context
- PiPER primary baselines should be:
  - seed-only anchor
  - vanilla real-only `K=1`, if hardware time permits
  - matched real-only uniform `K=4`
  - full CAVER `K=4`
  - selection-only and admission-only if pilot time permits
- Do not make `K=4 execute-all` a PiPER primary baseline:
  - it would spend up to `4x` more real executions at the same context count
  - it requires repeated resets from comparable physical states
  - it is useful only as a LIBERO diagnostic or as an explicitly costed hardware stress test later

## Post-Stage-E gate on 2026-06-01

- Stage E is now complete enough to proceed to Stage-F readiness work.
- The Stage-E result does not justify launching the full three-task PiPER study immediately.
- Updated after the vanilla `K=1` review:
  - do not freeze the current mainline CAVER rule as the final PiPER method yet
  - Stage E+ should first test a success-preserving / relaxed-admission CAVER variant against vanilla `K=1`
  - hardware planning may continue, but real seed collection and pilot execution should wait until the Stage E+ method decision is made
- Interpretation to carry forward:
  - mainline CAVER supports a selective-admission / backend-data-efficiency claim, not a raw-success dominance claim
  - vanilla real-only `K=1` remains a strong simulation baseline
  - success-only admission is the best raw-success attribution variant at `N=50`, but it admits more backend data than mainline CAVER
  - no-DR needs a caveat because two seeds skipped backend update after admitting no trajectories
- Allowed next work:
  - PiPER local environment setup
  - camera calibration
  - URDF/safety shield validation
  - no-motion shadow-mode logging
  - GE-Sim/provider latency measurement on real observations
  - labeler validation against manual inspection
- Method-side checks required before the full PiPER study:
  - shadow-mode audit of `mu`, `sigma`, `LCB`, admitted-positive rate, skipped-update frequency, and provider latency
  - one-task pilot comparing:
    - proposal-mainline CAVER
    - matched real-only `K=4`
    - success-only admission as a diagnostic or pre-registered secondary variant
  - decide after the pilot whether to:
    - keep the current LCB gate
    - relax the LCB/admission threshold
    - promote success-only admission to the full-study pilot variant
- Practical rule:
  - continue only non-budget-consuming Stage-F planning while Stage E+ runs
  - do not consume the shared 120-context real seed set or launch the one-task pilot until the Stage E+ `K=1` challenge is resolved
  - do not launch the full three-task study until the admission-gate audit and one-task pilot are complete

## Paper-aligned Stage-F readiness gate update on 2026-06-09

- The active paper now includes a dedicated Stage-F readiness gate before the full PiPER study.
- Required before any budgeted hardware learning run:
  - no-motion shadow-mode logging of observations, candidates, safety masks, propensities, and selected actions;
  - local safety-shield validation with scripted safe chunks and deliberately invalid chunks;
  - marker/geometric end-state labeler validation against manual inspection;
  - progress-label validation on every task where CAVER+FASR would be used;
  - latency logging for policy sampling, GE-Sim/provider inference, selector/calibrator scoring, and robot idle time.
- Full three-task Stage 1 should start only after a one-task, one-seed, `N=25` pilot validates the closed loop.
- If PiPER progress labels are unreliable, the paper-aligned fallback is strict CAVER, not CAVER+FASR.

## PiPER progress-label implementation scaffold on 2026-06-09

- Added `scripts/stage1/piper_progress_labeler.py`.
- The scaffold defines deterministic geometric progress scores for:
  - block-to-tray: normalized block-to-tray XY-distance reduction plus tray containment/resting-height bonuses;
  - can-to-bowl: normalized can-to-bowl XY-distance reduction plus bowl containment/resting-height bonuses;
  - two-block stack: normalized top-to-bottom XY alignment plus top-above-bottom, height-gap, and orientation/stability bonuses.
- These progress labels are not rewards and are not evidence yet; they are the planned FASR eligibility signals to validate in Stage F.
- Updated `scripts/stage1/validate_stage1_scaffold.py` to smoke-test the progress labeler.
- Validation passed under the project Python stack:
  - `scripts/env/with_openpi_pistepnft_libero_train.sh -- python scripts/stage1/validate_stage1_scaffold.py`
  - output included `OK progress_label=value:0.827,audit:False,reason:ok`.
- Required before enabling CAVER+FASR on PiPER:
  - collect at least 10 scripted or teleoperated calibration traces per task;
  - log marker/object poses, automatic progress scores, final success labels, and manual prefix audits;
  - require at least 90% final-label agreement and at least 90% FASR-prefix eligibility agreement with manual inspection;
  - require false-eligible repair rate at most 5%;
  - mark low-confidence marker traces audit-required and ineligible for FASR.
