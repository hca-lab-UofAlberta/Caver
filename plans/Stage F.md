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

Collect the shared 120-context real seed set:

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
