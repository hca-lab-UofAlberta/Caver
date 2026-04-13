# Stage G: Stage-1 Real Study and Packaging

## Goal

Run the matched-budget PiPER study and package the results so the paper claim is auditable.

## Fixed Stage-1 protocol

- tasks:
  - block-to-tray
  - can-to-bowl
  - two-block stack
- budgets: `N in {0, 25, 50, 100, 200}`
- rounds per nonzero budget: `N / 25`
- two full independent runs per budget, with a third run at `N=100`
- shared seed-warmed checkpoint at `N=0`

## Reporting rules

- report curves against matched training budget `N`
- also report against total real interaction count `N_total`
- keep validation and test interactions separate from training-budget accounting
- use the same held-out evaluation protocol across methods

## Study order

1. CAVER vs real-only `pi-StepNFT`
2. attribution ablations needed for the main claim
3. WoVR only if still feasible under the matched-budget rule

## Deliverables

- final Stage-1 tables and curves
- run manifests and archived logs
- bootstrap and permutation-test outputs
- a reproducibility bundle stored under `/rdss`

## Exit criteria

- The study meets or clearly misses the proposal's publishability threshold with transparent accounting.
- Every final figure can be traced back to a run manifest and archived log directory.
- No result depends on undocumented extra real interaction.

## Risks and watchpoints

- If WoVR requires extra real interaction or unsupported infrastructure, report it as infeasible under the matched-budget protocol instead of silently relaxing the rules.
- Keep the seed-only anchor visible so Stage-1 gains are not conflated with warm-start gains.
