# Stage E++ CAVER-LVD Strict Plan

## Objective

Resolve the main Stage-E ambiguity before PiPER:

- Does CAVER's candidate selection mechanism help beyond vanilla `K=1`?
- Does CAVER add value beyond progress-label/FASR repair?
- Does GE-Sim/provider information and DR correction matter?

The strict method pass is **CAVER-LVD: Learned Verification Distribution**. The core method is a learned distribution over candidate chunks, not another heuristic score stack.

## Current Diagnosis

- Strict CAVER is scientifically coherent but empirically mixed.
- Strict CAVER's strongest evidence is backend-data efficiency, not raw held-out success dominance.
- CAVER+FASR currently gives the best Stage-0 numbers, but the gain may come from progress-label segment repair rather than provider-aware selection.
- Vanilla `K=1` remains a strong pressure-test baseline.

## Strict Main Method

At each decision context:

1. Sample `K=4` candidate chunks from the unchanged `pi0.5` backend.
2. Build candidate feature vector `z(c,k)` from:
   - action/chunk features,
   - provider summary features when enabled,
   - proxy family id,
   - policy-query progress index.
3. Score each safe candidate with learned selector `g_psi(z(c,k))`.
4. Sample one candidate from:
   - `softmax(g_psi(z(c,k)) / T)` over safe candidates,
   - mixed with an epsilon-uniform floor for stable logged propensities.
5. Execute only the sampled candidate in LIBERO.
6. Log exact probability `p(c,k)`.
7. Build candidate-level DR pseudo-outcomes from the logged propensity and verified outcome.
8. Fit the next selector only from previous/lagged data.
9. Train the unchanged exact-offline NFT backend on the admitted/weighted data variant for that method.

## Non-Negotiable Protocol Rules

- `H=4` for all online and held-out execution.
- `K=1` baseline must remain in every diagnostic table.
- `K=4` uniform baseline must remain in every diagnostic table.
- CAVER-LVD must log exact candidate probabilities.
- The initial LVD selector used for reportable results must come from a counted/common seed-calibration source, not from uncounted Stage-E budget outcomes.
- Existing Stage-E K=4 DR datasets may be used for code smoke tests and development-only debugging, but not as the unreported initializer for final comparison jobs.
- No same-round selector leakage:
  - round `r` may only use LVD model fit before round `r`.
  - model fit from round `r` can only affect round `r+1`.
- LVD training must use listwise candidate groups, not independent binary labels only.
- DR targets must be clipped or stabilized for selector training.
- Propensity floor must remain nonzero.
- Failed or incomplete jobs are not results.
- All newly submitted jobs must be recorded with job IDs, roots, and ETA.
- Prefer RDSS for new heavy posttrain/recovery outputs because `/projects` is nearly full.

## CAVER-LVD Model

Artifact:

- `stagee_lvd_selector_mlp_v1`

Inputs:

- `base_feature_vector` from candidate metrics,
- proxy family one-hot features,
- normalized policy-query index,
- optional family/progress interaction features.

Training target:

- group candidates by `(context_id, policy_query_index)`;
- define target weights with `softmax(clipped_DR_target / tau_y)`;
- train listwise cross-entropy:
  - target distribution is stop-gradient;
  - model distribution is `softmax(g_psi(z) / T_train)`.

Inference:

- compute `g_psi(z)` for all safe candidates;
- sample from `softmax(g_psi(z) / T_select)` mixed with epsilon-uniform;
- store exact probabilities and scores in the trace.

Default stabilization:

- target key: `dr_pseudo_outcome_clipped`;
- target temperature: `0.20`;
- selector temperature: `0.50`;
- exploration floor: `0.10`;
- importance/DR clipping comes from the existing DR dataset builder and clipped target.

## Required Diagnostic Methods

Run at `N=50`, seeds `{7,13,29}` first.

Required baselines:

- `vanilla_k1`: `K=1`, first candidate, train on real-only/all successful or current vanilla protocol.
- `uniform_k4`: `K=4`, uniform selection, no CAVER selector.
- `uniform_k4_fasr`: `K=4`, uniform selection, FASR admission/repair.
- `strict_caver`: existing strict proposal-mainline CAVER.
- `caver_fasr`: existing CAVER+FASR.

Required new LVD methods:

- `caver_lvd`: learned distribution, no FASR, no hard LCB admission; use success-only admission first.
- `caver_lvd_fasr`: learned distribution plus FASR admission/repair.
- `caver_lvd_no_provider`: learned distribution with provider disabled.
- `caver_lvd_no_dr`: learned distribution trained without DR correction, using selected observed outcomes / nuisance target fallback.

## Primary Comparisons

Main checks:

- `caver_lvd` vs `vanilla_k1`
- `caver_lvd_fasr` vs `uniform_k4_fasr`
- `caver_lvd` vs `caver_lvd_no_provider`
- `caver_lvd` vs `caver_lvd_no_dr`

Interpretation:

- If `caver_lvd_fasr > uniform_k4_fasr`, CAVER selection adds value beyond progress repair.
- If `caver_lvd >= vanilla_k1`, CAVER selection is useful without relying on FASR.
- If `caver_lvd` is comparable to `vanilla_k1` but uses much less backend data, the claim is selective verification/data efficiency.
- If `uniform_k4_fasr` matches or beats `caver_lvd_fasr`, the current gain is likely FASR/progress-label driven.

## Success Criteria

For continuing toward a strong PiPER paper claim:

- `caver_lvd` at `N=50` should reach held-out test success at least comparable to `vanilla_k1`.
- `caver_lvd_fasr` should beat `uniform_k4_fasr` by at least `0.005` to `0.010` mean held-out test success.
- `caver_lvd_no_provider` should be worse than full `caver_lvd`, or GE-Sim should be demoted in the paper.
- `caver_lvd_no_dr` should be worse than full `caver_lvd`, or DR should be demoted in the paper.
- If these fail, reframe CAVER as selective-admission/data-compression rather than raw-success improvement.

## Implementation Checklist

1. Add LVD selector artifact loader/predictor.
2. Add LVD listwise trainer from `caver_dr_candidate_dataset.jsonl`.
3. Add `--lvd-selector-model-path` to LIBERO evaluator and Stage-E round scripts.
4. Add `selection_policy=caver_lvd`.
5. Add lagged LVD plumbing:
   - optional seed LVD model for round 1;
   - fit `caver_lvd_selector.json` after each round;
   - pass it into the next round.
6. Add launcher methods:
   - `caver_lvd`;
   - `caver_lvd_fasr`;
   - `caver_lvd_no_provider`;
   - `caver_lvd_no_dr`.
7. Fit a development-only LVD selector from existing completed K=4 CAVER DR datasets for code smoke tests.
8. Create a fair seed LVD source:
   - preferred: run a `T_seed_S0`, `K=4`, uniform-selection calibration pass and count/report it as the common Stage-0 seed-calibration source;
   - fallback: explicitly label any reused prior budget data as development-only and do not use it for paper results.
9. Dry-run Slurm command generation.
10. Submit only the `N=50` diagnostic cells after smoke tests and the fair seed selector pass.
11. Aggregate and update paper only after all diagnostic cells finish.

## Progress Log

- 2026-06-10: Implemented the first CAVER-LVD code path and found that `metadata/stage0/calibrator/stage0_seed_dr_candidate_dataset.jsonl` has 8,456 single-candidate groups, so it cannot train a listwise selector.
- 2026-06-10: Identified three completed K=4 CAVER N=50 DR datasets under RDSS with 11,448 total four-candidate groups. These are suitable for development smoke tests only; using them as an uncounted seed selector would leak budget outcomes into the method comparison.
- 2026-06-10: Fixed LVD trainer grouping to include dataset source in the candidate-menu key. Without this, repeated `context_id` values across seeds merged independent menus into invalid 12-candidate groups.
- 2026-06-10: Fitted development-only LVD smoke artifacts:
  - `metadata/stage0/lvd_selector/dev_stagee_budget50_lvd_selector_dr_clipped_mlp_v1.json`
  - `metadata/stage0/lvd_selector/dev_stagee_budget50_lvd_selector_observed_selected_else_nuisance_mlp_v1.json`
  - Corrected training summary: 45,792 records, 11,448 four-candidate groups, validation top-1 target match 0.286.
- 2026-06-10: Patched uniform K=4 evaluation to log diagnostic candidate metric tables without changing uniform sampling. This is required so a fair seed-calibration run can train LVD from uniform K=4 data rather than prior CAVER-selected data.
- 2026-06-10: Dry-run validation passed for `caver_lvd` and `caver_lvd_no_dr`; lagged dry-run JSON shows round 1 uses the seed selector, round 2 uses `round_001/caver_lvd_selector.json`, and the finalizer uses `round_002/caver_lvd_selector.json`.
- 2026-06-10: Submitted fair K=4 uniform seed-calibration job:
  - Slurm job: `9043`
  - Partition/GPU: `gpu-h200`, `h200`, 1 GPU
  - Run dir: `/rdss/p57098/euijin1/caver/runs/stagee__caver-lagged__manifest-t_seed_s0-all-lvd-seed-calib-k4-uniform__seed7__budget120__20260610T065003Z`
  - Stdout: `/rdss/p57098/euijin1/caver/logs/slurm/stagee__caver-lagged__manifest-t_seed_s0-all-lvd-seed-calib-k4-uniform__seed7__budget120__20260610T065003Z-9043.out`
  - Stderr: `/rdss/p57098/euijin1/caver/logs/slurm/stagee__caver-lagged__manifest-t_seed_s0-all-lvd-seed-calib-k4-uniform__seed7__budget120__20260610T065003Z-9043.err`
  - Started: `2026-06-10 00:50 MDT`
  - Slurm walltime limit/end: `5-00:00:00`, ending no later than `2026-06-15 00:50 MDT`
  - Practical ETA: likely 1-3 days, but use the Slurm walltime as the conservative bound.
- 2026-06-10 early health check for job `9043`:
  - State: running on `h200-03`.
  - Policy server started and LIBERO connected.
  - GE-Sim worker loaded assets successfully and processed 100+ requests.
  - Compact trace check: 30 records from the first context, all with K=4 probabilities summing to 1 and nonempty candidate metric tables.
  - No immediate import/path/provider failure observed.
- 2026-06-10 01:42 MDT progress for job `9043`:
  - Still running on `h200-03`.
  - Round 1 trace: 413 compact records across 6 seed contexts.
  - All checked records have K=4 probabilities summing to 1 and nonempty candidate metric tables.
  - Early practical ETA is roughly 12-24 hours if the observed rate holds; conservative Slurm walltime endpoint remains `2026-06-15 00:50 MDT`.
- 2026-06-10 02:22 MDT supporting FASR-baseline recovery:
  - Jobs `9040-9042` failed at FSDP checkpoint save on shared storage with `OSError: [Errno 5] Input/output error`.
  - Job `9044` fixed that checkpoint-save failure by writing RLinf checkpoints to node-local `/tmp`, but failed at eval because the RDSS `model.safetensors` export had an invalid all-zero header.
  - Patched posttrain export to stage and validate `model.safetensors` on node-local storage, copy the validated export to RDSS, then validate the RDSS copy before evaluation.
  - Submitted smoke job `9045` for reusable `k1_fasr`, `N=25`, seed `7`.
  - `9045` artifact root: `/rdss/p57098/euijin1/caver/stagee_fasr_fairness_posttrain_recovery_localckpt_v2/stagee__caver-lagged__manifest-t_train_s0-all-k1-fasr-n25__seed7__budget25__20260609T164124Z`.
  - `9045` checkpoint/export validation ETA: `2026-06-10 02:52-03:05 MDT`.
- 2026-06-10 02:52 MDT supporting FASR-baseline recovery:
  - Job `9045` passed local `/tmp` safetensors validation but failed copying the 7.5GB checkpoint to RDSS: `cp: failed to close ... Input/output error`.
  - The RDSS checkpoint copy again had an all-zero safetensors header, so large OpenPI checkpoints should not be stored on RDSS for this path.
  - Submitted project-checkpoint smoke job `9046` for reusable `k1_fasr`, `N=25`, seed `7`.
  - `9046` artifact root: `/projects/p57098/euijin1/caver_stagee_fasr_fairness_posttrain_recovery_projectckpt_v1/stagee__caver-lagged__manifest-t_train_s0-all-k1-fasr-n25__seed7__budget25__20260609T164124Z`.
  - `9046` checkpoint/export validation ETA: `2026-06-10 03:22-03:35 MDT`.
- 2026-06-10 03:22 MDT supporting FASR-baseline recovery:
  - Job `9046` passed local `/tmp` and project-filesystem safetensors validation with `812` keys and started held-out eval.
  - Submitted project-checkpoint recovery jobs:
    - `9047`: reusable `k1_fasr`, `N=25`, seed `13`.
    - `9048`: reusable `k1_fasr`, `N=25`, seed `29`.
  - `9047`/`9048` checkpoint validation ETA: `2026-06-10 03:50-04:05 MDT`.
  - Corrupt RDSS seed-7 checkpoint files from failed `localckpt_v1` and `localckpt_v2` attempts were removed.
- 2026-06-10 03:48 MDT supporting FASR-baseline recovery:
  - `9047` seed 13 passed local `/tmp` and project-filesystem `model.safetensors` validation with `812` keys.
  - `9048` seed 29 is still running and healthy at roughly step `17/20`; next gate is checkpoint validation.
  - `9046` seed 7 remains in held-out evaluation.
  - The storage rule is now explicit for this path: use RDSS for logs/compact summaries only; use project storage for large exported OpenPI checkpoints and eval artifacts.
- 2026-06-10 03:58 MDT supporting FASR-baseline recovery:
  - `9048` seed 29 passed local `/tmp` and project-filesystem `model.safetensors` validation with `812` keys and started held-out evaluation.
  - All three recovered `k1_fasr`, `N=25` posttrain seeds (`9046`, `9047`, `9048`) have passed the original recovery bug gate.
  - Remaining action is to wait for held-out summaries, verify outputs, and aggregate the recovered fairness baseline.
- 2026-06-10 04:05 MDT supporting progress:
  - `9046`, `9047`, and `9048` remain in held-out evaluation; summaries are still pending.
  - `9043` seed-calibration job is healthy at roughly context `22/25` of round 1/5.
  - Updated `scripts/stagee/plot_stagee_heldout_budget_curve.py` so it can aggregate `k1_fasr`, `uniform_k4_fasr`, and `caver_lvd*` results once summaries appear.
  - Dry-run aggregation against the project recovery root completed with `0` complete and `3` missing cells, as expected before held-out summaries are written.
- 2026-06-10 11:15 MDT supporting progress:
  - `9046`, `9047`, and `9048` completed and produced all recovered `k1_fasr`, `N=25` held-out summaries.
  - aggregate: validation `0.203 +/- 0.005`, test `0.223 +/- 0.003`, admitted demo items `400.3`, primitive steps `1585.7`.
  - wrote `figures/stagee_k1_fasr_n25_recovery_summary.{json,png,pdf}`.
  - `9043` failed after round 1 due RDSS gzip close `OSError: [Errno 5] Input/output error`; this is storage-path fragility, not method failure.
  - submitted replacement project-storage seed-calibration job `9056`.
  - `9056` run dir: `/projects/p57098/euijin1/caver_stagee_lvd_seed_calib_runs_v1/stagee__caver-lagged__manifest-t_seed_s0-all-lvd-seed-calib-k4-uniform-project-v1__seed7__budget120__20260610T171128Z`.
  - `9056` startup check passed: OpenPI exact policy server connected and context `1/25` started.
- 2026-06-10 15:07 MDT fairness-baseline expansion:
  - submitted the missing `uniform_k4_fasr` baseline at `N in {25, 50}` for seeds `{7, 13, 29}`.
  - job IDs:
    - `9073`: `N=25`, seed `7`.
    - `9074`: `N=25`, seed `13`.
    - `9075`: `N=25`, seed `29`.
    - `9076`: `N=50`, seed `7`.
    - `9077`: `N=50`, seed `13`.
    - `9078`: `N=50`, seed `29`.
  - all six jobs started on `gpu-h200`.
  - root paths:
    - run root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_runs_v3`.
    - posttrain root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_posttrain_v3`.
    - logs: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_logs_v3`.
    - submission record: `logs/runtime/stagee_uniform_k4_fasr_parallel_v3_20260610T210702Z.json`.
  - storage patch for this batch:
    - `scripts/stagee/run_stage0_posttrain_from_round.sh` now supports `--keep-export-node-local`.
    - `scripts/slurm/submit_stagee_heldout_budget_curve.py` now writes inline jobs with compute-node `/tmp` staging and node-local posttrain export.
    - RDSS is used for durable logs/summaries; large OpenPI checkpoints are not copied to RDSS.
  - conservative ETA:
    - `N=25`: no later than `2026-06-11 08:07 MDT`.
    - `N=50`: no later than `2026-06-11 14:07 MDT`.
  - interpretation gate:
    - if `uniform_k4_fasr` matches CAVER+FASR, the current improvement is mostly FASR/progress-label driven.
    - if CAVER+FASR or `caver_lvd_fasr` beats `uniform_k4_fasr`, then CAVER selection has evidence beyond FASR.
  - startup health check at `2026-06-10 15:11 MDT`:
    - all six jobs remained running.
    - every job connected to the OpenPI exact-policy server.
    - every job started LIBERO context `1/25`.
    - trace files were being written, so the jobs are past environment/import/startup failure modes.
- 2026-06-11 01:51 MDT failure diagnosis and recovery:
  - jobs `9073` through `9078` failed after online execution because the full-payload RDSS gzip demo traces failed at close with `OSError: [Errno 5] Input/output error`.
  - this confirms that RDSS is unsuitable for multi-GB full-payload gzip traces, not only for large OpenPI checkpoints.
  - job `9056` failed separately because `all_executed_nonerror` admission was paired with `success_only` demo trace writing, so artifact construction saw an admitted-record mismatch.
  - patched `run_stage0_caver_round.sh`:
    - full-payload demo/admitted traces can now live under `CAVER_STAGEE_HEAVY_TRACE_ROOT` on compute-node `/tmp`.
    - durable run directories receive lightweight trace-source manifests.
    - `all_executed_nonerror` now forces all executed trace writing.
  - patched Slurm submitters:
    - `submit_stagee_heldout_budget_curve.py` sets `CAVER_STAGEE_HEAVY_TRACE_ROOT="${TMPDIR}/heavy_traces"`.
    - `submit_stage0_caver_lagged_budget.sh` also sets the node-local heavy-trace root.
  - submitted `uniform_k4_fasr` recovery batch `v4`:
    - `9099`: `N=25`, seed `7`.
    - `9100`: `N=25`, seed `13`.
    - `9101`: `N=25`, seed `29`.
    - `9102`: `N=50`, seed `7`.
    - `9103`: `N=50`, seed `13`.
    - `9104`: `N=50`, seed `29`.
  - roots:
    - run root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_runs_v4`.
    - posttrain root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_posttrain_v4`.
    - logs: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_logs_v4`.
    - submission record: `logs/runtime/stagee_uniform_k4_fasr_parallel_v4_20260611T075105Z.json`.
  - conservative ETA:
    - `N=25`: `2026-06-11 18:51 MDT`.
    - `N=50`: `2026-06-12 00:51 MDT`.
  - seed-calibration recovery is still pending; do not resubmit it until the heavy-trace fix is confirmed on the simpler `uniform_k4_fasr` batch.
  - startup health check at `2026-06-11 01:56 MDT`:
    - jobs `9099` through `9104` remained running.
    - all six connected to OpenPI exact-policy servers and started LIBERO context `1/25`.
    - compact traces were being written.
    - the durable RDSS run directories did not contain `caver_online_demo_chunks.jsonl.gz`, confirming that the old RDSS gzip path is no longer active during online collection.
- 2026-06-11 08:32 MDT failure diagnosis and recovery:
  - jobs `9099` through `9104` failed after the online rollout phase; no held-out summaries were produced.
  - the node-local heavy-trace patch worked, but artifact construction still wrote `caver_selector_contexts.jsonl` directly to RDSS and hit `OSError: [Errno 5] Input/output error`.
  - `/projects` is full, and RDSS is unreliable for active high-churn writes, so the storage rule is now stricter:
    - active round artifacts: compute-node `/tmp`.
    - active posttrain/checkpoint/export artifacts: compute-node `/tmp`.
    - durable RDSS copy-back: compact JSON summaries only.
  - patched `submit_stagee_heldout_budget_curve.py` to rewrite generated per-cell round jobs onto node-local `/tmp` and copy compact summaries back to RDSS only after successful posttrain.
  - submitted `uniform_k4_fasr` recovery batch `v5`:
    - `9106`: `N=25`, seed `7`, walltime `17:00:00`.
    - `9107`: `N=25`, seed `13`, walltime `17:00:00`.
    - `9108`: `N=25`, seed `29`, walltime `17:00:00`.
    - `9109`: `N=50`, seed `7`, walltime `23:00:00`.
    - `9110`: `N=50`, seed `13`, walltime `23:00:00`.
    - `9111`: `N=50`, seed `29`, walltime `23:00:00`.
  - all six started on `gpu-h200`, node `h200-03`.
  - roots:
    - run root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_runs_v5`.
    - posttrain root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_posttrain_v5`.
    - logs: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_logs_v5`.
    - submission record: `logs/runtime/uniform_k4_fasr_v5_submission.json`.
  - conservative ETA:
    - `N=25`: `2026-06-12 01:32 MDT`.
    - `N=50`: `2026-06-12 07:32 MDT`.
  - seed-calibration recovery remains on hold until v5 proves that the fully node-local active-write path produces held-out summaries.
- 2026-06-11 15:14 MDT progress:
  - `9106`, `9107`, and `9108` (`uniform_k4_fasr`, `N=25`) failed after completing online rollout.
  - failure was not storage this time; it was a nested trace-source manifest bug in the lagged finalizer.
  - root cause: `build_caver_round_artifacts.py` handled one `stagee_trace_source_manifest_v1` level, but the lagged finalizer produced a manifest pointing to per-round manifests.
  - patched `build_caver_round_artifacts.py` to resolve nested trace-source manifests recursively.
  - syntax check passed.
  - `9109`, `9110`, and `9111` (`uniform_k4_fasr`, `N=50`) remain running in round `2/2`; the patch was applied before their finalizer stage.
  - current round-2 progress:
    - `9109`: context `16/25`.
    - `9110`: context `15/25`.
    - `9111`: context `14/25`.
  - current v5 held-out summary count: `0`.
  - next gate: let `9109-9111` validate the patched finalizer, then recover/resubmit the failed `N=25` cells.
- 2026-06-11 17:18 MDT immediate retry:
  - submitted `uniform_k4_fasr`, `N=25`, v6 retries with the nested-manifest patch:
    - `9138`: seed `7`, walltime `17:00:00`.
    - `9139`: seed `13`, walltime `17:00:00`.
    - `9140`: seed `29`, walltime `17:00:00`.
  - roots:
    - run root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_runs_v6`.
    - posttrain root: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_posttrain_v6`.
    - logs: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_logs_v6`.
    - submission record: `logs/runtime/uniform_k4_fasr_n25_v6_submission.json`.
  - Slurm placed all three on `h200-03` despite an environment-level exclude request.
  - startup check:
    - all three jobs are running.
    - all three entered the lagged wrapper and wrote policy-server log paths.
  - ETA:
    - conservative walltime upper bound: `2026-06-12 10:18 MDT`.
    - practical ETA if recent pace holds: `2026-06-11 23:30 MDT` to `2026-06-12 03:00 MDT`.
  - concurrent `N=50` status:
    - `9109-9111` completed online round `2/2`.
    - each had round-1 online success `20/25 = 0.800` and round-2 online success `0/25 = 0.000`.
    - they are in the patched finalizer/posttrain stage, with no nested-manifest `KeyError` visible in latest tails.
- 2026-06-11 21:01 MDT progress:
  - `9109`, `9110`, and `9111` passed the previous nested-manifest finalizer failure point and started posttrain.
  - posttrain start times:
    - `9109`: `2026-06-11 18:32:56 MDT`.
    - `9111`: `2026-06-11 18:51:42 MDT`.
    - `9110`: `2026-06-11 19:03:49 MDT`.
  - held-out eval is in progress; current logs show roughly:
    - `9109`: `79/100` contexts.
    - `9110`: `41/100` contexts.
    - `9111`: `33/100` contexts.
  - `9138`, `9139`, and `9140` are still running; all reached context `25/25` starts, and seed `29` logged online completion `10/25 = 0.400`.
  - no current `Traceback`, `KeyError`, RDSS `OSError`, or runtime crash is visible.
  - current v5/v6 held-out summary count: `0`.

## Next Required Action

When jobs `9106` through `9111` finish:

1. Verify every cell has `posttrain_holdout_summary.json`.
2. Aggregate `uniform_k4_fasr` against `k1_fasr`, current CAVER+FASR, and vanilla `K=1`.
3. Decide whether CAVER+FASR's advantage survives a fair same-wrapper FASR baseline.

When replacement job `9056` finishes:

1. Check `caver_round_summary.json`, `caver_dr_candidate_dataset.summary.json`, and Slurm stderr.
2. Fit reportable seed selectors from the project-storage `caver_dr_candidate_dataset.jsonl`:
   - `stage0_seed_lvd_selector_dr_clipped_mlp_v1.json`
   - `stage0_seed_lvd_selector_observed_selected_else_nuisance_mlp_v1.json`
3. Dry-run the N=50 diagnostic cells using the reportable seed selectors.
4. Submit only the strict N=50 diagnostic set after the seed-selector artifacts pass load/predict smoke tests.

## PiPER Gate

Proceed to PiPER full study only if:

- CAVER-LVD gives an interpretable selection benefit, or
- the paper is explicitly reframed as selective-admission/data-compression and PiPER Stage F is designed to test that narrower claim.

PiPER infrastructure/shadow-mode preparation can continue in parallel, but the full real-robot study should wait for this Stage-E++ diagnostic.

## Uniform K4 + FASR Control Status: 2026-06-12 03:30 MDT

- Purpose:
  - This control checks whether current CAVER+FASR gains come from provider-aware CAVER selection or mainly from FASR/progress-label admission/repair.
  - It is required before interpreting CAVER+FASR as a selector improvement over vanilla `K=1`.
- Prior v5/v6 outcome:
  - `9109`, `9110`, `9111`, `9138`, `9139`, and `9140` reached held-out evaluation.
  - They failed only at final compact-artifact copy-back to RDSS, so no clean `posttrain_holdout_summary.json` artifacts survived.
  - Log-derived triage results suggest `uniform_k4_fasr` is around `0.22-0.23` held-out test at `N=25`/`N=50`, but this must be confirmed from clean summaries.
- Fix:
  - `scripts/slurm/submit_stagee_heldout_budget_curve.py` now uses metadata-free byte copying with retries instead of `shutil.copy2`.
  - Node-local salvage job `9142` completed, but `/tmp` job directories were already gone.
- Clean v7 rerun:
  - `9143`, `9144`, `9145`: `uniform_k4_fasr`, `N=25`, seeds `7`, `13`, `29`.
  - `9146`, `9147`, `9148`: `uniform_k4_fasr`, `N=50`, seeds `7`, `13`, `29`.
  - roots: `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_runs_v7`, `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_posttrain_v7`, `/rdss/p57098/euijin1/caver/stagee_uniform_k4_fasr_logs_v7`.
  - submission record: `logs/runtime/uniform_k4_fasr_v7_submission.json`.
- ETA:
  - conservative upper bound: `2026-06-12 20:30 MDT` for `N=25`, `2026-06-13 02:30 MDT` for `N=50`.
  - expected practical completion if previous pace repeats: `2026-06-12 afternoon/evening MDT`.
