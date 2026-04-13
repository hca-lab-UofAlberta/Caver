# SDRE Bootstrap Workflow

This repo now contains enough Stage A/B scaffolding to create manifests, generate Slurm jobs, and bootstrap the first SDRE-native environments.

## Interactive shells

L40S:

```bash
scripts/slurm/interactive_l40s.sh --time 01:00:00 -- bash -l
```

H200:

```bash
scripts/slurm/interactive_h200.sh --time 01:00:00 -- bash -l
```

## Dry-run a batch submission

```bash
scripts/slurm/submit_experiment.sh \
  --stage stage0 \
  --method smoke \
  --task env \
  --seed 0 \
  --budget 0 \
  --partition gpu-l40s \
  --qos normal \
  --gpu-type l40s \
  --dry-run \
  -- bash -lc 'hostname; python3 --version'
```

This creates:

- a per-run directory under `runs/`
- a manifest under that run directory
- a generated `job.sbatch`

## Bootstrap environments

Dry-run:

```bash
scripts/env/bootstrap_openpi_env.sh --dry-run
scripts/env/bootstrap_libero_env.sh --dry-run
scripts/env/bootstrap_pistepnft_env.sh --dry-run
scripts/env/bootstrap_gesim_env.sh --dry-run
```

Actual bootstrap:

```bash
scripts/env/bootstrap_openpi_env.sh
scripts/env/bootstrap_libero_env.sh
scripts/env/bootstrap_pistepnft_env.sh
```

Policy-side combined runtime:

```bash
scripts/env/with_openpi_libero_eval.sh -- bash -l
scripts/env/with_openpi_libero_eval.sh -- python -c 'import toolkits.eval_scripts_openpi.libero_eval'
```

Train-side combined runtime:

```bash
scripts/env/with_openpi_pistepnft_libero_train.sh -- bash -l
scripts/env/with_openpi_pistepnft_libero_train.sh -- python -c 'import examples.embodiment.train_embodied_agent'
```

Stable split bridge for Stage 0 LIBERO:

```bash
scripts/bridge/run_libero_remote_eval.sh \
  --dummy-server \
  -- \
  --task-suite-name libero_spatial \
  --max-tasks 1 \
  --num-trials-per-task 1 \
  --max-steps 20 \
  --results-path logs/runtime/libero_dummy_smoke.json
```

Manifest-aware Stage-0 partition smoke with the dummy websocket server:

```bash
scripts/stage0/run_stage0_partition_eval.sh \
  --dummy-server \
  --partition-name T_seed_S0 \
  --family-ids drawer_open_proxy \
  --max-contexts 1 \
  --num-steps-wait 0 \
  --replan-steps 1 \
  --max-steps 1 \
  --results-path logs/runtime/stage0_partition_dummy.json \
  --context-log-path logs/runtime/stage0_partition_dummy.jsonl
```

Stage-0 seed warm-start collection wrapper:

```bash
scripts/stage0/collect_stage0_warm_start.sh \
  --openpi-native \
  --libero-gl-backend osmesa
```

Slurm submission for the full 120-context seed warm-start collection:

```bash
scripts/slurm/submit_stage0_seed_warm_start.sh \
  --partition gpu-l40s \
  --qos normal \
  --gpu-type l40s \
  --time 12:00:00
```

Native OpenPI Stage-0 smoke on SDRE:

```bash
scripts/bridge/run_stage0_openpi_libero_smoke.sh \
  --task-suite-name libero_spatial \
  --max-tasks 1 \
  --trials 1 \
  --max-steps 1 \
  --replan-steps 1 \
  --libero-gl-backend osmesa
```

Slurm submission for the same smoke:

```bash
scripts/slurm/submit_stage0_openpi_libero_smoke.sh \
  --task-suite-name libero_spatial \
  --max-tasks 1 \
  --trials 1 \
  --max-steps 1 \
  --replan-steps 1 \
  --libero-gl-backend osmesa \
  --partition gpu-l40s \
  --qos normal \
  --gpu-type l40s \
  --mem 128G \
  --time 02:00:00
```

OpenPI checkpoint conversion to the PyTorch format expected by RLinf:

```bash
scripts/openpi/convert_openpi_checkpoint_to_pytorch.sh \
  --checkpoint-dir /projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero \
  --output-path /projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch
```

Slurm submission for the same checkpoint conversion:

```bash
scripts/slurm/submit_openpi_checkpoint_conversion.sh \
  --checkpoint-dir /projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero \
  --output-path /projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch
```

Warm-start SAC smoke launcher against the Stage-0 demo manifest:

```bash
scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh --dry-run
```

Slurm submission for the same smoke, optionally chained after checkpoint conversion:

```bash
scripts/slurm/submit_stage0_seed_warm_start_smoke.sh \
  --dependency afterok:5395
```

Stage-E real-only one-round orchestrator on the split bridge:

```bash
scripts/stagee/run_stage0_real_only_round.sh \
  --task-suite libero_goal \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --count-legacy-contexts-as-online-budget \
  --candidate-count 4 \
  --selection-policy uniform \
  --libero-gl-backend osmesa \
  --max-env-steps 20
```

Slurm submission for the same Stage-E real-only smoke:

```bash
scripts/slurm/submit_stage0_real_only_round.sh \
  --task-suite libero_goal \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --count-legacy-contexts-as-online-budget \
  --candidate-count 4 \
  --selection-policy uniform \
  --libero-gl-backend osmesa \
  --max-env-steps 20
```

Balanced cross-task Stage-E budget submission from the Stage-0 manifest:

```bash
scripts/slurm/submit_stage0_real_only_budget.sh \
  --budget 25 \
  --seed 7
```

Notes:

- `openpi` and `pi-StepNFT` target the `StdEnv/2023 + CUDA 12.6 + Python 3.11.5` stack.
- `LIBERO` targets `StdEnv/2020 + Python 3.8.10`.
- GE-Sim is scaffolded on the `StdEnv/2023 + CUDA 12.6 + Python 3.10.13` path and still treated as a dependency-risk item until it installs cleanly.
- RLinf/WoVR is intentionally not bootstrapped yet because the current SDRE shell does not expose Docker or Apptainer.

## Current bring-up status

As of 2026-04-01:

- `openpi` installs in the `third_party/venvs/openpi` venv with local `uv` under `tools/uv`.
- `openpi` needs `UV_CACHE_DIR=/projects/p57098/euijin1/Caver/third_party/uv-cache` because the default cache placement is too small.
- `openpi` also needs `OPENPI_DATA_HOME=/projects/p57098/euijin1/Caver/third_party/openpi-cache` for checkpoint caching on SDRE.
- The SDRE headless install path for `openpi` currently excludes `pynput` and `evdev` during `uv sync`.
- `openpi` deeper training-path imports also need `pytest` present because upstream package imports leak test-only dependencies into runtime modules.
- `LIBERO` must override the cluster pip config with `PIP_CONFIG_FILE=/dev/null` and unset `PYTHONPATH`; otherwise SDRE wheelhouse constraints block the exact upstream pins.
- `LIBERO` also needs `TMPDIR` on `/projects` because `/tmp` on the login node is only 3 GB and is too small for the torch plus CUDA wheel unpack phase.
- `LIBERO` is configured non-interactively via `LIBERO_CONFIG_PATH=/projects/p57098/euijin1/Caver/third_party/config/libero`.
- `pi-StepNFT` upstream packaging was not editable-installable on SDRE as shipped; the local checkout now adds explicit setuptools package discovery in `third_party/src/pi-StepNFT/pyproject.toml`.
- `LIBERO` editable installation in the Python 3.11 `openpi` venv is unreliable under modern pip, so the policy-side runtime uses `PYTHONPATH` to expose the local `LIBERO` and `pi-StepNFT` source trees explicitly.
- `scripts/env/with_openpi_libero_eval.sh` is useful for OpenPI policy imports and websocket server startup, but the simulator still segfaults in that mixed interpreter on SDRE.
- The train-side embodied RL entrypoint now uses `scripts/env/with_openpi_pistepnft_libero_train.sh`, which exposes `openpi`, `pi-StepNFT`, and `LIBERO` together from the repaired `openpi` venv.
- That `openpi` venv now includes `hydra-core 1.3.2`, `ray 2.49.1+computecanada`, `accelerate`, and `tensorboard` because the upstream mix was not sufficient to import RLinf's embodied training entrypoint on Python `3.11`.
- The same train-side path also needs the upstream `transformers_replace` overlay for OpenPI PyTorch model construction; `scripts/openpi/install_transformers_replace.sh` now repairs that overlay automatically inside the active venv.
- The stable Stage 0 path is a split bridge: `scripts/env/with_openpi_libero_eval.sh` for the policy server and `scripts/env/with_libero_eval.sh` for the LIBERO simulator client.
- Repo-local bridge entrypoints now live under `scripts/bridge/`, with `openpi_policy_server.py` on the policy side and `libero_remote_eval.py` on the simulator side.
- Public `gs://openpi-assets/...` checkpoint downloads now fall back to anonymous gcsfs when `gsutil` is unavailable, and cached directories carry a completion marker to avoid reusing partial checkpoint trees.

## Smoke test summary

- `openpi`: `openpi`, `openpi_client`, `torch`, `jax`, `lerobot`, `openpi.training.checkpoints`, `openpi.shared.download`, and `openpi.training.config` import successfully.
- `LIBERO`: import succeeds and a minimal `OffScreenRenderEnv` can be constructed and reset when `LIBERO_CONFIG_PATH` is set.
- `pi-StepNFT`: `rlinf` and `rlinf.utils.omega_resolver` import successfully after installing `torch`, `omegaconf`, and `numpy`.
- `pi-StepNFT` openpi-adjacent imports now work in the `openpi` venv after light local import fixes, which is enough for websocket policy serving.
- Combined `openpi + RLinf + LIBERO` imports work in the `openpi` venv, but actual simulator startup still segfaults there on SDRE.
- The split websocket bridge is working: the OpenPI-side dummy server responds correctly to the LIBERO-side websocket client, and `scripts/bridge/run_libero_remote_eval.sh` is now the supported Stage 0 execution path.
- The manifest-aware Stage-0 harness is working in dummy-server mode:
  - wrapper: `scripts/stage0/run_stage0_partition_eval.sh`
  - seed-partition smoke artifact: `logs/runtime/stage0_partition_dummy.json`
  - context ledger artifact: `logs/runtime/stage0_partition_dummy.jsonl`
- The full native OpenPI Stage-0 seed collection launcher now exists:
  - submission wrapper: `scripts/slurm/submit_stage0_seed_warm_start.sh`
  - first live job: `5332`
  - live run directory: `runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T033646Z`
- The trace-authoritative rerun is also live:
  - job: `5333`
  - live run directory: `runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T043351Z`
  - live trace file: `runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T043351Z/results/stage0_seed_warm_start_chunks.jsonl`
- The `5333` rerun is not valid as the final warm-start trace:
  - its own OpenPI server failed to bind because port `8000` was already occupied by job `5332` on the same node
  - the client attached to the wrong server, then degraded into `ConnectionClosedError` once `5332` finished
  - treat that run as a concurrency-debug artifact, not the authoritative seed dataset
- The clean replacement rerun finished successfully:
  - job: `5334`
  - run directory: `runs/stage0__seed-warm-start__libero-stage0-seed__seed7__budget0__20260402T053122Z`
  - derived websocket port: `23334`
  - final state on `2026-04-02`: `COMPLETED`, exit code `0`, elapsed `05:36:10`
  - final counts: `120/120` contexts, `28` successes, `0` errors, `8456` chunk traces
- Stage-0 trace capture and demo conversion now exist:
  - trace-capable evaluator flag: `scripts/bridge/libero_remote_eval.py --transition-trace-path`
  - conversion wrapper: `scripts/stage0/convert_stage0_trace_to_rlinf_demo.sh`
  - smoke trace artifact: `logs/runtime/stage0_trace_smoke_chunks.jsonl`
  - smoke demo artifact: `logs/runtime/stage0_trace_smoke_demo.pt`
  - smoke demo summary: `logs/runtime/stage0_trace_smoke_demo.summary.json`
- The public `pi-StepNFT` demo target is now clearer:
  - `robot_demo` feeds `SACReplayBuffer`
  - the stock public NFT actor path still lacks native offline demo ingest
  - the current converter therefore emits SAC-compatible replay items
- The full authoritative `5334` trace is large enough to require sharded conversion:
  - trace file size: `71G`
  - the converter now supports `--output-mode sharded_manifest`
  - `SACReplayBuffer.create_from_demo(...)` now accepts the emitted sharded manifest and loads shards sequentially
  - sharded smoke artifact from the real `5334` trace: `logs/runtime/stage0_seed_warm_start_demo_smoke.manifest.json`
  - sharded smoke summary: `logs/runtime/stage0_seed_warm_start_demo_smoke.summary.json`
- The full sharded conversion completed on the CPU partition:
  - submission wrapper: `scripts/slurm/submit_stage0_seed_warm_start_conversion.sh`
  - job: `5392`
  - run directory: `runs/stage0__seed-demo-convert__libero-stage0-seed__seed7__budget0__20260402T203345Z`
  - output manifest target: `runs/stage0__seed-demo-convert__libero-stage0-seed__seed7__budget0__20260402T203345Z/results/stage0_seed_warm_start_demo.manifest.json`
  - final state on `2026-04-02`: `COMPLETED`, exit code `0`, elapsed `01:31:44`
  - final artifact shape: `8456` demo items across `67` shard files
- The Stage-D backend handoff from Stage C is now staged:
  - authoritative demo manifest: `runs/stage0__seed-demo-convert__libero-stage0-seed__seed7__budget0__20260402T203345Z/results/stage0_seed_warm_start_demo.manifest.json`
  - replay-item action shape: `(35,)`, matching RLinf's current `5 x 7` OpenPI-LIBERO expectation
  - first SAC smoke launcher: `scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh`
- RLinf still cannot train from the cached public checkpoint directly because the cache is JAX/Orbax only; it needs a PyTorch directory with `model.safetensors`.
- The first checkpoint-conversion attempt found one missing upstream runtime step:
  - job `5393` failed with `ValueError: transformers_replace is not installed correctly`
  - fix: `scripts/openpi/install_transformers_replace.sh` now copies the upstream overlay into the active `transformers` install before Stage-D model construction
- The corrected checkpoint-to-smoke chain is now active:
  - conversion wrapper: `scripts/slurm/submit_openpi_checkpoint_conversion.sh`
  - conversion job: `5395`
  - conversion run directory: `runs/staged__openpi-pytorch-convert__pi05_libero__seed0__budget0__20260403T041843Z`
  - conversion output target: `/projects/p57098/euijin1/Caver/third_party/openpi-cache/openpi-assets/checkpoints/pi05_libero_pytorch`
  - conversion final state: `COMPLETED`, exit code `0`, elapsed `00:01:19`
  - dependent smoke wrapper: `scripts/slurm/submit_stage0_seed_warm_start_smoke.sh`
  - smoke retries `5396` through `5399` only exposed Hydra config-shape issues
  - those fixes are now baked into `scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh`
  - smoke `5400` reached local Ray startup but hit the full `02:00:00` walltime and was cancelled by Slurm
  - the current debug-smoke defaults are now smaller and more observable:
    - demo manifest: `logs/runtime/stage0_seed_warm_start_demo_smoke.manifest.json` with `10` items
    - unbuffered Python output
    - `1` train env, `1` eval env, `5` rollout steps
    - `global_batch_size=2`, `micro_batch_size=1`
    - `+actor.model.openpi.pytorch_compile_mode=null`
  - smoke `5401` also timed out on `2026-04-03 04:07:33` America/Edmonton
  - `5401` produced no `results/rlinf_logs` tree and only flushed the same local Ray startup message
  - the smoke launcher now exports `CAVER_STAGE_DEBUG=1`
  - stage markers were added to:
    - `examples/embodiment/train_embodied_agent.py`
    - `rlinf.config.validate_cfg(...)` and `validate_embodied_cfg(...)`
    - `rlinf.scheduler.cluster.Cluster`
    - `rlinf.data.replay_buffer.SACReplayBuffer.create_from_demo(...)`
  - debug probe `5402` later timed out on `2026-04-03 07:17:03` America/Edmonton without any later marker
  - live `5402` markers show the startup reaches `ray.init(address='auto') failed, starting local Ray`, then Ray reports `Started a local Ray instance`, and no later RLinf marker appears; the blocker is therefore narrowed to local `ray.init(...)` return during RLinf cluster setup
  - a pure Ray follow-up is now tracked separately:
    - `5403` failed immediately due to a probe quoting bug
    - corrected probe `5404` run directory: `runs/staged__ray-init-probe__no-dashboard__seed1__budget0__20260403T180549Z`
    - `5404` disables the Ray dashboard and usage stats
    - `5404` later timed out on `2026-04-03 12:21:04` America/Edmonton with the same stop point: `[ray-probe] before ray.init()` plus `Started a local Ray instance.`
  - deep Ray instrumentation now exists:
    - `scripts/debug/ray_init_probe.py`
    - `scripts/debug/run_ray_connect_probe.sh`
  - that deeper probe chain established two SDRE-specific fixes:
    - keep `RAY_TMPDIR` short enough for AF_UNIX socket limits; wrappers now default it to `/projects/p57098/euijin1/ray/<jobid>`
    - constrain local `ray.init(...)` to the Slurm allocation instead of the full node; the successful proof run was `5409`, which completed after `00:00:19` once `num_cpus=8` and `num_gpus=1` were set explicitly
  - the RLinf fallback now mirrors that fix in `rlinf/scheduler/cluster/cluster.py`
  - smoke confirmation job `5410` proved the Ray fix and surfaced the next config constraint:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T184143Z`
    - node: `l40s-01`
    - final state: `FAILED` on `2026-04-03 12:42:05` America/Edmonton after `00:00:22`
    - `5410` got through RLinf cluster bring-up and Ray manager launch
    - new failure: `env.train.max_steps_per_rollout_epoch` must be divisible by `actor.model.num_action_chunks`
    - root cause: the extra-minimal `rollout_steps=1` debug override was invalid for the `libero_goal_ppo_openpi_pi05` config, which uses `num_action_chunks: 5`
  - the smoke launcher now checks that earlier and exits before GPU submission if the override is invalid
  - corrected smoke confirmation job `5411` then exposed the next packaging issue:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T184435Z`
    - node: `l40s-01`
    - final state: `FAILED` on `2026-04-03 12:44:59` America/Edmonton after `00:00:23`
    - `5411` got through RLinf cluster bring-up, placement validation, actor/rollout/env group launch, and resolved-config dump
    - new failure: `ModuleNotFoundError: No module named 'rlinf.data.datasets.math'`
    - fix: `rlinf/data/datasets/__init__.py` now treats `MathDataset` as optional and only raises for `data.type == "math"`
    - local validation under the train wrapper now confirms `create_rl_dataset(... type='robot_demo' ...)` returns `SACReplayBuffer` and loads the `10`-item smoke manifest
  - corrected smoke confirmation job `5412` then ran:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T184644Z`
    - node: `l40s-01`
    - final state: `FAILED` on `2026-04-03 12:48:10` America/Edmonton after `00:01:26`
    - `5412` got through replay-buffer load and into `runner.init_workers()`
    - real worker failure: OpenPI norm stats were missing from the converted checkpoint
    - fixes now applied:
      - the conversion wrapper copies `assets/` into the PyTorch checkpoint directory
      - RLinf's OpenPI loader now prefers `checkpoint_dir/assets` when loading norm stats
      - the current `pi05_libero_pytorch` checkpoint has been repaired in place with `assets/physical-intelligence/libero/norm_stats.json`
      - the SIGUSR1 cleanup path now treats `list_actors(...)` as best-effort so dashboard-less local Ray runs do not hide the original worker error
  - corrected smoke confirmation job `5413` then ran:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T185123Z`
    - node: `l40s-01`
    - final state: `FAILED` on `2026-04-03 12:53:33` America/Edmonton after `00:02:10`
    - `5413` got through norm-stat loading and actor model initialization
    - real worker failure: SAC optimizer construction only recognized `q_head`, but OpenPI exposes critic parameters under `value_head`
    - fix: `fsdp_sac_policy_worker.py` now treats both `q_head` and `value_head` as critic-side for optimizer and target-model handling
  - corrected smoke confirmation job `5414` then ran:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T185558Z`
    - node: `l40s-01`
    - final state: `FAILED` on `2026-04-03 12:59:03` America/Edmonton after `00:03:05`
    - `5414` got through actor and rollout setup and then failed in env import with `ModuleNotFoundError: No module named 'tensorflow'`
    - fix: `rlinf/envs/utils.py` now makes TensorFlow optional and only requires it when the TensorFlow-based crop helpers are called
    - local validation under the train wrapper now confirms `rlinf.envs.utils` imports without TensorFlow
  - corrected smoke confirmation job `5415` then ran:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T190019Z`
    - node: `l40s-01`
    - start time: `2026-04-03 13:00:19` America/Edmonton
    - walltime cap: `00:45:00`
    - ETA upper bound from Slurm walltime: `2026-04-03 13:45:19` America/Edmonton
  - `5415` failed on `2026-04-03 13:03:21` America/Edmonton after `00:03:02`
    - it got through env import and failed during LIBERO init-state loading with `_pickle.UnpicklingError`
    - root cause: PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`, which rejects LIBERO's trusted local init-state assets
    - fixes:
      - `third_party/src/LIBERO/libero/libero/benchmark/__init__.py` now loads init states with `weights_only=False`
      - the same init-state fix is applied in `third_party/src/LIBERO/libero/lifelong/metric.py` and `third_party/src/LIBERO/libero/lifelong/evaluate.py`
      - `third_party/src/LIBERO/libero/lifelong/utils.py` now also loads trusted local checkpoints with `weights_only=False`
      - local validation under the train wrapper now confirms the patched LIBERO benchmark path returns the real init-state array
  - corrected smoke confirmation job `5416` then showed the embedded-train-runtime LIBERO limit:
    - run directory: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T190505Z`
    - node: `l40s-01`
    - final state: `FAILED` on `2026-04-03 13:07:31` America/Edmonton after `00:02:24`
    - it got through replay ingest, actor init, rollout init, and patched LIBERO init-state loading
    - the remaining failure was train-runtime offscreen renderer construction inside the Python `3.11` stack
  - renderer retries then closed that branch:
    - `5417`: forced `MUJOCO_GL=osmesa`, failed with a segmentation fault during `OffScreenRenderEnv(...)`
    - `5418`: forced `MUJOCO_GL=egl`, failed during EGL device/display initialization
    - operational conclusion: on SDRE, keep online LIBERO execution on the split bridge instead of embedding it inside the Python `3.11` RLinf runtime
  - the warm-start smoke path then pivoted to the actual minimum backend objective: replay ingest plus one SAC update with no live env workers
    - `train_embodied_agent.py` now supports `runner.offline_demo_only=true`
    - `rlinf/runners/embodied_runner.py` now supports that demo-only execution mode
    - `rlinf/workers/actor/fsdp_sac_policy_worker.py` now allows demo-only SAC training when the demo buffer is populated
    - `scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh` now defaults to that mode and exposes `--online-rollout` for the old path
  - demo-only confirmation chain:
    - `5419`: failed on missing OpenPI RL `ForwardType.SAC` / `ForwardType.SAC_Q` support
    - `5420`: after adding `sac_forward(...)` / `sac_q_forward(...)`, failed because pi05 `value_after_vlm=true` still routed critic values through a no-grad path
    - `5421`: after forcing suffix-conditioned critic values, failed because suffix critic features are `1024`-dimensional but the existing pi05 `value_head` expected `2048`
    - `5422`: authoritative success, run directory `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T193632Z`, final state `COMPLETED` on `2026-04-03 13:39:10` America/Edmonton after `00:02:37`
  - `5422` proves the first Stage-D backend floor is now working on SDRE:
    - replay manifest ingest succeeds
    - actor initialization succeeds
    - critic initialization succeeds
    - one real SAC update completes from the Stage-0 demo artifact
    - logged metrics include `train/sac/critic_loss=0.000339`, `train/sac/actor_loss=0.0122`, and `train/sac/alpha=0.01`
    - current artifact path: `runs/staged__seed-sac-smoke__libero_goal-task-0__seed7__budget0__20260403T193632Z/results/rlinf_logs/replay_buffer_0.pkl`
- Stage-E real-only bring-up is now also validated:
  - new runner: `scripts/stagee/run_stage0_real_only_round.sh`
  - new submitter: `scripts/slurm/submit_stage0_real_only_round.sh`
  - `5423` failed because the new runner did not force the known-good `MUJOCO_GL=osmesa` backend and the LIBERO eval env fell back to EGL
  - `5424` failed because the earlier `weights_only=False` LIBERO patch was not backward-compatible with the older Python `3.8` eval-side torch
  - fix after `5424`: `third_party/src/LIBERO/libero/libero/torch_compat.py` now loads trusted local assets across both old and new torch releases
  - `5425` is the authoritative Stage-E smoke:
    - run directory: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget1__20260403T201608Z`
    - final state: `COMPLETED` on `2026-04-03 14:20:14` America/Edmonton after `00:04:06`
    - workflow summary: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget1__20260403T201608Z/results/real_only_round_summary.json`
    - online summary: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget1__20260403T201608Z/results/real_only_online_eval.json`
    - replay demo manifest: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget1__20260403T201608Z/results/real_only_round_demo.manifest.json`
    - replay snapshot after training: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget1__20260403T201608Z/results/rlinf_logs/replay_buffer_0.pkl`
  - `5425` proves the first real-only Stage-E loop now works:
    - `1` split-bridge episode completed
    - selector logging captured `candidate_count=4`, `selection_policy=uniform`, selected indices `[3, 2, 2, 3]`, and exact propensities `[0.25, 0.25, 0.25, 0.25]`
    - conversion produced `4` demo items from `20` primitive steps
    - the backend update completed and `train_embodied_agent` reached `runner.run completed`
  - legacy Stage-E real-only rounds now have an explicit online-budget ledger mode:
    - evaluator flag: `--count-legacy-contexts-as-online-budget`
    - the Stage-E runner and submitter enable that mode by default for legacy task selection
  - the first `25`-context one-task scale-up is now complete:
    - job: `5426`
    - run directory: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget25__20260403T202918Z`
    - start time: `2026-04-03 14:29:18` America/Edmonton
    - final state: `COMPLETED` on `2026-04-03 14:40:41` America/Edmonton after `00:11:23`
    - online budget validation: `online_training_budget_units=25` and all per-context budget records use `budget_domain=legacy_online`
    - workflow summary: `runs/stagee__real-only-round__libero_goal-task-0__seed7__budget25__20260403T202918Z/results/real_only_round_summary.json`
  - Stage-E now also has a balanced manifest path for the actual five-family Stage-0 task set:
    - builder: `scripts/stagee/build_stage0_balanced_manifest.py`
    - submitter: `scripts/slurm/submit_stage0_real_only_budget.sh`
  - the first cross-task `budget=25` run is now complete:
    - job: `5436`
    - run directory: `runs/stagee__real-only-round__manifest-t_train_s0-all__seed7__budget25__20260404T002000Z`
    - selection manifest: `logs/runtime/stagee_manifests/stagee_real_only__t_train_s0__budget25__offset0__seed7__all__20260404T002000Z.json`
    - final state: `COMPLETED` on `2026-04-03 18:31:00` America/Edmonton after `00:10:59`
    - validated workload: `25` total contexts, balanced as `5` contexts per family across all five Stage-0 proxy families
    - online budget validation: `25` online-training budget units with `budget_domain=online_train`
    - workflow summary: `runs/stagee__real-only-round__manifest-t_train_s0-all__seed7__budget25__20260404T002000Z/results/real_only_round_summary.json`
  - the full balanced seed-`7` ladder is now validated:
    - `5437`: `budget=50`, `COMPLETED` on `2026-04-03 20:12:13` America/Edmonton after `00:17:51`
    - `5438`: `budget=100`, `COMPLETED` on `2026-04-03 20:43:10` America/Edmonton after `00:30:57`
    - `5439`: `budget=200`, `COMPLETED` on `2026-04-03 21:41:13` America/Edmonton after `00:58:03`
    - validated counts by budget:
      - `50`: `50` contexts, `200` chunk traces, `200` demo items, `1000` primitive steps
      - `100`: `100` contexts, `400` chunk traces, `400` demo items, `2000` primitive steps
      - `200`: `200` contexts, `800` chunk traces, `800` demo items, `4000` primitive steps
    - all four seed-`7` balanced runs (`5436` to `5439`) reached `train_embodied_agent: runner.run completed`
  - Stage-E is now using the provisional three-seed working set `{7, 13, 29}` until the proposal text pins exact numeric seed IDs
  - the next seed ladder is now live on seed `13`:
    - authoritative chain: `5440 -> 5442 -> 5444 -> 5445`
    - final states:
      - `5440`: `budget=25`, `COMPLETED` on `2026-04-03 21:57:57` America/Edmonton after `00:10:38`
      - `5442`: `budget=50`, `COMPLETED` on `2026-04-03 22:15:27` America/Edmonton after `00:17:30`
      - `5444`: `budget=100`, `COMPLETED` on `2026-04-03 22:46:25` America/Edmonton after `00:30:58`
      - `5445`: `budget=200`, `COMPLETED` on `2026-04-03 23:44:35` America/Edmonton after `00:58:10`
    - validated counts by budget:
      - `25`: `25` contexts, `100` chunk traces, `100` demo items, `500` primitive steps
      - `50`: `50` contexts, `200` chunk traces, `200` demo items, `1000` primitive steps
      - `100`: `100` contexts, `400` chunk traces, `400` demo items, `2000` primitive steps
      - `200`: `200` contexts, `800` chunk traces, `800` demo items, `4000` primitive steps
    - all four seed-`13` balanced runs reached `train_embodied_agent: runner.run completed`
  - the final provisional seed ladder is now live on seed `29`:
    - authoritative chain: `5446 -> 5447 -> 5448 -> 5449`
    - final states:
      - `5446`: `budget=25`, `COMPLETED` on `2026-04-04 00:02:03` America/Edmonton after `00:10:41`
      - `5447`: `budget=50`, `COMPLETED` on `2026-04-04 00:19:32` America/Edmonton after `00:17:29`
      - `5448`: `budget=100`, `COMPLETED` on `2026-04-04 00:50:29` America/Edmonton after `00:30:57`
      - `5449`: `budget=200`, `COMPLETED` on `2026-04-04 01:48:52` America/Edmonton after `00:58:23`
    - validated counts by budget:
      - `25`: `25` contexts, `100` chunk traces, `100` demo items, `500` primitive steps
      - `50`: `50` contexts, `200` chunk traces, `200` demo items, `1000` primitive steps
      - `100`: `100` contexts, `400` chunk traces, `400` demo items, `2000` primitive steps
      - `200`: `200` contexts, `800` chunk traces, `800` demo items, `4000` primitive steps
    - all four seed-`29` balanced runs reached `train_embodied_agent: runner.run completed`
    - the provisional three-seed real-only cross-task ladder `{7, 13, 29}` is now fully validated
  - the first CAVER-specific trace upgrade is now in place:
    - `scripts/bridge/libero_remote_eval.py` now emits `caver_stage0_chunk_trace_v2`
    - each trace record now includes the full sampled candidate bank, the exact candidate propensity vector, and safety-mask placeholder fields for the later selector/admission layer
    - launch note: a mistaken dependency reference caused one rejected submission and a cancelled placeholder `5443`; the chain above is authoritative
  - the first end-to-end CAVER scaffold smoke is now validated:
    - new files:
      - `scripts/stagee/build_caver_round_artifacts.py`
      - `scripts/stagee/run_stage0_caver_round.sh`
      - `scripts/slurm/submit_stage0_caver_round.sh`
    - pre-submit fix:
      - both new shell wrappers initially lacked execute bits and were corrected before launch
    - authoritative job:
      - `5480`
      - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z`
      - final state: `COMPLETED` on `2026-04-04 02:49:11` America/Edmonton after `00:16:09`
    - workload:
      - derived manifest: `logs/runtime/caver_smoke_balanced_manifest.json`
      - selected family: `block_to_tray_proxy`
      - backend task suite: `libero_90`
      - backend task ids: `57,58,59`
    - validated outputs:
      - online summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z/results/caver_online_eval.json`
      - selector summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z/results/caver_selector_summary.json`
      - admission summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z/results/caver_admission_summary.json`
      - replay demo manifest: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z/results/caver_round_demo.manifest.json`
      - replay snapshot: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z/results/rlinf_logs/replay_buffer_0.pkl`
      - workflow summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T083302Z/results/caver_round_summary.json`
    - validated counts:
      - `5` online contexts
      - `5/5` successes
      - `159` `caver_stage0_chunk_trace_v2` records
      - `5` admitted contexts and `159` admitted trace records
      - `159` replay demo items, `788` primitive steps, and `2` shards
      - backend completion marker: `train_embodied_agent: runner.run completed`
  - the first balanced cross-family CAVER budget run is now validated:
    - new file:
      - `scripts/slurm/submit_stage0_caver_budget.sh`
    - authoritative job:
      - `5481`
      - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z`
      - final state: `COMPLETED` on `2026-04-04 03:08:58` America/Edmonton after `00:12:58`
    - manifest pair:
      - `logs/runtime/stagee_manifests/stagee_caver__t_train_s0__budget25__offset0__seed7__all__20260404T085600Z.json`
      - `logs/runtime/stagee_manifests/stagee_caver__t_train_s0__budget25__offset0__seed7__all__20260404T085600Z.summary.json`
    - validated workload:
      - `25` balanced online contexts from `T_train_S0`
      - `5` contexts per family across all five Stage-0 proxy families
      - backend task suite `libero_90` with task ids `6,7,11,16,17,46,47,48,57,58,59,63,73,74,75`
    - validated outputs:
      - online summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z/results/caver_online_eval.json`
      - selector summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z/results/caver_selector_summary.json`
      - admission summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z/results/caver_admission_summary.json`
      - replay demo manifest: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z/results/caver_round_demo.manifest.json`
      - replay snapshot: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z/results/rlinf_logs/replay_buffer_0.pkl`
      - workflow summary: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T085600Z/results/caver_round_summary.json`
    - validated counts:
      - `25` online contexts
      - `0` successes
      - `100` `caver_stage0_chunk_trace_v2` records
      - `25` admitted contexts and `100` admitted trace records
      - `100` replay demo items, `500` primitive steps, and `1` shard
      - backend completion marker: `train_embodied_agent: runner.run completed`
  - the first nontrivial CAVER selector/admission path is now validated on SDRE:
    - new file:
      - `scripts/stagee/caver_heuristic.py`
    - runtime changes:
      - `scripts/bridge/libero_remote_eval.py` now supports `caver_heuristic`
      - `scripts/stagee/build_caver_round_artifacts.py` now supports `success_lcb_v1`
      - `scripts/stagee/run_stage0_caver_round.sh` now skips conversion/training cleanly when admission returns zero contexts
    - first launch bug:
      - `5482` failed because `compute_selector_decision(...)` referenced `history` before defining it
      - that defect is fixed
    - authoritative post-fix smoke:
      - `5483`
      - run directory: `runs/stagee__caver-round__manifest-t_train_s0-block_to_tray_proxy__seed7__budget5__20260404T094447Z`
      - final state: `COMPLETED` on `2026-04-04 03:46:58` America/Edmonton after `00:02:11`
      - validated counts:
        - `5` online contexts
        - `20` `caver_stage0_chunk_trace_v2` records
        - `0` successes and `0` per-context errors
        - selector summary confirms non-uniform exact propensities under `caver_heuristic`
        - admission summary confirms `0` admitted contexts and an intentional zero-admission skip path
    - first balanced rerun on the new selector:
      - `5484`
      - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T094837Z`
      - final state: `COMPLETED` on `2026-04-04 03:56:11` America/Edmonton after `00:07:34`
      - validated counts:
        - `25` online contexts
        - `0` successes and `0` per-context errors
        - `100` `caver_stage0_chunk_trace_v2` records
        - `0` admitted contexts and `0` admitted trace records
        - replay conversion skipped intentionally
        - backend training skipped intentionally
    - current takeaway:
      - the nontrivial selector/admission plumbing is now stable
      - the current frozen action-space heuristic is too weak to feed the backend, so the next implementation step should fit the proxy-value input on Stage-0 seed/warm-start data
  - the fitted Stage-0 value-proxy path is now staged locally:
    - dataset builder: `scripts/stagee/build_stage0_value_proxy_dataset.py`
    - trainer: `scripts/stagee/train_stage0_value_proxy.py`
    - inference helpers: `scripts/stagee/stage0_value_proxy.py`
    - compact dataset: `metadata/stage0/value_proxy/stage0_seed_context_success_dataset.jsonl`
    - fitted model: `metadata/stage0/value_proxy/stage0_context_success_logreg_v1.json`
    - training summary: `metadata/stage0/value_proxy/stage0_context_success_logreg_v1.summary.json`
    - held-out validation metrics: `accuracy=0.7665`, `weighted_log_loss=0.5640`, `brier=0.1867`
  - learned-proxy live smoke:
    - job `5485`
    - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget5__20260404T101822Z`
    - start time: `2026-04-04 04:18:22` America/Edmonton
    - node: `l40s-01`
    - upper-bound finish from Slurm walltime: `2026-04-04 05:18:22` America/Edmonton
    - selector mode: `fitted_stage0_value_softmax_v1`
    - model path: `metadata/stage0/value_proxy/stage0_context_success_logreg_v1.json`
  - `5485` completed cleanly on `2026-04-04 04:20:45` America/Edmonton after `00:02:23`
    - `5` contexts, `20` chunk traces, `0` successes, `0` per-context errors
    - selector artifacts now carry `value_proxy_model_id=stage0_context_success_logreg_v1`
  - `5486` is the authoritative learned-proxy `budget=25` rerun:
    - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T102122Z`
    - final state: `COMPLETED` on `2026-04-04 04:28:51` America/Edmonton after `00:07:29`
    - balanced workload: `25` contexts, `5` per Stage-0 proxy family
    - validated outcomes:
      - `25/25` contexts completed
      - `0` successes and `0` per-context errors
      - `100` chunk traces
      - selector mode `fitted_stage0_value_softmax_v1`
      - value-proxy model id `stage0_context_success_logreg_v1`
      - `0` admitted contexts and `0` admitted trace records
      - replay conversion and backend training skipped intentionally
    - comparison to heuristic baseline `5484`: no verified success gain and no admitted-data gain at `budget=25`
  - progress-aware target revision is now available:
    - model: `metadata/stage0/value_proxy/stage0_context_success_progress_sq_logreg_v1.json`
    - summary: `metadata/stage0/value_proxy/stage0_context_success_progress_sq_logreg_v1.summary.json`
    - held-out metrics: `accuracy=0.8575`, `weighted_log_loss=0.4799`, `brier=0.1203`
  - live validation of that progress-aware model:
    - `5487` smoke completed on `2026-04-04 05:37:24` America/Edmonton after `00:02:28`
      - `5` contexts, `20` chunk traces, `0` successes, `0` admitted contexts
    - `5488` balanced rerun completed on `2026-04-04 05:45:26` America/Edmonton after `00:07:36`
      - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T113750Z`
      - selector model id `stage0_context_success_progress_sq_logreg_v1`
      - `25/25` contexts completed, `0` successes, `0` admitted contexts, `0` admitted trace records
      - replay conversion and backend training skipped intentionally
  - additional proxy experiments:
    - `scripts/stagee/stage0_value_proxy.py` and `scripts/stagee/train_stage0_value_proxy.py` now support optional squared-progress and family-progress interaction features
    - local sweep dir: `logs/runtime/value_proxy_sweep_20260404T0550`
    - the family-curve variants were worse than the simpler progress-aware model on held-out loss
    - best offline sweep artifact:
      - `metadata/stage0/value_proxy/stage0_context_success_progress_sq_logreg_v2.json`
      - `metadata/stage0/value_proxy/stage0_context_success_progress_sq_logreg_v2.summary.json`
      - held-out metrics: `accuracy=0.8606`, `weighted_log_loss=0.4780`, `brier=0.1192`
  - `5489` validated that regularized best-offline model online:
    - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T115303Z`
    - final state: `COMPLETED` on `2026-04-04 06:00:32` America/Edmonton after `00:07:28`
    - selector model id `stage0_context_success_progress_sq_logreg_v2`
    - outcomes:
      - `25/25` contexts completed
      - `0` successes and `0` per-context errors
      - `100` chunk traces
      - `0` admitted contexts and `0` admitted trace records
      - replay conversion and backend training skipped intentionally
    - comparison to `5488`: no verified online gain from the regularization-only revision
  - chunk-success selector follow-up:
    - `scripts/stagee/stage0_value_proxy.py` now supports base-feature-by-progress interaction blocks
    - `scripts/stagee/train_stage0_value_proxy.py` can train those interaction features directly against `chunk_success_label`
    - local sweep dir: `logs/runtime/value_proxy_chunk_success_sweep_20260404T0620`
    - authoritative model:
      - `metadata/stage0/value_proxy/stage0_chunk_success_logreg_v2.json`
      - `metadata/stage0/value_proxy/stage0_chunk_success_logreg_v2.summary.json`
      - held-out metrics: `accuracy=0.8121`, `weighted_log_loss=0.4705`, `brier=0.1487`
  - `5490` smoke completed on `2026-04-04 06:28:08` America/Edmonton after `00:02:26`
    - `5` contexts, `20` policy queries, `0` successes, `0` admitted contexts
    - selector artifacts now carry `value_proxy_model_id=stage0_chunk_success_logreg_v2`
  - `5491` validated that chunk-success model on the balanced `budget=25` round:
    - run directory: `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T122820Z`
    - final state: `COMPLETED` on `2026-04-04 06:35:54` America/Edmonton after `00:07:34`
    - outcomes:
      - `25/25` contexts completed
      - `0` successes and `0` per-context errors
      - `100` chunk traces
      - selector mode `fitted_stage0_value_softmax_v1`
      - value-proxy model id `stage0_chunk_success_logreg_v2`
      - `0` admitted contexts and `0` admitted trace records
      - replay conversion and backend training skipped intentionally
    - comparison to `5489`: no verified online gain from the first chunk-level objective either
- On `2026-04-04`, Stage E uncovered a hidden budget-submitter cap:
  - `scripts/slurm/submit_stage0_real_only_budget.sh`
  - `scripts/slurm/submit_stage0_caver_budget.sh`
  - both were defaulting `--max-env-steps 20`
- Because of that, the earlier balanced cross-family ladders should now be treated as smoke / orchestration evidence, not as authoritative Stage-E science:
  - real-only `5436` to `5449`
  - CAVER `5481`, `5484`, `5486`, `5488`, `5489`, `5491`
- The corrected native-horizon rerun grid is now authoritative:
  - the submitter defaults now defer to the suite-native LIBERO horizon
  - live seed-`7` budget-`25` reruns:
    - real-only `5499`, run directory `runs/stagee__real-only-round__manifest-t_train_s0-all__seed7__budget25__20260404T185442Z`
    - CAVER `5500`, run directory `runs/stagee__caver-round__manifest-t_train_s0-all__seed7__budget25__20260404T185442Z`
    - both live logs confirm `max_steps=400`
    - live outcome signal:
      - `5499` first four completed contexts all reached `success=True`
      - `5500` first three completed contexts all reached `success=True`
  - queued corrected seed-`7` ladder:
    - real-only: `5505`, `5501`, `5504`
    - CAVER: `5503`, `5502`, `5506`
  - launched corrected seed-`13` ladder:
    - real-only: `5507`, `5508`, `5509`, `5510`
    - CAVER: `5511`, `5512`, `5513`, `5514`
  - launched corrected seed-`29` ladder:
    - real-only: `5515`, `5516`, `5517`, `5518`
    - CAVER: `5519`, `5520`, `5521`, `5522`
- On `2026-04-04 13:37` America/Edmonton, the queued dependency descendants were cancelled and replaced with an independent parallel grid because `gpu-l40s` still had substantial idle capacity:
  - cancelled placeholders:
    - `5501`, `5502`, `5503`, `5504`, `5505`, `5506`
    - `5508`, `5509`, `5510`, `5512`, `5513`, `5514`
    - `5516`, `5517`, `5518`, `5520`, `5521`, `5522`
  - replacement independent real-only jobs:
    - seed `7`: `5523` (`50`), `5525` (`100`), `5527` (`200`)
    - seed `13`: `5529` (`50`), `5531` (`100`), `5533` (`200`)
    - seed `29`: `5535` (`50`), `5537` (`100`), `5539` (`200`)
  - replacement independent CAVER jobs:
    - seed `7`: `5524` (`50`), `5526` (`100`), `5528` (`200`)
    - seed `13`: `5530` (`50`), `5532` (`100`), `5534` (`200`)
    - seed `29`: `5536` (`50`), `5538` (`100`), `5540` (`200`)
  - preserved method settings:
    - real-only experiment name: `stage0_real_only_budget_native_horizon`
    - CAVER experiment name: `stage0_caver_budget_native_horizon`
    - CAVER selector: `fitted_stage0_value_softmax_v1`
    - CAVER admission: `success_lcb_v1`
    - CAVER value-proxy model: `metadata/stage0/value_proxy/stage0_chunk_success_logreg_v2.json`
  - fresh-job health checks:
    - `5523`, `5524`, `5527`, and `5528` all reached `connected to policy server`, `starting LIBERO evaluation: mode=manifest`, and `max_steps=400`
    - live websocket ports from stderr: `23523`, `23524`, `23527`, `23528`
    - this confirms the earlier native-horizon fix held under the parallel relaunch
  - queue snapshot checked on `2026-04-04 13:39:02` America/Edmonton:
    - running corrected `budget=25` jobs: `5499`, `5500`, `5507`, `5511`, `5515`, `5519`
    - running corrected independent jobs: `5523`, `5524`, `5525`, `5526`, `5527`, `5528`, `5529`, `5530`, `5531`, `5532`, `5533`, `5534`, `5535`, `5536`, `5537`
    - pending at that check: `5538`, `5539`, `5540`
    - longest active upper-bound finish times from Slurm:
      - `5528`: `2026-04-05 00:07:05` America/Edmonton
      - `5534`: `2026-04-05 00:07:09` America/Edmonton
  - Stage-E result collation helper:
    - script: `scripts/stagee/summarize_stagee_grid.py`
    - latest snapshot JSON: `logs/runtime/stagee_native_horizon_grid_snapshot.json`
    - latest snapshot Markdown table: `logs/runtime/stagee_native_horizon_grid_snapshot.md`
- On `2026-04-04 19:36:55` America/Edmonton, Stage E added a resume-only postprocess path for failed native-horizon cells:
  - `scripts/stagee/run_stage0_real_only_round.sh` and `scripts/stagee/run_stage0_caver_round.sh` now support `--skip-online`
  - `scripts/stagee/build_caver_round_artifacts.py` now streams trace JSONLs instead of loading them fully into memory
  - admission aggregation is shared through `scripts/stagee/caver_heuristic.py`
- Active salvage jobs launched against the original failed result directories:
  - real-only:
    - `5564` seed `7`, budget `50`, upper-bound finish `2026-04-04 23:31:34` America/Edmonton
    - `5566` seed `13`, budget `50`, upper-bound finish `2026-04-04 23:33:53` America/Edmonton
    - `5568` seed `29`, budget `50`, upper-bound finish `2026-04-04 23:33:54` America/Edmonton
  - CAVER:
    - `5563` seed `7`, budget `50`, upper-bound finish `2026-04-05 01:31:34` America/Edmonton
    - `5567` seed `13`, budget `50`, upper-bound finish `2026-04-05 01:33:54` America/Edmonton
    - `5569` seed `29`, budget `50`, upper-bound finish `2026-04-05 01:33:54` America/Edmonton
    - `5571` seed `7`, budget `100`, upper-bound finish `2026-04-05 03:33:54` America/Edmonton
    - `5570` seed `13`, budget `100`, upper-bound finish `2026-04-05 03:33:54` America/Edmonton
- Current proof that the salvage fix is real:
  - the resumed real-only job `5564` is already writing shard files under `real_only_round_demo.manifest_shards/`, so the old timeout case resumed past the failure point
  - the resumed CAVER job `5563` is executing the old OOM section at only `MaxRSS=9364984K` by `sstat`, which confirms the streamed builder removed the memory blow-up
- On `2026-04-05 01:07` America/Edmonton, the Stage-E backend path was tightened again for the no-checkpoint RLinf mode:
  - `scripts/pistepnft/run_stage0_seed_warm_start_smoke.sh` now writes `training_completed.marker` and `training_completed.json` when backend training exits cleanly
  - `scripts/stagee/run_stage0_real_only_round.sh`, `scripts/stagee/run_stage0_caver_round.sh`, and `scripts/stagee/summarize_stagee_grid.py` now treat that marker as a successful backend completion signal
  - this was needed because the repaired space-safe path disables RLinf checkpoint saves with `runner.save_interval=-1`, so `replay_buffer_0.pkl` is no longer guaranteed to exist
- Verified completed salvage jobs after that marker fix:
  - `5576` real-only seed `7`, budget `100`: `COMPLETED` on `2026-04-05 00:56:45` America/Edmonton
  - `5577` CAVER seed `7`, budget `50`: `COMPLETED` on `2026-04-05 00:56:02` America/Edmonton
  - both logs reached `train_embodied_agent: runner.run completed`
  - marker files were backfilled into their original `results/rlinf_logs/` directories so the Stage-E grid snapshot now reflects them as completed
- One submission-path regression was also found and fixed immediately:
  - direct `sbatch script.sh ...` resumes `5578` through `5584` failed because Slurm staged the runner scripts into `/var/spool/slurmd/...`, which broke their relative `source ../common.sh`
  - corrected resume wave `5591` through `5597` now runs through `/bin/bash /uhome/euijin1/projects/p57098/euijin1/Caver/scripts/stagee/...`
- Current live jobs as checked on `2026-04-05 01:07` America/Edmonton:
  - resumed original result directories:
    - `5591` real-only seed `13`, budget `100`
    - `5592` real-only seed `29`, budget `100`
    - `5593` CAVER seed `13`, budget `50`
    - `5594` CAVER seed `29`, budget `50`
    - `5595` CAVER seed `7`, budget `100`
    - `5596` CAVER seed `13`, budget `100`
    - `5597` CAVER seed `29`, budget `100`
  - fresh full reruns for incomplete `budget=200` cells:
    - `5585`, `5586`, `5587` real-only
    - `5588`, `5589`, `5590` CAVER
  - live proof points:
    - `5591` and `5592` are already loading the large sharded `budget=100` demo manifests from the original failed result directories
    - `5593` is reusing selector/admission artifacts in-place from the original failed CAVER `budget=50` directory
    - `5585` has started `mode=manifest contexts=200`
    - `5588` has already advanced to context `2/200` after logging the first context as `success=True`
- Rechecked on `2026-04-05 07:34` America/Edmonton:
  - completed repaired resumes:
    - `5591` real-only seed `13`, budget `100`
    - `5592` real-only seed `29`, budget `100`
    - `5593` CAVER seed `13`, budget `50`
    - `5594` CAVER seed `29`, budget `50`
    - `5597` CAVER seed `29`, budget `100`
  - two more CAVER `budget=100` resumes still needed intervention:
    - `5595` failed because its Ray owner node died
    - `5596` timed out after reaching backend training
  - best-effort response:
    - resubmitted `5595` as `5598` with `04:00:00` walltime and `--exclude=l40s-04`
    - resubmitted `5596` as `5599` with `04:00:00` walltime and `--exclude=l40s-04`
    - both retries are already running through the repaired absolute-path resume flow
  - live `budget=200` risk check:
    - `5585` through `5590` are all still `RUNNING`
    - `5585` and `5588` were at context `135/200` at the latest log check
    - attempted in-place walltime extension to `12:00:00` for all six live `budget=200` jobs
    - Slurm returned `Access/permission denied` on every `scontrol update`, so the account cannot extend running job limits directly
- Rechecked on `2026-04-05 21:29` America/Edmonton:
  - the repo path `/projects/p57098/euijin1/Caver` is writable again
  - the temporary `budget=200` recovery tooling was moved into the repo:
    - `scripts/stagee/finalize_stagee_budget200_cell.py`
    - `scripts/slurm/submit_stagee_budget200_finalize.sh`
  - those finalized `budget=200` runs are stored on `/rdss/p57098/euijin1/caver/stagee_budget200_finalized/` to avoid duplicating the giant merged traces under `/projects`
  - matching symlinked run dirs are created back under `/projects/p57098/euijin1/Caver/runs/` so `scripts/stagee/summarize_stagee_grid.py` still sees the latest Stage-E cells without modification
  - live finalize jobs:
    - `5612` real-only seed `13`
    - `5613` CAVER seed `13`
    - `5614` real-only seed `29`
    - `5615` real-only seed `7`
    - `5616` CAVER seed `7`
    - `5617` CAVER seed `29`
  - current proof points:
    - `5612` merged `real_only_online_chunks.jsonl` is already larger than `1.5G`
    - `5613` merged `caver_online_chunks.jsonl` is already larger than `1.1G`
    - `5615`, `5616`, and `5617` have all started writing nonzero merged traces
    - the finalize path now forces `CAVER_DEFAULT_RUNTIME_LOG_ROOT=/rdss/p57098/euijin1/caver/runtime_logs`, so it is isolated from the earlier `/projects` runtime-log exhaustion
- Rechecked on `2026-04-05 21:44` America/Edmonton:
  - cancelled the first finalize probe wave `5612` through `5617`
  - reason:
    - those jobs had already loaded the old materialized-merge process image, and their trace-copy rate was too low to be the final Stage-E closure path
  - corrected recovery path:
    - `scripts/stagee/finalize_stagee_budget200_cell.py` now defaults to `trace_reference_mode=manifest`
    - `scripts/stage0/convert_stage0_trace_to_rlinf_demo.py` and `scripts/stagee/build_caver_round_artifacts.py` now understand the Stage-E trace-source manifest directly
    - helper stderr now logs prefix-scan progress, so long recovery scans are visible in Slurm logs
  - authoritative live recovery jobs:
    - `5618`, `5619`, `5620`, `5621`, `5622`, `5623`
  - those six were resubmitted with `12:00:00` walltime and `--exclude=l40s-04`
  - immediate Slurm-log proof:
    - `5618` stderr shows `start mode=manifest ...` and `recovered prefix context 1/178`
    - `5619` stderr shows `start mode=manifest ...` and `recovered prefix context 1/176`
  - rechecked on `2026-04-06 10:29` America/Edmonton:
    - Stage E snapshot is now `24/25` completed cells
    - `5625` completed at `2026-04-06 10:28:01` America/Edmonton on `l40s-03`, closing the real-only seed-`13`, `budget=200` finalize retry
    - authoritative real-only seed-`13`, `budget=200` summary now exists at `/projects/p57098/euijin1/Caver/runs/stagee__real-only-round__manifest-t_train_s0-all__seed13__budget200__20260406T033634Z/results/real_only_round_summary.json`
    - validated metrics for that cell:
      - `200` contexts
      - `70` successes
      - success rate `0.35`
      - `12778` demo items
      - `63751` primitive steps
      - backend training completed
    - only `5626` remains live:
      - CAVER seed `29`, `budget=200`
      - running on `l40s-03`
      - no recurrence of the prior malformed-line failure
      - current best ETA from similar completed CAVER finalize jobs is `2026-04-06 15:40` to `16:05` America/Edmonton
- Operational caveat for the live seed job:
  - `5332` was submitted before trace capture existed and does not contain `--transition-trace-path`
  - that run can still finish for coverage and summary logging, but it cannot be retro-converted into a replay-buffer warm-start artifact
- Wrapper-default trace placement is now fixed:
  - `scripts/stage0/collect_stage0_warm_start.sh` now defaults traces into `${CAVER_RUN_DIR}/results/stage0_seed_warm_start_chunks.jsonl`
  - `scripts/stage0/run_stage0_partition_eval.sh` now defaults traces into `${CAVER_RUN_DIR}/results/stage0_${partition_name}_chunks.jsonl`
- Shared-port collisions are now also fixed:
  - Slurm jobs derive a run-local websocket port instead of all defaulting to `8000`
  - the bridge wrappers use that derived port automatically unless `--port` is passed explicitly
- The legacy non-manifest path still works after the evaluator changes:
  - regression smoke artifact: `logs/runtime/libero_legacy_dummy.json`
- Native OpenPI `pi05_libero` serving now works on `gpu-l40s` with the checkpoint cached under `/projects`.
- `MUJOCO_GL=egl` currently fails for LIBERO offscreen rendering on `gpu-l40s`, but `MUJOCO_GL=osmesa` completed a one-task, one-episode, one-step smoke run successfully in Slurm job `5329`.
