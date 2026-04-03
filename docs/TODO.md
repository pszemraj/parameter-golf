# HGDN Compile / Perf TODO

This file tracks follow-up work that is intentionally not enabled by default in the current hybrid trainer.

## Active next steps

### 0. Norm placement screen (`pre` vs `post` vs `keel`)

- Status: newly enabled, not quality-screened yet.
- Why: recent work argues that pre-norm may be leaving depth utilization on the table, especially when a KEEL-style residual path makes post-norm trainable again.
- Helper: `scripts/run_laptop_norm_compare.sh {hybrid|depth|both}`
- Helper default: `WANDB_WATCH=none` so the comparison screens keep online metric logging without gradient-watch stalls. Override with `WANDB_WATCH=gradients` only when histograms are actually needed.
- Scope for this branch:
  - use the hybrid trainer only
  - compare within the same residual shell
  - treat `GDN_RATIO=0` as the pure-attention control
  - treat `GDN_RATIO=1` as the current HGDN operating point
- First readout to collect:
  - fixed-step sampled-eval BPB at the same token budget
  - whether `post` is merely stable or actually better
  - whether `keel` beats both `pre` and naive `post`

Suggested local comparison contract:

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=0 MLP_MULT=4.0 NORM_STYLE=pre \
RUN_ID=norm_depth_pre_r0 scripts/sweep.sh depth
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=0 MLP_MULT=4.0 NORM_STYLE=post \
RUN_ID=norm_depth_post_r0 scripts/sweep.sh depth
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=0 MLP_MULT=4.0 NORM_STYLE=keel \
RUN_ID=norm_depth_keel_r0 scripts/sweep.sh depth
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=pre \
RUN_ID=norm_hybrid_pre_r1 scripts/sweep.sh single
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=post \
RUN_ID=norm_hybrid_post_r1 scripts/sweep.sh single
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=keel \
RUN_ID=norm_hybrid_keel_r1 scripts/sweep.sh single
```

### 1. Size-matched depth-control rerun

- Status: partially resolved.
- Why: the current hybrid quality win is real, but the logged artifact sizes are still not perfectly matched.
- What was tried:
  - `MLP_MULT=4.7` at a local 600-second screen: invalid, landed at `17,092,318` total bytes and was over budget.
  - `MLP_MULT=4.0` at the proper 2k-step fixed-step contract: valid, landed at `9,668,381` total bytes and `2.6550` / `2.6715` pre/post-roundtrip BPB. It is still smaller than the hybrid and only marginally better than `MLP_MULT=3.75`.
- Current takeaway:
  - Do not use a 600-second local 4070 run as the final size-matching proxy.
  - If exact size matching still matters locally, bracket the next fixed-step depth candidate between `4.0` and a slightly larger setting.
- Last fixed-step command used:

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=2000 TRAIN_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS=0 TRAIN_SEQ_LEN=2048 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=200 \
COMPILE_STRATEGY=model MLP_MULT=4.0 RUN_ID=quality_depth_mlp40_seq2k \
scripts/sweep.sh depth
```

### 2. H100 throughput calibration

- Status: pending target-hardware access.
- Why: the 4070 ratio improved from `1.38x` at `seq=1024` to `1.16x` at `seq=2048`, but the real question is the H100 ratio.
- Run only the existing perf harness first, not a full quality sweep.
- Helper: `scripts/run_h100_single_gpu_hgdn.sh perf`

### 3. Compute-optimal size sweep

- Status: no longer blocked by architecture viability; only blocked if exact local size-matching is considered a hard requirement first.
- Why: actual trained artifacts are around `11MB`, so the branch still needs a wall-clock-vs-size sweep instead of assuming "fill 16MB" is optimal.
- Candidate HGDN wall-clock sweep family:
  - `slim`: `MODEL_DIM=320`, `MLP_MULT=3.25`
  - `current`: `MODEL_DIM=384`, `MLP_MULT=3.25`
  - `mid`: `MODEL_DIM=416`, `MLP_MULT=3.25`
  - `wide`: `MODEL_DIM=448`, `MLP_MULT=3.0`

## Already landed

- Default compile path is now `COMPILE_STRATEGY=model`.
- The trainer has a dedicated perf harness:
  - `PERF_TIMING=1` for steady-state timing
  - `PERF_IGNORE_STEPS=N` to ignore early measured steps
  - `PERF_ISOLATE_COMPILE_CACHE=1` for fresh Inductor/Triton cache dirs per run
  - `PERF_SKIP_FINAL_EVAL=1` to stop after the measured training window
- The experimental `COMPILE_STRATEGY=hybrid` path remains available:
  - each GDN module is wrapped as an explicit eager boundary because the FLA path already dispatches Triton kernels and contains an internal `torch.compiler.disable()` wrapper
  - pure attention blocks are compiled with `fullgraph=True`
  - GDN-block MLPs are compiled with `fullgraph=True`
  - the top-level hybrid model still compiles with `fullgraph=False`
- Current local result on the RTX 4070: `COMPILE_STRATEGY=hybrid` underperformed `COMPILE_STRATEGY=model` by roughly `30%` on the 16-layer HGDN throughput screen, so it should stay experimental until an H100 test says otherwise.

## Break-Glass Items

### 1. Graph-break audit with `TORCH_LOGS` / `tlparse`

- Trigger: hybrid remains worse than `1.3x` the depth-control baseline after the current selective-compile quick hits, or compile-time behavior becomes erratic across repeated runs.
- What to do: run short screens with `TORCH_LOGS="graph_breaks,recompiles,perf_hints"` and inspect whether breaks are only at `GDNBlock.gdn` boundaries.
- Expected upside: `2-8%` on the 4070, `5-12%` on H100 if extra Python-side breaks are still present.
- Expected cost: low. Mostly logging and a small cleanup patch.

### 2. Backward-path `compiled_autograd`

- Trigger: forward graphs look clean but hybrid throughput is still lagging, or logs suggest backward fragmentation dominates runtime.
- What to do: test PyTorch compiled autograd on the hybrid trainer in a separate branch with the same 50-step throughput harness.
- Expected upside: `0-10%`, highly dependent on whether backward graph breaks are the real bottleneck.
- Expected cost: medium. Can increase recompiles and runtime overhead if shapes or control flow drift.

### 3. Regional compilation for repeated block stacks

- Trigger: compile warmup/cold-start becomes the main pain point, or we want faster iteration on short local screens without materially changing steady-state kernels.
- What to do: trial `torch.compiler.nested_compile_region` or a block-stack regional compile pattern around the repeated encoder/decoder loops.
- Expected upside: little steady-state gain, but compile latency can drop materially on short runs.
- Expected cost: medium. Good engineering payoff only if compile startup is the bottleneck.

### 4. Dynamic shape marking

- Trigger: frequent recompiles show up when sweeping `TRAIN_SEQ_LEN`, local batch size, or eval shapes inside the same process.
- What to do: add `mark_dynamic` only at the specific tensor boundaries causing recompiles.
- Expected upside: stability and lower recompile churn, not raw peak throughput.
- Expected cost: medium. For fixed-shape runs this can be neutral or negative, so it should stay off unless logs justify it.

### 5. FLA wrapper cleanup inside `model.py`

- Trigger: module-level eager boundaries still leave noisy compile traces, or we want a cleaner OLMo-style explicit wrapper around the FLA dispatch path.
- What to do: add a small local dispatch helper around `chunk_gated_delta_rule` and mark that wrapper as compiler-disabled instead of relying only on module-level disabling from the trainer.
- Expected upside: mostly cleaner boundaries and easier debugging, maybe `1-4%` if Dynamo is still poking at surrounding glue code.
- Expected cost: low to medium. Needs care to keep dtype and API handling identical.

### 6. Compile-mode shootout on finalists only

- Trigger: the hybrid passes the quality gates and is close enough to baseline that a few percent of throughput matters.
- What to do: compare PyTorch compile modes on fixed-shape finalists only, for example the default mode against a more aggressive autotune mode.
- Expected upside: `0-5%` if the workload lines up well with the mode.
- Expected cost: medium to high. More compile/autotune time, and results can differ across the 4070 and H100.

### 7. Nsight / kernel-level profiling

- Trigger: throughput remains unexplained after graph-break cleanup and matched baseline comparisons.
- What to do: profile attention vs GDN-heavy runs on the local GPU, but only after confirming local Nsight permissions are working.
- Expected upside: diagnostic clarity more than immediate speed.
- Expected cost: medium. Useful only after the higher-leverage compile quick hits are exhausted.
