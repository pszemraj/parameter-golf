# HGDN Compile / Perf TODO

This file tracks follow-up work that is intentionally not enabled by default in the current hybrid trainer.

## Active next steps

### 0. H100 profiling-driven HGDN kernel pass

- Status: active top priority.
- Why: the H100 runs confirmed that the architecture is worth optimizing, and the profiler says the current throughput tax is not just "FLA recurrence is slow." A large fraction of the overhead is HGDN-side glue code around the recurrence.
- Current H100 facts:
  - throughput: hybrid is about `1.40x` slower than the pure-attention depth control at `seq=2048`
  - quality: hybrid still wins strongly on H100 fixed-2k quality and roundtrip BPB
- Profiling helper:
  - `scripts/run_h100_single_gpu_hgdn_profile.sh {hybrid|depth|both|both-eager}`
  - default is now `USE_WANDB=0`
  - eager modes force `COMPILE=0` for attribution-only traces when compiled graphs hide the `record_function` labels
  - local phase-1 workflow is now scripted via:
    - `scripts/run_hgdn_local_phase1.sh`
    - `scripts/analyze_hgdn_phase1.py`
    - `scripts/profile_hgdn_local_hotpath.py`
- Measured hotspot read on the 4 active profiled H100 steps:
  - hybrid flash-attention self CUDA: about `1.15s`
  - depth flash-attention self CUDA: about `2.38s`
  - hybrid GDN recurrence-family kernels: about `0.71s`
  - hybrid depthwise conv stack: about `1.01s`
  - hybrid extra elementwise/Triton glue: about `1.26s`
  - hybrid `aten::copy_`: about `0.51s`
  - depth `aten::copy_`: about `0.04s`
- Takeaway:
  - the recurrence kernel matters, but it is not the whole problem
  - the immediate optimization target is the HGDN scaffolding around the recurrence
  - attention is already on a strong PyTorch flash-SDPA path, so it is not the first place to spend engineering time

Ranked optimization checklist:

1. Eliminate dtype/layout churn in the HGDN path.
   - Why: `aten::copy_` is materially larger in hybrid than depth on H100.
   - What to inspect:
     - casts around `q/k` normalization
     - casts around `g`, `beta`, and `g_out`
     - any view/contiguous path forcing copies before or after the FLA call
     - blanket `fp32 -> bf16` recasts in `CastedLinear` on large feature-map weights
   - Current branch support:
     - `GDN_LOG_LAYOUTS=1` prints one-shot tensor dtype/shape/stride summaries at the FLA boundary
     - `GDN_AUDIT_BOUNDARIES=1` now records structured boundary layouts through `project_qkv -> conv_qkv -> norm_qkv -> recurrence_inputs -> recurrence_output -> output_gate_inputs -> output_proj_input`
     - large feature-map `CastedLinear` weights now stay `bf16`; only low-dimensional and explicit control parameters are restored to `fp32`
   - Expected upside: high, because this is pure overhead and not model math.
2. Fuse or rewrite the `q_conv/k_conv/v_conv` preprocessing path.
   - Why: depthwise conv plus its backward path is one of the largest HGDN-only buckets.
   - What to try:
     - a custom Triton kernel or a more direct fused implementation for depthwise causal conv + SiLU
     - at minimum, benchmark whether `v_conv` can be removed again on H100 without hurting quality
   - Current branch support:
     - `GDN_USE_Q_CONV`, `GDN_USE_K_CONV`, `GDN_USE_V_CONV` now allow direct q/k/v ablations without code edits
   - Expected upside: high.
3. Fuse HGDN elementwise glue.
   - Why: the profiler shows a broad pile of Triton/elementwise kernels beyond the recurrence itself.
   - What to target:
     - q/k normalization path
     - output `rms_norm(o) * silu(g_out)`
     - simple gate/decay prep if it currently spills through multiple kernels
   - Expected upside: medium to high.
4. Only after 1-3, revisit the FLA recurrence kernel itself.
   - Why: the recurrence is important, but the trace says the branch is currently losing plenty of time outside it.
   - What to do:
     - compare current FLA kernel behavior against the surrounding glue cost
     - only invest in recurrence-kernel changes if it remains one of the top HGDN buckets after the glue cleanup
   - Expected upside: medium, but expensive engineering.
5. Run one attribution-only pass with `COMPILE=0`.
   - Why: compiled mode swallows some `record_function` labels, so a short eager profile will be easier to read.
   - Important: use it for diagnosis only, not throughput conclusions.
6. Consider attention-side alternatives only later.
   - Why: the attention stack is already using PyTorch flash SDPA, not a naive slow path.
   - Flex attention or external `flash-attn` should only be tested if HGDN-side kernels stop being the dominant issue.

Suggested immediate sequence:

```bash
scripts/run_hgdn_cuda_preflight.sh
```

Then:

```bash
RUN_PREFIX=h100prof scripts/run_h100_single_gpu_hgdn_profile.sh both
```

Then:

- inspect `profiles/<run_id>/key_averages.json`
- inspect the Chrome/Perfetto trace in `profiles/<run_id>/traces/`
- do one short eager attribution rerun with `COMPILE=0` only if compiled traces remain too anonymous

Latest local attribution checkpoint:

- `rtx4070_phase1` shows that the most concrete next target is the post-conv q/k/v layout path.
- The boundary audit shows q/k/v start contiguous after projection, become non-contiguous after `q_conv/k_conv/v_conv`, and stay non-contiguous into `recurrence_inputs`.
- That makes the next semantics-preserving kernel pass:
  - layout cleanup between `conv_qkv` and recurrence
  - then `norm_qkv` / `output_gate` glue
  - not generic trainer-wide `copy_` chasing
- First boundary candidate now exists and is worth keeping in the experiment surface:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - it fixes q/k/v contiguity all the way from `conv_qkv` to `recurrence_inputs`
  - local trainer self-device time improved from `25,990.59 ms` to `25,258.00 ms` (`-2.82%`)
  - the tradeoff is a more expensive `gdn.conv_qkv` bucket (`236.67 -> 322.01 ms`) in exchange for a cheaper recurrence/norm path
- Immediate next local target after this candidate:
  - `gdn.gates`
  - then `gdn.output_gate`
  - with explicit attention to avoidable fp32 promotion islands

### 1. Norm placement screen (`pre` vs `post` vs `keel`)

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

### 2. Size-matched depth-control rerun

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

### 3. H100 throughput calibration

- Status: done.
- Result:
  - H100 hybrid `GDN_RATIO=1, MLP_MULT=3.25` perf: `1002.72 ms`
  - H100 depth `MLP_MULT=4.0` perf: `714.74 ms`
  - hybrid slowdown: about `1.40x`
- Current takeaway:
  - the architecture remains interesting because the quality gap on H100 is stronger than local
  - but kernel work should come before broader size sweeps because the throughput penalty is larger than the 4070 suggested

### 4. Compute-optimal size sweep

- Status: temporarily deprioritized behind kernel work.
- Why: actual trained artifacts are around `11MB`, so the branch still needs a wall-clock-vs-size sweep instead of assuming "fill 16MB" is optimal.
- Candidate HGDN wall-clock sweep family:
  - `slim`: `MODEL_DIM=320`, `MLP_MULT=3.25`
  - `current`: `MODEL_DIM=384`, `MLP_MULT=3.25`
  - `mid`: `MODEL_DIM=416`, `MLP_MULT=3.25`
  - `wide`: `MODEL_DIM=448`, `MLP_MULT=3.0`

## Already landed

- Default compile path is now `COMPILE_STRATEGY=model`.
- The branch now has a profiling harness in the trainer:
  - `PROFILE=1`
  - scheduled `torch.profiler` capture
  - trace export under `profiles/<run_id>/traces/`
  - operator summary export to `profiles/<run_id>/key_averages.json` and `key_averages.csv`
- The branch now has a dedicated H100 profiling helper:
  - `scripts/run_h100_single_gpu_hgdn_profile.sh`
  - defaults to `USE_WANDB=0`
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

- Trigger: after the first HGDN glue/kernel cleanup pass, or when we need a firmer memory-bound vs compute-bound answer on H100.
- What to do:
  - use the current torch-profiler pass first
  - then move to Nsight Compute on the top HGDN kernels, especially depthwise conv and recurrence-adjacent kernels
- Expected upside: high diagnostic clarity for the second optimization pass, not necessarily an immediate speedup by itself.
- Expected cost: medium.

### 8. cuLA backend swap on Hopper, after the in-repo kernel pass

- Trigger: only after the current HGDN kernel work has already attacked copies, depthwise convs, and elementwise glue, and only if the linear-attention backend itself still looks like a serious remaining bottleneck on H100.
- Why it is interesting:
  - cuLA targets Hopper and Blackwell directly with CUDA/CuTe/CUTLASS kernels
  - it is intended to align with the FLA interface, so the eventual integration path may be relatively shallow
- Why it is not the first move:
  - the current profile says the branch is losing plenty of time outside the recurrence kernel
  - cuLA is early-stage and the README still marks modular GDN forward/backward support as incomplete
  - its published requirements and tested stack differ from this branch's current environment, so it is not a low-friction drop-in today
- What to do:
  - treat this as an H100-only side experiment, not a portability-preserving branch default
  - make a separate backend wrapper so the code can switch between FLA and cuLA cleanly
  - benchmark recurrence-heavy HGDN runs against the same FLA baseline on the same Hopper machine
  - only keep it if it wins materially without introducing integration debt or correctness drift
- Expected upside: unknown, potentially material, but currently speculative for this exact GDN training path.
- Expected cost: high. Environment churn plus backend-integration work.
