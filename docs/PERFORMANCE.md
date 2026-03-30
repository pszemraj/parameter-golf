# Performance

This file tracks the current ALlama performance state and the changes that have
materially improved or clarified it over time.

## Current State

Timestamp:

- `2026-03-30T02:20:00-04:00`

Current best ALlama quality run:

- run: `frontier_v1_shortfat_s4_ff15_pre_rms_gate005`
- sampled `val_bpb=1.434141`
- `tokens_per_s=133,833`
- `artifact_bytes=15,683,554`

Matched GPT reference from the same frontier sweep:

- run: `frontier_v1_gpt_baseline_reference`
- sampled `val_bpb=1.443775`
- `tokens_per_s=442,941`
- `artifact_bytes=11,404,805`

Current read on the project:

- ALlama is ahead on sampled validation quality, but the implementation is still
  too slow relative to GPT for comfortable search velocity.
- The quality winner is currently `shortfat_s4_ff15 + prenorm + rmsnorm +
  shortcut_gate005`.
- `wide` is no longer the best-model quality path. It remains useful only if the
  goal is an explicit speed/quality tradeoff study.
- `layernorm` is not part of the active search space. It lost to `rmsnorm` in
  both the reduced norm sweep families and layouts.

## Profiling Snapshot

Steady-state compiled CUDA profiling on the local 5090 shows that the slowdown
is dominated by real repeated GEMM work plus launch volume, not optimizer or
small pointwise overhead.

ALlama winner profile:

- output dir: `runs_allama_validation/profile_shortfat_prerms_gate005_20260330`
- profiled global step time: `4.034 s`
- profiler throughput: `64,984 tok/s`
- peak CUDA memory: `2.65 GB`
- kernel launches: `47,984`
- flash-attention forward calls: `1,280`
- flash-attention backward calls: `1,280`

GPT reference profile:

- output dir: `runs_allama_validation/profile_gpt_reference_20260330`
- profiled global step time: `1.896 s`
- profiler throughput: `138,284 tok/s`
- peak CUDA memory: `0.98 GB`
- kernel launches: `31,594`
- flash-attention forward calls: `576`
- flash-attention backward calls: `576`

Key interpretation:

- both models are GEMM-dominated
- ALlama is roughly `2.13x` slower than GPT in the profiler harness
- ALlama uses roughly `2.7x` the peak CUDA memory
- ALlama pays for repeated logical block execution; sharing saved bytes, not
  FLOPs
- hot shapes are already tensor-core-friendly, so the main problem is repeated
  compute and launch count, not obviously bad divisibility

## Repro Harness

Checked-in scripts now exist for repeatable local performance work:

- full-step benchmark or `torch.profiler` harness:
  - `bash scripts/run_allama_perf_harness.sh`
- Nsight Systems on the frozen ALlama anchor:
  - `bash scripts/run_allama_nsys_profile.sh allama_anchor`
- Nsight Compute on the representative hot kernels:
  - `bash scripts/run_allama_ncu_profile.sh`

These scripts assume the current local 5090 contract:

- `TORCH_BLAS_PREFER_CUBLASLT=1`
- single-GPU `4 x 1024 x 64` accumulation semantics
- compiled steady-state training on the frozen ALlama anchor or GPT reference

The harness owns the model contracts directly:

- `allama_anchor`: `shortfat_s4_ff15 + prenorm + rmsnorm + shortcut_gate005`
- `gpt_reference`: the current `train_gpt.py` reference defaults under the same
  batch contract

Latest harness smoke check:

- GPT reference: `441,358 tok/s`, `0.594 s/step`, `0.98 GB` peak CUDA memory
- ALlama anchor: `125,857 tok/s`, `2.083 s/step`, `2.65 GB` peak CUDA memory

## Improvement Log

### 2026-03-29

- Replaced the older ALlama path with the cleaned shared-model implementation in
  `train_allama.py` + `allama_shared.py`.
- Removed dead delta dispatch, kept both `prenorm` and `postnorm`, reused the
  Muon flat buffer, and made the SDPA GQA probe lazy.
- Result: cleaner compile path and lower implementation overhead, but not enough
  to close the large throughput gap to GPT.

### 2026-03-29 to 2026-03-30

- Ran the reduced norm-kind sweep on `wide_s4_e384_ff10` and
  `shortfat_s4_ff15`.
- Result:
  - `layernorm` lost to `rmsnorm` everywhere tested
  - `postnorm+rmsnorm` remained viable for `wide`
  - `prenorm+rmsnorm` stayed best for `shortfat`

### 2026-03-30

- Ran the frontier combination sweep focused on `shortfat` and `wide`.
- Result:
  - new best ALlama: `shortfat_s4_ff15 + prenorm + rmsnorm + shortcut_gate005`
  - sampled `val_bpb=1.434141`
  - improved the previous best ALlama run by `0.000543 bpb`
  - `shortcut_no_x0` did not help `wide`

### 2026-03-30

- Profiled the current ALlama winner and the matched GPT reference under the
  same steady-state compiled CUDA harness.
- Result:
  - confirmed the main bottleneck is repeated shared-block compute and launch
    count
  - ruled out Muon, norms, or miscellaneous Python glue as primary causes
  - established the need for a custom Triton or C++ kernel path aimed at
    block-level fusion and launch reduction

### 2026-03-30

- Checked in the reproducible performance harness:
  - `scripts/profile_training_step.py`
  - `scripts/profile_hot_kernels.py`
  - `scripts/run_allama_perf_harness.sh`
  - `scripts/run_allama_nsys_profile.sh`
  - `scripts/run_allama_ncu_profile.sh`
- Result:
  - benchmark and profiler work is now scriptable from the repo instead of
    living only in ad hoc local commands
  - the frozen quality anchor is now encoded directly in the harness
  - the harness smoke reproduced the expected throughput gap: GPT around
    `441k tok/s`, ALlama around `126k tok/s`

### 2026-03-30

- Ran `nsys` on the frozen ALlama anchor using the checked-in harness.
- Result:
  - the timeline confirms the launch-heavy repeated-block picture from
    `torch.profiler`
  - one profiled global step issued `47,984` `cuLaunchKernel` calls and
    `11,638` `cudaLaunchKernel` calls
  - the top GPU kernels are still the same CUTLASS GEMMs plus flash-attention
    backward and forward
  - representative report location:
    `runs_allama_validation/nsight_systems/20260330_024707_allama_anchor`

### 2026-03-30

- Started the checked-in `ncu` path for the anchor hot kernels.
- Result:
  - wrapper issues are fixed
  - current blocker is local GPU performance-counter permissions:
    `ERR_NVGPUCTRPERM`
  - once counters are re-enabled locally, rerun:
    `bash scripts/run_allama_ncu_profile.sh`

## Next Work

- Keep `frontier_v1_shortfat_s4_ff15_pre_rms_gate005` as the quality anchor.
- Stop spending best-model sweep budget on `wide` or `layernorm`.
- Use the profiler traces as the starting point for a custom kernel plan around
  the repeated shared block, especially the prenorm/x0 mix/norm/projection
  boundaries and residual epilogues.
- Finish the `ncu` pass on the hot kernels after GPU performance-counter access
  is restored, then pick the first kernel to replace.
