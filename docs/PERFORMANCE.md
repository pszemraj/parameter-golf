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

## Next Work

- Keep `frontier_v1_shortfat_s4_ff15_pre_rms_gate005` as the quality anchor.
- Stop spending best-model sweep budget on `wide` or `layernorm`.
- Use the profiler traces as the starting point for a custom kernel plan around
  the repeated shared block, especially the prenorm/x0 mix/norm/projection
  boundaries and residual epilogues.
