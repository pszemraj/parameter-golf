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
  - on this machine, `sudo`-launched `ncu` can also fail on
    `/tmp/nsight-compute-lock` because `fs.protected_regular=2` interacts badly
    with sticky `/tmp`
  - the wrapper now sets a per-run `TMPDIR` inside the run directory so `ncu`
    does not depend on `/tmp` for its lock file
  - output artifacts are chmodded open at the end of the run so root-owned
    reports are still easy to inspect locally

### 2026-03-30

- Completed `ncu` on the representative hot kernels at:
  `runs_allama_validation/nsight_compute/20260330_031014`
- Result:
  - the hot MLP GEMMs are large, latency-limited CUTLASS kernels rather than
    obviously broken shapes
  - `allama_mlp_up` primary GEMM:
    - `190.72 us`
    - `31.34%` compute throughput
    - `16.63%` achieved occupancy
    - `232` registers/thread
    - `73.73 KB` dynamic shared memory/block
  - `allama_mlp_down` primary GEMM:
    - `200.96 us`
    - `34.81%` compute throughput
    - `8.33%` achieved occupancy
    - `230` registers/thread
    - `81.92 KB` dynamic shared memory/block
  - `allama_qkv` primary GEMM:
    - `64.93 us`
    - `41.35%` compute throughput
    - `15.71%` achieved occupancy
    - `88` registers/thread
    - `49.15 KB` dynamic shared memory/block
  - `allama_attn_proj` primary GEMM:
    - `60.38 us`
    - `35.23%` compute throughput
    - `15.50%` achieved occupancy
    - `88` registers/thread
    - `49.15 KB` dynamic shared memory/block
  - flash-attention backward is also latency-limited:
    - `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` took `200.29 us`
    - `30.78%` compute throughput
    - `16.64%` achieved occupancy
    - `255` registers/thread
    - `73.73 KB` dynamic shared memory/block
  - implication:
    - the main cost is still the repeated shared-block compute itself
    - the vendor GEMMs are not trivially replaceable by a small custom matmul
      tweak
    - the first custom kernel effort should still target block-level launch
      reduction and fused glue around the repeated shared block, not the
      optimizer

### 2026-03-30

- Prototyped a first narrow Triton kernel for the prenorm prologue:
  fused `x0` mix plus RMSNorm before attention.
- Validation:
  - forward and backward stayed numerically close to the PyTorch reference on
    CUDA
  - compiled CUDA smoke worked under the local 5090 contract with
    `TORCH_BLAS_PREFER_CUBLASLT=1`
- Benchmark result using the checked-in harness on the frozen anchor:
  - baseline:
    `runs_allama_validation/perf_triton_prenorm_compare/off/allama_anchor/run_summary.json`
    - `126,416 tok/s`
    - `2.074 s/step`
  - Triton prenorm prototype:
    `runs_allama_validation/perf_triton_prenorm_compare/on/allama_anchor/run_summary.json`
    - `117,741 tok/s`
    - `2.226 s/step`
  - net result: about `6.9%` slower
- Follow-up profiler read:
  - profile location:
    `runs_allama_validation/perf_triton_prenorm_profile/on/allama_anchor`
  - the custom forward kernel itself was only about `0.52%` of self CUDA time
  - the regression came from worse end-to-end graph behavior and extra launch
    pressure, not from the fused forward kernel being individually expensive
- Conclusion:
  - do not keep the narrow forward-only Triton prenorm path in the model
  - Inductor already does enough on this small prologue that a custom op there
    is not a free win
  - the next custom-kernel target should be larger-grain fusion, likely a
    block-level epilogue/prologue path or a C++/CUTLASS operator that absorbs
    more of the repeated shared-block glue at once

### 2026-03-30

- Checked local extension-toolchain readiness for the next kernel path.
- Result:
  - local C++/CUDA extension builds are viable on this machine
  - `CUDA_HOME=/usr/local/cuda`
  - `nvcc` is `12.9`
  - the system C++ compiler is `g++ 13.3`
  - CUTLASS headers are not bundled under `/usr/local/cuda/include`
- Implication:
  - a custom C++/CUDA operator path is realistic locally
  - if we want CUTLASS epilogue fusion, we need to vendor CUTLASS or bring it
    in explicitly instead of assuming it ships with the CUDA toolkit

### 2026-03-30

- Added a standalone C++/CUDA benchmark scaffold for a larger shared-block
  prologue candidate:
  `residual_scale_rms_norm(x, branch, scale, weight, eps)`.
- Files:
  - `csrc/allama_ops.cpp`
  - `csrc/allama_residual_scale_rms_norm.cu`
  - `scripts/benchmark_allama_cpp_ops.py`
- This operator is relevant because it can model both:
  - the MLP prenorm prologue after attention:
    `rms_norm(x + attn_scale * attn_out)`
  - the attention prenorm shortcut mix if `scale=sigmoid(x0_gate)` and
    `branch=x0`
- Benchmark on representative ALlama anchor shape `[4, 1024, 896]`:
  - summary:
    `runs_allama_validation/cpp_ops_v1/residual_scale_rms_norm_summary.json`
  - eager PyTorch reference: `0.02485 ms`
  - compiled PyTorch reference: `0.01718 ms`
  - custom C++/CUDA op: `0.01443 ms`
  - speedup vs eager: `1.72x`
  - speedup vs compiled reference: `1.19x`
  - numerical drift stayed small for bf16:
    - `max_abs=0.125`
    - `max_rel=0.0095`
- Conclusion:
  - unlike the earlier narrow Triton prenorm attempt, this larger-grain
    operator does beat the compiled PyTorch baseline in isolation
  - the next question is no longer "can a custom op win at all"
  - the next question is whether an opt-in integration path preserves that win
    once autograd and full model compilation are involved

### 2026-03-30

- Extended the standalone C++/CUDA benchmark to a more realistic two-output
  boundary:
  `residual_scale_rms_norm_pair(x, branch, scale, weight, eps) ->
  (mixed, normed)`.
- Why this matters:
  - the single-output op only returns the normalized tensor
  - the real prenorm block also still needs the mixed residual tensor
  - this pair-output version is a more honest test of whether a fused prologue
    can pay for itself once integrated
- Benchmark on the same representative ALlama anchor shape `[4, 1024, 896]`:
  - summary:
    `runs_allama_validation/cpp_ops_v1/residual_scale_rms_norm_pair_summary.json`
  - eager PyTorch reference: `0.02699 ms`
  - compiled PyTorch reference: `0.01819 ms`
  - custom C++/CUDA pair op: `0.01240 ms`
  - speedup vs eager: `2.18x`
  - speedup vs compiled reference: `1.47x`
  - numerical drift stayed in the same bf16 band:
    - `max_abs=0.125`
    - `max_rel=0.0095`
- Conclusion:
  - the pair-output boundary is materially more promising than the earlier
    single-output sketch
  - the next practical step is an opt-in integration in the real model path,
    not more synthetic micro-benchmarks of smaller glue fragments

### 2026-03-30

- Tested an opt-in integration of `residual_scale_rms_norm_pair` in the real
  ALlama anchor, limited to the prenorm MLP boundary where the block genuinely
  consumes both outputs.
- End-to-end harness comparison under the normal compiled anchor contract:
  - baseline:
    `runs_allama_validation/perf_cpp_pair_compare/off/allama_anchor/baseline/run_summary.json`
    - `131,437 tok/s`
    - `1.994 s/step`
  - opt-in C++ pair op:
    `runs_allama_validation/perf_cpp_pair_compare/on/allama_anchor/cpp_mlp_pair_rmsnorm_on/run_summary.json`
    - `128,659 tok/s`
    - `2.038 s/step`
  - net result: about `2.1%` slower
- Follow-up profiler read:
  - profile location:
    `runs_allama_validation/perf_cpp_pair_profile/on/allama_anchor/cpp_mlp_pair_rmsnorm_on`
  - the custom op showed up as
    `allama_cpp::residual_scale_rms_norm_pair`
  - it accounted for only about `0.93%` of self CUDA time across `1,280` calls
  - implication:
    - this is not a case where the custom kernel itself is individually slow
    - the regression comes from the surrounding graph/autograd integration cost
- Conclusion:
  - do not keep the opt-in model integration path in the tree
  - keep the standalone C++/CUDA benchmark scaffold and extension sources
  - the next custom-op attempt needs to absorb more of the surrounding backward
    and graph structure, not just replace one forward prologue boundary

## Next Work

- Keep `frontier_v1_shortfat_s4_ff15_pre_rms_gate005` as the quality anchor.
- Stop spending best-model sweep budget on `wide` or `layernorm`.
- Use the profiler traces as the starting point for a custom kernel plan around
  the repeated shared block, especially larger-grain epilogue/prologue fusion
  around the attention and MLP boundaries.
- Design the next custom-op attempt around a larger boundary than the current
  pair op, so the real model path can avoid paying the current graph/autograd
  integration penalty.
