# Performance

This file tracks the current ALlama performance state and the changes that have
materially improved or clarified it over time.

## Current State

Timestamp:

- `2026-03-30T23:25:00-04:00`

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
- On the active local 5090 contract with `TORCH_BLAS_PREFER_CUBLASLT=1`, the
  best shipped kernel path is again `MLP_KERNEL=triton_gateup` on the ALlama
  anchor.
- Beyond the live MLP gate-up path, the strongest surviving kernel evidence is
  now benchmark-only:
  - the MLP gate/up boundary still beats compiled PyTorch in isolation
  - the attention out-proj boundary now wins strongly on forward-only under
    torch 2.11, but still loses badly once backward is included
- The first real block-scale integrations beyond `triton_gateup` have now been
  tested and rejected:
  - the attention bridge (`out-proj + residual + next RMSNorm`) regressed
  - the larger full MLP op (`gateup + down + residual`) also regressed
  - the current shipped path is still `MLP_KERNEL=triton_gateup`
- FlexAttention is only interesting for this project if the FA4/CuTe
  `BACKEND="FLASH"` path is installed. The built-in Triton flex backend does
  not beat SDPA on the dense causal GQA ALlama shape.
- `flash-attn 2.8.3` is now installed in the local `train` env, but the first
  direct FA2 benchmarks on the exact ALlama shape still lost to SDPA on the
  local 5090. That makes FA2 useful for research and possible H100 follow-up,
  but not an automatic drop-in model-path integration win.
- A proper FA2-native model-path adaptation now exists behind
  `ATTN_IMPL=fa2`, and unlike the raw backend swap, it does produce a real
  end-to-end gain on the anchor harness.
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

These scripts assume the current local single-GPU benchmark contract:

- `TORCH_BLAS_PREFER_CUBLASLT=1` on the local 5090
- `4 x 1024 x 64` accumulation semantics
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

- Implemented a real backward-side Triton kernel for the larger MLP gate/up
  boundary benchmark in `scripts/benchmark_allama_mlp_gateup_block.py`.
- Result:
  - the custom op now has a fused backward-parts kernel that recomputes the
    gate/up activations and emits `grad_gate` and `grad_up` directly, instead
    of delegating the whole derivative path to eager PyTorch
  - standalone benchmark at
    `runs_allama_validation/mlp_gateup_block_v2/summary.json`:
    - compiled forward: `0.22528 ms -> 0.19599 ms` for about `1.15x` speedup
    - compiled backward: `0.66785 ms -> 0.77036 ms`, still about `15.4%` slower
      than the compiled PyTorch reference
  - end-to-end compiled train-step probe on the real ALlama anchor was still
    positive despite the standalone backward loss:
    - baseline: `130,451 tok/s`, `2.010 s/step`, `2.65 GB` peak CUDA memory
    - Triton gate/up v2 monkeypatch: `133,582 tok/s`, `1.962 s/step`,
      `2.19 GB` peak CUDA memory
    - net: about `+2.4%` throughput with materially lower peak memory
- Conclusion:
  - the MLP gate/up boundary is the first custom backward path that survives a
    real compiled train-step contract
  - this is the current best candidate for model integration work

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

### 2026-03-30

- Extended the standalone C++/CUDA scaffold with a real backward kernel for the
  pair boundary and benchmarked it through a benchmark-local opaque custom op.
- Outputs:
  - directory: `runs_allama_validation/cpp_ops_v2`
  - combined summary: `runs_allama_validation/cpp_ops_v2/summary.json`
- Result on representative shape `[4, 1024, 896]`:
  - forward-only pair op still wins in isolation:
    - compiled PyTorch reference: `0.01837 ms`
    - custom C++/CUDA pair op: `0.01451 ms`
    - speedup vs compiled reference: `1.27x`
  - backward-inclusive isolated benchmark does not win yet:
    - compiled PyTorch reference: `0.21968 ms`
    - compiled custom forward+backward path: `0.25292 ms`
    - net result: about `13.1%` slower than compiled reference
  - numerical drift remained acceptable for bf16 prototype work:
    - `max_abs=1.1607`
    - `max_rel=0.00746`
- Conclusion:
  - making the backward path opaque fixed the compileability problem, but not
    the performance problem
  - the current pair boundary is still too small to beat compiled PyTorch once
    backward is part of the contract
  - the next custom-op candidate should be a larger boundary, not more polish
    on this same pair operator

### 2026-03-30

- Added an optional CUDA-graph benchmark mode to the checked-in perf harness for
  the fixed forward/backward microbatch loop.
- Outputs:
  - ALlama anchor baseline:
    `runs_allama_validation/perf_cuda_graph_compare/off/allama_anchor/baseline/run_summary.json`
  - ALlama anchor CUDA graph:
    `runs_allama_validation/perf_cuda_graph_compare/on/allama_anchor/cuda_graph_fwbw_on/run_summary.json`
  - GPT baseline:
    `runs_allama_validation/perf_cuda_graph_compare/off/gpt_reference/baseline/run_summary.json`
  - GPT CUDA graph:
    `runs_allama_validation/perf_cuda_graph_compare/on/gpt_reference/cuda_graph_fwbw_on/run_summary.json`
- Result:
  - ALlama anchor:
    - baseline: `137,601 tok/s`
    - CUDA graph forward/backward: `143,654 tok/s`
    - gain: about `4.4%`
  - GPT reference:
    - baseline: `244,580 tok/s`
    - CUDA graph forward/backward: `258,874 tok/s`
    - gain: about `5.8%`
- Interpretation:
  - static CUDA-graph capture is viable on the local contract and is worth
    keeping in mind as a systems optimization
  - but the gain is generic rather than ALlama-specific, and it is not large
    enough to solve the fundamental throughput gap by itself

### 2026-03-30

- Added a standalone Triton benchmark for a larger attention-side candidate
  boundary:
  - split fused `qkv`
  - reshape into SDPA layout
  - q/k RMSNorm
  - RoPE
  - q_gain
- File:
  - `scripts/benchmark_allama_attention_prep.py`
- Outputs:
  - summary:
    `runs_allama_validation/attention_prep_v1/summary.json`
- Result on representative anchor shape `qkv=[4, 1024, 1152]`,
  `num_heads=14`, `num_kv_heads=2`, `head_dim=64`:
  - prep-only boundary:
    - compiled PyTorch reference: `0.03960 ms`
    - Triton kernel: `0.04098 ms`
    - result: essentially tied, slightly slower than compiled reference
  - prep plus flash-attention forward:
    - compiled PyTorch reference: `0.08764 ms`
    - Triton prep + flash forward: `0.11069 ms`
    - result: about `20.8%` slower than compiled reference
  - numerical drift stayed in a reasonable bf16 range:
    - prep-only `max_rel=0.0070`
    - prep+flash `max_rel=0.0103`
- Conclusion:
  - this attention-prep boundary is not strong enough to be a first shipped
    kernel
  - even though it collapses a larger sequence of attention-side glue ops than
    the earlier prenorm/pair experiments, compiled PyTorch plus flash-attention
    is already strong here
  - the first serious custom kernels should be larger block-level boundaries,
    not more polish on attention prep alone

## Next Work

- Keep `frontier_v1_shortfat_s4_ff15_pre_rms_gate005` as the quality anchor.
- Stop spending best-model sweep budget on `wide` or `layernorm`.
- Use the profiler traces as the starting point for a custom kernel plan around
  the repeated shared block, especially larger-grain epilogue/prologue fusion
  around the attention and MLP boundaries.
- The current evidence has converged the initial kernel priorities:
  - first priority: a larger MLP-block kernel path that absorbs enough of the
    forward and backward structure around `gate_up`, `silu*up`, `down`, and the
    residual epilogue to matter beyond microbenchmarks
  - second priority: an attention-block boundary larger than prep alone,
    likely something that meaningfully reduces the qkv-to-flash-to-proj glue
    around the repeated shared block instead of just replacing q/k transforms
  - CUDA graphs are worth keeping available, but they are a secondary systems
    optimization, not the main answer

## Larger MLP Block Candidate

- Added `scripts/benchmark_allama_mlp_block.py`.
- Boundary under test:
  - vendor RMSNorm
  - vendor `gate_up` GEMM
  - Triton fused `SwiGLU + down projection + residual add/scale`
- The kernel reads `gate_up` output in `[M, 2H]`, computes `silu(gate) * up`
  on the fly, multiplies directly by `down_weight_t`, and writes the residual
  epilogue output without materializing the hidden MLP activation.
- Backward is intentionally straightforward PyTorch:
  - `grad_branch = grad_out * scale`
  - `grad_hidden = grad_branch @ W^T`
  - elementwise `SwiGLU` backward
  - `grad_W = hidden^T @ grad_branch`
- Representative anchor-shape result in
  `runs_allama_validation/mlp_block_v1/summary.json`:
  - shape: `batch=4`, `seq_len=1024`, `dim=896`, `hidden=1408`
  - forward:
    - eager reference: `0.28782 ms`
    - eager custom: `0.28372 ms`
    - compiled reference: `0.25822 ms`
    - compiled custom: `0.27940 ms`
  - backward:
    - eager reference: `0.94483 ms`
    - eager custom: `1.11486 ms`
    - compiled reference: `0.83704 ms`
    - compiled custom: `0.94980 ms`
  - numerical drift stayed within a reasonable bf16 range:
    - forward `max_rel=0.00676`
    - backward `max_rel=0.00838`
- Notes:
  - the first naive tiling was materially worse; grouped tiling and a bf16
    tensor-core-friendly dot path recovered most of the loss
  - even after that recovery, this boundary is still not a shipped kernel:
    compiled forward remains about `7.6%` slower than the compiled PyTorch
    baseline and compiled backward remains about `13.5%` slower
- Conclusion:
  - this is no longer an obviously bad idea, but the current MLP boundary is
    still not strong enough for end-to-end integration
  - if we stay on the MLP side, the next move should be an even larger boundary
    or a real custom backward, not more polish on this exact kernel

## Larger Attention Block Candidate

- Added `scripts/benchmark_allama_attention_block.py`.
- Boundary under test:
  - flash-attention output in head-major layout `[B, H, T, Dh]`
  - Triton fused output projection + residual add/scale
- The kernel avoids materializing the transposed `[B, T, D]` flash-attention
  output before the projection matmul.
- Backward is intentionally straightforward PyTorch:
  - `grad_branch = grad_out * scale`
  - `grad_W = attn_flat^T @ grad_branch`
  - `grad_attn = grad_branch @ W^T`, then reshape back to `[B, H, T, Dh]`
- Representative anchor-shape result in
  `runs_allama_validation/attention_block_v1/summary.json`:
  - shape: `batch=4`, `seq_len=1024`, `num_heads=14`, `head_dim=64`, `dim=896`
  - forward:
    - eager reference: `0.09131 ms`
    - eager custom: `0.04000 ms`
    - compiled reference: `0.05558 ms`
    - compiled custom: `0.05245 ms`
  - backward:
    - eager reference: `0.31109 ms`
    - eager custom: `0.49060 ms`
    - compiled reference: `0.34759 ms`
    - compiled custom: `0.46758 ms`
  - numerical agreement was exact in backward for this boundary and stayed
    within bf16 rounding tolerance in forward:
    - forward `max_rel=0.00442`
    - backward `max_rel=0.0`
- Conclusion:
  - this is the first larger custom boundary that shows a real compiled forward
    win: compiled custom forward is about `5.6%` faster than compiled PyTorch
  - it is not ready for training integration because the PyTorch backward path
    gives back that win and more; compiled backward is about `34.5%` slower
  - this is currently the most promising initial custom-kernel direction, but
    it needs a real backward kernel or a smarter integration strategy before it
    can help end-to-end throughput

## Larger MLP Gate-Up Candidate

- Added `scripts/benchmark_allama_mlp_gateup_block.py`.
- Boundary under test:
  - vendor RMSNorm
  - Triton fused gate projection + up projection + `SwiGLU` epilogue
  - vendor down projection
  - vendor residual add/scale
- This kernel avoids materializing the larger `[M, 2H]` gate-up activation
  tensor and instead emits the hidden `[M, H]` activation directly.
- Backward now includes a real Triton derivative kernel that recomputes the
  gate and up activations and emits `grad_gate` and `grad_up` directly, while
  the larger input-gradient and weight-gradient matmuls still use vendor paths.
- Current representative anchor-shape result in
  `runs_allama_validation/mlp_gateup_block_v2/summary.json`:
  - shape: `batch=4`, `seq_len=1024`, `dim=896`, `hidden=1408`
  - forward:
    - eager reference: `0.24228 ms`
    - eager custom: `0.20073 ms`
    - compiled reference: `0.22528 ms`
    - compiled custom: `0.19599 ms`
  - backward:
    - eager reference: `0.74162 ms`
    - eager custom: `0.90162 ms`
    - compiled reference: `0.66785 ms`
    - compiled custom: `0.77036 ms`
  - numerical drift stayed within a reasonable bf16 range:
    - forward `max_rel=0.00575`
    - backward `max_rel=0.01090`
- Conclusion:
  - this is the strongest MLP-side candidate so far
  - compiled custom forward remains about `14.9%` faster than compiled PyTorch
  - standalone compiled backward is still about `15.4%` slower than the
    compiled reference, so the matmul-heavy remainder of backward is still the
    limiting piece
  - the evidence now points to these two forward kernels as the best initial
    directions:
    - MLP: fused gate-up projection + `SwiGLU`
    - attention: fused post-flash output projection + residual epilogue

## End-to-End Harness Checks

- Measured the strongest benchmark candidates on the real compiled ALlama
  anchor harness.
- Result:
  - the MLP gate-up path is now the first candidate that survives full model
    integration and still helps throughput
  - the current attention output boundary still does not survive its backward
    path, so it remains benchmark-only

### MLP Gate-Up Integration Check

- Integrated the fused gate-up `SwiGLU` kernel into the shared MLP path behind
  `MLP_KERNEL=triton_gateup`, then measured the real compiled anchor step with
  the standard local accumulation contract.
- Current representative compare in
  `runs_allama_validation/perf_integrated_mlp/`:
  - baseline:
    - `mean_step_s=2.00143`
    - `tokens_per_s=130978.12`
    - `peak_cuda_mem_bytes=2648702464`
  - `MLP_KERNEL=triton_gateup`:
    - `mean_step_s=1.96022`
    - `tokens_per_s=133732.06`
    - `peak_cuda_mem_bytes=2194671104`
- Conclusion:
  - the integrated path improves the compiled anchor by about `2.1%`
  - it also cuts peak CUDA memory by about `453 MB`
  - this is the first custom-kernel path worth keeping in `allama_shared.py`
    instead of only in a benchmark script
  - the win is modest, so this is not the end state, but it is finally a real
    step in the right direction

### Torch 2.11 Recheck

- Rechecked the active backend and kernel picture after the local upgrade to
  `torch 2.11.0+cu128`.
- Fixed the lazy SDPA `enable_gqa` probe in `allama_shared.py` so it:
  - runs outside Dynamo/Inductor capture
  - probes on the real active device/backend instead of CPU only
  - resolves once during attention-module construction instead of inside the
    compiled forward path
- Rechecked the old `TORCH_BLAS_PREFER_CUBLASLT=1` workaround on torch 2.11.
  It is no longer needed to avoid the older 2.10 failure mode, but it still
  materially improves local 5090 throughput and remains part of the active
  local-script contract.
- Current whole-model backend compare in
  `runs_allama_validation/perf_torch211_backend_compare/`:
  - flash + PyTorch MLP, no local BLAS override:
    - `mean_step_s=2.25897`
    - `tokens_per_s=116045.98`
  - cuDNN + PyTorch MLP, no local BLAS override:
    - `mean_step_s=2.25389`
    - `tokens_per_s=116307.61`
  - flash + PyTorch MLP, with `TORCH_BLAS_PREFER_CUBLASLT=1`:
    - `mean_step_s=1.93177`
    - `tokens_per_s=135701.29`
  - flash + `MLP_KERNEL=triton_gateup`, with `TORCH_BLAS_PREFER_CUBLASLT=1`:
    - `mean_step_s=1.88445`
    - `tokens_per_s=139108.77`
  - cuDNN + PyTorch MLP, with `TORCH_BLAS_PREFER_CUBLASLT=1`:
    - `mean_step_s=4.20607`
    - `tokens_per_s=62325.23`
  - cuDNN + `MLP_KERNEL=triton_gateup`, with `TORCH_BLAS_PREFER_CUBLASLT=1`:
    - `mean_step_s=3.92872`
    - `tokens_per_s=66725.04`
- Conclusion:
  - on torch 2.11, `flash` and `cudnn` SDPA are effectively tied if the local
    BLAS override is removed
  - the old 5090 `cublaslt` knob is no longer a correctness guardrail, but it
    is still a very real local throughput knob on this machine
  - that same local BLAS override should be treated as flash-path-specific on
    this 5090; it catastrophically hurts the local `cudnn` path
  - with the active local 5090 BLAS override restored, `MLP_KERNEL=triton_gateup`
    again beats the plain compiled PyTorch baseline by about `2.5%`

### Torch 2.11 Larger-Boundary Recheck

- Re-ran the larger-boundary benchmark scripts on torch 2.11:
  - `runs_allama_validation/mlp_gateup_block_torch211/summary.json`
  - `runs_allama_validation/attention_block_torch211/summary.json`
- MLP gate/up boundary:
  - forward:
    - compiled reference: `0.40363 ms`
    - compiled custom: `0.36098 ms`
  - backward:
    - compiled reference: `1.34379 ms`
    - compiled custom: `1.03681 ms`
- Attention out-proj boundary:
  - forward:
    - compiled reference: `0.09545 ms`
    - compiled custom: `0.05880 ms`
  - backward:
    - compiled reference: `0.25580 ms`
    - compiled custom: `0.46556 ms`
- Conclusion:
  - torch 2.11 materially improved the standalone larger-boundary picture
  - the MLP gate/up boundary still wins in isolation, including backward
  - the attention out-proj boundary is now a strong forward win, but its
    backward path is still nowhere near acceptable
  - the remaining problem is integration, not the existence of a useful
    larger-boundary kernel candidate
  - a native packed-weight MLP integration was attempted and then removed again;
    it still landed only `112.2k-112.9k tok/s`, below the plain torch 2.11
    baseline

### FlexAttention Torch 2.11 Check

- Added `scripts/benchmark_allama_flex_attention.py` and ran it on the dense
  causal GQA ALlama attention shape.
- Current representative result in
  `runs_allama_validation/flex_attention_torch211/summary.json`:
  - `sdpa_flash`:
    - forward: `0.06802 ms`
    - forward+backward: `0.30628 ms`
  - `sdpa_cudnn`:
    - forward: `0.06859 ms`
    - forward+backward: `0.29532 ms`
  - `flex_triton_auto`:
    - forward: `0.07009 ms`
    - forward+backward: `0.43847 ms`
  - `flex_triton_tuned_tma`:
    - forward: `0.07695 ms`
    - forward+backward: `0.42711 ms`
  - `flex_flash_backend`:
    - fails because the CuTe flash-attention library is not installed in the
      current env
- Conclusion:
  - the built-in Triton flex backend is not a win for dense causal GQA here
  - the only flex path worth revisiting for this project is the FA4/CuTe
    `BACKEND="FLASH"` path after the matching FlashAttention install exists

### Attention Out-Proj Integration Check

- Added real backward kernels for the current post-flash attention boundary
  benchmark:
  - fused branch recompute + reduction for `grad_scale`
  - direct `grad_attn_y` writeback in head-major layout
  - direct `grad_proj_weight_t` without flattening the attention input
- Current representative result in
  `runs_allama_validation/attention_block_v2/summary.json`:
  - forward:
    - compiled reference: `0.05358 ms`
    - compiled custom: `0.05324 ms`
  - backward:
    - compiled reference: `0.24955 ms`
    - compiled custom: `0.31215 ms`
  - numerical drift:
    - forward `max_rel=0.00442`
    - backward `max_rel=0.00338`
- Conclusion:
  - the backward numerics are now fine, so the benchmark is honest
  - compiled forward is essentially tied with compiled PyTorch
  - compiled backward is still about `20.1%` slower
  - this boundary is not ready for model integration yet

### Updated State

- Keep the larger-boundary benchmark scripts.
- Keep the model default on plain compiled PyTorch for now.
- Do not integrate the attention output boundary into `allama_shared.py` yet.
- The benchmark evidence is still useful:
  - MLP gate-up fusion is still the best larger-boundary MLP candidate in
    standalone benchmarks
  - attention out-proj remains the best current attention benchmark, but it is
    still missing enough backward-side efficiency to matter end-to-end
- The next time these come back into the model path, they need either:
  - a stronger custom backward path for the attention boundary
  - a larger attention integration boundary that absorbs more launch overhead
  - a better integration story for the MLP boundary so its standalone win
    survives the full compiled training step

## Pre-Flash Attention Prep Candidate

- Extended `scripts/benchmark_allama_attention_prep.py` with a real Triton
  backward path for the pre-flash boundary:
  - backward through q RMSNorm + RoPE + `q_gain`
  - backward through k RMSNorm + RoPE
  - direct v-gradient writeback into the packed `qkv` tensor
- Current representative result in
  `runs_allama_validation/attention_prep_v2/summary.json`:
  - prep-only forward:
    - compiled reference: `0.03962 ms`
    - Triton: `0.03695 ms`
  - prep-only backward:
    - compiled reference: `0.23278 ms`
    - compiled Triton: `0.31218 ms`
  - prep + flash forward:
    - compiled reference: `0.08915 ms`
    - Triton: `0.11287 ms`
  - prep + flash backward:
    - compiled reference: `0.36928 ms`
    - compiled Triton: `0.42685 ms`
  - relevant end-to-end numerical drift stayed reasonable on the flash-backed
    case:
    - forward `max_rel=0.01031`
    - backward `max_rel=0.01064`
- Conclusion:
  - the fully fused pre-flash backward path is working, but it still does not
    survive the real flash-attention contract under `torch.compile`
  - this boundary is currently worse than the MLP gate-up path and not ready
  for model integration
  - attention-side kernel work should continue only on larger boundaries or
    after a clearer hypothesis about why Inductor still wins once flash is in
    the loop

### 2026-03-30

- Prototyped two real block-scale model integrations in `allama_shared.py`:
  - attention bridge:
    `head-major attn out-proj + residual add/scale + next RMSNorm`
  - full MLP op:
    `gateup + down-proj + residual add/scale`
- These were measured on the real compiled ALlama anchor harness under the
  active local 5090 flash contract:
  - current shipped path, `MLP_KERNEL=triton_gateup`:
    `runs_allama_validation/perf_blockkernels_compare/shipped/allama_anchor/baseline/run_summary.json`
    - `136,838.94 tok/s`
    - `1.91571 s/step`
  - attention bridge + shipped gateup:
    `runs_allama_validation/perf_blockkernels_compare/bridge_gateup/allama_anchor/baseline/run_summary.json`
    - `130,259.74 tok/s`
    - `2.01247 s/step`
    - about `4.8%` slower than shipped
  - full MLP op only:
    `runs_allama_validation/perf_blockkernels_compare/full_only/allama_anchor/baseline/run_summary.json`
    - `121,042.36 tok/s`
    - `2.16572 s/step`
    - about `11.5%` slower than shipped
  - attention bridge + full MLP op:
    `runs_allama_validation/perf_blockkernels_compare/bridge_full/allama_anchor/baseline/run_summary.json`
    - `117,385.70 tok/s`
    - `2.23319 s/step`
    - about `14.2%` slower than shipped
- Interpretation:
  - these are no longer toy kernels; they are real block-scale boundaries in
    the live model path, and they still lost
  - the attention bridge does reduce peak memory materially, but not enough to
    compensate for the step-time regression
  - the branch should not keep these integrations; the clean shipped path
    remains `MLP_KERNEL=triton_gateup`
- Next action:
  - if attention-side performance work continues, the next serious move is to
    use a real FlashAttention package path rather than only SDPA-side wrappers
  - installing `flash-attn` in the train env is now justified so the project
    can test FA2-style direct kernels, packed-QKV paths, and the newer Flex
    `BACKEND="FLASH"` route once compatible pieces are available

### 2026-03-30

- Installed `flash-attn 2.8.3` into the real `train` env and extended
  `scripts/benchmark_allama_flex_attention.py` to cover:
  - `flash_attn_func` on pretransposed `BSHD` tensors
  - `flash_attn_kvpacked_func` on prepacked KV tensors
  - both direct-layout and current-layout-plus-transpose cases
- Current representative result in
  `runs_allama_validation/flex_attention_fa2/summary.json` on the exact ALlama
  attention shape `B=4, H=14, H_kv=2, S=1024, D=64`:
  - `sdpa_flash`:
    - forward: `0.06876 ms`
    - forward+backward: `0.29398 ms`
  - `sdpa_cudnn`:
    - forward: `0.06864 ms`
    - forward+backward: `0.30139 ms`
  - `flash_attn2_pretransposed`:
    - forward: `0.07527 ms`
    - forward+backward: `0.33002 ms`
  - `flash_attn2_kvpacked_pretransposed`:
    - forward: `0.07773 ms`
    - forward+backward: `0.35318 ms`
  - `flash_attn2_with_transpose`:
    - forward: `0.10023 ms`
    - forward+backward: `0.37807 ms`
  - `flash_attn2_kvpacked_with_transpose`:
    - forward: `0.09858 ms`
    - forward+backward: `0.35442 ms`
- Conclusion:
  - FA2 is working and benchmarkable in the project env
  - on the local 5090 and this exact dense causal GQA shape, FA2 does not beat
    compiled SDPA flash or cuDNN
  - because the raw kernel is already slower locally, there is not yet a case
    for wiring FA2 into the model path on this machine
  - the result is still useful because:
    - it rules out “just switch to FA2” as an immediate local fix
    - it leaves open an H100-specific revisit, since official eval is on H100
  - it does not unblock Flex `BACKEND=\"FLASH\"`, which still needs the
      separate CuTe/FA4-capable FlashAttention stack rather than standard FA2

- Re-ran the same backend compare with explicit compiler resets between cases
  in `scripts/benchmark_allama_flex_attention.py` to avoid cross-case compile
  cache contamination.
- Current representative reset-per-case result in
  `runs_allama_validation/flex_attention_fa2_reset/summary.json`:
  - `sdpa_flash`:
    - forward: `0.06839 ms`
    - forward+backward: `0.28924 ms`
  - `sdpa_cudnn`:
    - forward: `0.07744 ms`
    - forward+backward: `0.28942 ms`
  - `flash_attn2_pretransposed`:
    - forward: `0.08310 ms`
    - forward+backward: `0.28953 ms`
  - `flash_attn2_kvpacked_pretransposed`:
    - forward: `0.07657 ms`
    - forward+backward: `0.30617 ms`
  - `flash_attn2_with_transpose`:
    - forward: `0.09826 ms`
    - forward+backward: `0.36999 ms`
- Reset-run conclusion:
  - the cache-clean rerun does not change the local decision
  - pretransposed FA2 comes very close on forward+backward, but it still does
    not beat SDPA flash on this 5090 benchmark
  - if FA2 is revisited from here, it should be as a model-layout adaptation
    question, not as a naïve drop-in backend swap

### 2026-03-30

- Added a real FA2-native attention path to `allama_shared.py` behind
  `ATTN_IMPL=fa2`.
- What changed in that path:
  - Q/K/V stay in `BSHD` layout after the `qkv` projection
  - qk-norm, RoPE, and `q_gain` are applied in the same `BSHD` layout
  - `flash_attn_func` is called directly on the native FA2 layout
  - the path flattens only once for the output projection
- This is deliberately different from the raw FA2 backend benchmark. It answers
  the model-path question rather than the backend-swap question.
- Sequential real-anchor compare on the local 5090 contract:
  - shipped path (`ATTN_IMPL=sdpa`, `MLP_KERNEL=triton_gateup`):
    `runs_allama_validation/perf_fa2_native_confirm_seq/shipped/allama_anchor/baseline/run_summary.json`
    - `134,045.96 tok/s`
    - `1.95563 s/step`
    - `2.1947 GB` peak CUDA memory
  - FA2-native path (`ATTN_IMPL=fa2`, `MLP_KERNEL=triton_gateup`):
    `runs_allama_validation/perf_fa2_native_confirm_seq/fa2_native/allama_anchor/baseline/run_summary.json`
    - `136,490.95 tok/s`
    - `1.92060 s/step`
    - `2.2189 GB` peak CUDA memory
- Interpretation:
  - the positive result survives a clean sequential rerun
  - the gain is modest but real: about `+1.8%` throughput on the full anchor
    training step
  - this is the first attention-side model-path change that actually helps the
    shipped stack end to end
- Practical conclusion:
  - keep `ATTN_IMPL=fa2` as an opt-in path for continued work
  - this does not prove FA2 is globally better than SDPA, but it does prove the
    right model-side adaptation can beat the naïve FA2 backend result

- Added a second FA2-native model path behind `ATTN_IMPL=fa2_kvpacked`.
- What changed in that path:
  - the `qkv` projection tail is viewed directly as packed KV with shape
    `[B, S, 2, H_kv, D]`
  - qk-norm, RoPE, and `q_gain` stay in the same `BSHD` attention layout
  - the model calls `flash_attn_kvpacked_func` directly instead of unpacked
    `flash_attn_func`
- Sequential real-anchor compare on the same local contract:
  - shipped path (`ATTN_IMPL=sdpa`, `MLP_KERNEL=triton_gateup`):
    `runs_allama_validation/perf_fa2_kvpacked_compare_seq/shipped/allama_anchor/baseline/run_summary.json`
    - `134,181.15 tok/s`
    - `1.95366 s/step`
    - `2.1947 GB` peak CUDA memory
  - FA2-native path (`ATTN_IMPL=fa2`, `MLP_KERNEL=triton_gateup`):
    `runs_allama_validation/perf_fa2_kvpacked_compare_seq/fa2_native/allama_anchor/baseline/run_summary.json`
    - `137,055.71 tok/s`
    - `1.91268 s/step`
    - `2.2189 GB` peak CUDA memory
  - FA2 KV-packed path (`ATTN_IMPL=fa2_kvpacked`, `MLP_KERNEL=triton_gateup`):
    `runs_allama_validation/perf_fa2_kvpacked_compare_seq/fa2_kvpacked/allama_anchor/baseline/run_summary.json`
    - `135,765.15 tok/s`
    - `1.93086 s/step`
    - `2.2399 GB` peak CUDA memory
- Interpretation:
  - `fa2_kvpacked` is a real positive model-path adaptation over shipped SDPA
  - it does not beat the simpler `fa2` path on the local 5090; it trails by
    about `0.9%`
  - the packed path remains worth keeping as an H100-facing reference because it
    is architecturally closer to how FA2 wants to consume GQA KV tensors
  - a short profiler compare did not show a dominant explicit packing tax; the
    extra `select/slice/stack` work is tiny, so the local gap appears to come
    from the FA2 kernel path itself rather than the model-side packing glue
