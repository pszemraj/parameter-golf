# Profiling Log

Last updated: 2026-04-07 01:35 EDT

This file records profiler-driven checkpoints that should survive beyond the raw
artifacts under `profiles/`.

## 2026-04-07 — Full-custom CUDA packed-conv backward is the wrong ownership on H100 (`h100k15`)

Bundle:

- H100 bundle:
  - `local-scratch/profiling-out-h100k15-hgdn.7z`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_PACKED_CONV=1`
- family read:
  - exact-length CUDA packed qkv depthwise conv
  - custom forward
  - custom input backward
  - custom weight backward

### Main finding

Reject the full-custom `winner-20260405-19-cuda-packed-conv` family on H100.

Same-day H100 compiled perf:

- controls:
  - `848.75 ms`
  - `848.76 ms`
- candidate:
  - `1796.70 ms`
  - `1793.67 ms`
- mean delta:
  - `848.75 -> 1795.19 ms` (`+111.51%`)

### What went wrong

This was not another frontend split/norm failure. The packed-conv direction is
still the right target stage, but the current backward ownership is wrong.

The important compiled-profile read is:

- `_PackedQKVConvFunctionBackward`:
  - `4161.58 ms`
- `causal_dwconv_weight_backward_kernel`:
  - `3973.02 ms`
- `causal_dwconv_preact_forward_kernel`:
  - `138.52 ms`
- `causal_dwconv_input_backward_kernel`:
  - `133.55 ms`

So the failure is concentrated in the custom weight-backward path, not the
forward kernel itself.

### Scoreboard read

The scoreboard classifies this run as `INTEGRATION_BOTTLENECK`, but the
practical diagnosis is more specific:

- compile-specific penalty:
  - `+751.28 ms`
- compiled copy tax:
  - `-46.51 ms`
- compiled external-kernel self time:
  - `8591.02 ms`

Copy tax actually improved a bit. The family still lost badly because the
custom backward kernels dominated the compiled step.

### Decision

Kill the full-custom packed-conv backward ownership.

Do **not** go back up to standalone frontend fusion ideas.

Reset the packed-conv family with different backward ownership:

- keep the exact-length CUDA packed-conv forward
- keep the narrow fused preact generation
- switch backward to ATen/cuDNN conv grads

That becomes the next live sidecar family.

## 2026-04-07 — Exact-length CUDA packed-conv forward plus ATen backward is locally positive (`rtx4070_phase1_cuda_packedconvaten_fix1`)

Bundle:

- fresh same-day local baseline:
  - `profiles/rtx4070_phase1_winner20260405_19_r5/`
- fresh same-day local candidate:
  - `profiles/rtx4070_phase1_cuda_packedconvaten_fix1/`
- direct comparison:
  - `profiles/rtx4070_phase1_cuda_packedconvaten_fix1/compare_vs_rtx4070_phase1_winner20260405_19_r5/comparison.md`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD=1`
- family read:
  - exact-length CUDA packed qkv depthwise conv forward
  - SiLU backward in packed layout
  - ATen/cuDNN `conv1d_input` and `conv1d_weight` for backward
  - same recurrence-facing contiguous contract as the promoted winner

### Main finding

This ownership change screens positive locally and is the correct next H100
sidecar.

Same-day local phase-1 result:

- trainer `ProfilerStep*`:
  - `6979.99 -> 4128.87 ms` (`-40.85%`)
- trainer `aten::copy_`:
  - `740.89 -> 415.25 ms`
- trainer `aten::mul`:
  - `1256.18 -> 756.61 ms`
- trainer `gdn.recurrence`:
  - `182.11 -> 109.95 ms`
- trainer `aten::convolution_backward`:
  - `182.77 -> 111.10 ms`

### What it got right

The recurrence-facing boundary stayed clean:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv q/k/v` stayed contiguous
- `recurrence_inputs q/k/v` stayed contiguous

And the hotpath improvements line up with the ownership change:

- the old depthwise ATen forward bucket disappears from the trainer view
- the old full-custom weight-backward ownership is gone
- the new path keeps the forward-side copy savings without reopening the
  frontend-NCT integration problem

### Decision

Keep `winner-20260405-19` active until H100 confirms or rejects this.

Promote `winner-20260405-19-cuda-packed-conv-aten-bwd` to the next H100
sidecar batch.

### H100 validation

- `python setup_hgdn_cuda.py build_ext --inplace`
- `python scripts/hgdn_cuda_parity.py`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k16ctl_a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k16ctl_b --offline`
- `python scripts/hgdn.py preflight --preset winner-20260405-19-cuda-packed-conv-aten-bwd --compile-strategy model`
- `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-cuda-packed-conv-aten-bwd --run-prefix h100k16a`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-packed-conv-aten-bwd --run-prefix h100k16a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-packed-conv-aten-bwd --run-prefix h100k16b --offline`
- `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-cuda-packed-conv-aten-bwd --run-prefix h100k16a`

## 2026-04-06 — Composed custom-backward + compile-visible NCT frontend is an H100 integration bottleneck (`h100k14`)

Bundle:

- H100 bundle:
  - `local-scratch/profiling-out-h100k14-hgdn.7z`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
  - `GDN_USE_CUDA_FRONTEND_NCT=1`
- family read:
  - same target stage as the earlier NCT-frontend work
  - still a different composition from standalone `h100k13` because it keeps
    the promoted exact-length packed custom-backward conv path

### Main finding

Reject `winner-20260405-19-cuda-frontend-nct-custom-bwd` on H100.

Same-day H100 compiled perf:

- controls:
  - `878.10 ms`
  - `882.01 ms`
- candidate:
  - `996.66 ms`
  - `996.98 ms`
- mean delta:
  - `880.06 -> 996.82 ms` (`+13.27%`)

### Scoreboard read

This family scores as `INTEGRATION_BOTTLENECK`.

- eager `ProfilerStep*` delta:
  - `-577.25 ms`
- compiled `ProfilerStep*` delta:
  - `+554.10 ms`
- compile-specific penalty:
  - `+1131.36 ms`
- compiled copy tax:
  - `-0.54 ms`
- compiled external-kernel self time:
  - `726.23 ms`
- compiled `CompiledFxGraph` delta:
  - `+542.31 ms`
- compiled `DDP.forward` delta:
  - `+183.48 ms`

### What went wrong

This was not a copy-tax failure. The composed NCT frontend path still improves
the eager view, but the compiled path pays too much around the custom-op
boundary itself.

The important compiled-profile read is:

- `hgdn_cuda_v2::preact_silu_split_l2norm_nct`:
  - `228.01 ms`
- `hgdn_cuda_v2::preact_silu_split_l2norm_nct_backward`:
  - `498.22 ms`
- `aten::copy_` stayed basically flat relative to the promoted winner
- `CompiledFxGraph` and `DDP.forward` both worsened materially

So the correct lesson from `h100k14` is:

- this is still an integration bottleneck at the current boundary
- do not keep iterating on the current standalone NCT-frontend family as if it
  were a near-promotion
- re-open this direction only if the boundary changes materially:
  - a combined conv+frontend kernel
  - or a different compile/decomposition story

### Decision

Keep `winner-20260405-19` active.

Park the current NCT-frontend family on H100.

Move the next H100 sidecar back down to the exact-length CUDA packed-conv
family.

## 2026-04-06 — Exact-length CUDA packed-conv sidecar stays locally positive after `h100k14` (`rtx4070_phase1_cuda_packedconv_fix2`)

Bundle:

- fresh same-day local baseline:
  - `profiles/rtx4070_phase1_winner20260405_19_r4/`
- fresh same-day local candidate:
  - `profiles/rtx4070_phase1_cuda_packedconv_fix2/`
- direct comparison:
  - `profiles/rtx4070_phase1_cuda_packedconv_fix2/compare_vs_rtx4070_phase1_winner20260405_19_r4/comparison.md`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_PACKED_CONV=1`
- purpose:
  - replace the packed qkv causal depthwise conv family itself with the narrow
    exact-length CUDA op and its custom backward
  - keep split, q/k norm, and recurrence in the normal PyTorch path
  - preserve the recurrence-facing contiguous contract

### Main finding

The earlier local packed-conv result was not stale luck. A fresh same-day local
rerun on current HEAD still supports this as the next H100 family.

Same-day local phase-1 result:

- console step average:
  - `3285.35 -> 3092.15 ms` (`-5.88%`)
- trainer `ProfilerStep*`:
  - `5543.80 -> 5119.98 ms` (`-7.64%`)
- peak allocated memory:
  - `6184 -> 6184 MiB`

### What it got right

The recurrence-facing boundary stayed clean:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv q/k/v` stayed contiguous
- `recurrence_inputs q/k/v` stayed contiguous

And the real trainer shell improved against the same-day local control:

- `aten::copy_`:
  - `586.65 -> 308.23 ms`
- `aten::mul`:
  - `995.28 -> 813.31 ms`
- `block.gdn`:
  - `993.14 -> 800.58 ms`
- `gdn.recurrence`:
  - `143.78 -> 142.33 ms`

The old ATen depthwise buckets disappear from the trainer view, which is
expected here because the family ownership changes:

- `aten::convolution_backward` disappears
- `aten::_conv_depthwise2d` disappears
- replaced by:
  - `_PackedQKVConvFunctionBackward`
  - `causal_dwconv_weight_backward_kernel`
  - `gdn.qkv_conv_cuda`

### Decision

Keep `winner-20260405-19` active until H100 confirms or rejects this.

Promote `winner-20260405-19-cuda-packed-conv` to the next H100 sidecar batch.

### H100 validation

- `python setup_hgdn_cuda.py build_ext --inplace`
- `python scripts/hgdn_cuda_parity.py`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k15ctl_a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k15ctl_b --offline`
- `python scripts/hgdn.py preflight --preset winner-20260405-19-cuda-packed-conv --compile-strategy model`
- `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k15a`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k15a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k15b --offline`
- `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k15a`

## 2026-04-06 — Standalone compile-visible NCT frontend is a hard H100 reject (`h100k13`)

Bundle:

- H100 bundle:
  - `local-scratch/profiling-out-h100k13-hgdn.7z`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_FRONTEND_NCT=1`
- important caveat:
  - this candidate did **not** include
    `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
  - so `h100k13` was not a true `k10 + k13` composition

### Main finding

Reject `winner-20260405-19-cuda-frontend-nct` on H100.

Same-day H100 compiled perf:

- controls:
  - `883.57 ms`
  - `883.29 ms`
- candidate:
  - `1050.79 ms`
  - `1051.40 ms`
- mean delta:
  - `883.43 -> 1051.10 ms` (`+18.98%`)

### What went wrong

The compile-visible NCT frontend idea by itself was not enough. The standalone
path gave back the promoted exact-length custom-backward conv win and reopened
the compiled front-end tax.

Important compiled-profile read:

- `aten::copy_` reopened materially
- `hgdn_cuda_v2::preact_silu_split_l2norm_nct`
- `hgdn_cuda_v2::preact_silu_split_l2norm_nct_backward`

So the correct lesson from `h100k13` is not “the NCT frontend idea is bad.” The
correct lesson is:

- standalone `frontend_nct` was the wrong composition
- keep the compile-visible NCT frontend as an ingredient
- pair it with the promoted exact-length packed custom-backward conv path

### Decision

Keep `winner-20260405-19` active.

Reject `winner-20260405-19-cuda-frontend-nct` as a standalone H100 sidecar.

Treat the compile-visible NCT frontend as a building block for a stronger
composed candidate, not as a path to promote by itself.

## 2026-04-06 — Composed custom-backward + compile-visible NCT frontend is the next H100 sidecar locally (`rtx4070_phase1_cuda_frontendnctcustom_fix2`)

Bundle:

- local control bundle:
  - `profiles/rtx4070_phase1_winner20260405_19_r3/`
- local candidate bundle:
  - `profiles/rtx4070_phase1_cuda_frontendnctcustom_fix2/`
- direct comparison:
  - `profiles/rtx4070_phase1_cuda_frontendnctcustom_fix2/compare_vs_rtx4070_phase1_winner20260405_19_r3/comparison.md`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
  - `GDN_USE_CUDA_FRONTEND_NCT=1`
- implementation idea:
  - keep the promoted exact-length packed custom-backward conv path
  - expose `preact_nct` directly to the compile-visible frontend op
  - let the frontend op perform `SiLU + q/k/v split + q/k L2 norm`
  - preserve the recurrence-facing contiguous contract

### Main finding

This is the first true `k10 + k13` composition.

The earlier local result looked promising but had an ambiguous same-day read
because the comparison control was stale. A fresh sequential rerun resolved that
ambiguity.

Same-day local phase-1 result:

- console step average:
  - `3479.42 -> 3042.16 ms` (`-12.57%`)
- peak allocated memory:
  - `6184 -> 5983 MiB`

### What it got right

The recurrence-facing boundary stayed clean:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv q/k/v` stayed contiguous
- `recurrence_inputs q/k/v` stayed contiguous

And the trainer buckets moved in the right direction against the same-day
control:

- `aten::copy_`:
  - `469.96 -> 236.48 ms`
- `aten::mul`:
  - `794.53 -> 511.73 ms`
- `gdn.recurrence`:
  - `115.13 -> 103.36 ms`
- `aten::convolution_backward`:
  - `116.08 -> 104.93 ms`
- `aten::_conv_depthwise2d`:
  - `88.44 -> 80.11 ms`

### Important nuance

The short hybrid forward/backward slice still shows some bucket reshaping:

- `gdn.qkv_frontend_nct_cuda` is now visible
- `gdn.q_norm` moved inside the custom frontend path rather than disappearing
  from the work entirely
- `bare_gdn` recurrence and pointwise buckets are not the main decision source
  here; the real local trainer step is

That is acceptable. The main question is no longer “did the trace move around?”
The main question is whether the composed path reduces the real trainer shell on
target hardware. The same-day local rerun is strong enough to justify asking
H100 that question.

### Decision

Keep `winner-20260405-19` active until H100 confirms or rejects this.

Promote `winner-20260405-19-cuda-frontend-nct-custom-bwd` to the next H100
sidecar.

### H100 validation

- `python setup_hgdn_cuda.py build_ext --inplace`
- `python scripts/hgdn_cuda_parity.py`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k14ctl_a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k14ctl_b --offline`
- `python scripts/hgdn.py preflight --preset winner-20260405-19-cuda-frontend-nct-custom-bwd --compile-strategy model`
- `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-cuda-frontend-nct-custom-bwd --run-prefix h100k14a`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-frontend-nct-custom-bwd --run-prefix h100k14a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-frontend-nct-custom-bwd --run-prefix h100k14b --offline`
- `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-cuda-frontend-nct-custom-bwd --run-prefix h100k14a`

## 2026-04-06 — Compile-visible NCT frontend supersedes the standalone packed-conv sidecar locally (`rtx4070_phase1_cuda_frontendnct_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_cuda_frontendnct_fix1/`
- direct comparison against the older local base:
  - `profiles/rtx4070_phase1_cuda_frontendnct_fix1/compare_vs_rtx4070_cuda_base/comparison.md`
- direct comparison against the earlier low-level CUDA packed-conv sidecar:
  - `profiles/rtx4070_phase1_cuda_frontendnct_fix1/compare_vs_rtx4070_phase1_cuda_packedconv_fix1/comparison.md`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_FRONTEND_NCT=1`
- implementation idea:
  - keep the promoted packed qkv projection
  - keep the depthwise conv itself in the normal ATen conv path
  - move the post-conv `SiLU + q/k/v split + q/k L2 norm` stage one boundary
    earlier into a compile-visible `torch.library` op that consumes
    `preact_nct`
  - explicitly avoid the old k12 failure mode:
    - no packed BTC `.contiguous()` right at the extension boundary
    - no `_dynamo_disable(...)` eager-island wrapper around the new frontend op

### Main finding

This is the first k12-driven follow-up that actually matches the failure
analysis from the H100 data and still looks positive after the full local
phase-1 screen.

Versus the older local base:

- console step average:
  - `3320.37 -> 2944.81 ms` (`-11.31%`)
- `ProfilerStep*` self-device total:
  - `6610.92 -> 5913.20 ms` (`-10.56%`)
- peak allocated memory:
  - `6184 -> 5984 MiB`

Versus the earlier low-level CUDA packed-conv sidecar:

- console step average:
  - `3065.44 -> 2944.81 ms` (`-3.94%`)
- `ProfilerStep*` self-device total:
  - `6126.34 -> 5913.20 ms` (`-3.48%`)

That is enough local evidence to promote this over the standalone packed-conv
sidecar as the next H100 check.

### What it got right

The recurrence-facing contract stayed clean:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv q/k/v` stayed contiguous
- `recurrence_inputs q/k/v` stayed contiguous

And the combined path did reduce real trainer-shell cost against the older
baseline:

- `aten::copy_`:
  - `785.65 -> 591.96 ms`
- `aten::mul`:
  - `1012.30 -> 850.40 ms`
- `gdn.recurrence`:
  - `177.23 -> 172.20 ms`
- `aten::convolution_backward`:
  - `174.81 -> 169.83 ms`
- `aten::_conv_depthwise2d`:
  - `141.31 -> 138.00 ms`

### Important nuance

This did **not** simply dominate every bucket from the older CUDA packed-conv
sidecar.

Compared directly with `rtx4070_phase1_cuda_packedconv_fix1`:

- `aten::copy_` got worse:
  - `369.11 -> 591.96 ms`
- but `aten::mul` got much better:
  - `977.57 -> 850.40 ms`
- and the end-to-end local trainer step still improved:
  - `3065.44 -> 2944.81 ms`

So the right read is not “this solved the front-end.” The right read is:

- the compile-visible NCT boundary is a better combined direction than the
  standalone exact-length packed-conv replacement
- the win mechanism is now mixed enough that H100 should decide the next
  promotion, not more laptop-only guesswork

### Decision

Keep `winner-20260405-19` active until H100 confirms or rejects this.

Supersede `winner-20260405-19-cuda-packed-conv` with
`winner-20260405-19-cuda-frontend-nct` as the next H100 sidecar.

Keep the older packed-conv candidate in-tree as a building block. It is still a
real kernel replacement and could matter again in a larger packed front-end
pipeline.

### H100 validation

Run the same-day control pattern:

- `python setup_hgdn_cuda.py build_ext --inplace`
- `python scripts/hgdn_cuda_parity.py`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k13ctl_a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k13ctl_b --offline`
- `python scripts/hgdn.py preflight --preset winner-20260405-19-cuda-frontend-nct --compile-strategy model`
- `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-cuda-frontend-nct --run-prefix h100k13a`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-frontend-nct --run-prefix h100k13a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-frontend-nct --run-prefix h100k13b --offline`
- `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-cuda-frontend-nct --run-prefix h100k13a`

## 2026-04-06 — Exact-length CUDA packed-conv kernel is locally positive and H100-worthy (`rtx4070_phase1_cuda_packedconv_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_cuda_packedconv_fix1/`
- direct comparison against the active local winner:
  - `profiles/rtx4070_phase1_cuda_packedconv_fix1/compare_vs_rtx4070_cuda_base/comparison.md`

Contract:

- active baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_PACKED_CONV=1`
- implementation idea:
  - keep the promoted packed qkv projection and the exact same recurrence math
  - replace the packed qkv causal depthwise conv itself with a narrow exact-length
    CUDA op and custom backward
  - keep split, q/k norm, and recurrence-facing layout inside the normal PyTorch
    path
  - do not stack on top of the promoted custom-backward path; this is a direct
    replacement candidate

### Main finding

This is the first post-`winner-20260405-19` candidate that looks both genuinely
low-level and locally strong enough to justify another H100 batch.

Local phase-1 result versus `rtx4070_cuda_base`:

- console step average:
  - `3320.37 -> 3065.44 ms` (`-7.68%`)
- `ProfilerStep*` self-device total:
  - `6610.92 -> 6126.34 ms` (`-7.33%`)
- peak allocated memory:
  - `6184 -> 6184 MiB`

That is now outside the rough “treat it as laptop noise” band and the mechanism
is cleaner than the failed split/norm and layout-only attempts.

### What it got right

The new depthwise kernel family is actually active:

- `_PackedQKVConvFunction`
- `_PackedQKVConvFunctionBackward`
- `causal_dwconv_preact_forward_kernel`
- `causal_dwconv_input_backward_kernel`
- `causal_dwconv_weight_backward_kernel`

The recurrence-facing contract also stayed clean:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv q/k/v` stayed contiguous
- `recurrence_inputs q/k/v` stayed contiguous

And the trainer view improved in the right buckets:

- `aten::copy_`:
  - `785.65 -> 369.11 ms`
- `aten::mul`:
  - `1012.30 -> 977.57 ms`
- `gdn.recurrence`:
  - `177.23 -> 169.22 ms`
- `gdn.q_norm`:
  - `48.81 -> 46.86 ms`
- `gdn.k_norm`:
  - `49.14 -> 46.87 ms`

### Important nuance

This did not make the depthwise family itself disappear. It changed how the
front-end shows up:

- old active path:
  - `aten::convolution_backward`
  - `aten::_conv_depthwise2d`
- new candidate:
  - `_PackedQKVConvFunctionBackward`
  - `causal_dwconv_weight_backward_kernel`
  - `gdn.qkv_conv_cuda`

That is acceptable for now because the real trainer step still moved in the
right direction. The question is no longer “does the kernel appear in the
trace?” It does. The question is whether H100 agrees with the local win.

### Decision

Keep `winner-20260405-19` active until H100 confirms or rejects this.

Promote `winner-20260405-19-cuda-packed-conv` to the next H100 sidecar.

### H100 validation

Run the same-day control pattern:

- `python setup_hgdn_cuda.py build_ext --inplace`
- `python scripts/hgdn_cuda_parity.py`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k13ctl_a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k13ctl_b --offline`
- `python scripts/hgdn.py preflight --preset winner-20260405-19-cuda-packed-conv --compile-strategy model`
- `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k13a`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k13a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k13b --offline`
- `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-cuda-packed-conv --run-prefix h100k13a`

### Composition note

Even if the standalone packed-conv candidate later loses on H100, keep it
bookmarked as a real building block. Unlike the Python/layout reshuffles, this
one replaces a real front-end kernel family and preserves the recurrence
boundary contract cleanly, so it is the kind of piece that could matter in a
larger packed front-end pipeline.

## 2026-04-06 — H100 rejects CUDA split+l2norm sidecar (`h100k12`)

Bundle:

- raw artifacts:
  - `local-scratch/profiling-out-h100k12-hgdn.7z`
- extracted review directory:
  - `local-scratch/_inspect_h100k12/`
- control reference:
  - `winner-20260405-19`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_SPLIT_NORM=1`
- purpose of the sidecar:
  - keep the promoted packed qkv front-end and custom depthwise backward
  - replace only the post-conv `split + q/k l2 norm + v materialization`
    stage with a narrow CUDA op
  - leave recurrence math unchanged

### Main finding

This candidate is a hard compiled-H100 reject and should not stay on the active
path.

Same-day compiled perf:

- controls:
  - `879.46 ms`
  - `878.20 ms`
- candidate:
  - `959.13 ms`
  - `961.33 ms`
- mean delta:
  - `878.83 -> 960.23 ms` (`+9.26%`)

The eager profile moved in the tempting direction, but compiled is what decides
this branch, and compiled lost badly.

### Why it lost

The compiled profile says the extension path reintroduced front-end boundary
cost instead of removing enough real work:

- `aten::copy_`:
  - `57.72 -> 210.65 ms`
- new forward island:
  - `_PackedQKVSplitL2NormFunction: 145.83 ms`
- new backward island:
  - `_PackedQKVSplitL2NormFunctionBackward: 194.82 ms`
- the true depthwise conv family barely moved:
  - `aten::convolution_backward: 287.23 -> 287.41 ms`
  - `aten::_conv_depthwise2d: 201.08 -> 201.23 ms`
  - `aten::mul: 53.98 -> 54.06 ms`

So the H100 trace is saying the same thing the earlier extension failures said
in a different form: adding another eager CUDA island on top of the compiled
front-end is not enough. It has to replace a much larger native bucket or stay
out of the way.

### Decision

Reject `winner-20260405-19-cuda-split-norm` on H100.

Keep `winner-20260405-19` as the active HGDN kernel baseline.

### Next direction

The next real kernel pass should be narrower and lower in the stack:

- keep split/norm inside the compiled PyTorch path
- target the exact-length packed causal depthwise conv forward/backward itself
- specifically attack the surviving compiled buckets:
  - `aten::convolution_backward`
  - `aten::_conv_depthwise2d`
  - `conv_depthwise2d_grad_weight_kernel`

This preserves the lesson from the last few runs:

- Python-side layout reshuffles are done
- extension-side split/norm islands are also done
- the next win has to come from replacing a large native depthwise-conv bucket
  directly

## 2026-04-06 — CUDA post-conv split+l2norm kernel is locally positive and H100-worthy (`rtx4070_phase1_cuda_splitnorm_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_cuda_splitnorm_fix1/`
- direct comparison against the active local winner:
  - `profiles/rtx4070_phase1_cuda_splitnorm_fix1/compare_vs_rtx4070_cuda_base/comparison.md`

Contract:

- active baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_USE_CUDA_SPLIT_NORM=1`
- implementation idea:
  - keep the promoted packed qkv front-end and custom depthwise backward
  - leave recurrence math unchanged
  - replace only the post-conv `split + q/k l2 norm + v materialization` stage
    with a narrow CUDA op and custom backward
  - do not revive the rejected fused frontend conv backward

### Main finding

This is the first front-end follow-up since the promotion that is both a real
kernel and locally directionally positive.

Local phase-1 result versus `rtx4070_cuda_base`:

- console step average:
  - `3320.37 -> 3192.68 ms` (`-3.85%`)
- `ProfilerStep*` self-device total:
  - `6610.92 -> 6384.67 ms` (`-3.42%`)
- peak allocated memory:
  - `6184 -> 5984 MiB`

Because this branch is screened on a laptop RTX 4070, that is still inside the
rough local noise band. But unlike the recent layout-only attempts, the
mechanism is plausible enough to justify H100 validation.

### What it got right

The new kernel is actually active in the trace:

- `_PackedQKVSplitL2NormFunction`
- `gdn.qkv_split_norm_cuda`

It also preserved the recurrence-facing contract cleanly:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv` stayed contiguous
- `recurrence_inputs` stayed contiguous

And it materially reduced trainer copy traffic:

- trainer `aten::copy_`:
  - `785.65 -> 604.06 ms`

### What is still unresolved

The new kernel did not produce a clean win everywhere:

- trainer `aten::mul`:
  - `1012.30 -> 1116.85 ms`
- trainer `gdn.recurrence`:
  - `177.23 -> 182.32 ms`
- trainer `aten::convolution_backward`:
  - `174.81 -> 182.88 ms`

So this should not be promoted from the laptop alone.

### Decision

Keep `winner-20260405-19` active.

Promote `winner-20260405-19-cuda-split-norm` to the next H100 sidecar.

### H100 validation

Run the same-day control pattern:

- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k12ctl_a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k12ctl_b --offline`
- `python scripts/hgdn.py preflight --preset winner-20260405-19-cuda-split-norm --compile-strategy model`
- `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-cuda-split-norm --run-prefix h100k12a`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-split-norm --run-prefix h100k12a --offline`
- `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-cuda-split-norm --run-prefix h100k12b --offline`
- `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-cuda-split-norm --run-prefix h100k12a`

### Composition note

Even if this candidate later loses standalone on H100, keep it in mind as a
real composable ingredient. It attacks the post-conv split/norm stage directly,
preserves the recurrence boundary contract, and materially reduces trainer copy
traffic. That makes it a plausible building block for a larger packed front-end
kernel pipeline, unlike the rejected Python-side reshuffles.

## 2026-04-06 — ATen split-copy packed output path rejected locally (`rtx4070_phase1_splitcopy_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_splitcopy_fix1/`
- direct comparison against the active local winner:
  - `profiles/rtx4070_phase1_splitcopy_fix1/compare_vs_rtx4070_cuda_base/comparison.md`

Contract:

- active baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_PACKED_QKV_SPLIT_COPY=1`
- implementation idea:
  - keep the promoted packed qkv front-end and custom backward
  - replace the packed split-plus-three-contiguous path with
    `aten.split_with_sizes_copy`
  - keep recurrence math unchanged
  - keep the packed recurrence-facing q/k/v contract contiguous

### Main finding

This was the right direction conceptually, but it still lost clearly on the
true trainer-eager bundle and should stop locally.

Local phase-1 result versus `rtx4070_cuda_base`:

- console step average:
  - `3320.37 -> 3752.76 ms` (`+13.02%`)
- `ProfilerStep*` self-device total:
  - `6610.92 -> 7546.44 ms` (`+14.15%`)
- peak allocated memory:
  - `6184 -> 6184 MiB`

### What it got right

Unlike the rejected single-contig path, this one preserved the packed conv
boundary cleanly:

- `conv_qkv q/k/v` stayed contiguous
- `norm_qkv` stayed contiguous
- `recurrence_inputs` stayed contiguous

And it did replace the old output materialization path with a real generated
op:

- `gdn.qkv_conv_split_copy`: `2.27 ms` in `hybrid_fwd_bwd`

### Why it still lost

The trainer view got worse in the same places that keep killing these
front-end layout experiments:

- `aten::mul`:
  - `1012.30 -> 1276.17 ms`
- `aten::copy_`:
  - `785.65 -> 798.02 ms`
- `gdn.recurrence`:
  - `177.23 -> 191.34 ms`
- `aten::convolution_backward`:
  - `174.81 -> 206.86 ms`
- `aten::_conv_depthwise2d`:
  - `141.31 -> 159.04 ms`

So this is not another “maybe laptop noise” case. The generated-path split-copy
experiment still made the real training step materially worse.

### Decision

Reject `winner-20260405-19-split-copy` locally.

Do not spend H100 time on it. The next low-level pass should go below this
ATen-only output-path change and target a genuinely lower-level packed output
path, not another layout-only variation.

## 2026-04-06 — H100 sidecar confirms single-contig is not promotable (`h100k11`)

Bundle:

- raw artifacts:
  - `local-scratch/profiling-out-h100k11-hgdn.7z`
- extracted review directory:
  - `local-scratch/_inspect_h100k11/`
- control reference:
  - `local-scratch/_inspect_h100k10/`

Contract:

- active H100 baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_PACKED_QKV_SINGLE_CONTIG=1`
- purpose of the sidecar:
  - local screening was done on a laptop RTX 4070 with non-dedicated GPU usage
  - because that local result sat near the rough laptop noise band, run the full
    same-day H100 control pattern before killing the idea outright
- commands:
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k11ctl_a --offline`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k11ctl_b --offline`
  - `python scripts/hgdn.py preflight --preset winner-20260405-19-single-contig --compile-strategy model`
  - `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19-single-contig --run-prefix h100k11a`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-single-contig --run-prefix h100k11a --offline`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-19-single-contig --run-prefix h100k11b --offline`
  - `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-19-single-contig --run-prefix h100k11a`

### Main finding

The H100 sidecar confirms the local read: this variant is not promotable.

Compiled perf:

- same-day controls:
  - `882.37 ms`
  - `877.13 ms`
- candidate:
  - `876.36 ms`
  - `878.32 ms`
- mean delta:
  - `879.75 -> 877.34 ms` (`-0.27%`)

That is far too small to matter. It sits fully inside the same-day H100 control
spread and does not justify replacing `winner-20260405-19`.

### Why it is not a real win

The saved profiles do not support a meaningful mechanism improvement.

Eager profile vs the promoted `h100k10` winner:

- `ProfilerStep*`: `6561.60 -> 6584.07 ms`
- `gdn.qkv_conv_output_contiguous`:
  - `147.16 ms`
  - replaced by `gdn.qkv_conv_output_contiguous_packed: 153.34 ms`
- `gdn.v_contiguous`:
  - `0.00 -> 16.76 ms`
- `gdn.q_norm`:
  - `77.58 -> 82.71 ms`
- `gdn.k_norm`:
  - `77.74 -> 82.78 ms`
- `aten::copy_`:
  - `972.51 -> 995.69 ms`
- `block.gdn`:
  - `1365.27 -> 1396.78 ms`

Compiled profile vs the promoted `h100k10` winner:

- `ProfilerStep*`: `3453.97 -> 3524.79 ms`
- `aten::copy_`: `57.72 -> 57.15 ms` (flat)
- `aten::mul`: `53.98 -> 54.08 ms` (flat)
- `aten::convolution_backward`: `287.23 -> 287.58 ms` (flat)
- `aten::_conv_depthwise2d`: `201.08 -> 201.35 ms` (flat)

So the sidecar result is not "slightly positive but noisy." It is "effectively
flat on H100 perf, with profiles that do not show a cleaner front-end."

### Decision

Reject `winner-20260405-19-single-contig` after H100 confirmation.

Keep `winner-20260405-19` as the active HGDN kernel preset. The next front-end
attempt should target a lower-level packed output path change, not another
Python-side materialization reshuffle.

## 2026-04-05 — Single packed post-conv materialization rejected locally (`rtx4070_phase1_singlecontig_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_singlecontig_fix1/`
- direct comparison against the active local winner:
  - `profiles/rtx4070_phase1_singlecontig_fix1/compare_vs_rtx4070_cuda_base/comparison.md`

Contract:

- active baseline:
  - `winner-20260405-19`
- candidate delta:
  - `GDN_PACKED_QKV_SINGLE_CONTIG=1`
- implementation idea:
  - keep the promoted packed qkv front-end and custom backward
  - replace three post-conv `q/k/v` contiguous materializations with one packed
    contiguous materialization before split
  - keep the Python-side `l2_norm` path unchanged

### Main finding

This candidate did the intended mechanical thing, but it still lost on the
true trainer-eager phase-1 bundle and should not go to H100.

Local phase-1 result versus `rtx4070_cuda_base`:

- trainer eager self-device total:
  - `25561.13 -> 26793.74 ms` (`+4.82%`)
- console step average:
  - `3320.37 -> 3392.40 ms` (`+2.17%`)
- peak allocated memory:
  - `6184 -> 6280 MiB`

### What improved

- `aten::copy_`:
  - `785.65 -> 727.70 ms`
- `aten::_conv_depthwise2d`:
  - `141.31 -> 135.20 ms`
- `gdn.q_norm`:
  - `48.81 -> 48.21 ms`
- `gdn.k_norm`:
  - `49.14 -> 48.85 ms`

The boundary audit also confirmed that the recurrence-facing contract stayed
clean:

- `norm_qkv` remained contiguous bf16
- `recurrence_inputs` remained contiguous bf16

### Why it still lost

The candidate moved the post-conv contiguity point later, so `conv_qkv` itself
became non-contiguous:

- baseline `conv_qkv q/k/v`:
  - contiguous with stride `(786432, 384, 1)`
- candidate `conv_qkv q/k/v`:
  - non-contiguous views with stride `(2359296, 1152, 1)`

That saved copy time, but the full trainer step paid more elsewhere:

- `aten::mul`:
  - `1012.30 -> 1219.95 ms`
- `gdn.recurrence`:
  - `177.23 -> 179.54 ms`
- `aten::convolution_backward`:
  - `174.81 -> 177.59 ms`

So this was not a free clone-tax removal. It shifted the front-end contract in a
way that made the true training step slower overall.

### Decision

This was enough to reject the variant locally. An optional H100 sidecar was run
later because the screening box is a noisy laptop, and that H100 pass also
confirmed no promotion.

## 2026-04-05 — Packed depthwise custom backward promoted to active H100 winner (`h100k10`)

Bundle:

- raw artifacts:
  - `local-scratch/profiling-out-h100k10-hgdn.7z`
- extracted review directory:
  - `local-scratch/_inspect_h100k10/`
- control references:
  - `local-scratch/_inspect_h100k5/`
  - `local-scratch/_inspect_h100k5v2/`

Contract:

- promoted winner name:
  - `winner-20260405-19`
- promoted env delta relative to `winner-20260405-11`:
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- same-day H100 controls:
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-11 --run-prefix h100k10ctl_a --offline`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-11 --run-prefix h100k10ctl_b --offline`
- candidate validation stack:
  - `python scripts/hgdn.py preflight --preset winner-20260405-11-custom-bwd --compile-strategy model`
  - `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-11-custom-bwd --run-prefix h100k10a`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-11-custom-bwd --run-prefix h100k10a --offline`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-11-custom-bwd --run-prefix h100k10b --offline`
  - `python scripts/hgdn.py h100-profile hybrid --preset winner-20260405-11-custom-bwd --run-prefix h100k10a`

### Main finding

The repeated H100 controls were tight enough to make the custom-backward result
decisive. This path is now the active non-extension winner.

Compiled perf:

- H100 controls:
  - `904.80 ms`
  - `904.12 ms`
- H100 custom-backward candidate:
  - `853.23 ms`
  - `853.20 ms`
- promoted delta:
  - `904.46 -> 853.21 ms` (`-5.67%`)
  - `579,671 -> 614,486 tok/s` (`+6.01%`)

Profiles:

- eager profile step average:
  - `1670.55 -> 1643.53 ms` (`-1.62%`)
- compiled profile step average:
  - `922.04 -> 874.83 ms` (`-5.12%`)

Memory:

- compiled perf peak memory stayed flat:
  - `19601 MiB allocated`
  - `20234 MiB reserved`
- eager profile memory also stayed effectively flat:
  - `23947 -> 23946 MiB allocated`

### Bucket read

The strongest H100 win is step-level copy reduction in compiled mode, not a big
change in the raw depthwise kernels themselves.

Compiled profile vs `h100k5v2`:

- `aten::copy_`: `174.42 -> 57.72 ms`
- direct-copy kernel: `160.69 -> 43.90 ms`
- `aten::convolution_backward`: `288.74 -> 287.23 ms`
- `aten::_conv_depthwise2d`: `202.72 -> 201.08 ms`
- `aten::mm`: `618.45 -> 620.45 ms` (flat)
- flash-attention backward: `420.39 -> 420.27 ms` (flat)

So the custom packed depthwise backward is a real H100 win, but the mechanism
is not "depthwise math got much cheaper." The dominant effect is that the new
path removes a large amount of copy churn around the packed front-end.

### Decision

Promote the custom-backward path as the active HGDN kernel winner:

- `winner-20260405-19`
- config:
  - `configs/hgdn/winner_20260405_19.toml`

Keep the older names for historical reproducibility:

- `winner-20260405-11`
- `winner-20260405-11-custom-bwd`

but use `winner-20260405-19` going forward in docs and launch commands.

Next prepared front-end candidate on top of the promoted winner:

- `winner-20260405-19-single-contig`
- delta:
  - `GDN_PACKED_QKV_SINGLE_CONTIG=1`
- intended mechanism:
  - replace three post-conv q/k/v contiguous materializations with one packed
    contiguous materialization before split
  - keep the promoted packed qkv front-end, custom backward, and Python-side
    `l2_norm` path unchanged otherwise
- status:
  - implementation and tests are in-tree
  - not yet screened locally or on H100

## 2026-04-05 — Packed depthwise custom backward promoted to H100-candidate status (`rtx4070_phase1_custombwd_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_custombwd_fix1/`
- direct comparison against the active non-extension winner:
  - `profiles/rtx4070_phase1_custombwd_fix1/compare_vs_rtx4070_cuda_base/comparison.md`
- stable local compiled perf pair:
  - baseline run id: `rtx4070_perf_currentwinner_cmp`
  - candidate run id: `rtx4070_perf_custombwd_cmp`

Contract:

- GPU: local RTX 4070 laptop
- active baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate delta:
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- important local perf guardrail:
  - `TORCH_BLAS_PREFER_CUBLASLT=1`
- compile mode used for the decision pass:
  - `COMPILE_STRATEGY=model`

### Main finding

This candidate stays alive and is worth H100 confirmation.

The first local read was mixed:

- local phase-1 trainer eager self-device total improved materially
- the short phase-1 trainer console step average moved only slightly and looked inconclusive
- an earlier tiny preflight under `COMPILE_STRATEGY=hybrid` looked much worse

That was not strong enough to promote on its own, so it was re-checked with a
stable compiled local perf pair under the branch's actual compile path:

- baseline compiled perf: `2191.34 ms`
- candidate compiled perf: `2126.57 ms`
- delta: `-64.77 ms` (`-2.96%`)

That resolves the ambiguity well enough to keep the path as the next H100
kernel candidate.

Because this branch is being screened on a laptop RTX 4070, that local win
should still be treated as within the rough `+/-5%` laptop noise band. The
right interpretation is:

- good enough to justify H100 time
- not good enough to promote on local data alone

### Why it is plausible

The custom path keeps the same packed qkv front-end math, but swaps the padded
tail-trimmed packed depthwise conv for an exact-length custom autograd path
with explicit input-grad and weight-grad attribution.

Local phase-1 and hotpath reads both show the expected depthwise reductions:

- trainer eager:
  - `aten::copy_`: `785.65 -> 535.24 ms`
  - `gdn.recurrence`: `177.23 -> 131.03 ms`
  - `aten::convolution_backward`: `174.81 -> 132.06 ms`
  - `aten::_conv_depthwise2d`: `141.31 -> 100.63 ms`
- hybrid forward/backward:
  - `aten::copy_`: `13.58 -> 10.68 ms`
  - `gdn.recurrence`: `46.06 -> 35.17 ms`

The new explicit backward subranges now identify where the packed front-end
backward time is going:

- `gdn.qkv_conv_bwd_silu`
- `gdn.qkv_conv_bwd_input_grad`
- `gdn.qkv_conv_bwd_weight_grad`
- `gdn.qkv_conv_bwd_input_trim`

### Important caveat

Do not treat the earlier `COMPILE_STRATEGY=hybrid` preflight regression as the
main decision signal. The branch-standard compile path is `model`, and that is
the path that produced the stable local perf win.

### Decision

Keep the code as an experimental preset:

- `winner-20260405-11-custom-bwd`

Do not make it the default yet. The next required gate is H100:

1. preflight
2. eager hybrid profile
3. compiled perf run A
4. compiled perf run B
5. compiled profile

## 2026-04-05 — Current-winner packed front-end subrange breakdown (`current_winner_qkvbreakdown`)

Bundle:

- local hotpath-only artifacts:
  - `profiles/current_winner_qkvbreakdown/`
- commit at profiling time:
  - `939a7e7`

Contract:

- GPU: local RTX 4070 laptop
- current confirmed winner:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- scope:
  - instrumentation-only profiler subranges inside `CausalConv1d` and
    `PackedCausalConv1d`
  - no math change, no new kernel path

### Main finding

The remaining packed front-end tax is **not** in transpose, trim, or split view
ops. On the current winner, the real costs are:

- `gdn.qkv_conv_depthwise`
- then `gdn.qkv_conv_output_contiguous`

Everything else in that front-end is small enough that it should not be the
next blind optimization target.

### Hybrid forward/backward read

From `profiles/current_winner_qkvbreakdown/hybrid_fwd_bwd.json`:

- `gdn.q_norm`: `51.07 ms`
- `gdn.recurrence`: `46.64 ms`
- `gdn.g_pointwise`: `27.87 ms`
- `gdn.qkv_conv_depthwise`: `13.76 ms`
- `aten::copy_`: `13.91 ms`
- `aten::mul`: `10.91 ms`
- `gdn.output_norm`: `7.74 ms`
- `gdn.qkv_conv_output_contiguous`: `3.76 ms`
- `gdn.qkv_conv_silu`: `0.42 ms`

Near-zero packed-front-end subranges:

- `gdn.qkv_conv_input_transpose`
- `gdn.qkv_conv_trim`
- `gdn.qkv_conv_output_transpose`
- `gdn.qkv_conv_split`

Interpretation:

- the packed front-end cost is dominated by real kernel work and forced
  materialization
- transpose/trim/split cleanup alone is not going to pay for the remaining
  H100 gap
- the next valid kernel targets stay:
  - packed qkv front-end depthwise work
  - q/k norm
  - gate/output glue
  - remaining copy/materialization churn after those

### Checkpoint policy result

An unfinished follow-up experiment (`packed qk conv + separate v conv`) was
screened after this attribution pass but **not** carried into the stable
checkpoint. The codebase keeps the subrange instrumentation and the confirmed
winner only; the partial candidate was reverted before checkpointing.

## 2026-04-05 — Rejected packed q/k normalization (`rtx4070_phase1_packedqknorm_fix1`)

Bundle:

- local hotpath-only artifacts:
  - `profiles/rtx4070_phase1_packedqknorm_fix1/hotpath/`
- comparison target:
  - `profiles/rtx4070_phase1_ctrlbf16_fix1/`
- commit at experiment start: `281dddc`

Contract:

- GPU: local RTX 4070 laptop
- baseline winner:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate delta:
  - `GDN_USE_PACKED_QK_NORM=1`
- implementation idea:
  - keep the packed qkv tensor alive through packed conv
  - normalize q/k together from one packed buffer
  - avoid separate q- and k-normalization prep

### Main finding

This candidate should be rejected **without** promoting it to a full local
phase-1 bundle.

The quick gate already failed:

- CUDA preflight hybrid eager:
  - baseline winner: about `122.89 ms`
  - candidate: `140.21 ms`
  - delta: about `+14%`

And the isolated bare-GDN path regressed hard:

- bare GDN self-device total:
  - baseline: `227.76 ms`
  - candidate: `552.22 ms`
  - delta: `+142.46%`
- `gdn.recurrence`: `34.71 -> 337.61 ms`
- `gdn.qkv_conv_packed`: `13.45 -> 18.17 ms`
- new `gdn.qk_norm_packed`: `51.28 ms`

That is enough to stop. The candidate is not a clean front-end win.

### Why it was tempting

It directly targeted the two remaining packed-front-end pain points:

1. packed qkv front-end cost
2. q/k normalization cost

And it was semantics-preserving in principle: normalize the same vectors, just
from one packed buffer instead of two split tensors.

### Why it was rejected early

The hotpath views became internally inconsistent:

- bare GDN looked dramatically worse
- hybrid forward/backward looked superficially better in total self-device time
- preflight got slower

That is exactly the kind of mixed signal that should **not** be promoted to a
full local phase-1 run or H100 time. The safe interpretation is:

- the candidate is not robustly better
- it is disturbing the recurrence/front-end balance in a bad way
- the measured upside is not trustworthy enough to justify more screening time

### Decision

Reject `GDN_USE_PACKED_QK_NORM`.

Do not keep the code path. Revert it and return to other kernel targets on the
confirmed winner.

## 2026-04-05 — Rejected packed `w_a/w_b` gate projection (`rtx4070_phase1_packedab_fix1`)

Bundle:

- local candidate bundle:
  - `profiles/rtx4070_phase1_packedab_fix1/`
- direct comparison against current local winner:
  - `profiles/rtx4070_phase1_packedab_fix1/compare_vs_ctrlbf16/comparison.md`
- commit at experiment start: `bf6ab3a`

Contract:

- GPU: local RTX 4070 laptop
- baseline winner:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate delta:
  - `GDN_USE_PACKED_AB_PROJ=1`

### Main finding

Packing the gate-side `w_a`/`w_b` projections into one shared projection is a
**full-step regression** and should not be kept.

Compared against `rtx4070_phase1_ctrlbf16_fix1`:

- trainer eager self-device total:
  - baseline: `16,588.06 ms`
  - candidate: `21,463.22 ms`
  - delta: `+4,875.17 ms` (`+29.39%`)

This is strong enough to reject the candidate without spending any H100 time on
it.

### Why the candidate was tempting

The isolated hotpath views looked good enough to merit one full local
confirmation:

- hybrid forward/backward:
  - `gdn.recurrence`: `48.46 -> 46.87 ms`
  - `gdn.q_norm`: `51.48 -> 47.17 ms`
  - `aten::mul`: `11.82 -> 10.83 ms`
- hybrid optimizer:
  - `aten::mm`: `14.54 -> 13.84 ms`
  - `Optimizer.step#Muon.step`: `115.43 -> 107.09 ms`

So this was a legitimate candidate to screen, not random churn.

### Why it was rejected

The real trainer step got materially worse in the transfer buckets that matter:

- `aten::copy_`: `510.77 -> 756.97 ms`
- `aten::mul`: `659.15 -> 975.56 ms`
- `gdn.qkv_conv_packed`: `234.58 -> 354.25 ms`
- `gdn.recurrence`: `114.97 -> 170.67 ms`
- `aten::convolution_backward`: `113.86 -> 167.50 ms`
- `aten::_conv_depthwise2d`: `91.58 -> 138.70 ms`

Boundary layouts stayed identical across the qkv-to-recurrence path, so this is
not a layout-fix candidate. The packed gate projection simply made the real
training step more expensive.

### Decision

Reject `GDN_USE_PACKED_AB_PROJ`.

Do not promote it to H100. Revert the code-path and keep the current local/H100
winner unchanged.

### Next step

Return to the remaining kernel backlog on the confirmed winner:

1. packed qkv front-end cost on H100
2. q/k norm
3. gate/output glue that does not disturb the packed qkv win
4. residual copy/layout churn after the packed-path improvements

## 2026-04-04 — H100 transfer confirmation for the current local winner (`h100k5`)

Bundle:

- raw artifacts: `local-scratch/profiling-out-h100k5-hgdn.7z`
- extracted eager profile:
  - `profiles/h100k5_profile_eager_hybrid_r1_mlp3.25_seq2048/`
- commit at run time: `6ef5c3f`

Contract:

- GPU: 1xH100
- launcher:
  - `python scripts/hgdn.py h100-profile hybrid-eager --preset current-winner --run-prefix h100k5`
  - `python scripts/hgdn.py h100-perf perf --preset current-winner --run-prefix h100k5 --offline`
- current winner:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`

### Main finding

The current local HGDN kernel winner **does transfer to H100** on compiled
throughput.

Compared against the prior H100 baseline `h100k1_fix2`:

- hybrid compiled perf:
  - old: `997.66 ms`, `525,518 tok/s`
  - new: `901.05 ms`, `581,860 tok/s`
  - delta: `-96.61 ms` (`-9.68%`)
  - throughput delta: `+10.72%`
- attention-only baseline compiled perf:
  - old: `708.95 ms`
  - new: `710.76 ms`
  - delta: `+1.81 ms`

Interpretation:

- the speedup is real and not explained by a looser environment or a faster
  baseline
- the hybrid-vs-depth ratio improved from `1.41x` slower to `1.27x` slower
- this is strong enough to keep the packed-path winner alive and move the branch
  forward from “local-only promising” to “confirmed H100 throughput improvement”

### Eager H100 profile read

The eager H100 profile also moved in the expected direction, but not uniformly.

Clear wins:

- `gdn.recurrence`: `336.60 -> 188.31 ms` (`-44%`)
- `aten::mul`: `1101 -> 974.59 ms` (`-11%`)
- `aten::convolution_backward`: `302.25 -> 289.17 ms` (`-4%`)

Mostly flat:

- `aten::copy_`: `1104 -> 1092.31 ms`
- `aten::_conv_depthwise2d`: `203.99 -> 202.75 ms`
- `attn.norm_rope`: `296.72 -> 296.52 ms`
- `aten::_flash_attention_backward`: `420.77 -> 421.25 ms`
- `gdn.q_norm`: `77.83 -> 77.83 ms`
- `gdn.k_norm`: `77.94 -> 77.94 ms`

Regression / tradeoff:

- the packed front-end is more expensive than the old conv-only HGDN front-end
  on this profile:
  - old `gdn.conv_qkv`: `401.32 ms`
  - new `gdn.qkv_conv_packed`: `544.76 ms`
  - plus `gdn.project_qkv_packed`: `26.29 ms`

Interpretation:

- the packed path is not “free”
- it is winning on H100 because the cheaper recurrence path and reduced
  step-level glue outweigh the more expensive packed qkv front-end
- this is a real systems tradeoff, not just local 4070 noise

### Caveat: missing compiled H100 trace

The archive does **not** contain the compiled H100 hybrid profile for `h100k5`.
Instead, the third command in the pasted run sequence launched a full
`local-phase1` bundle under `/content/parameter-golf/`, which is not part of
the 1xH100 confirmation contract.

So at this checkpoint we have:

- H100 eager hybrid profile: yes
- H100 compiled perf pair: yes
- H100 compiled hybrid trace: **missing**

### Decision

Keep the current winner:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

This is now the first HGDN kernel-path variant that is confirmed to improve the
compiled H100 perf harness.

### Next step

The remaining immediate follow-up is:

1. rerun the missing compiled H100 hybrid profile
2. then run fixed-step quality on the current winner only if the compiled trace
   does not reveal a new regression concern

## 2026-04-04 — H100 compiled-profile confirmation for `h100k5` (`h100k5v2`)

Bundle:

- raw artifacts: `local-scratch/profiling-out-h100k5v2-hgdn.7z`
- extracted compiled profile:
  - `profiles/h100k5_profile_compiled_hybrid_r1_mlp3.25_seq2048/`
- commit at run time: `6ef5c3f`

Contract:

- GPU: 1xH100
- launcher:
  - `python scripts/hgdn.py h100-profile hybrid --preset current-winner --run-prefix h100k5`
- current winner:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`

### Main finding

The compiled H100 profile confirms the same overall story as the eager H100
profile and the compiled perf harness:

- the current winner is genuinely better on H100
- the improvement is especially visible in step-level glue overhead
- the packed front-end remains a tradeoff rather than a free win

### Named compiled-bucket comparison vs `h100k1_fix2`

Clear wins:

- `aten::copy_`: `507.45 -> 174.42 ms` (`-65.6%`)
- `aten::mm`: `648.90 -> 618.45 ms` (`-4.7%`)
- `aten::convolution_backward`: `302.28 -> 288.75 ms` (`-4.5%`)
- `Optimizer.step#Muon.step`: `285.42 -> 258.46 ms` (`-9.4%`)

Mostly flat:

- `aten::_conv_depthwise2d`: `204.38 -> 202.72 ms`
- `aten::_flash_attention_backward`: `420.66 -> 420.39 ms`
- `aten::_flash_attention_forward`: `173.84 -> 174.30 ms`
- `ChunkGatedDeltaRuleFunctionBackward`: `254.41 -> 252.69 ms`
- `ChunkGatedDeltaRuleFunction`: `181.78 -> 180.15 ms`
- `aten::mul`: `54.46 -> 53.99 ms`

Interpretation:

- the compiled trace is even stronger than the eager trace on the key question
  of step-level data movement: compiled `aten::copy_` collapsed dramatically
- recurrence itself is not much faster in the compiled trace, which means the
  packed-path win is not “the recurrence kernel got magically better”
- instead, the packed path appears to help the compiled training step mostly by
  cutting glue/copy overhead while leaving the main recurrence and attention
  kernels roughly stable

### Caveat

The compiled trace is more anonymous than the eager trace because compiled
regions swallow many `record_function` labels. Several `CompiledFxGraph` entries
 are therefore large top-level rows.

That limits exact HGDN-subrange attribution, but it does **not** block the main
 conclusion because the key named buckets we care about are still visible.

### Decision

The current winner is now fully confirmed on H100 for:

- eager profile
- compiled perf harness
- compiled profile

So the branch should treat this as the new HGDN perf reference path until a
better candidate beats it.

### Next step

The next target-hardware question is no longer “does this perf change transfer?”
It is:

1. does the current winner preserve the fixed-step quality advantage?
2. if yes, what is the next HGDN kernel target beyond this packed-path win?

## 2026-04-05 — H100 fixed-step quality confirmation for the current winner (`h100k6`)

Bundle:

- raw artifacts: `local-scratch/profiling-out-h100k6-hgdn.7z`
- extracted logs:
  - `logs/h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048.txt`
  - `logs/h100k6_fixed2k_depth_mlp4.0_seq2048.txt`
- commit at run time: `7677396`
- W&B runs:
  - hybrid: `46fh5oih`
  - depth: `inq87cd4`

Contract:

- GPU: 1xH100
- launcher:
  - `python scripts/hgdn.py h100-perf fixed2k --preset current-winner --run-prefix h100k6 --online --set WANDB_PROJECT=pg-hconv-ablations --set WANDB_WATCH=gradients`
- current winner:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`

### Main finding

The current HGDN kernel winner preserves the fixed-step quality advantage on
H100.

At the matched `2000`-step / `seq=2048` / `train_batch_tokens=524288`
contract:

- hybrid final sampled eval:
  - `val_bpb = 2.4201`
  - `step_avg = 915.10 ms`
- attention-only baseline final sampled eval:
  - `val_bpb = 2.5373`
  - `step_avg = 724.72 ms`

Quality delta:

- hybrid beats the attention-only baseline by `0.1172` bpb at the final
  fixed-step eval
- hybrid also beats the attention-only baseline on the quantized roundtrip
  eval:
  - hybrid: `2.44379288`
  - attention-only baseline: `2.54975979`
  - delta: `0.10596691` bpb

Interpretation:

- the packed HGDN path is not just a throughput win
- the current winner keeps the hybrid quality edge on target hardware
- the speed/quality trade is still favorable even though hybrid remains about
  `1.263x` slower than depth at this contract

### Artifact read

Both models are disqualified on bytes, but hybrid is materially closer to the
limit than the attention-only baseline.

- hybrid artifact:
  - total bytes: `17,580,964`
  - over limit by `1,580,964` bytes (`+9.88%`)
- attention-only baseline artifact:
  - total bytes: `18,553,002`
  - over limit by `2,553,002` bytes (`+15.96%`)

Hybrid is better on bytes by `972,038` bytes (`5.24%` smaller total artifact).

Interpretation:

- the branch should not revert to depth on artifact grounds
- the branch should not revert to the attention-only baseline on artifact
  grounds
- the current winner is good enough to use as the next kernel/profiling
  baseline on H100
- the size target is now concrete: the current winner needs roughly a `10%`
  total artifact reduction without giving back too much of the fixed-step bpb
  advantage
- that byte target does **not** mean the branch is ready to lock model size yet;
  the resize shortlist is prepared, but more HGDN kernel work is still justified

Structured comparison bundle:

- `profiles/fixed2k_compare/h100k6_pair/comparison.md`
- `profiles/fixed2k_compare/h100k6_pair/rows.csv`

One W&B logging quirk to keep in mind for future retune comparisons:

- the final sampled eval and summary metrics are reliable
- the intermediate console evals at `500` and `1500` were not retained in W&B
  history for `h100k6`, so the structured comparison bundle only has `1000` and
  `2000` sampled-eval points

### Decision

Keep the current winner as the HGDN systems baseline:

- it is confirmed faster on H100
- it preserves the quality edge over the attention-only baseline on H100
- it is also the less-over-budget of the two matched fixed-step models

### Next step

The next active phase should be another HGDN kernel/profiling tranche, not
final model-size selection.

Near-term order:

1. keep the current winner as the H100 systems baseline
2. continue HGDN-native hotspot work, especially:
   - packed qkv front-end cost on H100
   - q/k norm and gate/output glue
   - any remaining copy/layout churn that still shows up after the packed-path
     win
3. use the architecture-retune shortlist only as a prepared follow-up once the
   next kernel tranche stops paying off

Immediate design question:

1. can another HGDN-native kernel pass cut the hybrid penalty further on H100?
2. only after that, what is the smallest architecture reduction that recovers
   about `10%` artifact bytes?

## 2026-04-04 — Local HGDN phase-1 attribution (`rtx4070_phase1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1/`
- synthesized analysis: `profiles/rtx4070_phase1/analysis/analysis.md`
- commit at run time: `6992e85`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- hybrid operating point: `16L x 384d`, `GDN_RATIO=1`, `MLP_MULT=3.25`, `seq=2048`
- sequential workflow:
  - CUDA preflight
  - bare GDN hotpath
  - HybridGPT forward/backward hotpath
  - optimizer-only hotpath
  - full trainer eager profile
  - bucket attribution + boundary audit

### Main findings

The first useful result is that the major step-level `aten::copy_` cost is not a
bare-GDN kernel issue.

Observed `aten::copy_`:

- bare GDN: `1.15 ms`
- hybrid forward/backward: `11.84 ms`
- optimizer only: `0.26 ms`
- full trainer eager step: `795.81 ms`

Observed `aten::mul`:

- bare GDN: `0.36 ms`
- hybrid forward/backward: `13.40 ms`
- optimizer only: `2.74 ms`
- full trainer eager step: `1047.57 ms`

Interpretation:

- `copy_` is mostly a full-step integration/training-shell problem, not a bare
  recurrence problem.
- `mul` is also mostly a hybrid-integration plus training-shell issue.
- the genuine HGDN model-path hotspots are still:
  - `gdn.recurrence`
  - `gdn.norm_qkv`
  - `gdn.conv_qkv`
  - `gdn.output_gate`

### Boundary audit

The boundary audit is the most actionable part of the run.

What it showed:

- after `w_q/w_k/w_v`, q/k/v are contiguous `(B, T, D)`
- after `q_conv/k_conv/v_conv`, q/k/v become non-contiguous with stride
  `(786432, 1, 2048)`
- that non-contiguous layout persists through:
  - `norm_qkv`
  - `recurrence_inputs`
- recurrence output `o` becomes contiguous again before the output-gate path

Interpretation:

- the most concrete next target is the **conv-to-recurrence layout path**
- this is stronger evidence than generic “copy is high” reasoning because it
  identifies an exact boundary where layout quality degrades and then persists
  into FLA inputs

### Decision

The next semantics-preserving optimization pass should target:

1. post-conv q/k/v layout cleanup before recurrence
2. only after that, adjacent HGDN glue around `norm_qkv` and `output_gate`

What should not be the first target from this run:

- generic trainer-wide `copy_` chasing without boundary evidence
- optimizer work
- GEMM-focused tuning

### Notes

- this run was intentionally local-only and eager for attribution
- H100 confirmation should happen only after a local conv-to-recurrence layout
  intervention shows a measurable win

## 2026-04-04 — Local conv-output layout variant (`rtx4070_phase1_convcontig`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_convcontig/`
- structured comparison: `profiles/rtx4070_phase1_convcontig/compare_vs_rtx4070_phase1/comparison.md`
- commit at run time: `82be69c`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- one change enabled:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- everything else matched the earlier `rtx4070_phase1` bundle

### Main findings

This is the first boundary variant that produced a clear local win rather than
just moving time around.

Boundary result:

- q/k/v now stay contiguous from `conv_qkv` through `norm_qkv` into
  `recurrence_inputs`
- the old non-contiguous stride `(786432, 1, 2048)` becomes contiguous
  `(786432, 384, 1)` at the conv output

Trainer-eager profiler result:

- total trainer self-device time:
  - baseline: `25,990.59 ms`
  - candidate: `25,258.00 ms`
  - delta: `-732.59 ms` (`-2.82%`)
- targeted trainer buckets:
  - `aten::copy_`: `795.81 -> 776.98 ms`
  - `aten::mul`: `1047.57 -> 1000.68 ms`
  - `gdn.recurrence`: `268.79 -> 194.16 ms`
  - `gdn.norm_qkv`: `113.18 -> 96.47 ms`
  - `gdn.output_gate`: `145.90 -> 141.58 ms`
  - `gdn.conv_qkv`: `236.67 -> 322.01 ms`

Interpretation:

- the explicit contiguous materialization is paying for itself overall
- the improvement is coming from a cheaper recurrence/norm path and slightly
  lower step-level `copy_` / `mul`
- the tradeoff is that `gdn.conv_qkv` itself becomes more expensive because the
  conv path is now doing the explicit layout materialization once up front

### Decision

`GDN_CONV_OUTPUT_CONTIGUOUS=1` is a real local candidate and should stay in the
experiment surface.

Current judgment:

- promising enough for an eventual 1xH100 confirmation
- not ready to become the default path yet
- next local target should move to the gate/output side, because after the
  layout fix the largest remaining bare-GDN bucket is now `gdn.gates`

### Next target

The next local-first investigation should focus on the remaining HGDN fp32
islands and elementwise glue:

1. `gdn.gates`
2. `gdn.output_gate`
3. any unnecessary dtype promotion around `softplus`, `sigmoid`, and
   `rms_norm(o.float()).to(x.dtype)`

## 2026-04-04 — Follow-up local HGDN experiments against `rtx4070_phase1_convcontig`

Bundles:

- gate precision candidate: `profiles/rtx4070_phase1_gatebf16/`
- output-norm precision candidate: `profiles/rtx4070_phase1_outputnormbf16/`
- q/k-only contiguous candidate: `profiles/rtx4070_phase1_qkcontig/`
- `v_conv` ablation candidate: `profiles/rtx4070_phase1_vconv0/`
- commit at run time: `318bc22`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- all comparisons used `rtx4070_phase1_convcontig` as the baseline
- runs were executed sequentially:
  - CUDA preflight
  - hotpath GDN forward/backward
  - hotpath hybrid forward/backward
  - compare against the structured conv-contiguous baseline

### Candidate matrix

1. `GDN_GATES_FP32=0`
2. `GDN_OUTPUT_NORM_FP32=0`
3. `GDN_Q_CONV_OUTPUT_CONTIGUOUS=1`, `GDN_K_CONV_OUTPUT_CONTIGUOUS=1`, `GDN_V_CONV_OUTPUT_CONTIGUOUS=0`
4. `GDN_USE_V_CONV=0`

### Main findings

None of the four follow-up candidates beat the blanket contiguous baseline.

#### `GDN_GATES_FP32=0`

This was the clearest negative result.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `255.20 ms`
  - delta: `+78.04 ms` (`+44.05%`)
- main regressions:
  - `gdn.gates`: `52.45 -> 107.01 ms`
  - `gdn.norm_qkv`: `30.05 -> 48.48 ms`
  - `gdn.recurrence`: `31.57 -> 34.77 ms`

Decision:

- keeping the gate softplus path in fp32 is currently the correct choice

#### `GDN_OUTPUT_NORM_FP32=0`

This improved one local bucket but still lost overall.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `218.98 ms`
  - delta: `+41.82 ms` (`+23.61%`)
- hybrid forward/backward total self-device time:
  - baseline: `1232.48 ms`
  - candidate: `1266.42 ms`
  - delta: `+33.94 ms` (`+2.75%`)
- targeted win:
  - `gdn.output_gate`: `8.29 -> 4.11 ms`
- but offsetting losses:
  - `gdn.norm_qkv`: `34.04 -> 38.57 ms`
  - `gdn.gates`: `39.31 -> 41.70 ms`
  - `aten::copy_`: `11.71 -> 13.74 ms`

Decision:

- the output-norm fp32 island is not the first thing to remove

#### q/k-only contiguous outputs

This was catastrophically worse than making all three conv outputs contiguous.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `511.72 ms`
  - delta: `+334.56 ms` (`+188.84%`)
- hybrid forward/backward total self-device time:
  - baseline: `1232.48 ms`
  - candidate: `1691.34 ms`
  - delta: `+458.86 ms` (`+37.23%`)
- dominant regression:
  - bare `gdn.recurrence`: `31.57 -> 314.31 ms`
  - hybrid `gdn.recurrence`: `43.99 -> 332.35 ms`

Decision:

- q/k-only contiguity is not a “cheaper version” of the contiguous fix
- if contiguity is forced, it needs to be coherent across q/k/v

#### `GDN_USE_V_CONV=0`

This reduced some conv-related work but still lost overall.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `199.40 ms`
  - delta: `+22.24 ms` (`+12.55%`)
- hybrid forward/backward total self-device time:
  - baseline: `1232.48 ms`
  - candidate: `1385.00 ms`
  - delta: `+152.53 ms` (`+12.38%`)
- small local wins:
  - hybrid `aten::copy_`: `11.71 -> 9.25 ms`
  - hybrid `aten::convolution_backward`: `3.17 -> 2.12 ms`
  - hybrid `aten::_conv_depthwise2d`: `2.52 -> 1.68 ms`
- larger regressions:
  - hybrid `gdn.norm_qkv`: `34.04 -> 50.56 ms`
  - hybrid `gdn.output_gate`: `8.29 -> 11.31 ms`
  - hybrid `gdn.recurrence`: `43.99 -> 48.43 ms`

Decision:

- removing `v_conv` is not a free speed win under the current HGDN path

### Decision

The winning local variant is still:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_GATES_FP32=1`
- `GDN_OUTPUT_NORM_FP32=1`
- `GDN_USE_Q_CONV=1`
- `GDN_USE_K_CONV=1`
- `GDN_USE_V_CONV=1`

What these follow-up runs changed in the plan:

- the next target should **not** be another coarse precision toggle
- the next target should **not** be a blind conv ablation
- the next step should be finer subrange attribution inside:
  - `gdn.gates`
  - `gdn.norm_qkv`
  - `gdn.output_gate`
  - optionally q/k/v conv paths separately

That is the missing measurement needed before the next real optimization pass.

## 2026-04-04 — Fine HGDN subrange attribution on the winning local variant (`rtx4070_phase1_convcontig_subranges`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_convcontig_subranges/`
- synthesized analysis: `profiles/rtx4070_phase1_convcontig_subranges/analysis/analysis.md`
- commit at run time: `c0ed0ae`
- analysis/tooling normalization commit: `8f0c626`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed winning config:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_GATES_FP32=1`
  - `GDN_OUTPUT_NORM_FP32=1`
  - q/k/v convs all enabled
- purpose:
  - replace coarse HGDN ranges with finer subranges so the next optimization
    target is based on specific code paths rather than whole-block labels

### Main findings

The subrange pass changed the next-step decision materially.

Under the current winning local variant, the most useful trainer-eager HGDN
sub-buckets are:

- `gdn.recurrence`: `180.70 ms`
- `gdn.q_conv`: `106.75 ms`
- `gdn.k_conv`: `109.86 ms`
- `gdn.v_conv`: `106.40 ms`
- `gdn.output_norm`: `93.83 ms`
- `gdn.q_norm`: `50.53 ms`
- `gdn.k_norm`: `49.46 ms`
- `gdn.output_gate_mul`: `25.32 ms`
- `gdn.output_gate_proj`: `23.32 ms`
- `gdn.g_proj`: `13.23 ms`
- `gdn.g_pointwise`: `6.66 ms`
- `gdn.beta_proj`: `6.28 ms`

Interpretation:

- the remaining cost is **not** primarily the old coarse `gdn.gates` bucket
- once first-call artifacts are separated out, the gate pointwise path is much
  smaller in the real training step than the earlier bare-GDN view suggested
- the major remaining HGDN costs are now:
  - recurrence
  - the three conv paths together
  - q/k normalization
  - output normalization

### Important caution

The fine-subrange pass also showed that bare-GDN microprofiles can overstate
first-call effects.

Examples:

- bare `gdn.g_pointwise`: `44.69 ms`
- trainer-eager `gdn.g_pointwise`: `6.66 ms`
- bare `gdn.q_conv`: `11.92 ms`
- trainer-eager `gdn.q_conv`: `106.75 ms` across `128` calls, with `k/v` now in
  the same ballpark

Decision:

- use bare-GDN hotpaths to find code-path ownership
- use trainer-eager totals to rank remaining payoff on the real training step

### Decision

The next local optimization pass should target **conv-path consolidation**, not
another precision toggle.

Reason:

- recurrence is still the single largest HGDN sub-bucket
- but the three conv paths together are materially larger than recurrence
- they are followed by depthwise-conv backward and `_conv_depthwise2d`
- that makes the highest-value next semantics-preserving experiment:
  - pack q/k/v depthwise causal conv into one shared conv path
  - preserve the current all-path contiguous layout
  - then remeasure recurrence, conv, and copy/mul spillover

What moves down the list after this run:

- more coarse fp32 on/off gate experiments
- q/k-only contiguity experiments
- `v_conv` ablation as the next default step

## 2026-04-04 — Corrected packed qkv conv follow-up (`rtx4070_phase1_packedqkv_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_packedqkv_fix1/`
- structured comparison:
  `profiles/rtx4070_phase1_packedqkv_fix1/compare_vs_rtx4070_phase1_convcontig_subranges/comparison.md`
- packed-fix commit at run time: `b17b80d`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- candidate:
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- purpose:
  - re-run the packed q/k/v depthwise-conv idea after fixing the earlier
    contiguity bug so the candidate is judged on real trainer data rather than
    on a broken layout path

### Main findings

The corrected packed path fixed the original structural flaw.

Boundary result:

- q/k/v are now contiguous after `conv_qkv`
- contiguity is preserved through:
  - `norm_qkv`
  - `recurrence_inputs`
  - `output_proj_input`

That repaired the earlier recurrence/layout regression. The isolated hotpaths
did improve:

- bare GDN total self-device time:
  - baseline: `171.05 ms`
  - candidate: `166.76 ms`
  - delta: `-4.29 ms` (`-2.51%`)
- hybrid forward/backward total self-device time:
  - baseline: `1273.39 ms`
  - candidate: `1162.71 ms`
  - delta: `-110.69 ms` (`-8.69%`)
- hybrid optimizer total self-device time:
  - baseline: `203.39 ms`
  - candidate: `176.65 ms`
  - delta: `-26.74 ms` (`-13.15%`)

But the full trainer-eager step still lost:

- trainer total self-device time:
  - baseline: `25797.53 ms`
  - candidate: `26082.79 ms`
  - delta: `+285.26 ms` (`+1.11%`)

### Why it still lost

The packed candidate bought the expected HGDN-path wins:

- trainer `gdn.recurrence`: `180.70 -> 173.34 ms`
- trainer `gdn.q_norm`: `50.53 -> 48.72 ms`
- trainer `gdn.k_norm`: `49.46 -> 48.99 ms`
- hybrid-fwd-bwd `aten::copy_`: `14.25 -> 11.99 ms`

But those gains were offset by new packing overhead and a slightly worse conv
backward path:

- trainer `gdn.qkv_conv_packed`: `+405.74 ms`
- trainer `aten::cat`: `16.45 -> 99.91 ms`
- trainer `aten::convolution_backward`: `159.30 -> 173.79 ms`
- trainer `aten::_conv_depthwise2d`: `125.02 -> 141.34 ms`
- trainer `aten::copy_`: `776.28 -> 788.81 ms`
- trainer `aten::mul`: `1003.44 -> 1011.34 ms`

Interpretation:

- the packed-conv idea itself is not dead
- but packing only at the conv stage is not enough
- the current implementation still pays too much to build the packed q/k/v
  tensor before the conv, and that cost comes back at trainer scope

### Decision

Do not promote the current packed qkv conv path to H100.

Current judgment:

- fixed packed qkv conv is a **hotpath win but trainer loss**
- keep the contiguity repair in the code because it makes the experiment valid
- reject the current packed-conv-only variant as a local winner

### Next target

If packing is revisited, it needs the next structural step:

1. remove the `aten::cat` tax by emitting packed q/k/v directly from projection
2. then remeasure packed projection + packed conv as one experiment

Absent that, the active default local winner remains:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`

## 2026-04-04 — Packed qkv projection + packed conv (`rtx4070_phase1_packedqkvproj_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_packedqkvproj_fix1/`
- structured comparison:
  `profiles/rtx4070_phase1_packedqkvproj_fix1/compare_vs_rtx4070_phase1_convcontig_subranges/comparison.md`
- experiment-surface commit at run time: `21f4d73`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- candidate:
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- purpose:
  - follow the packed-conv loss with the missing structural patch
  - remove the q/k/v packing tax at projection time instead of building the
    packed tensor with a late `cat` before the conv

### Main findings

This is the first packed-path candidate that wins on the real local training
step.

The microprofiles alone give a mixed story:

- bare GDN total self-device time:
  - baseline: `171.05 ms`
  - candidate: `208.44 ms`
  - delta: `+37.39 ms` (`+21.86%`)
- hybrid forward/backward total self-device time:
  - baseline: `1273.39 ms`
  - candidate: `1319.72 ms`
  - delta: `+46.33 ms` (`+3.64%`)
- hybrid optimizer total self-device time:
  - baseline: `203.39 ms`
  - candidate: `158.05 ms`
  - delta: `-45.34 ms` (`-22.29%`)

But the trainer-eager result is decisive:

- trainer total self-device time:
  - baseline: `25797.53 ms`
  - candidate: `18282.05 ms`
  - delta: `-7515.47 ms` (`-29.13%`)

Interpretation:

- this candidate is a reminder not to over-rank bare-GDN microprofiles
- the packed projection path helps where the real training step was paying,
  which was not visible from the bare recurrence view alone

### Why it won

Compared with the current conv-contiguous baseline, the trainer-eager trace
shows broad HGDN-path reductions:

- `aten::copy_`: `776.28 -> 560.86 ms`
- `aten::mul`: `1003.44 -> 719.87 ms`
- `gdn.recurrence`: `180.70 -> 123.78 ms`
- `aten::convolution_backward`: `159.30 -> 124.35 ms`
- `aten::_conv_depthwise2d`: `125.02 -> 100.19 ms`
- `gdn.q_norm`: `50.53 -> 34.57 ms`
- `gdn.k_norm`: `49.46 -> 34.82 ms`
- `gdn.output_norm`: `93.83 -> 67.79 ms`
- combined separate trainer conv buckets:
  - baseline `gdn.q_conv + gdn.k_conv + gdn.v_conv`: `323.01 ms`
  - candidate `gdn.qkv_conv_packed`: `256.35 ms`

The specific packed-conv-only failure mode was substantially reduced:

- packed-conv-only trainer `aten::cat`: `99.91 ms`
- packed-projection trainer `aten::cat`: `39.25 ms`

So the original hypothesis was correct:

- packed conv alone was not enough
- packed projection was the missing piece that made packing viable at trainer
  scope

### Boundary read

The candidate boundary audit is also coherent:

- `project_qkv` q/k/v are split views over one packed tensor and are not
  contiguous there
- `conv_qkv` restores contiguous q/k/v outputs
- contiguity is preserved through:
  - `norm_qkv`
  - `recurrence_inputs`
  - `output_proj_input`

This is acceptable because the expensive recurrence-facing path is still seeing
the same good layout as the earlier contiguous baseline.

### Decision

Promote this as the new local winner.

Current local winning HGDN speed candidate:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`

### Next target

This candidate has earned target-hardware confirmation.

Next step:

1. re-check the candidate on 1xH100
2. compare against the current H100 hybrid baseline
3. only then decide whether this becomes the default HGDN perf path

## 2026-04-04 — Keep HGDN control projections in bf16 (`rtx4070_phase1_ctrlbf16_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_ctrlbf16_fix1/`
- structured comparisons:
  - `profiles/rtx4070_phase1_ctrlbf16_fix1/compare_vs_refresh1/comparison.md`
  - `profiles/rtx4070_phase1_ctrlbf16_fix1/compare_vs_fix1/comparison.md`
- experiment-surface commit at run time: working tree after `eb08a00`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
- candidate:
  - fixed baseline plus `GDN_CONTROL_PROJ_FP32=0`
- purpose:
  - test whether the remaining trainer `copy_` and `mul` tax was still coming
    from repeated fp32 → bf16 casts on the tiny HGDN control projections
    `w_a/w_b/w_g`

### Main findings

This is a real local win, and it is stronger than the current packed-path
winner.

Compared with the refreshed packed-path baseline
`rtx4070_phase1_packedqkvproj_refresh1`, the full trainer-eager step improved
substantially:

- trainer total self-device time:
  - baseline: `25957.29 ms`
  - candidate: `16588.06 ms`
  - delta: `-9369.23 ms` (`-36.09%`)

Compared with the original packed-path winner
`rtx4070_phase1_packedqkvproj_fix1`, it still wins materially:

- trainer total self-device time:
  - baseline: `18282.05 ms`
  - candidate: `16588.06 ms`
  - delta: `-1693.99 ms` (`-9.27%`)

This is not just one noisy bucket moving around. The step-level HGDN path got
cheaper in multiple places while preserving the same good recurrence-facing
layout.

### Why it won

Against the refreshed baseline, the trainer-eager trace improved in the exact
family of buckets this experiment targeted:

- `aten::copy_`: `800.91 -> 510.77 ms`
- `aten::mul`: `1027.72 -> 659.15 ms`
- `gdn.qkv_conv_packed`: `366.43 -> 234.58 ms`
- `gdn.recurrence`: `176.97 -> 114.97 ms`
- `aten::convolution_backward`: `177.24 -> 113.86 ms`
- `aten::_conv_depthwise2d`: `143.16 -> 91.58 ms`
- `gdn.q_norm`: `49.55 -> 31.74 ms`
- `gdn.k_norm`: `49.84 -> 31.84 ms`
- `gdn.output_norm`: `97.06 -> 62.05 ms`
- `gdn.output_gate_proj`: `21.70 -> 13.45 ms`

The hotpath views are mixed but still consistent with the interpretation:

- bare GDN:
  - `gdn.g_pointwise`: `66.44 -> 49.85 ms`
  - `gdn.recurrence`: `37.30 -> 34.71 ms`
  - `gdn.output_norm`: `14.63 -> 9.29 ms`
- hybrid forward/backward:
  - `gdn.q_norm`: `57.32 -> 51.48 ms`
  - `gdn.g_pointwise`: `31.76 -> 29.34 ms`
  - `gdn.recurrence`: `46.08 -> 48.46 ms` (slightly worse)
  - `gdn.qkv_conv_packed`: `22.14 -> 25.93 ms` (slightly worse)

Interpretation:

- the local winner is still defined by the *real training step*, not the bare
  recurrence microprofile
- keeping `w_a/w_b/w_g` in bf16 appears to remove a large amount of recast
  overhead that was being paid repeatedly across the training shell

### Boundary read

The important part is that the winning packed-path boundary layout stayed
unchanged:

- `conv_qkv` outputs are still contiguous
- `norm_qkv` outputs are still contiguous
- `recurrence_inputs` remain contiguous bf16 tensors
- recurrence/output-gate/output-proj boundaries are unchanged

So this candidate did not win by giving up the packed-path layout cleanup.
It stacked on top of it.

### Decision

Promote this as the new local winner.

Current local winning HGDN speed candidate:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

### Next target

This candidate has earned target-hardware confirmation, but local work can
still continue first.

The next remaining HGDN-native hotspots on the local winner are:

- `gdn.output_norm`
- `gdn.q_norm`
- `gdn.k_norm`

The first normalization-path follow-up was a manual q/k norm experiment, and it
lost on the real training step. That means the next local follow-up should
shift away from hand-rolled q/k normalization and toward the remaining
output-side path before we spend H100 time.

## 2026-04-04 — Manual q/k norm regression (`rtx4070_phase1_qknorm_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_qknorm_fix1/`
- structured comparison:
  - `profiles/rtx4070_phase1_qknorm_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface commit at run time: working tree after `9cb9f25`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus a manual bf16 `q/k` L2 normalization path
- purpose:
  - test whether replacing `F.normalize` with an explicit bf16 pointwise
    normalization path would reduce the remaining `gdn.q_norm` / `gdn.k_norm`
    cost on the local winner

### Main findings

This candidate improved the isolated `q_norm` bucket but lost on the real
trainer step, so it should not remain in the branch.

Bare-GDN and hybrid-forward/backward wins:

- bare `gdn.q_norm`: `54.68 -> 47.86 ms`
- hybrid `gdn.q_norm`: `51.48 -> 46.80 ms`
- hybrid `gdn.recurrence`: `48.46 -> 44.96 ms`

But the full trainer-eager step regressed materially:

- trainer `aten::copy_`: `510.77 -> 791.84 ms`
- trainer `aten::mul`: `659.15 -> 1176.39 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 374.79 ms`
- trainer `gdn.recurrence`: `114.97 -> 183.09 ms`
- trainer `gdn.output_norm`: `62.05 -> 98.79 ms`
- trainer `gdn.q_norm`: `31.74 -> 63.04 ms`
- trainer `gdn.k_norm`: `31.84 -> 63.33 ms`

### Boundary read

The recurrence-facing layout did not improve at all:

- `conv_qkv` stayed contiguous bf16
- `norm_qkv` stayed contiguous bf16
- `recurrence_inputs` stayed contiguous bf16

So the manual q/k path did not buy anything at the boundary level. It only
changed kernel selection inside the normalization path, and the full-step side
effects were strongly negative.

### Decision

Reject this candidate and revert it.

What this means:

- `gdn.q_norm` and `gdn.k_norm` are still important buckets
- but a hand-written bf16 replacement for `F.normalize` is not the right fix
- future work on the norm path should preserve the current operator family
  unless a new experiment shows a step-level win

### Next target

Stay on the current winner:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And move the next local-first investigation to the output-side path:

1. re-screen `GDN_OUTPUT_NORM_FP32=0` on the **current** packed-path winner,
   not the older conv-only baseline
2. if it still loses, stop spending time on dtype-only output-norm changes and
   move to a more structural output-gate consolidation

## 2026-04-04 — Output-norm bf16 retest on the packed-path winner (`rtx4070_phase1_outputnormbf16_packed_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_outputnormbf16_packed_fix1/`
- structured comparison:
  - `profiles/rtx4070_phase1_outputnormbf16_packed_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface commit at run time: `6609643`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `GDN_OUTPUT_NORM_FP32=0`
- purpose:
  - re-test the old output-norm precision idea on the **current** local winner,
    not on the older conv-only baseline

### Main findings

This candidate still loses on the real trainer step.

It does exactly what it is supposed to do locally inside the narrow target:

- bare `gdn.output_norm`: `9.29 -> 0.11 ms`
- hybrid `gdn.output_norm`: `7.72 -> 0.88 ms`
- trainer `gdn.output_norm`: `62.05 -> 49.78 ms`

But the step-level regressions elsewhere are large enough to swamp that win:

- trainer `aten::copy_`: `510.77 -> 728.00 ms`
- trainer `aten::mul`: `659.15 -> 1030.60 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 368.55 ms`
- trainer `gdn.recurrence`: `114.97 -> 179.07 ms`
- trainer `gdn.output_gate_proj`: `13.45 -> 21.13 ms`
- trainer `gdn.output_gate_mul`: `16.57 -> 26.30 ms`

### Boundary read

The boundary audit is unchanged:

- `conv_qkv`, `norm_qkv`, and `recurrence_inputs` stay contiguous bf16
- the output-side tensors stay contiguous bf16 too

So again, this is not fixing a bad layout boundary. It is just changing the
internal kernel mix around the output side, and the overall step gets worse.

### Decision

Reject `GDN_OUTPUT_NORM_FP32=0` again.

What this means:

- dtype-only output-norm changes are not paying off in this branch
- we should stop spending local iterations on output-norm precision toggles
- the next local target needs to be a structural path change, not another fp32
  switch

### Next target

Stay on the current winner:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And move the next local experiment to a structural path cleanup:

1. in-place or otherwise consolidated SiLU work on the packed conv path and/or
   output-gate path
2. if that does not win locally, stop local output-side tuning and spend the
   next budget on H100 confirmation of the current winner

## 2026-04-04 — In-place SiLU structural candidate regressed (`rtx4070_phase1_inplacesilu_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_inplacesilu_fix1/`
- structured comparison:
  - `profiles/rtx4070_phase1_inplacesilu_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface commit at run time: `118aca8`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus structural in-place SiLU on:
    - packed conv activation
    - output-gate activation
- purpose:
  - test whether a small structural cleanup could reduce activation traffic in
    the packed conv path and the output-gate path without changing math

### Main findings

This candidate improved some isolated HGDN buckets but still lost on the real
trainer step.

Local hotpath wins:

- bare `gdn.g_pointwise`: `49.85 -> 36.91 ms`
- bare `gdn.recurrence`: `34.71 -> 30.19 ms`
- bare `gdn.q_norm`: `54.68 -> 27.76 ms`
- hybrid `gdn.recurrence`: `48.46 -> 38.73 ms`
- hybrid `gdn.qkv_conv_packed`: `25.93 -> 21.29 ms`
- hybrid `gdn.output_gate_mul`: `0.33 -> 0.12 ms`

But the full trainer-eager step still regressed materially:

- trainer `aten::copy_`: `510.77 -> 1033.81 ms`
- trainer `aten::mul`: `659.15 -> 1051.13 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 431.57 ms`
- trainer `gdn.recurrence`: `114.97 -> 181.37 ms`
- trainer `gdn.output_norm`: `62.05 -> 98.47 ms`
- trainer `gdn.q_norm`: `31.74 -> 50.31 ms`
- trainer `gdn.k_norm`: `31.84 -> 50.53 ms`

### Boundary read

The boundary audit remained identical to the current winner:

- packed conv outputs stayed contiguous bf16
- `norm_qkv` and `recurrence_inputs` stayed contiguous bf16
- output-gate inputs stayed contiguous bf16

So the loss is not a boundary-layout regression. It is another case where a
micro-level kernel mix change looks promising in isolation but gets worse once
it is inside the real training shell.

### Decision

Reject this candidate and revert it.

This hits the local stop condition for the current output-side loop:

- dtype-only output-norm change lost
- structural in-place SiLU cleanup also lost

### Next target

Keep the current local winner unchanged:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And move the next step to target-hardware confirmation on 1xH100 rather than
burning more local iterations on this output-side path.

## 2026-04-05 — Packed projection with separate convs regressed locally (`rtx4070_phase1_packedproj_sepconv_fix1`)

Bundle:

- raw artifacts:
  - `profiles/rtx4070_phase1_packedproj_sepconv_fix1/hotpath/`
- experiment-surface working tree was reverted after the screen

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=0`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- purpose:
  - test whether the packed projection win could be kept while undoing the
    H100 packed-conv regression

### Main findings

This candidate passed CUDA preflight but failed the local hotpath gate, so it
should not be promoted to the local full-step profile or H100.

The hoped-for tradeoff did not happen. Removing packed conv got rid of the
`gdn.qkv_conv_packed` row, but the replacement front-end and surrounding glue
got materially worse.

Bare-GDN regressions:

- `gdn.q_conv + gdn.k_conv + gdn.v_conv`: `0.00 + 0.00 + 0.00 -> 16.66 + 0.29 + 1.51 ms`
- `gdn.q_norm`: `21.34 -> 49.34 ms`
- `gdn.g_pointwise`: `26.88 -> 64.78 ms`
- `gdn.output_norm`: `5.24 -> 14.10 ms`
- `aten::copy_`: `0.24 -> 0.94 ms`

Hybrid forward/backward regressions:

- `gdn.qkv_conv_packed`: `8.00 -> 0.00 ms`
- `gdn.q_conv + gdn.k_conv + gdn.v_conv`: `0.00 + 0.00 + 0.00 -> 3.72 + 18.54 + 3.22 ms`
- `gdn.recurrence`: `44.23 -> 35.84 ms`
- `gdn.output_norm`: `1.00 -> 21.93 ms`
- `aten::copy_`: `2.75 -> 10.18 ms`
- `aten::mul`: `2.22 -> 12.07 ms`
- `aten::mm`: `2.31 -> 27.12 ms`

Interpretation:

- packed projection by itself is not enough
- on this branch, the separate-conv replacement reintroduced too much front-end
  and glue overhead
- the next HGDN kernel candidate should stay on the current winner and attack a
  different remaining hotspot

### Decision

Reject this candidate and revert it.

It failed the local stop condition:

- intended target did not improve at the hotpath level
- surrounding `copy_`, `mul`, and `mm` overhead got worse
- there is no reason to spend a trainer-eager run or H100 run on it

### Next target

Keep the current winner as the systems baseline:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And shift the next local candidate away from front-end decoupling and toward a
different structural HGDN-native cleanup, with the best current bet being the
gate/output projection path.

## 2026-04-05 — Packed-conv `split_with_sizes -> narrow` cleanup regressed locally (`rtx4070_phase1_narrowsplit_fix1`)

Bundle:

- raw artifacts:
  - `profiles/rtx4070_phase1_narrowsplit_fix1/`
- comparison against the current local winner:
  - `profiles/rtx4070_phase1_narrowsplit_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface working tree was reverted after the screen

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus a packed-conv split cleanup:
    - replace `x.split(self.dims, dim=-1)` with `narrow(...)` views inside
      `PackedCausalConv1d`
- purpose:
  - test whether the H100 compiled `split_with_sizes` / `clone_transpose`
    traces could be reduced by swapping the packed-conv output split primitive
    without changing the packed-path boundary contract

### Main findings

This candidate looked mildly promising in preflight, but it failed the full
local phase-1 gate and should not be promoted to H100.

The local trainer-eager bundle regressed exactly in the transfer buckets that
matter:

- trainer `aten::copy_`: `510.77 -> 653.08 ms`
- trainer `aten::mul`: `659.15 -> 785.87 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 280.92 ms`
- trainer `gdn.recurrence`: `114.97 -> 136.84 ms`
- trainer `aten::convolution_backward`: `113.86 -> 135.11 ms`
- trainer `aten::_conv_depthwise2d`: `91.58 -> 109.67 ms`

There were a few isolated wins, but they did not survive the real step:

- bare `gdn.q_norm`: `54.68 -> 39.23 ms`
- hybrid-forward/backward `gdn.qkv_conv_packed`: `25.93 -> 21.86 ms`
- hybrid optimizer `aten::copy_`: `0.27 -> 0.24 ms`

Interpretation:

- replacing `split_with_sizes` with `narrow` is too small and too local to be
  the real fix for the packed-conv front-end
- the H100 compiled `split_with_sizes` rows are real, but this operator swap
  did not reduce the actual step-level HGDN tax
- the packed front-end still needs a more structural cleanup than this

### Boundary read

The recurrence-facing layout remained identical to the current winner:

- packed conv outputs stayed contiguous bf16
- `norm_qkv` stayed contiguous bf16
- `recurrence_inputs` stayed contiguous bf16
- output-gate inputs stayed contiguous bf16

So this was not a boundary-layout regression. The loss came from the new
kernel mix inside the packed-conv front-end and the surrounding trainer shell.

### Decision

Reject this candidate and revert it.

It failed the actual promotion rule:

- preflight alone was not enough
- the full local trainer-eager bundle got worse
- there is no reason to spend H100 time on it

### Next target

Keep the current winner unchanged:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And shift the next packed-front-end experiment away from post-conv split
micro-optimizations and toward the remaining transpose / clone / conv-input
path around `PackedCausalConv1d`.

## 2026-04-05 — Explicit packed-conv input materialization regressed locally (`rtx4070_phase1_packedconvinput_fix1`)

Bundle:

- raw artifacts:
  - `profiles/rtx4070_phase1_packedconvinput_fix1/`
- comparison against the current local winner:
  - `profiles/rtx4070_phase1_packedconvinput_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface working tree was reverted after the screen

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus an explicit packed-conv input copy:
    - materialize `x.transpose(1, 2).contiguous()` before the packed depthwise
      `Conv1d`
- purpose:
  - test whether the remaining H100 `clone + transpose + convolution` kernels
    were being driven by the non-contiguous packed-conv input rather than the
    post-conv output split

### Main findings

This candidate improved preflight, but it still failed the real local
trainer-eager gate and should not be promoted.

The full local phase-1 bundle regressed in the same trainer buckets that matter
for H100 transfer:

- trainer `aten::copy_`: `510.77 -> 689.51 ms`
- trainer `aten::mul`: `659.15 -> 1000.15 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 356.32 ms`
- trainer `gdn.recurrence`: `114.97 -> 173.27 ms`
- trainer `aten::convolution_backward`: `113.86 -> 172.55 ms`
- trainer `aten::_conv_depthwise2d`: `91.58 -> 139.24 ms`
- trainer `gdn.q_norm`: `31.74 -> 48.31 ms`
- trainer `gdn.k_norm`: `31.84 -> 48.37 ms`

There were still local screen wins:

- preflight `hybrid_eager`: `146.49 -> 133.88 ms`
- bare `aten::copy_`: `1.18 -> 0.95 ms`
- bare `gdn.q_norm`: `54.68 -> 46.28 ms`
- hybrid optimizer `aten::copy_`: `0.27 -> 0.24 ms`

Interpretation:

- the packed front-end remains a real hotspot, but an explicit input-side
  `.contiguous()` is not the right fix
- this is the second straight packed-conv structural tweak that looked better
  in preflight and worse in the real trainer step
- the remaining packed-front-end tax is not going to yield to obvious
  transpose/split micro-surgery

### Boundary read

The recurrence-facing boundary again stayed identical to the current winner:

- packed conv outputs stayed contiguous bf16
- `norm_qkv` stayed contiguous bf16
- `recurrence_inputs` stayed contiguous bf16
- output-gate inputs stayed contiguous bf16

So the loss was not a recurrence-boundary regression. It came from the changed
kernel mix around the packed-conv front-end and its downstream shell overhead.

### Decision

Reject this candidate and revert it.

It failed the same promotion rule as the `split -> narrow` attempt:

- preflight was not enough
- hotpath looked interesting enough to escalate
- the real trainer-eager bundle got materially worse

### Next target

Keep the current winner unchanged:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And move the next hypothesis away from packed-conv input/output surgery and
toward the repeated fp32 shell/control casts that still show up as
`to_copy + add + mul + unsqueeze` style kernels in the compiled H100 trace.

## 2026-04-05 — Residual-shell bf16 candidate regressed locally (`rtx4070_phase1_blockshellbf16_fix1`)

Bundle:

- raw artifacts:
  - `profiles/rtx4070_phase1_blockshellbf16_fix1/`
- comparison against the current local winner:
  - `profiles/rtx4070_phase1_blockshellbf16_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface working tree was reverted after the screen

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `BLOCK_SHELL_FP32=0`
  - keep residual-shell control params on activation dtype instead of
    restoring them to fp32:
    - `attn_scale`
    - `mlp_scale`
    - `resid_mix`
    - `q_gain`
    - `skip_weights`
- purpose:
  - test whether the remaining compiled H100 `to_copy + add + mul +
    unsqueeze` style kernels were being driven by repeated forward-side casts
    of the residual shell rather than the HGDN recurrence path itself

### Main findings

This candidate looked real in preflight and hotpath, then failed the actual
trainer-eager gate.

Local screens that looked promising:

- preflight:
  - `hybrid_eager`: `146.49 -> 124.23 ms`
  - `hybrid_compiled`: `3223.56 -> 1689.94 ms`
- hotpath:
  - bare GDN total: `227.76 -> 221.99 ms`
  - hybrid forward/backward total: `1467.43 -> 336.86 ms`
  - hybrid optimizer total: `190.68 -> 134.76 ms`

But the real local trainer-eager bundle regressed across the transfer buckets
that decide promotion:

- trainer `aten::copy_`: `510.77 -> 771.96 ms`
- trainer `aten::mul`: `659.15 -> 1000.82 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 357.30 ms`
- trainer `gdn.recurrence`: `114.97 -> 173.41 ms`
- trainer `aten::convolution_backward`: `113.86 -> 173.00 ms`
- trainer `aten::_conv_depthwise2d`: `91.58 -> 139.65 ms`
- trainer `gdn.q_norm`: `31.74 -> 48.22 ms`
- trainer `gdn.k_norm`: `31.84 -> 48.20 ms`

### Boundary read

The recurrence-facing layout stayed identical to the current winner:

- packed conv outputs stayed contiguous bf16
- `norm_qkv` stayed contiguous bf16
- `recurrence_inputs` stayed contiguous bf16
- output-gate inputs stayed contiguous bf16

So this was not a layout-boundary failure. The regression came from the changed
kernel mix and/or numerics after removing fp32 restoration from the residual
shell.

### Interpretation

The residual shell is not a free cast-elimination target.

Two conclusions matter:

- the remaining compiled `to_copy + add + mul + unsqueeze` style kernels are
  not evidence that the whole residual shell should simply be moved to bf16
- preflight plus hotpath wins are still insufficient when the true trainer step
  says the shell change made the real training path worse

This result does **not** prove that every smaller shell-side change is bad. It
does rule out the broad "`BLOCK_SHELL_FP32=0`" version as a promotion
candidate.

### Decision

Reject this candidate and revert it.

### Next target

Keep the current winner unchanged:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And move the next hypothesis to a narrower HGDN-native target:

- either a smaller shell-side carve-out rather than the full residual shell
- or another packed-front-end / norm / gate path that is still showing up in
  the H100 traces

## 2026-04-05 — Manual fp32 q/k norm regressed at preflight (`GDN_MANUAL_QK_NORM=1`)

Screen:

- type:
  - local preflight-only quick screen
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `GDN_MANUAL_QK_NORM=1`
  - replace the generic `F.normalize` q/k path with an explicit fp32
    `square -> sum -> rsqrt -> mul` formulation

### Main findings

This candidate failed at the first gate and was not promoted to hotpath or full
phase-1.

Preflight deltas versus the current winner:

- `gdn_eager`: `1032.32 -> 1092.91 ms`
- `hybrid_eager`: `146.49 -> 151.18 ms`
- `hybrid_compiled`: `3223.56 -> 4966.38 ms`

The compiled preflight regression is especially decisive here. The manual norm
path did not merely fail to help; it made the compiled path dramatically worse.

### Decision

Reject this candidate at the preflight gate and revert it.

### Interpretation

This rules out the simple "replace `F.normalize` with a manual fp32 rsqrt path"
version of q/k norm cleanup. If q/k norm gets revisited again, it should be a
more targeted formulation than this broad manual replacement.

## 2026-04-05 — `resid_mix`-only bf16 carve-out regressed at preflight (`RESID_MIX_FP32=0`)

Screen:

- type:
  - local preflight-only quick screen
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `RESID_MIX_FP32=0`
  - keep `resid_mix` on activation dtype while leaving the rest of the
    residual shell on the default fp32 restore path

### Main findings

This narrower shell-side carve-out also failed at the first gate and was not
promoted.

Preflight deltas versus the current winner:

- `gdn_eager`: `1032.32 -> 1076.62 ms`
- `hybrid_eager`: `146.49 -> 143.97 ms`
- `hybrid_compiled`: `3223.56 -> 9879.93 ms`

The decisive signal is again the compiled preflight. Even though eager was only
slightly mixed, compiled performance collapsed hard enough that this is not a
reasonable next candidate.

### Decision

Reject this candidate at the preflight gate and revert it.

### Interpretation

This is strong evidence that the remaining compiled `add + mul + unsqueeze`
shell kernels are not going to be solved by simply dropping fp32 restoration on
`resid_mix`. The next meaningful branch of work should move back toward the
packed-conv implementation itself rather than more shell-side dtype tweaks.

## 2026-04-05 — Manual packed-qkv shift-conv regressed at preflight (`GDN_USE_MANUAL_PACKED_QKV_CONV=1`)

Screen:

- type:
  - local preflight-only quick screen
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `GDN_USE_MANUAL_PACKED_QKV_CONV=1`
  - replace the packed depthwise `Conv1d` with an explicit causal
    shift-and-sum implementation that reuses the same packed conv weights

### Main findings

This was the first real packed-front-end rewrite candidate after the shell-side
 and q/k-norm failures, and it still failed immediately at preflight.

Preflight deltas versus the current winner:

- `gdn_eager`: `1032.32 -> 1136.29 ms`
- `hybrid_eager`: `146.49 -> 150.94 ms`
- `hybrid_compiled`: `3223.56 -> 7863.94 ms`

The compiled preflight regression is again decisive. The manual packed shift
kernel did not reduce the packed-front-end tax; it made the compiled path much
worse.

### Decision

Reject this candidate at the preflight gate and revert it.

### Interpretation

This rules out the simple high-level shift-and-sum rewrite of the packed
depthwise conv. If the next packed-front-end attempt stays in-repo, it likely
needs to be lower-level or otherwise much closer to the actual kernel/memory
behavior we are trying to fix, rather than another high-level algebraic rewrite
of the same operation.

## 2026-04-05 — Delayed packed qkv split regressed at preflight (`GDN_DELAY_QKV_SPLIT=1`)

Screen:

- type:
  - local preflight-only quick screen
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `GDN_DELAY_QKV_SPLIT=1`
  - keep one packed q/k/v buffer through the packed depthwise conv and delay
    the q/k/v split until recurrence prep

### Main findings

This was the first structural packed-front-end candidate that explicitly tried
to remove the immediate post-conv q/k/v clone tax without changing the
normalization operator family. It still failed decisively at the preflight
gate.

Preflight deltas versus the current winner:

- `gdn_eager`: `1032.32 -> 1546.84 ms`
- `hybrid_eager`: `146.49 -> 211.45 ms`
- `hybrid_compiled`: `3223.56 -> 5258.08 ms`

### Decision

Reject this candidate at the preflight gate and revert it.

### Interpretation

Even though the H100 compiled trace still contains `clone/split/transposed`
front-end rows, delaying the q/k/v split inside the current Python-level packed
path is not the right fix. The current packed front-end appears to rely on the
existing post-conv materialization pattern more than expected.

This shifts the next front-end attempt away from split timing and toward either:

- a lower-level packed conv implementation, or
- a different HGDN-native path entirely, such as a projection/gate-side
  structural cleanup that does not disturb the current packed conv contract

## 2026-04-05 — FLA in-kernel q/k l2 norm regressed locally (`GDN_QK_L2NORM_IN_KERNEL=1`)

Bundle:

- raw artifacts:
  - `profiles/rtx4070_phase1_qknorminkernel_fix1/`
- comparison against the current local winner:
  - `profiles/rtx4070_phase1_qknorminkernel_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface working tree was reverted after the screen

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus `GDN_QK_L2NORM_IN_KERNEL=1`
  - skip Python-side `l2_norm(q)` / `l2_norm(k)` and delegate q/k
    normalization to the FLA kernel via
    `use_qk_l2norm_in_kernel=True`
- purpose:
  - test whether moving q/k normalization into the recurrence backend would
    reduce HGDN glue overhead and improve the packed-path winner without
    disturbing the current recurrence-facing layout

### Main findings

This candidate looked promising at steady-state preflight, then failed
decisively at the full local phase-1 gate.

Preflight versus the current winner:

- `gdn_eager`: `1032.32 -> 1064.78 ms`
- `hybrid_eager`: `146.49 -> 153.04 ms`
- `hybrid_compiled`: `3223.56 -> 1808.31 ms`

The preflight compiled number was directionally interesting, but the real
trainer-eager bundle regressed badly:

- trainer `aten::copy_`: `510.77 -> 662.86 ms`
- trainer `aten::mul`: `659.15 -> 888.74 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 365.93 ms`
- trainer `gdn.recurrence`: `114.97 -> 204.03 ms`
- trainer `aten::convolution_backward`: `113.86 -> 177.45 ms`
- trainer `aten::_conv_depthwise2d`: `91.58 -> 143.26 ms`
- trainer step timing from the console moved from the accepted winner's
  sub-`1s` regime to roughly `3.0s` per step on the same local phase-1
  contract

The hybrid forward/backward microprofile also showed the same failure mode:

- `gdn.recurrence`: `48.46 -> 5136.19 ms`

### Boundary read

The recurrence-facing layout stayed unchanged:

- packed conv outputs stayed contiguous bf16
- `norm_qkv` stayed contiguous bf16
- `recurrence_inputs` stayed contiguous bf16
- output-side tensors stayed contiguous bf16

So this was not a layout-fix candidate that accidentally broke contiguity. It
changed the kernel mix behind the recurrence call and made the actual training
path much worse.

### Interpretation

The FLA in-kernel q/k normalization switch is not a free replacement for the
current Python-side `l2_norm` path in this HGDN training stack.

Two conclusions matter:

- the current packed-path winner is relying on the existing q/k normalization
  behavior more than the preflight alone suggested
- for this branch, a better compiled micro-case is irrelevant if the true
  trainer-eager step gets much slower

### Decision

Reject this candidate and revert it.

### Next target

Keep the current winner unchanged:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And return to the remaining HGDN-native targets that are still open after the
packed-path win:

- packed qkv front-end cost
- gate/output glue
- remaining copy/layout churn

## 2026-04-05 — Optional fused CUDA front-end/output path imported and passed the local phase-1 gate

Bundles:

- baseline:
  - `profiles/rtx4070_cuda_base/`
- fused candidate:
  - `profiles/rtx4070_cuda_fused/`
- structured comparison:
  - `profiles/rtx4070_cuda_fused/compare_vs_rtx4070_cuda_base/comparison.md`

Implementation surface:

- optional extension package:
  - `hgdn_cuda/`
- build entrypoint:
  - `setup_hgdn_cuda.py`
- validation helpers:
  - `scripts/build_hgdn_cuda.sh`
  - `scripts/hgdn_cuda_parity.py`
- preset:
  - `configs/hgdn/current_winner_cuda_fused.toml`

Flags:

- baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- fused candidate:
  - baseline plus:
    - `GDN_OUTPUT_NORM_FP32=1`
    - `GDN_USE_CUDA_FUSED_FRONTEND=1`
    - `GDN_USE_CUDA_FUSED_OUTPUT=1`

### What was imported

The external branch added two optional fused paths:

- packed HGDN front-end:
  - packed post-projection qkv buffer
  - depthwise causal conv
  - SiLU
  - split
  - q/k normalization
- output glue:
  - `RMSNorm(o) * SiLU(g_out)`

This branch adaptation kept the extension optional and added compile-safe
eager-island wrapping so TorchDynamo does not try to trace the raw pybind
extension directly.

### Local validation

Passed locally in `pg`:

- `python setup_hgdn_cuda.py build_ext --inplace`
- `python scripts/hgdn_cuda_parity.py`
- `python test_model.py`

The direct parity script passed with the extension loaded. CPU fallback and
flag-validation coverage were also added to `test_model.py`.

### Main findings

This candidate passed the local phase-1 gate and is worth H100 validation.

Local trainer-eager result:

- self-device total:
  - `25561.13 -> 21352.84 ms` (`-16.46%`)
- console step average:
  - `3320.37 -> 2804.76 ms` (`-15.53%`)
- peak allocated memory:
  - `6184 -> 5696 MiB`

The most important trainer-eager replacement on the current winner was the
front-end/output glue cluster:

- baseline cluster:
  - `gdn.qkv_conv_depthwise = 228.25 ms`
  - `gdn.qkv_conv_output_contiguous = 89.32 ms`
  - `gdn.q_norm = 48.81 ms`
  - `gdn.k_norm = 49.14 ms`
  - `gdn.output_norm = 95.59 ms`
  - `gdn.output_gate_mul = 25.46 ms`
  - total `536.57 ms`
- fused replacement:
  - `gdn.qkv_frontend_fused = 158.06 ms`
  - `gdn.output_fused = 68.34 ms`
  - total `226.40 ms`

Trainer-eager glue buckets also moved materially:

- `aten::copy_`: `785.65 -> 137.73 ms`
- `aten::mul`: `1012.30 -> 837.35 ms`

The hybrid forward/backward hotpath also improved overall:

- self-device total:
  - `1354.73 -> 1258.45 ms` (`-7.11%`)

Important nuance:

- the hotpath view still shows `gdn.q_norm` prominently
- the real local keep/drop call came from the trainer-eager step-level result,
  not from a clean disappearance of every HGDN-native micro-bucket

### Compile note

The raw external pybind integration is not acceptable as-is.

Before adapting the branch-local integration, the same fused path produced:

- TorchDynamo tracing warnings
- a large compiled-preflight regression on the local machine

The accepted branch version wraps the extension dispatch in
`torch._dynamo.disable(...)`, which restored a compile-safe local path and let
the fused candidate pass the real local gate.

### Decision

Keep the optional fused CUDA path in-tree as an experimental candidate.

Do not promote it to the default winner yet.

### H100 gate (`h100k8`)

Bundle:

- raw artifacts: `local-scratch/profiling-out-h100k8-hgdn.7z`
- extracted profiles:
  - `local-scratch/_inspect_h100k8/profiles/h100k8_profile_eager_hybrid_r1_mlp3.25_seq2048/`
  - `local-scratch/_inspect_h100k8/profiles/h100k8_profile_compiled_hybrid_r1_mlp3.25_seq2048/`
- extracted logs:
  - `local-scratch/_inspect_h100k8/logs/h100k8_profile_eager_hybrid_r1_mlp3.25_seq2048.txt`
  - `local-scratch/_inspect_h100k8/logs/h100k8_perf_hybrid_r1_mlp3.25_seq2048.txt`
  - `local-scratch/_inspect_h100k8/logs/h100k8_profile_compiled_hybrid_r1_mlp3.25_seq2048.txt`

Contract:

- H100 build/parity:
  - `python setup_hgdn_cuda.py build_ext --inplace`
  - `python scripts/hgdn_cuda_parity.py`
- H100 fused preset:
  - `python scripts/hgdn.py preflight --preset current-winner-cuda-fused --compile-strategy hybrid`
  - `python scripts/hgdn.py h100-profile hybrid-eager --preset current-winner-cuda-fused --run-prefix h100k8`
  - `python scripts/hgdn.py h100-perf perf --preset current-winner-cuda-fused --run-prefix h100k8 --offline`
  - `python scripts/hgdn.py h100-profile hybrid --preset current-winner-cuda-fused --run-prefix h100k8`

Main finding:

- build passed on H100
- direct parity passed on H100
- fused preflight passed on H100
- the fused preset is still a **clear H100 performance regression**

Compared against the current winner `h100k5`:

- eager hybrid profile step average:
  - `1670.55 -> 2248.98 ms` (`+34.6%`)
- compiled hybrid perf:
  - `901.05 -> 1863.41 ms` (`+106.8%`)
  - `581,860 -> 281,359 tok/s` (`-51.6%`)

Named-bucket read:

- things that did get cheaper:
  - eager `aten::copy_`: `1092.31 -> 184.25 ms`
  - compiled `aten::copy_`: `174.42 -> 34.37 ms`
  - eager `aten::_fused_rms_norm`: `293.18 -> 188.02 ms`
  - eager `aten::_fused_rms_norm_backward`: `293.42 -> 203.51 ms`
- but the fused frontend backward exploded and dominated the step:
  - eager `_PackedQKVFrontendFunctionBackward = 4341.02 ms`
  - eager `causal_dwconv_weight_backward = 3983.38 ms`
  - compiled `_PackedQKVFrontendFunctionBackward = 4316.51 ms`
  - compiled `causal_dwconv_weight_backward = 3959.04 ms`

Interpretation:

- this is not just a compile-mode problem
- the fused packed frontend itself is currently too expensive on H100,
  especially in backward
- output-side fusion may still be individually salvageable, but the combined
  frontend+output preset is not competitive

Decision:

Keep the optional fused CUDA extension in-tree for future work, but reject
`current-winner-cuda-fused` as an active H100 candidate.

Next step:

Return to the non-extension current winner for the active kernel path.

If the extension is revisited later, the first targets are:

1. packed frontend backward
2. depthwise-conv weight gradient
3. output fusion only after the frontend backward is no longer dominant

## 2026-04-05 — Output-only fused preset added as the next salvage experiment

Checkpoint:

- preset added:
  - `current-winner-cuda-output-only`
- config added:
  - `configs/hgdn/current_winner_cuda_output_only.toml`
- local validation:
  - `conda run -s --name pg python scripts/hgdn.py preflight --preset current-winner-cuda-output-only --compile-strategy hybrid`
  - extension loaded
  - preflight passed

Interpretation:

- the output-only fused path is now wired as an isolated experiment surface
- this does **not** mean it is a local or H100 win yet
- the next actual keep/drop gate is a local phase-1 run against the non-extension current winner

## 2026-04-05 — Output-only fused preset failed the local phase-1 gate

Bundle:

- candidate:
  - `profiles/rtx4070_cuda_output_only/`
- structured comparison:
  - `profiles/rtx4070_cuda_output_only/compare_vs_rtx4070_cuda_base/comparison.md`
- baseline:
  - `profiles/rtx4070_cuda_base/`

Contract:

- launcher:
  - `conda run -s --name pg python scripts/hgdn.py local-phase1 --preset current-winner-cuda-output-only --run-prefix rtx4070_cuda_output_only`
- candidate flags:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
  - `GDN_OUTPUT_NORM_FP32=1`
  - `GDN_USE_CUDA_FUSED_OUTPUT=1`

Main finding:

The output-only fused preset is a clean local wiring success, but it still loses
slightly at the full-step level and does **not** earn an H100 run.

Compared against the non-extension current winner:

- trainer eager self-device total:
  - `25561.13 -> 25978.10 ms` (`+1.63%`)
- trainer eager console step average:
  - `3320.37 -> 3379.27 ms` (`+1.77%`)
- trainer eager peak allocated memory:
  - `6184 -> 5896 MiB`

What did improve:

- `gdn.output_norm`: `95.59 -> 0.00 ms`
- `gdn.output_gate_mul`: `25.46 -> 0.00 ms`
- `aten::copy_`: `785.65 -> 742.30 ms`
- `aten::mul`: `1012.30 -> 1001.08 ms`

What got worse enough to offset that:

- `gdn.qkv_conv_depthwise`: `228.25 -> 236.92 ms`
- `gdn.qkv_conv_output_contiguous`: `89.32 -> 92.65 ms`
- `gdn.q_norm`: `48.81 -> 50.82 ms`
- `gdn.k_norm`: `49.14 -> 50.95 ms`
- `gdn.recurrence`: `177.23 -> 182.66 ms`

Interpretation:

- the output-side fusion itself is real
- but on the current winner path, its local gain is too small to overcome the
  small regressions it introduces or exposes elsewhere
- that makes it a local reject for now, not an H100 candidate

Decision:

Do not promote `current-winner-cuda-output-only` to H100.

Keep the preset/config around for future isolated rework if needed, but return
the active kernel path to the non-extension current winner.

## 2026-04-05 — Output-only fused preset also failed the H100 check (`h100k10out`)

Bundle:

- raw artifacts:
  - `local-scratch/profiling-out-h100k10-hgdn.7z`
- extracted review directory:
  - `local-scratch/_inspect_h100k10/`

Contract:

- launcher:
  - `python scripts/hgdn.py preflight --preset winner-20260405-11-cuda-output-only --compile-strategy hybrid`
  - `python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-11-cuda-output-only --run-prefix h100k10out`
  - `python scripts/hgdn.py h100-perf perf --preset winner-20260405-11-cuda-output-only --run-prefix h100k10out --offline`

Main finding:

The output-only fused sidecar did not flip on target hardware. It is a clean
build/parity path, but it remains an H100 perf regression.

Compared against the same-day H100 control mean (`904.46 ms`):

- compiled hybrid perf:
  - `904.46 -> 944.37 ms` (`+4.41%`)
  - `579,671 -> 555,174 tok/s` (`-4.23%`)
- eager profile step average:
  - `1670.55 -> 1644.61 ms` (`-1.55%`)

Interpretation:

- the small eager improvement is not enough to matter because compiled perf is
  the real decision surface for this branch
- do not spend more H100 time on the output-only fused preset until its
  implementation changes materially

## 2026-04-05 — Packed q/k conv + separate v conv regressed locally (`rtx4070_phase1_packedqkconv_fix2`)

Bundle:

- candidate:
  - `profiles/rtx4070_phase1_packedqkconv_fix2/`
- structured comparison:
  - `profiles/rtx4070_phase1_packedqkconv_fix2/compare_vs_rtx4070_cuda_base/comparison.md`
- baseline:
  - `profiles/rtx4070_cuda_base/`

Contract:

- launcher:
  - `conda run -s --name pg python scripts/hgdn.py local-phase1 --preset current-winner --run-prefix rtx4070_phase1_packedqkconv_fix2 --set GDN_USE_PACKED_QKV_CONV=0 --set GDN_USE_PACKED_QK_CONV=1`
- candidate flags:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=0`
  - `GDN_USE_PACKED_QK_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`

Main finding:

The packed `q/k` conv idea worked mechanically but lost at the full local
phase-1 gate, so it does **not** stay in the branch.

Compared against `rtx4070_cuda_base`:

- trainer eager self-device total:
  - `25561.13 -> 26367.32 ms` (`+3.15%`)
- trainer eager console step average:
  - `3320.37 -> 3422.05 ms` (`+3.06%`)
- trainer eager peak allocated memory:
  - `6184 -> 6184 MiB`

What did improve:

- `gdn.qk_conv_output_contiguous` replaced the old three-way materialization path:
  - baseline `gdn.qkv_conv_output_contiguous = 89.32 ms`
  - candidate `gdn.qk_conv_output_contiguous = 59.47 ms`
- `aten::copy_`: `785.65 -> 750.49 ms`

What made it lose anyway:

- the front-end stopped being one packed qkv conv and turned into:
  - packed `qk` conv
  - separate `v_conv`
  - extra packing work via `aten::cat`
- that changed the trainer front-end subtotal from:
  - baseline:
    - `gdn.qkv_conv_depthwise + gdn.qkv_conv_output_contiguous`
    - `228.25 + 89.32 = 317.57 ms`
  - candidate:
    - `gdn.qk_conv_depthwise + gdn.qk_conv_output_contiguous + gdn.conv_depthwise + aten::cat`
    - `157.64 + 59.47 + 74.58 + 87.84 = 379.53 ms`
    - net `+61.96 ms`
- secondary regressions followed:
  - `gdn.q_norm`: `48.81 -> 50.62 ms`
  - `gdn.k_norm`: `49.14 -> 50.89 ms`
  - `gdn.recurrence`: `177.23 -> 182.28 ms`
  - `aten::mul`: `1012.30 -> 1050.52 ms`

Boundary read:

- the intended boundary change did happen:
  - `conv_qkv.q/k` are non-contiguous after the packed-`qk` conv
  - `norm_qkv.q/k` are contiguous again after normalization
- that means the design goal was achieved
- it just was not worth the extra packing + separate-`v` cost

Decision:

Reject packed `q/k` conv + separate `v` conv as an active kernel path.

Keep the non-extension current winner unchanged, and avoid future front-end
experiments that split `v` back out of the packed qkv path unless they also
eliminate the added `aten::cat` / separate-depthwise cost.

## Depthwise attribution screen and `CUDNN_BENCHMARK=1` promotion

I ran a depthwise-specific attribution tranche on the non-extension current
winner instead of guessing at another front-end rewrite.

What was screened locally:

- fresh hotpath baseline under current code
- `GDN_FREEZE_CONV_WEIGHTS=1` in the profiler/preflight only
- `CUDNN_BENCHMARK=1`

Fresh hotpath baseline vs attribution screens:

- `hybrid_fwd_bwd` total self-device:
  - baseline: `397.06 ms`
  - freeze conv weights: `387.22 ms`
  - `CUDNN_BENCHMARK=1`: `369.45 ms`
- key `hybrid_fwd_bwd` buckets:
  - `gdn.conv_depthwise`: `7.56 -> 7.57 -> 6.78 ms`
  - `aten::convolution_backward`: `3.35 -> 2.18 -> 3.12 ms`
  - `aten::_conv_depthwise2d`: `2.65 -> 2.67 -> 2.65 ms`
  - `gdn.q_norm`: `3.46 -> 3.47 -> 2.81 ms`
  - `gdn.k_norm`: `2.06 -> 1.85 -> 1.34 ms`
  - `block.gdn`: `58.03 -> 57.96 -> 53.85 ms`

Interpretation:

- freezing conv weights does **not** produce a decisive collapse in the real
  hybrid path
- that means the problem is not "just conv weight grad" in isolation
- `CUDNN_BENCHMARK=1` is the only runtime-only knob that showed a credible
  hybrid hotpath win, so I promoted it to a full local phase-1 gate

Full local phase-1 result for `CUDNN_BENCHMARK=1`:

- candidate bundle:
  - `profiles/rtx4070_phase1_cudnnbench_fix1/`
- comparison:
  - `profiles/rtx4070_phase1_cudnnbench_fix1/compare_vs_rtx4070_cuda_base/comparison.md`
- trainer eager self-device total:
  - `25561.13 -> 26518.46 ms` (`+3.75%`)
- trainer eager step average:
  - `3320.37 -> 3396.07 ms` (`+2.28%`)

Why it was rejected:

- the short hotpath slice improved, especially around the packed depthwise
  front-end
- the real trainer-eager window still lost:
  - `aten::copy_`: `785.65 -> 812.82 ms`
  - `aten::mul`: `1012.30 -> 1048.52 ms`
  - `gdn.recurrence`: `177.23 -> 182.33 ms`
  - `aten::convolution_backward`: `174.81 -> 180.83 ms`
  - `aten::_conv_depthwise2d`: `141.31 -> 145.90 ms`

Decision:

Reject `CUDNN_BENCHMARK=1` as an active runtime setting for the current winner.

The depthwise tranche should now pivot away from runtime knobs and toward a
targeted implementation effort:

1. keep one packed q/k/v front-end buffer
2. replace only the packed depthwise causal conv backward path
3. specifically target the input-grad / weight-grad behavior without reopening
   split-`v` or output-fusion experiments
