# HGDN fused CUDA path

This branch adds an optional CUDA extension for the current HGDN hotspot cluster:

- packed `qkv` front-end: packed projection result -> packed depthwise causal conv -> SiLU -> split -> q/k L2 norm
- output glue: `RMSNorm(o) * SiLU(g_out)`

The intended targets are the HGDN-native buckets that remain after the packed qkv winner:

- `gdn.qkv_conv_packed`
- `gdn.q_norm`
- `gdn.k_norm`
- `gdn.output_norm`
- `gdn.output_gate_mul`

## Files

- `hgdn_cuda/csrc/binding.cpp`
- `hgdn_cuda/csrc/frontend_kernel.cu`
- `hgdn_cuda/csrc/output_kernel.cu`
- `hgdn_cuda/ops.py`
- `hgdn_cuda/reference.py`
- `setup_hgdn_cuda.py`
- `scripts/build_hgdn_cuda.sh`
- `scripts/hgdn_cuda_parity.py`

## Build

From the repo root on a CUDA machine:

```bash
conda run -s --name pg python setup_hgdn_cuda.py build_ext --inplace
```

or

```bash
conda run -s --name pg bash scripts/build_hgdn_cuda.sh
```

A slower opt-in JIT build path also exists through `hgdn_cuda/ops.py` when:

```bash
export GDN_CUDA_ALLOW_JIT_BUILD=1
```

Build the extension on the machine where it will run. The locally built `.so`
is not treated as portable across the laptop and H100 boxes.

## Runtime flags

These flags are off by default.

```bash
export GDN_USE_PACKED_QKV_CONV=1
export GDN_USE_PACKED_QKV_PROJ=1
export GDN_USE_CUDA_FUSED_FRONTEND=1
export GDN_USE_CUDA_FUSED_OUTPUT=1
```

`GDN_USE_CUDA_FUSED_FRONTEND=1` currently requires both packed qkv projection and packed qkv conv.

`GDN_USE_CUDA_FUSED_OUTPUT=1` currently requires `GDN_OUTPUT_NORM_FP32=1`.

## Validation

Use the single-process smoke test before long runs:

```bash
conda run -s --name pg python scripts/hgdn_cuda_preflight.py
```

For direct kernel-vs-reference checks on GPU after the extension is built, run:

```bash
conda run -s --name pg python scripts/hgdn_cuda_parity.py
```

The fused path currently uses eager-island wrapping around the extension calls so TorchDynamo does not try to trace the raw pybind extension directly.

## Fallback behavior

If the extension is not available, the model falls back to PyTorch reference implementations in `hgdn_cuda/reference.py`.

That keeps CPU testing and non-CUDA environments working while preserving the same model math.

## Local checkpoint

Current local status on the laptop/WSL `pg` environment:

- extension build passed with `python setup_hgdn_cuda.py build_ext --inplace`
- direct parity passed with `python scripts/hgdn_cuda_parity.py`
- constructor validation and CPU fallback parity landed in `test_model.py`
- compile-safe eager-island wrapping is required
  - the raw external pybind path caused TorchDynamo tracing warnings and a large
    compiled-preflight regression
  - wrapping the extension calls behind `torch._dynamo.disable(...)` recovered a
    compile-safe local path

First full local phase-1 checkpoint versus the current winner:

- baseline bundle:
  - `profiles/rtx4070_cuda_base/`
- fused candidate bundle:
  - `profiles/rtx4070_cuda_fused/`
- structured comparison:
  - `profiles/rtx4070_cuda_fused/compare_vs_rtx4070_cuda_base/comparison.md`

Main local read:

- trainer eager self-device total:
  - `25561.13 -> 21352.84 ms` (`-16.46%`)
- trainer eager step average from the console:
  - `3320.37 -> 2804.76 ms` (`-15.53%`)
- trainer eager peak memory:
  - `6184 -> 5696 MiB`
- replaced trainer buckets:
  - baseline front-end/output cluster:
    - `gdn.qkv_conv_depthwise`
    - `gdn.qkv_conv_output_contiguous`
    - `gdn.q_norm`
    - `gdn.k_norm`
    - `gdn.output_norm`
    - `gdn.output_gate_mul`
    - summed to `536.57 ms`
  - fused replacement buckets:
    - `gdn.qkv_frontend_fused = 158.06 ms`
    - `gdn.output_fused = 68.34 ms`
    - summed to `226.40 ms`
- trainer eager glue deltas:
  - `aten::copy_`: `785.65 -> 137.73 ms`
  - `aten::mul`: `1012.30 -> 837.35 ms`

Interpretation:

- this is a real local step-level win, not just a microbenchmark win
- the fused path is still experimental
- the next gate is H100 validation, not default promotion

## H100 gate (`h100k8`)

Completed on the H100 box with plain `python`:

```bash
python setup_hgdn_cuda.py build_ext --inplace
python scripts/hgdn_cuda_parity.py
python scripts/hgdn.py preflight --preset current-winner-cuda-fused --compile-strategy hybrid
python scripts/hgdn.py h100-profile hybrid-eager --preset current-winner-cuda-fused --run-prefix h100k8
python scripts/hgdn.py h100-perf perf --preset current-winner-cuda-fused --run-prefix h100k8 --offline
python scripts/hgdn.py h100-profile hybrid --preset current-winner-cuda-fused --run-prefix h100k8
```

### Result

Build, parity, and preflight all passed on H100.

The fused preset did **not** transfer as a performance win.

Compared against the current H100 winner `h100k5`:

- eager hybrid profile step average:
  - `1670.55 -> 2248.98 ms` (`+34.6%`)
- compiled hybrid perf:
  - `901.05 -> 1863.41 ms` (`+106.8%`)
  - `581,860 -> 281,359 tok/s` (`-51.6%`)

### Root cause from the H100 profiles

The regression is dominated by the fused packed-front-end backward path, not by
build issues and not primarily by TorchDynamo.

Largest compiled-profile regressors on `h100k8`:

- `_PackedQKVFrontendFunctionBackward = 4316.51 ms`
- `causal_dwconv_weight_backward = 3959.04 ms`
- `_PackedQKVFrontendFunction = 291.11 ms`
- `_RMSNormSiluGateFunctionBackward = 105.65 ms`

The eager profile shows the same pattern:

- `_PackedQKVFrontendFunctionBackward = 4341.02 ms`
- `causal_dwconv_weight_backward = 3983.38 ms`
- `gdn.qkv_frontend_fused = 291.47 ms`
- `gdn.output_fused = 102.59 ms`

This matters because it rules out the simple story that “compile interaction
made it slow.” The fused frontend itself is currently too expensive on H100,
especially in backward.

### Decision

Keep the extension in-tree for future kernel work, but drop
`current-winner-cuda-fused` from the active H100 kernel path.

Do **not** promote it, and do **not** use it for architecture retuning or
quality runs.

If this path is revisited later, the first targets are:

1. packed frontend backward
2. depthwise-conv weight gradient kernel
3. only then output-side fusion as an isolated follow-up
