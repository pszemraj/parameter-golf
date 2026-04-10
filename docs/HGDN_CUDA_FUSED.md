# HGDN CUDA Extension

Last updated: 2026-04-10

## What is in the extension

- packed `qkv` front-end:
  - packed projection result
  - packed depthwise causal conv
  - SiLU
  - split
  - q/k L2 norm
- output glue:
  - `RMSNorm(o) * SiLU(g_out)`

## Files

- [`../hgdn_cuda/csrc/binding.cpp`](../hgdn_cuda/csrc/binding.cpp)
- [`../hgdn_cuda/csrc/frontend_kernel.cu`](../hgdn_cuda/csrc/frontend_kernel.cu)
- [`../hgdn_cuda/csrc/output_kernel.cu`](../hgdn_cuda/csrc/output_kernel.cu)
- [`../hgdn_cuda/ops.py`](../hgdn_cuda/ops.py)
- [`../hgdn_cuda/reference.py`](../hgdn_cuda/reference.py)
- [`../setup_hgdn_cuda.py`](../setup_hgdn_cuda.py)
- [`../scripts/build_hgdn_cuda.sh`](../scripts/build_hgdn_cuda.sh)
- [`../scripts/hgdn_cuda_parity.py`](../scripts/hgdn_cuda_parity.py)

## Build

Build on the machine where the extension will run.

```bash
conda run -s --name pg python setup_hgdn_cuda.py build_ext --inplace
```

or

```bash
conda run -s --name pg bash scripts/build_hgdn_cuda.sh
```

Opt-in JIT build remains available through `hgdn_cuda/ops.py` when:

```bash
export GDN_CUDA_ALLOW_JIT_BUILD=1
```

## Runtime flags

These flags are off by default:

```bash
export GDN_USE_PACKED_QKV_CONV=1
export GDN_USE_PACKED_QKV_PROJ=1
export GDN_USE_CUDA_FUSED_FRONTEND=1
export GDN_USE_CUDA_FUSED_OUTPUT=1
```

`GDN_USE_CUDA_FUSED_FRONTEND=1` requires packed qkv projection and packed qkv conv.

`GDN_USE_CUDA_FUSED_OUTPUT=1` requires `GDN_OUTPUT_NORM_FP32=1`.

## Validation

Run the single-process smoke test before long jobs:

```bash
conda run -s --name pg python scripts/hgdn_cuda_preflight.py
```

Run direct kernel-vs-reference checks after the extension is built:

```bash
conda run -s --name pg python scripts/hgdn_cuda_parity.py
```

## Fallback behavior

If the extension is unavailable, the model falls back to the PyTorch reference path in [`../hgdn_cuda/reference.py`](../hgdn_cuda/reference.py).

## Current status

- The extension builds and parity checks pass on supported CUDA machines.
- The extension is not part of the active H100 winner path.
- The full fused front-end variants lost on H100 (`h100k18`, `h100k19`).
- The compile-visible split / norm follow-up also lost on H100 (`h100k20`).
- Output-only fusion also failed to justify promotion.
- Keep the extension in-tree only for narrower future kernel work.
- Do not use extension presets for architecture ranking or quality sweeps unless the kernel family itself is the question.

Exact result history and scoreboards live in [PROFILING_LOG.md](PROFILING_LOG.md).
