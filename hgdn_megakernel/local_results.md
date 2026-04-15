# HGDN Megakernel Local Results

## Environment

- Date: 2026-04-15
- Host: local WSL laptop helper
- Python env: `conda run -s --name pg`
- Build command:
  `python setup_hgdn_megakernel.py build_ext --inplace`
- Validation command:
  `python hgdn_megakernel/test_megakernel.py`

## Branch direction

- Branch rule as of 2026-04-15: old HGDN sidecar kernels are retired for new
  kernel work on this branch.
- New kernel implementation work should stay on the fused or fused-like
  end-to-end megakernel path.
- The older `hgdn_cuda` sidecar family remains in-tree only for historical
  reference and old result inspection.
- Current candidate here is the chunk-checkpointed megakernel:
  - `THREADS = 128`
  - `REC_V_TILE = 8`
  - `REC_CHUNK_T = 8`
  - bf16 WMMA tensor-core path for full dense tiles on `sm_80+`
  - warp-local scalar fallback for dense edge tiles
  - forward saves only recurrence chunk-start checkpoints
  - backward replays each chunk inside the same cooperative kernel
  - warp-shuffle-backed CTA reductions replace the old full-block tree
    reductions in q/k norm, output RMSNorm, and backward scalar reductions
  - winner-shape `Dv=8` recurrence dot products now use all block warps instead
    of only the first `8` threads for the per-column dot loops
  - backward no longer keeps per-token `chunk_states` or `dv0_hist` in shared
    memory
  - backward reconstructs each token's in-chunk recurrence state from the chunk
    checkpoint during the reverse sweep
  - forward launch count remains `1`
  - backward launch count remains `1`

## Device report

```
GPU_NAME=NVIDIA GeForce RTX 4070 Laptop GPU
major=8
minor=9
multiProcessorCount=36
l2CacheSize=33554432
sharedMemPerMultiprocessor=102400
sharedMemPerBlockOptin=101376
regsPerMultiprocessor=65536
warpSize=32
maxThreadsPerMultiProcessor=1536
cooperativeLaunch=1
scope=correctness_only_non_h100
```

This device is `sm_89`, so these results are for compile/parity/sanity only.
They are not evidence about H100 timing quality.

## Parity

Reference cases were generated from the live packed `GatedDeltaNet` module with:

- `D=384`
- `H=8`
- `Dk=48`
- `Dv=48`
- `K=4`
- `expand_v=1.0`
- `use_packed_qkv_proj=True`
- `use_packed_qkv_conv=True`
- `gates_fp32=True`
- `output_norm_fp32=True`
- `gdn_control_proj_fp32=False`
- eager recurrence reference (`use_fla=False`)

### `B=1, T=8`

- forward `y`: pass
  max abs `0.00585938`
  max rel `39.9146`
- `grad_x`: pass
  max abs `0.000671387`
- `grad_w_qkv`: pass
  max abs `0.0117188`
- `grad_w_a`: pass
  max abs `0.000366211`
- `grad_w_b`: pass
  max abs `9.15527e-05`
- `grad_w_g`: pass
  max abs `0.0012207`
- `grad_w_out`: pass
  max abs `0.000518799`
- `grad_conv_w`: pass
  max abs `0.00488281`
- `grad_A_log`: pass
  max abs `0.000276035`
- `grad_dt_bias`: pass
  max abs `0.000178222`

### `B=1, T=32`

- forward `y`: pass
  max abs `0.0078125`
  max rel `166.647`
- `grad_x`: pass
  max abs `0.000747681`
- `grad_w_qkv`: pass
  max abs `0.0158691`
- `grad_w_a`: pass
  max abs `0.000976562`
- `grad_w_b`: pass
  max abs `0.000244141`
- `grad_w_g`: pass
  max abs `0.00244141`
- `grad_w_out`: pass
  max abs `0.000915527`
- `grad_conv_w`: pass
  max abs `0.00878906`
- `grad_A_log`: pass
  max abs `0.000457941`
- `grad_dt_bias`: pass
  max abs `0.000311824`

The parity tolerances used by the harness were:

- forward: `atol=3e-2`, `rtol=3e-2`
- backward: `atol=1.2e-1`, `rtol=1.2e-1`

These are still diagnostic bf16 tolerances, not the final tightened threshold set.

## Launch count

The measured megakernel path was isolated with a preallocated `grad_out` and direct
`torch.autograd.backward((out,), (grad_out,))`.

- forward launch count: `1`
- backward launch count: `1`

Observed CUDA-side entries for the HGDN path:

- `hgdn_forward_bf16_kernel`
- `hgdn_backward_bf16_kernel`
- host/autograd bookkeeping around the custom function

No extra CUDA helper kernels from the HGDN block path itself appeared in the
isolated launch-count region.

## CUDA-event timing

These timings are for the local `sm_89` device only.

### `B=1, T=8`

- forward: `0.79258 ms`
- forward + backward: `1.20525 ms`

### `B=1, T=32`

- forward: `0.21402 ms`
- forward + backward: `0.69325 ms`

## 4070 speed comparison vs current packed HGDN CUDA path

I also benchmarked the new megakernel against the current packed `GatedDeltaNet`
CUDA block path on the same local RTX 4070 laptop GPU.

Comparison setup:

- same weights copied into both modules
- same live packed HGDN block contract
- direct block timing, not end-to-end trainer timing
- backward isolated with a preallocated `grad_out`
- baseline path had `use_fla=True` active on this machine

### `B=1, T=128`

- current path:
  - forward `0.86925 ms`
  - forward + backward `2.83228 ms`
- megakernel:
  - forward `0.41193 ms`
  - forward + backward `2.06364 ms`
- relative:
  - forward speedup `2.110x`
  - forward + backward speedup `1.372x`

### `B=1, T=512`

- current path:
  - forward `0.87076 ms`
  - forward + backward `3.86432 ms`
- megakernel:
  - forward `1.23343 ms`
  - forward + backward `5.29029 ms`
- relative:
  - forward speedup `0.706x`
  - forward + backward speedup `0.730x`

### `B=1, T=1024`

- current path:
  - forward `0.85423 ms`
  - forward + backward `3.04876 ms`
- megakernel:
  - forward `1.72012 ms`
  - forward + backward `10.56481 ms`
- relative:
  - forward speedup `0.497x`
  - forward + backward speedup `0.289x`

### `B=1, T=2048`

- current path:
  - forward `0.92774 ms`
  - forward + backward `3.82874 ms`
- megakernel:
  - forward `3.42753 ms`
  - forward + backward `21.49407 ms`
- relative:
  - forward speedup `0.271x`
  - forward + backward speedup `0.178x`

### `B=2, T=512`

- current path:
  - forward `0.85545 ms`
  - forward + backward `2.92895 ms`
- megakernel:
  - forward `1.02694 ms`
  - forward + backward `6.01738 ms`
- relative:
  - forward speedup `0.833x`
  - forward + backward speedup `0.487x`

Local conclusion:

- parity is good
- launch structure is good
- the backward shared-memory shrink is a real local improvement
- removing shared `chunk_states` and `dv0_hist` cut the candidate's dynamic
  backward recurrence tile from about `21.25 KiB` to about `9.00 KiB`
- that shared-memory drop also materially improved the forward timing even
  though the forward math did not change
- relative to the previous clean checkpoint, the current candidate improved by:
  - about `2.04x` at `B=1,T=512` for forward + backward
  - about `1.87x` at `B=1,T=1024` for forward + backward
  - about `1.61x` at `B=1,T=2048` for forward + backward
  - about `1.93x` at `B=2,T=512` for forward + backward
- the current megakernel is still a clear local win at `B=1,T=128`
- the current megakernel is still behind the packed HGDN path for `T=512+`,
  but the gap is now materially smaller than the previous checkpoint
- larger local batch points are still not competitive:
  - `B=4,T=512`: baseline `3.39333 ms`, megakernel `14.28992 ms`
  - `B=8,T=512`: baseline `3.97802 ms`, megakernel `19.89581 ms`
- the next meaningful speed step is still backward-focused:
  fewer global accumulations, better dense phases, or a more aggressive
  Hopper-specific implementation

## Readiness call

Current status: **ready for H100 compile/parity testing**

Rationale:

- builds cleanly on the local GPU
- forward parity passes for `B=1,T=8` and `B=1,T=32`
- backward parity passes for the requested gradients on both shapes
- isolated launch counting shows exactly one HGDN forward kernel launch and one HGDN backward kernel launch
- device properties are printed explicitly

Current status is **not** “ready for H100 timing testing” because:

- local `sm_89` timing is not a proxy for H100
- the current tensor-core path is WMMA-based and portable, not yet Hopper-tuned
- the recurrent core still loses to the live packed path at `T=512+` on the
  local 4070
- backward recurrence/replay and its global accumulation traffic are still too
  expensive
