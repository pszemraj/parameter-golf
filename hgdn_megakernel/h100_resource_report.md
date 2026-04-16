# HGDN Megakernel H100 Resource Report

## Candidate shape

This report targets the live winner-style HGDN contract:

- `B = 32`
- `T = 2048`
- `D = 384`
- `H = 8`
- `Dk = 48`
- `Dv = 48`
- `K = 4`
- packed qkv projection
- packed depthwise causal qkv conv
- dense cross-head `W_out`

## Current checkpointed candidate

The current repo-backed candidate is no longer the save-heavy version.

- `THREADS = 128`
- `REC_V_TILE = 8`
- `REC_CHUNK_T = 8`
- bf16 WMMA tensor-core path for full dense tiles on `sm_80+`
- forward saves only recurrence chunk-start checkpoints
- backward replays each chunk inside the same cooperative kernel
- backward reconstructs each token-local in-chunk recurrence state from the
  chunk checkpoint instead of storing a full shared-memory `chunk_states` table
- backward now consumes `grad_q_norm_accum`, `grad_k_norm_accum`,
  `grad_g_log_accum`, and `grad_beta_accum` directly instead of staging
  separate global copies before the tail backward phases
- `REC_V_TILE=8` recurrence dot products use all block warps for the
  column-dot loops instead of leaving most of the CTA idle
- launch contract is still exactly one forward kernel and one backward kernel

The immediate parity-contract hardening pass is now complete:

- megakernel mode no longer silently falls back to eager when CUDA `bf16`
  execution requested the extension but the extension is unavailable
- megakernel mode now hard-fails on CUDA activations that are not `bf16`
- selective/hybrid compile no longer preemptively force megakernel GDN blocks
  into eager islands; compile stats now record
  `gdn_megakernel_left_enabled`
- trainer preflight now prints extension status and rejects
  `GDN_CONTROL_PROJ_FP32=1` before training starts in megakernel mode
- the autograd wrapper now rejects non-contiguous or wrongly typed inputs
  instead of inserting hidden `.contiguous()` or `.to()` kernels
- correctness builds now omit `--use_fast_math`
- local eager parity now covers `B=1,T=8/32/128/512` and optional `B=2,T=512`
- the harness now exposes an optional `B=1,T=2048` parity gate for the real
  submission sequence length, and that case also passes locally on the 4070
  helper
- isolated launch counting still shows exactly one forward launch and one
  backward launch, with no extra CUDA-side copy or memset events

That means the honest status of this checkpoint is:

- **ready for H100 compile/parity**
- **not ready for H100 timing**

One extra local finding matters for interpretation:

- the installed `use_fla=True` control path diverges materially from the eager
  HGDN reference on the same saved weights and inputs, so eager remains the
  megakernel numerical contract until that separate control-path mismatch is
  resolved
- a dedicated recurrence-boundary diagnostic confirms that this FLA drift is
  immediate rather than only a late chunk-boundary artifact:
  - `allow_neg_eigval=False`: first failing `T=1`, worst sampled `T=28`, max
    abs `0.152049`, norm-rel `0.524228`
  - `allow_neg_eigval=True`: first failing `T=3`, worst sampled `T=4`, max abs
    `0.0744246`, norm-rel `0.327112`
- the first cooperative split-K weight-gradient trial for `grad_w_out`,
  `grad_w_qkv`, and `grad_w_g` stayed parity-clean and kept the one-launch
  contract, but it was slightly slower on the local 4070 helper and was
  therefore reverted instead of kept live

The main activation-state change is:

| Tensor strategy | Shape | Dtype | Bytes |
| --- | --- | --- | ---: |
| old `state_prev` | `(32, 2048, 8, 48, 48)` | fp32 | 4,831,838,208 |
| new `state_ckpt` | `(32, 256, 8, 48, 48)` | fp32 | 603,979,776 |

That cuts the recurrence-state save from **4.50 GiB** to **576 MiB** per GDN
block.

Total saved forward state in the current checkpointed version is about
**1.18 GiB per GDN block** instead of about **5.12 GiB**.

## Save-vs-recompute decisions

| Saved tensor | Shape | Dtype | Bytes | If not saved | Decision |
| --- | --- | --- | ---: | --- | --- |
| `qkv` | `(32, 2048, 1152)` | bf16 | 150,994,944 | rerun packed `W_qkv` dense phase, about `57.98 GF` | keep saved |
| `pre` | `(32, 2048, 1152)` | bf16 | 150,994,944 | rerun packed causal conv and pre-SiLU staging, about `0.60 GF` plus extra conv bookkeeping | keep saved |
| `q_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 | recompute q conv, SiLU, and L2 norm path | keep saved |
| `k_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 | recompute k conv, SiLU, and L2 norm path | keep saved |
| `v_post` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 | recompute v conv and SiLU path | keep saved |
| `inv_q` | `(32, 2048, 8)` | fp32 | 2,097,152 | cheap recompute from q preactivations | keep for parity simplicity |
| `inv_k` | `(32, 2048, 8)` | fp32 | 2,097,152 | cheap recompute from k preactivations | keep for parity simplicity |
| `g_pre` | `(32, 2048, 8)` | bf16 | 1,048,576 | rerun `w_a` projection, about `0.40 GF` | keep for now |
| `beta_pre` | `(32, 2048, 8)` | bf16 | 1,048,576 | rerun `w_b` projection, about `0.40 GF` | keep for now |
| `g_log` | `(32, 2048, 8)` | bf16 | 1,048,576 | recompute from `g_pre`, `A_log`, and `dt_bias` | keep for now |
| `beta` | `(32, 2048, 8)` | bf16 | 1,048,576 | recompute from `beta_pre` | keep for now |
| `g_out` | `(32, 2048, 384)` | bf16 | 50,331,648 | rerun dense `w_g`, dominates the control-path dense work | keep saved |
| `o_raw` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 | rerun recurrence readout, part of about `12.08 GF` recurrence work | keep saved |
| `o_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 | recompute RMSNorm from `o_raw` | future drop candidate |
| `z` | `(32, 2048, 384)` | bf16 | 50,331,648 | recompute from `o_norm` and `g_out` | future drop candidate |
| `state_ckpt` | `(32, 256, 8, 48, 48)` | fp32 | 603,979,776 | replay recurrence inside backward, adds about one extra forward-style recurrence pass, about `12.08 GF` | keep saved |

The implemented save-vs-recompute changes so far are:

- replacement of full per-token `state_prev` with chunk checkpoints plus
  within-chunk replay
- removal of backward shared-memory `chunk_states` and `dv0_hist`, replaced by
  token-local replay from the chunk checkpoint during the reverse sweep

The latest backward-scratch cleanup also removed these global staging tensors
from the backward workspace:

| Removed staging tensor | Shape | Dtype | Bytes |
| --- | --- | --- | ---: |
| `grad_q_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `grad_k_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `grad_g_log` | `(32, 2048, 8)` | fp32 | 2,097,152 |
| `grad_beta` | `(32, 2048, 8)` | fp32 | 2,097,152 |
| total removed |  |  | 104,857,600 |

The only implemented dense-phase speed change so far is the replacement of the
old scalar `16 x 16` shared-memory dot-product tiles with a portable bf16 WMMA
path on `sm_80+`, while keeping scalar edge fallback for non-full tiles.

The latest reduction-side cleanup is:

- warp-shuffle-backed CTA reductions for q/k norm, output RMSNorm, and the
  backward scalar gate accumulations
- one dead CTA reduction buffer removed from the backward shared-memory layout
- `REC_V_TILE=8` recurrence column-dot helpers now use warp-partitioned block
  cooperation for `tmp_dv0`, `o_raw`, `tmp_dv1`, and `grad_v_post`
- later backward phases no longer write and reread temporary global copies of
  q/k norm grads or gate-log grads before finishing `grad_pre`,
  `grad_g_pre`, `grad_beta_pre`, `grad_A_log`, and `grad_dt_bias`

## Shared-memory map

The current cooperative kernel uses:

- warp-level bf16 WMMA tiles for dense projections:
  - one `16 x 16 x 16` tensor-core tile per warp for full tiles
  - scalar edge fallback for partial tiles
- one CTA-local recurrence tile per `(batch, head, value_tile)` stream
  - `REC_V_TILE = 8`
  - `REC_CHUNK_T = 8`
  - `THREADS = 128`

- `S0`: `Dk * REC_V_TILE = 384` fp32 values
- `S1`: `384` fp32 values
- `adj`: `384` fp32 values
- `q`: `48` fp32 values
- `k`: `48` fp32 values
- `v`: `8` fp32 values
- `go`: `8` fp32 values
- `tmp_dv0`: `8` fp32 values
- `tmp_dv1`: `8` fp32 values
- `tmp_dk0`: `48` fp32 values
- one `THREADS=128` scratch buffer for CTA-local reductions
- replay caches for one chunk:
  - `q_hist`: `8 * 48 = 384` fp32 values
  - `k_hist`: `384` fp32 values
  - `v_hist`: `8 * 8 = 64` fp32 values
  - `alpha_hist`: `8` fp32 values
  - `beta_hist`: `8` fp32 values

Recurrence-tile shared memory for `Dk=48`, `REC_V_TILE=8`, `REC_CHUNK_T=8` is:

- `3 * 384 + 3 * 48 + 4 * 8 + 128 + 8 * (2 * 48 + 8 + 2)`
  fp32 values
- `2304` fp32 values
- `2304 * 4 = 9,216` bytes
- about **9.00 KiB per CTA**

Dense-phase WMMA scratch is smaller:

- `WARPS_PER_BLOCK * 16 * 16 = 4 * 256 = 1024` fp32 values
- `1024 * 4 = 4,096` bytes

So the recurrence tile still sets the dynamic shared-memory request.

## Occupancy estimate

The repo-backed launch wrapper uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor` instead of a hard-coded block count.

With `THREADS=128` and about `9.00 KiB` dynamic shared memory per CTA:

- shared memory is no longer the first obvious occupancy limiter
- on a `100 KiB`-class SM, the shared-memory ceiling rises from about `4` CTAs
  per SM to about `11` CTAs per SM
- on H100 SXM, shared memory should not be the gating factor at all for this
  checkpoint
- the real limiter is now more likely register pressure from:
  - the explicit backward math
  - the replayed in-chunk prefix reconstruction
  - the WMMA accumulator fragments in the dense phases
- this is a cleaner occupancy trade than the previous save-heavy shared-memory
  layout, even though it adds extra recurrence arithmetic

Recurrence parallelism is now structurally capped at `B * H * ceil(Dv / REC_V_TILE)` CTAs.
At the candidate shape that is:

- `32 * 8 * 6 = 1,536` recurrence streams

On a 132-SM H100 SXM this means the recurrence portion can only expose about:

- `11.64` streams per SM at best

That is a real improvement over the first draft, but it still does not guarantee a final throughput winner because each stream is still a lightweight sequential recurrence tile rather than a tensor-core-friendly bulk kernel.

## FLOP and traffic sanity

Approximate forward work at the candidate shape:

- packed dense `W_qkv`: about `57.98 GF`
- `w_a + w_b + w_g`: about `20.13 GF`
- packed depthwise causal conv: about `0.60 GF`
- gated-delta recurrence update + readout: about `12.08 GF`
- dense `W_out`: about `19.33 GF`

Total forward work is about **110 GF per block**.

Backward is analytic and explicit in the same kernel and should land in the
rough range of **2.5x to 3.5x forward work** even after checkpointing, because
the saved-state reduction is partly offset by the added replay pass.

HBM traffic is dominated by:

- dense reads of `x`, `w_qkv`, `w_g`, and `w_out`
- saved activation traffic other than recurrence state
- checkpoint reads plus replay staging instead of full `state_prev` rereads
- global-atomic traffic for `grad_q_norm_accum` and `grad_k_norm_accum`

The latest checkpoint does trim about `100 MiB` of backward workspace and one
full-grid scratch staging pass, but the current version is still expected to be:

- materially better than the first scalar-loop draft
- materially less memory-heavy than the save-heavy parity version
- materially better in dense-phase throughput than the old scalar dense helpers
- still memory-traffic heavy
- still not Hopper-tuned tensor-core efficient

## Roofline-style conclusion

This candidate is good enough for:

- local compile/parity
- local launch-count proof
- H100 compile/parity testing

This candidate is **not** a credible final H100 throughput kernel yet.

The blunt bottlenecks are:

1. Dense `W_qkv`, `W_g`, `W_out`, and the major backward matmuls are now portable WMMA tensor-core kernels for full tiles, but they are not yet Hopper-specific kernels.
2. The recurrence-state save is much smaller now, and the backward kernel no
   longer carries the old shared-memory chunk-state table, but backward still
   pays for replay arithmetic inside the kernel.
3. The recurrence is value-tiled, but the core algorithm is still sequential within each tile and remains the scaling limiter at longer sequence lengths.
4. Backward global accumulation for `grad_q_norm_accum` and `grad_k_norm_accum` is still an obvious cost center.

If H100 parity is clean, the next speed branch should be one of:

- split-K or other token-dimension parallelization for the dense backward
  weight-gradient phases inside the same cooperative backward kernel
- Hopper-specific dense phases (`wgmma` / better scheduling) inside the megakernel
- or a justified split where cuBLAS/cuBLASLt owns dense GEMMs and one fused HGDN kernel owns the recurrent shell
- or a recurrence-focused rewrite that removes or sharply reduces the backward global-atomic path

The first branch is the purist megakernel route.
The second branch may be the faster route to a competition-worthy H100 candidate.

The first split-K attempt was enough to rule out one naive design:

- writing one set of fp32 partial tiles and reducing them later inside the same
  cooperative kernel did not beat the hardened baseline on the local 4070
- any next dense-phase rewrite needs a better ownership/scheduling story than
  the simple partial-buffer reduction path
