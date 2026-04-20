# Archived Full-Block HGDN Megakernel H100 Resource Report

The active performance path is the packed HGDN stack. The archived core-kernel
notes live in [../docs/HGDN_CORE_KERNEL_PLAN.md](../docs/HGDN_CORE_KERNEL_PLAN.md).

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

## Batch interpretation

The current report uses two different batch meanings and they should not be
mixed:

- `B=32,T=2048` is the current **1xH100 compile/parity proxy**
- `B=128,T=2048` is the likely **exact 8x bridge per-rank stress shape** under
  the live bridge-style contract:
  - `TRAIN_BATCH_TOKENS = 2,097,152`
  - `TRAIN_SEQ_LEN = 2,048`
  - `WORLD_SIZE = 8`
  - `GRAD_ACCUM_STEPS = 1`
  - `local_batch_size = 2,097,152 / (8 * 1 * 2,048) = 128`

Do not extrapolate saved-state size, HBM traffic, or allocator pressure from
`B=32` alone when the eventual question is the exact 8x bridge.

## First bounded 1xH100 checkpoint

The first real target-silicon validation is already complete. I inspected the
bundle from:

- `local-scratch/h100mk_1xh100.7z`

Bundle provenance:

- commit: `e883b9b9f187f6c92719b59fb6946a6c35fd0643`
- branch: `exp/hgdn`
- mode: `all`
- target arch: `TORCH_CUDA_ARCH_LIST=9.0`
- runtime cadence: `rec_chunk_t=8`
- timing repeats: `3`

Observed target device from the bundle:

```text
GPU_NAME=NVIDIA H100 80GB HBM3
major=9
minor=0
multiProcessorCount=132
l2CacheSize=52428800
sharedMemPerMultiprocessor=233472
sharedMemPerBlockOptin=232448
regsPerMultiprocessor=65536
warpSize=32
maxThreadsPerMultiProcessor=2048
cooperativeLaunch=1
scope=target_h100_class
```

What that H100 bundle proved:

- `setup_hgdn_megakernel.py build_ext --inplace` succeeded for `sm_90`
- `scripts/audit_hgdn_megakernel_contract.py` passed
- eager-contract parity passed through `B=1,T=2048`
- isolated launch counting still showed exactly `1` forward launch and `1`
  backward launch, with `suspicious_cuda_keys=[]`
- parity-harness H100 timing at `B=1,T=2048` was:
  - forward `8.90 ms`
  - forward + backward `37.21 ms`
- bounded trainer smoke completed all `5` steps with:
  - `compile_plan:strategy:hybrid gdn_disabled:0 gdn_megakernel_left_enabled:7`
  - `step_avg:10232.29 ms`
  - peak memory `17447 MiB` allocated, `17726 MiB` reserved

What that H100 bundle did not prove:

- it did not resolve the eager-vs-FLA recurrence mismatch
- it did not make the branch ready for paid H100 timing claims
- it did not come from a fully producer-grade helper yet

The operational caveat is important. That first bundle came from `e883b9b`,
before three helper fixes landed:

1. archive-on-failure was missing, so failed remote runs were not guaranteed to
   return a `.7z` bundle
2. the helper did not yet hard-refuse
   `GDN_MEGAKERNEL_ALLOW_JIT_BUILD=1`, so a polluted shell could have changed
   the run contract
3. the bundle did not yet record compiled binary provenance such as
   `THREADS`, `REC_V_TILE`, `REC_CHUNK_T_MAX`, and
   `GEMM_ATB_BLOCK_SPLIT_M_THRESHOLD`

Current HEAD fixes those gaps:

- the H100 helper now archives once on both success and failure through an
  `EXIT` trap
- the helper now hard-refuses any nonzero
  `GDN_MEGAKERNEL_ALLOW_JIT_BUILD`
- `extension_status()` now reports `module_file` and parsed
  `build_config_json()`, and the helper writes that into both
  `metadata.txt` and `bundle_manifest.json`

Status after that first H100 bundle is therefore:

- **ready for H100 compile/parity against eager**
- **not ready for H100 timing claims**
- **not yet validated as a drop-in replacement for a historically FLA-trained winner**

After the 2026-04-17 recurrence correction, that bundle should be read as a
**pre-beta-write checkpoint**. It still matters operationally, but it no longer
validates the exact current recurrence contract.

Approximate saved forward tensors per GDN block at `B=128,T=2048`:

| Tensor group | Bytes | MiB |
| --- | ---: | ---: |
| `qkv` | `603,979,776` | `576` |
| `g_pre + beta_pre + g_log + beta` | `16,777,216` | `16` |
| `g_out + o_raw` | `402,653,184` | `384` |
| `state_ckpt` | `2,415,919,104` | `2,304` |
| **Total per GDN block** | **`3,439,329,280`** | **`3,280`** |

That is about **3.20 GiB per GDN block** at the exact-bridge per-rank stress
shape. For a 14-layer 1:1 model with 7 GDN blocks, that is roughly **22.4 GiB**
of saved GDN block state before attention/MLP activations, optimizer state,
workspace, and allocator overhead.

## Current checkpointed candidate

The current repo-backed candidate is no longer the save-heavy version.

- `THREADS = 128`
- `REC_V_TILE = 8`
- compiled `REC_CHUNK_T_MAX = 8`
- live runtime `rec_chunk_t = 8`
- `THREADS`, `GEMM_ATB_BLOCK_SPLIT_M_THRESHOLD`, `REC_V_TILE`, and
  `REC_CHUNK_T_MAX` are now explicit compile-time knobs through `HGDN_THREADS`,
  `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD`, `HGDN_REC_V_TILE`, and
  `HGDN_REC_CHUNK_T`
- runtime checkpoint cadence is selected through
  `GDN_MEGAKERNEL_REC_CHUNK_T` or `module.megakernel_rec_chunk_t`, bounded by
  the compiled `REC_CHUNK_T_MAX`
- the live default remains `128 / 2048 / 8 / 8`
- bf16 WMMA tensor-core path for full dense tiles on `sm_80+`
- forward saves only recurrence chunk-start checkpoints
- forward keeps conv preactivations only in a temporary `pre_tmp`, not as a
  saved activation
- forward no longer materializes long-lived `q_norm`, `k_norm`, `v_post`,
  `inv_q`, or `inv_k` tensors; it forms q/k norms and v post-activations
  on the fly from `pre_tmp` inside the owned recurrence path
- forward keeps `g_out` and `o_raw` but no longer saves `o_norm` or `z`
- backward replays each chunk inside the same cooperative kernel
- backward recomputes conv preactivations from `qkv` and `conv_w` before the
  SiLU derivative path
- backward also recomputes q/k/v post-conv state and q/k inverse norms from
  `qkv` and `conv_w` instead of saving `q_norm`, `k_norm`, `v_post`, `inv_q`,
  or `inv_k` from forward
- backward recomputes output RMSNorm and gated `z` from `o_raw` and `g_out`
  before the dense `W_out` gradient phases
- backward now accumulates `grad_A_log` and `grad_dt_bias` across the existing
  `BT * H` control loop instead of leaving a serialized `H=8` tail at the end
  of the cooperative kernel
- backward reconstructs each token-local in-chunk recurrence state from the
  chunk checkpoint instead of storing a full shared-memory `chunk_states` table
- backward now consumes `grad_q_norm_accum`, `grad_k_norm_accum`,
  `grad_g_log_accum`, and `grad_beta_accum` directly instead of staging
  separate global copies before the tail backward phases
- `REC_V_TILE=8` recurrence dot products use all block warps for the
  column-dot loops instead of leaving most of the CTA idle
- the recurrence column-dot helper no longer assumes the whole tile width is
  exactly `8`; wider tile experiments now reuse the same 8-column primitive in
  slices inside the same owned kernels
- `THREADS` is now an explicit compile-time knob so CTA-width tuning can be
  tested on H100 without source edits; a local `HGDN_THREADS=256` trial was
  materially worse than the `128`-thread default, so `128` remains the live
  starting point until real H100 data says otherwise
- the qkv weight-gradient splitM activation threshold is now an explicit
  compile-time knob via `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD`; a local trial at
  `1024` was worse than the live `2048` default on the repeated timing gate,
  so `2048` remains the starting point until H100 evidence says otherwise
- the local harness now supports repeated CUDA-event timing; on the current
  `THREADS=128`, `REC_V_TILE=8`, `REC_CHUNK_T=8` build, the latest reduced
  control-tail checkpoint measured about `25.03 ms` median forward+backward at
  `B=1,T=2048`, versus the prior local `30.73 ms` point on the previous live
  checkpoint. Shorter helper cases were flat to worse, so this is still a
  target-length structural hint rather than an H100 timing claim.
- launch contract is still exactly one forward kernel and one backward kernel

The immediate parity-contract hardening pass is now complete:

- megakernel mode no longer silently falls back to eager when CUDA `bf16`
  execution requested the extension but the extension is unavailable
- megakernel mode now hard-fails on CUDA activations that are not `bf16`
- selective/hybrid compile no longer preemptively force megakernel GDN blocks
  into eager islands; compile stats now record
  `gdn_megakernel_left_enabled`
- the Python binding now exposes the HGDN block as a compile-visible
  `torch.library` custom op with registered fake/autograd formulas instead of
  an eager-only `torch.autograd.Function` island
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
- a fresh-cache local trainer smoke with `TORCH_LOGS=graph_breaks,recompiles`
  and `PERF_SKIP_FINAL_EVAL=1` now completes warmup plus the real training step
  with `gdn_disabled:0` and no HGDN-path graph-break log lines

That means the honest status of this checkpoint is:

- **ready for H100 compile/parity**
- **not ready for H100 timing**
- **not yet validated as a drop-in replacement for a historically FLA-trained winner**

One extra local finding matters for interpretation:

- the branch no longer treats no-beta-write as the live HGDN contract
- on 2026-04-17, eager fallback and the megakernel were both corrected to the
  canonical beta-write recurrence, matching the Gated DeltaNet / FLA-style
  contract rather than the older eager fallback variant
- the dedicated recurrence-boundary diagnostic now shows eager and FLA aligned
  within bf16-level drift rather than a semantic mismatch:
  - `allow_neg_eigval=False`: first failing `T=None`, worst sampled `T=55`, max
    abs `0.000866883`, norm-rel `0.0035972`
  - `allow_neg_eigval=True`: first failing `T=None`, worst sampled `T=4`, max
    abs `0.00111803`, norm-rel `0.00443445`
- the closed-form `T=1` diagnostic now matches the intended ordering:
  - eager matches `canonical_beta_write` essentially exactly
  - local FLA also matches `canonical_beta_write` within small bf16 drift
  - `diagnostic_no_beta_write` is now only the intentional non-canonical
    comparison
- regenerated megakernel parity cases now record
  `recurrence_contract='beta_write'` so future recurrence math changes force
  case regeneration instead of reusing stale saved references
- the first cooperative split-K weight-gradient trial for `grad_w_out`,
  `grad_w_qkv`, and `grad_w_g` stayed parity-clean and kept the one-launch
  contract, but it was slightly slower on the local 4070 helper and was
  therefore reverted instead of kept live
- the next backward-structure experiment keeps the same one-launch contract but
  is narrower:
  - only the dominant `grad_w_qkv` dense weight-gradient phase switches to a
    CTA-local split over the token-reduction dimension
  - that split is gated to long reductions only (`BT >= 2048`)
  - the warp partials are reduced in shared memory inside the same cooperative
    backward launch
  - earlier point-in-time local result on the 4070 helper:
    `B=1,T=2048` parity-harness forward+backward improved from about
    `31.28 ms` to about `26.83 ms`
- a later bounded local sweep tested wider recurrence value tiles on the same
  source after fixing the hidden 8-column assumption in the helper:
  - `REC_V_TILE=8`: `32.61 ms` forward+backward at `B=1,T=2048`
  - `REC_V_TILE=16`: `43.59 ms`
  - `REC_V_TILE=24`: `53.78 ms`
  - `REC_V_TILE=48`: `76.64 ms`
  - local conclusion: keep `REC_V_TILE=8` as the live default unless H100 data
    proves a different value wins there
- the current pre-drop checkpoint keeps that `REC_V_TILE=8` default, cuts an
  additional `144 MiB` of saved forward state per GDN block at
  `B=32,T=2048`, and moved the local `B=1,T=2048` parity-harness point to
  `30.89 ms`
- the current reduced-control-tail checkpoint keeps the same memory contract but
  reduces `grad_A_log` / `grad_dt_bias` global atomic traffic through per-head
  block reductions, moving the local `B=1,T=2048` parity-harness point again to
  `25.03 ms`

The main activation-state change is:

| Tensor strategy | Shape | Dtype | Bytes |
| --- | --- | --- | ---: |
| old `state_prev` | `(32, 2048, 8, 48, 48)` | fp32 | 4,831,838,208 |
| new `state_ckpt` | `(32, 256, 8, 48, 48)` | fp32 | 603,979,776 |

That cuts the recurrence-state save from **4.50 GiB** to **576 MiB** per GDN
block.

Total saved forward state in the current checkpointed version is about
**0.80 GiB per GDN block** at `B=32,T=2048` and about **3.20 GiB per GDN
block** at `B=128,T=2048`, instead of about **5.12 GiB** in the original
save-heavy layout.

A bounded high-memory speed candidate is now also identified:

- the current portable build compiles `REC_CHUNK_T_MAX=8` and can now select
  `rec_chunk_t=4` on the same binary while keeping the one-launch
  forward/backward structure
- `rec_chunk_t=4` still doubles checkpoint count versus the live `8`
- that pushes saved forward state to about **1.36 GiB per GDN block** at
  `B=32,T=2048` and about **5.45 GiB per GDN block** at `B=128,T=2048`
- same-build local helper timing with the current runtime-selectable path:
  - default `rec_chunk_t=8`: `B=1,T=128` about `2.45 ms`,
    `B=1,T=512` about `9.15 ms`, `B=2,T=512` about `9.11 ms`,
    `B=1,T=2048` about `30.73 ms`
  - runtime `rec_chunk_t=4`: `2.20 ms`, `8.16 ms`, `8.94 ms`, `26.19 ms`
- interpretation: promising speed-vs-memory trade, worth a bounded H100
  compile/parity check if memory headroom allows it, but not safe to flip the
  live default from local `sm_89` data alone

That makes the current rerun gate straightforward:

- rerun the bounded `1xH100` helper from current HEAD, not from `e883b9b`
- this rerun is now required not only for helper hardening, but also because
  the live recurrence contract has changed from the pre-beta-write H100 bundle
- helper command:
  `scripts/run_h100_single_gpu_hgdn_megakernel.sh all`
- helper Python runtime selection is now portable:
  - `PYTHON_BIN` if explicitly set
  - else plain `python3` / `python`
- structured launcher:
  `python scripts/hgdn.py h100-megakernel all --offline`
- helper bundle behavior now includes:
  - `MK_OUTPUT_DIR=/path/to/output_dir`
  - optional explicit archive path:
    `MK_ARCHIVE_OUTPUT=/path/to/output_dir.7z`
  - `build.log`, `audit.log`, `parity.log`, `trainer_smoke.log` when present,
    plus `commands.sh`, `metadata.txt`, and `bundle_manifest.json`
  - archive-on-failure as well as archive-on-success
  - `bundle_exit_status`
  - `extension_status_json`
  - compiled binary provenance via `build_config_json()`
- after the corrected default rerun, do at most one narrow `1xH100` timing
  sanity comparing `rec_chunk_t=8` against the existing `rec_chunk_t=4`
  candidate
- do not upgrade the label to “ready for H100 timing” until the corrected rerun
  passes and the next structural bottleneck branch lands

## Save-vs-recompute decisions

| Saved tensor | Shape | Dtype | Bytes | If not saved | Decision |
| --- | --- | --- | ---: | --- | --- |
| `qkv` | `(32, 2048, 1152)` | bf16 | 150,994,944 | rerun packed `W_qkv` dense phase, about `57.98 GF` | keep saved |
| `g_pre` | `(32, 2048, 8)` | bf16 | 1,048,576 | rerun `w_a` projection, about `0.40 GF` | keep for now |
| `beta_pre` | `(32, 2048, 8)` | bf16 | 1,048,576 | rerun `w_b` projection, about `0.40 GF` | keep for now |
| `g_log` | `(32, 2048, 8)` | bf16 | 1,048,576 | recompute from `g_pre`, `A_log`, and `dt_bias` | keep for now |
| `beta` | `(32, 2048, 8)` | bf16 | 1,048,576 | recompute from `beta_pre` | keep for now |
| `g_out` | `(32, 2048, 384)` | bf16 | 50,331,648 | rerun dense `w_g`, dominates the control-path dense work | keep saved |
| `o_raw` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 | rerun recurrence readout, part of about `12.08 GF` recurrence work | keep saved |
| `state_ckpt` | `(32, 256, 8, 48, 48)` | fp32 | 603,979,776 | replay recurrence inside backward, adds about one extra forward-style recurrence pass, about `12.08 GF` | keep saved |

The implemented save-vs-recompute changes so far are:

- replacement of full per-token `state_prev` with chunk checkpoints plus
  within-chunk replay
- removal of saved `pre`, recomputed inside backward from `qkv` and `conv_w`
- removal of saved `q_norm`, `k_norm`, `v_post`, `inv_q`, and `inv_k`,
  recomputed inside backward from `qkv` and `conv_w`
- removal of saved `o_norm` and `z`, recomputed inside backward from `o_raw`
  and `g_out`
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
5. The new qkv-only CTA-local split is a better direction than the rejected
   global-partial split-K layout, but it has only moved the long-sequence local
   point; it has not closed the overall gap to the packed HGDN path.

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
