# HGDN Megakernel Local Results

## Environment

- Date: 2026-04-16
- Host: local WSL laptop helper
- Python env: `conda run -s --name pg`
- Build command:
  `python setup_hgdn_megakernel.py build_ext --inplace`
- Optional recurrence-tile sweep build:
  `HGDN_THREADS=256 python setup_hgdn_megakernel.py build_ext --inplace`
  `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD=1024 python setup_hgdn_megakernel.py build_ext --inplace`
  `HGDN_REC_V_TILE=16 python setup_hgdn_megakernel.py build_ext --inplace`
- Validation command:
  `python hgdn_megakernel/test_megakernel.py`
- Optional median timing pass:
  `python hgdn_megakernel/test_megakernel.py --timing-repeats 5`
- Optional long-sequence gate:
  `python hgdn_megakernel/test_megakernel.py --include-b1-t2048`
- Correctness build note: the current parity binary removes `--use_fast_math`

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
  - `THREADS`, `GEMM_ATB_BLOCK_SPLIT_M_THRESHOLD`, `REC_V_TILE`, and
    `REC_CHUNK_T` are now explicit compile-time knobs via `HGDN_THREADS`,
    `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD`, `HGDN_REC_V_TILE`, and
    `HGDN_REC_CHUNK_T`, but the live defaults remain `128 / 2048 / 8 / 8`
  - bf16 WMMA tensor-core path for full dense tiles on `sm_80+`
  - warp-local scalar fallback for dense edge tiles
  - forward saves only recurrence chunk-start checkpoints
  - forward keeps conv preactivations only in a temporary `pre_tmp`, not as a
    saved activation
  - forward no longer materializes long-lived `q_norm`, `k_norm`, `v_post`,
    `inv_q`, or `inv_k` tensors; it uses `pre_tmp` plus on-the-fly q/k norm
    formation inside the owned forward recurrence path
  - forward keeps `g_out` and `o_raw` but no longer saves `o_norm` or `z`
  - backward replays each chunk inside the same cooperative kernel
  - backward recomputes conv preactivations from `qkv` and `conv_w` before the
    SiLU derivative path
  - backward also recomputes q/k/v post-conv state and q/k inverse norms from
    `qkv` and `conv_w` instead of saving `q_norm`, `k_norm`, `v_post`, `inv_q`,
    or `inv_k` from forward
  - backward recomputes output RMSNorm and gated `z` from `o_raw` and `g_out`
    before the dense `W_out` gradient phases
  - warp-shuffle-backed CTA reductions replace the old full-block tree
    reductions in q/k norm, output RMSNorm, and backward scalar reductions
  - `REC_V_TILE=8` recurrence dot products now use all block warps instead of
    only the first `8` threads for the per-column dot loops
  - the recurrence column-dot helper now supports wider `REC_V_TILE` sweeps by
    slicing tiles through the same 8-column reduction primitive instead of
    assuming the whole tile width is exactly `8`
- backward no longer keeps per-token `chunk_states` or `dv0_hist` in shared
  memory
- backward reconstructs each token's in-chunk recurrence state from the chunk
  checkpoint during the reverse sweep
- backward no longer stages `grad_q_norm`, `grad_k_norm`, `grad_g_log`, or
  `grad_beta` as separate global scratch tensors before the final backward
  phases
- later backward phases now read the existing fp32 accumulators directly
- forward launch count remains `1`
- backward launch count remains `1`
- for the dominant long-sequence `grad_w_qkv` phase only, the backward kernel
  now switches to a CTA-local split over the token-reduction dimension when
  `BT >= 2048`, then reduces those warp partials in shared memory inside the
  same cooperative launch
- the kernel CTA width is now an explicit compile-time knob via
  `HGDN_THREADS`; a local `HGDN_THREADS=256` trial remained architecture-faithful
  but regressed the long-sequence helper point, so the live default stays
  `THREADS=128` until H100 data says otherwise

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
- eager recurrence reference (`use_fla=False`) is the hard parity gate
- the harness also records the live `use_fla=True` control path when FLA is
  installed

The parity tolerances used by the harness were:

- forward: `atol=3e-2`, `rtol=3e-2`
- backward: `atol=1.2e-1`, `rtol=1.2e-1`

These are still diagnostic bf16 tolerances, not the final tightened threshold set.

### Eager reference pass cases

- `B=1,T=8`
  - `forward_y`: max abs `0.00585938`, rmse `0.00148067`, norm-rel `0.00854068`
  - `grad_x`: max abs `0.000671387`
  - `grad_w_qkv`: max abs `0.0117188`
  - `grad_w_out`: max abs `0.000518799`
  - `grad_conv_w`: max abs `0.00488281`
  - `grad_A_log`: max abs `0.000276035`
  - `grad_dt_bias`: max abs `0.000178221`
- `B=1,T=32`
  - `forward_y`: max abs `0.0078125`, rmse `0.00150397`, norm-rel `0.00797955`
  - `grad_x`: max abs `0.000747681`
  - `grad_w_qkv`: max abs `0.0161133`
  - `grad_w_out`: max abs `0.000915527`
  - `grad_conv_w`: max abs `0.0078125`
  - `grad_A_log`: max abs `0.000457931`
  - `grad_dt_bias`: max abs `0.000311816`
- `B=1,T=128`
  - `forward_y`: max abs `0.0136719`, rmse `0.00146453`, norm-rel `0.0080668`
  - `grad_x`: max abs `0.00183105`
  - `grad_w_qkv`: max abs `0.0390625`
  - `grad_w_out`: max abs `0.00149536`
  - `grad_conv_w`: max abs `0.0175781`
  - `grad_A_log`: max abs `0.00159452`
  - `grad_dt_bias`: max abs `0.00105244`
- `B=1,T=512`
  - `forward_y`: max abs `0.0117188`, rmse `0.00139698`, norm-rel `0.00762117`
  - `grad_x`: max abs `0.00305176`
  - `grad_w_qkv`: max abs `0.0605469`
  - `grad_w_out`: max abs `0.00195312`
  - `grad_conv_w`: max abs `0.0332031`
  - `grad_A_log`: max abs `0.00456837`
  - `grad_dt_bias`: max abs `0.00265132`
- optional `B=2,T=512`
  - `forward_y`: max abs `0.0185547`, rmse `0.00143757`, norm-rel `0.00786403`
  - `grad_x`: max abs `0.00256348`
  - `grad_w_qkv`: max abs `0.0585938`
  - `grad_w_out`: max abs `0.00230026`
  - `grad_conv_w`: max abs `0.0664062`
  - `grad_A_log`: max abs `0.00595432`
  - `grad_dt_bias`: max abs `0.00418161`
- optional `B=1,T=2048`
  - `forward_y`: max abs `0.0185547`, rmse `0.00145426`, norm-rel `0.00795628`
  - `grad_x`: max abs `0.00305176`
  - `grad_w_qkv`: max abs `0.101562`
  - `grad_w_out`: max abs `0.00390625`
  - `grad_conv_w`: max abs `0.0429688`
  - `grad_A_log`: max abs `0.00585778`
  - `grad_dt_bias`: max abs `0.00422701`

### FLA control result

- the live `use_fla=True` control path is now measured on the same saved weights
  and inputs
- on this local machine, FLA differs materially from eager even at `B=1,T=8`
  and continues to drift through `T=512`
- representative forward drift:
  - `B=1,T=8`: max abs `0.134766`, norm-rel `0.138049`
  - `B=1,T=32`: max abs `0.255859`, norm-rel `0.183047`
  - `B=1,T=128`: max abs `0.177734`, norm-rel `0.171794`
  - `B=1,T=512`: max abs `0.197266`, norm-rel `0.166777`
- representative backward-control drift:
  - `B=1,T=512 grad_w_qkv`: max abs `0.333984`, norm-rel `0.264985`
  - `B=1,T=512 grad_w_b`: max abs `0.43642`, norm-rel `6.75898`
- because the FLA control itself does not match eager, the harness records FLA
  comparisons as diagnostics only where the control drifts; eager remains the
  contract gate for the megakernel path
- a dedicated recurrence-boundary diagnostic now confirms the control mismatch
  starts immediately rather than only after long chunked replay:
  - `allow_neg_eigval=False`: first failing `T=1`, worst sampled `T=28`, max
    abs `0.152049`, norm-rel `0.524228`
  - `allow_neg_eigval=True`: first failing `T=3`, worst sampled `T=4`, max abs
    `0.0744246`, norm-rel `0.327112`
  - the new closed-form `T=1` check is decisive:
    - eager matches the formal no-beta-write recurrence essentially exactly
    - local FLA matches a beta-gated write candidate essentially exactly
    - this is a recurrence-semantics mismatch, not only late-sequence numeric drift
  - artifact: `hgdn_megakernel/cases/fla_recurrence_diag.json`

## Launch count

The measured megakernel path was isolated with a preallocated `grad_out` and direct
`torch.autograd.backward((out,), (grad_out,))`.

- forward launch count: `1`
- backward launch count: `1`

Observed CUDA-side entries for the HGDN path:

- `hgdn_forward_bf16_kernel`
- `hgdn_backward_bf16_kernel`

No extra CUDA helper kernels, memsets, or memcpys from the HGDN block path
itself appeared in the isolated launch-count region.

## CTA-width experiment

The default kernel build remains `THREADS=128`. A bounded local compile/test
trial with `HGDN_THREADS=256` preserved eager parity and the one-forward /
one-backward launch contract, but it was clearly worse on the local `sm_89`
helper:

| `HGDN_THREADS` | `B=1,T=512` fwd+bwd | `B=2,T=512` fwd+bwd | `B=1,T=2048` fwd+bwd | Result |
| ---: | ---: | ---: | ---: | --- |
| `128` | about `7.45 ms` | about `8.56 ms` | about `29.27 ms` | live default |
| `256` | `11.45 ms` | `14.91 ms` | `34.70 ms` | reject locally |

Local conclusion:

- keep `THREADS=128` as the default local/H100 starting point
- keep `HGDN_THREADS` exposed so H100-specific CTA-width tuning can happen
  without hand-editing the CUDA source
- the harness now also supports repeated CUDA-event timing via
  `--timing-repeats`; on the live `128 / 8 / 8` build, a `3`-repeat median pass
  on the local helper measured:
  - `B=1,T=512`: `7.53 ms` median forward+backward
  - `B=2,T=512`: `8.47 ms` median
  - `B=1,T=2048`: `27.78 ms` median, range `26.37-27.96 ms`
  Use the repeated timing path for future local kernel comparisons rather than
  trusting single-sample outliers.

## QKV splitM threshold experiment

The qkv weight-gradient splitM path remains the only dense-phase splitM variant
that has paid locally so far. The activation threshold for that path is now an
explicit compile-time knob via `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD`. A bounded
local trial lowering the trigger from the live default `2048` to `1024`
preserved eager parity and the one-forward / one-backward launch contract, but
it did not beat the current default on the repeated timing gate:

| `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD` | `B=1,T=512` median fwd+bwd | `B=2,T=512` median fwd+bwd | `B=1,T=2048` median fwd+bwd | Result |
| ---: | ---: | ---: | ---: | --- |
| `2048` | `7.53 ms` | `8.47 ms` | `27.78 ms` | live default |
| `1024` | `7.51 ms` | `8.54 ms` | `29.75 ms` | reject locally |

Local conclusion:

- keep `HGDN_GEMM_ATB_SPLIT_M_THRESHOLD=2048` as the live default
- keep the threshold exposed for future H100-specific qkv splitM tuning

## QKV post/norm recompute checkpoint

The current checkpoint removes `q_norm`, `k_norm`, `v_post`, `inv_q`, and
`inv_k` from the saved forward contract entirely:

- forward now keeps only `qkv`, `g_pre`, `beta_pre`, `g_log`, `beta`, `g_out`,
  `o_raw`, and `state_ckpt`
- backward reconstructs q/k/v post-conv state and q/k inverse norms from
  `qkv` and `conv_w` inside the same cooperative kernel
- the one-forward / one-backward launch contract remains intact

Local repeated timing on the `sm_89` helper is mixed but directionally useful:

| Checkpoint | `B=1,T=128` median fwd+bwd | `B=1,T=512` median fwd+bwd | `B=2,T=512` median fwd+bwd | `B=1,T=2048` median fwd+bwd | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| prior live default | `2.04 ms` | `7.53 ms` | `8.47 ms` | `27.78 ms` | `3`-repeat median |
| qkv post/norm recompute | `2.41-2.42 ms` | `9.00-9.04 ms` | `9.38-9.84 ms` | `24.65-26.86 ms` | three `3`-repeat confirmation passes |

Local conclusion:

- this recompute trade is worse on the helper at short/medium lengths
- it improves the long `T=2048` helper point by about `0.9-3.1 ms`
- that makes it worth keeping as an H100 compile/parity candidate because it
  materially reduces saved state and improves the long-sequence helper point,
  but it is still not a local proof of H100 timing quality

## Chunk replay cadence experiment

I also tested a higher-memory replay-cadence variant by rebuilding the same
source with `HGDN_REC_CHUNK_T=4` instead of the live `8`.

This keeps the same one-forward / one-backward launch contract and the same
model math, but it doubles the number of saved recurrence checkpoints and cuts
the within-chunk replay span in half.

Same-session repeated timing summary on the local `sm_89` helper:

| Build | `B=1,T=128` median fwd+bwd | `B=1,T=512` median fwd+bwd | `B=2,T=512` median fwd+bwd | `B=1,T=2048` median fwd+bwd | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| default `REC_CHUNK_T=8` | `2.41 ms` | `9.08 ms` | `9.81 ms` | `29.50 ms` | same-session rerun after restoring the default build |
| `REC_CHUNK_T=4` pass 1 | `2.23 ms` | `8.09 ms` | `8.93 ms` | `23.24 ms` | best long-sequence sample |
| `REC_CHUNK_T=4` pass 2 | `2.23 ms` | `8.10 ms` | `8.87 ms` | `28.30 ms` | second confirmation pass |

Memory cost of the same variant:

- `state_ckpt` doubles because chunk count doubles
- saved forward state per GDN block rises from about `0.80 GiB` to about
  `1.36 GiB` at `B=32,T=2048`
- saved forward state per GDN block rises from about `3.20 GiB` to about
  `5.45 GiB` at `B=128,T=2048`

Local conclusion:

- `REC_CHUNK_T=4` looks like a real speed-vs-memory trade rather than pure
  noise; it beat the same-session default on every measured case
- the long `T=2048` helper point improved materially but was less stable than
  the medium-length points
- the checkpoint-state increase is large enough that this should stay a bounded
  H100 compile/parity candidate for now, not the new live default

## Rejected local variants

These bounded local `sm_89` trials were architecture-faithful and stayed inside
the one-forward / one-backward owned megakernel path, but they did not survive
the local timing gate:

| Variant | Local result | Decision |
| --- | --- | --- |
| long-`BT` splitM for `grad_w_g` | one fast-looking single sample did not reproduce; repeat landed back near the committed baseline | reject |
| long-`BT` splitM for `grad_w_out` | about `29.89 ms` at `B=1,T=2048`, worse than the live default repeated gate | reject |
| `REC_CHUNK_T=16` | about `9.53 ms` at `B=1,T=512`, `10.69 ms` at `B=2,T=512`, and `33.98 ms` at `B=1,T=2048` | reject |

Keep these as recorded local dead ends unless a future H100-specific reason
justifies reopening them under a different measurement contract.

## Trainer compile smoke

The binding path now presents the HGDN block through a compile-visible
`torch.library` custom op rather than a local `torch.autograd.Function` island.

Fresh-cache trainer smoke used:

- `TORCH_LOGS=graph_breaks,recompiles`
- `TORCHINDUCTOR_CACHE_DIR=/tmp/pg_inductor_mk_grad_tail`
- `USE_WANDB=0`
- `WANDB_WATCH=none`
- `PERF_SKIP_FINAL_EVAL=1`
- `COMPILE=1`
- `COMPILE_STRATEGY=hybrid`
- `WARMUP_STEPS=1`
- `ITERATIONS=1`
- `TRAIN_BATCH_TOKENS=1024`
- `TRAIN_SEQ_LEN=128`
- `VAL_LOSS_EVERY=0`
- `NUM_LAYERS=2`
- `MODEL_DIM=384`
- `MLP_MULT=3.25`
- `GDN_RATIO=1`
- `GDN_USE_CUDA_MEGAKERNEL=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=0`
- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_CONTROL_PROJ_FP32=0`

Observed trainer-path result on the local 4070 helper:

- preflight reported `loaded=true`
- compile plan reported `gdn_disabled:0` and `gdn_megakernel_left_enabled:1`
- warmup completed
- the real training step executed
- peak memory log completed
- `PERF_SKIP_FINAL_EVAL=1` exited the run before artifact roundtrip/eval
- no graph-break or recompile log lines were emitted around the HGDN block path

Representative output:

```text
hgdn_megakernel_preflight:{"allow_jit_build": false, "arch_list": null, "loaded": true}
launch_contract:planned_train_tokens:1024 train_batch_tokens:1024 train_seq_len:128 grad_accum_steps:8 local_batch_size:1
compile_plan:strategy:hybrid gdn_disabled:0 gdn_megakernel_left_enabled:1 gdn_mlps_compiled:1 attn_blocks_compiled:1 model_compiled:1
warmup_step:1/1
compile_prewarm: muon_shapes:5 rotary_modules:1
step:1/1 train_loss:6.9387 train_time:35ms step_avg:35.30ms
peak memory allocated: 46 MiB reserved: 68 MiB
perf_mode: skipping serialization and final roundtrip eval
```

## CUDA-event timing

These timings are for the local `sm_89` device only.
Unless otherwise noted, the numbers below reflect the latest rebuilt
control-tail checkpoint with the live defaults
`REC_V_TILE=8`, `REC_CHUNK_T=8`.

### `B=1, T=8`

- forward: `0.78746 ms`
- forward + backward: `1.12538 ms`

### `B=1, T=32`

- forward: `0.24474 ms`
- forward + backward: `0.76800 ms`

### `B=1, T=128`

- forward: `0.48845 ms`
- forward + backward: `2.02957 ms`

### `B=1, T=512`

- forward: `1.50323 ms`
- forward + backward: `7.49875 ms`

### optional `B=2, T=512`

- forward: `1.79610 ms`
- forward + backward: `8.60365 ms`

### optional `B=1, T=2048`

- forward: `5.37395 ms`
- forward + backward: `29.76870 ms`

## REC_V_TILE sweep

The recurrence dot helper originally had a hidden 8-column assumption. I fixed
that by routing wider tiles through repeated 8-column slices inside the same
owned forward/backward kernels, then reran a bounded local sweep on the 4070
helper.

Same-source local timing summary, all with `REC_CHUNK_T=8` and the same
single-launch forward/backward contract:

| `REC_V_TILE` | `B=1,T=512` fwd+bwd | `B=1,T=2048` fwd+bwd | Result |
| ---: | ---: | ---: | --- |
| `8` | `8.01280 ms` | `32.60621 ms` | current default |
| `16` | `13.51475 ms` | `43.58963 ms` | slower |
| `24` | `16.87142 ms` | `53.78151 ms` | slower |
| `48` | `27.98899 ms` | `76.64436 ms` | much slower |

Local conclusion from this bounded sweep:

- wider recurrence value tiles are now parity-correct, so the earlier
  `REC_V_TILE=16` failure was an implementation bug rather than a math issue
- on this `sm_89` helper, larger `REC_V_TILE` values reduce recurrence-stream
  parallelism more than they save cross-tile accumulation work
- `REC_V_TILE=8` remains the live default until H100 data says otherwise

## Historical 4070 speed comparison vs current packed HGDN CUDA path

I also benchmarked an earlier pre-activation-recompute megakernel checkpoint
against the current packed `GatedDeltaNet` CUDA block path on the same local
RTX 4070 laptop GPU.

Important interpretation note:

- this section is a runtime-path speed comparison, not a same-equation parity
  comparison
- the baseline side used the historical local `use_fla=True` path
- the megakernel numerical contract is currently the eager HGDN path because
  local FLA still diverges materially from eager
- these numbers are kept as historical reference only
- they predate the current pre-drop checkpoint
- use the explicit timing table above plus the checkpoint delta notes below for
  the current same-source local status

Comparison setup:

- same weights copied into both modules
- same live packed HGDN block contract
- direct block timing, not end-to-end trainer timing
- backward isolated with a preallocated `grad_out`
- baseline path had `use_fla=True` active on this machine

### `B=1, T=128`

- current path:
  - forward `1.03690 ms`
  - forward + backward `3.26577 ms`
- megakernel:
  - forward `0.41160 ms`
  - forward + backward `2.03656 ms`
- relative:
  - forward speedup `2.519x`
  - forward + backward speedup `1.604x`

### `B=1, T=512`

- current path:
  - forward `0.91169 ms`
  - forward + backward `3.09097 ms`
- megakernel:
  - forward `0.88722 ms`
  - forward + backward `5.23036 ms`
- relative:
  - forward speedup `1.028x`
  - forward + backward speedup `0.591x`

### `B=1, T=1024`

- current path:
  - forward `0.94669 ms`
  - forward + backward `3.40004 ms`
- megakernel:
  - forward `1.72600 ms`
  - forward + backward `10.45432 ms`
- relative:
  - forward speedup `0.548x`
  - forward + backward speedup `0.325x`

### `B=1, T=2048`

- current path:
  - forward `0.93286 ms`
  - forward + backward `3.19488 ms`
- megakernel:
  - forward `3.42569 ms`
  - forward + backward `21.20602 ms`
- relative:
  - forward speedup `0.272x`
  - forward + backward speedup `0.151x`

### `B=2, T=512`

- current path:
  - forward `0.86564 ms`
  - forward + backward `3.30463 ms`
- megakernel:
  - forward `1.02582 ms`
  - forward + backward `5.95576 ms`
- relative:
  - forward speedup `0.844x`
  - forward + backward speedup `0.555x`

Local conclusion:

- parity is good
- launch structure is good
- the backward shared-memory shrink is a real local improvement
- the latest backward-scratch cleanup is also a real local improvement
- removing shared `chunk_states` and `dv0_hist` cut the candidate's dynamic
  backward recurrence tile from about `21.25 KiB` to about `9.00 KiB`
- removing global staging copies for `grad_q_norm`, `grad_k_norm`,
  `grad_g_log`, and `grad_beta` removed one full-grid sync and one dead
  backward scratch-write/read pass
- relative to the previous clean checkpoint `f6d0a23`, this checkpoint
  improved forward + backward by:
  - about `1.013x` at `B=1,T=128`
  - about `1.011x` at `B=1,T=512`
  - about `1.011x` at `B=1,T=1024`
  - about `1.014x` at `B=1,T=2048`
  - about `1.010x` at `B=2,T=512`
- the current megakernel is still a clear local win at `B=1,T=128`
- the current megakernel is still behind the packed HGDN path for `T=512+`,
  and especially for larger batch
- forward-only timing differences in this checkpoint should be treated as noise
  because the forward kernel itself did not change
- larger local batch points have not been rerun on this exact checkpoint yet;
  the last measured previous checkpoint was still not competitive:
  - `B=4,T=512`: baseline `3.39333 ms`, megakernel `14.28992 ms`
  - `B=8,T=512`: baseline `3.97802 ms`, megakernel `19.89581 ms`
- the next meaningful speed step is still backward-focused:
  fewer global accumulations, better dense phases, or a more aggressive
  Hopper-specific implementation

Latest control-tail checkpoint delta versus `243bda5`:

- the one-forward and one-backward launch structure stayed intact
- eager parity still passed through optional `B=1,T=2048`
- the checkpoint now parallelizes `grad_A_log` and `grad_dt_bias` accumulation
  across the existing `BT * H` control loop instead of leaving a serialized
  `H=8` tail at the end of the cooperative backward kernel
- local parity-harness forward + backward moved:
  - `B=1,T=128`: `2.25379 ms` -> `2.02957 ms`
  - `B=1,T=512`: `7.83974 ms` -> `7.49875 ms`
  - `B=2,T=512`: `8.97126 ms` -> `8.60365 ms`
  - `B=1,T=2048`: `30.88589 ms` -> `29.76870 ms`
- saved forward state is unchanged from `243bda5`
- on this `sm_89` helper this is the cleanest local long-sequence win since the
  earlier qkv-only long-`BT` dense-gradient split

## Rejected follow-up branch

I also tried the first structural follow-up after the hardening commit:

- split the backward dense weight-gradient phases over the token dimension
  inside the same cooperative backward kernel
- wrote fp32 partial tiles into a reusable workspace
- reduced those partials back to `grad_w_out`, `grad_w_qkv`, and `grad_w_g`
  inside the same single backward launch
- eager parity still passed
- launch count stayed at `1` forward and `1` backward
- no extra CUDA-side helper kernels appeared

Local result: not a win on this `sm_89` 4070 laptop helper, so it was reverted
instead of kept live in the branch.

Representative forward + backward timing deltas versus the hardened baseline:

- `B=1,T=128`: baseline `2.12275 ms`, split-K trial `2.12787 ms`
- `B=1,T=512`: baseline `7.84794 ms`, split-K trial `7.94010 ms`
- `B=2,T=512`: baseline `8.92211 ms`, split-K trial `9.09414 ms`

Interpretation:

- the added partial-write plus reduction pass inside the cooperative kernel
  did not pay for itself on this local GPU
- dense weight-gradient parallelism is still the right problem to attack
  next, but this specific split-K layout is not yet the answer

## Readiness call

Current status: **ready for H100 compile/parity**

Rationale:

- builds cleanly on the local GPU
- parity now passes against the eager repo contract for `B=1,T=8/32/128/512`
  plus optional `B=2,T=512` and `B=1,T=2048`
- isolated launch counting shows exactly one HGDN forward kernel launch and one HGDN backward kernel launch
- no hidden CUDA-side copy, cast, memset, or helper kernel appears in the
  measured region
- trainer-side preflight now rejects missing extension availability and
  `GDN_CONTROL_PROJ_FP32=1` in megakernel mode
- autograd wrapper now hard-fails instead of inserting hidden `.contiguous()`
  or `.to()` kernels
- device properties are printed explicitly

Current status is **not** “ready for H100 timing testing” because:

- local `sm_89` timing is not a proxy for H100
- the current local FLA control path does not match eager, so the megakernel
  must continue to treat eager as the numerical contract until that divergence
  is separately resolved
- the current tensor-core path is WMMA-based and portable, not yet Hopper-tuned
- the recurrent core still loses to the live packed path at `T=512+` on the
  local 4070
- backward recurrence/replay and its global accumulation traffic are still too
  expensive
