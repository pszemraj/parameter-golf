# HGDN Megakernel Local Results

## Environment

- Date: 2026-04-16
- Host: local WSL laptop helper
- Python env: `conda run -s --name pg`
- Build command:
  `python setup_hgdn_megakernel.py build_ext --inplace`
- Validation command:
  `python hgdn_megakernel/test_megakernel.py`
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
  - bf16 WMMA tensor-core path for full dense tiles on `sm_80+`
  - warp-local scalar fallback for dense edge tiles
  - forward saves only recurrence chunk-start checkpoints
  - backward replays each chunk inside the same cooperative kernel
  - warp-shuffle-backed CTA reductions replace the old full-block tree
    reductions in q/k norm, output RMSNorm, and backward scalar reductions
  - `REC_V_TILE=8` recurrence dot products now use all block warps instead of
    only the first `8` threads for the per-column dot loops
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

## Trainer compile smoke

The binding path now presents the HGDN block through a compile-visible
`torch.library` custom op rather than a local `torch.autograd.Function` island.

Fresh-cache trainer smoke used:

- `TORCH_LOGS=graph_breaks,recompiles`
- `PERF_SKIP_FINAL_EVAL=1`
- `COMPILE=1`
- `COMPILE_STRATEGY=hybrid`
- `COMPILE_WARMUP_STEPS=1`
- `ITERATIONS=1`
- `TRAIN_BATCH_TOKENS=1024`
- `TRAIN_SEQ_LEN=128`
- `NUM_LAYERS=2`
- `MODEL_DIM=384`
- `GDN_RATIO=1`
- `GDN_USE_CUDA_MEGAKERNEL=1`

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
compile_plan:strategy:hybrid gdn_disabled:0 gdn_megakernel_left_enabled:1 gdn_mlps_compiled:1 attn_blocks_compiled:1 model_compiled:1
warmup_step:10/20
warmup_step:20/20
compile_prewarm: muon_shapes:4 rotary_modules:1
step:1/1 train_loss:6.9398 train_time:22ms step_avg:22.30ms
peak memory allocated: 75 MiB reserved: 92 MiB
perf_mode: skipping serialization and final roundtrip eval
```

## CUDA-event timing

These timings are for the local `sm_89` device only.

### `B=1, T=8`

- forward: `0.77722 ms`
- forward + backward: `1.20013 ms`

### `B=1, T=32`

- forward: `0.23552 ms`
- forward + backward: `0.72397 ms`

### `B=1, T=128`

- forward: `0.47821 ms`
- forward + backward: `2.12275 ms`

### `B=1, T=512`

- forward: `1.44077 ms`
- forward + backward: `7.84794 ms`

### optional `B=2, T=512`

- forward: `1.70701 ms`
- forward + backward: `8.92211 ms`

### optional `B=1, T=2048`

- forward: `5.37907 ms`
- forward + backward: `31.2832 ms`

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
  and optional `B=2,T=512`
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
