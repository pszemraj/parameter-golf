# HGDN Core-Kernel Pivot

Last updated: 2026-04-18

## Why this pivot exists

The repo-backed full-block HGDN megakernel preserved the right modeling pieces,
but the `compare100` evidence killed it as a finalist-training path.

Matched `1xH100` fixed-step contract:

- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=2048`
- `grad_accum_steps=8`
- `local_batch_size=32`

Observed trainer throughput:

- full-block `base_rc8_v8`: about `8765.65 ms/step`
- full-block `rc6_v8`: about `8464.21 ms/step`
- full-block `rc4_v8`: about `8205.93 ms/step`
- live packed HGDN reference: about `915.10 ms/step`

That is not a micro-optimization problem. It means the custom kernel swallowed
the wrong ownership boundary: it replaced dense GEMM phases that H100 libraries
already do well, then paid cooperative-grid barrier costs around them.

## Active boundary

The active performance direction is now an **HGDN core kernel**.

The owned CUDA path should own:

- packed depthwise QKV conv
- post-conv SiLU
- q/k normalization
- gate math from `g_pre`, `beta_pre`, `A_log`, and `dt_bias`
- gated-delta recurrence
- output RMSNorm + SiLU gate

The owned CUDA path should not own:

- `W_qkv`
- `W_a`
- `W_b`
- `W_g`
- `W_out`

Those dense projections stay on the normal vendor/compiler GEMM path.

## Non-goals

- Do not keep paying H100 budget to polish the current full-block kernel.
- Do not reopen the old `hgdn_cuda` sidecar family as the main direction.
- Do not pretend a Python-level stitch-up of old sidecars is the new answer.
- Do not claim victory from fixed-token kernel microbenchmarks if trainer
  wallclock stays far from the packed control.

## Keep / kill gate

The first serious goal is not beating the packed HGDN control. The first goal is
to get back into the same order of magnitude.

Keep the core-kernel branch only if it can get within roughly `20-30%` of the
packed HGDN control on the matched `1xH100` fixed-step contract.

If it cannot clear that bar after the core boundary is implemented and compiled
cleanly, stop and keep the packed winner path as the mainline.

## Task list

### Phase 0: freeze the diagnosis

- [x] keep the full-block path labeled research-only in docs and helper text
- [x] make the active branch rules say “core kernel mainline, full-block research”
- [x] keep the exact `compare100` numbers nearby so future work does not drift

### Phase 1: create the new runtime boundary

- [x] add `GDN_USE_CUDA_COREKERNEL=1`
- [x] wire it through trainer config, W&B config, compile preflight, and model flags
- [x] keep `GDN_USE_CUDA_MEGAKERNEL=1` available only as the archived full-block research path

### Phase 2: implement the owned core op contract

- [x] forward inputs:
  - `qkv`
  - `g_pre`
  - `beta_pre`
  - `g_out`
  - `conv_w`
  - `A_log`
  - `dt_bias`
- [x] forward output:
  - `z` shaped `(B, T, H, Dv)`
- [x] backward outputs:
  - `dqkv`
  - `dg_pre`
  - `dbeta_pre`
  - `dg_out`
  - `dconv_w`
  - `dA_log`
  - `ddt_bias`

### Phase 3: verify architecture fidelity

- [x] preserve packed causal conv tap ordering
- [x] preserve beta-write recurrence
- [x] preserve fp32 gate math, fp32 recurrence state, fp32 output-norm accumulation
- [x] parity against eager reference on small and medium shapes
- [ ] parity against FLA when FLA is the intended semantic control

### Phase 4: measure the right thing

- [x] add a fixed-step compare helper that includes:
  - live packed HGDN control
  - HGDN core-kernel candidate
  - optional full-block research path
- [x] keep same batch contract / step budget / eval cadence across candidates
- [x] use `MAX_WALLCLOCK_SECONDS=0`
- [ ] treat the first `1xH100` result as a keep/kill checkpoint, not as a final leaderboard claim

## Current local checkpoint

- Static audit passes with both the archived full-block path and the new
  core-kernel boundary checks enabled.
- The new isolated harness is:
  [`../hgdn_megakernel/test_corekernel.py`](../hgdn_megakernel/test_corekernel.py)
- Local `sm_89` parity passed for:
  - `B=1,T=8`
  - `B=1,T=32`
  - `B=1,T=128`
  - `B=1,T=512`
  - optional `B=2,T=512`
- The owned core path currently proves exactly:
  - one `hgdn_core_forward_bf16_kernel`
  - one `hgdn_core_backward_bf16_kernel`
  inside the profiled region
- A tiny compiled trainer smoke also passed on the local GPU with:
  - `GDN_USE_CUDA_COREKERNEL=1`
  - `compile_plan: ... gdn_corekernel_left_enabled:7 ...`
- Important caveat:
  - the isolated local timing numbers in `test_corekernel.py` compare against
    the eager HGDN control, not the packed trainer control
  - do not treat those local ratios as H100 finalist evidence
  - the real keep/kill measurement is the new bounded `1xH100` `compare100`
    helper below

## Contract hygiene

- The active H100 core helper should default to the live packed control, which
  means `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1` for `packed_control`.
- The active core path should prefer `GDN_COREKERNEL_REC_CHUNK_T`; the legacy
  `GDN_MEGAKERNEL_REC_CHUNK_T` name remains a fallback until the archived
  full-block docs and helpers are physically split out.
- The active `1xH100` core helper should default to:
  - `HK_TRAINER_LAUNCHER_MODE=plain`
  - `COMPILE_STRATEGY=selective`
  so the validation path does not add single-rank DDP wrapper noise or an
  unnecessary top-level compile shell.
- The active `GatedDeltaNet.forward` core path should call
  `torch.ops.hgdn_corekernel_v1.run(...)` directly. Runtime cadence resolution,
  compatibility checks, and tensor-shape-to-Python-int conversions do not
  belong in the traced hot path.
- Trainer preflight should pin `corekernel_rec_chunk_t` onto the built GDN
  modules once, and owned GDN blocks should be compiled as submodules rather
  than merely "left enabled".
- Follow-up cleanup, after the first honest H100 compare:
  - physically split archived full-block notes from active core-kernel notes in
    `hgdn_megakernel/local_results.md` and
    `hgdn_megakernel/h100_resource_report.md`

## Immediate implementation order

1. run one bounded `1xH100` compile/parity smoke through the new helper
2. run the fixed-100-step packed-vs-core comparison
3. keep only if the core candidate gets back into the packed-control ballpark
4. only after that consider core-internal schedule tuning or save/recompute work

## References

- current branch status: [README.md](README.md)
- active next steps: [TODO.md](TODO.md)
- full-block experiment history: [../hgdn_megakernel/local_results.md](../hgdn_megakernel/local_results.md)
- profiling chronology: [PROFILING_LOG.md](PROFILING_LOG.md)
- H100 helper: [../scripts/run_h100_single_gpu_hgdn_corekernel.sh](../scripts/run_h100_single_gpu_hgdn_corekernel.sh)
