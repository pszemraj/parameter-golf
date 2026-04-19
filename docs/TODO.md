# HGDN Next Steps

Last updated: 2026-04-18 23:44 CDT

## 0. Reconcile the packed H100 winner path on the current branch

- The clean bounded core compare is done and fair enough to trust.
- Result:
  - packed control under the cleaned core helper: `1191.52 ms/step`
  - core `rc8`: `6369.37 ms/step`
  - losses matched; the systems boundary lost
- Keep both custom-kernel forks archived only:
  - full-block megakernel: research-only
  - core-kernel pivot: research-only after the clean keep/kill miss
- The next active work is on the packed HGDN winner path.
- Immediate paid check:
  - rerun the packed current-winner path on the current branch under the
    historical `COMPILE_STRATEGY=model` contract
  - exact command:
    `USE_WANDB=0 WANDB_MODE=offline COMPILE_STRATEGY=model RUN_PREFIX=h100packed_recheck python scripts/hgdn.py h100-perf fixed2k-hybrid --preset winner-20260405-19`
- Main question:
  - is the current branch's packed path still near the historical
    `~915 ms/step` H100 reference, or did the packed stack itself drift?
- Local packed-path runtime cleanup now landed on this branch:
  - both trainers use reusable pinned `int64` host staging buffers for the
    rank-local token batches instead of pageable CPU widening every step
  - both trainers now stage validation batches through reusable pinned `int64`
    host buffers as well, instead of doing pageable eval-batch widening on each
    validation slice
  - both trainers now reuse one representative warmup batch for compile priming
    instead of rereading and recopying fresh data that is thrown away after the
    warmup state reset
  - both trainers now keep a single `zero_grad(set_to_none=True)` pass per
    optimizer step instead of paying a redundant extra pass at step start
  - the baseline trainer now prewarms rotary caches before top-level compile,
    matching the hybrid trainer's cache discipline
  - the baseline trainer no longer runs step-0 validation by default; it now
    uses the same explicit `LOG_STEP0_EVAL=1` escape hatch as the hybrid path
  - disabled profiling ranges now reuse a shared no-op context instead of
    allocating fresh `nullcontext()` objects in the hot eager shell
  - both trainers now keep one line-buffered logfile handle open instead of
    reopening the run log on every emitted line
  - both attention implementations now cache rotary tables by
    `(seq_len, device, dtype)` rather than recasting cached fp32 tables on
    every call
- Local packed frontend probes on `2026-04-18` after `fce8e24`:
  - on the local `sm_89` helper, `l2_norm` on `bf16` already returns `bf16`, so
    the current q/k `.to(dtype=x.dtype)` sites are not triggering real cast
    kernels on this stack
  - a direct `_project_recurrence_inputs()` microbench at `B=1,T=1024,D=384`
    did **not** show a win from flipping the packed split path to
    `use_packed_qkv_single_contig=1` or `use_packed_qkv_split_copy=1`
  - keep the current packed split default until target-hardware evidence says
    otherwise
  - these are local/runtime cleanups only until the next bounded H100 compare
    confirms or rejects their end-to-end value
- Core history and keep/kill notes remain in
  [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md).

## 1. Keep the exact 8x bridge result as the architecture gate

- Keep the active kernel baseline at [`winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml).
- Keep `h100pack3_b_fixed2k_hybrid_r1_mlp3.25_seq2048` as the live H100 proxy reference that fed the exact bridge.
- The bounded H100 proxy ladder is done:
  - `local128` improved all three tested families
  - the live `14L x 384d x mlp3.25` anchor stayed in front
  - the two `15L x 384d` finalists did not survive the H100 proxy stage
- Exact `8xH100` matched-control result:
  - HGDN finalist:
    - run: `h100bridge1_exact_hybrid_r1_mlp3.25_seq2048`
    - stop-step eval: `2.3949` bpb at step `1564`
    - final roundtrip: `2.4206`
    - artifact: `UNDER_LIMIT`, headroom `834,652`
  - attention-only baseline:
    - run: `h100bridge1_exact_depth_mlp4.0_seq2048`
    - stop-step eval: `2.5638` bpb at step `1858`
    - final roundtrip: `2.6320`
    - artifact: `OVER_LIMIT`, headroom `-1,922,740`
- Keep HGDN as the main record-path architecture.
- Do not reopen broad cross-family architecture comparison unless a later exact run contradicts this result.

## 2. Run one absolute-competitiveness check on the naive-baseline contract

- The exact bridge answered the within-branch keep/kill question.
- It did **not** answer whether the current HGDN stack is remotely competitive
  with the repo's published naive baseline at `1.2244`.
- Run one bounded `8xH100` sanity batch under the official naive-baseline
  contract:
  - `TRAIN_SEQ_LEN=1024`
  - `TRAIN_BATCH_TOKENS=524288`
  - `VAL_LOSS_EVERY=200`
  - `TRAIN_LOG_EVERY=50`
  - `MAX_WALLCLOCK_SECONDS=600`
- Use [`../scripts/run_h100_hgdn_naive_contract_round.sh`](../scripts/run_h100_hgdn_naive_contract_round.sh).
- This batch should contain:
  - the exact repo naive baseline from `train_gpt.py`
  - the live HGDN finalist
  - one baseline-like attention-only control inside the hybrid trainer
- The exact repo naive baseline is the real calibration target here.
  - The hybrid-trainer attention-only control is only a secondary diagnostic
    run for isolating architecture effects inside the HGDN trainer stack.
  - It is not an acceptable substitute for the published baseline.
- Pin the hybrid-trainer legs to `WEIGHT_DECAY=0` for this check.
  - `train_gpt.py` does not apply optimizer weight decay.
  - Leaving the hybrid default `WEIGHT_DECAY=0.04` on makes this a different contract and can collapse both hybrid runs after roughly `1.6k-2k` steps.
- Compare all three against the recorded naive-baseline reference:
  - stop-step eval `1.2172`
  - final roundtrip `1.22436570`
- If the exact repo naive baseline is still near `1.22x` but the hybrid-trainer
  control and HGDN are nowhere near that scale, stop pretending the current
  HGDN training stack is ready for finalist garnish and shift work toward the
  trainer/optimization path instead.

## 3. Rerun for confidence only as needed

- Do not reopen H100 proxy architecture search by default.
- The exact bridge margin is large enough that the next paid runs should be confirmation or HGDN-only improvement work, not another cross-family ladder.
- If more exact runs are paid for, keep them tightly bounded:
  - one additional exact HGDN confirmation seed if needed
  - only rerun the attention-only baseline again if a regression check genuinely requires it

## 4. Run HGDN-only finalist work on the live bracket

- Compare `NORM_STYLE=pre`, `post`, and `keel`.
- Keep the architecture and training contract fixed.
- Compare within HGDN first.
- Prefer changes that preserve or improve artifact headroom because the current exact HGDN result is only `834,652` bytes under the cap.

## 5. Finish the recurrence-side compile cleanup

- Done:
  - the HGDN recurrence now runs behind an owned
    `hgdn_fla_v1::chunk_gated_delta_rule` boundary that bypasses the upstream
    FLA backend-registry lock/context-manager path
  - the full trainer smoke no longer shows HGDN graph breaks in the block loop
  - the custom-op fake contract now keeps `grad_g` as `float32`, which avoids
    the Inductor buffer-planning failure seen in the first owned-boundary pass
  - Muon Newton-Schulz helpers now prewarm on the live matrix-shape family, so
    the optimizer-side shape-family recompile no longer lands in the real
    training path
  - rotary caches now prewarm before compile, so the lazy attention cache fill
    no longer triggers a training-side recompile
- Remaining:
  - decide whether the train/eval `grad_mode` split should be prewarmed,
    isolated, or simply treated as a one-time eval-side dual-graph cost
  - resolve the remaining `torch.library.opcheck(... test_aot_dispatch_dynamic)`
    failure for the owned recurrence op, or document why trainer-path smoke is
    the acceptance gate
- Only move back to recurrence-adjacent kernel work if compiled H100 profiles
  put recurrence back in the top hotspot tier after this cleanup.

Do not reopen the current post-conv front-end seam for more ownership or packing tweaks unless the decomposition changes materially.

## 6. Use compile/backend work only on finalists

- graph-break and recompile audit with `TORCH_LOGS`
- `compiled_autograd`
- regional compilation for repeated block stacks
- dynamic-shape handling only where the logs justify it
- Nsight or backend swaps only if the compiled H100 profile says the backend is the bottleneck

## 7. Finish the cleanup backlog

- profiler/report toolchain consolidation
- shared env-flag and launch-contract helpers
- trainer utility dedup where behavior matches
- test parameterization cleanup

See [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md) for the concrete code targets.

## 8. Record the 2026-04-19 local runtime-cleanup checkpoint

- Base commit before the cleanup pass: `f00e9b4a`.
- Scope of the validated must-have cleanup:
  - `train_gpt.py` and `train_gpt_hybrid.py` now use current-stream CUDA event
    waits around timed boundaries instead of repeated device-wide
    `torch.cuda.synchronize()` calls.
  - `train_gpt_hybrid.py` now uses CUDA events for `PERF_TIMING` and final
    roundtrip-eval timing on CUDA instead of host timers plus broad
    synchronization.
  - `model.py` now uses `match_reference_tensor(...)` on the hot HGDN/attention
    path to avoid redundant `.to(...)` casts when the tensor already matches the
    reference dtype/device.
  - The temporary shared `_NoOpContext` optimization was reverted after it
    caused a real Dynamo failure in a selective-compile hybrid smoke.
    Use `nullcontext()` for disabled profile scopes on compiled paths unless a
    future replacement is proven compile-safe.
- Validation artifacts from this checkpoint:
  - synthetic shards: `local-scratch/smoke_data/fineweb_train_000000.bin`,
    `local-scratch/smoke_data/fineweb_val_000000.bin`
  - baseline trainer smoke: `local-scratch/smoke_train_gpt.log`
  - hybrid trainer smoke: `local-scratch/smoke_train_gpt_hybrid.log`
- Keep this checkpoint as the minimum local proof before asking for another
  H100 rerun on the packed path.

## 9. Record the 2026-04-19 packed-loader follow-up checkpoint

- Base commit before the loader follow-up: `e08be3a`.
- Scope of the validated packed-path follow-up:
  - both trainers now stage `x` tokens as `int32` and keep `y` as `int64`
    across train and eval batch staging
  - `train_gpt.py` now times the final int8 roundtrip eval with CUDA events on
    CUDA instead of broad `torch.cuda.synchronize()` calls
  - both distributed token loaders now materialize only the local rank span and
    skip the other ranks' spans in-stream instead of reading the full global
    chunk on every rank and slicing one local view out of it
- Contract validation:
  - `PYTHONPATH=$PWD conda run -s --name pg python /tmp/check_loader_contract.py`
    returned `{'loader_contract_ok': True}` after matching the new loader
    against the old global-chunk reference for `world_size` `1`, `2`, and `4`
    across multiple steps on `local-scratch/smoke_data`
  - `local-scratch/smoke_train_gpt.log` passed again with compile enabled
  - `local-scratch/smoke_train_gpt_hybrid.log` passed again with compile
    enabled
- Local proxy evidence:
  - `PYTHONPATH=$PWD conda run -s --name pg python /tmp/bench_token_input_width.py`
    measured:
    - `int64_ms = 4.2373`
    - `int32_ms = 3.2885`
    - `speedup_x = 1.2885x`
  - treat this as local directionality only; the real keep/kill still belongs
    to the next bounded H100 packed-path rerun.

## Stop rules

- The current post-conv front-end seam is closed after `h100k20`.
- Stop any seam when the scoreboard says `FLAT`, `SATURATED`, or `INTEGRATION_BOTTLENECK`.
- Do not stop or continue work based on run count or commit count alone.

## References

- profiling chronology and exact scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- branch status and operating rules: [README.md](README.md)
- core-kernel pivot checklist: [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- W&B logging contract: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
