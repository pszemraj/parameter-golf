# HGDN Next Steps

Last updated: 2026-04-18

## 0. Pivot kernel work to the HGDN core boundary

- The full-block HGDN megakernel is no longer the active performance path.
- Keep it as a research/archive path only.
- The active kernel goal is now an **HGDN core kernel**:
  - dense `W_qkv`, `W_a`, `W_b`, `W_g`, `W_out` outside
  - owned CUDA path for conv + gate math + recurrence + output gate
- Do not pay for more full-block H100 compare work unless the question is
  explicitly about that archived design.
- Active plan and keep/kill gate:
  [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- Current local checkpoint:
  - `GDN_USE_CUDA_COREKERNEL=1` is wired through model + trainer + compile plan
  - the owned core op has a dedicated parity/launch harness:
    [`../hgdn_megakernel/test_corekernel.py`](../hgdn_megakernel/test_corekernel.py)
  - local `sm_89` parity passed through `B=2,T=512`
  - local compiled trainer smoke passed with `gdn_corekernel_left_enabled:7`
- Next paid gate:
  - run the bounded `1xH100` helper:
    [`../scripts/run_h100_single_gpu_hgdn_corekernel.sh`](../scripts/run_h100_single_gpu_hgdn_corekernel.sh)
  - default fixed-step compare there is:
    - packed control
    - core-kernel `rec_chunk_t=8`
  - only keep the branch active if that result gets back near the packed control

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

## Stop rules

- The current post-conv front-end seam is closed after `h100k20`.
- Stop any seam when the scoreboard says `FLAT`, `SATURATED`, or `INTEGRATION_BOTTLENECK`.
- Do not stop or continue work based on run count or commit count alone.

## References

- profiling chronology and exact scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- branch status and operating rules: [README.md](README.md)
- core-kernel pivot checklist: [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- W&B logging contract: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
