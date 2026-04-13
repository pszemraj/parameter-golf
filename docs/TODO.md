# HGDN Next Steps

Last updated: 2026-04-13

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

## 2. Rerun for confidence only as needed

- Do not reopen H100 proxy architecture search by default.
- The exact bridge margin is large enough that the next paid runs should be confirmation or HGDN-only improvement work, not another cross-family ladder.
- If more exact runs are paid for, keep them tightly bounded:
  - one additional exact HGDN confirmation seed if needed
  - only rerun the attention-only baseline again if a regression check genuinely requires it

## 3. Run HGDN-only finalist work on the live bracket

- Compare `NORM_STYLE=pre`, `post`, and `keel`.
- Keep the architecture and training contract fixed.
- Compare within HGDN first.
- Prefer changes that preserve or improve artifact headroom because the current exact HGDN result is only `834,652` bytes under the cap.

## 4. Work the remaining non-seam HGDN hotspots

- output / gate projection path
- residual-shell glue on the compiled path
- recurrence-adjacent work only if compiled H100 profiles move recurrence back to the top tier

Do not reopen the current post-conv front-end seam for more ownership or packing tweaks unless the decomposition changes materially.

## 5. Use compile/backend work only on finalists

- graph-break and recompile audit with `TORCH_LOGS`
- `compiled_autograd`
- regional compilation for repeated block stacks
- dynamic-shape handling only where the logs justify it
- Nsight or backend swaps only if the compiled H100 profile says the backend is the bottleneck

## 6. Finish the cleanup backlog

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
- W&B logging contract: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
