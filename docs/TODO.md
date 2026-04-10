# HGDN Next Steps

Last updated: 2026-04-10

## 1. Close the H100 resize ranking

- Keep the active kernel baseline at [`winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml).
- Keep the fixed-token H100 reference anchored on `h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048`.
- Current fixed-token H100 winner:
  - `h100retune3_b_fixed2k_hybrid_r1_mlp3.375_seq2048`
  - `14L x 384d x mlp3.375`
  - roundtrip `2.4365`
  - headroom `641,667`
- `h100retune3` rejected the active `15L x 384d` family:
  - the best `15L` point was still over limit and slower than the `14L` leader
  - the lower-MLP `15L` points gave back too much quality
- `16L x 384d x mlp2.6666666666666665` also lost the under-limit ranking.
- If rerunning for noise control, rerun only:
  - `h100retune3_b_fixed2k_hybrid_r1_mlp3.375_seq2048`
  - `h100retune3_a_fixed2k_hybrid_r1_mlp3.25_seq2048` as the over-limit ceiling anchor
- Do not spend more H100 fixed-token runs on the `15L x 384d` bracket unless a new size bracket changes the artifact tradeoff.

## 2. Run the H100 batch-scale / packing follow-up

- Start with:
  - `14L x 384d x mlp3.375`
  - optional over-limit ceiling control: `16L x 384d x mlp3.25`
- Vary:
  - `TRAIN_BATCH_TOKENS`
  - `GRAD_ACCUM_STEPS`
  - resulting `LOCAL_BATCH_SIZE`
- Separate:
  - microbatch / kernel-efficiency effects
  - effective-batch optimizer dynamics
- Use this pass to decide whether H100 headroom converts into wall-clock wins under the current stack.
- Do not restart the fixed-token ranking just because VRAM headroom exists.

## 3. Run the norm-placement screen

- Compare `NORM_STYLE=pre`, `post`, and `keel`.
- Keep the architecture and training contract fixed.
- Compare within HGDN first.
- Add the attention-only baseline only after the HGDN-side direction is clear.

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
