# HGDN Next Steps

Last updated: 2026-04-12

## 1. Finish the narrow H100 exact-mappable packing refinement

- Keep the active kernel baseline at [`winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml).
- Keep the live bracket narrowed to:
  - `14L x 384d x mlp3.25`
  - `14L x 384d x mlp3.5`
- Run only the missing exact-mappable packing cross-term at `TRAIN_BATCH_TOKENS=1048576`:
  - `local32`
  - `local64`
- Use [`../scripts/run_h100_hgdn_resize_round.sh`](../scripts/run_h100_hgdn_resize_round.sh) for this batch.
- The output of this pass is a single bridge candidate, not another broad architecture screen.

## 2. Run one decisive exact 8x matched-control bridge

- After the exact-mappable H100 refinement picks the live finalist, run:
  - one exact HGDN submission-style training/eval contract run
  - one exact matched attention-only control run
- Keep trainer contract, tokenizer, eval path, and artifact accounting aligned.
- Do not add leaderboard garnish before this bridge.
- Treat this as the architecture go/no-go:
  - if HGDN wins clearly and stays legal, keep it as the main record path
  - if HGDN loses or only ties while staying materially more painful, demote it from the main record path

## 3. Run the norm-placement screen on the live bracket only

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
