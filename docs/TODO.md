# HGDN Next Steps

Last updated: 2026-04-22 15:00 EDT

## 1. Run the local baseline-shaped HGDN search before more H100 spend

- The fair exact naive-contract batch already answered the branch-goal
  question for the current live14 replay shell:
  - exact repo baseline: `44.00 ms/step`, exact roundtrip `1.23710448`,
    `UNDER_LIMIT`
  - live HGDN finalist: `98.08 ms/step`, exact roundtrip `1.24735121`,
    `OVER_LIMIT`
- That means the next work is **not** another blind H100 rerun of live14.
  The next work is a contract-native HGDN shell search on the baseline-shaped
  surface:
  - `TRAIN_SEQ_LEN=1024`
  - `TRAIN_BATCH_TOKENS=65536`
  - `ITERATIONS=500`
  - `VAL_LOSS_EVERY=100`
  - `TRAIN_LOG_EVERY=25`
  - `WEIGHT_DECAY=0`
- New helper:
  [`../scripts/run_local_hgdn_naive_contract_search.sh`](../scripts/run_local_hgdn_naive_contract_search.sh)
- New size-screen config:
  [`../configs/hgdn/naive_contract_search.toml`](../configs/hgdn/naive_contract_search.toml)
- Initial artifact-size screen (`2026-04-22`) says all five first-pass
  baseline-shaped candidates are under the init-time proxy cap:
  - `l9_d512_r1_m2`
  - `l9_d512_r1_m1p75`
  - `l8_d512_r1_m2`
  - `l9_d480_r1_m2`
  - `l9_d512_r0_m2`
- Local `4070` GPU screen (`2026-04-23`, `300` steps, `seq=1024`,
  `65536` tokens/step, `WEIGHT_DECAY=0`) now ranks the baseline-shaped shells:
  - `l8_d512_r1_m2`: `560.56 ms/step`, `val_bpb=1.8709`
  - `l9_d512_r1_m1p75`: `630.64 ms/step`, `val_bpb=1.8758`
  - `l9_d480_r1_m2`: `631.19 ms/step`, `val_bpb=1.8803`
  - `l9_d512_r1_m2`: `646.60 ms/step`, `val_bpb=1.8515`
  - same-shell anchor `l9_d512_r0_m2`: `452.26 ms/step`,
    `val_bpb=1.9459`
- Same-shell winner-shell control on the same local contract:
  - `l8_d512_r1_m2`: `560.56 ms/step`, `val_bpb=1.8709`
  - `l8_d512_r0_m2`: `407.30 ms/step`, `val_bpb=1.9888`
- So the current contract-native HGDN leader is `l8_d512_r1_m2`.
  It pays about `1.38x` step time versus the same-shell attention-only control
  while buying back about `0.118` bpb at step `300`.
- The helper now defaults `PERF_SKIP_FINAL_EVAL=1` for local screening so the
  size screen handles artifact triage and the local run is not dominated by the
  quantized roundtrip tail.
- The attention-only `l9_d512_r0_m2` shell exists only to isolate the HGDN tax
  on the exact baseline-shaped architecture. It is not a replacement baseline.

```bash
USE_WANDB=0 WANDB_MODE=offline \
RUN_PREFIX_BASE=localnaivehgdn1 \
bash scripts/run_local_hgdn_naive_contract_search.sh
```

- Immediate local follow-up from this screen:
  - trim around `l8_d512_r1_m2`, not around the old live14 replay shell
  - compare the provisional `l8_d512_r1_m2` HGDN winner against the exact repo
    baseline only after one more bounded local trim pass or a targeted `1xH100`
    sanity run

## 2. Re-run the exact 8x packed HGDN tiebreak on the patched surface

- The `1xH100` compile matrix closed at `59d0817a` on the live14 replay shell:
  - `hybrid`: `799.05 ms/step`
  - `model`: `799.32 ms/step`, worse quality than `hybrid`
  - `selective`: `875.67 ms/step`, still alive because its final roundtrip
    stayed stronger than `hybrid`
- The first exact `8xH100` tiebreak closed on `2026-04-21 06:15 UTC`:
  - `hybrid`: `373.13 ms/step`, final exact roundtrip `2.39929889`
  - `selective`: `403.46 ms/step`, final exact roundtrip `2.39044828`
- That run answered the old compile-strategy question, but it did **not** yet
  include the current local fairness/perf patchset:
  - FA3 attention path on Hopper
  - `DISTRIBUTED_MODE=parallel_muon`
  - hard FLA fast-path preflight
  - manifest provenance fields
  - `cb026ab` distributed-runtime cleanup:
    - bucketed replicated-grad all-reduces in `parallel_muon`
    - baseline-style Muon bank sharding restored on the DDP path
    - bridge helper aligned with the same FA3 / distributed flags as the other
      exact-8x helpers
    - exact-8x bridge/tiebreak helpers default to `WANDB_WATCH=none`
- The latest distributed compile follow-up suppresses top-level model compile
  on multi-rank runs. On the exact `8xH100` `parallel_muon` surface,
  `COMPILE_STRATEGY=hybrid` now normalizes to the same effective plan as
  `selective`.
- Re-run the exact `8xH100` packed-HGDN helper under the same live14 finalist
  shell with the patched runtime surface. Treat the resulting bundle as the
  new distributed-safe packed compile reference instead of a meaningful
  `hybrid` vs `selective` tiebreak:

```bash
USE_WANDB=1 WANDB_MODE=online \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
WANDB_WATCH=none \
RUN_PREFIX_BASE=h100packed_tiebreak \
bash scripts/run_h100_hgdn_compile_tiebreak_round.sh
```

## 3. Re-run the naive-contract sanity batch against the exact repo baseline

- Use [`../scripts/run_h100_hgdn_naive_contract_round.sh`](../scripts/run_h100_hgdn_naive_contract_round.sh).
- Include exactly three legs:
  - exact repo baseline from `train_gpt.py`
  - live HGDN finalist
  - hybrid-trainer attention-only control
- The first fair naive-contract run on `2026-04-21 06:15 UTC` said:
  - exact repo baseline: `44.00 ms/step`, exact roundtrip `1.23710448`,
    `UNDER_LIMIT`
  - live HGDN finalist: `98.08 ms/step`, exact roundtrip `1.24735121`,
    `OVER_LIMIT`
  - hybrid attention-only control: `46.09 ms/step`, exact roundtrip
    `1.24098267`, `OVER_LIMIT`
- That is the branch-goal comparison. Packed HGDN lost badly on the exact
  baseline contract.
- The next rerun should keep the same comparison surface but include the
  patched HGDN runtime stack. Pin the hybrid-trainer legs to `WEIGHT_DECAY=0`.
- Pin the direct baseline leg explicitly to:
  - `DATA_PATH`
  - `TOKENIZER_PATH`
  - `VOCAB_SIZE`
- Treat the repo baseline as the calibration anchor. The hybrid attention-only
  control remains diagnostic only.

```bash
USE_WANDB=0 WANDB_MODE=offline \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
WANDB_WATCH=none \
RUN_PREFIX_BASE=h100naive1 \
bash scripts/run_h100_hgdn_naive_contract_round.sh
```

## 4. Keep HGDN-only finalist work bounded to closing the exact-baseline gap

- The branch objective is no longer ambiguous: HGDN only counts if it improves
  against the exact repo baseline in `train_gpt.py`.
- Do not spend more H100 time on HGDN-internal wins that do not move that
  comparison.
- Keep the live `14L x 384d x mlp3.25` family fixed while closing the speed gap.
- Current local priorities are:
  - remove attention-path delta versus the record-grade baseline stack
  - remove distributed optimizer / communication tax versus the record-grade
    baseline stack
  - fail fast if HGDN loses the FLA fast recurrence path
- Only after the patched reruns land should the next HGDN ablation screen be
  revisited.

## 5. Finish packed-path compile/runtime cleanup only where it still matters

- Treat the packed FLA recurrence path as compile-eligible; do not force it
  into eager-only islands again.
- Keep `TORCH_LOGS=recompiles,graph_breaks` opt-in only.
- The live local patchset now includes:
  - FA3 on Hopper for standard attention blocks
  - `DISTRIBUTED_MODE=parallel_muon`
  - `COMPILE_STRATEGY=hybrid` defaults in the active packed helpers
  - `--compile-strategy selective` in the structured launcher
  - W&B-off default for the naive-contract helper
  - helper manifests with git/branch/host/timestamp/attention/distributed fields
  - bucketed replicated-grad sync for the non-Muon params in `parallel_muon`
  - baseline-style Muon bank sharding on the DDP path
  - `WANDB_WATCH=none` default on the exact-8x bridge and tiebreak helpers
- Use explicit overrides instead of changing shared defaults again when running
  controlled comparisons.

## 6. Leave archived kernel paths archived

- Full-block megakernel: research-only.
- Core-kernel pivot: research-only.
- Do not spend new H100 time on either archived path unless a local rewrite
  produces a materially different result first.

## Recent checkpoints

- `7df6f8d1` (2026-04-18 branch point): core-kernel fork split from `exp/hgdn`.
- `c96ff08` (2026-04-18 19:03 CDT / 2026-04-19 00:03 UTC): clean `1xH100`
  core compare; packed `1191.52 ms/step`, core `6369.37 ms/step`.
- `0a433c9` (2026-04-19 12:47 CDT): packed helper defaults switched to the live
  `single-live14` replay shell.
- `d3842b6` (2026-04-19 12:50 CDT): packed eval path moved to `torch.no_grad()`
  after compile prewarm.
- `f3d8c2f` (2026-04-19 19:08 CDT): packed FLA-backed GDN blocks became
  compile-eligible in `selective` / `hybrid`.
- `59d0817a` (2026-04-20 02:38 UTC / 2026-04-19 21:38 CDT): live14 packed
  `1xH100` compile matrix
  closed with `hybrid` as the speed default and `selective` retained for the
  exact-8x tiebreak.
- `2026-04-21 06:15 UTC`: first exact `8xH100` packed tiebreak and first fair
  naive-contract sanity batch. Those runs are now historical references, not
  the final patched-runtime answer.
- `cb026ab` (2026-04-21 07:01 UTC / 2026-04-21 03:01 EDT): packed distributed
  runtime cleanup before the next exact-8x rerun.

## References

- branch status and commands: [README.md](README.md)
- chronology and exact scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- packed replay audit: [HGDN_PACKED_DRIFT_AUDIT.md](HGDN_PACKED_DRIFT_AUDIT.md)
- archived core-kernel notes: [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- cleanup backlog: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
