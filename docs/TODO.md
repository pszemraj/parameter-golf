# HGDN Next Steps

Last updated: 2026-04-20 13:40 CDT

## 1. Run the exact 8x packed compile tiebreak

- The `1xH100` compile matrix closed at `59d0817a` on the live14 replay shell:
  - `hybrid`: `799.05 ms/step`
  - `model`: `799.32 ms/step`, worse quality than `hybrid`
  - `selective`: `875.67 ms/step`, still alive because its final roundtrip
    stayed stronger than `hybrid`
- Do not kill `selective` yet.
- Run the exact `8xH100` packed-HGDN tiebreak under the same live14 finalist
  shell:

```bash
USE_WANDB=0 WANDB_MODE=offline \
RUN_PREFIX_BASE=h100packed_tiebreak \
bash scripts/run_h100_hgdn_compile_tiebreak_round.sh
```

## 2. Run the naive-contract sanity batch against the exact repo baseline

- Use [`../scripts/run_h100_hgdn_naive_contract_round.sh`](../scripts/run_h100_hgdn_naive_contract_round.sh).
- Include exactly three legs:
  - exact repo baseline from `train_gpt.py`
  - live HGDN finalist
  - hybrid-trainer attention-only control
- Pin the hybrid-trainer legs to `WEIGHT_DECAY=0`.
- Pin the direct baseline leg explicitly to:
  - `DATA_PATH`
  - `TOKENIZER_PATH`
  - `VOCAB_SIZE`
- Treat the repo baseline as the calibration anchor. The hybrid attention-only
  control remains diagnostic only.

```bash
USE_WANDB=0 WANDB_MODE=offline \
RUN_PREFIX_BASE=h100naive1 \
bash scripts/run_h100_hgdn_naive_contract_round.sh
```

## 3. Keep HGDN-only finalist work tightly bounded

- After the packed compile matrix and naive-contract batch, keep HGDN-only work
  inside the live `14L x 384d x mlp3.25` family.
- First learning-dynamics screen:
  - `NORM_STYLE=pre`
  - `NORM_STYLE=post`
  - `NORM_STYLE=keel`
- Keep architecture, batch contract, and artifact accounting fixed while
  screening those changes.

## 4. Finish packed-path compile/runtime cleanup only where it still matters

- Treat the packed FLA recurrence path as compile-eligible; do not force it
  into eager-only islands again.
- Keep `TORCH_LOGS=recompiles,graph_breaks` opt-in only.
- Only spend more compile/backend work where the packed H100 matrix or packed
  H100 profiles still show meaningful payoff:
  - `compiled_autograd`
  - regional compilation
  - train/eval grad-mode handling
  - backend swaps
- Packed helper defaults now use `COMPILE_STRATEGY=hybrid`.
- Use explicit overrides instead of changing shared defaults again when running
  controlled comparisons.

## 5. Leave archived kernel paths archived

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

## References

- branch status and commands: [README.md](README.md)
- chronology and exact scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- packed replay audit: [HGDN_PACKED_DRIFT_AUDIT.md](HGDN_PACKED_DRIFT_AUDIT.md)
- archived core-kernel notes: [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- cleanup backlog: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
