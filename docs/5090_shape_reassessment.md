# 5090 Shape Reassessment

Last updated: `2026-04-26`

This note reopens decisions that were made under the old `core_dim=48`
parallel-minGRU geometry. The independent review is right: do not treat
`128x4x4` as the aligned answer. It is a speed-frontier probe that discards a
large amount of stacked recurrent memory.

## Core Rule

Think in terms of:

```text
core_dim x layers x inner_dim
```

not just:

```text
core_dim x layers x expansion
```

For minGRU, `inner_dim = int(core_dim * expansion)` is the positive recurrent
state scanned over time. It is not just an MLP width.

Current leader:

```text
48 x 12 x inner480 = 5760 recurrent cells
```

Shallow aligned speed probe:

```text
128 x 4 x inner512 = 2048 recurrent cells
```

So `128x4x4` is a real architectural shift from recurrent memory depth toward
wide frozen statistical basis resolution. That may win on wallclock, but it
must be measured as one point on a frontier.

## Completed Shape Evidence

Local synthetic full-step benchmarks use blocks0 + top-2 trigram memory at
`B=256`, `T=512`, no `torch.compile`, and `scan_backend=assoc_accel`.

| Shape | Intent | Trainable params | tok/s | Peak MiB |
|---|---|---:|---:|---:|
| `d48_l12_e10` | current quality leader geometry | `839,130` | `586,519` | `22,726.5` |
| `d64_l8_e8` | compact aligned inner-512 successor | `796,182` | `802,329` | `17,042.7` |
| `d64_l10_e8` | depth-preserving aligned successor | `993,944` | `651,124` | `20,729.9` |
| `d64_l12_e8` | same depth as current, inner 512 | `1,191,706` | `547,596` | `24,417.0` |
| `d96_l8_e5.333` | middle frozen width, inner 512 | `1,192,462` | `767,276` | `17,574.1` |
| `d128_l4_e4` | wide shallow speed control | `806,418` | `1,371,298` | `10,332.0` |
| `d128_l6_e4` | wide medium-depth candidate | `1,200,916` | `964,038` | `14,245.8` |
| `d128_l8_e4` | wide depth/capacity candidate | `1,595,414` | `743,737` | `18,159.6` |

Read:

- `d128_l4_e4` is a diagnostic speed control, not the main recommendation.
- `d96_l6_inner512` is the balanced first serious alternative: better frozen
  basis width than `64`, more recurrent memory than `128x4`, and clean
  `inner_dim=512`.
- `d64_l10_inner512` is the memory-preserving successor to `48x12x10`.
- `d128_l5_inner512` is the speed-frontier refinement to run before `128x6x4`.
- Synthetic benchmark names are blocks0 unless `--num-blocks` is set.

## Decisions To Reopen

High-impact:

- Controller shape ranking:
  - prior winners `12x6`, `12x8`, `12x10`, `10x12`, and `16x8` all used
    `core_dim=48`
  - expansion should now be derived from target inner widths like `512`, not
    treated as an abstract scalar
- Structure ranking:
  - `blocks0` beating heavier frozen blocks remains plausible
  - at `core_dim=96+`, blocks0 should remain the default with top-K memory tensors
  - if blocks return, test them through `d64` or reduced branch count first
- Schedule and LR:
  - `h3500/h7000` and `lr=3.5e-3` were tuned on the old geometry
  - retest only after a new geometry actually wins
- Gate / EMA / router:
  - failures were measured on saturated deep `core_dim=48` controllers
  - rerun only after base shape and trigram top-K are settled

Lower-impact:

- dense top-2 trigram memory:
  - the `~0.134` bpb `1B` gain is large enough that the direction is robust
  - exact top-K and artifact headroom should be rechecked after shape
    selection
- seed policy:
  - use seed `1337` for shape screens
  - add seeds only for final evidence or threshold ambiguity

## Three-Day Frontier Batch

Use top-2 trigram memory first. Do not test K=4 until shape is no longer the
dominant unknown.

Run the adaptive closeout:

```bash
bash scripts/run_5090_adaptive_closeout.sh --dry-run --frontier-batch-id geom1 --run-version geom1 --seed 1337 --no-run-benchmark --count-workers 2 --max-confirmations 2 --stop-after k4
bash scripts/run_5090_adaptive_closeout.sh --frontier-batch-id geom1 --run-version geom1 --seed 1337 --no-run-benchmark --count-workers 2 --max-confirmations 2 --stop-after k4
```

Manual staged equivalent:

```bash
bash scripts/run_5090_final3day_frontier_batch.sh --dry-run --run-version geom1 --seeds 1337
bash scripts/run_5090_final3day_frontier_batch.sh --run-version geom1 --seeds 1337
```

The batch does two things and then stops:

1. Stage 0 benchmark:
   - `current_d48_l12_i480`
   - `d64_l10_i512`
   - `d96_l6_i512`
   - `d96_l8_i512`
   - `d128_l4_i512`
   - `d128_l5_i512`
   - `d128_l6_i384`
   - `d160_l4_i512`
2. Stage 1 fixed-token K2/blocks0 screen:
   - `blocks0_d96_l6_i512`
   - `blocks0_d64_l10_i512`
   - `blocks0_d128_l4_i512`
   - `blocks0_d128_l5_i512`

The default serious contract remains:

- `4096` steps / `512M` planned tokens
- `TARGET_EFFECTIVE_STEP_TOKENS=131072`
- `seq_len=512`
- `batch_size=256`
- `carry_chunks=8`
- `bptt_chunks=1`
- `lr=3.5e-3`
- `lr_hold_steps=3500`
- `TRIGRAM_TOP_K=2`
- `blocks=0`
- 12 branch lags

Analyze after Stage 1:

```bash
conda run -s --name train python tools/analyze_5090_geometry_frontier.py \
  --run-version geom1 \
  --benchmark logs/5090_final3day/<batch_id>/geometry_frontier_benchmark.json
```

Promotion read against the current top-2 seed-`1337` screen
(`val_bpb=2.0751715673`, steady throughput about `571,660` tok/s):

- better by any clear margin:
  - promote to `8192`-step / `1B` confirmation
- within `0.020` bpb and at least `1.5x` faster:
  - run an `8192`-step time-matched confirmation
- within `0.035` bpb and at least `2.0x` faster:
  - run one time-matched confirmation before killing
- worse by `>0.040` bpb:
  - kill unless the curve slope is obviously unfinished

## After Geometry

For the best aligned survivor only, test longer BPTT:

```text
batch_size=128
seq_len=512
bptt_chunks=2
TARGET_EFFECTIVE_STEP_TOKENS=131072
```

Only after geometry and BPTT are read should top-K headroom run:

```bash
bash scripts/run_5090_trigram_memory_screen.sh --run-version v2 --trigram-top-k 4 --seeds 1337
```

If the winning geometry is not the old `48x12x10`, run K4 through
`scripts/run_5090_trigram_aligned_geometry_screen.sh` with that geometry
instead of the legacy memory launcher.

## Cache Policy

Trigram memory specs are materialized per frozen spec because the final
`spec.pt` must contain the memory tensors.

The expensive counted table is cached separately under:

```text
${TRIGRAM_MEMORY_TABLE_CACHE_ROOT:-~/.cache/experiments/param-golf-coreamp/trigram_memory_tables}
```

The key uses:

- data path
- training-shard fingerprint
- storage dtype
- vocab size
- base-bigram logits hash
- top-K / smoothing / residual clip / confidence cap
- optional `max_tokens`

Compatible `core_dim` shape ablations can attach the same counted trigram table
to different frozen specs instead of recounting the full 19.5B training-token
stream every time.
