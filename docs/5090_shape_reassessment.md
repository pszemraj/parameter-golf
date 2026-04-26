# 5090 Shape Reassessment

Last updated: `2026-04-26`

This note reopens decisions that were made under the old `core_dim=48`
parallel-minGRU geometry.

## Why This Matters

The old controller shapes were not just slow. They were architecturally
entangled with the conclusions:

- `core_dim=48` makes the recurrent input/output projections small and
  awkward for CUDA.
- the common `12x10` winner uses `dim_inner=480`, which is also an awkward
  scan/projection width.
- `core_dim` changes the frozen spec too:
  - token embedding width
  - lag-operator width
  - branch/readout width
  - frozen block cost
  - artifact budget left for top-K sidecars

So the shape problem can affect both quality and throughput. Treat the previous
shape as an implementation bottleneck, not as a settled architecture optimum.

## Completed Shape Evidence

Local synthetic full-step benchmarks use blocks0 + top-2 trigram sidecar at
`B=256`, `T=512`, no `torch.compile`, and `scan_backend=assoc_accel`.

| Shape | Intent | Trainable params | tok/s | Peak MiB |
|---|---|---:|---:|---:|
| `d48_l12_e10` | current quality leader geometry | `839,130` | `586,519` | `22,726.5` |
| `d64_l8_e8` | compact aligned inner-512 successor | `796,182` | `802,329` | `17,042.7` |
| `d64_l10_e8` | depth-preserving aligned successor | `993,944` | `651,124` | `20,729.9` |
| `d64_l12_e8` | same depth as current, inner 512 | `1,191,706` | `547,596` | `24,417.0` |
| `d64_l8_e10` | inner 640 capacity check | `993,814` | `654,104` | `20,504.9` |
| `d96_l8_e5.333` | middle frozen width, inner 512 | `1,192,462` | `767,276` | `17,574.1` |
| `d96_l10_e5.333` | middle width plus more depth | `1,488,014` | `623,334` | `21,367.8` |
| `d128_l4_e4` | wide shallow speed control | `806,418` | `1,371,298` | `10,332.0` |
| `d128_l6_e4` | wide medium-depth candidate | `1,200,916` | `964,038` | `14,245.8` |
| `d128_l8_e4` | wide depth/capacity candidate | `1,595,414` | `743,737` | `18,159.6` |

Read:

- `d128_l4_e4` is a diagnostic lower-bound on recurrent depth, not the main
  recommendation.
- `d64_l10_e8` is the conservative successor to `48x12x10`: it keeps real
  depth and aligns the inner width at `512`.
- `d128_l6_e4` and `d128_l8_e4` are the serious wide candidates because they
  preserve enough recurrent depth while improving frozen-spec resolution and
  still beat current throughput.
- `d96` is a useful diagnostic because `amp_dim=1152` and `dim_inner=512` are
  aligned, but it is less clean than `64` or `128` under the local CUDA-shape
  guardrail.

## Decisions To Reopen

High-impact:

- Controller shape ranking:
  - prior winners `12x6`, `12x8`, `12x10`, `10x12`, and `16x8` all used
    `core_dim=48`
  - expansion should now be derived from target inner widths like `512` or
    `640`, not treated as an abstract scalar
- Structure ranking:
  - `blocks0` beating heavier frozen blocks remains plausible, but
    `core_dim` changes frozen readout quality and frozen block bytes
  - at `core_dim=128`, blocks0 is probably the only practical frozen-block
    setting with top-K sidecars
  - at `core_dim=64`, a single frozen block may still be feasible and should
    be considered only after the blocks0 shape read
- Schedule and LR:
  - `h3500/h7000` and `lr=3.5e-3` were tuned on the bad geometry
  - do not run a large schedule sweep first, but retest LR/hold after a new
    geometry actually wins
- Gate / EMA / router:
  - these were measured on saturated deep `core_dim=48` controllers
  - gate/EMA/router failures should not be treated as global negative results
  - rerun only after the base shape and trigram top-K are settled

Medium-impact:

- `blocks0` vs `blocks1` as the final structural counterweight:
  - old results are still useful for direction
  - exact ranking may change with `core_dim=64`
- readout-delta and base-bigram-delta:
  - likely less important than literal context memory
  - revisit only if diagnostics show residual readout calibration is the
    bottleneck under the winning shape

Lower-impact:

- dense top-2 trigram memory:
  - the `~0.134` bpb `1B` gain is large enough that the direction is robust
  - exact top-K, trigram scale, and artifact headroom should be rechecked
    after shape selection
- seed policy:
  - unchanged
  - use seed `1337` for shape screens, add seeds only for finalists

## Rerun Priority

Use top-2 trigram sidecar first. Do not test K=4 until shape is no longer the
dominant unknown.

First matrix, single seed, fixed `512M` contract:

```bash
DRY_RUN=1 RUN_VERSION=v1 SEEDS=1337 bash scripts/run_5090_trigram_geometry_matrix.sh
RUN_VERSION=v1 SEEDS=1337 bash scripts/run_5090_trigram_geometry_matrix.sh
```

Default rows:

```text
blocks0_d64_l8_e8    64  8  8.0
blocks0_d64_l10_e8   64  10 8.0
blocks0_d128_l6_e4   128 6  4.0
blocks0_d128_l8_e4   128 8  4.0
```

Promotion read:

- compare every row to current top-2 `512M` seed `1337`:
  - `blocks0_resid12_e10_trigramk2_lr0035_h3500_512m_s1337`
  - `val_bpb=2.0751715673`
  - steady throughput about `571,660` tok/s
- if a shape is better, promote it directly
- if a shape is within `0.01-0.015` bpb and materially faster, keep it alive
  because H100 wallclock training may turn speed into quality
- if all aligned shapes lose badly, keep `48x12x10` for quality and use shape
  work only for packaging/runtime notes

Second step after a winner:

1. confirm the winning shape at `1B`
2. only then run K=4 top-K headroom on that shape
3. rerun gate/EMA/router only if diagnostics show controller selectivity or
   temporal hidden-state memory is still the bottleneck

## Cache Policy

Trigram sidecar specs are still materialized per frozen spec because the final
`spec.pt` must contain the sidecar tensors.

The expensive counted table is now cached separately under
`${TRIGRAM_TABLE_CACHE_ROOT:-~/.cache/experiments/param-golf-coreamp/trigram_tables}`.
The key uses:

- data path
- storage dtype
- vocab size
- base-bigram logits hash
- top-K / smoothing / residual clip / confidence cap
- optional `max_tokens`

That means compatible `core_dim` shape ablations can attach the same counted
trigram table to different frozen specs instead of recounting the full 19.5B
training-token stream every time.
