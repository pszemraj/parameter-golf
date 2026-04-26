# 5090 Next Experiments

Last updated: `2026-04-26`

This note is the short working summary. The full architecture source of truth is now:

- [docs/5090_architecture_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_architecture_plan.md)
- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)

## Frontier Snapshot

Current best completed `1B` local points by three-seed mean:

| Rank | Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---|---:|---:|---:|---:|
| 1 | `blocks0_resid12_e10_trigramk2_lr0035_h7000_1b` | `2.0415615686` | `0.0018559894` | `574,798` | `7,331,969` |
| 2 | `blocks0_resid12_e10_lr0035_final_h7000_1b` | `2.1757877509` | `0.0029076698` | `576,426` | `4,007,243` |
| 3 | `blocks1_resid10_e12_lr0035_final_h7000_1b` | `2.1765101841` | `0.0061762455` | `552,843` | `4,853,073` |

Interpretation:

- `blocks0 12x10 + trigram top-2` is now the local mean leader by a large
  margin.
- `blocks1 10x12` is still close enough to keep as the quality counterweight.
- The top-2 trigram confirmation improved over the previous blocks0 finalist
  mean by about `0.1342` bpb with full final validation coverage.
- Artifact estimate is still only about `7.33 MB`, so the remaining
  non-transformer budget should go into top-K/context-memory headroom rather
  than schedule microtuning.

## 5090 Performance Read

The current quality leader is not slow because of the trigram sidecar. Local
CUDA microbenchmarks at the maintained `B=256`, `T=512` contract show the
dominant cost is the recurrent controller geometry:

| Path | Shape | Synthetic full tok/s | Peak MiB | Trainable params |
|---|---|---:|---:|---:|
| current leader | `core_dim=48`, `layers=12`, `exp=10` | `586,729` | `22,726.5` | `839,130` |
| aligned probe | `core_dim=64`, `layers=8`, `exp=8` | `802,372` | `17,042.7` | `796,182` |
| aligned probe | `core_dim=128`, `layers=4`, `exp=4` | `1,371,917` | `10,332.0` | `806,418` |
| aligned probe | `core_dim=128`, `layers=6`, `exp=4` | `964,576` | `14,245.8` | `1,200,916` |

Core-only measurements tell the same story: `core_dim=128`, `layers=4`,
`exp=4` is about `2.6x` faster per token than the current `48x12x10`
controller at similar trainable parameter count.

Interpretation:

- the current `48`/`480` shape is CUDA-unfriendly and memory-heavy
- `torch.compile` is not a quick escape hatch because the current
  `accelerated-scan` Triton path fails under Inductor tracing
- this is a controlled GPU-friendly geometry probe, not an arbitrary
  transformer-style pivot
- the next metric read should test whether the faster aligned controller keeps
  or improves fixed-token `val_bpb`

The detailed reassessment and rerun matrix now live in:

- [docs/5090_shape_reassessment.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_shape_reassessment.md)

## Locked Schedule Defaults

The post-confirmation hold-retune screen is complete.

| Representative | `h2500` mean | `h3500` mean | `h4096` mean | Winner |
|---|---:|---:|---:|---|
| `blocks1 10x12` | `2.2724088059` | `2.2680191476` | `2.2892407482` | `h3500` |
| `blocks0 12x10` | `2.2732337556` | `2.2716177172` | `2.2823704208` | `h3500` |
| `blocks2 12x8` | `2.2821037577` | `2.2801177077` | `2.2941590571` | `h3500` |

Working defaults now:

- `lr_hold_steps=3500` for the `4096`-step / `512M` screen
- `lr_hold_steps=7000` for the `8192`-step / `1B` confirmation contract

`h4096` lost on all three reps, so “no decay” is not the answer.

## Training Budget Policy

- Full-dataset frozen spec/statistics build is always required:
  - all `195` train shards
  - `19,473,201,340` train tokens
  - no validation tokens
- The explicit validation shard is separate:
  - `1` shard
  - `62,021,846` tokens
  - used only for eval/full-val evidence, not frozen statistics
- Trigram sidecar specs are cached outside the repo under
  `${TRIGRAM_SPEC_CACHE_ROOT:-~/.cache/experiments/param-golf-coreamp}` and are
  keyed by the source spec hash plus sidecar parameters, so compatible ablations
  reuse the same full-data build.
- `512M` is the default serious screening budget.
- `1B` is the confirmation budget for finalists.
- Anything shorter than `512M` is only for smoke tests or harness checks.

## Seed Policy

Seeds are not a search axis. They only protect us from mistaking a lucky or
unlucky controller initialization/order for an architecture result.

Current policy:

- default screens use one canonical seed: `1337`
- rerun another seed only when a screen is close to a promotion threshold
- use `SEEDS="1337 2027 3141"` only for final evidence or when a result is
  likely to become the submission candidate
- do not pick winners by best seed

The top-2 trigram confirmation already showed low seed variation
(`std=0.0018559894`) relative to the architecture gain (`~0.1342` bpb), so the
remaining top-K/headroom probes should stay single-seed unless a result is
being packaged as final evidence.

## Final-Week Execution Read

The original final-week sequence has run far enough to close its main loop:

Launchers:

```bash
bash scripts/run_5090_safe_maxlr_probe.sh
bash scripts/run_5090_architecture_gate_screen.sh
bash scripts/run_5090_architecture_temporal_screen.sh
bash scripts/run_5090_architecture_router_screen.sh
bash scripts/run_5090_finalist_confirm1b.sh
```

Serious final-week launchers now fail before training if a shell override would
change the maintained protocol. Defaults are:

- `WANDB_PROJECT=pg-hconv-ablations`
- `SCAN_BACKEND=auto`
- `TORCH_BLAS_PREFER_CUBLASLT=1`
- `COMPILE=0`
- `GRADIENT_CHECKPOINTING=0`
- `RUN_VERSION=v2`

The deadline-oriented source of truth for ordering, promotion rules, and stop rules is:

- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)

## Current Batch Read

Completed so far in the final-week lane:

- safe `max_lr` screen:
  - `blocks1 10x12`: `lr=3.5e-3` beat `3.0e-3` by about `0.01303` bpb on seed `1337`
  - `blocks0 12x10`: `lr=3.5e-3` beat `3.0e-3` by about `0.01273` bpb on seed `1337`
  - decision:
    - promote `lr=3.5e-3` for both reps to seed `2027`

- gate screen:
  - `blocks1 10x12`:
    - `base` improved mean `val_bpb` by about `0.00241`
    - throughput loss was about `3.7%`
    - decision:
      - does **not** clear the `>= 0.003` promotion bar
      - treat gating as flat on the primary quality rep for now
  - `blocks0 12x10`:
    - `base` improved mean `val_bpb` by about `0.00582`
    - throughput loss was about `3.9%`
    - decision:
      - `gate=base` clears the promotion bar on the lean control
  - `core_base` did not clear the bar on either rep

- temporal screen:
  - primary lane ran on `blocks1 10x12` with `gate=none`
  - `ema` was worse than the matching `current` baseline by about `0.00457` bpb on seed `1337`
  - `ema_hybrid` was worse by about `0.01230` bpb on seed `1337`
  - decision:
    - EMA lane is flat/negative
    - skip router
    - move straight to queued `1B` finalist confirmations

- cleaned `v2` finalist confirmations:
  - `blocks0_resid12_e10_lr0035_final`: mean `2.1757877509`
  - `blocks1_resid10_e12_lr0035_final`: mean `2.1765101841`
  - both completed under exact `val_bpb`, explicit validation shards, and
    `scan_backend=assoc_accel`
  - decision:
    - keep both as current finalists
    - treat the ranking as close enough that new architecture probes should
      screen both, not only the current mean leader

- gate/LR sidecar:
  - completed `blocks1 gate=base lr=3.5e-3` seed `1337`
  - result: `2.2575790952`, slower than the matching no-gate safe-lane point
  - decision:
    - stop this sidecar
    - do not spend more queue time on the existing tokenwise gate formulation

## Plan Delta From The No-Fallback Audit

The high-level batch order did not change.

What changed is the acceptance contract for serious runs:

- serious runs now count only if they keep exact `val_bpb`
- serious CUDA runs now count only if `scan_backend=auto` resolves to `assoc_accel`
- directory-style FineWeb runs still require the explicit validation shard
- spectral basis builds no longer silently degrade to `svd`

Practical consequence:

- the next steps are still `max_lr`, gating, EMA, router, and then `1B` finalists
- but any run that only “works” by slipping onto approximate `bpb`, a slower scan backend, or another degraded path should be treated as invalid and rerun
- explicit smoke/debug opt-ins still exist, but they are no longer ambiguous with maintained-path experiment results

## Immediate Next Commands

The top-2 blocks0 trigram sidecar is confirmed under the `1B` contract. The
performance read makes one bounded geometry probe higher priority than top-K
headroom, because the current controller shape is both slow and memory-heavy.

Optional repeatable benchmark:

```bash
TORCH_BLAS_PREFER_CUBLASLT=1 conda run -s --name train python tools/benchmark_core_amp_perf.py --mode full --shape current_blocks0_d48_l12_e10:48:12:10 --shape aligned_d128_l4_e4:128:4:4 --batch-size 256 --seq-len 512 --warmup 4 --steps 10
```

Dry run the aligned top-2 geometry matrix first:

```bash
DRY_RUN=1 RUN_VERSION=v1 SEEDS=1337 bash scripts/run_5090_trigram_geometry_matrix.sh
```

Then run:

```bash
RUN_VERSION=v1 SEEDS=1337 bash scripts/run_5090_trigram_geometry_matrix.sh
```

Promotion rule:

- compare fixed-token `512M` `val_bpb` against the current top-2 seed-`1337`
  screen (`2.0751715673`)
- if an aligned shape is better, or within about `0.01-0.015` bpb while
  retaining a large throughput gain, promote it to a `1B` confirmation
- if it is clearly worse at fixed tokens, keep current `48x12x10` for quality
  and return to top-K headroom

After the geometry read, the top-K headroom question remains:

```bash
DRY_RUN=1 RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_sidecar_screen.sh
```

Then run:

```bash
RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_sidecar_screen.sh
```

Promotion rule:

- compare `K=4` to the matching `K=2` `512M` seed-`1337` screen
  (`2.0751715673`)
- if `K=4` improves by at least `0.015` bpb and artifact estimate stays under
  roughly `12 MB`, confirm `K=4` at `1B`
- if `K=4` is flat or worse, keep top-2 and move to packaging / optional
  blocks1 check

```bash
RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_confirm1b.sh
```

Replay `blocks1` only as a geometry check after the blocks0 top-K and aligned
geometry decisions:

```bash
RUN_VERSION=v1 SEEDS=1337 RUN_BLOCKS1=1 RUN_BLOCKS0=0 bash scripts/run_5090_trigram_sidecar_screen.sh
```

Use diagnostics on completed or partial runs before adding secondary adapters:

```bash
conda run -s --name train python tools/analyze_core_amp_run.py /path/to/run_dir --checkpoint /path/to/run_dir/final.pt --steps 64 --batch-size 64 --device cuda
```

The old `gate x lr` sidecar is no longer recommended.
The base-bigram delta and readout-delta scripts remain available, but they are
now secondary to the trigram memory probe.

Readout-delta is still available, but it is a secondary adapter lane. Use it
only after the trigram sidecar read is understood:

```bash
RUN_VERSION=v1 SEEDS=1337 RANKS="128 256" bash scripts/run_5090_readout_delta_screen.sh
```

## Why The Architecture Lane Changed

The schedule question is no longer the biggest uncertainty. The stronger thesis now is:

- frozen statistics should absorb easy tokens
- the recurrent controller should intervene selectively on hard tokens
- temporal structure should come from real causal multi-timescale taps, not just alternate projections of the current state

That means the next architecture order is now:

1. test top-K headroom (`K=4`, then only consider `K=8` if artifact estimate
   remains comfortably under cap)
2. confirm `K=4` only if it clearly beats top-2 on the same screen contract
3. replay `blocks1` only after the blocks0 top-K decision
4. base-bigram delta or readout-delta only as secondary calibration/capacity
   checks
5. optional score-first adaptive n-gram cache only after the static sidecar is
   validated and compliance is documented

Current practical interpretation:

- EMA and EMA-hybrid did not survive on the primary `blocks1` lane
- router stays skipped
- the existing gate cross-term did not survive at `lr=3.5e-3`
- current lag operators are not evidence against temporal structure because
  they do not expose literal high-order context identity
- top-2 trigram sidecar is now the strongest non-transformer use of artifact
  budget and should own the remaining confirmation window

## Fast-Scan Note

`scan_backend=auto` is now wired through the active Core/Amplifier path as the fail-loud default.

- on CUDA it requires `accelerated-scan`, then resolves to `assoc_accel`
- on non-CUDA devices it resolves to the repo-local Torch `assoc` path
- it no longer silently downgrades to a slower backend on the maintained CUDA path

Local 5090 microbenchmarks on representative recurrence shapes:

- minGRU layer (`B=256`, `T=512`, `dim=48`, `expansion=12`)
  - `heinsen`: `30.893 ms`
  - `assoc_accel`: `21.041 ms`
- EMA-style branch recurrence (`B=64`, `T=512`, `12 x 48`)
  - `sequential`: `95.314 ms`
  - `assoc_accel`: `2.844 ms`

Those are local primitive-level checks, not end-to-end training claims, but they are strong enough to keep `scan_backend=auto` as the working default for the new architecture lane, with `accelerated-scan` treated as core CUDA infrastructure and the slower `assoc` path owned in-repo.
