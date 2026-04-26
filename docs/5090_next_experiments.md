# 5090 Next Experiments

Last updated: `2026-04-25`

This note is the short working summary. The full architecture source of truth is now:

- [docs/5090_architecture_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_architecture_plan.md)
- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)

## Frontier Snapshot

Current best completed `1B` local points by three-seed mean after the cleaned
`v2` finalist rerun:

| Rank | Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---|---:|---:|---:|---:|
| 1 | `blocks0_resid12_e10_lr0035_final_h7000_1b` | `2.1757877509` | `0.0029076698` | `576,426` | `4,007,243` |
| 2 | `blocks1_resid10_e12_lr0035_final_h7000_1b` | `2.1765101841` | `0.0061762455` | `552,843` | `4,853,073` |

Interpretation:

- `blocks0 12x10` is now the local mean leader by a small margin and is much faster.
- `blocks1 10x12` is still close enough to keep as the quality counterweight.
- Both finalists are far under the 16 MB artifact budget, so the next useful
  architecture ideas should spend some of that budget on non-transformer
  capacity instead of rechecking small schedule variants.
- The top-2 trigram sidecar is not yet a `1B` confirmation, but its first
  `512M` screen is a decisive new local architecture signal:
  - `blocks0_resid12_e10_trigramk2_lr0035_h3500_512m_s1337`: `2.0751715673`
  - matching no-sidecar point: `2.2529073228`
  - sampled gain: about `0.1777` bpb
  - artifact estimate: `7,333,039` bytes, still `8,666,961` bytes under cap

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

- Full-dataset frozen spec/statistics build is always required.
- `512M` is the default serious screening budget.
- `1B` is the confirmation budget for finalists.
- Anything shorter than `512M` is only for smoke tests or harness checks.

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

The dense SP1024 trigram sidecar has cleared its first continuation bar. The
next step is to confirm the top-2 blocks0 lane under the `1B` contract with a
full validation final pass:

Dry run first:

```bash
DRY_RUN=1 RUN_VERSION=v1 SEEDS="1337 2027 3141" bash scripts/run_5090_trigram_confirm1b.sh
```

Then run:

```bash
RUN_VERSION=v1 SEEDS="1337 2027 3141" bash scripts/run_5090_trigram_confirm1b.sh
```

If the `1B` top-2 confirmation keeps a real margin, spend the next slot on
top-K headroom, still on blocks0 first:

```bash
RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_sidecar_screen.sh
```

Replay `blocks1` only after blocks0 top-2 is confirmed or after top-K headroom
still looks strong:

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

1. confirm top-2 trigram on blocks0 under the `1B` contract
2. test top-K headroom (`K=4`, then only consider `K=8` if artifact estimate
   remains comfortably under cap)
3. replay `blocks1` only if blocks0 confirmation/headroom stays strong
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
