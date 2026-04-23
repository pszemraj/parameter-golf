# 5090 Next Experiments

Last updated: `2026-04-23`

This note is the short working summary. The full architecture source of truth is now:

- [docs/5090_architecture_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_architecture_plan.md)
- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)

## Frontier Snapshot

Current best completed `1B` local points by three-seed mean:

| Rank | Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---|---:|---:|---:|---:|
| 1 | `blocks1_resid10_e12_h7000_1b` | `2.1865341393` | `0.0052472581` | `372,888` | `4,790,559` |
| 2 | `blocks1_resid12_e10_h7000_1b` | `2.1866023565` | `0.0051325073` | `374,271` | `4,791,951` |
| 3 | `blocks0_resid12_e10_h7000_1b` | `2.1899359311` | `0.0039254056` | `386,331` | `3,945,168` |
| 4 | `blocks2_resid12_e8_h7000_1b` | `2.2005760974` | `0.0018741094` | `442,062` | `5,326,970` |

Interpretation:

- `blocks1` is still the quality frontier, but the top two one-block geometries remain effectively tied.
- `blocks0 12x10` is the lean control.
- `blocks2 12x8` is still useful as the fast, stable structural control.

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

## Final-Week Execution Order

The week is now split into:

- safe lane:
  - a compact `max_lr` probe on the incumbent reps
- aggressive lane:
  - tokenwise gating
  - EMA / EMA-hybrid temporal taps
  - branch routing only if the temporal lane wins
- finalist lane:
  - `1B` three-seed confirmations on the promoted winners

Launchers:

```bash
bash scripts/run_5090_safe_maxlr_probe.sh
bash scripts/run_5090_architecture_gate_screen.sh
bash scripts/run_5090_architecture_temporal_screen.sh
bash scripts/run_5090_architecture_router_screen.sh
bash scripts/run_5090_finalist_confirm1b.sh
```

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

Safe-lane and temporal read now imply the next serious batch is the queued `1B` confirmation set:

```bash
bash scripts/run_5090_post_temporal_queue.sh
```

This queues:

- `blocks1_resid10_e12_lr0035_final`
- `blocks0_resid12_e10_lr0035_final`
- seeds `1337 2027 3141`

Sidecar note:

- `blocks0 gate=base` remains the one aggressive-lane result that actually cleared a promotion bar
- `blocks1 gate=base` missed the bar at `lr=3e-3`, but only narrowly
- that makes `gate=base x lr=3.5e-3` the one remaining architecture cross-term worth queueing behind the safe finalists

Exact sidecar command:

```bash
bash scripts/run_5090_gate_lr_sidecar.sh
```

Optional queueing wrapper:

```bash
bash scripts/run_5090_final_week_pueue.sh enqueue
```

That helper:

- boots a dedicated `pueue` daemon under `${HOME}/.cache/parameter-golf/pueue/5090_final_week`
- queues the safe finalist `1B` batch first
- queues the `gate=base x lr=3.5e-3` sidecar second with a success dependency

## Why The Architecture Lane Changed

The schedule question is no longer the biggest uncertainty. The stronger thesis now is:

- frozen statistics should absorb easy tokens
- the recurrent controller should intervene selectively on hard tokens
- temporal structure should come from real causal multi-timescale taps, not just alternate projections of the current state

That means the next architecture order is still:

1. tokenwise residual gating
2. EMA / EMA-hybrid temporal taps
3. per-token branch routing
4. `1B` confirmation on the architecture winners

Current practical interpretation:

- EMA and EMA-hybrid did not survive on the primary `blocks1` lane
- router stays skipped
- the only remaining architecture-side follow-up with good expected value is the `gate=base x lr=3.5e-3` cross-term

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
