# 5090 Next Experiments

Last updated: `2026-04-22`

This note is the short working summary. The full architecture source of truth is now:

- [docs/5090_architecture_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_architecture_plan.md)

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

## Immediate Next Batch

The next architecture step is tokenwise residual gating, not more hold work.

Exact launcher:

```bash
bash scripts/run_5090_architecture_gate_screen.sh
```

Default batch:

- representatives:
  - `blocks1_resid10_e12`
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
  - `2027`
- gate modes:
  - `none`
  - `base`
  - `core_base`
- fixed controls:
  - `branch_temporal_mode=current`
  - `branch_router_mode=none`
  - `scan_backend=auto`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - `4096` steps
  - `TARGET_EFFECTIVE_STEP_TOKENS=131072`
  - `lr_hold_steps=3500`

Promotion rule:

- only promote a gate variant if it improves two-seed mean `val_bpb` by at least `0.003` with no more than `5%` throughput loss

## Why The Architecture Lane Changed

The schedule question is no longer the biggest uncertainty. The stronger thesis now is:

- frozen statistics should absorb easy tokens
- the recurrent controller should intervene selectively on hard tokens
- temporal structure should come from real causal multi-timescale taps, not just alternate projections of the current state

That means the next order is:

1. tokenwise residual gating
2. EMA / EMA-hybrid temporal taps
3. per-token branch routing
4. `1B` confirmation on the architecture winners

## Fast-Scan Note

`scan_backend=auto` is now wired through the active Core/Amplifier path.

- on CUDA it resolves to `assoc_accel` when `assoc-scan` and `accelerated-scan` are available
- otherwise it falls back cleanly to `assoc`, `heinsen`, or `sequential`

Local 5090 microbenchmarks on representative recurrence shapes:

- minGRU layer (`B=256`, `T=512`, `dim=48`, `expansion=12`)
  - `heinsen`: `30.893 ms`
  - `assoc_accel`: `21.041 ms`
- EMA-style branch recurrence (`B=64`, `T=512`, `12 x 48`)
  - `sequential`: `95.314 ms`
  - `assoc_accel`: `2.844 ms`

Those are local primitive-level checks, not end-to-end training claims, but they are strong enough to keep `scan_backend=auto` as the working default for the new architecture lane.
