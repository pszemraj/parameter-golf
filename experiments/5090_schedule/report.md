# 5090 Schedule Report

## Status

The schedule lane is effectively closed for the current local architecture frontier.

Completed:

- original `blocks0` hold sweep
- hold-edge follow-up
- `1B` hold transfer confirmation
- wide `blocks0` / `blocks1` / `blocks2` multi-seed confirmation
- post-confirmation two-seed hold-retune screen

Working schedule defaults:

- `lr_hold_steps=3500` for the `4096`-step / `512M` screening contract
- `lr_hold_steps=7000` for the `8192`-step / `1B` contract

Next move:

- architecture, not more hold work

## Training Budget Policy

- full-dataset frozen spec/statistics build is mandatory
- `512M` is the default schedule-screening budget
- `1B` is the confirmation budget
- shorter than `512M` is only for smoke tests or harness checks

## What Was Verified

- warmup-hold-cosine is the active training path
- `lr_hold_steps` is wired end to end
- the trainer records exact `val_bpb`, throughput, memory, and artifact estimates in structured outputs
- sweep comparisons can preserve a fixed effective-step-token contract even when local microbatch changes

## Blocks0 Hold Result

The inherited `1500`-step hold was too short for the zero-block radical controllers.

| Controller | `h0` | `h500` | `h1500` | `h2500` | `h3500` | `h4096` | Best tested |
|---|---:|---:|---:|---:|---:|---:|---:|
| `blocks0 12x10` | `2.2949891937` | `2.2876574385` | `2.2777913795` | `2.2696659544` | `2.2690508796` | `2.2830053665` | `h3500` |
| `blocks0 10x12` | `2.3017589365` | `2.2910034081` | `2.2794286891` | `2.2715466346` | `2.2669840064` | `2.2834303277` | `h3500` |

Interpretation:

- later decay helped materially
- no-decay was clearly worse

## `1B` Hold Transfer Result

The proportional `h3500 -> h7000` transfer held on the `1B` contract for the zero-block finalists.

| Controller | inherited `h1500` @ `1B` | tuned `h7000` @ `1B` | Delta |
|---|---:|---:|---:|
| `blocks0 12x10` | `2.2113941366` | `2.1954688682` | `-0.0159252684` |
| `blocks0 10x12` | `2.2128156660` | `2.1878016930` | `-0.0250139730` |

## Wide Multi-Seed Confirmation

The three-seed `1B` batch changed the frontier interpretation.

| Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---:|---:|---:|---:|
| `blocks1_resid10_e12_h7000_1b` | `2.1865341393` | `0.0052472581` | `372,888` | `4,790,559` |
| `blocks1_resid12_e10_h7000_1b` | `2.1866023565` | `0.0051325073` | `374,271` | `4,791,951` |
| `blocks0_resid12_e10_h7000_1b` | `2.1899359311` | `0.0039254056` | `386,331` | `3,945,168` |
| `blocks0_resid10_e12_h7000_1b` | `2.1947452986` | `0.0061822180` | `384,669` | `3,943,662` |
| `blocks2_resid12_e8_h7000_1b` | `2.2005760974` | `0.0018741094` | `442,062` | `5,326,970` |

Key read:

- the two top `blocks1` geometries are effectively tied
- `blocks0 12x10` is the correct zero-block control
- `blocks2 12x8` remains useful as the fast, stable structural control

## Post-Confirmation Hold Retune

The two-seed `512M` retune screen rechecked the hold default on the true representatives.

| Representative | `h2500` mean | `h3500` mean | `h4096` mean | Winner |
|---|---:|---:|---:|---|
| `blocks1 10x12` | `2.2724088059` | `2.2680191476` | `2.2892407482` | `h3500` |
| `blocks0 12x10` | `2.2732337556` | `2.2716177172` | `2.2823704208` | `h3500` |
| `blocks2 12x8` | `2.2821037577` | `2.2801177077` | `2.2941590571` | `h3500` |

Interpretation:

- `h3500` won on every representative
- `h4096` lost on every representative
- the schedule story is stable enough now that more hold work is not the best use of the GPU

## Conclusion

The schedule lane delivered real gains and reduced uncertainty, but it is no longer the bottleneck.

The next question is architectural:

- can the controller intervene more selectively on hard tokens?
- can real causal multi-timescale taps beat `current`?
- can per-token routing help without drifting toward a transformer?

See:

- [docs/5090_architecture_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_architecture_plan.md)
- [docs/5090_next_experiments.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_next_experiments.md)
