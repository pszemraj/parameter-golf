# 5090 Schedule Report

## Status
- Harness ready
- Controller frontier is now stable enough to start schedule isolation
- Phase 1 hold sweep and edge follow-up are complete on the two best completed `blocks0` `1B` controllers
- The best tested `512M` screening hold is currently `3500` for both leaders
- No fully promoted global default yet because `3500` is still screening evidence until the corresponding `1B` confirmation lands

## What Was Verified
- Warmup-hold-cosine is the real root schedule path.
- `lr_hold_steps` is wired end to end.
- The trainer now records structured train/eval metrics and final run results, so schedule comparisons no longer depend on ad hoc log parsing.
- The sweep harness now supports a fixed effective-step-token contract, so schedule runs can later drop the local microbatch and recover apples-to-apples comparisons with derived `grad_accum` instead of silently changing the optimizer-step budget.

## Current Evidence
- There is now strong evidence that the inherited `1500`-step hold is too short for the radical `blocks0` controllers at the `512M` screening budget.
- Hold helped monotonically across the tested range for both leading controllers:

| Controller | `h0` | `h500` | `h1500` | `h2500` | Best tested |
|---|---:|---:|---:|---:|---:|
| `blocks0 12x10` | `2.2949891937` | `2.2876574385` | `2.2777913795` | `2.2696659544` | `h2500` |
| `blocks0 10x12` | `2.3017589365` | `2.2910034081` | `2.2794286891` | `2.2715466346` | `h2500` |

Delta relative to the current inherited `h1500` default:

- `blocks0 12x10`: `h2500` improved by about `0.00813` bpb
- `blocks0 10x12`: `h2500` improved by about `0.00788` bpb

Important systems note:

- steady-state throughput was effectively unchanged across the hold sweep
- the win is schedule quality, not a hidden speed artifact

## Edge Follow-Up Result

The edge probe resolved the main ambiguity from phase 1a.

| Controller | `h2500` | `h3000` | `h3500` | `h4096` | Best tested |
|---|---:|---:|---:|---:|---:|
| `blocks0 12x10` | `2.2696659544` | `2.2719321948` | `2.2690508796` | `2.2830053665` | `h3500` |
| `blocks0 10x12` | `2.2715466346` | `2.2684898656` | `2.2669840064` | `2.2834303277` | `h3500` |

Delta relative to the inherited `h1500` default:

- `blocks0 12x10`: `h3500` improved by about `0.00874` bpb
- `blocks0 10x12`: `h3500` improved by about `0.01244` bpb

Interpretation:

- `h3500` is the best tested setting for both top controllers on the `512M` screening contract
- effective no-decay (`h4096`) is clearly worse for both
- the schedule wants a very late cosine tail, not zero decay

What can now be stated with evidence:

- hold-then-cosine is helping this family
- decay is still starting too early under the current default
- the `512M` screening default should move from `h1500` to `h3500`

What cannot yet be stated:

- whether `h3500` is the right transferred hold for the `1B` budget
- whether `min_lr` is too high or too low once hold is tuned farther out

## Phase 1 Screening Contract

- same frozen structure: corrected full-dataset `blocks0` shared spec
- same effective step tokens: `131,072`
- same planned screening budget: `4,096` steps = `536,870,912` tokens
- same logging:
  - `VAL_EVERY=256`
  - `VAL_STEPS=8`
  - `LOG_EVERY=64`
  - `LOG_STATE_EVERY=256`
  - `SAVE_EVERY=2048`
- same schedule defaults except for `lr_hold_steps`:
  - `max_lr = 3e-3`
  - `warmup_steps = 100`
  - `min_lr = 3e-4`
  - `weight_decay = 1e-3`

Run the queue:

```bash
conda run -s --name train python tools/run_core_amp_schedule_sweeps.py
```

List or dry-run families:

```bash
conda run -s --name train python tools/run_core_amp_schedule_sweeps.py --list
conda run -s --name train python tools/run_core_amp_schedule_sweeps.py --dry-run
```

Current phase-1 families:

```bash
blocks0_12x10_hold_screen_v1:
  h0, h500, h1500, h2500

blocks0_10x12_hold_screen_v1:
  h0, h500, h1500, h2500
```

## Phase 2 Confirmation

Because `4096` no-decay lost cleanly and `3500` won for both families, the next disciplined move is a budget-matched `1B` confirmation using the scaled hold:

- `4096 -> 8192` total steps is an exact 2x step-budget increase
- so the direct proportional transfer of `h3500` is `h7000`

```bash
blocks0_12x10_hold_confirm1b_v1:
  h7000 @ 1B

blocks0_10x12_hold_confirm1b_v1:
  h7000 @ 1B
```

Only after that `1B` confirmation lands should the next isolated sweeps move to `max_lr`, `warmup_steps`, `min_lr`, and `weight_decay`.

## Questions This Sweep Should Answer
- Does the `h3500 -> h7000` transfer hold up on the `1B` budget?
- Does the same late-tail schedule remain best for both `12 x 10.0` and `10 x 12.0` after the budget change?
- Is hold-then-cosine actually helping this recurrent-controller family?
- Are we decaying too early for the chosen local token budget?
- Is the current `min_lr` floor too conservative?
