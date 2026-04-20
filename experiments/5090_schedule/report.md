# 5090 Schedule Report

## Status
- Harness ready
- Controller frontier is now stable enough to start schedule isolation
- Phase 1a `lr_hold_steps` sweep finished on the two best completed `blocks0` `1B` controllers
- The best tested hold is currently `2500` for both leaders
- No final schedule winner yet because the best hold landed on the edge of the tested range

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

What can now be stated with evidence:

- hold-then-cosine is helping this family
- decay is still starting too early under the current default

What cannot yet be stated:

- whether `h2500` is the real optimum
- whether the best setting is close to no decay for this screening budget
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

## Phase 1b Edge Follow-Up

Because both families won on the edge of the tested range, the next disciplined move is not a different LR yet. It is a direct hold-edge probe:

```bash
blocks0_12x10_hold_edge_v2:
  h3000, h3500, h4096

blocks0_10x12_hold_edge_v2:
  h3000, h3500, h4096
```

Interpretation of `h4096`:

- with `4096` total steps and `100` warmup steps, this is effectively a no-decay boundary check under the same cosine-with-hold code path

Only after the edge follow-up lands should the next isolated sweeps move to `max_lr`, `warmup_steps`, `min_lr`, and `weight_decay`.

## Questions This Sweep Should Answer
- Does the inherited `1500`-step hold actually earn its keep on the radical `blocks0` controllers?
- Is the same hold value preferred by both `12 x 10.0` and `10 x 12.0`, or is schedule preference geometry-dependent?
- Does the monotonic improvement continue past `h2500`, or do we hit a plateau / reversal closer to no decay?
- Is hold-then-cosine actually helping this recurrent-controller family?
- Are we decaying too early for the chosen local token budget?
- Is the current `min_lr` floor too conservative?
