# 5090 Schedule Report

## Status
- Harness ready
- Controller frontier is now stable enough to start schedule isolation
- Phase 1 schedule work is an isolated `lr_hold_steps` sweep on the two best completed `blocks0` `1B` controllers
- No evidence-backed schedule winner yet

## What Was Verified
- Warmup-hold-cosine is the real root schedule path.
- `lr_hold_steps` is wired end to end.
- The trainer now records structured train/eval metrics and final run results, so schedule comparisons no longer depend on ad hoc log parsing.
- The sweep harness now supports a fixed effective-step-token contract, so schedule runs can later drop the local microbatch and recover apples-to-apples comparisons with derived `grad_accum` instead of silently changing the optimizer-step budget.

## Current Evidence
- No evidence-backed schedule winner yet.
- No claim is made yet about:
  - whether hold-then-cosine helps this controller family
  - whether decay is starting too early
  - whether `min_lr` is too high or too low
- The next schedule sweep should be run on the post-confirmation contenders, not on the invalidated older `blocks3` default.
- The first variable to move should be `lr_hold_steps`, because it can be isolated cleanly while keeping:
  - controller geometry fixed
  - `max_lr`, `min_lr`, `warmup_steps`, and `weight_decay` fixed
  - full-dataset `blocks0` shared spec fixed
  - planned screening tokens fixed at `512M`

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

After phase 1, use the winning hold value to stage isolated sweeps for `max_lr`, `warmup_steps`, `min_lr`, and finally `weight_decay`.

## Questions This Sweep Should Answer
- Does the inherited `1500`-step hold actually earn its keep on the radical `blocks0` controllers?
- Is the same hold value preferred by both `12 x 10.0` and `10 x 12.0`, or is schedule preference geometry-dependent?
- Is hold-then-cosine actually helping this recurrent-controller family?
- Are we decaying too early for the chosen local token budget?
- Is the current `min_lr` floor too conservative?
