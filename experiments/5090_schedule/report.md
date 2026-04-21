# 5090 Schedule Report

## Status
- Harness ready
- Controller frontier is now stable enough to start schedule isolation
- Phase 1 hold sweep, edge follow-up, and `1B` hold confirmation are complete on the two best completed `blocks0` controllers
- The tuned-hold `blocks1` family is now also complete on the `1B` contract
- Working schedule defaults are now:
  - `lr_hold_steps=3500` for the `4096`-step / `512M` screening contract
  - `lr_hold_steps=7000` for the `8192`-step / `1B` contract
- The tuned hold changed the top local ranking twice:
  - inside `blocks0`, `10x12` moved ahead of `12x10`
  - once `blocks1` inherited the same tuned hold, `blocks1 12x10` became the best completed single-seed local point overall

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

## 1B Hold Confirmation Result

The proportional transfer held up and materially improved the two leading `blocks0` controllers.

| Controller | inherited `h1500` @ `1B` | tuned `h7000` @ `1B` | Delta |
|---|---:|---:|---:|
| `blocks0 12x10` | `2.2113941366` | `2.1954688682` | `-0.0159252684` |
| `blocks0 10x12` | `2.2128156660` | `2.1878016930` | `-0.0250139730` |

Direct comparison at the tuned `1B` schedule:

- `blocks0 10x12 h7000 1b = 2.1878016930`
- `blocks0 12x10 h7000 1b = 2.1954688682`
- `blocks0 10x12` now leads `blocks0 12x10` by about `0.00767` bpb

Important systems note:

- throughput stayed effectively unchanged relative to the inherited `1B` schedule runs
- this is genuine optimization gain, not a speed artifact

## Blocks1 Tuned-Hold Result

The same late-tail schedule also helped the one-block frozen-amplifier family enough to change the current local frontier.

| Controller | tuned `h7000` @ `1B` | Throughput | Artifact bytes | Comparison note |
|---|---:|---:|---:|---|
| `blocks1 12x10` | `2.1831753851` | `374,227` | `4,791,890` | best completed single-seed point overall |
| `blocks1 10x12` | `2.1935951525` | `372,809` | `4,790,848` | beats `blocks0 12x10 h7000` by about `0.00187` bpb |
| `blocks1 12x6` | `2.2132622271` | `588,014` | `4,167,982` | improved over old `blocks1 12x6 h1500` by about `0.02241` bpb |

Key direct comparisons:

- `blocks1 12x10 h7000 1b` beats `blocks0 10x12 h7000 1b` by about `0.00463` bpb
- `blocks1 10x12 h7000 1b` beats `blocks0 12x10 h7000 1b` by about `0.00187` bpb
- `blocks1 12x6 h7000 1b` is now the best fast anchor and improved over the inherited-hold `blocks1 12x6` point by about `0.02241` bpb

Interpretation:

- the late hold is not a `blocks0`-only effect
- one frozen amplifier block is now earning its bytes at the current single-seed frontier
- the next optimization work should center `blocks1`, with `blocks0 10x12 h7000` retained as the lean control
- this is still a single-seed ranking change; multi-seed confirmation is still required before treating the new order as final

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

Because `4096` no-decay lost cleanly and `3500` won for both families, the next disciplined move was a budget-matched `1B` confirmation using the scaled hold:

- `4096 -> 8192` total steps is an exact 2x step-budget increase
- so the direct proportional transfer of `h3500` is `h7000`

```bash
blocks0_12x10_hold_confirm1b_v1:
  h7000 @ 1B

blocks0_10x12_hold_confirm1b_v1:
  h7000 @ 1B
```

That confirmation has now landed successfully for both `blocks0` leaders and the follow-up `blocks1` family.

The next disciplined move is:

- keep `h7000` as the working `1B` schedule default for this family
- treat `blocks1 12x10 h7000` as the current single-seed quality leader
- run multi-seed confirmation on the leading `blocks1` and `blocks0` contenders before making stronger architecture claims
- then move to `max_lr`, `min_lr`, and only afterward to `warmup_steps` and `weight_decay`
- do not go back to extra frozen depth before the `blocks1` optimization lane is exhausted

## Open Schedule Questions
- Does `blocks1 12x10 h7000` stay ahead across seeds, or is the current lead fragile?
- Is `max_lr=3e-3` still the right peak LR now that `blocks1` is the frontier family?
- Is the current `min_lr=3e-4` floor leaving quality on the table on the one-block structure?
- Once LR is retuned, does `blocks1 10x12` close the remaining gap to `blocks1 12x10`, or is deeper geometry now the stable winner?
- After schedule work, does longer context help `blocks1` more than the leaner `blocks0` control?
