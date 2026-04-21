# 5090 Next Experiments

Last updated: `2026-04-21`

This note is the working next-step plan after the tuned-hold `blocks1` family finished and changed the single-seed local frontier.

## Frontier Snapshot

Current best completed local points:

| Rank | Run | `val_bpb` | Steady tok/s | Artifact bytes | Note |
|---|---|---:|---:|---:|---|
| 1 | `blocks1_resid12_e10_h7000_1b` | `2.1831753851` | `374,227` | `4,791,890` | best completed single-seed point |
| 2 | `blocks0_resid10_e12_h7000_1b` | `2.1878016930` | `384,758` | `3,899,812` | lean best-control point |
| 3 | `blocks1_resid10_e12_h7000_1b` | `2.1935951525` | `372,809` | `4,790,848` | second one-block geometry |
| 4 | `blocks0_resid12_e10_h7000_1b` | `2.1954688682` | `386,106` | `3,900,555` | close `blocks0` alternate |
| 5 | `blocks1_resid12_e6_h7000_1b` | `2.2132622271` | `588,014` | `4,167,982` | best fast anchor |

Main takeaways:

- one frozen amplifier block is now competitive enough to be the current single-seed winner
- the late hold is helping both `blocks0` and `blocks1`
- this is still not a transformer drift story:
  - no attention
  - no token-token mixing
  - still parallel minGRU over a frozen statistical basis

## Locked Comparison Contract

Do not change these unless the experiment explicitly studies that knob.

- data:
  - full local `fineweb10B_sp1024`
  - official tokenizer path
- trainer family:
  - frozen spec / frozen amplifier
  - parallel minGRU controller
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - `branch_temporal_mode=current`
- screening contract:
  - `seq_len=512`
  - effective step tokens `= 131,072`
  - `4096` steps = `512M` tokens
  - `lr_hold_steps=3500`
- confirmation contract:
  - `seq_len=512`
  - effective step tokens `= 131,072`
  - `8192` steps = `1B` tokens
  - `lr_hold_steps=7000`
- optimization defaults until explicitly changed:
  - `max_lr=3e-3`
  - `min_lr=3e-4`
  - `warmup_steps=100`
  - `weight_decay=1e-3`
- systems defaults for model-quality comparisons:
  - `COMPILE=0`
  - `TORCH_BLAS_PREFER_CUBLASLT=1`
  - no attention-style changes
  - no token-token mixing

## Phase A: Seed Confirmation And Structural Control

This is the most important immediate next step because the current frontier change is still single-seed.

Minimum set:

- `blocks1_resid12_e10_h7000_1b`
- `blocks0_resid10_e12_h7000_1b`

Preferred set:

- `blocks1_resid10_e12_h7000_1b`
- `blocks0_resid12_e10_h7000_1b`

Add this structural comparison arm:

- `blocks2_resid12_e8_h7000_1b`

Important nuance:

- `blocks2_resid12_e8_h7000_1b` is not a replay of an existing tuned `1B` result.
- It is a promoted `blocks2` structural control arm because the strongest completed `blocks2` point on disk is still the `512M` `blocks2_resid12_e8_c8t1_r3_current_512m` screen.
- That makes the widened batch a mix of:
  - true multi-seed confirmation on `blocks0` and `blocks1`
  - first fair tuned-`1B` structural comparison on `blocks2`

Seed set:

- `1337`
- `2027`
- `3141`

Success criteria:

- if `blocks1 12x10` stays ahead on mean `val_bpb`, promote it from single-seed leader to working winner
- if the lead collapses, keep `blocks0 10x12` as the lean default and treat `blocks1` as a promising but not yet stable branch
- if `blocks2 12x8 h7000` lands unexpectedly close, keep `blocks2` alive as a real structural branch instead of closing the door on deeper frozen stacks too early

## Phase B: Max-LR Screen On The One-Block Family

Only do this after at least the top two runs above have multi-seed coverage.

Models:

- `blocks1 12x10`
- `blocks1 10x12`

Contract:

- `512M`
- `h3500`
- same effective step tokens
- one seed for broad screening

Grid:

- `max_lr in {2.5e-3, 3.0e-3, 3.5e-3, 4.0e-3}`

Reason:

- `blocks1` is now the frontier family
- `3e-3` was inherited from the earlier `blocks0`-centered schedule lane
- the next disciplined question is whether the one-block winner wants a different peak LR

Promotion rule:

- rerun the best 2 `max_lr` points at `1B`
- keep `blocks0 10x12` as the lean control when interpreting whether the gain is real or architecture-specific

## Phase C: Min-LR Follow-Up

Only do this after `max_lr` is narrowed.

Primary target:

- the best `blocks1` shape from Phase B

Grid:

- `min_lr in {1e-4, 2e-4, 3e-4, 5e-4}`

Keep fixed:

- `warmup_steps=100`
- `weight_decay=1e-3`

Reason:

- the late hold moved the optimum
- the new leader may want a lower floor than the inherited `3e-4`

## Phase D: Controller-Up Neighborhood

Only do this after schedule tuning settles enough that we are not mixing controller and optimizer confounds.

Primary points:

- `blocks1 14x10`
- `blocks1 12x12`

Optional stretch if memory is acceptable:

- `blocks1 14x12`

Rules:

- keep `num_blocks=1`
- keep `core_dim` fixed
- use smaller local microbatch plus derived `grad_accum` if the 5090 needs it
- turn on gradient checkpointing only if necessary, and keep it explicit in summaries

Reason:

- the user explicitly wants controller-only capacity before changing frozen width
- these points push depth and width around the current `blocks1 12x10` winner without drifting toward a transformer

## Phase E: Longer Context

Only do this after schedule and controller shape are reasonably settled.

Compare:

- best tuned `blocks1` winner
- `blocks0 10x12` as lean control

Grid:

- `seq_len in {768, 1024}`

Rules:

- match total train tokens
- preserve the same effective-step-token discipline
- report both quality and throughput

## What Not To Do Next

- do not add more frozen blocks before the one-block lane is exhausted
- do not touch `core_dim` yet
- do not re-open lag-heavy frozen temporal variants yet
- do not add attention or token-token mixing
- do not mix multi-seed confirmation and schedule screening into one noisy batch

## Practical Recommendation

If only one experiment batch can be run next, make it this:

1. three-seed confirmation of `blocks1_resid12_e10_h7000_1b`
2. three-seed confirmation of `blocks0_resid10_e12_h7000_1b`

That is the highest-value evidence upgrade available right now.
