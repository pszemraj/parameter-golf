# 5090 Next Experiments

Last updated: `2026-04-21`

This note is the working next-step plan after the wide `blocks0` / `blocks1` / `blocks2` multi-seed confirmation finished.

## Frontier Snapshot

Current best completed local points by three-seed mean:

| Rank | Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes | Note |
|---|---|---:|---:|---:|---:|---|
| 1 | `blocks1_resid10_e12_h7000_1b` | `2.1865341393` | `0.0052472581` | `372,888` | `4,790,559` | best mean; one-block quality rep |
| 2 | `blocks1_resid12_e10_h7000_1b` | `2.1866023565` | `0.0051325073` | `374,271` | `4,791,951` | effectively tied with rank 1 |
| 3 | `blocks0_resid12_e10_h7000_1b` | `2.1899359311` | `0.0039254056` | `386,331` | `3,945,168` | best zero-block control |
| 4 | `blocks0_resid10_e12_h7000_1b` | `2.1947452986` | `0.0061822180` | `384,669` | `3,943,662` | earlier single-seed leader did not hold up |
| 5 | `blocks2_resid12_e8_h7000_1b` | `2.2005760974` | `0.0018741094` | `442,062` | `5,326,970` | fastest and most stable structural control |

Main takeaways:

- the two `blocks1` geometries are effectively tied on mean:
  - `12x10 - 10x12 = +0.0000682172` bpb
- the best zero-block representative is now `blocks0 12x10`, not `blocks0 10x12`
- `blocks2 12x8` is behind on quality but still matters:
  - fastest of the confirmed structural reps
  - lowest seed variance in the batch
- the known schedule lever is still larger than the remaining architecture gaps
- this is still not a transformer drift story:
  - no attention
  - no token-token mixing
  - still parallel minGRU over a frozen statistical basis

## Go-Forward Training Budget Policy

This is the working iteration policy for this family.

- full-dataset frozen spec/statistics build is always required
- `512M` is the default serious screening budget
- `1B` is the confirmation budget for finalists and stronger claims
- below `512M` is only for smoke tests, wiring checks, or obvious loser screens
- any `512M` result should be labeled as screening unless it is later confirmed at `1B`

Reason:

- the full-dataset spec build materially changed conclusions, so that part is non-negotiable
- the finished schedule work shows that `512M` is long enough to be directionally useful
- the finished wide confirmation also shows that close architecture rankings can move around, so the final calls still need `1B`

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
  - `TARGET_EFFECTIVE_STEP_TOKENS=131072`
  - `4096` steps = `512M` tokens
  - `VAL_EVERY=256`
  - `VAL_STEPS=8`
  - `LOG_EVERY=64`
  - `LOG_STATE_EVERY=256`
  - `SAVE_EVERY=2048`
- confirmation contract:
  - `seq_len=512`
  - `TARGET_EFFECTIVE_STEP_TOKENS=131072`
  - `8192` steps = `1B` tokens
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

## Phase A: Hold-Transfer Retune Screen

This is the highest-value immediate next step because the wide confirmation showed that:

- `blocks1` is best on mean, but its two controller geometries are still tied
- `blocks2` has never had a proper hold screen on the same disciplined contract
- schedule moved quality more than the current architecture gaps

Convenience launcher:

- `bash scripts/run_5090_schedule_retune.sh`
- useful overrides:
  - `SEEDS="1337 2027"`
  - `HOLDS="2500 3500 4096"`
  - `RUN_BLOCKS2=0`
  - `DRY_RUN=1`

Representative configs for this batch:

- `blocks1_resid10_e12`
- `blocks0_resid12_e10`
- `blocks2_resid12_e8`

Why these reps:

- `blocks1 10x12` has the best three-seed mean and slightly smaller artifact bytes than `blocks1 12x10`
- `blocks0 12x10` is now the best zero-block control on mean
- `blocks2 12x8` is the fairest surviving two-block structural control

Grid:

- seeds: `1337`, `2027`
- holds: `2500`, `3500`, `4096`

Success criteria:

- keep the best hold per representative by two-seed mean `val_bpb`
- if `blocks2` closes to within about `0.006` bpb of the best rep, keep it fully alive
- if `4096` wins on any rep, confirm that at `1B` before assuming the no-decay edge generalizes

## Phase B: Max-LR Screen

Only do this after the hold-transfer screen lands.

Use:

- the best two representatives from Phase A

Contract:

- `512M`
- best hold from Phase A per representative
- same effective step tokens
- seeds `1337`, `2027`

Grid:

- `max_lr in {2.5e-3, 3.0e-3, 3.5e-3, 4.0e-3}`

Promotion rule:

- only move forward points that beat the inherited `3e-3` setting clearly on mean `val_bpb`, or that tie while materially improving throughput or memory

## Phase C: Min-LR Follow-Up

Only do this after `max_lr` is narrowed.

Primary target:

- the best representative from Phase B

Grid:

- `min_lr in {1e-4, 2e-4, 3e-4, 5e-4}`

Keep fixed:

- best hold from Phase A
- best `max_lr` from Phase B
- `warmup_steps=100`
- `weight_decay=1e-3`

## Phase D: 1B Confirmation

Once the schedule lane is narrowed, rerun the top two tuned configs at:

- `8192` steps
- `1B` planned tokens
- seeds `1337`, `2027`, `3141`

This is the batch that should decide the new working winner.

## Phase E: Controller-Up Neighborhood

Only do this after schedule tuning is no longer a confound.

Primary points:

- `blocks1 14x10`
- `blocks1 12x12`

Optional stretch:

- `blocks2 12x10` if the tuned schedule keeps `blocks2` alive and memory remains comfortable

Rules:

- keep `num_blocks` fixed per family
- keep `core_dim` fixed
- use smaller local microbatch plus derived `grad_accum` only if the 5090 needs it
- turn on gradient checkpointing only if necessary and keep it explicit in summaries

## What Not To Do Next

- do not add more frozen blocks beyond `blocks2` yet
- do not touch `core_dim` yet
- do not re-open lag-heavy frozen temporal variants yet
- do not add attention or token-token mixing
- do not mix schedule screening and controller-up architecture changes into one batch

## Practical Recommendation

If only one experiment batch can be run next, make it this:

1. `bash scripts/run_5090_schedule_retune.sh`
2. keep the default reps:
   - `blocks1 10x12`
   - `blocks0 12x10`
   - `blocks2 12x8`
3. keep the default two-seed hold grid:
   - `2500`
   - `3500`
   - `4096`

That is the cleanest way to test whether the current `h7000` transfer is actually optimal for the true post-confirmation frontier.
