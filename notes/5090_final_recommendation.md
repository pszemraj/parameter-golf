# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The schedule-retune question is now materially reduced. The next real uncertainty is architectural: selective controller intervention, real temporal taps, and branch routing inside the same frozen-statistics + parallel minGRU family.

## Current Status

- W&B logging is clean in project `pg-core-amp`.
- Exact `val_bpb` is active through the official tokenizer path.
- Artifact accounting follows the record-style convention:
  - repo code bytes
  - `gzip(spec.pt)`
  - trainable int8-zlib payload
- The local dataset is complete for this family:
  - `195` train shards
  - `1` dedicated validation shard
- The regression-to-transformer guardrail is still intact:
  - no attention
  - no token-token mixing
  - winners remain parallel minGRU controllers over a frozen statistical basis
- Full-dataset frozen spec building is the standing requirement.
- Training-budget policy is now explicit:
  - `512M` for serious screening
  - `1B` for confirmation
- Final-week execution order is now documented in:
  - [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)
- Final-week lane status:
  - safe `max_lr` screening promoted `lr=3.5e-3` on both primary reps from seed `1337`
  - two-seed gating was flat on `blocks1 10x12`
  - two-seed gating promoted `gate=base` on `blocks0 12x10`
  - `blocks1` temporal screening with `gate=none` was negative for both `ema` and `ema_hybrid`
  - router is now skipped under the plan stop rules
  - next queued serious batch is the `1B` safe-lane confirmation set at `lr=3.5e-3`

## Top 3 Current Contenders

These are still the strongest completed local contenders by three-seed mean quality.

1. `blocks1_resid10_e12_h7000_1b`
   - mean `val_bpb = 2.1865341393`
   - std `= 0.0052472581`
   - mean steady `tok/s = 372,888`
   - mean artifact estimate `= 4,790,559`

2. `blocks1_resid12_e10_h7000_1b`
   - mean `val_bpb = 2.1866023565`
   - std `= 0.0051325073`
   - mean steady `tok/s = 374,271`
   - mean artifact estimate `= 4,791,951`

3. `blocks0_resid12_e10_h7000_1b`
   - mean `val_bpb = 2.1899359311`
   - std `= 0.0039254056`
   - mean steady `tok/s = 386,331`
   - mean artifact estimate `= 3,945,168`

Important nuance:

- the two one-block finalists are still effectively tied:
  - `blocks1 12x10 - blocks1 10x12 = +0.0000682172` bpb on mean
- `blocks2_resid12_e8_h7000_1b` remains strategically useful even though it is outside the top three on quality:
  - mean `val_bpb = 2.2005760974`
  - std `= 0.0018741094`
  - mean steady `tok/s = 442,062`

## Locked Schedule Defaults

The post-confirmation retune screen now supports using these as working defaults:

- `lr_hold_steps=3500` for the `4096`-step / `512M` screen
- `lr_hold_steps=7000` for the `8192`-step / `1B` contract

Revalidated representatives:

- `blocks1 10x12`
- `blocks0 12x10`
- `blocks2 12x8`

`h4096` lost on all three, so no-decay is still not the answer.

## Exact Reproduction Commands

### Current frontier confirmation batch

One-block multi-seed family:

```bash
SEEDS="1337 2027 3141" RUN_BLOCKS0=0 RUN_BLOCKS2=0 bash scripts/run_5090_wide_confirm.sh
```

Zero-block multi-seed family:

```bash
SEEDS="1337 2027 3141" RUN_BLOCKS1=0 RUN_BLOCKS2=0 bash scripts/run_5090_wide_confirm.sh
```

Two-block structural control:

```bash
SEEDS="1337 2027 3141" RUN_BLOCKS1=0 RUN_BLOCKS0=0 bash scripts/run_5090_wide_confirm.sh
```

### Next architecture batch

Tokenwise residual-gate screen:

```bash
bash scripts/run_5090_architecture_gate_screen.sh
```

Safe compact `max_lr` probe:

```bash
bash scripts/run_5090_safe_maxlr_probe.sh
```

Temporal screen:

```bash
bash scripts/run_5090_architecture_temporal_screen.sh
```

Router stretch:

```bash
bash scripts/run_5090_architecture_router_screen.sh
```

Finalist `1B` confirmation harness:

```bash
bash scripts/run_5090_finalist_confirm1b.sh
```

## Best Current Calls

Best pure-quality contender:

- `blocks1_resid10_e12_h7000_1b`

Caveat:

- it is only ahead of `blocks1_resid12_e10_h7000_1b` by about `0.00007` bpb on mean, so this is a working pick, not a decisive geometry claim

Best quality/speed tradeoff on the 5090:

- `blocks0_resid12_e10_h7000_1b`

Reason:

- only about `0.00340` bpb behind the best mean point
- about `13k` tok/s faster than the best mean point
- about `845k` fewer artifact bytes than the one-block finalists

Best fast structural control:

- `blocks2_resid12_e8_h7000_1b`

Reason:

- fastest of the confirmed structural reps
- lowest seed variance in the current frontier batch

## Most Likely To Transfer Cleanly To `1x H100`

- `blocks1_resid10_e12_h7000_1b` as the current quality candidate
- `blocks0_resid12_e10_h7000_1b` as the lean control candidate

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom
- compile warmup economics
- local `TORCH_BLAS_PREFER_CUBLASLT=1` interactions
- exact end-to-end impact of `assoc_accel` versus the Heinsen path on this device

## Findings Less Likely To Be 5090-Specific

- full-dataset frozen spec building is mandatory for serious claims
- `blocks1` is a real frontier family
- `blocks0 12x10` is the correct lean control
- the inherited `lr_hold_steps=1500` default was too short
- the tuned late-hold schedule transfers
- naive lag-heavy temporal variants were not good enough

## Code Improvements Vs Pure Hyperparameter Findings

Code improvements:

- structured W&B logging and resolved-config snapshots
- exact environment/runtime capture
- artifact accounting with trainable int8-zlib payload
- gradient checkpointing support for larger local controllers
- fixed effective-step-token contracts with derived local `grad_accum`
- explicit validation-shard enforcement
- scan backend selection for the active Core/Amplifier path:
  - `auto`
  - `heinsen`
  - `assoc`
  - `assoc_accel`
  - `sequential`
- new architectural controls:
  - tokenwise residual gating
  - EMA / EMA-hybrid temporal taps
  - per-token branch routing

Pure hyperparameter / architecture findings:

- the one-block pair remains tied on mean
- `blocks0 12x10` is the best zero-block control
- `blocks2 12x8` is still worth keeping alive as a fast structural control
- `3500 / 7000` is the working late-hold schedule pair

## Unresolved Questions

- does `lr=3.5e-3` hold up on seed `2027` for both the `blocks1` and `blocks0` safe-lane reps?
- does tokenwise residual gating make the controller meaningfully more selective on hard tokens?
- do EMA or EMA-hybrid taps recover useful temporal structure without drifting toward token mixing?
- does per-token branch routing help once real temporal taps exist?
- after the architecture lane settles, does `blocks1 10x12` finally separate cleanly from `blocks1 12x10`?
- if the architecture lane stalls, is the next lever `max_lr` or controller-up scaling?

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- best moves remain recurrent-controller and schedule work over a frozen statistical basis
- no attention was added
- no token-token mixing was added
- the trainable core remains a parallel minGRU stack
