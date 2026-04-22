# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The corrected full-spec replay, the main `1B` controller confirmations, the tuned-hold `blocks1` follow-up, and the wide multi-seed `blocks0` / `blocks1` / `blocks2` confirmation are complete.
> The next real source of uncertainty is schedule transfer on the post-confirmation representatives, not whether the single-seed frontier was real.

## Current Status

- Logging is clean in W&B project `pg-core-amp`.
- Exact `val_bpb` is active through the official tokenizer path.
- Artifact accounting matches the record-style convention:
  - repo code bytes
  - `gzip(spec.pt)`
  - trainable int8-zlib payload
- The local dataset is complete for this family:
  - `195` train shards
  - about `19.47B` train tokens available to frozen-spec builds
- The regression-to-transformer guardrail is still intact:
  - no attention
  - no token-token mixing
  - winners are still parallel minGRU controllers over a frozen statistical basis
- The wide confirmation changed the earlier single-seed story:
  - the two `blocks1` finalists are effectively tied on mean
  - `blocks0 12x10` is now the best zero-block control
  - `blocks2 12x8` is slower on quality but remains the fastest and most seed-stable structural control
- Go-forward training-budget policy is now explicit:
  - full-dataset frozen spec build always
  - `512M` for serious screening
  - `1B` for confirmations and stronger claims
  - shorter than `512M` only for smoke or harness checks

## Top 3 Current Contenders

These are the three strongest completed local contenders right now by three-seed mean quality.

1. `blocks1_resid10_e12_h7000_1b`
   - mean `val_bpb = 2.1865341393`
   - std `= 0.0052472581`
   - mean steady `tok/s = 372,888`
   - mean artifact estimate `= 4,790,559`
   - why it is in:
     - best mean quality after the wide confirmation
     - slightly smaller artifact than `blocks1 12x10`

2. `blocks1_resid12_e10_h7000_1b`
   - mean `val_bpb = 2.1866023565`
   - std `= 0.0051325073`
   - mean steady `tok/s = 374,271`
   - mean artifact estimate `= 4,791,951`
   - why it is in:
     - effectively tied with the top point
     - keeps the deeper one-block geometry alive

3. `blocks0_resid12_e10_h7000_1b`
   - mean `val_bpb = 2.1899359311`
   - std `= 0.0039254056`
   - mean steady `tok/s = 386,331`
   - mean artifact estimate `= 3,945,168`
   - why it is in:
     - best zero-block control on the confirmed batch
     - meaningfully smaller artifact than the one-block finalists

Important nuance:

- The two one-block finalists are functionally tied:
  - `blocks1 12x10 - blocks1 10x12 = +0.0000682172` bpb on mean
- `blocks0_resid10_e12_h7000_1b` fell to fourth on the multi-seed mean:
  - `2.1947452986`
- `blocks2_resid12_e8_h7000_1b` is not top-3 on quality, but it is still strategically useful:
  - mean `val_bpb = 2.2005760974`
  - std `= 0.0018741094`
  - mean steady `tok/s = 442,062`
- The fast anchor is still `blocks1_resid12_e6_h7000_1b`:
  - `final val_bpb = 2.2132622271`
  - `steady tok/s = 588,014`

## Exact Reproduction Commands

These are the exact family launch commands currently used for the confirmed results. The three-seed means come from seeds `1337`, `2027`, and `3141` on the same contract.

### 1. One-block multi-seed family

```bash
SEEDS="1337 2027 3141" RUN_BLOCKS0=0 RUN_BLOCKS2=0 bash scripts/run_5090_wide_confirm.sh
```

This reproduces:

- `blocks1_resid10_e12_h7000_1b`
- `blocks1_resid12_e10_h7000_1b`

### 2. Zero-block multi-seed family

```bash
SEEDS="1337 2027 3141" RUN_BLOCKS1=0 RUN_BLOCKS2=0 bash scripts/run_5090_wide_confirm.sh
```

This reproduces:

- `blocks0_resid12_e10_h7000_1b`
- `blocks0_resid10_e12_h7000_1b`

### 3. Two-block structural control

```bash
SEEDS="1337 2027 3141" RUN_BLOCKS1=0 RUN_BLOCKS0=0 bash scripts/run_5090_wide_confirm.sh
```

This reproduces:

- `blocks2_resid12_e8_h7000_1b`

## Best Current Calls

Best pure-quality contender:

- `blocks1_resid10_e12_h7000_1b`
- caveat:
  - it is only ahead of `blocks1_resid12_e10_h7000_1b` by about `0.00007` bpb on mean, so this is a working pick, not a decisive geometry claim

Best quality/speed tradeoff on the 5090:

- `blocks0_resid12_e10_h7000_1b`
- reason:
  - only about `0.00340` bpb behind the best mean point
  - about `13k` tok/s faster than the best mean point
  - about `845k` fewer artifact bytes than the one-block finalists

Best fast anchor:

- `blocks1_resid12_e6_h7000_1b`
- reason:
  - still the fastest quality-relevant point we have at `1B`
  - keeps a clearly non-transformer recurrent profile

Most likely to transfer cleanly to `1x H100`:

- `blocks1_resid10_e12_h7000_1b` as the current pure-quality candidate
- `blocks0_resid12_e10_h7000_1b` as the lean control candidate

Why that split:

- the one-block pair is currently best on mean quality
- the zero-block control is smaller, faster, and more stable than the old `blocks0 10x12` story suggested

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom numbers
- compile warmup economics
- local interactions with `TORCH_BLAS_PREFER_CUBLASLT=1`
- part of the `blocks2` speed advantage may be device-specific even if the quality ordering transfers

## Findings Less Likely To Be 5090-Specific

- the capped frozen-spec default was invalid and materially changed conclusions
- extra frozen amplifier depth beyond the first couple blocks is not obviously earning its bytes
- controller-up/spec-down reallocation remains stronger than the old moderate-depth frontier
- the inherited `lr_hold_steps=1500` default was too short
- `carry=8` and `bptt=1` are the clean current defaults
- the first lag-heavy frozen temporal variants lost clearly to `current`

## Code Improvements Vs Pure Hyperparameter Findings

Code improvements:

- W&B logging is now structured cleanly for this family
- exact environment and runtime metadata are saved per run
- artifact accounting includes the trainable int8-zlib payload
- gradient checkpointing is available for larger recurrent controllers
- the sweep harness supports smaller local microbatches plus derived `grad_accum` from a fixed effective-step-token contract
- the rerun flow is restart-safe and summary-safe
- there is now a dedicated next-step launcher for the post-confirmation schedule retune batch:
  - `scripts/run_5090_schedule_retune.sh`

Pure hyperparameter / architecture findings:

- the one-block family is real, but its two top controller geometries are currently tied on mean
- `blocks0 12 x 10.0` is the best zero-block control after three-seed confirmation
- `blocks2 12 x 8.0` is worse on quality but still viable enough to keep alive as a structural control
- `blocks1 12 x 6.0` with tuned hold is still the fast anchor
- naive lag-heavy temporal variants are not winning
- the current working schedule defaults remain `3500` for the `512M` screen and `7000` for the `1B` contract, but they should now be rechecked on the post-confirmation representatives

## Unresolved Questions

- Does the inherited `h3500 / h7000` transfer remain optimal on `blocks1 10x12` and `blocks2 12x8`?
- After rechecking hold, is `max_lr=3e-3` still right for the one-block winner?
- Does one of the two tied `blocks1` geometries separate once schedule is retuned?
- Does `blocks2` stay alive after a fair hold retune, or does it drop back enough to close that branch?
- After schedule retuning, is the next better move controller-up scaling or longer context?

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- best moves are recurrent-controller scaling, tuned late decay, and at most a minimal frozen amplifier
- no attention was added
- no token-token mixing was added
- the trainable core remains a parallel minGRU stack
