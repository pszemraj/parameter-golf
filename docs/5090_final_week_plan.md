# 5090 Final Week Plan

Last updated: `2026-04-26`

This is the deadline-focused execution note for the final week on the local RTX 5090.

- Short working status: [docs/5090_next_experiments.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_next_experiments.md)
- Shape rationale: [docs/5090_shape_reassessment.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_shape_reassessment.md)

## Goal

Use the remaining week to maximize the chance of ending with a stronger repo-root candidate for a credible non-record submission.

This is a dual-track week:

- safe lane:
  - keep one compact optimization lane alive on the current incumbent family
- aggressive lane:
  - test the highest-upside thesis-aligned architecture ideas that are already implemented

Out of scope for this note:

- record-folder packaging
- PR assembly
- broader speculative architecture branches

## Locked Context

Current confirmed frontier by three-seed mean:

| Rank | Run | Mean `val_bpb` | Mean steady tok/s |
|---|---|---:|---:|
| 1 | `blocks0_resid12_e10_trigramk2_lr0035_h7000_1b` | `2.0415615686` | `574,798` |
| 2 | `blocks0_resid12_e10_lr0035_final_h7000_1b` | `2.1757877509` | `576,426` |
| 3 | `blocks1_resid10_e12_lr0035_final_h7000_1b` | `2.1765101841` | `552,843` |

Locked schedule defaults:

- `512M` screens use `lr_hold_steps=3500`
- `1B` confirmations use `lr_hold_steps=7000`

Fixed screening contract:

- full-dataset frozen shared spec only
- `4096` steps
- `seq_len=512`
- `TARGET_EFFECTIVE_STEP_TOKENS=131072`
- `carry_chunks=8`
- `bptt_chunks=1`
- `warmup_steps=100`
- `min_lr=3e-4`
- `weight_decay=1e-3`
- `COMPILE=0`
- `GRADIENT_CHECKPOINTING=0`
- `scan_backend=auto`
- `TORCH_BLAS_PREFER_CUBLASLT=1`
- W&B project:
  - `pg-core-amp`
- output version:
  - `RUN_VERSION=v2` for the cleaned post-audit rerun

Frozen spec/statistics coverage:

- train shards: `195`
- train tokens used for specs/trigram memories: `19,473,201,340`
- validation shards: `1`
- validation tokens: `62,021,846`
- validation tokens are not used for frozen statistics
- quick coverage check:
  - `conda run -s --name train python tools/check_dataset_shards.py data/datasets/fineweb10B_sp1024 --expected-train-files 195 --expected-val-files 1`

Trigram memory specs are cached under
`${TRIGRAM_MEMORY_SPEC_CACHE_ROOT:-~/.cache/experiments/param-golf-coreamp}` and keyed
by source `spec.pt` hash, training-shard fingerprint, and memory parameters.
Compatible ablations reuse the same full-data top-K build instead of
rebuilding inside each experiment root.

Primary reps:

- quality baseline:
  - `blocks1_resid10_e12`
- lean control:
  - `blocks0_resid12_e10`
- fast structural control:
  - `blocks2_resid12_e8`

Current interpretation:

- the tuned safe-lane finalists are now the incumbent local choices
- the `gate=base x lr=3.5e-3` follow-up lane was negative on `blocks1` seed `1337`
  and should be stopped
- both finalists leave more than 11 MB of artifact budget unused, so the next
  architecture work should spend budget on thesis-aligned non-transformer
  capacity rather than replaying the same gate/temporal formulations

Performance interruption:

- the local 5090 bottleneck is the recurrent controller, not data loading or
  the trigram memory
- synthetic full-step benchmarks at `B=256`, `T=512`, top-2 memory:
  - current `core_dim=48`, `layers=12`, `exp=10`: about `586,729` tok/s
  - aligned `core_dim=64`, `layers=8`, `exp=8`: about `802,372` tok/s
  - aligned `core_dim=128`, `layers=4`, `exp=4`: about `1,371,917` tok/s
- the aligned `128x4x4` controller has similar trainable parameter count to
  the current leader and still fits the artifact budget, but it is a
  speed-frontier probe rather than the aligned answer
- `torch.compile` currently fails on the `accelerated-scan` Triton path, so
  compile remains out of the serious local protocol
- this makes one bounded GPU-friendly geometry probe in-scope before more
  top-K headroom, despite the earlier stop rule against arbitrary `core_dim`
  churn
- the independent shape review changes the matrix from a shallow/wide probe
  into a real frontier over recurrent memory cells:
  - balanced: `96 x 6 x inner512`
  - memory-preserving: `64 x 10 x inner512`
  - speed frontier: `128 x 4/5 x inner512`
- detailed shape rationale and rerun priorities:
  - [docs/5090_shape_reassessment.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_shape_reassessment.md)

## Protocol Invariants

The final-week plan is now explicitly fail-loud on maintained competition-path choices.

Runs only count toward screening or confirmation if all of the following are true:

- `validation_source=explicit_val_shard` for directory-style FineWeb runs
- exact `val_bpb` is active
  - no implicit approximate-`bpb` fallback
- eval rows expose `eval_tokens`, `eval_bytes`, and `eval_coverage_frac`
  - small `VAL_STEPS=8` results are screen metrics, not full validation claims
- `scan_backend=auto` on CUDA resolves to `assoc_accel`
  - no silent scan-backend downgrade
- the shared spec matches the intended structure and embedding-init contract
  - spectral basis build failure is not treated as “close enough” to `svd`
- serious 5090 launchers fail before training on protocol drift:
  - `SCAN_BACKEND` must be `auto`
  - `TORCH_BLAS_PREFER_CUBLASLT` must be `1`
  - `COMPILE` and `GRADIENT_CHECKPOINTING` must stay `0`
  - `SPEC_MAX_TOKENS` and `DATA_MAX_TOKENS` must be unset
  - W&B must be online in `pg-core-amp`

Explicit low-quality or convenience modes remain allowed only for local smoke or debugging:

- `--allow-train-frac-val-split`
- `--allow-approx-bpb`
- `--scan-backend heinsen|assoc|sequential`

If a serious run hits one of those modes unintentionally, the run should be treated as invalid and rerun under the maintained contract.

## Active Batch Order

The pre-pivot safe-LR, token gate, EMA, router, old finalist-confirmation,
schedule-retune, wide-confirm, and gate/LR follow-up launchers have completed
or been stopped under the rules above. They have been removed from the active
script surface so new runs do not accidentally replay obsolete lanes.

### 1. Dense trigram memory

Script:

```bash
bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
  --run-version geom1 \
  --seeds 1337 \
  --geometry-label blocks0_d128_l5_i512 \
  --geometry-core-dim 128 \
  --geometry-core-layers 5 \
  --geometry-core-inner-dim 512 \
  --trigram-top-k 2
```

Default contract:

- reps:
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
- `4096` steps / `512M` planned tokens
- `learning_rate=3.5e-3`
- frozen shared specs from the cleaned finalist families
- trigram memory:
  - `trigram_memory=frozen`
  - `TRIGRAM_TOP_K=2`
  - `(x[t-1], x[t]) -> top-K residual logits over the frozen bigram base`
  - current K2 memory is a sparse additive boost into base logits with one
    learned global scale; controller features/gates/routers do not arbitrate the
    boost yet
- `base_bigram_delta=none`
- `residual_readout_delta_rank=0`
- all serious-run protocol invariants still apply

Why this is in-scope:

- it spends unused artifact bytes on literal high-order context identity
- it stays non-transformer: no attention, no learned token-token mixing
- it directly tests whether the current branch is missing memory rather than
  controller capacity
- the previous token is part of recurrent chunk state, so memory lookup is
  causal and consistent between full forward, chunked training, and `step()`

Promotion rule:

- compare to the matching no-memory `512M` safe-lane point
- `>= 0.05` bpb gain on seed `1337`: continue immediately
- `< 0.02` bpb gain: stop the Core/Amplifier leaderboard-record attempt and
  package as non-record research unless diagnostics show an obvious bug
- between those: run diagnostics and a full-val sanity check before replaying
- replay `blocks1` only after `blocks0` shows a real signal

Current read:

- `blocks0_resid12_e10_trigramk2_lr0035_h3500_512m_s1337`: `2.0751715673`
- matching no-memory `blocks0_resid12_e10_lr0035_h3500_512m_s1337`:
  `2.2529073228`
- sampled gain: about `0.1777` bpb
- final-checkpoint diagnostic on `2.1M` validation tokens:
  - full gain over frozen bigram base: about `0.4171` bpb
  - disabling only the trigram memory costs about `0.2648` bpb
- artifact estimate: `7,333,039` bytes, leaving `8,666,961` bytes headroom

The continuation bar is cleared. Use the aligned geometry launcher for
confirmation:

```bash
bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
  --run-version geom1_confirm \
  --seeds 1337 \
  --geometry-label blocks0_d128_l5_i512 \
  --geometry-core-dim 128 \
  --geometry-core-layers 5 \
  --geometry-core-inner-dim 512 \
  --num-steps 8192 \
  --lr-hold-steps 7000 \
  --full-val-final
```

Default confirmation contract:

- reps:
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
- `8192` steps / `1B` planned tokens
- `lr_hold_steps=7000`
- `FULL_VAL_FINAL=1`
- trigram memory defaults remain `TRIGRAM_TOP_K=2`, `trigram_memory=frozen`

Seed policy:

- seeds are not a tuning axis
- use seed `1337` for normal screens and first confirmations
- add `2027` / `3141` only for final evidence or when a result is near a
  threshold
- the completed top-2 trigram three-seed confirmation already showed low seed
  variation relative to effect size, so remaining headroom probes should not
  run all seeds by default

Next headroom test after top-2 confirmation:

Before increasing top-K, run the reviewer-aligned geometry frontier. The
adaptive closeout runner is the preferred command because it selects follow-ups from
completed summaries rather than running a permutation grid:

```bash
bash scripts/run_5090_adaptive_closeout.sh \
  --dry-run \
  --frontier-batch-id geom1 \
  --run-version geom1 \
  --seed 1337 \
  --no-run-benchmark \
  --count-workers 2 \
  --max-confirmations 2 \
  --stop-after k4

bash scripts/run_5090_adaptive_closeout.sh \
  --frontier-batch-id geom1 \
  --run-version geom1 \
  --seed 1337 \
  --no-run-benchmark \
  --count-workers 2 \
  --max-confirmations 2 \
  --stop-after k4
```

Manual staged equivalent:

```bash
bash scripts/run_5090_final3day_frontier_batch.sh --dry-run --run-version geom1 --seeds 1337
bash scripts/run_5090_final3day_frontier_batch.sh --run-version geom1 --seeds 1337
```

Promotion rule:

- compare against the current top-2 seed-`1337` `512M` screen
  (`2.0751715673`)
- promote an aligned shape if it is better at fixed tokens
- if within `0.020` bpb and at least `1.5x` faster, run an `8192`-step
  time-matched confirmation before killing
- if within `0.035` bpb and at least `2.0x` faster, run one time-matched
  confirmation before killing
- if worse by `>0.040` bpb, kill unless the curve slope is clearly unfinished
- if all aligned shapes die, keep current `48x12x10` as the quality leader and
  run top-K headroom through the aligned launcher or adaptive closeout:

```bash
bash scripts/run_5090_adaptive_closeout.sh \
  --frontier-batch-id geom1 \
  --run-version geom1 \
  --seed 1337 \
  --no-run-benchmark \
  --count-workers 2 \
  --stop-after k4
```

Only consider `K=8` if `K=4` improves and the measured artifact estimate still
leaves enough submission bytes for code and trainable payload.

Confirmation read:

| Seed | Full-val `val_bpb` | Steady tok/s | Artifact bytes | W&B |
|---:|---:|---:|---:|---|
| `1337` | `2.0394517881` | `574,571` | `7,331,661` | `d6r3b7uc` |
| `2027` | `2.0429425206` | `574,798` | `7,332,800` | `9thh1ep8` |
| `3141` | `2.0422903971` | `575,025` | `7,331,445` | `mue7m0o2` |

Mean `val_bpb=2.0415615686`, std `0.0018559894`. This improves over the
previous cleaned blocks0 finalist mean `2.1757877509` by about `0.1342` bpb.
All three final evals used exact BPB and full validation coverage. Learned
trigram boost scale converged tightly around `1.17x`, so log-scale-init tuning
is lower priority than top-K headroom.

Aligned-geometry closeout read:

| Run | Contract | `val_bpb` | Steady tok/s | Artifact bytes |
|---|---:|---:|---:|---:|
| `d96_l6_i512` K2 | `512M` | `2.0668155804` | `990,977` | `7,856,990` |
| `d64_l10_i512` K2 | `512M` | `2.0677617650` | `645,010` | `7,954,836` |
| `d128_l4_i512` K2 | `512M` | `2.0681435993` | `1,372,453` | `8,568,493` |
| `d128_l5_i512` K2 | `512M` | `2.0563568016` | `1,128,480` | `8,806,358` |
| `d96_l6_i512` K2 | `1B` full-val | `2.0264627708` | `1,006,543` | `7,942,889` |
| `d128_l5_i512` K2 | `1B` full-val | `2.0031207874` | `1,137,730` | `8,830,483` |
| `d128_l5_i512` K2 BPTT2 | `512M` | `2.0560344760` | `1,140,618` | `8,805,457` |
| `d128_l5_i512` K4 BPTT2 | `512M` | `2.0155952297` | `1,140,404` | `11,281,814` |
| `d128_l5_i512` K4 seq2048 | `1B` full-val | `1.9731361526` | `1,182,049` | `11,371,671` |
| `d128_l5_i512` K4 seq2048 BPTT2 | `1B` full-val | `1.9722313128` | `1,177,934` | `11,405,945` |
| `d128_l5_i512` K6 seq2048 BPTT2 | preflight only |  |  | `14,520,129` |

Current interpretation:

- `d128_l5_i512` is the aligned geometry winner and K4 seq2048 BPTT2 is the
  current local finalist.
- BPTT2 is not independently established; the K2 gain is only `0.0003` bpb at
  `512M`.
- K4 is established: K4 seq2048 BPTT2 beats K2 `1B` by about `0.0309` bpb and
  still leaves about `4.59 MB` artifact headroom.
- K6 has only passed artifact preflight. It did not train because the old
  launcher/sweep rebuild boundary rejected the explicit shared spec. Reuse the
  valid K6 cache after the boundary fix.

Top-K headroom confirmation:

```bash
bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
  --run-version geom1_seq2048_bptt2_k6 \
  --seeds 1337 \
  --geometry-label blocks0_d128_l5_i512 \
  --geometry-core-dim 128 \
  --geometry-core-layers 5 \
  --geometry-core-inner-dim 512 \
  --geometry-batch-size 32 \
  --geometry-seq-len 2048 \
  --geometry-bptt-chunks 2 \
  --num-steps 8192 \
  --lr-hold-steps 7000 \
  --geometry-train-label 1b_seq2048_bptt2_k6 \
  --trigram-top-k 6 \
  --count-workers 4 \
  --full-val-final
```

Promotion rule:

- compare K6 to the K4 seq2048 BPTT2 `1B` result (`1.9722313128`)
- if K6 improves by at least `0.004` bpb and remains under the artifact cap,
  promote K6 to final evidence
- if K6 is flat within noise, keep K4 and stop top-K expansion
- if K6 regresses or artifact headroom becomes too tight, keep K4 and move to
  controller/trigram arbitration features instead of K8

### 2. Optional adapter probes

Base-bigram delta and residual readout delta remain in the model/trainer code,
but their dedicated launchers are removed from the active script surface. They
can be recovered from git if diagnostics specifically show calibration or
frozen-readout capacity is the bottleneck. Do not run them by default during the
geometry/top-K closeout.

Diagnostic command for completed or partial runs:

```bash
conda run -s --name train python tools/analyze_core_amp_run.py /path/to/run_dir --steps 16
```

## Stop Rules

Stop adding complexity and keep the best confirmed trigram-memory candidate if
any of the following happen:

- every aligned geometry row is worse by `>0.040` bpb at fixed tokens and has
  no compensating wallclock argument
- `K=4` is flat or worse than the matching top-2 screen
- diagnostics show no hard-token-bucket improvement after the best
  trigram-memory geometry/top-K run

If the whole aggressive lane stalls, the remaining budget should go to:

- target-hardware verification of the current finalists
- one carefully bounded frozen-structure probe with GPU-friendly dimensions

Do not spend the final week on:

- more frozen blocks under the already-tested `core_dim=48` setup
- arbitrary `core_dim` changes that create unfriendly CUDA shapes
- open-ended geometry sweeps beyond the bounded matrix in
  [docs/5090_shape_reassessment.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_shape_reassessment.md)
  unless one row produces a real metric read
- larger controller ladders unless a new mechanism wins decisively
- attention-like ideas
- token-token mixing
- submission packaging

## Final Reserve

Reserve the last `1` to `2` days for target-hardware verification if H100 access is available.

Use that reserve for:

- current incumbent
- best new local winner, if there is one

Do not start a new model-family branch once the reserve window begins.
