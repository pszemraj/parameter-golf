# 5090 Next Experiments

Last updated: `2026-04-26`

This note is the short working summary. Current protocol details live in:

- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)
- [docs/5090_shape_reassessment.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_shape_reassessment.md)

## Frontier Snapshot

Current best completed `1B` local points by three-seed mean:

| Rank | Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---|---:|---:|---:|---:|
| 1 | `blocks0_resid12_e10_trigramk2_lr0035_h7000_1b` | `2.0415615686` | `0.0018559894` | `574,798` | `7,331,969` |
| 2 | `blocks0_resid12_e10_lr0035_final_h7000_1b` | `2.1757877509` | `0.0029076698` | `576,426` | `4,007,243` |
| 3 | `blocks1_resid10_e12_lr0035_final_h7000_1b` | `2.1765101841` | `0.0061762455` | `552,843` | `4,853,073` |

Interpretation:

- `blocks0 12x10 + trigram top-2` is now the local mean leader by a large
  margin.
- `blocks1 10x12` is still close enough to keep as the quality counterweight.
- The top-2 trigram confirmation improved over the previous blocks0 finalist
  mean by about `0.1342` bpb with full final validation coverage.
- Artifact estimate is still only about `7.33 MB`, so the remaining
  non-transformer budget should go into top-K/context-memory headroom rather
  than schedule microtuning.

## 5090 Performance Read

The current quality leader is not slow because of the trigram memory. Local
CUDA microbenchmarks at the maintained `B=256`, `T=512` contract show the
dominant cost is the recurrent controller geometry:

| Path | Shape | Synthetic full tok/s | Peak MiB | Trainable params |
|---|---|---:|---:|---:|
| current leader | `core_dim=48`, `layers=12`, `exp=10` | `586,729` | `22,726.5` | `839,130` |
| aligned probe | `core_dim=64`, `layers=8`, `exp=8` | `802,372` | `17,042.7` | `796,182` |
| aligned probe | `core_dim=128`, `layers=4`, `exp=4` | `1,371,917` | `10,332.0` | `806,418` |
| aligned probe | `core_dim=128`, `layers=6`, `exp=4` | `964,576` | `14,245.8` | `1,200,916` |

Core-only measurements tell the same story: `core_dim=128`, `layers=4`,
`exp=4` is about `2.6x` faster per token than the current `48x12x10`
controller at similar trainable parameter count.

Interpretation:

- the current `48`/`480` shape is CUDA-unfriendly and memory-heavy
- `torch.compile` is not a quick escape hatch because the current
  `accelerated-scan` Triton path fails under Inductor tracing
- this is a controlled GPU-friendly geometry probe, not an arbitrary
  transformer-style pivot
- the next metric read should test whether the faster aligned controller keeps
  or improves fixed-token `val_bpb`

The detailed reassessment and rerun matrix now live in:

- [docs/5090_shape_reassessment.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_shape_reassessment.md)

## Locked Schedule Defaults

The post-confirmation hold-retune screen is complete.

| Representative | `h2500` mean | `h3500` mean | `h4096` mean | Winner |
|---|---:|---:|---:|---|
| `blocks1 10x12` | `2.2724088059` | `2.2680191476` | `2.2892407482` | `h3500` |
| `blocks0 12x10` | `2.2732337556` | `2.2716177172` | `2.2823704208` | `h3500` |
| `blocks2 12x8` | `2.2821037577` | `2.2801177077` | `2.2941590571` | `h3500` |

Working defaults now:

- `lr_hold_steps=3500` for the `4096`-step / `512M` screen
- `lr_hold_steps=7000` for the `8192`-step / `1B` confirmation contract

`h4096` lost on all three reps, so “no decay” is not the answer.

## Training Budget Policy

- Full-dataset frozen spec/statistics build is always required:
  - all `195` train shards
  - `19,473,201,340` train tokens
  - no validation tokens
- The explicit validation shard is separate:
  - `1` shard
  - `62,021,846` tokens
  - used only for eval/full-val evidence, not frozen statistics
- Validate local shard coverage before long cached builds:
  - `conda run -s --name train python tools/check_dataset_shards.py data/datasets/fineweb10B_sp1024 --expected-train-files 195 --expected-val-files 1`
- Trigram memory specs are cached outside the repo under
  `${TRIGRAM_MEMORY_SPEC_CACHE_ROOT:-~/.cache/experiments/param-golf-coreamp}` and are
  keyed by the source spec hash, the training-shard fingerprint, and memory
  parameters, so compatible ablations reuse the same full-data build.
- `512M` is the default serious screening budget.
- `1B` is the confirmation budget for finalists.
- Anything shorter than `512M` is only for smoke tests or harness checks.

## Seed Policy

Seeds are not a search axis. They only protect us from mistaking a lucky or
unlucky controller initialization/order for an architecture result.

Current policy:

- default screens use one canonical seed: `1337`
- rerun another seed only when a screen is close to a promotion threshold
- use `SEEDS="1337 2027 3141"` only for final evidence or when a result is
  likely to become the submission candidate
- do not pick winners by best seed

The top-2 trigram confirmation already showed low seed variation
(`std=0.0018559894`) relative to the architecture gain (`~0.1342` bpb), so the
remaining top-K/headroom probes should stay single-seed unless a result is
being packaged as final evidence.

## Final-Week Execution Read

The original final-week sequence has run far enough to close its main loop. The
safe-LR, gate, EMA, router, old finalist-confirmation, schedule-retune,
wide-confirm, and gate/LR follow-up launchers are retired and removed from the
active script surface.

Serious maintained launchers fail before training if a shell override would
change the maintained protocol. Defaults are:

- `WANDB_PROJECT=pg-core-amp`
- `SCAN_BACKEND=auto`
- `TORCH_BLAS_PREFER_CUBLASLT=1`
- `COMPILE=0`
- `GRADIENT_CHECKPOINTING=0`
- `RUN_VERSION=v2`

The deadline-oriented source of truth for ordering, promotion rules, and stop rules is:

- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)

## Current Batch Read

Completed so far in the final-week lane:

- safe `max_lr` screen:
  - `blocks1 10x12`: `lr=3.5e-3` beat `3.0e-3` by about `0.01303` bpb on seed `1337`
  - `blocks0 12x10`: `lr=3.5e-3` beat `3.0e-3` by about `0.01273` bpb on seed `1337`
  - decision:
    - promote `lr=3.5e-3` for both reps to seed `2027`

- gate screen:
  - `blocks1 10x12`:
    - `base` improved mean `val_bpb` by about `0.00241`
    - throughput loss was about `3.7%`
    - decision:
      - does **not** clear the `>= 0.003` promotion bar
      - treat gating as flat on the primary quality rep for now
  - `blocks0 12x10`:
    - `base` improved mean `val_bpb` by about `0.00582`
    - throughput loss was about `3.9%`
    - decision:
      - `gate=base` clears the promotion bar on the lean control
  - `core_base` did not clear the bar on either rep

- temporal screen:
  - primary lane ran on `blocks1 10x12` with `gate=none`
  - `ema` was worse than the matching `current` baseline by about `0.00457` bpb on seed `1337`
  - `ema_hybrid` was worse by about `0.01230` bpb on seed `1337`
  - decision:
    - EMA lane is flat/negative
    - skip router
    - move straight to queued `1B` finalist confirmations

- cleaned `v2` finalist confirmations:
  - `blocks0_resid12_e10_lr0035_final`: mean `2.1757877509`
  - `blocks1_resid10_e12_lr0035_final`: mean `2.1765101841`
  - both completed under exact `val_bpb`, explicit validation shards, and
    `scan_backend=assoc_accel`
  - decision:
    - keep both as current finalists
    - treat the ranking as close enough that new architecture probes should
      screen both, not only the current mean leader

- gate/LR follow-up:
  - completed `blocks1 gate=base lr=3.5e-3` seed `1337`
  - result: `2.2575790952`, slower than the matching no-gate safe-lane point
  - decision:
    - stop this lane
    - do not spend more queue time on the existing tokenwise gate formulation

## Plan Delta From The No-Fallback Audit

The pre-pivot batch order is now closed. What remains from that work is the
acceptance contract for serious runs:

- serious runs now count only if they keep exact `val_bpb`
- serious CUDA runs now count only if `scan_backend=auto` resolves to `assoc_accel`
- directory-style FineWeb runs still require the explicit validation shard
- spectral basis builds no longer silently degrade to `svd`

Practical consequence:

- any run that only “works” by slipping onto approximate `bpb`, a slower scan
  backend, or another degraded path should be treated as invalid and rerun
- explicit smoke/debug opt-ins still exist, but they are no longer ambiguous
  with maintained-path experiment results

## Immediate Next Commands

The `geom1` adaptive closeout completed. Current read:

| Run | Contract | `val_bpb` | Steady tok/s | Artifact |
|---|---:|---:|---:|---:|
| `d96_l6_i512` K2 | `512M` screen | `2.0668155804` | `990,977` | `7,856,990` |
| `d64_l10_i512` K2 | `512M` screen | `2.0677617650` | `645,010` | `7,954,836` |
| `d128_l4_i512` K2 | `512M` screen | `2.0681435993` | `1,372,453` | `8,568,493` |
| `d128_l5_i512` K2 | `512M` screen | `2.0563568016` | `1,128,480` | `8,806,358` |
| `d96_l6_i512` K2 | `1B` full-val | `2.0264627708` | `1,006,543` | `7,942,889` |
| `d128_l5_i512` K2 | `1B` full-val | `2.0031207874` | `1,137,730` | `8,830,483` |
| `d128_l5_i512` K2 BPTT2 | `512M` screen | `2.0560344760` | `1,140,618` | `8,805,457` |
| `d128_l5_i512` K4 BPTT2 | `512M` screen | `2.0155952297` | `1,140,404` | `11,281,814` |
| `d128_l5_i512` K4 seq2048 | `1B` full-val | `1.9731361526` | `1,182,049` | `11,371,671` |
| `d128_l5_i512` K4 seq2048 BPTT2 | `1B` full-val | `1.9722313128` | `1,177,934` | `11,405,945` |
| `d128_l5_i512` K6 seq2048 BPTT2 | preflight only |  |  | `14,520,129` |

Interpretation:

- `d128_l5_i512` is the current geometry winner. It beats `d96_l6_i512` by
  about `0.0233` bpb at K2 `1B` while running faster.
- `BPTT2` is not established. The K2 BPTT2 gain over K2 BPTT1 at `512M` is
  only `0.0003` bpb, which is noise for this screen.
- `K4` is established as a real signal. The current best local result is
  `d128_l5_i512` K4 seq2048 BPTT2 at `1.9722313128` full-val BPB, under the
  artifact cap with about `4.59 MB` headroom.
- `seq4096` underperformed (`2.0467157896` at `512M`), so do not expand context
  further by default.
- `K6` has only passed artifact preflight. It did not train because the old
  launcher/sweep rebuild boundary rejected the explicit shared spec. The K6
  cache is valid and should be reused.

Current trigram memory is a sparse additive boost into base logits with one
learned global scale. The recurrent controller does not yet receive trigram
hit/confidence/margin features and does not arbitrate when to trust the memory.

Next run:

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

Replay `blocks1` only as a geometry check after the blocks0 top-K and aligned
geometry decisions:

```bash
bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
  --run-version blocks1_check \
  --seeds 1337 \
  --geometry-label blocks1_d64_l10_i512 \
  --geometry-core-dim 64 \
  --geometry-core-layers 10 \
  --geometry-core-inner-dim 512 \
  --geometry-num-blocks 1
```

Use diagnostics on completed or partial runs before recovering any secondary
adapter probe from history:

```bash
conda run -s --name train python tools/analyze_core_amp_run.py /path/to/run_dir --checkpoint /path/to/run_dir/final.pt --steps 64 --batch-size 64 --device cuda
```

The old `gate x lr` follow-up is removed. The base-bigram delta and
readout-delta launcher scripts are also removed from the active script surface;
recover them from git only if diagnostics make them clearly relevant.

## Why The Architecture Lane Changed

The schedule question is no longer the biggest uncertainty. The stronger thesis now is:

- frozen statistics should absorb easy tokens
- the recurrent controller should intervene selectively on hard tokens
- temporal structure should come from real causal multi-timescale taps, not just alternate projections of the current state

That means the next architecture order is now:

1. run the reviewer-aligned geometry frontier batch and stop after Stage 1
2. confirm only geometry rows that clear the fixed-token plus speed promotion
   rules
3. test longer BPTT on the best aligned survivor only
4. test top-K headroom (`K=4`, then only consider `K=8` if artifact estimate
   remains comfortably under cap)
5. optional score-first adaptive n-gram cache only after the static memory is
   validated and compliance is documented

Current practical interpretation:

- EMA and EMA-hybrid did not survive on the primary `blocks1` lane
- router stays skipped
- the existing gate cross-term did not survive at `lr=3.5e-3`
- current lag operators are not evidence against temporal structure because
  they do not expose literal high-order context identity
- top-2 trigram memory is now the strongest non-transformer use of artifact
  budget, but controller geometry is the immediate bottleneck to resolve before
  spending more runs on K4/K8

## Fast-Scan Note

`scan_backend=auto` is now wired through the active Core/Amplifier path as the fail-loud default.

- on CUDA it requires `accelerated-scan`, then resolves to `assoc_accel`
- on non-CUDA devices it resolves to the repo-local Torch `assoc` path
- it no longer silently downgrades to a slower backend on the maintained CUDA path

Local 5090 microbenchmarks on representative recurrence shapes:

- minGRU layer (`B=256`, `T=512`, `dim=48`, `expansion=12`)
  - `heinsen`: `30.893 ms`
  - `assoc_accel`: `21.041 ms`
- EMA-style branch recurrence (`B=64`, `T=512`, `12 x 48`)
  - `sequential`: `95.314 ms`
  - `assoc_accel`: `2.844 ms`

Those are local primitive-level checks, not end-to-end training claims, but they are strong enough to keep `scan_backend=auto` as the working default for the new architecture lane, with `accelerated-scan` treated as core CUDA infrastructure and the slower `assoc` path owned in-repo.
