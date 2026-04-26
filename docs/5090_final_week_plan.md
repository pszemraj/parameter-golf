# 5090 Final Week Plan

Last updated: `2026-04-24`

This is the deadline-focused execution note for the final week on the local RTX 5090. It complements the architecture note instead of replacing it.

- Full architecture rationale: [docs/5090_architecture_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_architecture_plan.md)
- Short working status: [docs/5090_next_experiments.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_next_experiments.md)

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

Current confirmed frontier by three-seed mean after the cleaned `v2`
confirmation batch:

| Rank | Run | Mean `val_bpb` | Mean steady tok/s |
|---|---|---:|---:|
| 1 | `blocks0_resid12_e10_lr0035_final_h7000_1b` | `2.1757877509` | `576,426` |
| 2 | `blocks1_resid10_e12_lr0035_final_h7000_1b` | `2.1765101841` | `552,843` |

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
  - `pg-hconv-ablations`
- output version:
  - `RUN_VERSION=v2` for the cleaned post-audit rerun

Primary reps:

- quality baseline:
  - `blocks1_resid10_e12`
- lean control:
  - `blocks0_resid12_e10`
- fast structural control:
  - `blocks2_resid12_e8`

Current interpretation:

- the tuned safe-lane finalists are now the incumbent local choices
- the `gate=base x lr=3.5e-3` sidecar was negative on `blocks1` seed `1337`
  and should be stopped
- both finalists leave more than 11 MB of artifact budget unused, so the next
  architecture work should spend budget on thesis-aligned non-transformer
  capacity rather than replaying the same gate/temporal formulations

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
  - W&B must be online in `pg-hconv-ablations`

Explicit low-quality or convenience modes remain allowed only for local smoke or debugging:

- `--allow-train-frac-val-split`
- `--allow-approx-bpb`
- `--scan-backend heinsen|assoc|sequential`

If a serious run hits one of those modes unintentionally, the run should be treated as invalid and rerun under the maintained contract.

## Batch Order

### 1. Safe lane: compact `max_lr` probe

Script:

```bash
bash scripts/run_5090_safe_maxlr_probe.sh
```

Default contract:

- reps:
  - `blocks1_resid10_e12`
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
- `max_lr in {2.5e-3, 3.0e-3, 3.5e-3}`

Promotion rule:

- promote at most one LR per rep
- require `>= 0.003` bpb gain over that rep’s current single-seed incumbent, or a tie within `0.002` with clearly cleaner curve behavior
- confirm promoted LR on seed `2027`
- invalidate and rerun any point that does not satisfy the protocol invariants above

Current status:

- completed screening on seed `1337`
- `lr=3.5e-3` was promoted for:
  - `blocks1_resid10_e12`
  - `blocks0_resid12_e10`

### 2. Aggressive lane: tokenwise residual gating

Script:

```bash
bash scripts/run_5090_architecture_gate_screen.sh
```

Default contract:

- reps:
  - `blocks1_resid10_e12`
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
  - `2027`
- `residual_token_gate_mode in {none, base, core_base}`
- fixed:
  - `branch_temporal_mode=current`
  - `branch_router_mode=none`

Promotion rule:

- compare against `gate=none`
- promote at most one non-`none` gate mode per rep
- require `>= 0.003` bpb gain
- allow at most `7%` throughput loss
- invalidate and rerun any point that does not satisfy the protocol invariants above

Current status:

- completed on seeds `1337` and `2027`
- `blocks1_resid10_e12`:
  - no gate mode cleared the promotion bar
  - use `gate=none` for the primary temporal lane
- `blocks0_resid12_e10`:
  - `gate=base` cleared the promotion bar
  - keep it alive as a sidecar aggressive-lane result

### 3. Aggressive lane: EMA temporal taps

Script:

```bash
bash scripts/run_5090_architecture_temporal_screen.sh
```

Default contract:

- primary rep:
  - `blocks1_resid10_e12`
- seeds:
  - `1337`
- default launcher modes:
  - `ema`
  - `ema_hybrid`
- gate mode:
  - set `GATE_MODE` to the winner from the gate screen
  - for the current primary lane, that means `GATE_MODE=none`
- default launcher behavior:
  - reuse the prior `current` baseline instead of rerunning it

Promotion rule:

- compare to the matching `current` baseline from the gate batch
- promote only the best non-`current` mode
- require `>= 0.004` bpb gain on the first-pass screen
- confirm the promoted temporal mode on seed `2027`
- invalidate and rerun any point that does not satisfy the protocol invariants above

Replay rule:

- only if the promoted temporal mode survives on `blocks1`
- then replay that single temporal winner on:
  - `blocks0_resid12_e10`
  - `blocks2_resid12_e8`
- start with seed `1337` only for the control replays

Current status:

- completed on seed `1337` for the primary `blocks1` lane
- `ema` was worse than the matching `current` baseline by about `0.00457` bpb
- `ema_hybrid` was worse by about `0.01230` bpb
- this lane is flat/negative under the promotion rules
- do not replay temporal variants on `blocks0` or `blocks2`

### 4. Stretch lane: branch routing

Script:

```bash
bash scripts/run_5090_architecture_router_screen.sh
```

Run this only if the EMA lane yields a real winner.

Default launcher behavior:

- reuse the matching `router=none` temporal baseline
- launch only `router=softmax` unless `INCLUDE_BASELINE_NONE=1`

Promotion rule:

- require `>= 0.0025` bpb gain
- allow at most `10%` throughput loss
- confirm on seed `2027` only if the first pass clears the bar
- invalidate and rerun any point that does not satisfy the protocol invariants above

Current status:

- skipped
- the primary EMA lane did not yield a winner

### 5. Finalist `1B` confirmations

Script:

```bash
bash scripts/run_5090_finalist_confirm1b.sh
```

Default finalists:

- `blocks1_resid10_e12_lr0035_final`
- `blocks0_resid12_e10_lr0035_final`

Default contract:

- seeds:
  - `1337`
  - `2027`
  - `3141`
- `8192` steps
- `1B` planned tokens
- `lr_hold_steps=7000`
- all protocol invariants above still apply
- current queued launcher uses `learning_rate=3.5e-3`
- default output roots use `_v2` so cleaned reruns do not reuse mixed `v1`
  artifacts

Custom finalist format:

```text
name shared_spec_dir core_layers core_expansion gate_mode temporal_mode router_mode learning_rate
```

Example:

```bash
FINALIST_SPECS=$'blocks1_gate_base_ema /abs/path/to/shared 10 12.0 base ema none 0.003\nblocks1_gate_base_ema_router /abs/path/to/shared 10 12.0 base ema softmax 0.003' \
bash scripts/run_5090_finalist_confirm1b.sh
```

Completed queued batch:

```bash
bash scripts/run_5090_post_temporal_queue.sh
```

The cleaned `v2` run completed for seeds `1337 2027 3141`. The optional
`gate x lr` sidecar no longer has promotion value after the completed
`blocks1` seed-`1337` result:

- `blocks1_resid10_e12_gate_base_lr0035_h3500_512m_s1337`: `2.2575790952`
- matching no-gate safe-lane screen was better and faster

### 6. Pivot lane: dense trigram sidecar

Script:

```bash
bash scripts/run_5090_trigram_sidecar_screen.sh
```

Default contract:

- reps:
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
- `4096` steps / `512M` planned tokens
- `learning_rate=3.5e-3`
- frozen shared specs from the cleaned finalist families
- sidecar:
  - `trigram_sidecar=frozen`
  - `TRIGRAM_TOP_K=2`
  - `(x[t-1], x[t]) -> top-K residual logits over the frozen bigram base`
- `base_bigram_delta=none`
- `residual_readout_delta_rank=0`
- all serious-run protocol invariants still apply

Why this is in-scope:

- it spends unused artifact bytes on literal high-order context identity
- it stays non-transformer: no attention, no learned token-token mixing
- it directly tests whether the current branch is missing memory rather than
  controller capacity
- the previous token is part of recurrent chunk state, so sidecar lookup is
  causal and consistent between full forward, chunked training, and `step()`

Promotion rule:

- compare to the matching no-sidecar `512M` safe-lane point
- `>= 0.05` bpb gain on seed `1337`: continue immediately
- `< 0.02` bpb gain: stop the Core/Amplifier leaderboard-record attempt and
  package as non-record research unless diagnostics show an obvious bug
- between those: run diagnostics and a full-val sanity check before replaying
- replay `blocks1` only after `blocks0` shows a real signal

Current read:

- `blocks0_resid12_e10_trigramk2_lr0035_h3500_512m_s1337`: `2.0751715673`
- matching no-sidecar `blocks0_resid12_e10_lr0035_h3500_512m_s1337`:
  `2.2529073228`
- sampled gain: about `0.1777` bpb
- final-checkpoint diagnostic on `2.1M` validation tokens:
  - full gain over frozen bigram base: about `0.4171` bpb
  - disabling only the trigram sidecar costs about `0.2648` bpb
- artifact estimate: `7,333,039` bytes, leaving `8,666,961` bytes headroom

The continuation bar is cleared. Use the dedicated confirmation launcher:

```bash
bash scripts/run_5090_trigram_confirm1b.sh
```

Default confirmation contract:

- reps:
  - `blocks0_resid12_e10`
- seeds:
  - `1337 2027 3141`
- `8192` steps / `1B` planned tokens
- `lr_hold_steps=7000`
- `FULL_VAL_FINAL=1`
- sidecar defaults remain `TRIGRAM_TOP_K=2`, `trigram_sidecar=frozen`

Next headroom test after top-2 confirmation:

```bash
RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_sidecar_screen.sh
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

Next run:

```bash
RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_sidecar_screen.sh
```

Promotion rule:

- compare to the top-2 seed-`1337` `512M` screen (`2.0751715673`)
- if `K=4` improves by at least `0.015` bpb and artifact estimate stays under
  roughly `12 MB`, confirm `K=4` at `1B`
- if flat or worse, keep top-2 and stop architecture changes unless the
  remaining time can absorb a blocks1 geometry check

### 7. Secondary lane: base-bigram delta

Script:

```bash
bash scripts/run_5090_base_delta_screen.sh
```

This remains implemented, but it is now secondary. It trains a current-token
bigram correction and does not add literal high-order context identity. Use it
only if the trigram sidecar shows promise and diagnostics suggest calibration
rather than memory is the remaining bottleneck.

### 8. Secondary lane: residual readout delta

Script:

```bash
bash scripts/run_5090_readout_delta_screen.sh
```

Default contract:

- reps:
  - `blocks1_resid10_e12`
  - `blocks0_resid12_e10`
- seeds:
  - `1337`
- ranks:
  - `128`
  - `256`
- `4096` steps / `512M` planned tokens
- `learning_rate=3.5e-3`
- frozen shared specs from the cleaned finalist families
- all serious-run protocol invariants still apply

Use this only after the trigram sidecar result is understood, or when
diagnostics specifically show the frozen residual readout dictionary is the
bottleneck. It is still zero-init and non-transformer, but it adds per-token
matmuls, so it has a higher speed bar than lookup sidecars.

Diagnostic command for completed or partial runs:

```bash
conda run -s --name train python tools/analyze_core_amp_run.py /path/to/run_dir --steps 16
```

## Stop Rules

Stop adding complexity and keep the cleaned finalists if any of the following happen:

- the gate lane is flat
- the EMA lane is flat
- the router lane fails its first-pass bar
- the trigram sidecar gives `< 0.02` bpb on the first serious `blocks0` screen
- the readout-delta lane is flat after diagnostics show no hard-token-bucket
  improvement

If the whole aggressive lane stalls, the remaining budget should go to:

- target-hardware verification of the current finalists
- one carefully bounded frozen-structure probe with GPU-friendly dimensions

Do not spend the final week on:

- more frozen blocks under the already-tested `core_dim=48` setup
- arbitrary `core_dim` changes that create unfriendly CUDA shapes
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
