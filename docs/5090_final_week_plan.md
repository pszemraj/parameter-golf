# 5090 Final Week Plan

Last updated: `2026-04-23`

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

Current confirmed frontier by three-seed mean:

| Rank | Run | Mean `val_bpb` | Mean steady tok/s |
|---|---|---:|---:|
| 1 | `blocks1_resid10_e12_h7000_1b` | `2.1865341393` | `372,888` |
| 2 | `blocks1_resid12_e10_h7000_1b` | `2.1866023565` | `374,271` |
| 3 | `blocks0_resid12_e10_h7000_1b` | `2.1899359311` | `386,331` |
| 4 | `blocks2_resid12_e8_h7000_1b` | `2.2005760974` | `442,062` |

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

Primary reps:

- quality baseline:
  - `blocks1_resid10_e12`
- lean control:
  - `blocks0_resid12_e10`
- fast structural control:
  - `blocks2_resid12_e8`

## Protocol Invariants

The final-week plan is now explicitly fail-loud on maintained competition-path choices.

Runs only count toward screening or confirmation if all of the following are true:

- `validation_source=explicit_val_shard` for directory-style FineWeb runs
- exact `val_bpb` is active
  - no implicit approximate-`bpb` fallback
- `scan_backend=auto` on CUDA resolves to `assoc_accel`
  - no silent scan-backend downgrade
- the shared spec matches the intended structure and embedding-init contract
  - spectral basis build failure is not treated as “close enough” to `svd`

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

Custom finalist format:

```text
name shared_spec_dir core_layers core_expansion gate_mode temporal_mode router_mode learning_rate
```

Example:

```bash
FINALIST_SPECS=$'blocks1_gate_base_ema /abs/path/to/shared 10 12.0 base ema none 0.003\nblocks1_gate_base_ema_router /abs/path/to/shared 10 12.0 base ema softmax 0.003' \
bash scripts/run_5090_finalist_confirm1b.sh
```

Queued next batch:

```bash
bash scripts/run_5090_post_temporal_queue.sh
```

## Stop Rules

Stop adding complexity and move straight to finalist confirmations if any of the following happen:

- the gate lane is flat
- the EMA lane is flat
- the router lane fails its first-pass bar

If the whole aggressive lane stalls, the remaining budget should go to:

- confirming the best safe-lane point
- optionally one last clean `max_lr` follow-up on the incumbent `blocks1` baseline

Do not spend the final week on:

- more frozen blocks
- `core_dim` changes
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
