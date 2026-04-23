# 5090 Architecture Plan

Last updated: `2026-04-23`

This is the official plan for the next Core/Amplifier lane on the local RTX 5090.

Deadline-oriented execution order now lives in:

- [docs/5090_final_week_plan.md](/home/pszemraj/workspace/projects/parameter-golf/docs/5090_final_week_plan.md)

## Thesis

The current evidence supports a refined version of the original idea:

- the frozen statistical basis is useful
- the trainable controller should spend its capacity correcting hard tokens, not relearning easy local structure
- temporal structure matters, but the old lag path was too weak and too close to “alternate projections of the current state”

The durable idea is therefore:

> frozen statistics for easy structure, plus a small parallel minGRU that selectively corrects what the statistics miss

This is still explicitly not a transformer path:

- no attention
- no token-token mixing
- no TTT / LoRA-TTT

## Current Evidence

### Frontier

Best completed `1B` results by three-seed mean:

| Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---:|---:|---:|---:|
| `blocks1_resid10_e12_h7000_1b` | `2.1865341393` | `0.0052472581` | `372,888` | `4,790,559` |
| `blocks1_resid12_e10_h7000_1b` | `2.1866023565` | `0.0051325073` | `374,271` | `4,791,951` |
| `blocks0_resid12_e10_h7000_1b` | `2.1899359311` | `0.0039254056` | `386,331` | `3,945,168` |
| `blocks2_resid12_e8_h7000_1b` | `2.2005760974` | `0.0018741094` | `442,062` | `5,326,970` |

### Schedule Is Locked Enough

The hold-retune screen is complete and consistent:

| Representative | `h2500` mean | `h3500` mean | `h4096` mean | Winner |
|---|---:|---:|---:|---|
| `blocks1 10x12` | `2.2724088059` | `2.2680191476` | `2.2892407482` | `h3500` |
| `blocks0 12x10` | `2.2732337556` | `2.2716177172` | `2.2823704208` | `h3500` |
| `blocks2 12x8` | `2.2821037577` | `2.2801177077` | `2.2941590571` | `h3500` |

Working defaults:

- `512M` screens: `lr_hold_steps=3500`
- `1B` confirmations: `lr_hold_steps=7000`

That means the next uncertainty is architecture, not more hold work.

## New Implementation Surface

The active Core/Amplifier code now supports:

### 1. Tokenwise Residual Gating

`residual_token_gate_mode`:

- `none`
- `base`
- `core_base`

Behavior:

- one scalar gate per token multiplies `residual_logits` before the global residual scale
- `base` uses frozen-base confidence only:
  - entropy
  - top-1 probability
  - top-1 minus top-2 margin
- `core_base` appends RMS-normalized controller state

Init:

- zero weights
- bias `-1.5`

So the controller starts conservative.

### 2. Real Causal Temporal Taps

`branch_temporal_mode` now supports:

- `current`
- `lagged`
- `hybrid`
- `ema`
- `ema_hybrid`

EMA definition for lag `L`:

```text
s_L[t] = d_L * s_L[t-1] + (1 - d_L) * core_out[t]
d_L = exp(-1 / L)
```

`ema_hybrid` mixes the current state with the EMA state using the existing lag-mix scale.

### 3. Per-Token Branch Routing

`branch_router_mode`:

- `none`
- `softmax`

Behavior:

- router input uses the same conditioning features as `core_base`
- outputs one weight per branch per token
- softmax is rescaled by `num_branches` so the zero-init router is exactly uniform

### 4. Fast Scan Backend Selection

`scan_backend`:

- `auto`
- `heinsen`
- `assoc`
- `assoc_accel`
- `sequential`

`auto` behavior:

- CUDA + required scan deps installed: `assoc_accel`
- non-CUDA: `assoc` via the repo-local Torch associative scan
- otherwise: fail loudly instead of silently downgrading the maintained CUDA path

The accelerated CUDA scan dependency is now treated as part of the core path, and the slower associative fallback is owned in-repo rather than delegated to a wrapper package.

This backend is now used in:

- the parallel minGRU path
- the EMA / EMA-hybrid branch path

## Local Speed Signal

These are local primitive-level microbenchmarks on the 5090, not end-to-end trainer claims.

### minGRU recurrence

Representative shape:

- batch `256`
- sequence `512`
- model dim `48`
- expansion `12`

Measured average training-step time per layer:

- `heinsen`: `30.893 ms`
- `assoc_accel`: `21.041 ms`

That is about a `1.47x` speedup for the recurrent primitive.

### EMA branch recurrence

Representative shape:

- batch `64`
- sequence `512`
- branches `12`
- branch width `48`

Measured average forward+backward time:

- `sequential`: `95.314 ms`
- `assoc_accel`: `2.844 ms`

That is a large enough difference that the EMA lane should treat accelerated scan as the intended runtime path.

## Immediate Experiment Order

### Batch A: Residual-Gate Screen

Use:

- `blocks1_resid10_e12`
- `blocks0_resid12_e10`

Contract:

- `512M`
- `4096` steps
- `seq_len=512`
- `TARGET_EFFECTIVE_STEP_TOKENS=131072`
- `carry_chunks=8`
- `bptt_chunks=1`
- `max_lr=3e-3`
- `min_lr=3e-4`
- `warmup_steps=100`
- `lr_hold_steps=3500`
- `weight_decay=1e-3`
- `branch_temporal_mode=current`
- `branch_router_mode=none`
- `scan_backend=auto`
- seeds `1337`, `2027`

Grid:

- `residual_token_gate_mode in {none, base, core_base}`

Promotion rule:

- require at least `0.003` mean `val_bpb` gain
- allow at most `5%` throughput loss

Convenience launcher:

```bash
bash scripts/run_5090_architecture_gate_screen.sh
```

### Batch B: Temporal-Tap Screen

Run only after Batch A.

Primary representative:

- best Batch-A `blocks1` point

Compare:

- `branch_temporal_mode=current`
- `branch_temporal_mode=ema`
- `branch_temporal_mode=ema_hybrid`

Keep:

- gate winner from Batch A
- `branch_router_mode=none`
- `scan_backend=auto`

Promotion rule:

- require at least `0.003` mean `val_bpb` gain

If the best temporal winner is real on `blocks1`, replay that single temporal winner on:

- `blocks0_resid12_e10`
- `blocks2_resid12_e8`

Convenience launcher:

```bash
bash scripts/run_5090_architecture_temporal_screen.sh
```

### Batch C: Branch-Router Screen

Run only after Batch B.

Use:

- best temporal winner

Compare:

- `branch_router_mode=none`
- `branch_router_mode=softmax`

Promotion rule:

- require at least `0.0025` mean `val_bpb` gain
- allow at most `10%` throughput loss

Convenience launcher:

```bash
bash scripts/run_5090_architecture_router_screen.sh
```

### Batch D: `1B` Confirmation

Confirm the top one or two architecture winners at:

- `8192` steps
- `1B` planned tokens
- seeds `1337`, `2027`, `3141`
- `lr_hold_steps=7000`

Convenience launcher:

```bash
bash scripts/run_5090_finalist_confirm1b.sh
```

## What We Are Not Doing Next

- more frozen blocks beyond `blocks2`
- `core_dim` changes
- attention-like augmentations
- token-token mixing
- eval-only hacks
- more hold sweeps unless the architecture lane fully stalls

## Success Criteria

The next lane succeeds if any of the following happen:

- tokenwise residual gating makes the controller visibly more selective and improves mean `val_bpb`
- EMA temporal taps beat `current` and show that real multi-timescale temporal structure helps
- branch routing improves quality without turning into an expensive learned mixer

If none of those clear the thresholds, stop the architecture lane and return to a clean `max_lr` screen on the current `blocks1` baseline rather than layering on more complexity.
