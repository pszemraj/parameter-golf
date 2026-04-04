# Hardware Transfer Notes: RTX 4070 vs 1xH100

Last updated: 2026-04-04 15:35 EDT

Branch: `exp/hgdn`

## Purpose

This note records what did and did not transfer from local RTX 4070 Laptop profiling to 1xH100 profiling for the current HGDN operating point.

The main question is practical:

- can local profiling be trusted to guide kernel work?
- if yes, which hotspot classes transfer well?
- which buckets need H100 confirmation before drawing conclusions?

## Runs Compared

The comparison below uses the same hybrid architecture:

- `16L x 384d`
- `GDN_RATIO=1` (`8G+8A`)
- `gdn_head_k_dim=48`
- `MLP_MULT=3.25`
- `TRAIN_SEQ_LEN=2048`
- `NORM_STYLE=pre`

Compared runs:

- H100 eager attribution profile:
  - `h100k1_fix2_profile_eager_hybrid_r1_mlp3.25_seq2048`
  - `TRAIN_BATCH_TOKENS=524288`
  - full eager profiler capture
- H100 compiled profile:
  - `h100k1_fix2_profile_compiled_hybrid_r1_mlp3.25_seq2048`
  - same contract as above
- local 4070 eager attribution profile:
  - `rtx4070mini_profile_eager_hybrid_r1_mlp3.25_seq2048`
  - `TRAIN_BATCH_TOKENS=131072`
  - shorter eager capture used for hotspot ranking only

Important caveat:

- the local 4070 profile had to use a smaller batch contract to fit memory
- therefore raw milliseconds are not directly comparable
- the correct comparison axis is normalized hotspot ranking, mainly `Self CUDA %`

## Main Result

The hotspot overlap is strong enough that local profiling is useful for HGDN kernel work.

What transfers well:

- `aten::copy_` / dtype-layout churn
- `aten::mul` and surrounding elementwise glue
- `gdn.conv_qkv`
- `gdn.recurrence`
- depthwise conv backward and forward
- `gdn.norm_qkv`
- `gdn.output_gate`
- `attn.norm_rope`

What does not transfer cleanly:

- GEMM-heavy buckets such as `aten::mm`
- MLP-heavy buckets
- absolute speed ratios or tokens/s

Bottom line:

- use the 4070 to optimize HGDN-specific non-matmul overhead
- do not use the 4070 to prioritize GEMM or MLP work for this branch
- confirm final ranking and payoff on H100 after each meaningful kernel pass

## Hotspot Comparison

Normalized `Self CUDA %` for the hybrid eager profile:

| Bucket | RTX 4070 | 1xH100 | Read |
|---|---:|---:|---|
| `aten::mul` | `16.29%` | `16.46%` | same dominant issue |
| `aten::copy_` | `12.47%` | `16.51%` | same issue, worse on H100 |
| `gdn.conv_qkv` | `3.74%` | `6.00%` | same issue, worse on H100 |
| `gdn.recurrence` | `4.18%` | `5.03%` | same tier |
| `ChunkGatedDeltaRuleFunctionBackward` | `4.33%` | `3.81%` | same tier |
| `aten::convolution_backward` | `2.58%` | `4.52%` | worse on H100 |
| `aten::_conv_depthwise2d` | `2.01%` | `3.05%` | worse on H100 |
| `aten::_flash_attention_backward` | `7.04%` | `6.29%` | similar tier |
| `Command Buffer Full` | `6.65%` | `7.53%` | similar tier |
| `attn.norm_rope` | `2.54%` | `4.44%` | worse on H100 |
| `gdn.norm_qkv` | `1.79%` | `2.76%` | worse on H100 |
| `gdn.output_gate` | `2.24%` | `2.59%` | similar tier |

Two buckets are notably misleading on the 4070:

| Bucket | RTX 4070 | 1xH100 | Read |
|---|---:|---:|---|
| `aten::mm` | `21.50%` | `9.83%` | 4070 overstates GEMM cost |
| `mlp.forward` | `7.01%` | `3.42%` | 4070 overstates MLP cost |

Interpretation:

- the H100 is much better at chewing through dense GEMMs and related MLP work
- the HGDN-specific glue and conv path do not get the same relative hardware advantage
- that is why the hybrid penalty looks worse on H100 even though the same slow buckets are visible locally

## What This Means for Kernel Work

### Good local optimization targets

These are worth iterating on locally first:

1. `aten::copy_`
   - dtype churn
   - redundant recasts
   - avoidable layout materialization
2. `aten::mul`
   - broadcast-heavy scalar/gate/residual elementwise glue
   - output gate path
3. `gdn.conv_qkv`
   - conv implementation
   - conv fusion opportunities
   - per-path conv ablations like `GDN_USE_V_CONV=0`
4. `gdn.norm_qkv`
   - q/k normalization path
   - recurrence input preparation
5. `attn.norm_rope`
   - only insofar as it is part of the hybrid-vs-depth gap in this trainer stack

### Buckets that must be judged on H100

These should be rechecked on H100 before making strong claims:

1. `aten::mm`
2. `mlp.forward`
3. overall throughput ratio
4. whether a local win still matters once flash-attention and Hopper GEMM kernels dominate more strongly

## Current Working Rule

For this branch:

- local 4070 profiling is a good proxy for HGDN-specific glue cleanup
- local 4070 profiling is not a good proxy for GEMM optimization priority
- H100 remains the authority for final throughput decisions

In practical terms:

- if a patch reduces `copy/mul/conv/norm/gate` cost locally, it is worth testing on H100
- if a patch only improves GEMM-heavy buckets locally, it is lower priority unless H100 also agrees

## Why This Is Useful

This result materially changes the workflow.

We do not need to treat every kernel pass as an H100-only exercise. The local GPU is useful for the high-frequency iteration loop, as long as the optimization target is chosen correctly.

That means the current branch can use:

- local 4070 for fast HGDN glue/kernel iteration
- H100 for confirmation and final payoff measurement

This should shorten the optimization cycle substantially without pretending that local throughput numbers themselves are target-hardware truth.
