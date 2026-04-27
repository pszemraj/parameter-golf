# 5090 Shape Reassessment

Last updated: `2026-04-27`

The geometry pivot replaced the old `core_dim=48`, deep-controller frontier
with a CUDA-friendlier shape search over:

```text
core_dim x layers x inner_dim
```

For minGRU, `inner_dim = int(core_dim * expansion)` is the recurrent state
scanned over time. It is not only an MLP width.

## Shape Principle

The old quality leader had many recurrent cells but poor local GPU geometry:

```text
48 x 12 x inner480 = 5760 recurrent cells
```

The shallow speed probe was much faster but discarded recurrent depth:

```text
128 x 4 x inner512 = 2048 recurrent cells
```

The winning compromise was:

```text
128 x 5 x inner512 = 2560 recurrent cells
```

It doubled frozen basis width versus `core_dim=64`, kept a clean `inner_dim`,
and preserved enough stacked recurrence to beat the other aligned candidates.

## Completed Geometry Read

| Run | Contract | Full-val / screen BPB | Steady tok/s | Artifact bytes |
|---|---:|---:|---:|---:|
| `d96_l6_i512` K2 | `512M` | `2.0668155804` | `990,977` | `7,856,990` |
| `d64_l10_i512` K2 | `512M` | `2.0677617650` | `645,010` | `7,954,836` |
| `d128_l4_i512` K2 | `512M` | `2.0681435993` | `1,372,453` | `8,568,493` |
| `d128_l5_i512` K2 | `512M` | `2.0563568016` | `1,128,480` | `8,806,358` |
| `d96_l6_i512` K2 | `1B` full-val | `2.0264627708` | `1,006,543` | `7,942,889` |
| `d128_l5_i512` K2 | `1B` full-val | `2.0031207874` | `1,137,730` | `8,830,483` |

Read:

- `d128_l5_i512` is the aligned geometry winner.
- `d96_l6_i512` was the balanced alternative but lost clearly at `1B`.
- `d64_l10_i512` preserved more recurrent cells but did not translate that into
  better BPB.
- `d128_l4_i512` was the speed frontier, but the fifth layer was worth it.

## Current Defaults

Use this geometry for current finalist work:

```text
core_dim = 128
core_layers = 5
core_inner_dim = 512
core_expansion = 4.0
num_blocks = 0
branch_lags = 1,2,3,4,6,8,12,16,24,32,48,64
```

With K6 `seq2048` BPTT2, this is the current local leader:

```text
val_bpb = 1.9572908661
artifact estimate = 13,798,090 bytes
```

## Reopen Only With Evidence

Do not rerun the old shape frontier by default. Reopen geometry only if a later
diagnostic shows a specific bottleneck:

- underfitting despite unused artifact headroom
- K7/K8 cannot fit without reducing frozen/readout bytes
- H100 profiling shows a shape-specific runtime issue not visible on the 5090

If blocks return, test them through `d64` or reduced branch counts first.
`d128+` with 12 branches and frozen blocks grows square matrix costs quickly.

Seed policy for shape work stays simple: seed `1337` only unless the user
explicitly asks for a stability report.
