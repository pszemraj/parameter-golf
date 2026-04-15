# HGDN Megakernel H100 Resource Report

## Candidate shape

This report targets the live winner-style HGDN contract:

- `B = 32`
- `T = 2048`
- `D = 384`
- `H = 8`
- `Dk = 48`
- `Dv = 48`
- `K = 4`
- packed qkv projection
- packed depthwise causal qkv conv
- dense cross-head `W_out`

## Save-heavy parity version

The first repo-backed candidate intentionally saves a large forward context so backward parity stays simple.

| Saved tensor | Shape | Dtype | Approx bytes |
| --- | --- | --- | ---: |
| `qkv` | `(B, T, H * (2 * Dk + Dv)) = (32, 2048, 1152)` | bf16 | 150,994,944 |
| `pre` | `(32, 2048, 1152)` | bf16 | 150,994,944 |
| `q_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `k_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `v_post` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `inv_q` | `(32, 2048, 8)` | fp32 | 2,097,152 |
| `inv_k` | `(32, 2048, 8)` | fp32 | 2,097,152 |
| `g_pre` | `(32, 2048, 8)` | bf16 | 1,048,576 |
| `beta_pre` | `(32, 2048, 8)` | bf16 | 1,048,576 |
| `g_log` | `(32, 2048, 8)` | bf16 | 1,048,576 |
| `beta` | `(32, 2048, 8)` | bf16 | 1,048,576 |
| `g_out` | `(32, 2048, 384)` | bf16 | 50,331,648 |
| `o_raw` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `o_norm` | `(32, 2048, 8, 48)` | bf16 | 50,331,648 |
| `z` | `(32, 2048, 384)` | bf16 | 50,331,648 |
| `state_prev` | `(32, 2048, 8, 48, 48)` | fp32 | 4,831,838,208 |

Total saved forward state is about **5.45 GiB per GDN block**.

## Shared-memory map

The current cooperative kernel uses:

- tiled shared-memory GEMM staging for dense projections:
  - two `16 x 16` fp32 tiles
- one CTA-local recurrence tile per `(batch, head, value_tile)` stream
  - `REC_V_TILE = 8` on the current local candidate

- `S0`: `Dk * REC_V_TILE = 384` fp32 values
- `S1`: `384` fp32 values
- `adj`: `384` fp32 values
- `q`: `48` fp32 values
- `k`: `48` fp32 values
- `v`: `8` fp32 values
- `go`: `8` fp32 values
- `tmp_dv0`: `8` fp32 values
- `tmp_dv1`: `8` fp32 values
- `tmp_dk0`: `48` fp32 values
- `tmp_dk1`: `48` fp32 values
- two `THREADS=256` reduction buffers for CTA-local reductions

Recurrence-tile shared memory for `Dk=48`, `REC_V_TILE=8` is:

- `3 * 384 + 4 * 48 + 4 * 8 + 2 * 256 = 1888` fp32 values
- `1888 * 4 = 7,552` bytes
- about **7.38 KiB per CTA**

GEMM staging shared memory is smaller:

- `2 * 16 * 16 = 512` fp32 values
- `512 * 4 = 2,048` bytes

So the recurrence tile still sets the dynamic shared-memory request.

## Occupancy estimate

The repo-backed launch wrapper uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor` instead of a hard-coded block count.

With `THREADS=256` and about `7.38 KiB` dynamic shared memory per CTA:

- shared memory alone allows multiple CTAs per SM on H100 SXM
- the real limiter is expected to be register pressure from:
  - the explicit backward math
  - the tiled GEMM accumulators
  - the recurrence temporaries
- practical cooperative occupancy should be materially better than the first scalar-loop draft, but this is still not a high-occupancy kernel

Recurrence parallelism is now structurally capped at `B * H * ceil(Dv / REC_V_TILE)` CTAs.
At the candidate shape that is:

- `32 * 8 * 6 = 1,536` recurrence streams

On a 132-SM H100 SXM this means the recurrence portion can only expose about:

- `11.64` streams per SM at best

That is a real improvement over the first draft, but it still does not guarantee a final throughput winner because each stream is still a lightweight sequential recurrence tile rather than a tensor-core-friendly bulk kernel.

## FLOP and traffic sanity

Approximate forward work at the candidate shape:

- packed dense `W_qkv`: about `57.98 GF`
- `w_a + w_b + w_g`: about `20.13 GF`
- packed depthwise causal conv: about `0.60 GF`
- gated-delta recurrence update + readout: about `12.08 GF`
- dense `W_out`: about `19.33 GF`

Total forward work is about **110 GF per block**.

Backward is analytic and explicit in the same kernel and should land in the rough range of **2.5x to 3.5x forward work** for this save-heavy implementation.

HBM traffic is dominated by:

- dense reads of `x`, `w_qkv`, `w_g`, and `w_out`
- large saved-state reads in backward, especially `state_prev`

The current version is therefore expected to be:

- materially better than the first scalar-loop draft
- still memory-traffic heavy
- still not tensor-core efficient

## Roofline-style conclusion

This candidate is good enough for:

- local compile/parity
- local launch-count proof
- H100 compile/parity testing

This candidate is **not** a credible final H100 throughput kernel yet.

The blunt bottlenecks are:

1. Dense `W_qkv`, `W_g`, `W_out`, and the major backward matmuls are tiled shared-memory GEMMs, but they are still not tensor-core kernels.
2. The backward path still stores and rereads a huge `state_prev` tensor.
3. The recurrence is now value-tiled, but the core algorithm is still sequential within each tile and remains the scaling limiter at longer sequence lengths.

If H100 parity is clean, the next speed branch should be one of:

- tensor-core `W_qkv`, `W_g`, `W_out`, and backward matmuls inside the megakernel, likely Hopper-specific
- or a justified split where cuBLAS/cuBLASLt owns dense GEMMs and one fused HGDN kernel owns the recurrent shell

The first branch is the purist megakernel route.
The second branch may be the faster route to a competition-worthy H100 candidate.
