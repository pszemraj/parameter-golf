# HGDN Branch Status

Last updated: 2026-04-12

Branch: `exp/hgdn`

## Current state

- Active H100 kernel winner: [`winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml)
- Active winner flags:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- The current post-conv front-end seam is closed after `h100k20`.
- The current architecture stage is fixed-token screening, then an H100 batch-scale finalist pass, then one exact-mappable H100 packing refinement on the live bracket, then one exact 8x matched-control go/no-go run.
- Real HGDN ablations go to `pg-hgdn-ablations`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

## Active architecture bracket

- Live exact-mappable finalists:
  - `14L x 384d x mlp3.25`
    - current exact-mappable leader: `h100pack1_c_fixed2k_hybrid_r1_mlp3.25_seq2048`
  - `14L x 384d x mlp3.5`
    - current exact-mappable companion: `h100pack1_g_fixed2k_hybrid_r1_mlp3.5_seq2048`
- `15L x 384d` is out of the live bracket.
- `double-global local32` is the current exact-mappable contract to beat.
- The only unresolved packing question on the live bracket is whether `double-global local64` beats `double-global local32`.
- Exact metrics, reject rationale, and run history live in [PROFILING_LOG.md](PROFILING_LOG.md).

## Operating rules

- Use the local GPU for broad fixed-token architecture search when the candidate family fits memory.
- Use 1xH100 for finalist ranking under the same fixed-token contract.
- After fixed-token ranking, run a separate H100 batch-scale / packing pass before the final architecture call.
- If that pass exposes a missing exact-8x cross-term, run one narrow exact-mappable H100 refinement before paying for the full 8x bridge.
- Because the trainer defaults to `grad_accum_steps = 8 / world_size`, the 1xH100 proxy preserves the same per-GPU local-batch mapping for a given `TRAIN_BATCH_TOKENS` unless that knob is explicitly overridden. The unresolved question after the proxy pass is still the exact 8x contract result.
- Fixed-token sweeps answer learning efficiency. The final architecture call still has to survive the H100 wallclock contract.
- Low VRAM use during saturated fixed-token H100 runs is not a failure by itself. Treat it as a reason to study packing, not to discard the architecture.
- On the compiled HGDN path, default to changes that alter the generated path. Python-side view reshuffles and `.contiguous()` edits are not the main lever.
- Enable compile diagnostics only when needed:
  - `TORCH_LOGS=recompiles,graph_breaks`
  - optional `TORCH_TRACE=/tmp/tracedir`

## Competition timing

- The challenge has separate training-time and evaluation-time budgets.
- In both `train_gpt.py` and `train_gpt_hybrid.py`, the training timer excludes warmup/compile priming, validation passes, and final serialization plus roundtrip eval.
- Fixed-token sweeps are therefore a screening tool for learning efficiency, not the final leaderboard objective.

## Screening checkpoints

- Keep local and H100 screening as separate stages. They do not stabilize at the same step count.
- Local broad architecture screen:
  - default to `300` steps
  - keep `VAL_LOSS_EVERY=100`
- Local shortlist confirmation:
  - rerun the survivors for `500` steps
- H100 fixed-token finalist screen:
  - default to `1500` steps
  - keep `VAL_LOSS_EVERY=500`
- Only pay for `2000` H100 steps when the `1500`-step finalists are still crossing, the margin is tiny, or a final tie-break is needed.
- Current evidence behind those defaults:
  - local `retune2`: ranking was already mostly stable by `300` and very stable by `500`
  - H100 `retune3`: `500` was misleading, `1000` was only a coarse filter, and `1500` was the first checkpoint that looked like the `2000` result
- Do not use the `500`-step H100 screen to kill viable candidates.

## Current boundaries

- Treat norm placement as a learning-dynamics screen, not as kernel work. Compare `NORM_STYLE=pre|post|keel` inside the HGDN family first, then against the attention-only baseline via `GDN_RATIO=0`.
- Use compile/backend work only on finalists.
- Do not reopen the closed front-end seam for more ownership or packing tweaks unless the decomposition changes materially.

## Launch entrypoints

Structured launcher:

- [`../scripts/hgdn.py`](../scripts/hgdn.py)

Single-entry batch helpers:

- [`../scripts/run_local_hgdn_resize_round.sh`](../scripts/run_local_hgdn_resize_round.sh)
- [`../scripts/run_h100_hgdn_resize_round.sh`](../scripts/run_h100_hgdn_resize_round.sh)

Data / setup helpers:

- [`../scripts/bootstrap_challenge_data.sh`](../scripts/bootstrap_challenge_data.sh)

1xH100 perf / profiling helpers:

- [`../scripts/run_h100_single_gpu_hgdn.sh`](../scripts/run_h100_single_gpu_hgdn.sh)
- [`../scripts/run_h100_single_gpu_hgdn_profile.sh`](../scripts/run_h100_single_gpu_hgdn_profile.sh)
- [`../scripts/hgdn_cuda_preflight.py`](../scripts/hgdn_cuda_preflight.py)

## Related docs

- W&B schema and project rules: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
- local/H100 transfer limits: [HARDWARE_TRANSFER.md](HARDWARE_TRANSFER.md)
- CUDA extension build/runtime notes: [HGDN_CUDA_FUSED.md](HGDN_CUDA_FUSED.md)
- profiling chronology and scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- active next steps: [TODO.md](TODO.md)
- cleanup targets: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
- OLMo/Hybrid reference notes: [REFERENCE.md](REFERENCE.md)
