# HGDN Branch Status

Last updated: 2026-04-10

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
- The current architecture stage is compute-optimal resize under matched fixed-token contracts, followed by a separate H100 batch-scale / packing pass on the top `2-3` finalists.
- Real HGDN ablations go to `pg-hgdn-ablations`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

## Working rules

- Use the local GPU for broad fixed-data architecture search when the candidate family fits.
- Use 1xH100 for finalist ranking under the same fixed-token contract.
- After fixed-token ranking, run a separate H100 batch-scale / packing follow-up before the final architecture call.
- Treat low VRAM use during saturated fixed-token H100 runs as headroom for the follow-up pass, not as a reason to restart the ranking.
- On the compiled HGDN path, default to changes that alter the generated path. Python-side view reshuffles and `.contiguous()` edits are not the main lever.
- Enable compile diagnostics only when needed:
  - `TORCH_LOGS=recompiles,graph_breaks`
  - optional `TORCH_TRACE=/tmp/tracedir`

## Current architecture read

- Fixed-2k H100 reference:
  - `h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048`
  - `16L x 384d x mlp3.25`
- First H100 resize leader:
  - `h100retune_a_fixed2k_hybrid_r1_mlp3.25_seq2048`
  - `14L x 384d x mlp3.25`
  - roundtrip improved `2.4438 -> 2.4243`
  - last-step time improved `915.10 -> 897.96 ms`
  - artifact status moved from over-limit to under-limit
- Rejected resize branch:
  - `16L x 320d x mlp3.25`
  - worse quality and slower than the reference
- Later local broad search shifted the live family toward `15L x 384d` with `MLP_MULT` around `2.5-3.0`.
- Current H100 finalist batch is the `h100retune3` round.
- Final architecture selection waits for the fixed-token H100 ranking and the separate batch-scale / packing follow-up.

Exact run history, scoreboards, and reject rationale live in [PROFILING_LOG.md](PROFILING_LOG.md).

## Norm placement

HGDN supports `NORM_STYLE=pre|post|keel`.

- `pre`: current default
- `post`: plain post-residual RMSNorm
- `keel`: inner transform-branch norm plus post-residual norm, with `residual_alpha`

Treat norm placement as a learning-dynamics screen, not as kernel work. Compare it inside the HGDN family first, then against the attention-only baseline via `GDN_RATIO=0`.

## Launch helpers

Structured launcher:

- [`../scripts/hgdn.py`](../scripts/hgdn.py)

Single-entry batch helpers:

- [`../scripts/run_local_hgdn_resize_round.sh`](../scripts/run_local_hgdn_resize_round.sh)
- [`../scripts/run_h100_hgdn_resize_round.sh`](../scripts/run_h100_hgdn_resize_round.sh)

Data / setup helpers:

- [`../scripts/bootstrap_challenge_data.sh`](../scripts/bootstrap_challenge_data.sh)
- [`../scripts/build_hgdn_cuda.sh`](../scripts/build_hgdn_cuda.sh)

1xH100 perf / profiling helpers:

- [`../scripts/run_h100_single_gpu_hgdn.sh`](../scripts/run_h100_single_gpu_hgdn.sh)
- [`../scripts/run_h100_single_gpu_hgdn_profile.sh`](../scripts/run_h100_single_gpu_hgdn_profile.sh)
- [`../scripts/run_hgdn_cuda_preflight.sh`](../scripts/run_hgdn_cuda_preflight.sh)

Local analysis helpers:

- [`../scripts/profile_hgdn_local_hotpath.py`](../scripts/profile_hgdn_local_hotpath.py)
- [`../scripts/analyze_hgdn_phase1.py`](../scripts/analyze_hgdn_phase1.py)
- [`../scripts/compare_hgdn_phase1.py`](../scripts/compare_hgdn_phase1.py)
- [`../scripts/compare_hgdn_fixed2k.py`](../scripts/compare_hgdn_fixed2k.py)
- [`../scripts/export_wandb_hgdn_runs.py`](../scripts/export_wandb_hgdn_runs.py)
- [`../scripts/hgdn_kernel_scoreboard.py`](../scripts/hgdn_kernel_scoreboard.py)
- [`../scripts/screen_hgdn_arch_sizes.py`](../scripts/screen_hgdn_arch_sizes.py)

## Related docs

- W&B schema and project rules: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
- local/H100 transfer limits: [HARDWARE_TRANSFER.md](HARDWARE_TRANSFER.md)
- CUDA extension build/runtime notes: [HGDN_CUDA_FUSED.md](HGDN_CUDA_FUSED.md)
- profiling chronology and scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- active next steps: [TODO.md](TODO.md)
- cleanup targets: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
- OLMo/Hybrid reference notes: [REFERENCE.md](REFERENCE.md)

## Files worth touching

- HGDN model stack:
  - [`../model.py`](../model.py)
  - [`../train_gpt_hybrid.py`](../train_gpt_hybrid.py)
- Active winner config:
  - [`../configs/hgdn/winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml)
- Norm helper:
  - [`../scripts/run_laptop_norm_compare.sh`](../scripts/run_laptop_norm_compare.sh)
- Baseline norm ablation helper:
  - [`../scripts/run_train_gpt_norm_ablate.sh`](../scripts/run_train_gpt_norm_ablate.sh)
- CUDA extension entrypoints:
  - [`../setup_hgdn_cuda.py`](../setup_hgdn_cuda.py)
  - [`../scripts/hgdn_cuda_parity.py`](../scripts/hgdn_cuda_parity.py)

## Practical read

- HGDN already shows a learning-per-step edge over the attention-only baseline on matched H100 quality checks.
- The unresolved question is compute-optimal sizing and packing, not whether the branch should keep relitigating the closed front-end seam.
- The next useful evidence is the H100 resize ranking plus the separate batch-scale / packing pass, not another Python-side layout experiment.
