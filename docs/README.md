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
- The current architecture stage is fixed-token screening followed by a separate H100 wallclock-aware finalist pass on the top `2-3` candidates.
- Real HGDN ablations go to `pg-hgdn-ablations`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

## Working rules

- Use the local GPU for broad fixed-data architecture search when the candidate family fits.
- Use 1xH100 for finalist ranking under the same fixed-token contract.
- After fixed-token ranking, run a separate H100 wallclock-aware batch-scale / packing pass before the final architecture call.
- Treat low VRAM use during saturated fixed-token H100 runs as a signal to study batch-scale behavior later, not as evidence that the fixed-token winner should be replaced.
- On the compiled HGDN path, default to changes that alter the generated path. Python-side view reshuffles and `.contiguous()` edits are not the main lever.
- Enable compile diagnostics only when needed:
  - `TORCH_LOGS=recompiles,graph_breaks`
  - optional `TORCH_TRACE=/tmp/tracedir`

## Checkpoint proxy protocol

- Keep local and H100 screening as separate stages. They do not stabilize at the same step count.
- Local broad architecture screen:
  - default to `300` steps
  - keep `VAL_LOSS_EVERY=100`
  - use this to kill bad families cheaply
- Local shortlist confirmation:
  - rerun the survivors for `500` steps
  - use this only when the `300`-step top group is close
- H100 fixed-token finalist screen:
  - default to `1500` steps
  - keep `VAL_LOSS_EVERY=500`
  - use this as the main fixed-token ranking pass
- Only pay for `2000` H100 steps when the `1500`-step finalists are still crossing, the margin is tiny, or a final tie-break is needed.
- The current evidence behind those defaults:
  - local `retune2`: ranking was already mostly stable by `300` and very stable by `500`
  - H100 `retune3`: `500` was misleading, `1000` was only a coarse filter, and `1500` was the first checkpoint that looked like the `2000` result
- Do not use the `500`-step H100 screen to kill viable candidates. It was early enough to bury the eventual legal winner.

## Competition timing

- The challenge has two separate 10-minute budgets:
  - training time inside the trainer
  - evaluation time enforced externally
- In both `train_gpt.py` and `train_gpt_hybrid.py`, the training timer excludes:
  - compile/kernel warmup steps
  - validation passes
  - final serialization and roundtrip eval
- Fixed-token sweeps are therefore a screening tool for learning efficiency, not the final leaderboard objective.
- Final architecture selection must still answer the wallclock question on H100.

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
- Fixed-token H100 ranking after the broader local search:
  - `h100retune3_b_fixed2k_hybrid_r1_mlp3.375_seq2048`
  - `14L x 384d x mlp3.375`
  - sampled eval `2.4245`
  - final roundtrip `2.4365`
  - last step time `948.72 ms`
  - artifact total `15,358,333`
- `h100retune3` disciplined the local `15L x 384d` family:
  - best `15L` point was `h100retune3_i_fixed2k_hybrid_r1_mlp3.125_seq2048`
  - it still finished over limit by `68,592` bytes and slower than the `14L` leader
  - lower-MLP `15L` points gave back too much quality
- The depth-preserving under-limit check also lost:
  - `h100retune3_j_fixed2k_hybrid_r1_mlp2.6666666666666665_seq2048`
  - `16L x 384d x mlp2.666...`
  - under limit, but slower and worse than the `14L x 384d x mlp3.375` leader
- Historical absolute `step_ms` deltas to `h100k6` are not the clean speed control anymore:
  - same-arch rerun `h100retune3_a_fixed2k_hybrid_r1_mlp3.25_seq2048` improved quality but ran much slower than `h100k6`
  - use the within-round H100 ranking for the fixed-token screen, then rerun finalists in the separate wallclock-aware batch-scale / packing pass
- Live architecture call:
  - keep `14L x 384d x mlp3.375` as the active fixed-token leader
  - keep the current `16L x 384d x mlp3.25` rerun only as an over-limit quality ceiling
  - do not keep the `15L x 384d` family as the active branch unless a new size bracket changes the result

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

1xH100 perf / profiling helpers:

- [`../scripts/run_h100_single_gpu_hgdn.sh`](../scripts/run_h100_single_gpu_hgdn.sh)
- [`../scripts/run_h100_single_gpu_hgdn_profile.sh`](../scripts/run_h100_single_gpu_hgdn_profile.sh)
- [`../scripts/hgdn_cuda_preflight.py`](../scripts/hgdn_cuda_preflight.py)

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
- CUDA extension entrypoints:
  - [`../setup_hgdn_cuda.py`](../setup_hgdn_cuda.py)
  - [`../scripts/hgdn_cuda_parity.py`](../scripts/hgdn_cuda_parity.py)

## Practical read

- HGDN already shows a learning-per-step edge over the attention-only baseline on matched H100 quality checks.
- The unresolved question is compute-optimal sizing and packing, not whether the branch should keep relitigating the closed front-end seam.
- The next useful evidence is the H100 resize ranking plus the separate batch-scale / packing pass, not another Python-side layout experiment.
