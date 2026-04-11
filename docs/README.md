# HGDN Branch Status

Last updated: 2026-04-11

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
- The current architecture stage is fixed-token screening, then an H100 batch-scale finalist pass on the top `2` candidates, then one exact 8x matched-control go/no-go run.
- Real HGDN ablations go to `pg-hgdn-ablations`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

## Working rules

- Use the local GPU for broad fixed-data architecture search when the candidate family fits.
- Use 1xH100 for finalist ranking under the same fixed-token contract.
- After fixed-token ranking, run a separate H100 wallclock-aware batch-scale / packing pass before the final architecture call.
- Because the trainer defaults to `grad_accum_steps = 8 / world_size`, the 1xH100 proxy preserves the same per-GPU local-batch mapping for a given `TRAIN_BATCH_TOKENS` unless that knob is explicitly overridden; the unresolved question after the proxy pass is still the exact 8x contract result.
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
  - over limit at `17,580,964` bytes
- Current fixed-token H100 leader:
  - `h100retune6_f_fixed2k_hybrid_r1_mlp3.5_seq2048`
  - `14L x 384d x mlp3.5`
  - final roundtrip `2.4224`
  - last-step time `905.68 ms`
  - artifact total `15,878,878`
  - headroom `121,122`
- Current speed-and-headroom companion:
  - `h100retune6_d_fixed2k_hybrid_r1_mlp3.25_seq2048`
  - `14L x 384d x mlp3.25`
  - final roundtrip `2.4245`
  - last-step time `898.55 ms`
  - artifact total `15,052,320`
  - headroom `947,680`
- The old `14L x 384d x mlp3.375` point was superseded:
  - `h100retune6_e_fixed2k_hybrid_r1_mlp3.375_seq2048`
  - slower and worse than both `mlp3.25` and `mlp3.5`
- `h100retune6` killed the revived `15L x 384d` family again:
  - `mlp2.375`: roundtrip `2.4594`, last step `970.81 ms`
  - `mlp2.4375`: roundtrip `2.4663`, last step `979.52 ms`
  - `mlp2.5`: roundtrip `2.4640`, last step `987.09 ms`
  - read: slower and materially worse than the `14L x 384d` bracket
- Historical absolute `step_ms` deltas to `h100k6` are still not the clean speed control:
  - use the within-round H100 ordering for the fixed-token screen
  - use the separate H100 batch-scale / packing pass for the wallclock decision
- Live architecture call:
  - carry only `14L x 384d x mlp3.25` and `14L x 384d x mlp3.5` into the next H100 finalist pass
  - treat `mlp3.5` as the quality lead and `mlp3.25` as the efficiency hedge
  - stop spending fixed-token H100 runs on the `15L x 384d` bracket

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
- The next useful evidence is the H100 batch-scale / packing pass and then one exact 8x HGDN-vs-attention-only control run, not another Python-side layout experiment.
