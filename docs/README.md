# HGDN Branch Status

Branch: `exp/hgdn-final`

## Current Position

- Active path: packed FLA-backed sparse HGDN in `train_gpt_hybrid.py`.
- Exact comparator: `train_gpt.py`. The hybrid `GDN_RATIO=0` path is only an
  attention-only diagnostic control.
- Removed path: the owned full-block CUDA kernel and core-kernel experiments.
  The live trainer no longer has owned-kernel flags, branches, preflight, or
  tests.
- Preserved path: `hgdn_cuda/`, `setup_hgdn_cuda.py`, packed QKV projection,
  packed depthwise QKV conv, and FLA recurrence support.
- GDN dynamics now match the important upstream FLA priors more closely:
  `A_log` and `dt_bias` use timescale-aware initialization, those decay params
  run in a no-weight-decay Adam group, negative eigenvalues stay enabled by
  default, and each GDN layer has a learned output RMSNorm weight.
- `GDN_FLA_RECURRENCE_MODE=direct` is available for wrapper-tax ablations.
  `GDN_FLA_RECURRENCE_MODE=direct_fused` or
  `GDN_USE_DIRECT_FLA_LAYER_SEMANTICS=1` additionally tests upstream-style
  in-kernel q/k L2 normalization and decay-gate activation. The default remains
  `compile_visible`. All FLA recurrence modes now use the public FLA default
  scale, `1 / sqrt(head_k_dim)`.
- Requested optional `GDN_USE_CUDA_*` paths now fail fast in the trainer when
  `hgdn_cuda_ext` is not loaded instead of silently falling back.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

The active branch question is narrow: can a sparse HGDN shell beat the exact
repo baseline on the exact baseline-shaped contract after accounting for speed
and artifact size?

## OLMo/FLA Alignment Audit

| Item | Status | Branch handling |
|---|---|---|
| Public FLA recurrence | Resolved | Custom HGDN calls public `fla.ops.gated_delta_rule.chunk_gated_delta_rule`; naive recurrence is refused for active CUDA HGDN training. |
| Decay init | Resolved | `A_log` / `dt_bias` use timescale-aware FLA-style initialization and log `gdn_decay_init` startup stats. |
| Decay weight decay | Resolved | `A_log` / `dt_bias` use a separate Adam group with `weight_decay=0.0`. |
| Recurrence scale | Resolved | Compile-visible and direct custom recurrence paths keep upstream FLA's default `1 / sqrt(head_k_dim)` scale instead of forcing `1.0`. |
| Negative eigenvalues | Resolved | Custom HGDN defaults `GDN_ALLOW_NEG_EIGVAL=1`; native FLA configs set `FLA_ALLOW_NEG_EIGVAL=true`. |
| Output norm/gate parameterization | Resolved enough for custom path | Custom HGDN now has learned per-`head_v_dim` `o_norm_weight`; it still uses PyTorch/sidecar norm+SiLU instead of `FusedRMSNormGated` directly. |
| Output norm epsilon | Documented deviation | Custom HGDN keeps the branch default `eps=1e-6`; native FLA defaults to `1e-5`. Treat `1e-5` as a strict-fidelity ablation, not the active default. |
| `expand_v=2.0` prior | Added as candidates | Practical sparse search still includes cheaper `1.0`/`1.5` variants, but OLMo-prior `v2` candidates are now first-class configs. |
| 3:1 GDN:attention prior | Added as candidate | `l8_d512_olmoish_6g2a_v2_m1p25` is the 6G/2A reality check; it is not the default promotion candidate until wallclock evidence supports it. |
| Native FLA control | Clarified | `train_gpt_fla_control.py` remains pure native GDN calibration, not OLMo Hybrid; an OLMo-ish SP1024 native config exists for dimension/value-width calibration. |
| Wrapper overhead | Measurable | `scripts/bench_fla_recurrence_paths.py` times wrapper, direct custom recurrence, direct fused FLA semantics, and native `fla.layers.GatedDeltaNet`. |
| Optional local CUDA extension fallback | Resolved in trainer | Requested `GDN_USE_CUDA_*` paths fail fast if `hgdn_cuda_ext` is absent. |

## Local Sparse Search

Completed bundle: `local-scratch/localnaivehgdn_sparse3_bundle`.

Analyze it with:

```bash
conda run -s --name pg python scripts/analyze_local_naive_contract_bundle.py --top 20
```

The completed sparse3 bundle predates the decay-init/no-WD/output-norm fix, so
it should be treated as pre-fix evidence. It still identifies useful shapes,
but exact promotion should rerun the shortlist under the corrected GDN dynamics.

Pre-fix promotion call:

| Role | Config | Local result |
|---|---|---|
| Primary H100 sparse candidate | `configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml` | `458.84 ms/step`, sampled BPB `1.6884`, speed-rank HGDN `1`, `UNDER_LIMIT` |
| Secondary quality ceiling | `configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml` | `586.24 ms/step`, sampled BPB `1.6839`, fixed-step rank HGDN `1`, `UNDER_LIMIT` |
| Matched attention-only diagnostic control | `configs/hgdn/naive_contract_l8_d512_r0_m2.toml` | `444.40 ms/step`, sampled BPB `1.6921`, `UNDER_LIMIT` |

The practical rerun shortlist starts with the same primary candidate, the
matched attention-only diagnostic control, and two new OLMo-prior checks:

- `configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml`
- `configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml`

The local helper includes both OLMo-prior configs plus matched attention-only
diagnostic controls for `mlp1.5` and `mlp1.25`; the H100 helper can infer those
controls when `ATTN_CONFIG` is not set.

## H100 Commands

Primary exact-baseline comparison:

```bash
USE_WANDB=0 WANDB_MODE=offline \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
MUON_DISTRIBUTED_MODE=packed_allreduce \
GDN_W_G_OPTIMIZER=matrix \
HGDN_CONFIG=configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml \
ATTN_CONFIG=configs/hgdn/naive_contract_l8_d512_r0_m2.toml \
WANDB_WATCH=none \
RUN_PREFIX_BASE=h100naive_sparse_primary \
bash scripts/run_h100_hgdn_naive_contract_round.sh
```

Optional quality-ceiling substitution:

```bash
HGDN_CONFIG=configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml
```

The helper runs three legs: exact `train_gpt.py` baseline, config-driven sparse
HGDN, and the matched attention-only diagnostic control. It pins `DATA_PATH`,
`TOKENIZER_PATH`, and `VOCAB_SIZE` for the exact baseline leg.

Local sparse search helper:

```bash
USE_WANDB=0 WANDB_MODE=offline \
DISTRIBUTED_MODE=parallel_muon \
RUN_PREFIX_BASE=localnaivehgdn_sparse3 \
bash scripts/run_local_hgdn_naive_contract_search.sh
```

Active helpers launch `torchrun train_gpt_hybrid.py` directly.

## FLA Calibration

The native FLA stack is now checked explicitly:

```bash
conda run -s --name pg python scripts/probe_fla_stack.py
```

The isolated native-FLA control lives in `train_gpt_fla_control.py` with two
separate configs:

```bash
configs/fla/native_prlike_gdn10_d544_sp8192.toml
configs/fla/native_olmoish_gdn8_d512_sp1024.toml
```

The PR-like config keeps the 10 pure native `fla.layers.GatedDeltaNet` layers,
`d_model=544`, 8 heads, `head_dim=64`, MLP multiplier `3.0`, and SP8192
data/tokenizer defaults. The OLMo-ish config uses `d_model=512`, 8 heads,
`head_dim=48`, `FLA_VALUE_EXPAND=2.0`, `FLA_ALLOW_NEG_EIGVAL=true`, and SP1024
so it can be compared against the exact-contract HGDN surface. Optional knobs
include `FLA_SHARE_QK`, `FLA_SHARE_KV`, `BIGRAM_HASH_BUCKETS`, and
`TRIGRAM_HASH_BUCKETS`.

Both native FLA configs default `FLA_STORAGE_DTYPE=fp32` for numerical sanity.
For speed calibration against bf16 custom HGDN, override
`FLA_STORAGE_DTYPE=bf16` explicitly. These configs remain pure-GDN calibration;
the custom HGDN 6G/2A config is the OLMo-style hybrid test.

Tiny smoke:

```bash
env COMPILE=0 VOCAB_SIZE=128 NUM_LAYERS=1 MODEL_DIM=64 NUM_HEADS=2 \
HEAD_DIM=32 MLP_MULT=2 FLA_VALUE_EXPAND=1 TRAIN_SEQ_LEN=32 \
SMOKE_SEQ_LEN=32 SMOKE_BATCH_SIZE=2 \
conda run -s --name pg python train_gpt_fla_control.py --smoke
```

Full calibration launch, after loading the TOML as environment:

```bash
set -a
source <(conda run -s --name pg python scripts/hgdn_helper_cli.py load-env \
  --path configs/fla/native_prlike_gdn10_d544_sp8192.toml)
set +a
torchrun --standalone --nproc_per_node=1 train_gpt_fla_control.py
```

This is calibration only. It must not change `HybridGPT` behavior or replace
the sparse HGDN finalist path unless its canonical score, artifact size, and
speed justify promotion.

Wrapper/direct/native timing ablation:

```bash
conda run -s --name pg python scripts/bench_fla_recurrence_paths.py --iters 20
```

Run this benchmark by itself. Timings are invalid if other tests, training jobs,
or CUDA benchmarks are active in background terminals.

## Analysis And Sanity Tools

- `scripts/analyze_local_naive_contract_bundle.py`: local sparse bundle ranking.
- `scripts/check_bpb_sanity.py`: nats/BPB/implied bytes-per-token checks.
- `scripts/probe_fla_stack.py`: FLA package/API/kernel import probe.
- `scripts/bench_fla_recurrence_paths.py`: wrapper versus direct public FLA
  recurrence timing, upstream-style direct fused FLA semantics timing, plus
  native `fla.layers.GatedDeltaNet` timing.
- `scripts/screen_hgdn_arch_sizes.py`: artifact-size screen for active configs.

Example BPB sanity check:

```bash
conda run -s --name pg python scripts/check_bpb_sanity.py \
  local-scratch/localnaivehgdn_sparse3_bundle/logs/localnaivehgdn_sparse3_l8_d512_mid2_dk48_m2.txt
```

## Kept Entrypoints

- `train_gpt.py`: exact repo baseline.
- `train_gpt_hybrid.py`: packed sparse HGDN and attention-only diagnostic path.
- `train_gpt_fla_control.py`: isolated native-FLA calibration path.
- `scripts/run_local_hgdn_naive_contract_search.sh`: local sparse search.
- `scripts/run_h100_hgdn_naive_contract_round.sh`: H100 exact-baseline comparison.
- `scripts/bundle_hgdn_run.py`: bundle helper.
- `scripts/bootstrap_challenge_data.sh`: data bootstrap helper.

Related docs:

- [TODO.md](TODO.md)
- [REFERENCE.md](REFERENCE.md)
- [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
