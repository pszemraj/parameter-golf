# HGDN TODO

## 1. Run The Primary H100 Sparse Confirmation

Use the exact-baseline helper with:

- HGDN: `configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml`
- attention-only diagnostic control:
  `configs/hgdn/naive_contract_l8_d512_r0_m2.toml`
- exact comparator: direct `train_gpt.py` leg inside the helper

Command:

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

Promotion rule: sparse HGDN must improve the exact `train_gpt.py` comparison
after speed and artifact size are accounted for. Same-shell attention-only
wins are diagnostic, not leaderboard evidence.

## 2. Optional Quality-Ceiling Probe

Only run this if the primary H100 result leaves a real quality/speed tradeoff
unresolved:

```bash
HGDN_CONFIG=configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml
```

The local evidence says this is the fixed-step quality leader but slower enough
to lose the speed-aware HGDN rank.

## 3. Calibrate Native FLA After Source Reinstall

First probe the local stack:

```bash
conda run -s --name pg python scripts/probe_fla_stack.py
```

Then run only bounded smoke locally:

```bash
env COMPILE=0 VOCAB_SIZE=128 NUM_LAYERS=1 MODEL_DIM=64 NUM_HEADS=2 \
HEAD_DIM=32 MLP_MULT=2 FLA_VALUE_EXPAND=1 TRAIN_SEQ_LEN=32 \
SMOKE_SEQ_LEN=32 SMOKE_BATCH_SIZE=2 \
conda run -s --name pg python train_gpt_fla_control.py --smoke
```

H100 calibration, if requested, uses
`configs/fla/native_gdn10_d544_sp8192.toml` and `train_gpt_fla_control.py`.
Keep it isolated from `HybridGPT`.

## 4. Keep Hygiene Checks Green

Before handing off a run bundle or branch:

```bash
bash -n scripts/hgdn_shell_common.sh \
  scripts/run_local_hgdn_naive_contract_search.sh \
  scripts/run_h100_hgdn_naive_contract_round.sh \
  scripts/bootstrap_challenge_data.sh

conda run -s --name pg python -m py_compile \
  model.py train_gpt.py train_gpt_hybrid.py train_gpt_fla_control.py \
  hgdn_runtime_utils.py scripts/hgdn_helper_cli.py \
  scripts/screen_hgdn_arch_sizes.py \
  scripts/analyze_local_naive_contract_bundle.py \
  scripts/check_bpb_sanity.py scripts/probe_fla_stack.py

conda run -s --name pg ruff check --fix
conda run -s --name pg ruff format
conda run -s --name pg ruff check
git diff --check
```

Do not launch multi-run sweeps or non-trivial training unless the user requests
the exact run.
