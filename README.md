# Parameter Golf

OpenAI Model Craft Challenge: Parameter Golf is a language-model compression
challenge. The target artifact is at most `16,000,000` decimal bytes, training
must fit the 10-minute `8xH100 SXM` contract for record submissions, and quality
is evaluated on FineWeb validation compression in bits per byte.

This branch is focused on the HGDN finalist path. Historical leaderboard
snapshots and old `records/` submissions were removed from this branch; use Git
history when old submission artifacts are needed.

## Active Entrypoints

- `train_gpt.py`: exact repo baseline and absolute comparator.
- `train_gpt_hybrid.py`: packed sparse HGDN trainer and attention-only
  diagnostic control.
- `train_gpt_fla_control.py`: isolated native `fla.layers.GatedDeltaNet`
  calibration path.
- `train_gpt_mlx.py`: small Apple Silicon starter path from the original
  challenge repo.

For the HGDN branch state, run commands, and next experiment order, see
[docs/README.md](docs/README.md).

## Setup

Install dependencies in the environment used for this checkout:

```bash
python -m pip install -r requirements.txt
```

Download the cached FineWeb `sp1024` assets:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

This populates:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`

More data options are described in [data/README.md](data/README.md).

## Baseline Smoke

Run the exact baseline on a single CUDA worker:

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

`train_gpt.py` keeps its wallclock cap by default. For fixed-step local
debugging, override the relevant environment variables explicitly, for example
`MAX_WALLCLOCK_SECONDS=0`.

## HGDN Workflow

The sparse HGDN helpers live under `scripts/`:

```bash
bash scripts/run_local_hgdn_naive_contract_search.sh
bash scripts/run_h100_hgdn_naive_contract_round.sh
```

The H100 helper runs three legs: exact `train_gpt.py` baseline, config-driven
sparse HGDN, and the matched attention-only diagnostic control. The attention-only
control is useful for same-shell diagnosis; it is not a replacement for the exact
baseline.

## Artifact And Evaluation Rules

- The artifact cap is `16,000,000` total bytes, not `16 MiB`.
- Counted artifact bytes are code bytes plus compressed model bytes.
- Evaluation must not download external data, inspect validation tokens before
  scoring them, or rely on network access.
- Training and evaluation have separate 10-minute `8xH100` limits for record
  submissions.
- Imported packages are allowed when they do not violate the compute, artifact,
  or evaluation rules; keep runtime dependencies in `requirements.txt`.

## Docs

- [docs/README.md](docs/README.md): HGDN branch status, commands, and next runs.
- [docs/REFERENCE.md](docs/REFERENCE.md): OLMo Hybrid and GDN architecture notes.
- [docs/WANDB_SCHEMA.md](docs/WANDB_SCHEMA.md): W&B logging schema for HGDN runs.
- [data/README.md](data/README.md): dataset and tokenizer workflows.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): upstream attribution.

## Support

Use the OpenAI Discord Parameter Golf channels for challenge discussion:
`#parameter-golf-discussions` and `#parameter-golf-announcements`.
