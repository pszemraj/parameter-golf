# Data Workflows

This directory contains dataset download helpers and export scripts.

Local layout:

- `data/datasets/<dataset_name>/`
- `data/tokenizers/`
- `data/manifest.json`
- `data/docs_selected.jsonl`
- `data/docs_selected.source_manifest.json`

## Current 5090 Dataset

Maintained Core/Amplifier runs use the local SP1024 export:

```text
data/datasets/fineweb10B_sp1024
```

Serious 5090 runs expect:

```text
195 train shards
1 validation shard
19,473,201,340 train tokens
62,021,846 validation tokens
```

Check the local layout before long spec or trigram-memory builds:

```bash
conda run -s --name train python tools/check_dataset_shards.py \
  data/datasets/fineweb10B_sp1024 \
  --expected-train-files 195 \
  --expected-val-files 1
```

Validation shards are used only for evaluation, not for frozen statistics or
trigram-memory counts.

## Downloading Published Data

Download the cached FineWeb export for a tokenizer variant with:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

To fetch more training shards, pass `--train-shards`:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 180
```

The downloader is manifest-driven and can fetch only a prefix of train shards
from a larger published export. The current serious-run local dataset was built
from 195 train shards.

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 195
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split.
Training on the first `N` train shards means training on the prefix of the same
frozen shuffled export, so the data order stays aligned with the baseline for
that tokenizer family.

The default published repo is `willdepueoai/parameter-golf`, with the export
rooted under the repo subdirectory `datasets/`.

## Rebuilding Tokenizers From Published Docs

To retrain a tokenizer or re-export shards from exactly the same selected
documents, run the standalone retokenizer against the published docs cache:

```bash
python3 data/download_hf_docs_and_tokenize.py \
  --repo-id your-hf-username/your-dataset-repo \
  --remote-root your_50B_export_root \
  --output-root /tmp/my_custom_tokenizer_export \
  --tokenizer-config ./data/tokenizer_specs.json
```

The sidecar `docs_selected.source_manifest.json` includes `docs_sha256`, so
users can verify they are rebuilding from the exact same document list and order
as the baseline export.

## Useful Knobs

For CPU-heavy SentencePiece exports, useful knobs are:

```bash
MATCHED_FINEWEB_SP_BATCH_SIZE=2048
MATCHED_FINEWEB_TOKENIZER_THREADS=16
```

These control batched tokenizer encoding and tokenizer thread count.
