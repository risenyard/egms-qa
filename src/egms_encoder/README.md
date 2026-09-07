# EGMS-QA Encoder

> 🤗 [Encoder weights](https://huggingface.co/risenyard/egms-qa-encoder) ·
> [tiles and token cache](https://huggingface.co/datasets/risenyard/egms-qa-dataset)

The EGMS-QA Encoder maps a variable number of persistent-scatterer histories
from one 7 km tile to a fixed representation. Each point contributes a
294-step displacement history and centred coordinates. A temporal Transformer
and spatial Transformer produce one 256-dimensional embedding per point;
deterministic 8×8 pooling produces 65 tokens: one tile summary followed by 64
row-major spatial cells.

The package contains only the public model, data-contract, training, loading,
and token-extraction code. It does not depend on another source checkout,
private paths, or QA task implementations.

## Direct inference from Hugging Face

From an installed GitHub checkout, the extractor can resolve both published HF
repositories itself. It resolves `main` to immutable revisions before
downloading files:

```bash
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --output-dir outputs/tokens \
    --device cuda:0
```

Use `--max-tiles 1` for a small smoke run. The full release writes:

```text
outputs/tokens/
├── egms_tokens_10k.pt
└── egms_tokens_10k_metadata.json
```

The tensor payload contains `spatial_tokens [10000,65,256]`, `token_mask`, tile
IDs, split labels, per-cell point counts, and reproducibility metadata.

## Compatible local inputs

Local model inputs are atomic: provide the checkpoint, model config, and
normalization together. Local data inputs likewise require a manifest and data
config together:

```bash
python -m egms_encoder.extract_tokens \
    --checkpoint /path/to/encoder.safetensors \
    --model-config /path/to/config.json \
    --normalization /path/to/normalization.json \
    --manifest /path/to/split_manifest.parquet \
    --data-config /path/to/data_config.json \
    --source-tiles-root /path/to/artifacts/source_tiles \
    --output-dir outputs/tokens \
    --device cuda:0
```

`--source-tiles-root` is needed only when relative manifest paths do not resolve
from the working directory. Local inputs are recorded by current file hashes;
they are not labelled as official EGMS-QA repositories.

## Install the released data for training

The Encoder has its own data installer and does not require the QA installer:

```bash
hf download risenyard/egms-qa-dataset \
    --repo-type dataset --local-dir release/egms-qa-dataset
python -m egms_encoder.install_data \
    --release-dir release/egms-qa-dataset --target-root .
hf download risenyard/egms-qa-encoder \
    --local-dir data/encoder/checkpoint
```

This creates only the runtime paths used by the Encoder: tiles, manifest, data
config, and the optional released token cache.

## Training

Reproduce pretraining with the released architecture and recipe:

```bash
python -m egms_encoder.pretrain \
    --output-dir outputs/my_encoder \
    --device cuda:0
```

The output contains both resumable training state and a directly reusable
inference bundle:

```text
outputs/my_encoder/
├── best.safetensors
├── best.pt
├── latest.pt
├── config.json
├── training_args.json
├── normalization.json
├── run_args.json
└── metrics.csv
```

Resume training with the generated files:

```bash
python -m egms_encoder.pretrain \
    --model-config outputs/my_encoder/config.json \
    --training-args outputs/my_encoder/training_args.json \
    --normalization outputs/my_encoder/normalization.json \
    --resume-from outputs/my_encoder/latest.pt \
    --output-dir outputs/my_encoder \
    --device cuda:0
```

Use the best trained model without conversion code:

```bash
python -m egms_encoder.extract_tokens \
    --checkpoint outputs/my_encoder/best.safetensors \
    --model-config outputs/my_encoder/config.json \
    --normalization outputs/my_encoder/normalization.json \
    --manifest data/encoder/manifest/split.parquet \
    --data-config data/encoder/manifest/data_config.json \
    --output-dir outputs/my_tokens \
    --device cuda:0
```

`latest.pt` is for `--resume-from`; `best.safetensors` is pure model state for
strict inference loading.

## Data boundary

The public runtime accepts the EGMS-QA NPZ contract with stored
`time_series [N,294]`, schema `egms-qa-data-config-1.1`, and stored window
`[0,294)`. This corresponds scientifically to indices `[8,302)` on the
304-step prepared source axis. The code deliberately rejects implicit or
repeated cropping.

It does not authenticate to the official EGMS service, convert arbitrary
ZIP/CSV products, choose a valid time window, or estimate normalization for a
new reference period. A different product version must be prepared and audited
before it is supplied to this Encoder.
