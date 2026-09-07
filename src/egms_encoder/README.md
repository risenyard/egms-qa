# EGMS-QA Encoder

The encoder converts displacement time series within each 7 km tile into
65 tokens: one tile-summary token and 64 spatial-cell tokens. Each token has
256 dimensions.

[Project guide](../../README.md) ·
[Model and training recipe](https://huggingface.co/risenyard/egms-qa-encoder) ·
[Dataset and token cache](https://huggingface.co/datasets/risenyard/egms-qa-dataset)

## Extract tokens

Start with one tile. Run the following commands from the repository root.

```bash
pip install -e .
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --max-tiles 1 \
    --output-dir outputs/tokens
```

The command downloads the released encoder and one tile, then saves
`outputs/tokens/egms_tokens_1.pt` with token shape `[1,65,256]`. The outputs
include validity masks, tile IDs, and metadata recording the model and dataset
revisions. Add `--device cpu` for a CPU run.

Remove `--max-tiles 1` to process all 10,000 tiles and write
`outputs/tokens/egms_tokens_10k.pt` with shape `[10000,65,256]`. GPU execution
is recommended for the full dataset.

The Dataset also provides a precomputed token cache. After installing the
Dataset with `egms_encoder.install_data`, the cache is available at
`data/encoder/tokens/egms_tokens_10k.pt`. Translator use also requires the QA
labels and task tables, installed through the
[Translator guide](../egms_qa/translator/README.md).

## Input requirements

| input | released contract |
|---|---|
| displacement | vertical displacement in mm, stored as `[N,294]` |
| coordinates | EPSG:3035 easting and northing in meters, `[N,2]` |
| tile geometry | 7 km side length, with a variable number of points |
| time axis | stored `[0,294)`, corresponding to source indices `[8,302)` |
| preprocessing | normalization paired with the encoder checkpoint |

The data config records the six-day cadence and source index offset.
Coordinates are centered within each tile and scaled using the model config.

New collections must match the input requirements above. Keep the released
normalization when using the frozen encoder, and check its performance on the
new data. When training a new encoder, fit normalization on the new training
split and save it with the checkpoint.

## Use local inputs

For local tiles, provide `--manifest` and `--data-config` together. Use
`--source-tiles-root` to resolve relative tile paths against a different
directory. Encoder inference uses displacement histories and coordinates.

For a local checkpoint, provide `--checkpoint`, `--model-config`, and
`--normalization` together. The token-extraction example below shows how to
use a checkpoint produced by training.

## Reproduce training

The encoder provides a data installer that links the tiles, manifests, and
token cache into its runtime paths. Download the Dataset and training files:

```bash
hf download risenyard/egms-qa-dataset --repo-type dataset \
    --local-dir release/egms-qa-dataset
python -m egms_encoder.install_data \
    --release-dir release/egms-qa-dataset --target-root .
hf download risenyard/egms-qa-encoder --include '*.json' \
    --local-dir data/encoder/checkpoint
python -m egms_encoder.pretrain \
    --output-dir outputs/my_encoder --device cuda:0
```

Training reads the architecture, recipe, and normalization from
`data/encoder/checkpoint/`. It starts from scratch with the released
train-fitted normalization. Command-line overrides support custom experiments.

The output directory contains `best.safetensors` for inference, `latest.pt`
for resuming training, and the corresponding `config.json`,
`training_args.json`, and `normalization.json`. Resume with those files:

```bash
python -m egms_encoder.pretrain \
    --model-config outputs/my_encoder/config.json \
    --training-args outputs/my_encoder/training_args.json \
    --normalization outputs/my_encoder/normalization.json \
    --resume-from outputs/my_encoder/latest.pt \
    --output-dir outputs/my_encoder --device cuda:0
```

To extract tokens with the trained encoder, pass its inference bundle and the
installed data contract:

```bash
python -m egms_encoder.extract_tokens \
    --checkpoint outputs/my_encoder/best.safetensors \
    --model-config outputs/my_encoder/config.json \
    --normalization outputs/my_encoder/normalization.json \
    --manifest data/encoder/manifest/split.parquet \
    --data-config data/encoder/manifest/data_config.json \
    --output-dir outputs/my_tokens --device cuda:0
```

## Architecture

![EGMS Encoder framework](../../docs/assets/egms-encoder.png)

Each normalized history is divided into temporal patches. A temporal
Transformer and mean pooling form a 256-dimensional representation for each
point. A coordinate embedding is added before spatial attention exchanges
information across points. Pretraining reconstructs a synchronized masked
interval shared by all points in a tile. Inference uses the frozen encoder without masking and pools
the point features into an 8×8 grid.

| module | purpose |
|---|---|
| `models/tile_encoder.py` | temporal and spatial encoder |
| `data/tile_store.py` | NPZ loading and stored-window validation |
| `checkpoint.py` | model configuration and weight loading |
| `extract_tokens.py` | encoding and spatial pooling |
| `pretrain.py` | masked-reconstruction training |

## Scope

The encoder consumes prepared EGMS-QA tiles. Downloading official EGMS products,
converting their source formats, and selecting a valid window for another
reference period require a separate data-preparation workflow. Its
representations describe observed deformation histories and do not establish
causes, predict future motion, or certify structural safety.
