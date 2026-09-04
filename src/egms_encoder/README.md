# EGMS-QA Encoder

> 🤗 [Encoder weights](https://huggingface.co/risenyard/egms-qa-encoder) ·
> [tiles and token cache](https://huggingface.co/datasets/risenyard/egms-qa-dataset)

The EGMS-QA Encoder maps a variable-size tile of persistent-scatterer displacement
histories to a fixed 65-token representation used by the rest of EGMS-QA. It is a
self-supervised spatio-temporal model:

![EGMS Encoder framework](../../docs/assets/egms-encoder.png)

- each point's 294-step history is normalized and split into 37 eight-step
  temporal patches; a temporal Transformer + mean pooling gives one temporal
  feature per point;
- point coordinates (relative to the tile centre, scaled by the tile half-width)
  are projected and added;
- a spatial Transformer exchanges information across points, producing one
  256-d contextual feature per point;
- training is masked reconstruction: a synchronized block hides the same 30%
  interval in every point history within a tile, recovered from the contextual
  features.

At inference the encoder is frozen and applied without masking. A deterministic
pooling step assigns points to an 8×8 grid and mean-pools features per cell,
yielding 65 tokens (1 tile summary + 64 cells) with a validity mask.

This directory contains the encoder model and data code (`models/`, `data/`,
`pretrain.py`). Loading the checkpoint, extracting tokens, and
**retraining the encoder from scratch** are all self-contained on the released
data: the model-ready 294-step EGMS tiles ship as NPZ under
`artifacts/source_tiles/` in `risenyard/egms-qa-dataset`; the release installer
links them to `data/tiles/`. The split manifest and normalization ship with the
HF repositories. The encoder was trained on this 10k tile set's train split.
The code does not depend on a private repository, another source checkout, or
a machine-specific path.

## Data support boundary

The public encoder code consumes the EGMS-QA NPZ tile contract together with a
split manifest, data config, and normalization file. It supports reproducing
the released encoder and training or inference on already prepared compatible
tiles. The released NPZ files store `[N,294]` displacement arrays and the
encoder reads `[0,294)` directly; this is the same physical window as
`[8,302)` on the original 304-step prepared axis. It does not download official EGMS products, convert arbitrary EGMS
ZIP/CSV releases, or infer a valid time window and normalization for another
reference period. New product versions require a separate, empirically audited
preparation step before this encoder entrypoint can be used.

## Token extraction

```bash
# encoder checkpoint + split manifest come from the data release (data/encoder/)
python -m egms_encoder.extract_tokens \
    --checkpoint data/encoder/checkpoint/encoder.safetensors \
    --model-config data/encoder/checkpoint/config.json \
    --normalization data/encoder/checkpoint/normalization.json \
    --manifest   data/encoder/manifest/split.parquet \
    --data-config data/encoder/manifest/data_config.json \
    --output-dir outputs/tokens
# -> outputs/tokens/egms_tokens_10k.pt   (spatial_tokens [10000, 65, 256], mask, ids, splits)
```

The released token cache (`data/encoder/tokens/egms_tokens_10k.pt`) lets you
skip this step and train/evaluate the translator directly. Encoder provenance is
documented in the
[dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset) and
[encoder card](https://huggingface.co/risenyard/egms-qa-encoder).

Install the structured dataset before pretraining or token extraction:

```bash
hf download risenyard/egms-qa-dataset \
    --repo-type dataset --local-dir release/egms-qa-dataset
python -m egms_encoder.install_data \
    --release-dir release/egms-qa-dataset --target-root .
hf download risenyard/egms-qa-encoder \
    --local-dir data/encoder/checkpoint
```

## Training

Reproduce pretraining with the released recipe:

```bash
python -m egms_encoder.pretrain \
    --output-dir outputs/my_encoder \
    --device cuda:0
```

The output contains a reusable inference bundle and resumable training state:

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

Resume an interrupted run without writing conversion code:

```bash
python -m egms_encoder.pretrain \
    --model-config outputs/my_encoder/config.json \
    --training-args outputs/my_encoder/training_args.json \
    --normalization outputs/my_encoder/normalization.json \
    --resume-from outputs/my_encoder/latest.pt \
    --output-dir outputs/my_encoder \
    --device cuda:0
```

Use the trained encoder directly for token extraction:

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

`latest.pt` is for `--resume-from`; `best.safetensors` is for inference. Users
do not need to write conversion code between training and token extraction.

Token metadata records input hashes automatically. Add
`--encoder-repository` and `--dataset-repository` only when those repository
identifiers are true provenance for the supplied files; custom inputs are not
labelled as official EGMS-QA artifacts by default.
