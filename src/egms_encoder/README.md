# Encoder (tile representation)

> 🤗 Released weights & token cache: [`risenyard/egms-qa-encoder`](https://huggingface.co/risenyard/egms-qa-encoder)

The EGMS encoder maps a variable-size tile of persistent-scatterer displacement
histories to a fixed 65-token representation used by the rest of EGMS-QA. It is a
self-supervised spatio-temporal model:

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

This directory vendors the encoder model and data code (`models/`, `data/`,
`train_tile_aware_v4.py`) so the released checkpoint can be loaded and tokens can
be extracted. Reproduction from raw tiles additionally needs the per-tile point
store from the EGMS encoder project, located via `EGMS_ENCODER_HOME` (see
`../egms_qa/paths.py`).

## Token extraction

```bash
# encoder checkpoint + split manifest come from the data release (data/encoder/)
python -m egms_encoder.extract_tokens \
    --checkpoint data/encoder/checkpoint/encoder.pt \
    --manifest   data/encoder/manifest/split.parquet \
    --output-dir outputs/tokens
# -> outputs/tokens/encoder_tokens_10k.pt   (spatial_tokens [10000, 65, 256], mask, ids, splits)
```

The released token cache (`data/encoder/tokens/encoder_tokens_10k.pt`) lets you
skip this step and train/evaluate the translator directly. Encoder provenance is
in `../../data/encoder/PROVENANCE.md`.
