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
`train_tile_aware.py`). Loading the checkpoint, extracting tokens, and
**retraining the encoder from scratch** are all self-contained on the released
data: the raw tiles (`data/tiles/`), split manifest and normalization ship with
the dataset repo, and the encoder was trained on this 10k tile set's train split.
The release has no external dependency. (One family-C4 threshold is
corpus-relative, derived from the full European candidate pool, but its value
ships in the C4 reference JSON, so re-deriving it is optional and never required
to reproduce the release.)

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
