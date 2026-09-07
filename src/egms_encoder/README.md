# Encoder (tile representation)

> 🤗 Released weights & token cache: [`risenyard/egms-qa-encoder`](https://huggingface.co/risenyard/egms-qa-encoder)

The EGMS encoder maps a variable-size tile of persistent-scatterer displacement
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

This directory vendors the encoder model and data code (`models/`, `data/`,
`pretrain.py`). Loading the checkpoint, extracting tokens, and
**retraining the encoder from scratch** are all self-contained on the released
data: the model-ready 294-step EGMS tiles ship as NPZ under
`artifacts/source_tiles/` in `risenyard/egms-qa-dataset`; the release installer
links them to `data/tiles/`. The split manifest and normalization ship with the
code/model release. The encoder was trained on this 10k tile set's train split.
The release has no external dependency. (One family-C4 threshold is
corpus-relative, derived from the full European candidate pool, but its value
ships in the C4 reference JSON, so re-deriving it is optional and never required
to reproduce the release.)

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

From the published GitHub checkout, identify both published HF repositories
directly. The command resolves and pins their current revisions in the HF cache:

```bash
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --output-dir outputs/tokens
# -> outputs/tokens/egms_tokens_10k.pt   (spatial_tokens [10000, 65, 256], mask, ids, splits)
```

The repository mode reads `encoder.safetensors`, `config.json`, and
`normalization.json` from the published Encoder tree. It reads
`metadata/{split_manifest.parquet,data_config.json}` and
`artifacts/source_tiles/` from the published Dataset tree.

For another compatible local NPZ collection, provide `--checkpoint`,
`--model-config`, and `--normalization` together, plus `--manifest` and
`--data-config`. Use `--source-tiles-root` when manifest paths are relative to a
separate tile root.

Compatibility requires consistent displacement units (mm), vertical component,
temporal sampling and preprocessing, and metric point coordinates appropriate
for the 7 km tile geometry. Having 294 columns alone does not establish this.
Keep the released normalization when applying the released frozen encoder;
do not refit it on each inference collection. Distribution shifts require
validation. For a new encoder trained on another corpus, fit normalization on
that corpus's training split and pair it with the new checkpoint. Auxiliary
NPZ fields can affect QA construction even though encoder inference uses
displacement histories and coordinates.

The released token cache (`data/encoder/tokens/egms_tokens_10k.pt`) lets you
skip this step and train/evaluate the translator directly. Encoder provenance is
documented in the
[dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset) and
[encoder card](https://huggingface.co/risenyard/egms-qa-encoder).

To reproduce the published encoder training recipe, first install the Dataset
as described in the top-level README, then download the model configuration
and run the recipe. The runner reads architecture, masking, sampling, loss,
optimizer, and validation settings from the downloaded JSON files:

```bash
hf download risenyard/egms-qa-encoder --include '*.json' \
    --local-dir release/egms-qa-encoder
python -m egms_qa.reproduce encoder \
    --model-dir release/egms-qa-encoder --output-dir outputs/encoder_pretrain
```

Add `--dry-run` to inspect the resolved command. This starts training from
scratch; weights from the released encoder are not loaded. The generic
`egms_encoder.pretrain` defaults are not the published training recipe.
