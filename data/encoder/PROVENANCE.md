# EGMS encoder artifacts

This folder holds the coordinate-normalized EGMS encoder used by EGMS-QA and its
lightweight metadata. The heavy files (checkpoint, token cache) are not in the
git repository; download them from the data release and place them here.

## Files

Tracked in git (lightweight):
- `checkpoint/args.json` — training/config arguments paired with the checkpoint.
- `checkpoint/normalization.json` — normalization statistics paired with the checkpoint.
- `manifest/data_config.json` — data configuration reference.
- `manifest/split.parquet` — the 10,000-tile train/val/test split manifest.

From the data release (git-ignored, place here):
- `checkpoint/encoder.pt` — the frozen encoder checkpoint.
- `tokens/encoder_tokens_10k.pt` — token cache for the 10,000 EGMS-QA tiles,
  with `spatial_tokens` of shape `(10000, 65, 256)`, a validity mask, tile ids,
  and split labels.
- `tokens/encoder_tokens_10k_metadata.json` — extraction metadata for the cache.

## Notes

- The encoder is trained with `coord_scale = 3500`; the token cache records the
  paired checkpoint and normalization in its metadata.
- The processed per-tile point store ships in the dataset repo under
  `artifacts/source_tiles/`; `python -m egms_qa.release install` links it to
  `data/tiles/` so recomputing tokens or task reference values is self-contained.
  (Only re-deriving the family-C4 corpus threshold needs the full European pool,
  which is not shipped; its value ships in the C4 reference JSON.)
- The underlying EGMS Level-3 product is Copernicus data; see `../METADATA.md`.
