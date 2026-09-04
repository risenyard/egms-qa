# EGMS-QA data release

The structured Hugging Face release is
[`risenyard/egms-qa-dataset`](https://huggingface.co/datasets/risenyard/egms-qa-dataset).
It separates user-facing QA from large experiment artifacts:

```text
data/qa/                    Hugging Face-loadable QA splits
artifacts/source_tiles/     10,000 processed EGMS tiles (NPZ)
artifacts/representations/  frozen 65×256 token cache
artifacts/labels/           canonical task labels
artifacts/reference_tables/ deterministic task-family tables
metadata/                   split, provenance, audit, and SHA256 records
```

The default Hugging Face configuration loads only `data/qa/`; it does not pull
the 7.5 GiB source-tile store.

## Source scope

The encoder uses EGMS Level-3 Ortho Vertical (`U`) measurements for 2019–2023.
The release contains 10,000 geographically sampled 7 km tiles selected from
131,281 prepared candidates. The fixed split is 8,000 train, 1,000 validation,
and 1,000 test tiles.

Each tile contains 304 prepared time indices. The encoder consumes `[8,302)`,
i.e. 294 steps. Tiles overlap, so a persistent-scatterer `pid` can intentionally
occur in multiple tiles. The files are processed source tiles, not untouched
official EGMS archives.

## Install the full release

```bash
hf download risenyard/egms-qa-dataset \
    --repo-type dataset --local-dir release/egms-qa-dataset
python -m egms_qa.release audit \
    --release-dir release/egms-qa-dataset
python -m egms_qa.release install \
    --release-dir release/egms-qa-dataset --target-root .
```

The installer creates non-destructive symbolic links into the runtime paths
expected by the code. Use `audit --verify-hashes` for a complete byte-level
check.

## Attribution

European Union's Copernicus Land Monitoring Service information;
https://doi.org/10.2909/4a14a29b-7db7-40e4-81ad-df3aa8dfbc6f

See [`../DATA_LICENSE`](../DATA_LICENSE) for the modification and
non-endorsement notices.
