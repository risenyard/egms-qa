# A1: Representation stability

## Task overview

| task | description |
|---|---|
| **A1 group** | Assess how stable the tile representation remains when observations are removed. |
| A11 | Mean normalized angular drift between the full-tile summary token and summaries from repeated point subsamples. |
| A12 | Representation stability class obtained by comparing A11 with train-fitted tail percentiles. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv) · [Implementation](a1_compute.py)

## Key concepts

A11 measures severe global representation instability:

> When a tile loses most observations, does the encoder still form a stable global tile representation?

## Algorithm steps

### Target

For each tile:

1. Use the cached full-tile EGMS summary token as `summary_full`.
2. Keep 20% of points with deterministic random seeds `0,1,2,3,4`.
3. Re-run the frozen EGMS encoder on each subsampled tile, using the original full-tile center.
4. Compute per-seed angular drift:

```text
drift_seed = arccos(clip(cosine(summary_full, summary_sub), -1, 1)) / pi
```

5. Average across seeds:

```text
A11_global_angular_drift = mean_seed(drift_seed)
```

Lower drift means the global representation is more stable. A11 does not use coverage score, patch-token similarity, cluster consistency, scalar reconstruction, or manual coverage thresholds.

### Four Classes

The continuous target is primary. The four-class label is a corpus-relative tail
label. Thresholds are fitted on the train split only and then applied to all 10k
tiles:

| class | rule | threshold |
|---|---|---|
| `stable` | drift <= train p75 | <= 0.007791 |
| `mildly_sensitive` | train p75 < drift <= train p95 | <= 0.011695 |
| `highly_sensitive` | train p95 < drift <= train p99 | <= 0.016930 |
| `extreme` | drift > train p99 | > 0.016930 |

The same thresholds in degrees are 1.402, 2.105, and 3.047 degrees.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.a1.a1_compute \
    --out-dir outputs/tasks-rebuilt/a1/work/shards/shard_0 \
    --num-shards 1 --shard-index 0 --device cuda:0
python -m egms_qa.qa_construction.tasks.a1.a1_combine_shards \
    --base-dir outputs/tasks-rebuilt/a1/work --num-shards 1 \
    --out-path outputs/tasks-rebuilt/a1/a1_final_table.csv
```

This example uses one shard and CUDA. For multiple shards, use matching
`--num-shards` values in the compute and combine commands.

The new table is written to `outputs/tasks-rebuilt/a1/a1_final_table.csv`.
The installed reference remains at `outputs/tasks/a1/a1_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |
| Data configuration | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) | `data/encoder/manifest/data_config.json` |
| Encoder token cache | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/representations/egms_tokens_10k.pt) | `data/encoder/tokens/egms_tokens_10k.pt` |
| Encoder weights | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/encoder.safetensors) | `data/encoder/checkpoint/encoder.safetensors` |
| Encoder configuration | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/config.json) | `data/encoder/checkpoint/config.json` |
| Encoder normalization | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/normalization.json) | `data/encoder/checkpoint/normalization.json` |

Intermediate files are written under `work/shards/`. The combiner removes
that work directory by default. Add `--keep-work` to retain those local files.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| A11 | 10,000 | 0 | 0.00329956 | 0.00586604 | 0.0116798 |

### A12 label distribution

| label | count | share |
|---|---:|---:|
| `stable` | 7,472 | 74.72% |
| `mildly_sensitive` | 2,032 | 20.32% |
| `highly_sensitive` | 395 | 3.95% |
| `extreme` | 101 | 1.01% |
