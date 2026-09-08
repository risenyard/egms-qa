# A4: Measurement noise

## Task overview

| task | description |
|---|---|
| **A4 group** | Summarize typical noise in the source point measurements. |
| A41 | Median of the finite point-level EGMS RMSE values within the tile, in millimeters. |
| A42 | Noise class assigned by comparing the median RMSE with fixed millimeter thresholds. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv) · [Implementation](a4_compute.py)

## Key concepts

A41 measures measurement noise level:

> Is the tile's typical EGMS point-level RMSE low enough for downstream monitoring?

## Algorithm steps

### Target

For each tile, read the EGMS point-level `rmse` column and compute:

```text
A41_median_rmse_mm = median(point_rmse_mm)
```

The median is used because A41 is meant to describe typical measurement noise,
not a few local outlier points.

### Classes

The class label uses fixed absolute RMSE thresholds in millimeters:

| class | rule | meaning |
|---|---:|---|
| `low_noise` | median RMSE < 1.0 mm | low typical noise |
| `moderate_noise` | 1.0 <= median RMSE < 1.5 mm | normal usable noise |
| `high_noise` | 1.5 <= median RMSE < 2.0 mm | elevated noise |
| `very_high_noise` | median RMSE >= 2.0 mm | high-noise tile; use caution |

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.a4.a4_compute \
    --out-path outputs/tasks-rebuilt/a4/a4_final_table.csv
```

The new table is written to `outputs/tasks-rebuilt/a4/a4_final_table.csv`.
The installed reference remains at `outputs/tasks/a4/a4_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |
| Data configuration | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) | `data/encoder/manifest/data_config.json` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| A41 | 10,000 | 0 | 0.6 | 1.1 | 1.9 |

### A42 label distribution

| label | count | share |
|---|---:|---:|
| `moderate_noise` | 4,303 | 43.03% |
| `low_noise` | 3,822 | 38.22% |
| `high_noise` | 1,493 | 14.93% |
| `very_high_noise` | 382 | 3.82% |
