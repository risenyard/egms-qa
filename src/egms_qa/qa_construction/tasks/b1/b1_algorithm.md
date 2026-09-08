# B1: Subsidence signal

## Task overview

| task | description |
|---|---|
| **B1 group** | Assess whether the observed downward motion is distinguishable from its reported uncertainty. |
| B11 | Average subsidence signal-to-noise ratio computed from point velocities and their uncertainties. |
| B12 | Clear-subsidence decision obtained by applying the specified SNR criterion to B11. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b1/b1_final_table.csv) · [Implementation](b1_compute.py)

## Key concepts

Does this tile show clear average subsidence relative to its observation noise?

### Inputs

- `mean_velocity`: point-level mean velocity in mm/yr.
- `rmse`: point-level RMSE/noise estimate.

Both arrays are read from the EGMS encoder 10k tile manifest:

[HF split.parquet](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) (installed at `data/encoder/manifest/split.parquet`)

### Intentional Exclusions

- B11/B12 do not classify uplift.
- B11/B12 do not use p10/p90 tail velocity.
- B11/B12 do not assign mild/moderate/strong severity; that belongs to the B/C/D derived monitoring layer.
- B11/B12 do not replace A41/A51 quality gates; it uses RMSE only to normalize this specific direction signal.

## Algorithm steps

### Formula

For each tile:

```text
v_mean = mean(point mean_velocity)
rmse_median = median(point rmse)
B11_subsidence_snr = -v_mean / (rmse_median + eps)
```

The minus sign makes stronger subsidence a larger positive value.

### Class Rule

```text
if B11_subsidence_snr >= 1.0:
    B12_clear_subsidence_class = clear_subsidence
else:
    B12_clear_subsidence_class = no_clear_subsidence
```

The threshold `1.0` is a signal-to-noise rule: the tile's average subsidence must be at least as large as the median point RMSE. It is not a European severity threshold and is not fitted from the corpus distribution.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.b1.b1_compute \
    --out-dir outputs/tasks-rebuilt/b1
```

The new table is written to `outputs/tasks-rebuilt/b1/b1_final_table.csv`.
The installed reference remains at `outputs/tasks/b1/b1_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b1/b1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B11 | 10,000 | 0 | 0.0807331 | 0.99613 | 2.1774 |

### B12 label distribution

| label | count | share |
|---|---:|---:|
| `no_clear_subsidence` | 5,029 | 50.29% |
| `clear_subsidence` | 4,971 | 49.71% |
