# C2: Spatial concentration

## Task overview

| task | description |
|---|---|
| **C2 group** | Measure how unevenly motion is distributed across the tile's spatial cells. |
| C21 | Spatial concentration score computed from the distribution of cell-level motion magnitudes. |
| C22 | Concentration class derived from C21 using the specified reference thresholds. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv) · [Implementation](c2_compute.py)

## Key concepts

Is the motion magnitude spatially concentrated in a few parts of the tile, or spread more evenly?

### Inputs

- `coords`: point coordinates.
- `mean_velocity`: point-level mean velocity in mm/yr.

Arrays are read from the EGMS encoder 10k tile manifest:

[HF split.parquet](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) (installed at `data/encoder/manifest/split.parquet`)

### Intentional Exclusions

- C21/C22 do not use RMSE or motion SNR. RMSE belongs to A41 and noise-aware point activity belongs to C11.
- C21/C22 do not count active bins; the old active spatial extent task was deleted.
- C21/C22 do not use a fixed `2 mm/yr` active threshold.
- C21/C22 do not classify moving-support location; C13 handles the C1-derived bin string.

## Algorithm steps

### Formula

For each point:

```text
point_motion_magnitude = abs(point mean_velocity)
```

For each tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split the tile into an 8x8 local grid.
3. Keep bins with at least 5 points.
4. For each valid bin:
   bin_mean_abs_velocity = mean(point_motion_magnitude)
5. C21_spatial_concentration_score = Gini(bin_mean_abs_velocity over valid bins)
```

Higher values mean the motion magnitude is more spatially concentrated.

### Distribution Class

C22 is derived from the observed C21 train-split distribution. The distribution
is unimodal and right-skewed with a long high-concentration tail, so C22 uses
train p10/p50/p90 rather than equal quartiles. It is a corpus-relative spatial
organization class:

| class | rule |
|---|---:|
| `diffuse` | C21 <= 0.093756 |
| `mildly_concentrated` | 0.093756 < C21 <= 0.181511 |
| `concentrated` | 0.181511 < C21 <= 0.312813 |
| `highly_concentrated` | C21 > 0.312813 |

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.c2.c2_compute \
    --out-dir outputs/tasks-rebuilt/c2
```

The new table is written to `outputs/tasks-rebuilt/c2/c2_final_table.csv`.
The installed reference remains at `outputs/tasks/c2/c2_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| C21 | 10,000 | 0 | 0.0784815 | 0.18154 | 0.362198 |

### C22 label distribution

| label | count | share |
|---|---:|---:|
| `mildly_concentrated` | 4,015 | 40.15% |
| `concentrated` | 3,998 | 39.98% |
| `highly_concentrated` | 1,003 | 10.03% |
| `diffuse` | 984 | 9.84% |
