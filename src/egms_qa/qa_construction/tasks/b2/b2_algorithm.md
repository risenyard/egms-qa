# B2: Mean motion

## Task overview

| task | description |
|---|---|
| **B2 group** | Describe mean vertical motion and its subsidence intensity while preserving uplift cases. |
| B21 | Mean of the valid point velocities in the tile, in millimeters per year. |
| B22 | Mean-subsidence intensity band with an uplift-protected direction rule. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv) · [Implementation](b2_compute.py)

## Key concepts

What is the tile's average velocity, and which European relative subsidence intensity band does it fall into?

### Interpretation

B22 is a corpus-relative European intensity band. It is not a legal, physical, or causal severity class. Uplift is kept as a separate direction override.

## Algorithm steps

### Targets

```text
B21_mean_velocity_mm_yr = mean(point mean_velocity)
```

B22 is derived from B21 and the uplift-protected direction rule:

```text
if B34_uplift_protected_direction == uplift:
    B22_mean_subsidence_intensity_band = uplift
elif B21_mean_velocity_mm_yr <= -1.47:
    B22_mean_subsidence_intensity_band = high
elif B21_mean_velocity_mm_yr <= -1.215:
    B22_mean_subsidence_intensity_band = high_mid
elif B21_mean_velocity_mm_yr <= -0.971:
    B22_mean_subsidence_intensity_band = mid
elif B21_mean_velocity_mm_yr <= -0.657:
    B22_mean_subsidence_intensity_band = low_mid
else:
    B22_mean_subsidence_intensity_band = low
```

The band cutoffs are **corpus-relative**: fixed quantiles of the full European
candidate-pool velocity distribution, baked into the compute script as constants.

The B34 direction used here is copied into the B2 final table as an upstream explanation column so B22 is reproducible within the folder:

```text
B34_uplift_protected_direction = uplift
    if B32_velocity_p90_mm_yr > abs(B31_velocity_p10_mm_yr)
    else non_uplift
```

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.b2.b2_compute \
    --out-dir outputs/tasks-rebuilt/b2
```

The new table is written to `outputs/tasks-rebuilt/b2/b2_final_table.csv`.
The installed reference remains at `outputs/tasks/b2/b2_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B21 | 10,000 | 0 | -2.07322 | -1.09149 | -0.095449 |

### B22 label distribution

| label | count | share |
|---|---:|---:|
| `high` | 2,013 | 20.13% |
| `high_mid` | 1,952 | 19.52% |
| `low` | 1,914 | 19.14% |
| `mid` | 1,906 | 19.06% |
| `low_mid` | 1,879 | 18.79% |
| `uplift` | 336 | 3.36% |
