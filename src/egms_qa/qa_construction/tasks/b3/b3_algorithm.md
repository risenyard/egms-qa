# B3: Velocity tails and direction

## Task overview

| task | description |
|---|---|
| **B3 group** | Describe the velocity distribution's tails, direction, significance, and European reference typicality. |
| B31 | 10th percentile of point velocity, describing the sinking tail. |
| B32 | 90th percentile of point velocity, describing the upper tail. |
| B33 | 90th percentile of absolute point velocity, describing the magnitude of the fast tail. |
| B34 | Direction class that retains evidence of uplift instead of treating every strong tail as subsidence. |
| B35 | Significance class comparing extreme point motion with its measurement uncertainty. |
| B36 | Velocity typicality class relative to the specified European reference distribution. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) · [Implementation](b3_compute.py)

## Key concepts

What do the tile's velocity tails say about sinking-side motion, uplift protection, local worst-point strength, and European velocity typicality?

## Algorithm steps

### Targets

```text
B31_velocity_p10_mm_yr = percentile(point mean_velocity, 10)
B32_velocity_p90_mm_yr = percentile(point mean_velocity, 90)
B33_vel_abs_p90_mm_yr = percentile(abs(point mean_velocity), 90)
```

B31 captures the sinking-side tail. B32 captures the upper-side tail. B33 captures local absolute motion strength.

### Derived Labels

Uplift-protected direction:

```text
B34_uplift_protected_direction = uplift
    if B32_velocity_p90_mm_yr > abs(B31_velocity_p10_mm_yr)
    else non_uplift
```

Worst-point significance:

```text
B35_worst_point_significance =
    very_low   if B33 < 1.5
    low        if 1.5 <= B33 < 1.9
    moderate   if 1.9 <= B33 < 2.24
    high       if 2.24 <= B33 < 2.9
    very_high  if B33 >= 2.9
```

European velocity typicality:

```text
B36_european_velocity_typicality =
    low       if B33 <= 1.367
    typ_low   if 1.367 < B33 <= 2.075
    typ_high  if 2.075 < B33 <= 3.149
    high      if 3.149 < B33 <= 4.781
    extreme   if B33 > 4.781
```

B36 is a European reference distribution label, not a causal anomaly claim. The
cutoffs are **corpus-relative**: fixed quantiles of the full European
candidate-pool distribution, baked into the compute script as constants.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.b3.b3_compute \
    --out-dir outputs/tasks-rebuilt/b3
```

The new table is written to `outputs/tasks-rebuilt/b3/b3_final_table.csv`.
The installed reference remains at `outputs/tasks/b3/b3_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B31 | 10,000 | 0 | -4.243 | -1.9 | -0.8 |
| B32 | 10,000 | 0 | -1.1 | -0.3 | 1 |
| B33 | 10,000 | 0 | 0.9 | 1.9 | 4.4 |

### B34 label distribution

| label | count | share |
|---|---:|---:|
| `non_uplift` | 9,664 | 96.64% |
| `uplift` | 336 | 3.36% |

### B35 label distribution

| label | count | share |
|---|---:|---:|
| `very_low` | 2,307 | 23.07% |
| `low` | 2,206 | 22.06% |
| `moderate` | 2,022 | 20.22% |
| `very_high` | 1,752 | 17.52% |
| `high` | 1,713 | 17.13% |

### B36 label distribution

| label | count | share |
|---|---:|---:|
| `typ_low` | 3,768 | 37.68% |
| `typ_high` | 3,056 | 30.56% |
| `low` | 1,857 | 18.57% |
| `high` | 922 | 9.22% |
| `extreme` | 397 | 3.97% |
