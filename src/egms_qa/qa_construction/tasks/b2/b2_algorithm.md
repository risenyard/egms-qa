# B2: Mean motion

## Task overview

B2 describes average vertical motion across the observation points in a tile. It reports the mean velocity and a subsidence-intensity band, while retaining a separate label when upward motion dominates the velocity tails.

| Task | Type | Output and relationship |
|---|---|---|
| B21 | Numeric value (mm/yr) | Mean point velocity. Negative values indicate subsidence; positive values indicate uplift. |
| B22 | Classification | Converts B21 into a corpus-relative subsidence band, with a separate uplift label when the upper velocity tail dominates. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv) · [Implementation](b2_compute.py) · [Run and files](../README.md#run-b2)

The subsidence-band cutoffs are corpus-relative quantiles from the European candidate pool. The uplift rule is calculated directly from the same tile’s velocity percentiles, so B2 does not need a B3 input table.

## Algorithm steps

### Targets

```text
B21_mean_velocity_mm_yr = mean(point mean_velocity)
```

Before classifying B21, compute the 10th and 90th percentiles of the same point
velocities. The rule labels uplift if the upper percentile is larger than the
absolute lower percentile:

```text
velocity_p10 = percentile(point mean_velocity, 10)
velocity_p90 = percentile(point mean_velocity, 90)
direction = uplift if velocity_p90 > abs(velocity_p10) else non_uplift
```

This is also the [B34 direction rule](../b3/b3_algorithm.md). B2 stores its locally
computed result as `B34_uplift_protected_direction` for traceability.
B22 then applies the uplift label or a subsidence band:

```text
if direction == uplift:
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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

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
