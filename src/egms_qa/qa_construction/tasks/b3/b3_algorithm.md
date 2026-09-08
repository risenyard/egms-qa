# B3: Velocity tails and direction

## Task overview

B3 describes the slow and fast ends of the point-velocity distribution in a tile. Negative vertical velocity means subsidence and positive velocity means uplift. A percentile marks a position in that distribution: p10 has 10% of values below it, and p90 has 90% below it.

| Task | Type | Output and relationship |
|---|---|---|
| B31 | Numeric value (mm/yr) | 10th percentile of point velocity: the lower, more downward-moving tail. |
| B32 | Numeric value (mm/yr) | 90th percentile of point velocity: the upper, more upward-moving tail. |
| B33 | Numeric value (mm/yr) | 90th percentile of absolute point velocity: fast-motion magnitude regardless of direction. |
| B34 | Classification | Uses B31 and B32 to label uplift when the upper tail exceeds the lower tail's magnitude; otherwise labels non-uplift. |
| B35 | Classification | Converts B33 into five motion-strength levels using fixed velocity cutoffs. |
| B36 | Classification | Places B33 in five levels relative to the European candidate-pool distribution. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) · [Implementation](b3_compute.py) · [Run and files](../README.md#run-b3)

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

B35 motion-strength class (stored as `worst_point_significance`):

```text
B35_worst_point_significance =
    very_low   if B33 < 1.5
    low        if 1.5 <= B33 < 1.9
    moderate   if 1.9 <= B33 < 2.24
    high       if 2.24 <= B33 < 2.9
    very_high  if B33 >= 2.9
```

Boundary comparisons use the released NPZ's float32 precision: a stored
1.9 mm/yr belongs to `moderate`. B35 uses velocity strength rather than a
point-specific uncertainty ratio.

B36 European reference class:

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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

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
