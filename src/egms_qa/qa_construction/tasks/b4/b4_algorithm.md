# B4: Acceleration strength

## Task overview

B4 summarizes strong acceleration across the observation points in a tile. Acceleration measures change in velocity; its absolute value measures the size of that change regardless of direction.

| Task | Type | Output and relationship |
|---|---|---|
| B41 | Numeric value (mm/yr²) | 90th percentile of absolute point acceleration: the stronger end of changes in velocity. |
| B42 | Classification | Places B41 in five levels relative to the European candidate-pool distribution. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) · [Implementation](b4_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Targets

```text
B41_acc_abs_p90 = percentile(abs(point acceleration), 90)
```

B42 is derived from B41:

```text
B42_european_acceleration_typicality =
    low       if B41 <= 0.491
    typ_low   if 0.491 < B41 <= 0.697
    typ_high  if 0.697 < B41 <= 0.989
    high      if 0.989 < B41 <= 1.404
    extreme   if B41 > 1.404
```

B42 is a European reference distribution label, not a causal anomaly claim. The
cutoffs are **corpus-relative**: fixed quantiles of the full European
candidate-pool distribution, baked into the compute script as constants.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B41 | 10,000 | 0 | 0.33 | 0.62 | 1.23615 |

### B42 label distribution

| label | count | share |
|---|---:|---:|
| `typ_low` | 3,153 | 31.53% |
| `low` | 2,945 | 29.45% |
| `typ_high` | 2,589 | 25.89% |
| `high` | 1,025 | 10.25% |
| `extreme` | 288 | 2.88% |
