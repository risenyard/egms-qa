# A4: Measurement noise

## Task overview

A4 summarizes typical measurement noise in a tile, a spatial collection of EGMS observation points. Each point has a reported root mean squared error (RMSE); A4 uses the median of these errors.

| Task | Type | Output and relationship |
|---|---|---|
| A41 | Numeric value (mm) | Median point measurement error, expressed as root mean squared error (RMSE). |
| A42 | Classification | Converts A41 into four noise levels using fixed millimeter thresholds. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv) · [Implementation](a4_compute.py) · [Run and files](../README.md#run-a4)

## Algorithm steps

### Target

For each tile, read the EGMS point-level `rmse` column and compute:

```text
A41_median_rmse_mm = median(point_rmse_mm)
```

The median is used because A41 is meant to describe typical measurement noise,
not a few local outlier points.

### A42 noise classes

The class label uses fixed absolute RMSE thresholds in millimeters:

| class | rule | meaning |
|---|---:|---|
| `low_noise` | median RMSE < 1.0 mm | low typical noise |
| `moderate_noise` | 1.0 <= median RMSE < 1.5 mm | normal usable noise |
| `high_noise` | 1.5 <= median RMSE < 2.0 mm | elevated noise |
| `very_high_noise` | median RMSE >= 2.0 mm | high-noise tile; use caution |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

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
