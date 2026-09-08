# B1: Subsidence signal

## Task overview

B1 checks whether mean downward motion in a tile is large relative to its reported point measurement error. A tile contains EGMS observation points; negative vertical velocity means subsidence, and RMSE means root mean squared error.

| Task | Type | Output and relationship |
|---|---|---|
| B11 | Numeric score | Mean downward velocity divided by median point RMSE. Larger positive values indicate stronger subsidence relative to this noise measure. |
| B12 | Yes/no classification | Reports clear subsidence when B11 is at least 1; otherwise reports no clear subsidence. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b1/b1_final_table.csv) · [Implementation](b1_compute.py) · [Run and files](../README.md#run-b1)

The input arrays `mean_velocity` (mm/yr) and `rmse` (mm) are read from each source NPZ tile. The split manifest identifies these files. The released score is a velocity-to-RMSE screening ratio, not a statistical significance test; `eps = 1e-12` prevents division by zero.

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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b1/b1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B11 | 10,000 | 0 | 0.0807331 | 0.99613 | 2.1774 |

### B12 label distribution

| label | count | share |
|---|---:|---:|
| `no_clear_subsidence` | 5,029 | 50.29% |
| `clear_subsidence` | 4,971 | 49.71% |
