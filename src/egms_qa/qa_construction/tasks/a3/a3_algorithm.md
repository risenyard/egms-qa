# A3: Spatial observation coverage

## Task overview

A3 describes how widely observation points cover a tile. The tile is divided into an 8×8 grid, and a cell is occupied if it contains at least one point.

| Task | Type | Output and relationship |
|---|---|---|
| A31 | Numeric fraction | Share of the 64 spatial cells that contain observations, from 0 to 1. |
| A32 | Classification | Converts A31 into four coverage levels using fixed occupied-cell thresholds. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv) · [Implementation](a3_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Target

A31 uses the 8x8 spatial bin counts already stored in the EGMS encoder token cache:

```text
occupied_bins = count(point_count_per_bin > 0)
A31_valid_bin_fraction_8x8 = occupied_bins / 64
```

### A32 coverage classes

The class label uses fixed structural thresholds, not empirical percentiles:

| class | rule | meaning |
|---|---:|---|
| `well_spread` | fraction >= 0.75 | observations cover most of the tile |
| `moderate_gaps` | 0.50 <= fraction < 0.75 | visible coverage gaps |
| `sparse` | 0.25 <= fraction < 0.50 | sparse spatial support |
| `highly_fragmented` | fraction < 0.25 | very fragmented support |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| A31 | 10,000 | 0 | 0.484375 | 0.9375 | 1 |

### A32 label distribution

| label | count | share |
|---|---:|---:|
| `well_spread` | 7,836 | 78.36% |
| `moderate_gaps` | 1,663 | 16.63% |
| `sparse` | 434 | 4.34% |
| `highly_fragmented` | 67 | 0.67% |
