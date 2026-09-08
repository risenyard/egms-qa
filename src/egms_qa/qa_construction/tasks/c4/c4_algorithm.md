# C4: Fast-tail spatial extent

## Task overview

C4 measures the area occupied by fast-moving parts of a tile. Observation points are grouped into an 8×8 grid; each valid cell is represented by the 90th percentile of absolute vertical velocity, meaning the value below which 90% of its point magnitudes fall.

| Task | Type | Output and relationship |
|---|---|---|
| C41 | Numeric fraction | Share of valid cells whose 90th-percentile absolute velocity reaches 4.8 mm/yr, a fixed corpus-relative fast-motion cutoff. |
| C42 | Classification | Combines C41 with the number of fast cells to label their extent as none, sparse, localized, or extensive. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv) · [Implementation](c4_compute.py) · [Run and files](../README.md#run-c4)

The fast-motion cutoff is corpus-relative: 4.8 mm/yr is the 95th percentile of cell-level velocity-tail values from 83,323 European candidate tiles. This reference pool is distinct from the released 10,000-tile dataset.

## Algorithm steps

### Reference Distribution

For every European candidate tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split a 7000 m square around that center into an 8x8 grid (875 m cells).
3. Keep bins with at least 5 finite points.
4. For each valid bin:
   bin_abs_velocity_p90 = percentile(abs(point mean_velocity), 90)
5. Pool all valid-bin bin_abs_velocity_p90 values across Europe.
```

The full-Europe reference run used 83,323 candidate tiles and produced 3,351,762
valid-bin values.

| statistic | bin_abs_velocity_p90 mm/yr |
|---|---:|
| p50 | 1.800000 |
| p75 | 2.580000 |
| p90 | 3.720000 |
| p95 | 4.800000 |
| p97.5 | 6.120000 |
| p99 | 8.200000 |

The distribution is unimodal and right-skewed with a long high-velocity tail.
`p90=3.72` is too broad for a fast-tail extent target, while `p99=8.20` is closer
to an extreme-only cutoff. EGMS-QA freezes:

```text
T_fast = full-Europe bin-level p95 = 4.800000 mm/yr
```

### C41 fraction and C42 extent class

For each of the 10,000 released tiles:

```text
fast_tail_bin = bin_abs_velocity_p90 >= T_fast
C41_fast_tail_bin_fraction =
    count(fast_tail_bin over valid bins) / count(valid bins)
```

C42 is a derived extent class from C41. It uses structure-based thresholds rather
than positive-C41 quantile cuts:

```text
none       : fast_tail_bin_count = 0
sparse     : fast_tail_bin_count = 1 or 2
localized  : fast_tail_bin_count >= 3 and C41 < 0.25
extensive  : fast_tail_bin_count >= 3 and C41 >= 0.25
```

The `sparse` class uses absolute bin count because one or two high-tail bins are
not enough to claim a spatial area. The `extensive` class uses a quarter of valid
bins as an interpretable area-coverage threshold.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| C41 | 10,000 | 0 | 0 | 0 | 0.230769 |

### C42 label distribution

| label | count | share |
|---|---:|---:|
| `none` | 5,740 | 57.40% |
| `sparse` | 2,321 | 23.21% |
| `localized` | 1,482 | 14.82% |
| `extensive` | 457 | 4.57% |

### Reference and selection evidence

#### Reference calibration

Additional reference percentiles from the 83,323-tile European candidate pool:

| statistic | value (mm/yr) |
|---|---|
| p50 | 1.80 |
| p90 | 3.72 |
| **p95 (= `T_fast`)** | **4.80** |
| p99 | 8.20 |
| p99.9 | 18.60 |

For an additional distribution check, the natural-log cell velocities were
trimmed to their 1st–99th percentiles. Their fitted mean was 0.594 and standard
deviation 0.536. The reference pool contained 3,351,762 valid cells.
