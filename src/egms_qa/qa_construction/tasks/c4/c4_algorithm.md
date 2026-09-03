# C41/C42 Fast-Tail Spatial Extent

## Task Question

Do high-velocity tail bins occupy a spatial area, or are they confined to a very small part of the tile?

## Inputs

- `coords`: point coordinates.
- `mean_velocity`: point-level mean velocity in mm/yr.

The 10k final table uses:

`./data/encoder/manifest/split.parquet`

The reference threshold is a corpus-relative value estimated from the full
European candidate pool (not shipped; the derived threshold is already in the
released c4 table). Only re-deriving it needs the encoder project:

`../egms-encoder/data/processed/v4/v4_pool_candidates.parquet`

## Reference Distribution

For every European candidate tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split the tile into an 8x8 local grid.
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

For reference, the trimmed log-bulk fit gives `z2 = 5.287757 mm/yr`, close to p95,
which supports p95 as a stable high-tail threshold rather than an arbitrary cut.

## Final Formula

For each 10k VQA tile:

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

## Final Distribution

All 10k EGMS encoder tiles:

| statistic | C41_fast_tail_bin_fraction |
|---|---:|
| p50 | 0.000000 |
| p75 | 0.035714 |
| p90 | 0.125000 |
| p95 | 0.230769 |
| p99 | 0.577806 |

C42 class counts:

| class | count | fraction |
|---|---:|---:|
| `none` | 5740 | 0.5740 |
| `sparse` | 2321 | 0.2321 |
| `localized` | 1482 | 0.1482 |
| `extensive` | 457 | 0.0457 |

## Intentional Exclusions

- C4 does not measure the strongest velocity itself; B33/B36 handle velocity-tail magnitude.
- C4 does not measure all meaningful motion; C11 handles noise-aware moving-point fraction.
- C4 does not measure global concentration; C21/C22 handle spatial concentration over all motion.
- C4 does not measure a deformation front; C31/C33 handle adjacent-bin jumps.

## File Inventory

- `c4_compute.py`: reference and final computation script.
- `c4_bin_level_reference_thresholds.json`: frozen full-Europe reference statistics.
- `c4_bin_level_reference_distribution.png`: diagnostic reference distribution plot.
- `c4_final_table.csv`: canonical final table with C41 and C42.
- `c4_final_summary.json`: frozen final distribution and class-count summary.
