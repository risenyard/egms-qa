# C21/C22 Spatial Concentration

## Task Question

Is the motion magnitude spatially concentrated in a few parts of the tile, or spread more evenly?

## Inputs

- `coords`: point coordinates.
- `mean_velocity`: point-level mean velocity in mm/yr.

Arrays are read from the EGMS encoder 10k tile manifest:

`./data/encoder/manifest/split.parquet`

## Formula

For each point:

```text
point_motion_magnitude = abs(point mean_velocity)
```

For each tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split the tile into an 8x8 local grid.
3. Keep bins with at least 5 points.
4. For each valid bin:
   bin_mean_abs_velocity = mean(point_motion_magnitude)
5. C21_spatial_concentration_score = Gini(bin_mean_abs_velocity over valid bins)
```

Higher values mean the motion magnitude is more spatially concentrated.

## Distribution Class

C22 is derived from the observed C21 train-split distribution. The distribution
is unimodal and right-skewed with a long high-concentration tail, so C22 uses
train p10/p50/p90 rather than equal quartiles. It is a corpus-relative spatial
organization class:

| class | rule |
|---|---:|
| `diffuse` | C21 <= 0.093756 |
| `mildly_concentrated` | 0.093756 < C21 <= 0.181511 |
| `concentrated` | 0.181511 < C21 <= 0.312813 |
| `highly_concentrated` | C21 > 0.312813 |

## Intentional Exclusions

- C21/C22 do not use RMSE or motion SNR. RMSE belongs to A41 and noise-aware point activity belongs to C11.
- C21/C22 do not count active bins; the old active spatial extent task was deleted.
- C21/C22 do not use a fixed `2 mm/yr` active threshold.
- C21/C22 do not classify moving-support location; C13 handles the C1-derived bin string.

## Final Distribution

All 10k EGMS encoder tiles:

| statistic | value |
|---|---:|
| p01 | 0.0550 |
| p05 | 0.0785 |
| p25 | 0.1282 |
| p50 | 0.1815 |
| p75 | 0.2461 |
| p95 | 0.3622 |
| p99 | 0.4870 |

C22 class counts:

| class | count | fraction |
|---|---:|---:|
| `diffuse` | 984 | 0.0984 |
| `mildly_concentrated` | 4015 | 0.4015 |
| `concentrated` | 3998 | 0.3998 |
| `highly_concentrated` | 1003 | 0.1003 |

## File Inventory

- `c2_final_table.csv`: canonical final table with C21 and C22.
- `c2_compute.py`: reproducible computation script.
- `c2_gini_distribution.png`: diagnostic distribution plot.
- `c1_c2_hexbin.png`: diagnostic C11-C21 relationship plot.
- `examples/`: diagnostic 8x8 bin example plots.
