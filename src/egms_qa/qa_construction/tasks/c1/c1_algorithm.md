# C11/C12/C13 Noise-Aware Moving Fraction

## Task Question

What fraction of points in this tile show motion larger than their own observation noise, how broad is that moving-point extent, and which 8x8 bin has the strongest mean motion magnitude?

## Inputs

- `mean_velocity`: point-level mean velocity in mm/yr.
- `rmse`: point-level RMSE/noise estimate.
- `coords`: point coordinates.

Both arrays are read from the EGMS encoder 10k tile manifest:

`./data/encoder/manifest/split.parquet`

## Formula

For each point:

```text
point_motion_snr = abs(point mean_velocity) / (point rmse + eps)
```

For each tile:

```text
C11_noise_aware_moving_fraction =
    count(point_motion_snr >= 1.0) / total_points
```

The `1.0x RMSE` rule means a point counts as moving only when its motion magnitude is at least as large as its own RMSE.

C13 locates the strongest bin by bin-level average velocity magnitude:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split the tile into an 8x8 local grid.
3. Keep bins with at least 5 finite points.
4. For each valid bin:
   bin_mean_abs_velocity = mean(abs(point mean_velocity))
5. C13_moving_bin_location = bin with the largest bin_mean_abs_velocity.
```

Ties are resolved by larger valid point count, then by bin order. The location
string uses `r{row}c{col}`, for example `r4c3`. If no valid bin exists, C13 is
`none`.

## Distribution Class

C12 is derived from the observed C11 train-split distribution. The distribution
has a broad middle body and two thinner tails, so C12 uses train p10/p50/p90
rather than equal quartiles. It is a corpus-relative extent class, not a
physical risk threshold:

| class | rule |
|---|---:|
| `limited` | C11 <= 0.080806 |
| `partial` | 0.080806 < C11 <= 0.423910 |
| `broad` | 0.423910 < C11 <= 0.776414 |
| `widespread` | C11 > 0.776414 |

## Intentional Exclusions

- C11 does not use the old fixed `abs(mean_velocity) > 2 mm/yr` threshold.
- C11/C12 do not judge noise level by itself; A41 handles tile-level RMSE.
- C11/C12 do not decide direction or European intensity; B-family tasks handle those.
- C13 is an explanation string derived from the same 8x8 binning used by the C family. It is not a separate location family and should not be treated as a core probe target.

## Final Distribution

All 10k EGMS encoder tiles:

| statistic | value |
|---|---:|
| p01 | 0.0179 |
| p05 | 0.0466 |
| p25 | 0.2073 |
| p50 | 0.4226 |
| p75 | 0.6266 |
| p95 | 0.8416 |
| p99 | 0.9386 |

C12 class counts:

| class | count | fraction |
|---|---:|---:|
| `limited` | 994 | 0.0994 |
| `partial` | 4026 | 0.4026 |
| `broad` | 3981 | 0.3981 |
| `widespread` | 999 | 0.0999 |

C13 uses all 64 possible 8x8 bin strings in the 10k table; `none` does not occur
in the current run. It is kept as a location explanation rather than a balanced
classification target.

Most frequent C13 locations:

| bin | count |
|---|---:|
| `r7c0` | 213 |
| `r0c7` | 209 |
| `r7c7` | 196 |
| `r0c0` | 195 |
| `r6c0` | 190 |
| `r0c6` | 188 |
| `r4c0` | 186 |
| `r1c0` | 186 |

## File Inventory

- `c1_final_table.csv`: canonical final table with C11, C12, and C13.
- `c1_compute.py`: reproducible computation script.
