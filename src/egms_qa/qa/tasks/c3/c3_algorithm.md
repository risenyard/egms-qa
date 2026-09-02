# C31/C32/C33 Deformation Front

## Task Question

Does this tile contain a strong spatial velocity jump, where is the strongest jump located, and how sharp is the front relative to the 10k corpus?

## Inputs

- `coords`: point coordinates.
- `mean_velocity`: point-level mean velocity in mm/yr.

Arrays are read from the EGMS encoder 10k tile manifest:

`./data/encoder/manifest/split.parquet`

## Formula

For each tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split the tile into an 8x8 local grid.
3. Keep bins with at least 5 points.
4. For each valid bin:
   bin_mean_velocity = mean(point mean_velocity)
5. For each horizontal or vertical adjacent valid-bin pair:
   neighbor_diff = abs(bin_mean_velocity_a - bin_mean_velocity_b)
6. C31_deformation_front_strength_mm_yr = percentile(neighbor_diff, 90)
7. C32_front_location = adjacent bin pair with max(neighbor_diff)
```

`C32_front_location` is stored as a compact bin-pair string:

```text
r{row_a}c{col_a}-r{row_b}c{col_b}
```

Example: `r4c2-r4c3`.

## Distribution Class

C33 is derived from the observed C31 train-split distribution. The distribution
is strongly right-skewed with a long high-front tail, so C33 uses train
p10/p50/p90 rather than equal quartiles. It is a corpus-relative front-strength
class:

| class | rule |
|---|---:|
| `weak` | C31 <= 0.354557 |
| `moderate` | 0.354557 < C31 <= 0.653784 |
| `strong` | 0.653784 < C31 <= 1.583580 |
| `very_sharp` | C31 > 1.583580 |

## Intentional Exclusions

- C31/C32/C33 do not use RMSE; A41 and C11 handle noise-aware reliability.
- C33 is the only sharp-front class kept in EGMS-QA. There is no separate sharp-differential score/flag; downstream monitoring context should use `C33=very_sharp` when it needs a sharp-front predicate.
- C31 uses p90 neighbor difference as the stable main scalar and uses max neighbor difference only for location.

## Final Distribution

All 10k EGMS encoder tiles:

| statistic | value |
|---|---:|
| p01 | 0.2117 |
| p05 | 0.2955 |
| p25 | 0.4735 |
| p50 | 0.6550 |
| p75 | 0.9853 |
| p95 | 2.2751 |
| p99 | 3.9454 |

C33 class counts:

| class | count | fraction |
|---|---:|---:|
| `weak` | 1009 | 0.1009 |
| `moderate` | 3978 | 0.3978 |
| `strong` | 4032 | 0.4032 |
| `very_sharp` | 981 | 0.0981 |

## File Inventory

- `c3_final_table.csv`: canonical final table with C31, C32, and C33.
- `c3_compute.py`: reproducible computation script.
