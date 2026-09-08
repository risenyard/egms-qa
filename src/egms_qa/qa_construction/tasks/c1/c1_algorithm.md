# C1: Motion extent and location

## Task overview

C1 describes how many observation points in a tile show motion above the task’s noise criterion, and where motion is strongest. It reads point vertical velocity, root mean squared error (RMSE), and coordinates. Negative velocity means subsidence; positive velocity means uplift.

| Task | Type | Output and relationship |
|---|---|---|
| C11 | Numeric fraction | Share of points whose absolute velocity is at least their reported RMSE, using the task's motion-to-noise rule. |
| C12 | Classification | Converts C11 into four moving-point extent levels using training-split percentiles. |
| C13 | Grid location | Identifies the cell with the largest mean absolute velocity. Complements C11 with a location, using all eligible cells. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c1/c1_final_table.csv) · [Implementation](c1_compute.py) · [Run and files](../README.md#run-c1)

The source NPZ files contain `mean_velocity` (mm/yr), `rmse` (mm), and `coords`. The velocity-to-RMSE ratio below is the released motion-screening rule, not a statistical significance test. `eps = 1e-12` prevents division by zero.

## Algorithm steps

### Formula

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
2. Split a 7000 m square around that center into an 8x8 grid (875 m cells).
3. Keep bins with at least 5 finite points.
4. For each valid bin:
   bin_mean_abs_velocity = mean(abs(point mean_velocity))
5. C13_moving_bin_location = bin with the largest bin_mean_abs_velocity.
```

Ties are resolved by larger valid point count, then by bin order. The location
string uses `r{row}c{col}`, for example `r4c3`. Rows and columns run from 0 to 7, increasing with the centered y and x coordinates. If no valid bin exists, C13 is
`none`.

### C12 extent class

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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c1/c1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| C11 | 10,000 | 0 | 0.0465551 | 0.422582 | 0.841572 |

### C12 label distribution

| label | count | share |
|---|---:|---:|
| `partial` | 4,026 | 40.26% |
| `broad` | 3,981 | 39.81% |
| `widespread` | 999 | 9.99% |
| `limited` | 994 | 9.94% |

### C13 label distribution

<details>
<summary>All 64 released labels</summary>

| label | count | share |
|---|---:|---:|
| `r7c0` | 213 | 2.13% |
| `r0c7` | 209 | 2.09% |
| `r7c7` | 196 | 1.96% |
| `r0c0` | 195 | 1.95% |
| `r6c0` | 190 | 1.90% |
| `r0c6` | 188 | 1.88% |
| `r4c0` | 186 | 1.86% |
| `r1c0` | 186 | 1.86% |
| `r5c7` | 185 | 1.85% |
| `r6c7` | 182 | 1.82% |
| `r2c0` | 181 | 1.81% |
| `r0c5` | 180 | 1.80% |
| `r1c7` | 180 | 1.80% |
| `r7c1` | 173 | 1.73% |
| `r1c6` | 171 | 1.71% |
| `r0c2` | 170 | 1.70% |
| `r3c3` | 166 | 1.66% |
| `r1c5` | 166 | 1.66% |
| `r6c6` | 164 | 1.64% |
| `r3c0` | 164 | 1.64% |
| `r7c5` | 163 | 1.63% |
| `r5c0` | 162 | 1.62% |
| `r7c6` | 162 | 1.62% |
| `r2c7` | 160 | 1.60% |
| `r5c1` | 160 | 1.60% |
| `r5c5` | 157 | 1.57% |
| `r0c3` | 157 | 1.57% |
| `r1c1` | 157 | 1.57% |
| `r6c1` | 156 | 1.56% |
| `r0c4` | 156 | 1.56% |
| `r6c2` | 155 | 1.55% |
| `r7c3` | 154 | 1.54% |
| `r7c4` | 154 | 1.54% |
| `r1c3` | 153 | 1.53% |
| `r0c1` | 153 | 1.53% |
| `r4c1` | 152 | 1.52% |
| `r2c4` | 151 | 1.51% |
| `r2c1` | 150 | 1.50% |
| `r7c2` | 149 | 1.49% |
| `r4c5` | 147 | 1.47% |
| `r2c6` | 146 | 1.46% |
| `r2c5` | 144 | 1.44% |
| `r3c4` | 143 | 1.43% |
| `r1c4` | 143 | 1.43% |
| `r3c6` | 142 | 1.42% |
| `r5c4` | 142 | 1.42% |
| `r1c2` | 141 | 1.41% |
| `r3c2` | 141 | 1.41% |
| `r6c3` | 140 | 1.40% |
| `r4c6` | 139 | 1.39% |
| `r6c4` | 138 | 1.38% |
| `r3c7` | 137 | 1.37% |
| `r3c5` | 137 | 1.37% |
| `r5c3` | 136 | 1.36% |
| `r3c1` | 135 | 1.35% |
| `r4c2` | 134 | 1.34% |
| `r5c6` | 133 | 1.33% |
| `r2c3` | 133 | 1.33% |
| `r4c7` | 132 | 1.32% |
| `r2c2` | 132 | 1.32% |
| `r6c5` | 130 | 1.30% |
| `r5c2` | 127 | 1.27% |
| `r4c4` | 120 | 1.20% |
| `r4c3` | 102 | 1.02% |

</details>
