# C3: Deformation fronts

## Task overview

C3 looks for sharp changes in mean vertical velocity between neighboring cells in a tile’s 8×8 grid. A deformation front here means a spatial velocity contrast; the task reports its strength and location.

| Task | Type | Output and relationship |
|---|---|---|
| C31 | Numeric value (mm/yr) | 90th percentile of velocity differences between neighboring cells: typical strong spatial contrast. |
| C32 | Grid-pair location | Locates the largest neighboring-cell contrast among the pairs used for C31. |
| C33 | Classification | Converts C31 into four front-strength levels using training-split percentiles. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c3/c3_final_table.csv) · [Implementation](c3_compute.py) · [Run and files](../README.md#run-c3)

## Algorithm steps

### Formula

For each tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split a 7000 m square around that center into an 8x8 grid (875 m cells).
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

Example: `r4c2-r4c3`. Rows and columns run from 0 to 7, increasing with the centered y and x coordinates.

### C33 front-strength class

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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c3/c3_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| C31 | 10,000 | 0 | 0.295513 | 0.655021 | 2.27508 |

### C32 label distribution

<details>
<summary>All 112 released labels</summary>

| label | count | share |
|---|---:|---:|
| `r7c0-r7c1` | 133 | 1.33% |
| `r2c0-r3c0` | 125 | 1.25% |
| `r0c0-r1c0` | 122 | 1.22% |
| `r6c7-r7c7` | 118 | 1.18% |
| `r0c7-r1c7` | 118 | 1.18% |
| `r5c7-r6c7` | 114 | 1.14% |
| `r1c0-r2c0` | 112 | 1.12% |
| `r4c7-r5c7` | 112 | 1.12% |
| `r4c0-r5c0` | 112 | 1.12% |
| `r6c0-r7c0` | 111 | 1.11% |
| `r0c6-r0c7` | 111 | 1.11% |
| `r0c0-r0c1` | 110 | 1.10% |
| `r6c6-r7c6` | 102 | 1.02% |
| `r0c6-r1c6` | 102 | 1.02% |
| `r1c6-r1c7` | 101 | 1.01% |
| `r7c6-r7c7` | 101 | 1.01% |
| `r5c0-r6c0` | 101 | 1.01% |
| `r4c1-r5c1` | 101 | 1.01% |
| `r0c3-r0c4` | 100 | 1.00% |
| `r0c5-r0c6` | 99 | 0.99% |
| `r2c4-r3c4` | 99 | 0.99% |
| `r6c2-r6c3` | 98 | 0.98% |
| `r5c3-r5c4` | 98 | 0.98% |
| `r7c4-r7c5` | 97 | 0.97% |
| `r0c2-r0c3` | 97 | 0.97% |
| `r2c7-r3c7` | 96 | 0.96% |
| `r0c2-r1c2` | 96 | 0.96% |
| `r7c1-r7c2` | 95 | 0.95% |
| `r6c5-r7c5` | 95 | 0.95% |
| `r4c1-r4c2` | 95 | 0.95% |
| `r7c3-r7c4` | 95 | 0.95% |
| `r5c1-r5c2` | 95 | 0.95% |
| `r5c1-r6c1` | 94 | 0.94% |
| `r1c4-r2c4` | 94 | 0.94% |
| `r6c0-r6c1` | 94 | 0.94% |
| `r5c6-r5c7` | 94 | 0.94% |
| `r4c2-r5c2` | 94 | 0.94% |
| `r6c4-r7c4` | 93 | 0.93% |
| `r7c5-r7c6` | 93 | 0.93% |
| `r3c1-r4c1` | 92 | 0.92% |
| `r4c2-r4c3` | 92 | 0.92% |
| `r6c2-r7c2` | 92 | 0.92% |
| `r3c0-r4c0` | 92 | 0.92% |
| `r4c4-r5c4` | 92 | 0.92% |
| `r2c0-r2c1` | 91 | 0.91% |
| `r6c1-r6c2` | 90 | 0.90% |
| `r0c4-r0c5` | 90 | 0.90% |
| `r0c4-r1c4` | 90 | 0.90% |
| `r0c5-r1c5` | 89 | 0.89% |
| `r5c4-r5c5` | 89 | 0.89% |
| `r4c6-r5c6` | 89 | 0.89% |
| `r3c0-r3c1` | 88 | 0.88% |
| `r3c4-r4c4` | 88 | 0.88% |
| `r2c5-r2c6` | 88 | 0.88% |
| `r2c6-r2c7` | 87 | 0.87% |
| `r1c5-r2c5` | 87 | 0.87% |
| `r4c4-r4c5` | 87 | 0.87% |
| `r1c1-r2c1` | 86 | 0.86% |
| `r6c4-r6c5` | 86 | 0.86% |
| `r5c2-r6c2` | 86 | 0.86% |
| `r3c2-r4c2` | 86 | 0.86% |
| `r1c0-r1c1` | 86 | 0.86% |
| `r1c4-r1c5` | 86 | 0.86% |
| `r2c6-r3c6` | 86 | 0.86% |
| `r2c2-r3c2` | 86 | 0.86% |
| `r2c2-r2c3` | 86 | 0.86% |
| `r2c3-r3c3` | 86 | 0.86% |
| `r2c3-r2c4` | 85 | 0.85% |
| `r4c5-r5c5` | 85 | 0.85% |
| `r2c1-r2c2` | 85 | 0.85% |
| `r0c3-r1c3` | 85 | 0.85% |
| `r3c7-r4c7` | 85 | 0.85% |
| `r7c2-r7c3` | 85 | 0.85% |
| `r4c5-r4c6` | 84 | 0.84% |
| `r6c3-r7c3` | 84 | 0.84% |
| `r6c1-r7c1` | 84 | 0.84% |
| `r1c7-r2c7` | 84 | 0.84% |
| `r1c5-r1c6` | 84 | 0.84% |
| `r6c5-r6c6` | 83 | 0.83% |
| `r3c4-r3c5` | 82 | 0.82% |
| `r1c2-r1c3` | 82 | 0.82% |
| `r3c6-r3c7` | 82 | 0.82% |
| `r5c6-r6c6` | 81 | 0.81% |
| `r5c3-r6c3` | 81 | 0.81% |
| `r5c0-r5c1` | 81 | 0.81% |
| `r2c4-r2c5` | 80 | 0.80% |
| `r3c3-r4c3` | 80 | 0.80% |
| `r6c6-r6c7` | 79 | 0.79% |
| `r4c0-r4c1` | 79 | 0.79% |
| `r3c2-r3c3` | 79 | 0.79% |
| `r5c4-r6c4` | 79 | 0.79% |
| `r4c3-r5c3` | 79 | 0.79% |
| `r1c1-r1c2` | 79 | 0.79% |
| `r4c3-r4c4` | 78 | 0.78% |
| `r3c5-r4c5` | 77 | 0.77% |
| `r5c5-r5c6` | 76 | 0.76% |
| `r0c1-r0c2` | 76 | 0.76% |
| `r2c1-r3c1` | 76 | 0.76% |
| `r3c6-r4c6` | 76 | 0.76% |
| `r1c6-r2c6` | 75 | 0.75% |
| `r2c5-r3c5` | 74 | 0.74% |
| `r1c3-r2c3` | 74 | 0.74% |
| `r5c2-r5c3` | 73 | 0.73% |
| `r3c1-r3c2` | 72 | 0.72% |
| `r1c3-r1c4` | 72 | 0.72% |
| `r4c6-r4c7` | 71 | 0.71% |
| `r1c2-r2c2` | 71 | 0.71% |
| `r3c5-r3c6` | 71 | 0.71% |
| `r0c1-r1c1` | 71 | 0.71% |
| `r3c3-r3c4` | 69 | 0.69% |
| `r6c3-r6c4` | 62 | 0.62% |
| `r5c5-r6c5` | 60 | 0.60% |

</details>

### C33 label distribution

| label | count | share |
|---|---:|---:|
| `strong` | 4,032 | 40.32% |
| `moderate` | 3,978 | 39.78% |
| `weak` | 1,009 | 10.09% |
| `very_sharp` | 981 | 9.81% |
