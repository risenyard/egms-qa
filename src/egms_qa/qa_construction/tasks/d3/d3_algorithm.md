# D3: Motion intensification

## Task overview

D3 describes whether motion is speeding up or slowing down across the observation points in a tile. It combines each point’s vertical velocity and acceleration so that positive intensification means faster motion in its existing direction, whether downward or upward.

| Task | Type | Output and relationship |
|---|---|---|
| D31 | Numeric value (mm/yr²) | Median acceleration signed by each point's motion direction. Positive means intensifying motion; negative means weakening. |
| D32 | Numeric fraction | Share of valid moving points whose intensification sign agrees with D31. |
| D33 | Numeric value (mm/yr²) | 90th minus 10th percentile of the point intensification values used for D31: their spread across the tile. |
| D34 | Numeric value (mm/yr²) | Largest cell mean absolute intensification, using the same valid moving points as D31. |
| D35 | Grid location | Identifies the cell that supplies the D34 hotspot strength. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d3/d3_final_table.csv) · [Implementation](d3_compute.py) · [Run and files](../README.md#run-d3)

## Algorithm steps

### Point-Level Definition

For each point, read vertical velocity `v_i` (mm/yr), acceleration `a_i` (mm/yr²), and root mean squared measurement error `r_i` (mm). The small constant `eps = 1e-9` prevents division by zero:

```text
v_i = mean_velocity_i
a_i = acceleration_i
r_i = rmse_i
```

Keep only valid moving points:

```text
abs(v_i) / (r_i + eps) >= 1
```

Then apply a tile-level validity gate:

```text
valid_moving_point_count >= 100
valid_moving_point_fraction >= 0.10
```

Tiles that fail this gate keep raw diagnostics, but their final D31-D35 target
values are set to NaN or `none`. The point count/fraction gate prevents a small
local subset of points from defining the whole-tile acceleration label.

Then compute direction-aware point intensification:

```text
point_intensification_i = sign(v_i) * a_i
```

This gives:

| velocity | acceleration | point intensification | meaning |
|---:|---:|---:|---|
| negative | negative | positive | subsidence intensifying |
| negative | positive | negative | subsidence weakening |
| positive | positive | positive | uplift intensifying |
| positive | negative | negative | uplift weakening |

### D31 Formula

```text
D31_motion_intensification_mm_yr2 =
    median_i(point_intensification_i over valid moving points)
```

If the tile-level validity gate fails, final D31 is undefined.

Interpretation:

- positive: dominant moving points are intensifying
- negative: dominant moving points are weakening
- near zero: central acceleration change is weak or balanced

### D32 Formula

If D31 has a nonzero direction:

```text
D32_acceleration_support_fraction =
    count(sign(point_intensification_i) == sign(D31)) / valid_moving_points
```

If D31 is zero/undefined, D32 is undefined because there is no direction to
support. D32 is also undefined when the tile-level validity gate fails.

Interpretation:

- near 1: most valid moving points support the D31 direction
- near 0.5: support is mixed and spatially weak

Because D31 is a median, D32 is expected to be at least about 0.5 whenever it is
defined.

### D33 Formula

D33 reuses the same `point_intensification_i` and the same valid moving point
gate as D31/D32:

```text
D33_intensification_spread_mm_yr2 =
    p90(point_intensification_i) - p10(point_intensification_i)
```

If the tile-level validity gate fails, final D33 is undefined.

Interpretation:

- high: direction-aware acceleration is spread out across the valid moving
  points
- low: direction-aware acceleration is more compact or uniform

### D34/D35 Formula

D34 and D35 reuse the same `point_intensification_i` and the same valid moving
point gate as D31-D33. Center the coordinates on their tile mean and divide a 7000 m square into an 8×8 grid of 875 m cells. Keep cells with at least 5 valid moving points.

For every valid bin:

```text
bin_hotspot_score =
    mean(abs(point_intensification_i) for points in the bin)
```

Then:

```text
D34_intensification_hotspot_strength_mm_yr2 = max(bin_hotspot_score)
D35_intensification_hotspot_location = r{row}c{col} of the max bin
```

If the tile-level validity gate fails, D34 is undefined and D35 is `none`. The location format is `r{row}c{col}`, with row and column indices from 0 to 7 increasing along centered y and x.

Interpretation:

- D34 high: one local 8x8 bin has strong direction-aware acceleration activity
- D35: the local bin where that activity is strongest

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d3/d3_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| D31 | 8,508 | 1,492 | -0.22 | 0.01 | 0.33 |
| D32 | 7,967 | 2,033 | 0.50643 | 0.583893 | 0.883951 |
| D33 | 8,508 | 1,492 | 0.352 | 0.75 | 1.772 |
| D34 | 8,508 | 1,492 | 0.295 | 0.648278 | 1.72036 |

### D35 label distribution

<details>
<summary>All 65 released labels</summary>

| label | count | share |
|---|---:|---:|
| `none` | 1,492 | 14.92% |
| `r7c7` | 161 | 1.61% |
| `r0c0` | 158 | 1.58% |
| `r3c0` | 157 | 1.57% |
| `r2c0` | 157 | 1.57% |
| `r7c0` | 155 | 1.55% |
| `r6c0` | 154 | 1.54% |
| `r2c3` | 154 | 1.54% |
| `r0c5` | 153 | 1.53% |
| `r7c5` | 153 | 1.53% |
| `r6c7` | 153 | 1.53% |
| `r4c0` | 151 | 1.51% |
| `r0c7` | 150 | 1.50% |
| `r7c1` | 150 | 1.50% |
| `r5c7` | 148 | 1.48% |
| `r0c4` | 144 | 1.44% |
| `r3c5` | 143 | 1.43% |
| `r7c4` | 142 | 1.42% |
| `r5c0` | 142 | 1.42% |
| `r7c3` | 141 | 1.41% |
| `r0c1` | 141 | 1.41% |
| `r1c3` | 140 | 1.40% |
| `r4c2` | 139 | 1.39% |
| `r7c6` | 137 | 1.37% |
| `r0c3` | 136 | 1.36% |
| `r5c1` | 136 | 1.36% |
| `r5c4` | 136 | 1.36% |
| `r6c5` | 134 | 1.34% |
| `r0c6` | 133 | 1.33% |
| `r3c1` | 133 | 1.33% |
| `r3c2` | 133 | 1.33% |
| `r4c7` | 132 | 1.32% |
| `r1c1` | 132 | 1.32% |
| `r1c7` | 132 | 1.32% |
| `r1c0` | 131 | 1.31% |
| `r2c6` | 130 | 1.30% |
| `r2c5` | 128 | 1.28% |
| `r0c2` | 128 | 1.28% |
| `r2c7` | 128 | 1.28% |
| `r3c3` | 127 | 1.27% |
| `r5c3` | 127 | 1.27% |
| `r2c4` | 126 | 1.26% |
| `r4c5` | 126 | 1.26% |
| `r6c1` | 125 | 1.25% |
| `r7c2` | 125 | 1.25% |
| `r3c7` | 125 | 1.25% |
| `r1c2` | 125 | 1.25% |
| `r6c2` | 123 | 1.23% |
| `r4c4` | 123 | 1.23% |
| `r6c4` | 122 | 1.22% |
| `r4c6` | 122 | 1.22% |
| `r5c6` | 121 | 1.21% |
| `r4c1` | 120 | 1.20% |
| `r6c3` | 120 | 1.20% |
| `r1c6` | 120 | 1.20% |
| `r5c2` | 119 | 1.19% |
| `r2c1` | 118 | 1.18% |
| `r3c4` | 117 | 1.17% |
| `r5c5` | 114 | 1.14% |
| `r6c6` | 114 | 1.14% |
| `r1c4` | 113 | 1.13% |
| `r3c6` | 113 | 1.13% |
| `r1c5` | 112 | 1.12% |
| `r4c3` | 109 | 1.09% |
| `r2c2` | 97 | 0.97% |

</details>
