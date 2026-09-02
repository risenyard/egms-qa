# D3 Acceleration Algorithm

## Current Scope

This folder is the D3 acceleration family. The implemented targets are:

- `D31_motion_intensification_mm_yr2`
- `D32_acceleration_support_fraction`
- `D33_intensification_spread_mm_yr2`
- `D34_intensification_hotspot_strength_mm_yr2`
- `D35_intensification_hotspot_location`

D3 is a point-level acceleration-field story. It asks whether moving points in
the tile are intensifying or weakening, whether that acceleration direction is
spatially supported, how spread out the direction-aware acceleration field is,
and where the strongest local hotspot sits in an 8x8 tile grid. It does not use
a tile-level direction such as B34 to flip point-level acceleration.

## Point-Level Definition

For each point, read:

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

## D31 Formula

```text
D31_motion_intensification_mm_yr2 =
    median_i(point_intensification_i over valid moving points)
```

If the tile-level validity gate fails, final D31 is undefined.

Interpretation:

- positive: dominant moving points are intensifying
- negative: dominant moving points are weakening
- near zero: central acceleration change is weak or balanced

## D32 Formula

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

## D33 Formula

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

D33 is not raw acceleration strength. Raw absolute acceleration strength remains
the B41/B42 story. D33 only measures the distribution width of the D31
direction-aware acceleration field.

## D34/D35 Formula

D34 and D35 reuse the same `point_intensification_i` and the same valid moving
point gate as D31-D33. Split the tile into the same 8x8 local grid style used by
the C-family location tasks. Keep only bins with at least 5 valid moving points.

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

If the tile-level validity gate fails, D34 is undefined and D35 is `none`.

Interpretation:

- D34 high: one local 8x8 bin has strong direction-aware acceleration activity
- D35: the local bin where that activity is strongest

D34 is not raw acceleration strength. It is the strongest local bin of the D31
direction-aware acceleration field.

## Relationship To B41

| task | formula | meaning |
|---|---|---|
| B41 | `p90(abs(point acceleration))` | acceleration strength |
| D31 | `median(sign(mean_velocity) * acceleration)` over valid moving points | dominant-motion intensification |
| D32 | support fraction for D31 direction | spatial consistency of intensification/weakening |
| D33 | `p90(point_intensification) - p10(point_intensification)` | spread of the direction-aware acceleration field |
| D34 | max 8x8-bin `mean(abs(point_intensification))` | hotspot strength of the direction-aware acceleration field |
| D35 | `r{row}c{col}` of the D34 max bin | hotspot location |

D31, D32, D33 and D34 remain scalar-only. D35 is location-only. They have no
hard acceleration class thresholds.

## Exclusions

- D3 does not include changepoint slope jump. That belongs to the D1 trend/regime
  story, because it depends on D13/D14 and tile-level trend timing rather than
  point-level acceleration fields.
- D3 does not classify D31/D32/D33/D34 into hard bands.
- D3 does not replace B41 acceleration strength; B41 measures raw acceleration
  magnitude, while D31-D35 measure direction-aware intensification, support,
  spread, hotspot strength, and hotspot location.

## Files

- `d3_final_table.csv`: canonical D31-D35 target table plus validity reason.
- `d3_final_diagnostics.csv`: final and raw D31-D35, point-intensification summaries, raw acceleration diagnostics, valid motion counts, and support reasons.
- `d3_compute.py`: recomputes D31-D35.

## Current All-10k Result

`D31_motion_intensification_mm_yr2`:

| statistic | value |
|---|---:|
| count defined | 8508 |
| mean | 0.025996 |
| p01 | -0.520000 |
| p05 | -0.220000 |
| p10 | -0.110000 |
| p25 | -0.030000 |
| p50 | 0.010000 |
| p75 | 0.070000 |
| p90 | 0.210000 |
| p95 | 0.330000 |
| p99 | 0.783950 |

`D32_acceleration_support_fraction`:

| statistic | value |
|---|---:|
| count defined | 7967 |
| mean | 0.624230 |
| p25 | 0.535377 |
| p50 | 0.583893 |
| p75 | 0.677198 |
| p90 | 0.810806 |
| p95 | 0.883951 |
| p99 | 0.975410 |

`D33_intensification_spread_mm_yr2`:

| statistic | value |
|---|---:|
| count defined | 8508 |
| mean | 0.874792 |
| p01 | 0.300000 |
| p05 | 0.352000 |
| p10 | 0.400000 |
| p25 | 0.520000 |
| p50 | 0.750000 |
| p75 | 1.078000 |
| p90 | 1.484000 |
| p95 | 1.772000 |
| p99 | 2.518880 |

`D34_intensification_hotspot_strength_mm_yr2`:

| statistic | value |
|---|---:|
| count defined | 8508 |
| mean | 0.793356 |
| p01 | 0.217861 |
| p05 | 0.295000 |
| p10 | 0.345714 |
| p25 | 0.458000 |
| p50 | 0.648278 |
| p75 | 0.927143 |
| p90 | 1.341129 |
| p95 | 1.720361 |
| p99 | 3.007306 |

`D35_intensification_hotspot_location` top locations:

| location | count |
|---|---:|
| `none` | 1492 |
| `r7c7` | 161 |
| `r0c0` | 158 |
| `r3c0` | 157 |
| `r2c0` | 157 |
| `r7c0` | 155 |
| `r6c0` | 154 |
| `r2c3` | 154 |
| `r0c5` | 153 |
| `r7c5` | 153 |

D31/D32 validity reason counts:

| reason | count |
|---|---:|
| `valid` | 7967 |
| `insufficient_valid_motion_points` | 1492 |
| `zero_or_undefined_d31_direction` | 541 |

`D31`, `D33`, and `D34` are defined for all 8508 gate-passed tiles. `D35` is
`none` for the 1492 gate-failed tiles. `D32` is defined only for the 7967
gate-passed tiles where D31 has a nonzero direction.

Raw pre-gate diagnostics are retained in `d3_final_diagnostics.csv` as
`D31_raw_motion_intensification_mm_yr2` and
`D32_raw_acceleration_support_fraction`. D33 raw pre-gate spread is retained as
`D33_raw_intensification_spread_mm_yr2`. D34/D35 raw pre-gate hotspot outputs
are retained as `D34_raw_intensification_hotspot_strength_mm_yr2` and
`D35_raw_intensification_hotspot_location`.
