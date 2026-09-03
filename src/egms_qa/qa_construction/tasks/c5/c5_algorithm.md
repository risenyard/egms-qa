# C5 Algorithm: Spatial Monitoring Context

## Delivered Tasks

`C51_monitoring_priority`

C51 asks whether a tile that already triggers monitoring should be treated as a
standard watch-list case or a high-priority spatial case because it also has a
very sharp deformation front.

`C52_hidden_local_risk`

C52 asks whether the tile's mean subsidence band looks low, while the local
worst-point velocity tail is still high. This captures a spatially hidden local
signal that can be diluted by tile-level averaging.

## Inputs

C5 is a composite family using existing EGMS-QA outputs:

- `B61_monitoring_trigger`
- `C33_deformation_front_strength_class`
- `B22_mean_subsidence_intensity_band`
- `B35_worst_point_significance`

No new point-level, bin-level, or encoder computation is introduced.

## Rules

C51:

```text
if B61_monitoring_trigger == no:
    C51_monitoring_priority = none
elif C33_deformation_front_strength_class == very_sharp:
    C51_monitoring_priority = high
else:
    C51_monitoring_priority = standard
```

C52:

```text
if B22_mean_subsidence_intensity_band in {low, low_mid}
and B35_worst_point_significance in {high, very_high}:
    C52_hidden_local_risk = yes
else:
    C52_hidden_local_risk = no
```

The C52 rule preserves the old EGMS-QA C10 logic: mean severity was only slight or
mild, but the local worst-point significance was high.

## Current All10k Result

C51 counts:

```text
none      9405
standard   199
high       396
```

C52 counts:

```text
no   9593
yes   407
```

## File Inventory

- `c5_final_table.csv`: final C5 table with upstream B/C labels and C51/C52.
- `c5_compute.py`: reproducible composite computation script.
- `c5_summary.json`: counts and rule summary.
