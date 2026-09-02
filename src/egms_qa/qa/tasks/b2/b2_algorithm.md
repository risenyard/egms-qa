# B21/B22 Mean Velocity Family

## Task Question

What is the tile's average velocity, and which European relative subsidence intensity band does it fall into?

## Targets

```text
B21_mean_velocity_mm_yr = mean(point mean_velocity)
```

B22 is derived from B21 and the uplift-protected direction rule:

```text
if B34_uplift_protected_direction == uplift:
    B22_mean_subsidence_intensity_band = uplift
elif B21_mean_velocity_mm_yr <= -1.47:
    B22_mean_subsidence_intensity_band = high
elif B21_mean_velocity_mm_yr <= -1.215:
    B22_mean_subsidence_intensity_band = high_mid
elif B21_mean_velocity_mm_yr <= -0.971:
    B22_mean_subsidence_intensity_band = mid
elif B21_mean_velocity_mm_yr <= -0.657:
    B22_mean_subsidence_intensity_band = low_mid
else:
    B22_mean_subsidence_intensity_band = low
```

The B34 direction used here is copied into the B2 final table as an upstream explanation column so B22 is reproducible within the folder:

```text
B34_uplift_protected_direction = uplift
    if B32_velocity_p90_mm_yr > abs(B31_velocity_p10_mm_yr)
    else non_uplift
```

## Interpretation

B22 is a corpus-relative European intensity band. It is not a legal, physical, or causal severity class. Uplift is kept as a separate direction override.

## Final Counts

All 10k EGMS encoder tiles:

| class | count | fraction |
|---|---:|---:|
| `high` | 2013 | 0.2013 |
| `high_mid` | 1952 | 0.1952 |
| `mid` | 1906 | 0.1906 |
| `low_mid` | 1879 | 0.1879 |
| `low` | 1914 | 0.1914 |
| `uplift` | 336 | 0.0336 |

## File Inventory

- `b2_final_table.csv`: canonical family table with B21, upstream B34 direction, and B22.
- `b2_compute.py`: reproducible computation script.
