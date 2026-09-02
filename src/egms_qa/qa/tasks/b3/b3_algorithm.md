# B31-B36 Velocity Tail Family

## Task Question

What do the tile's velocity tails say about sinking-side motion, uplift protection, local worst-point strength, and European velocity typicality?

## Targets

```text
B31_velocity_p10_mm_yr = percentile(point mean_velocity, 10)
B32_velocity_p90_mm_yr = percentile(point mean_velocity, 90)
B33_vel_abs_p90_mm_yr = percentile(abs(point mean_velocity), 90)
```

B31 captures the sinking-side tail. B32 captures the upper-side tail. B33 captures local absolute motion strength.

## Derived Labels

Uplift-protected direction:

```text
B34_uplift_protected_direction = uplift
    if B32_velocity_p90_mm_yr > abs(B31_velocity_p10_mm_yr)
    else non_uplift
```

Worst-point significance:

```text
B35_worst_point_significance =
    very_low   if B33 < 1.5
    low        if 1.5 <= B33 < 1.9
    moderate   if 1.9 <= B33 < 2.24
    high       if 2.24 <= B33 < 2.9
    very_high  if B33 >= 2.9
```

European velocity typicality:

```text
B36_european_velocity_typicality =
    low       if B33 <= 1.367
    typ_low   if 1.367 < B33 <= 2.075
    typ_high  if 2.075 < B33 <= 3.149
    high      if 3.149 < B33 <= 4.781
    extreme   if B33 > 4.781
```

B36 is a European reference distribution label, not a causal anomaly claim.

## Final Counts

All 10k EGMS encoder tiles:

| B34 class | count | fraction |
|---|---:|---:|
| `non_uplift` | 9664 | 0.9664 |
| `uplift` | 336 | 0.0336 |

| B35 class | count | fraction |
|---|---:|---:|
| `very_low` | 2307 | 0.2307 |
| `low` | 2206 | 0.2206 |
| `moderate` | 2022 | 0.2022 |
| `high` | 1713 | 0.1713 |
| `very_high` | 1752 | 0.1752 |

| B36 class | count | fraction |
|---|---:|---:|
| `low` | 1857 | 0.1857 |
| `typ_low` | 3768 | 0.3768 |
| `typ_high` | 3056 | 0.3056 |
| `high` | 922 | 0.0922 |
| `extreme` | 397 | 0.0397 |

## File Inventory

- `b3_final_table.csv`: canonical family table with B31, B32, B33, B34, B35, and B36.
- `b3_compute.py`: reproducible computation script.
