# B41/B42 Acceleration Strength Family

## Task Question

How strong is the tile's acceleration signal, and how typical is that acceleration relative to the European reference distribution?

## Targets

```text
B41_acc_abs_p90 = percentile(abs(point acceleration), 90)
```

B42 is derived from B41:

```text
B42_european_acceleration_typicality =
    low       if B41 <= 0.491
    typ_low   if 0.491 < B41 <= 0.697
    typ_high  if 0.697 < B41 <= 0.989
    high      if 0.989 < B41 <= 1.404
    extreme   if B41 > 1.404
```

B42 is a European reference distribution label, not a causal anomaly claim. The
cutoffs are **corpus-relative**: fixed quantiles of the full European
candidate-pool distribution, baked into the compute script as constants.

## Final Counts

All 10k EGMS encoder tiles:

| class | count | fraction |
|---|---:|---:|
| `low` | 2945 | 0.2945 |
| `typ_low` | 3153 | 0.3153 |
| `typ_high` | 2589 | 0.2589 |
| `high` | 1025 | 0.1025 |
| `extreme` | 288 | 0.0288 |

## File Inventory

- `b4_final_table.csv`: canonical family table with B41 and B42.
- `b4_compute.py`: reproducible computation script.
