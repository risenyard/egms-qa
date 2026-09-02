# B61 Monitoring Trigger

## Task Question

Should this tile enter the European monitoring watch list based on extreme velocity or acceleration typicality?

## Inputs

- `B36_european_velocity_typicality`
- `B42_european_acceleration_typicality`

## Rule

```text
B61_monitoring_trigger = yes
    if B36_european_velocity_typicality == extreme
    or B42_european_acceleration_typicality == extreme
else:
    B61_monitoring_trigger = no
```

B61 is intentionally the only formal B6 task. Trigger type and broad monitoring summaries are not kept as separate leaf tasks because they are composite explanations rather than new targets.

## Final Counts

All 10k EGMS encoder tiles:

| class | count | fraction |
|---|---:|---:|
| `no` | 9405 | 0.9405 |
| `yes` | 595 | 0.0595 |

## File Inventory

- `b6_final_table.csv`: canonical family table with upstream B36/B42 labels and B61.
- `b6_compute.py`: reproducible computation script from B3 and B4 final tables.
