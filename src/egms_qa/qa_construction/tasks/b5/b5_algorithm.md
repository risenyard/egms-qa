# B51 Seasonality Strength

## Task Question

How strong is the tile's seasonality signal?

## Formula

```text
B51_seasonality_p90 = percentile(point seasonality, 90)
```

## Output

- `B51_seasonality_p90`: continuous scalar.

## Intentional Exclusions

- No classification label is produced.
- No seasonal phase is inferred here; phase belongs to the D group.
- No European typicality threshold is used here; B5 only measures seasonality strength.

## Final Distribution

All 10k EGMS encoder tiles:

| statistic | value |
|---|---:|
| p01 | 0.700 |
| p05 | 0.900 |
| p50 | 1.400 |
| p95 | 3.100 |
| p99 | 6.201 |

## File Inventory

- `b5_final_table.csv`: canonical final table.
- `b5_compute.py`: reproducible computation script.
