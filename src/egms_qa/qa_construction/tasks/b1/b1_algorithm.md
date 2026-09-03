# B11/B12 Clear Subsidence Signal

## Task Question

Does this tile show clear average subsidence relative to its observation noise?

## Inputs

- `mean_velocity`: point-level mean velocity in mm/yr.
- `rmse`: point-level RMSE/noise estimate.

Both arrays are read from the EGMS encoder 10k tile manifest:

`./data/encoder/manifest/split.parquet`

## Formula

For each tile:

```text
v_mean = mean(point mean_velocity)
rmse_median = median(point rmse)
B11_subsidence_snr = -v_mean / (rmse_median + eps)
```

The minus sign makes stronger subsidence a larger positive value.

## Class Rule

```text
if B11_subsidence_snr >= 1.0:
    B12_clear_subsidence_class = clear_subsidence
else:
    B12_clear_subsidence_class = no_clear_subsidence
```

The threshold `1.0` is a signal-to-noise rule: the tile's average subsidence must be at least as large as the median point RMSE. It is not a European severity threshold and is not fitted from the corpus distribution.

## Intentional Exclusions

- B11/B12 do not classify uplift.
- B11/B12 do not use p10/p90 tail velocity.
- B11/B12 do not assign mild/moderate/strong severity; that belongs to the B/C/D derived monitoring layer.
- B11/B12 do not replace A41/A51 quality gates; it uses RMSE only to normalize this specific direction signal.

## Final Counts

All 10k EGMS encoder tiles:

| class | count |
|---|---:|
| clear_subsidence | 4971 |
| no_clear_subsidence | 5029 |

Split counts:

| split | clear_subsidence | no_clear_subsidence |
|---|---:|---:|
| train | 4003 | 3997 |
| val | 472 | 528 |
| test | 496 | 504 |

## File Inventory

- `b1_final_table.csv`: canonical final table.
- `b1_compute.py`: reproducible computation script.
