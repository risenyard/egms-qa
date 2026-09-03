# A41/A42 Algorithm

## Task

A41 measures measurement noise level:

> Is the tile's typical EGMS point-level RMSE low enough for downstream monitoring?

## Target

For each tile, read the EGMS point-level `rmse` column and compute:

```text
A41_median_rmse_mm = median(point_rmse_mm)
```

The median is used because A41 is meant to describe typical measurement noise,
not a few local outlier points.

## Classes

The class label uses fixed absolute RMSE thresholds in millimeters:

| class | rule | meaning |
|---|---:|---|
| `low_noise` | median RMSE < 1.0 mm | low typical noise |
| `moderate_noise` | 1.0 <= median RMSE < 1.5 mm | normal usable noise |
| `high_noise` | 1.5 <= median RMSE < 2.0 mm | elevated noise |
| `very_high_noise` | median RMSE >= 2.0 mm | high-noise tile; use caution |

Class counts in the final table:

| class | count | fraction |
|---|---:|---:|
| `low_noise` | 3822 | 0.3822 |
| `moderate_noise` | 4303 | 0.4303 |
| `high_noise` | 1493 | 0.1493 |
| `very_high_noise` | 382 | 0.0382 |

## Files

- `a4_final_table.csv`: final all10k table.
- `a4_compute.py`: deterministic computation from EGMS encoder tile data.

The delivery folder intentionally does not retain distribution plots or scratch
JSON summaries.
