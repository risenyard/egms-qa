# A21/A22 Algorithm

## Task

A21 measures temporal masked reconstructability:

> When a synchronized temporal block is hidden, can the EGMS encoder reconstruct the missing tile time structure?

## Target

For each tile:

1. Load the EGMS encoder checkpoint from `data/encoder/checkpoint/encoder.pt`.
2. Read the tile data config from `data/encoder/manifest/data_config.json`.
3. Keep at most 4096 points per tile; oversized tiles use deterministic tile-id sampling.
4. Mask the centered 30% synchronized temporal block:

```text
mask_start = 103
mask_end   = 191
```

5. Run the frozen encoder and compute error only on masked finite positions:

```text
A21_masked_global_mse_z = mean((recon_z - target_z)^2)
A21_masked_global_rmse_mm = sqrt(A21_masked_global_mse_z) * normalization_std
```

A21 does not use coverage score, point/bin masking, or coverage breakpoints.

## Thresholds

The continuous target is primary. Class thresholds are fitted on the train split
only (`n=8000`) and then applied to all 10k tiles.

| class | rule | MSE_z threshold | RMSE_mm threshold |
|---|---:|---:|---:|
| `reconstructable` | mse <= train p75 | <= 0.099960 | <= 1.801 |
| `mildly_hard` | train p75 < mse <= train p95 | <= 0.154634 | <= 2.240 |
| `high_error` | train p95 < mse <= train p99 | <= 0.245728 | <= 2.824 |
| `unreliable` | mse > train p99 | > 0.245728 | > 2.824 |

Applied to all 10k tiles:

| class | count | fraction |
|---|---:|---:|
| `reconstructable` | 7509 | 0.7509 |
| `mildly_hard` | 1997 | 0.1997 |
| `high_error` | 402 | 0.0402 |
| `unreliable` | 92 | 0.0092 |

## Files

- `a2_final_table.csv`: final all10k table.
- `a2_compute.py`: shard-level GPU computation.
- `a2_combine_shards.py`: shard combiner; defaults to `--threshold-pool train`.

The delivery folder intentionally does not retain plots, scratch JSON summaries,
shards, Slurm logs, or bytecode caches. Thresholds and counts are recorded in
this note and in the final table's class labels.
