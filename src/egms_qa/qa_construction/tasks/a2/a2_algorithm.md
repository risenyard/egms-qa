# A2: Masked reconstruction reliability

## Task overview

A2 checks how well the encoder reconstructs a time interval hidden at every observation point in a tile. A tile contains point displacement histories; the encoder predicts the missing displacement values from the remaining observations.

| Task | Type | Output and relationship |
|---|---|---|
| A21 | Numeric error | Mean squared error when reconstructing a hidden time interval. Smaller values mean better reconstruction; values use normalized displacement. |
| A22 | Classification | Groups A21 into four reconstruction-reliability levels using training-split percentiles. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv) · [Implementation](a2_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Target

For each tile:

1. Load the released encoder and its displacement normalization. Normalized displacement is `z = (displacement_mm - normalization_mean) / normalization_std`.
2. Read the tile data configuration. The stored displacement series has 294 observation times.
3. Keep at most 4096 points per tile; oversized tiles use deterministic tile-id sampling.
4. Mask the centered 30% synchronized temporal block:

```text
mask_start = 103
mask_end   = 191  # exclusive; hide stored indices 103 through 190
```

5. Run the frozen encoder and compute error only on masked finite positions:

```text
A21_masked_global_mse_z = mean((recon_z - target_z)^2)
A21_masked_global_rmse_mm = sqrt(A21_masked_global_mse_z) * normalization_std
```

### A22 reliability classes

The continuous target is primary. These corpus-relative class thresholds are fitted on the train split
only (`n=8000`) and then applied to all 10k tiles.

| class | rule | MSE_z threshold | RMSE_mm threshold |
|---|---:|---:|---:|
| `reconstructable` | mse <= train p75 | <= 0.099960 | <= 1.801 |
| `mildly_hard` | train p75 < mse <= train p95 | <= 0.154634 | <= 2.240 |
| `high_error` | train p95 < mse <= train p99 | <= 0.245728 | <= 2.824 |
| `unreliable` | mse > train p99 | > 0.245728 | > 2.824 |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| A21 | 10,000 | 0 | 0.0311969 | 0.0736837 | 0.153716 |

### A22 label distribution

| label | count | share |
|---|---:|---:|
| `reconstructable` | 7,509 | 75.09% |
| `mildly_hard` | 1,997 | 19.97% |
| `high_error` | 402 | 4.02% |
| `unreliable` | 92 | 0.92% |
