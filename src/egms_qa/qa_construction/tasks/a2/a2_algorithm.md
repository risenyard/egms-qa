# A2: Masked reconstruction reliability

## Task overview

| task | description |
|---|---|
| **A2 group** | Assess whether the encoder reconstructs a shared missing temporal interval reliably. |
| A21 | Mean squared reconstruction error on the synchronized masked interval, measured in normalized displacement units. |
| A22 | Reconstruction reliability class obtained from train-fitted percentiles of A21. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv) · [Implementation](a2_compute.py)

## Key concepts

A21 measures temporal masked reconstructability:

> When a synchronized temporal block is hidden, can the EGMS encoder reconstruct the missing tile time structure?

## Algorithm steps

### Target

For each tile:

1. Load the EGMS encoder weights and config from
   [HF encoder.safetensors](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/encoder.safetensors) (installed at `data/encoder/checkpoint/encoder.safetensors`) and [encoder config.json](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/config.json).
2. Read the tile data config from [HF data_config.json](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) (installed at `data/encoder/manifest/data_config.json`).
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

### Thresholds

The continuous target is primary. Class thresholds are fitted on the train split
only (`n=8000`) and then applied to all 10k tiles.

| class | rule | MSE_z threshold | RMSE_mm threshold |
|---|---:|---:|---:|
| `reconstructable` | mse <= train p75 | <= 0.099960 | <= 1.801 |
| `mildly_hard` | train p75 < mse <= train p95 | <= 0.154634 | <= 2.240 |
| `high_error` | train p95 < mse <= train p99 | <= 0.245728 | <= 2.824 |
| `unreliable` | mse > train p99 | > 0.245728 | > 2.824 |

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.a2.a2_compute \
    --out-dir outputs/tasks-rebuilt/a2/work/shards/shard_0 \
    --num-shards 1 --shard-index 0 --device cuda:0
python -m egms_qa.qa_construction.tasks.a2.a2_combine_shards \
    --base-dir outputs/tasks-rebuilt/a2/work --num-shards 1 \
    --out-path outputs/tasks-rebuilt/a2/a2_final_table.csv
```

This example uses one shard and CUDA. For multiple shards, use matching
`--num-shards` values in the compute and combine commands.

The new table is written to `outputs/tasks-rebuilt/a2/a2_final_table.csv`.
The installed reference remains at `outputs/tasks/a2/a2_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |
| Data configuration | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) | `data/encoder/manifest/data_config.json` |
| Encoder weights | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/encoder.safetensors) | `data/encoder/checkpoint/encoder.safetensors` |
| Encoder configuration | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/config.json) | `data/encoder/checkpoint/config.json` |
| Encoder normalization | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/normalization.json) | `data/encoder/checkpoint/normalization.json` |
| Encoder training recipe | [HF file](https://huggingface.co/risenyard/egms-qa-encoder/blob/main/training_args.json) | `data/encoder/checkpoint/training_args.json` |

Intermediate files are written under `work/shards/`. The combiner removes
that work directory by default. Add `--keep-work` to retain those local files.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

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
