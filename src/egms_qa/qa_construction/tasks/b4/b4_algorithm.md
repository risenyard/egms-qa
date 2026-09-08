# B4: Acceleration strength

## Task overview

| task | description |
|---|---|
| **B4 group** | Summarize strong acceleration and its reference-distribution typicality. |
| B41 | 90th percentile of absolute point acceleration. |
| B42 | Acceleration typicality class relative to the specified European reference distribution. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) · [Implementation](b4_compute.py)

## Key concepts

How strong is the tile's acceleration signal, and how typical is that acceleration relative to the European reference distribution?

## Algorithm steps

### Targets

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

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.b4.b4_compute \
    --out-dir outputs/tasks-rebuilt/b4
```

The new table is written to `outputs/tasks-rebuilt/b4/b4_final_table.csv`.
The installed reference remains at `outputs/tasks/b4/b4_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B41 | 10,000 | 0 | 0.33 | 0.62 | 1.23615 |

### B42 label distribution

| label | count | share |
|---|---:|---:|
| `typ_low` | 3,153 | 31.53% |
| `low` | 2,945 | 29.45% |
| `typ_high` | 2,589 | 25.89% |
| `high` | 1,025 | 10.25% |
| `extreme` | 288 | 2.88% |
