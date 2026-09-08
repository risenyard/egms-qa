# B5: Seasonality strength

## Task overview

| task | description |
|---|---|
| **B5 group** | Summarize the upper tail of the source seasonality measurements. |
| B51 | 90th percentile of the valid point-level seasonality field. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) · [Implementation](b5_compute.py)

## Key concepts

How strong is the tile's seasonality signal?

### Intentional Exclusions

- No classification label is produced.
- No seasonal phase is inferred here; phase belongs to the D group.
- No European typicality threshold is used here; B5 only measures seasonality strength.

## Algorithm steps

### Formula

```text
B51_seasonality_p90 = percentile(point seasonality, 90)
```

### Output

- `B51_seasonality_p90`: continuous scalar.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.b5.b5_compute \
    --out-dir outputs/tasks-rebuilt/b5
```

The new table is written to `outputs/tasks-rebuilt/b5/b5_final_table.csv`.
The installed reference remains at `outputs/tasks/b5/b5_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B51 | 10,000 | 0 | 0.9 | 1.4 | 3.1 |
