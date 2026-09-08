# B5: Seasonality strength

## Task overview

B5 summarizes seasonal strength across the observation points in a tile. It reads the source `seasonality` field, which records the point’s seasonal displacement signal.

| Task | Type | Output and relationship |
|---|---|---|
| B51 | Numeric value | 90th percentile of the source point seasonality field. It summarizes seasonal strength; no class is assigned. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) · [Implementation](b5_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Formula

```text
B51_seasonality_p90 = percentile(point seasonality, 90)
```

### Output

- `B51_seasonality_p90`: continuous scalar.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| B51 | 10,000 | 0 | 0.9 | 1.4 | 3.1 |
