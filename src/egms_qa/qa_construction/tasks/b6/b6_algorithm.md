# B6: Monitoring trigger

## Task overview

B6 combines two existing classifications: [B36 velocity typicality](../b3/b3_algorithm.md) and [B42 acceleration typicality](../b4/b4_algorithm.md). Each compares motion strength with a fixed European candidate-pool distribution; an extreme value in either triggers the task’s monitoring flag.

| Task | Type | Output and relationship |
|---|---|---|
| B61 | Yes/no classification | Triggers monitoring when either the velocity or acceleration reference class is extreme. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv) · [Implementation](b6_compute.py) · [Run and files](../README.md#run-b6)

## Algorithm steps

### Rule

```text
B61_monitoring_trigger = yes
    if B36_european_velocity_typicality == extreme
    or B42_european_acceleration_typicality == extreme
else:
    B61_monitoring_trigger = no
```

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### B61 label distribution

| label | count | share |
|---|---:|---:|
| `no` | 9,405 | 94.05% |
| `yes` | 595 | 5.95% |
