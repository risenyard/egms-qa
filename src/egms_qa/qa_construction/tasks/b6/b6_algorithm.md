# B6: Monitoring trigger

## Task overview

| task | description |
|---|---|
| **B6 group** | Combine velocity and acceleration typicality into a monitoring trigger. |
| B61 | Trigger when B36 velocity typicality or B42 acceleration typicality is classified as extreme. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv) · [Implementation](b6_compute.py)

## Key concepts

Should this tile enter the European monitoring watch list based on extreme velocity or acceleration typicality?

### Inputs

- `B36_european_velocity_typicality`
- `B42_european_acceleration_typicality`

## Algorithm steps

### Rule

```text
B61_monitoring_trigger = yes
    if B36_european_velocity_typicality == extreme
    or B42_european_acceleration_typicality == extreme
else:
    B61_monitoring_trigger = no
```

B61 is intentionally the only formal B6 task. Trigger type and broad monitoring summaries are not kept as separate leaf tasks because they are composite explanations rather than new targets.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.b6.b6_compute \
    --out-dir outputs/tasks-rebuilt/b6
```

The new table is written to `outputs/tasks-rebuilt/b6/b6_final_table.csv`.
The installed reference remains at `outputs/tasks/b6/b6_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| B3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) | `outputs/tasks/b3/b3_final_table.csv` |
| B4 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) | `outputs/tasks/b4/b4_final_table.csv` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### B61 label distribution

| label | count | share |
|---|---:|---:|
| `no` | 9,405 | 94.05% |
| `yes` | 595 | 5.95% |
