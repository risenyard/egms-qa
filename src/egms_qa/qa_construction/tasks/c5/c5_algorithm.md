# C5: Spatial monitoring context

## Task overview

C5 combines overall motion and local spatial evidence. It reads [B61 monitoring trigger](../b6/b6_algorithm.md), [C33 front strength](../c3/c3_algorithm.md), [B22 mean subsidence band](../b2/b2_algorithm.md), and [B35 local motion strength](../b3/b3_algorithm.md). The resulting labels describe monitoring evidence, rather than an assessed probability of damage.

| Task | Type | Output and relationship |
|---|---|---|
| C51 | Combined classification | Uses the monitoring-trigger and front-strength classes to assign no, standard, or high monitoring priority. |
| C52 | Yes/no classification | Flags strong local motion when the mean subsidence band is low or low-mid. Complements C51 with evidence that an average can obscure. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c5/c5_final_table.csv) · [Implementation](c5_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Rules

C51:

```text
if B61_monitoring_trigger == no:
    C51_monitoring_priority = none
elif C33_deformation_front_strength_class == very_sharp:
    C51_monitoring_priority = high
else:
    C51_monitoring_priority = standard
```

C52:

```text
if B22_mean_subsidence_intensity_band in {low, low_mid}
and B35_worst_point_significance in {high, very_high}:
    C52_hidden_local_risk = yes
else:
    C52_hidden_local_risk = no
```

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c5/c5_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### C51 label distribution

| label | count | share |
|---|---:|---:|
| `none` | 9,405 | 94.05% |
| `high` | 396 | 3.96% |
| `standard` | 199 | 1.99% |

### C52 label distribution

| label | count | share |
|---|---:|---:|
| `no` | 9,593 | 95.93% |
| `yes` | 407 | 4.07% |
