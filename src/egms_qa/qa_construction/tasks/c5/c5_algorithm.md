# C5: Spatial monitoring context

## Task overview

| task | description |
|---|---|
| **C5 group** | Combine overall motion and local spatial evidence into a monitoring interpretation. |
| C51 | Monitoring priority derived from the prescribed upstream motion and spatial classes. |
| C52 | Local-risk category indicating spatial evidence that the mean-motion class may obscure. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c5/c5_final_table.csv) · [Implementation](c5_compute.py)

## Key concepts

### Inputs

C5 is a composite family using existing EGMS-QA outputs:

- `B61_monitoring_trigger`
- `C33_deformation_front_strength_class`
- `B22_mean_subsidence_intensity_band`
- `B35_worst_point_significance`

No new point-level, bin-level, or encoder computation is introduced.

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

The C52 rule preserves the old EGMS-QA C10 logic: mean severity was only slight or
mild, but the local worst-point significance was high.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.c5.c5_compute \
    --out-dir outputs/tasks-rebuilt/c5
```

The new table is written to `outputs/tasks-rebuilt/c5/c5_final_table.csv`.
The installed reference remains at `outputs/tasks/c5/c5_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| B2 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b2/b2_final_table.csv) | `outputs/tasks/b2/b2_final_table.csv` |
| B3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) | `outputs/tasks/b3/b3_final_table.csv` |
| B6 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b6/b6_final_table.csv) | `outputs/tasks/b6/b6_final_table.csv` |
| C3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c3/c3_final_table.csv) | `outputs/tasks/c3/c3_final_table.csv` |

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c5/c5_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

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
