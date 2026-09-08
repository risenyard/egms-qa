# A5: Monitoring usability

## Task overview

| task | description |
|---|---|
| **A5 group** | Combine representation stability, reconstruction reliability, coverage, and noise into a monitoring usability decision. |
| A51 | Usability class derived from the severity of the A12, A22, A32, and A42 quality flags. |
| A52 | Reason for the usability decision, including rules for multiple simultaneous quality issues. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a5/a5_final_table.csv) · [Implementation](a5_compute.py)

## Key concepts

A51/A52 are the A-group monitoring usability gate and reason tasks:

> Can this tile be used normally for downstream QA and monitoring, or should the interpretation carry a quality warning?

A51/A52 are deterministic roll-ups of A12/A22/A32/A42. They are not encoder-advantage tasks and
does not introduce a new learned model.

### Inputs

- `A12_representation_stability_class`
- `A22_reconstruction_reliability_class`
- `A32_spatial_coverage_class`
- `A42_noise_level_class`

A12 and A22 use train-fitted corpus-relative thresholds before this roll-up is
computed. A5 itself does not fit any thresholds.

## Algorithm steps

### Class Rules

`unreliable` if any severe issue is present:

```text
A12 == extreme
OR A22 == unreliable
OR A32 == highly_fragmented
OR A42 == very_high_noise
```

`caution` if no severe issue is present, but any caution issue is present:

```text
A12 == highly_sensitive
OR A22 == high_error
OR A32 == sparse
OR A42 == high_noise
```

Otherwise:

```text
usable
```

### Reason Rules

Severe reasons:

- `unstable_representation`: A12 extreme
- `poor_reconstruction`: A22 unreliable
- `fragmented_coverage`: A32 highly_fragmented
- `very_high_noise`: A42 very_high_noise

Caution reasons:

- `sensitive_representation`: A12 highly_sensitive
- `high_reconstruction_error`: A22 high_error
- `sparse_coverage`: A32 sparse
- `high_noise`: A42 high_noise

If multiple severe issues are present, or a severe issue appears together with a
caution issue, the reason is `multiple_quality_issues`. If multiple caution
issues are present without a severe issue, the reason is `multiple_minor_issues`.
If no issue is present, the reason is `stable_inputs`.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.a5.a5_compute \
    --out-path outputs/tasks-rebuilt/a5/a5_final_table.csv
```

The new table is written to `outputs/tasks-rebuilt/a5/a5_final_table.csv`.
The installed reference remains at `outputs/tasks/a5/a5_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| A1 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv) | `outputs/tasks/a1/a1_final_table.csv` |
| A2 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a2/a2_final_table.csv) | `outputs/tasks/a2/a2_final_table.csv` |
| A3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv) | `outputs/tasks/a3/a3_final_table.csv` |
| A4 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv) | `outputs/tasks/a4/a4_final_table.csv` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a5/a5_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### A51 label distribution

| label | count | share |
|---|---:|---:|
| `usable` | 7,588 | 75.88% |
| `caution` | 1,897 | 18.97% |
| `unreliable` | 515 | 5.15% |

### A52 label distribution

| label | count | share |
|---|---:|---:|
| `stable_inputs` | 7,588 | 75.88% |
| `high_noise` | 1,138 | 11.38% |
| `multiple_quality_issues` | 443 | 4.43% |
| `multiple_minor_issues` | 356 | 3.56% |
| `sparse_coverage` | 294 | 2.94% |
| `sensitive_representation` | 108 | 1.08% |
| `fragmented_coverage` | 40 | 0.40% |
| `very_high_noise` | 20 | 0.20% |
| `unstable_representation` | 10 | 0.10% |
| `poor_reconstruction` | 2 | 0.02% |
| `high_reconstruction_error` | 1 | 0.01% |
