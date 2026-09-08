# A5: Monitoring usability

## Task overview

A5 combines four quality checks into a tile usability decision and its reason. It reads [A12 representation stability](../a1/a1_algorithm.md), [A22 reconstruction reliability](../a2/a2_algorithm.md), [A32 spatial coverage](../a3/a3_algorithm.md), and [A42 measurement noise](../a4/a4_algorithm.md).

| Task | Type | Output and relationship |
|---|---|---|
| A51 | Combined classification | Combines four input quality classes: representation stability, reconstruction reliability, spatial coverage, and measurement noise. |
| A52 | Reason category | Explains the A51 decision by naming the quality issue, multiple issues, or stable inputs. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a5/a5_final_table.csv) · [Implementation](a5_compute.py) · [Run and files](../README.md#run-a5)

A12 and A22 use training-split, corpus-relative thresholds. A5 applies the class rules below without fitting additional thresholds.

## Algorithm steps

### A51 usability rules

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

### A52 reason rules

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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a5/a5_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

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
