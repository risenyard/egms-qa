# A51/A52 Algorithm

## Task

A51/A52 are the A-group monitoring usability gate and reason tasks:

> Can this tile be used normally for downstream QA and monitoring, or should the interpretation carry a quality warning?

A51/A52 are deterministic roll-ups of A12/A22/A32/A42. They are not encoder-advantage tasks and
does not introduce a new learned model.

## Inputs

- `A12_representation_stability_class`
- `A22_reconstruction_reliability_class`
- `A32_spatial_coverage_class`
- `A42_noise_level_class`

A12 and A22 use train-fitted corpus-relative thresholds before this roll-up is
computed. A5 itself does not fit any thresholds.

## Class Rules

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

## Reason Rules

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

## Counts

Class counts in the final table:

| class | count | fraction |
|---|---:|---:|
| `usable` | 7588 | 0.7588 |
| `caution` | 1897 | 0.1897 |
| `unreliable` | 515 | 0.0515 |

## Files

- `a5_final_table.csv`: final all10k usability table.
- `a5_compute.py`: deterministic computation from A11/A21/A31/A41 final tables.
