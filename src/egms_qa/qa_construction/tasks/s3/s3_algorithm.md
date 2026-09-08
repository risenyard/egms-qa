# S3: Representation–monitoring consistency

## Task overview

| task | description |
|---|---|
| **S3 group** | Compare representation rarity with rarity in the scalar monitoring indicators. |
| S31 | Difference between the training-reference percentile ranks of representation rarity and monitoring rarity. |
| S32 | Relation class derived from the S31 gap using train-fitted standardized thresholds. |
| S33 | Monitoring dimension with the largest axis-level reference score: quality, motion, spatial, or temporal. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s3/s3_final_table.csv) · [Implementation](s3_compute.py)

## Key concepts

S31 is a difference of percentile ranks on a train-defined `p0-p99` scale.
S32 classifies this difference. S33 identifies the most distinctive monitoring
dimension.

### Inputs

Encoder-side input:

- `S21_local_isolation_score`

A/B/C/D monitoring sentinel scalars:

- quality axis: `A41_median_rmse_mm`
- motion axis: `B33_vel_abs_p90_mm_yr`, `B41_acc_abs_p90`, `B51_seasonality_p90`
- spatial axis: `C11_noise_aware_moving_fraction`, `C21_spatial_concentration_score`,
  `C31_deformation_front_strength_mm_yr`, `C41_fast_tail_bin_fraction`
- temporal axis: `D12_curvature_strength`, `D13_changepoint_strength`,
  `D22_phase_coherence`, `D31_motion_intensification_mm_yr2`

The temporal axis reads curvature and changepoint strength directly from the
[D1 reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv).
The [D1 method](../d1/d1_algorithm.md) computes these geometry scores from the
same tile-median displacement series. Undefined scores are omitted when taking
the temporal-axis maximum; the remaining temporal indicators still contribute.

These sentinels are oriented so larger values mean stronger monitoring signal,
stronger structure, or worse observation quality.

## Algorithm steps

### Formula

1. Use train tiles only as the reference population.
2. Convert `S21_local_isolation_score` to `embedding_rarity_p`.
3. Convert every sentinel scalar to its train empirical percentile `p0-p99`.
4. For each axis, take the maximum sentinel percentile inside that axis:
   `quality`, `motion`, `spatial`, and `temporal`.
5. Average the four axis scores to form a raw monitoring-system score.
6. Convert that raw monitoring-system score again to train empirical percentile
   `monitoring_rarity_p`.
7. Output:

```text
S31_representation_monitoring_rarity_gap_p
    = embedding_rarity_p - monitoring_rarity_p
```

Positive values mean the encoder representation is rarer than expected from the
A/B/C/D monitoring scalar system. Negative values mean the scalar monitoring
system is rarer than the encoder representation.

### S32 Class Rule

Use only train tiles to estimate the S31 gap mean and standard deviation:

```text
train mean = 0.0000000
train std  = 35.0091216

-1.96 sigma = -68.6178783
-1.00 sigma = -35.0091216
+1.00 sigma = 35.0091216
+1.96 sigma = 68.6178783
```

Then assign:

```text
S31 <= -1.96 sigma                 -> strong_monitoring_excess
-1.96 sigma < S31 <= -1 sigma      -> moderate_monitoring_excess
-1 sigma < S31 < +1 sigma          -> aligned
+1 sigma <= S31 < +1.96 sigma      -> moderate_encoder_excess
S31 >= +1.96 sigma                 -> strong_encoder_excess
```

The thresholds are corpus-relative train z-score thresholds, not physical
thresholds. The strong classes correspond to an approximate two-sided 95%
normal-style deviation.

### S33 Dimension Rule

S33 reuses the same monitoring axes used by S31:

```text
quality_axis  = p(A41)
motion_axis   = max(p(B33), p(B41), p(B51))
spatial_axis  = max(p(C11), p(C21), p(C31), p(C41))
temporal_axis = max(p(D12), p(D13), p(D22), p(D31))

S33_monitoring_distinctive_dimension
    = argmax(quality_axis, motion_axis, spatial_axis, temporal_axis)
```

S33 is a monitoring-side explanation task. It explains which scalar monitoring
dimension is most distinctive, not why the encoder embedding itself is rare.
Exact ties are resolved deterministically by axis order:
`quality -> motion -> spatial -> temporal`.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.s3.s3_compute \
    --out-dir outputs/tasks-rebuilt/s3
```

The new table is written to `outputs/tasks-rebuilt/s3/s3_final_table.csv`.
The installed reference remains at `outputs/tasks/s3/s3_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| A4 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a4/a4_final_table.csv) | `outputs/tasks/a4/a4_final_table.csv` |
| B3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) | `outputs/tasks/b3/b3_final_table.csv` |
| B4 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) | `outputs/tasks/b4/b4_final_table.csv` |
| B5 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) | `outputs/tasks/b5/b5_final_table.csv` |
| C1 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c1/c1_final_table.csv) | `outputs/tasks/c1/c1_final_table.csv` |
| C2 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv) | `outputs/tasks/c2/c2_final_table.csv` |
| C3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c3/c3_final_table.csv) | `outputs/tasks/c3/c3_final_table.csv` |
| C4 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv) | `outputs/tasks/c4/c4_final_table.csv` |
| D1 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv) | `outputs/tasks/d1/d1_final_table.csv` |
| D2 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv) | `outputs/tasks/d2/d2_final_table.csv` |
| D3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d3/d3_final_table.csv) | `outputs/tasks/d3/d3_final_table.csv` |
| S2 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv) | `outputs/tasks/s2/s2_final_table.csv` |

To use a newly computed D1 table, add
`--d1-table outputs/tasks-rebuilt/d1/d1_final_table.csv`. Other input tables
continue to come from `--tasks-root` (the installed reference directory by default).

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s3/s3_final_table.csv). Numeric summaries use finite values.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| S31 | 10,000 | 0 | -58.1142 | -0.0773437 | 59.2546 |

### S32 label distribution

| label | count | share |
|---|---:|---:|
| `aligned` | 6,686 | 66.86% |
| `moderate_encoder_excess` | 1,435 | 14.35% |
| `moderate_monitoring_excess` | 1,387 | 13.87% |
| `strong_encoder_excess` | 250 | 2.50% |
| `strong_monitoring_excess` | 242 | 2.42% |

### S33 label distribution

| label | count | share |
|---|---:|---:|
| `spatial` | 3,871 | 38.71% |
| `temporal` | 3,461 | 34.61% |
| `motion` | 1,602 | 16.02% |
| `quality` | 1,066 | 10.66% |
