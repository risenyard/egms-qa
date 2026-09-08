# S3: Representation–monitoring consistency

## Task overview

S3 compares two kinds of rarity: isolation of the encoder representation and unusual values in measured monitoring indicators. It reads [S21 local representation isolation](../s2/s2_algorithm.md) and the quality, motion, spatial, and temporal indicators listed below. Both sides are ranked against training tiles before comparison.

| Task | Type | Output and relationship |
|---|---|---|
| S31 | Numeric rank gap | Representation-rarity rank minus monitoring-rarity rank. Positive values mean greater rarity on the representation side. |
| S32 | Classification | Converts S31 into aligned, monitoring-excess, or encoder-excess levels using the training mean and standard deviation. |
| S33 | Explanation category | Names the strongest monitoring dimension used in S31: quality, motion, spatial, or temporal. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s3/s3_final_table.csv) · [Implementation](s3_compute.py) · [Run and files](../README.md#run-s3)

The monitoring inputs are grouped into four dimensions. A percentile rank
places a value within the training distribution; this task uses a 0–99 scale.

| Dimension | Input tasks and meaning |
|---|---|
| Quality | [A41](../a4/a4_algorithm.md): median point measurement RMSE |
| Motion | [B33](../b3/b3_algorithm.md): absolute velocity p90; [B41](../b4/b4_algorithm.md): absolute acceleration p90; [B51](../b5/b5_algorithm.md): seasonality p90 |
| Spatial | [C11](../c1/c1_algorithm.md): moving-point fraction; [C21](../c2/c2_algorithm.md): motion concentration; [C31](../c3/c3_algorithm.md): neighboring-cell velocity contrast; [C41](../c4/c4_algorithm.md): fast-cell fraction |
| Temporal | [D12 and D13](../d1/d1_algorithm.md): curvature and changepoint strengths; [D22](../d2/d2_algorithm.md): phase coherence; [D31](../d3/d3_algorithm.md): signed motion intensification |

Here p90 means the 90th percentile of point values, and RMSE means root mean
squared error. Larger input values represent higher noise, stronger motion or
structure, greater phase agreement, or more positive intensification.
Undefined values are omitted from the maximum within each dimension.

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
monitoring indicators listed above. Negative values mean the scalar monitoring
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
