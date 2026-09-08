# S2: Local representation support

## Task overview

S2 measures how isolated a tile’s encoder summary vector is from nearby training-tile vectors. A summary vector is the encoder’s representation of the whole tile; nearby vectors indicate similar representations, rather than geographic proximity.

| Task | Type | Output and relationship |
|---|---|---|
| S21 | Numeric distance | Mean cosine distance to the 20 nearest training tiles after reducing the representation to 25 features. Larger values mean greater isolation. |
| S22 | Classification | Converts S21 into common, unusual, rare, or extreme using training-split percentiles. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv) · [Implementation](s2_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Algorithms

Steps:

1. Extract summary embeddings from `spatial_tokens[:, 0, :]`.
2. Standardize each vector dimension using its training mean and standard deviation (`StandardScaler`).
3. Fit principal component analysis (`PCA`) on training vectors to retain 25 dimensions of variation.
4. Scale each reduced vector to unit length (L2 normalization).
5. Use train tiles as the reference library.
6. Find the `k=20` nearest training vectors by cosine distance (`1 - cosine similarity`).
7. For train queries, exclude the tile itself.
8. Output:

```text
S21_local_isolation_score = mean(distance to nearest 20 train neighbors)
```

Higher values mean the tile is more isolated from the training-vector population.

S22 uses train-only p75/p95/p99 thresholds on S21:

| S22 class | rule |
|---|---|
| `common` | `S21 <= 0.1768` |
| `unusual` | `0.1768 < S21 <= 0.2625` |
| `rare` | `0.2625 < S21 <= 0.3286` |
| `extreme` | `S21 > 0.3286` |

These thresholds are corpus-relative representation rarity labels, not physical or regulatory thresholds.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| S21 | 10,000 | 0 | 0.055714 | 0.132214 | 0.263487 |

### S22 label distribution

| label | count | share |
|---|---:|---:|
| `common` | 7,480 | 74.80% |
| `unusual` | 2,011 | 20.11% |
| `rare` | 403 | 4.03% |
| `extreme` | 106 | 1.06% |

### Reference and selection evidence

#### k Selection

Here `k` is the number of training neighbors. Candidate values were compared: `5, 10, 20, 50, 100, 200, 500`.

`k=20` is retained because `k=10/20/50` form a stable local-neighborhood range, while `k>=100` changes the rare tail substantially and measures isolation over a broader part of the reference population.

Key train quantiles for `k=20`:

| statistic | value |
|---|---:|
| p50 | 0.1323 |
| p75 | 0.1768 |
| p95 | 0.2625 |
| p99 | 0.3286 |

#### S22 Threshold Choice

S21 is unimodal and right-skewed, so no clear natural valley is used as a threshold.

Candidate rules compared (q denotes a quantile; IQR is the 75th minus 25th percentile; Tukey cutoffs add 1.5 or 3 IQRs to q75):

| rule | thresholds | all10k class counts |
|---|---|---|
| `q75_q95_q99` | 0.1768 / 0.2625 / 0.3286 | 7480 / 2011 / 403 / 106 |
| `q80_q95_q99` | 0.1919 / 0.2625 / 0.3286 | 7997 / 1494 / 403 / 106 |
| `q90_q975_q99` | 0.2306 / 0.2913 / 0.3286 | 8997 / 741 / 156 / 106 |
| `tukey_q75_1p5iqr_3iqr` | 0.1768 / 0.2965 / 0.4162 | 7480 / 2285 / 222 / 13 |

Released S22 rule: `q75_q95_q99`.

The selected cutoffs define a common core, an unusual upper quartile, a rare top 5%, and an extreme top 1%. The thresholds are train-only and corpus-relative.
