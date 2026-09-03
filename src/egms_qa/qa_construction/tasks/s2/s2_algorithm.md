# S2 Local Representation Support

## Task Scope

S2 describes whether a tile is locally supported by nearby train reference tiles in the encoder representation space. It is a representation construct, not an external geophysical truth label.

Official leaf tasks:

| ID | Field | Meaning |
|---|---|---|
| S21 | `S21_local_isolation_score` | mean cosine distance to the nearest 20 train neighbors in the train-defined CLS representation space |
| S22 | `S22_representation_rarity_class` | corpus-relative rarity class derived from S21 |

## Algorithms

Input token cache:

```text
data/encoder/tokens/encoder_tokens_10k.pt
```

Steps:

1. Extract CLS embeddings from `spatial_tokens[:, 0, :]`.
2. Fit `StandardScaler` on train CLS embeddings only.
3. Fit `PCA(n_components=25)` on train embeddings only.
4. L2-normalize PCA features.
5. Use train tiles as the reference library.
6. For every tile, find the nearest `k=20` train neighbors using cosine distance.
7. For train queries, exclude the tile itself.
8. Output:

```text
S21_local_isolation_score = mean(distance to nearest 20 train neighbors)
```

Higher values mean the tile is more isolated from the train reference manifold.

The final table stores only one S21 task value: `S21_local_isolation_score`.
The neighbor count `k=20` is an algorithm parameter, not a task output.

S22 uses train-only p75/p95/p99 thresholds on S21:

| S22 class | rule |
|---|---|
| `common` | `S21 <= 0.1768` |
| `unusual` | `0.1768 < S21 <= 0.2625` |
| `rare` | `0.2625 < S21 <= 0.3286` |
| `extreme` | `S21 > 0.3286` |

These thresholds are corpus-relative representation rarity labels, not physical or regulatory thresholds.

## k Selection

Candidate values were swept: `5, 10, 20, 50, 100, 200, 500`.

`k=20` is retained because `k=10/20/50` form a stable local-neighborhood range, while `k>=100` changes the rare tail substantially and behaves more like broad manifold sparsity than local isolation.

Key train quantiles for `k=20`:

| statistic | value |
|---|---:|
| p50 | 0.1323 |
| p75 | 0.1768 |
| p95 | 0.2625 |
| p99 | 0.3286 |

## S22 Threshold Choice

S21 is unimodal and right-skewed, so no clear natural valley is used as a threshold.

Candidate rules compared:

| rule | thresholds | all10k class counts |
|---|---|---|
| `q75_q95_q99` | 0.1768 / 0.2625 / 0.3286 | 7480 / 2011 / 403 / 106 |
| `q80_q95_q99` | 0.1919 / 0.2625 / 0.3286 | 7997 / 1494 / 403 / 106 |
| `q90_q975_q99` | 0.2306 / 0.2913 / 0.3286 | 8997 / 741 / 156 / 106 |
| `tukey_q75_1p5iqr_3iqr` | 0.1768 / 0.2965 / 0.4162 | 7480 / 2285 / 222 / 13 |

Official S22 rule: `q75_q95_q99`.

Reason: it directly matches the rarity story: common core, unusual upper quartile, rare top 5%, and extreme top 1%. The thresholds are train-only and corpus-relative.

Official all10k S22 counts:

| label | count | fraction |
|---|---:|---:|
| `common` | 7480 | 0.7480 |
| `unusual` | 2011 | 0.2011 |
| `rare` | 403 | 0.0403 |
| `extreme` | 106 | 0.0106 |

## File Inventory

```text
src/egms_qa/qa_construction/tasks/s2/s2_compute.py   # this script
src/egms_qa/qa_construction/tasks/s2/s2_algorithm.md # this document
outputs/tasks/s2/s2_final_table.csv     # generated canonical table (data release)
```

Running `s2_compute.py` also writes auxiliary diagnostics (class counts,
distribution plot, summary) next to the final table. The cluster count k and the
rarity thresholds were fixed by an offline selection step and are baked into the
script as constants.
