# S2: Local representation support

## Task overview

| task | description |
|---|---|
| **S2 group** | Measure local isolation from the training-set reference representations. |
| S21 | Mean cosine distance to the nearest 20 training neighbors after train-fitted standardization and PCA. |
| S22 | Corpus-relative rarity class derived from train-only percentiles of S21. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv) · [Implementation](s2_compute.py)

## Key concepts

S2 describes whether a tile is locally supported by nearby train reference tiles in the encoder representation space. It is a representation construct, not an external geophysical truth label.

## Algorithm steps

### Algorithms

Input token cache:

[HF input](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/representations/egms_tokens_10k.pt) · installed path: `data/encoder/tokens/egms_tokens_10k.pt`

Steps:

1. Extract summary embeddings from `spatial_tokens[:, 0, :]`.
2. Fit `StandardScaler` on train summary embeddings only.
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

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.s2.s2_compute \
    --out-dir outputs/tasks-rebuilt/s2
```

The new table is written to `outputs/tasks-rebuilt/s2/s2_final_table.csv`.
The installed reference remains at `outputs/tasks/s2/s2_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| Encoder token cache | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/representations/egms_tokens_10k.pt) | `data/encoder/tokens/egms_tokens_10k.pt` |

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s2/s2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

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

Candidate values were swept: `5, 10, 20, 50, 100, 200, 500`.

`k=20` is retained because `k=10/20/50` form a stable local-neighborhood range, while `k>=100` changes the rare tail substantially and behaves more like broad manifold sparsity than local isolation.

Key train quantiles for `k=20`:

| statistic | value |
|---|---:|
| p50 | 0.1323 |
| p75 | 0.1768 |
| p95 | 0.2625 |
| p99 | 0.3286 |

#### S22 Threshold Choice

S21 is unimodal and right-skewed, so no clear natural valley is used as a threshold.

Candidate rules compared:

| rule | thresholds | all10k class counts |
|---|---|---|
| `q75_q95_q99` | 0.1768 / 0.2625 / 0.3286 | 7480 / 2011 / 403 / 106 |
| `q80_q95_q99` | 0.1919 / 0.2625 / 0.3286 | 7997 / 1494 / 403 / 106 |
| `q90_q975_q99` | 0.2306 / 0.2913 / 0.3286 | 8997 / 741 / 156 / 106 |
| `tukey_q75_1p5iqr_3iqr` | 0.1768 / 0.2965 / 0.4162 | 7480 / 2285 / 222 / 13 |

Released S22 rule: `q75_q95_q99`.

Reason: it directly matches the rarity story: common core, unusual upper quartile, rare top 5%, and extreme top 1%. The thresholds are train-only and corpus-relative.
