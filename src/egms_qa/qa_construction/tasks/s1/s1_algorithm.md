# S1: Representation anchors

## Task overview

| task | description |
|---|---|
| **S1 group** | Describe a tile's position relative to reference anchors in the encoder representation space. |
| S11 | Assigned reference-anchor profile. |
| S12 | Distance to the nearest reference anchor. |
| S13 | Assignment margin between the nearest competing anchors. |
| S14 | Assignment status describing how the representation is supported by the reference profiles. |
| S15 | Descriptive category for the assigned reference-anchor profile. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s1/s1_final_table.csv) · [Implementation](s1_compute.py)

## Key concepts

S1 describes where each tile sits relative to train-defined encoder summary-token reference anchors. It is a representation construct, not an external geophysical class.

## Algorithm steps

### Algorithm

Input token cache:

[HF input](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/representations/egms_tokens_10k.pt) · installed path: `data/encoder/tokens/egms_tokens_10k.pt`

Steps:

1. Extract summary embeddings from `spatial_tokens[:, 0, :]`.
2. Fit `StandardScaler` on train summary embeddings only.
3. Fit `PCA(n_components=25)` on train embeddings only.
4. L2-normalize PCA features.
5. Fit `sklearn.cluster.HDBSCAN(min_cluster_size=50, min_samples=80)` on train features.
6. Keep the 6 train dense-core clusters as reference anchors.
7. Use each cluster medoid as the anchor vector.
8. For every tile, compute nearest-anchor distance and nearest-vs-second margin.
9. Fit a train-only 2D Gaussian mixture over `[S12 distance, S13 margin]`; BIC selects `k=6`.
10. Merge GMM components into S14:
    - `strongly_anchored`
    - `transition_or_weakly_anchored`
    - `far_or_ambiguous_from_reference_anchors`

### Anchor Profiles

| anchor | S11 profile | S15 description |
|---:|---|---|
| 0 | `mixed_acceleration_complex_trend_reference` | large mixed dynamic reference with elevated acceleration and complex trend behavior |
| 1 | `spring_trend_acceleration_reference` | spring-associated trend and acceleration mixed reference |
| 2 | `coherent_autumn_seasonal_reference` | compact autumn-seasonal reference with high phase coherence |
| 3 | `extreme_localized_deformation_front_reference` | small extreme reference with strong localized deformation, front strength, and fast-tail extent |
| 4 | `stable_low_activity_background_reference` | large low-activity stable background reference with low velocity and acceleration |
| 5 | `summer_trend_seasonal_mixed_reference` | summer-associated trend-seasonal mixed reference with relatively diffuse spatial structure |

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.s1.s1_compute \
    --out-dir outputs/tasks-rebuilt/s1
```

The new table is written to `outputs/tasks-rebuilt/s1/s1_final_table.csv`.
The installed reference remains at `outputs/tasks/s1/s1_final_table.csv`.
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
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s1/s1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| S12 | 10,000 | 0 | 0.095143 | 0.414417 | 0.807097 |
| S13 | 10,000 | 0 | 0.0356778 | 0.467927 | 0.902359 |

### S11 label distribution

| label | count | share |
|---|---:|---:|
| `stable_low_activity_background_reference` | 2,687 | 26.87% |
| `mixed_acceleration_complex_trend_reference` | 2,574 | 25.74% |
| `spring_trend_acceleration_reference` | 1,789 | 17.89% |
| `extreme_localized_deformation_front_reference` | 1,134 | 11.34% |
| `summer_trend_seasonal_mixed_reference` | 963 | 9.63% |
| `coherent_autumn_seasonal_reference` | 853 | 8.53% |

### S14 label distribution

| label | count | share |
|---|---:|---:|
| `transition_or_weakly_anchored` | 7,576 | 75.76% |
| `far_or_ambiguous_from_reference_anchors` | 1,242 | 12.42% |
| `strongly_anchored` | 1,182 | 11.82% |

### S15 label distribution

| label | count | share |
|---|---:|---:|
| `large low-activity stable background reference with low velocity and acceleration` | 2,687 | 26.87% |
| `large mixed dynamic reference with elevated acceleration and complex trend behavior` | 2,574 | 25.74% |
| `spring-associated trend and acceleration mixed reference` | 1,789 | 17.89% |
| `small extreme reference with strong localized deformation, front strength, and fast-tail extent` | 1,134 | 11.34% |
| `summer-associated trend-seasonal mixed reference with relatively diffuse spatial structure` | 963 | 9.63% |
| `compact autumn-seasonal reference with high phase coherence` | 853 | 8.53% |
