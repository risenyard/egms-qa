# S1: Representation anchors

## Task overview

S1 compares each tile’s encoder summary vector with six reference anchors learned from the training tiles. An anchor is a representative vector from a dense cluster of similar tiles. The profile names describe these reference groups; they are not independently verified physical classes.

| Task | Type | Output and relationship |
|---|---|---|
| S11 | Profile assignment | Names the nearest training-derived reference anchor: a representative tile vector from a dense cluster. |
| S12 | Numeric distance | Cosine distance to the anchor assigned by S11. Smaller values mean closer resemblance. |
| S13 | Numeric margin | Second-nearest anchor distance minus S12. Larger values mean a clearer preference for the assigned anchor. |
| S14 | Classification | Combines S12 distance and S13 margin into strong, weak/transition, or far/ambiguous anchor support. |
| S15 | Profile description | Provides the fixed text description of the S11 anchor profile; it is not a separate measurement. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s1/s1_final_table.csv) · [Implementation](s1_compute.py) · [Run and files](../README.md#run-s1)

## Algorithm steps

### Algorithm

Steps:

1. Extract summary embeddings from `spatial_tokens[:, 0, :]`.
2. Standardize each vector dimension using its training mean and standard deviation (`StandardScaler`).
3. Fit principal component analysis (`PCA(n_components=25)`) on the training vectors to retain 25 dimensions of variation.
4. Scale each reduced vector to unit length (L2 normalization).
5. Find dense training clusters with `sklearn.cluster.HDBSCAN(min_cluster_size=50, min_samples=80)`, a density-based clustering method.
6. Keep the 6 train dense-core clusters as reference anchors.
7. Use each cluster medoid as its anchor: the member vector with the smallest total cosine distance to other members.
8. For each tile, S12 is the nearest-anchor cosine distance (`1 - cosine similarity`), and S13 is second-nearest distance minus nearest distance.
9. Standardize distance and margin using their training statistics, then fit a Gaussian mixture model (GMM) to the training pairs `[S12 distance, S13 margin]`. This groups similar assignment patterns; the Bayesian information criterion (BIC) selects six components from candidates with one to six components.
10. Convert mixture components to S14 labels using their training medians in
    the original distance and margin units. Start with
    `transition_or_weakly_anchored`; assign `strongly_anchored` to the component
    with the smallest median distance minus median margin. Then assign
    `far_or_ambiguous_from_reference_anchors` to the components with the largest
    median distance or smallest median margin; these assignments take precedence.

### Anchor Profiles

| anchor | S11 profile | S15 description |
|---:|---|---|
| 0 | `mixed_acceleration_complex_trend_reference` | large mixed dynamic reference with elevated acceleration and complex trend behavior |
| 1 | `spring_trend_acceleration_reference` | spring-associated trend and acceleration mixed reference |
| 2 | `coherent_autumn_seasonal_reference` | compact autumn-seasonal reference with high phase coherence |
| 3 | `extreme_localized_deformation_front_reference` | small extreme reference with strong localized deformation, front strength, and fast-tail extent |
| 4 | `stable_low_activity_background_reference` | large low-activity stable background reference with low velocity and acceleration |
| 5 | `summer_trend_seasonal_mixed_reference` | summer-associated trend-seasonal mixed reference with relatively diffuse spatial structure |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s1/s1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

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
