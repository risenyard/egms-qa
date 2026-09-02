# S1 Global Representation Anchor

## Task Scope

S1 describes where each tile sits relative to train-defined encoder CLS reference anchors. It is a representation construct, not an external geophysical class.

Official leaf tasks:

| ID | Field | Meaning |
|---|---|---|
| S11 | `S11_reference_anchor_profile` | readable profile of the nearest reference anchor |
| S12 | `S12_reference_anchor_distance` | cosine distance from the tile to the nearest train-defined anchor |
| S13 | `S13_reference_anchor_margin` | distance gap between the nearest and second-nearest anchors |
| S14 | `S14_reference_assignment_status` | distribution-driven assignment status |
| S15 | `S15_reference_anchor_profile_description` | concise explanation of the nearest anchor profile |

`reference_anchor_id` is an auxiliary technical identifier for reproducibility. It is not a separate task.

## Algorithm

Input token cache:

```text
data/encoder/tokens/encoder_tokens_10k.pt
```

Steps:

1. Extract CLS embeddings from `spatial_tokens[:, 0, :]`.
2. Fit `StandardScaler` on train CLS embeddings only.
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

## Anchor Profiles

| anchor | S11 profile | S15 description |
|---:|---|---|
| 0 | `mixed_acceleration_complex_trend_reference` | large mixed dynamic reference with elevated acceleration and complex trend behavior |
| 1 | `spring_trend_acceleration_reference` | spring-associated trend and acceleration mixed reference |
| 2 | `coherent_autumn_seasonal_reference` | compact autumn-seasonal reference with high phase coherence |
| 3 | `extreme_localized_deformation_front_reference` | small extreme reference with strong localized deformation, front strength, and fast-tail extent |
| 4 | `stable_low_activity_background_reference` | large low-activity stable background reference with low velocity and acceleration |
| 5 | `summer_trend_seasonal_mixed_reference` | summer-associated trend-seasonal mixed reference with relatively diffuse spatial structure |

## Result Counts

S14:

| label | count | fraction |
|---|---:|---:|
| `transition_or_weakly_anchored` | 7576 | 0.7576 |
| `far_or_ambiguous_from_reference_anchors` | 1242 | 0.1242 |
| `strongly_anchored` | 1182 | 0.1182 |

Nearest anchor:

| anchor | count | fraction |
|---:|---:|---:|
| 0 | 2574 | 0.2574 |
| 1 | 1789 | 0.1789 |
| 2 | 853 | 0.0853 |
| 3 | 1134 | 0.1134 |
| 4 | 2687 | 0.2687 |
| 5 | 963 | 0.0963 |

## File Inventory

```text
src/egms_qa/qa/tasks/s1/s1_compute.py   # this script
src/egms_qa/qa/tasks/s1/s1_algorithm.md # this document
outputs/tasks/s1/s1_final_table.csv     # generated canonical table (data release)
```

Running `s1_compute.py` also writes auxiliary diagnostics (anchor counts,
GMM components, distribution plot, summary) next to the final table.
