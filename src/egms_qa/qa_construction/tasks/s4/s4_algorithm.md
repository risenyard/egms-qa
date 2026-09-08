# S4: Local representation structure

## Task overview

S4 measures variation among the encoder vectors for a tile’s 8×8 spatial cells. Each cell vector, also called a patch token, represents observations in that part of the tile. These outputs describe the encoder representation, rather than a directly measured deformation pattern.

| Task | Type | Output and relationship |
|---|---|---|
| S41 | Numeric score | Variation among spatial-cell vectors divided by their overall magnitude. Larger values mean stronger local differences in the representation. |
| S42 | Classification | Converts S41 into four local-structure levels using training-split percentiles. |
| S43 | Numeric score | Gini inequality of cell-vector deviations from their mean. Complements S41 by showing whether variation is concentrated in a few cells. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv) · [Implementation](s4_compute.py) · [Run the task system](../README.md#run-the-task-system)

The cached `spatial_tokens[:, 1:65, :]` contain the 64 cell vectors, and `token_mask[:, 1:65]` identifies valid cells. The centroid is their mean vector; RMS means root mean square. S41 and S43 use only the current tile’s valid cells, while S42 uses thresholds fitted on training tiles.

## Algorithm steps

### Formula

For a tile with valid patch tokens `p_i`:

```text
patch_centroid = mean_i(p_i)
S41_encoder_perceived_local_structure_strength
    = RMS_i(p_i - patch_centroid) / RMS_i(p_i)
```

Low values indicate that the encoder sees the tile as locally coherent. High
values indicate stronger encoder-perceived local spatial structure.

For S43:

```text
residual_i = ||p_i - patch_centroid||
S43_encoder_perceived_local_structure_concentration
    = Gini_i(residual_i)
```

Low values indicate that local structure is more evenly distributed across
valid patches. High values indicate that local structure is concentrated in
fewer valid patches. Gini measures inequality: zero indicates equal residual magnitudes; larger values indicate greater inequality.

### S42 Class Rule

S41 is right-skewed with a high local-structure tail, so S42 uses train-only
tail-aware thresholds:

```text
S41 <= train p50 = 0.1240134898
    -> spatially_coherent

train p50 < S41 <= train p90 = 0.1638370953
    -> weak_local_structure

train p90 < S41 <= train p95 = 0.1817143741
    -> clear_local_structure

S41 > train p95
    -> strong_local_structure
```

These thresholds are corpus-relative train-distribution labels, not physical
thresholds.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| S41 | 10,000 | 0 | 0.0884594 | 0.124018 | 0.180914 |
| S43 | 10,000 | 0 | 0.144162 | 0.189268 | 0.250342 |

### S42 label distribution

| label | count | share |
|---|---:|---:|
| `spatially_coherent` | 5,000 | 50.00% |
| `weak_local_structure` | 4,016 | 40.16% |
| `strong_local_structure` | 493 | 4.93% |
| `clear_local_structure` | 491 | 4.91% |
