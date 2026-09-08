# S4: Local representation structure

## Task overview

| task | description |
|---|---|
| **S4 group** | Describe variation and concentration among the valid spatial-cell representations inside a tile. |
| S41 | RMS deviation from the cell-token centroid divided by the RMS magnitude of the cell tokens. |
| S42 | Local-structure class derived from train-reference thresholds on S41. |
| S43 | Gini concentration of the cell-token residual magnitudes around their centroid. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv) · [Implementation](s4_compute.py)

## Key concepts

### Inputs

Current tile tokens only:

- `spatial_tokens[:, 1:65, :]`: the 64 patch tokens.
- `token_mask[:, 1:65]`: valid patch-token mask.

S4 does not use the tile-summary token, geographic neighbors, A/B/C/D labels, S11 anchors, or
reference libraries.

### Interpretation Boundary

S4 is not a direct physical ground-truth label. It describes
encoder-perceived local spatial structure inside a tile. Because the 64 patch
tokens are tied to the 8x8 tile layout, the task has geographic monitoring
meaning, but the claim remains representation-level:

```text
Does the encoder see local spatial structure inside this tile?
```

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
fewer valid patches. S43 is delivered as a continuous scalar and is not
classified because its empirical distribution is continuous without a clear
natural breakpoint.

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

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.s4.s4_compute \
    --out-dir outputs/tasks-rebuilt/s4
```

The new table is written to `outputs/tasks-rebuilt/s4/s4_final_table.csv`.
The installed reference remains at `outputs/tasks/s4/s4_final_table.csv`.
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
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/s4/s4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

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
