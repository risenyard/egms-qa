# S4 Algorithm: Encoder-Perceived Local Spatial Structure

## Delivered Tasks

`S41_encoder_perceived_local_structure_strength`

S41 asks whether the encoder sees a tile as spatially coherent, or whether the
valid 8x8 patch tokens contain stronger local spatial structure.

`S42_encoder_perceived_local_structure_class`

S42 is the question-friendly class derived from S41.

`S43_encoder_perceived_local_structure_concentration`

S43 asks whether the encoder-perceived local structure is broadly distributed
across valid patch positions or concentrated in fewer local patches.

## Inputs

Current tile tokens only:

- `spatial_tokens[:, 1:65, :]`: the 64 patch tokens.
- `token_mask[:, 1:65]`: valid patch-token mask.

S4 does not use CLS, geographic neighbors, A/B/C/D labels, S11 anchors, or
reference libraries.

## Interpretation Boundary

S4 is not a direct physical ground-truth label. It describes
encoder-perceived local spatial structure inside a tile. Because the 64 patch
tokens are tied to the 8x8 tile layout, the task has geographic monitoring
meaning, but the claim remains representation-level:

```text
Does the encoder see local spatial structure inside this tile?
```

## Formula

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

## S42 Class Rule

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

## Current All10k Result

`S41_encoder_perceived_local_structure_strength`:

```text
mean = 0.128880
std  = 0.034542
p05  = 0.088459
p25  = 0.108963
p50  = 0.124018
p75  = 0.141626
p95  = 0.180914
p99  = 0.249896
```

`S42_encoder_perceived_local_structure_class` all10k counts:

```text
spatially_coherent       5000
weak_local_structure     4016
clear_local_structure     491
strong_local_structure    493
```

`S43_encoder_perceived_local_structure_concentration`:

```text
mean = 0.192186
std  = 0.033269
p05  = 0.144162
p25  = 0.168749
p50  = 0.189268
p75  = 0.211706
p95  = 0.250342
p99  = 0.288332
```

Diagnostic correlations:

```text
Spearman(S43, S41) = 0.391856
Spearman(S43, valid_patch_count) = -0.406812
```

## File Inventory

- `s4_final_table.csv`: final task table with `tile_id`, `split`, S41, S42, and S43.
- `s4_compute.py`: reproducible computation script.
- `s4_summary.json`: distribution and reproducibility summary.
- `s4_distribution.png`: final distribution diagnostic.
