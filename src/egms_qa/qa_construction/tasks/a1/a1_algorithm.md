# A11/A12 Algorithm

## Task

A11 measures severe global representation instability:

> When a tile loses most observations, does the encoder still form a stable global tile representation?

## Target

For each tile:

1. Use the cached full-tile EGMS encoder CLS token as `CLS_full`.
2. Keep 20% of points with deterministic random seeds `0,1,2,3,4`.
3. Re-run the frozen EGMS encoder on each subsampled tile, using the original full-tile center.
4. Compute per-seed angular drift:

```text
drift_seed = arccos(clip(cosine(CLS_full, CLS_sub), -1, 1)) / pi
```

5. Average across seeds:

```text
A11_global_angular_drift = mean_seed(drift_seed)
```

Lower drift means the global representation is more stable. A11 does not use coverage score, patch-token similarity, cluster consistency, scalar reconstruction, or manual coverage thresholds.

## Four Classes

The continuous target is primary. The four-class label is a corpus-relative tail
label. Thresholds are fitted on the train split only and then applied to all 10k
tiles:

| class | rule | threshold |
|---|---|---|
| `stable` | drift <= train p75 | <= 0.007791 |
| `mildly_sensitive` | train p75 < drift <= train p95 | <= 0.011695 |
| `highly_sensitive` | train p95 < drift <= train p99 | <= 0.016930 |
| `extreme` | drift > train p99 | > 0.016930 |

The same thresholds in degrees are 1.402, 2.105, and 3.047 degrees.

Class counts in the final table:

| class | count | fraction |
|---|---:|---:|
| `stable` | 7472 | 0.7472 |
| `mildly_sensitive` | 2032 | 0.2032 |
| `highly_sensitive` | 395 | 0.0395 |
| `extreme` | 101 | 0.0101 |

## Files

- `a1_final_table.csv`: final table only.
- `a1_compute.py`: shard-level encoder run.
- `a1_combine_shards.py`: shard combiner; writes only `a1_final_table.csv` and removes temporary `work/` by default.

Current EGMS encoder package: `data/encoder/`.
Current EGMS encoder token cache: `data/encoder/tokens/encoder_tokens_10k.pt`.
