# A1: Representation stability

## Task overview

A1 checks whether removing most observation points changes the encoder’s representation of a tile. A tile is one spatial sample containing point displacement histories; its summary token is the encoder’s vector for the whole sample.

| Task | Type | Output and relationship |
|---|---|---|
| A11 | Numeric score | Average change in the whole-tile representation after keeping only 20% of its points. Smaller values mean greater stability. |
| A12 | Classification | Groups A11 into four stability levels using percentiles fitted on the training split. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv) · [Implementation](a1_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Target

For each tile:

1. Use the cached full-tile EGMS summary token as `summary_full`.
2. Keep 20% of points with deterministic random seeds `0,1,2,3,4`.
3. Re-run the frozen EGMS encoder on each subsampled tile, using the original full-tile center. Call each new summary vector `summary_sub`.
4. Compute per-seed angular drift:

```text
drift_seed = arccos(clip(cosine(summary_full, summary_sub), -1, 1)) / pi
```

5. Average across seeds:

```text
A11_global_angular_drift = mean_seed(drift_seed)
```

Lower drift means the global representation is more stable. The cosine measures alignment between the two vectors; dividing their angle by pi puts drift on a 0–1 scale.

### A12 stability classes

The continuous target is primary. The four-class label is a corpus-relative tail
label. Thresholds are fitted on the train split only and then applied to all 10k
tiles. Here p75, p95, and p99 mean the 75th, 95th, and 99th percentiles:

| class | rule | threshold |
|---|---|---|
| `stable` | drift <= train p75 | <= 0.007791 |
| `mildly_sensitive` | train p75 < drift <= train p95 | <= 0.011695 |
| `highly_sensitive` | train p95 < drift <= train p99 | <= 0.016930 |
| `extreme` | drift > train p99 | > 0.016930 |

The same thresholds in degrees are 1.402, 2.105, and 3.047 degrees.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a1/a1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| A11 | 10,000 | 0 | 0.00329956 | 0.00586604 | 0.0116798 |

### A12 label distribution

| label | count | share |
|---|---:|---:|
| `stable` | 7,472 | 74.72% |
| `mildly_sensitive` | 2,032 | 20.32% |
| `highly_sensitive` | 395 | 3.95% |
| `extreme` | 101 | 1.01% |
