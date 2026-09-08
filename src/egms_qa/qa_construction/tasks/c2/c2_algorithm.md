# C2: Spatial concentration

## Task overview

C2 measures whether motion is spread evenly across a tile or concentrated in a few places. It groups observation points into an 8×8 grid and compares the mean absolute vertical velocity of its cells.

| Task | Type | Output and relationship |
|---|---|---|
| C21 | Numeric score | Gini inequality of cell mean motion magnitudes. Larger values mean motion is concentrated in fewer cells. |
| C22 | Classification | Converts C21 into four concentration levels using training-split percentiles. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv) · [Implementation](c2_compute.py) · [Run the task system](../README.md#run-the-task-system)

The Gini score measures inequality: zero means equal cell magnitudes, and larger values mean greater differences between cells. Only cells with enough observations enter the calculation.

## Algorithm steps

### Formula

For each point:

```text
point_motion_magnitude = abs(point mean_velocity)
```

For each tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split a 7000 m square around that center into an 8x8 grid (875 m cells).
3. Keep bins with at least 5 points.
4. For each valid bin:
   bin_mean_abs_velocity = mean(point_motion_magnitude)
5. C21_spatial_concentration_score = Gini(bin_mean_abs_velocity over valid bins)
```

Higher values mean the motion magnitude is more spatially concentrated.

### C22 concentration class

C22 is derived from the observed C21 train-split distribution. The distribution
is unimodal and right-skewed with a long high-concentration tail, so C22 uses
train p10/p50/p90 rather than equal quartiles. It is a corpus-relative spatial
organization class:

| class | rule |
|---|---:|
| `diffuse` | C21 <= 0.093756 |
| `mildly_concentrated` | 0.093756 < C21 <= 0.181511 |
| `concentrated` | 0.181511 < C21 <= 0.312813 |
| `highly_concentrated` | C21 > 0.312813 |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c2/c2_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| C21 | 10,000 | 0 | 0.0784815 | 0.18154 | 0.362198 |

### C22 label distribution

| label | count | share |
|---|---:|---:|
| `mildly_concentrated` | 4,015 | 40.15% |
| `concentrated` | 3,998 | 39.98% |
| `highly_concentrated` | 1,003 | 10.03% |
| `diffuse` | 984 | 9.84% |
