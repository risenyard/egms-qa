# A3: Spatial observation coverage

## Task overview

| task | description |
|---|---|
| **A3 group** | Describe how widely the observations occupy the tile's 8×8 spatial grid. |
| A31 | Fraction of the 64 grid cells containing observations, computed from the cached point counts. |
| A32 | Coverage class assigned from fixed thresholds on the occupied-cell fraction. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv) · [Implementation](a3_compute.py)

## Key concepts

A31 measures spatial observation coverage:

> Are the observations spread across the tile, or concentrated into only a few spatial bins?

## Algorithm steps

### Target

A31 uses the 8x8 spatial bin counts already stored in the EGMS encoder token cache:

```text
occupied_bins = count(point_count_per_bin > 0)
A31_valid_bin_fraction_8x8 = occupied_bins / 64
```

This is a raw observation-support target. It is not an encoder-advantage task.

### Classes

The class label uses fixed structural thresholds, not empirical percentiles:

| class | rule | meaning |
|---|---:|---|
| `well_spread` | fraction >= 0.75 | observations cover most of the tile |
| `moderate_gaps` | 0.50 <= fraction < 0.75 | visible coverage gaps |
| `sparse` | 0.25 <= fraction < 0.50 | sparse spatial support |
| `highly_fragmented` | fraction < 0.25 | very fragmented support |

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.a3.a3_compute \
    --out-path outputs/tasks-rebuilt/a3/a3_final_table.csv
```

The new table is written to `outputs/tasks-rebuilt/a3/a3_final_table.csv`.
The installed reference remains at `outputs/tasks/a3/a3_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| Encoder token cache | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/representations/egms_tokens_10k.pt) | `data/encoder/tokens/egms_tokens_10k.pt` |

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/a3/a3_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| A31 | 10,000 | 0 | 0.484375 | 0.9375 | 1 |

### A32 label distribution

| label | count | share |
|---|---:|---:|
| `well_spread` | 7,836 | 78.36% |
| `moderate_gaps` | 1,663 | 16.63% |
| `sparse` | 434 | 4.34% |
| `highly_fragmented` | 67 | 0.67% |
