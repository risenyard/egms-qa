# C4: Fast-tail spatial extent

## Task overview

| task | description |
|---|---|
| **C4 group** | Measure whether high-velocity tail cells occupy a substantial spatial area. |
| C41 | Fraction of valid cells whose absolute-velocity tail statistic exceeds the fixed reference threshold. |
| C42 | Extent class combining the number of high-tail cells and their fraction of valid cells. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv) · [Implementation](c4_compute.py)

## Key concepts

Do high-velocity tail bins occupy a spatial area, or are they confined to a very small part of the tile?

### Inputs

- `coords`: point coordinates.
- `mean_velocity`: point-level mean velocity in mm/yr.

The 10k final table uses:

[HF split.parquet](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) (installed at `data/encoder/manifest/split.parquet`)

The fast-motion threshold `T_fast` is **corpus-relative**: it is the p95 of the
per-bin abs-velocity p90 over the full European candidate pool (83,323 tiles).
The `final` subcommand uses the fixed value in the code
(`FAST_THRESHOLD_MM_YR = 4.8`, in mm/yr). Re-estimating `T_fast` with the
optional `reference` subcommand requires the full candidate pool, which is
not included in the Dataset. That subcommand generates a reference JSON that
can be supplied to `final` through `--reference-json`.



### Intentional Exclusions

- C4 does not measure the strongest velocity itself; B33/B36 handle velocity-tail magnitude.
- C4 does not measure all meaningful motion; C11 handles noise-aware moving-point fraction.
- C4 does not measure global concentration; C21/C22 handle spatial concentration over all motion.
- C4 does not measure a deformation front; C31/C33 handle adjacent-bin jumps.

## Algorithm steps

### Reference Distribution

For every European candidate tile:

```text
1. Center point coordinates by the tile coordinate mean.
2. Split the tile into an 8x8 local grid.
3. Keep bins with at least 5 finite points.
4. For each valid bin:
   bin_abs_velocity_p90 = percentile(abs(point mean_velocity), 90)
5. Pool all valid-bin bin_abs_velocity_p90 values across Europe.
```

The full-Europe reference run used 83,323 candidate tiles and produced 3,351,762
valid-bin values.

| statistic | bin_abs_velocity_p90 mm/yr |
|---|---:|
| p50 | 1.800000 |
| p75 | 2.580000 |
| p90 | 3.720000 |
| p95 | 4.800000 |
| p97.5 | 6.120000 |
| p99 | 8.200000 |

The distribution is unimodal and right-skewed with a long high-velocity tail.
`p90=3.72` is too broad for a fast-tail extent target, while `p99=8.20` is closer
to an extreme-only cutoff. EGMS-QA freezes:

```text
T_fast = full-Europe bin-level p95 = 4.800000 mm/yr
```

For reference, the trimmed log-bulk fit gives `z2 = 5.287757 mm/yr`, close to p95,
which supports p95 as a stable high-tail threshold rather than an arbitrary cut.

### Final Formula

For each 10k VQA tile:

```text
fast_tail_bin = bin_abs_velocity_p90 >= T_fast
C41_fast_tail_bin_fraction =
    count(fast_tail_bin over valid bins) / count(valid bins)
```

C42 is a derived extent class from C41. It uses structure-based thresholds rather
than positive-C41 quantile cuts:

```text
none       : fast_tail_bin_count = 0
sparse     : fast_tail_bin_count = 1 or 2
localized  : fast_tail_bin_count >= 3 and C41 < 0.25
extensive  : fast_tail_bin_count >= 3 and C41 >= 0.25
```

The `sparse` class uses absolute bin count because one or two high-tail bins are
not enough to claim a spatial area. The `extensive` class uses a quarter of valid
bins as an interpretable area-coverage threshold.

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.c4.c4_compute final \
    --out-dir outputs/tasks-rebuilt/c4
```

The new table is written to `outputs/tasks-rebuilt/c4/c4_final_table.csv`.
The installed reference remains at `outputs/tasks/c4/c4_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |

The `final` subcommand uses the published fixed threshold. The optional
`reference` subcommand needs a separate full candidate-pool manifest, which is
not distributed with this Dataset. Its threshold JSON and diagnostics are
generated locally; they are not downloadable release files.

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/c4/c4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| C41 | 10,000 | 0 | 0 | 0 | 0.230769 |

### C42 label distribution

| label | count | share |
|---|---:|---:|
| `none` | 5,740 | 57.40% |
| `sparse` | 2,321 | 23.21% |
| `localized` | 1,482 | 14.82% |
| `extensive` | 457 | 4.57% |

### Reference and selection evidence

#### Reference calibration

Derived reference distribution over the pool (for provenance):

| statistic | value (mm/yr) |
|---|---|
| p50 | 1.80 |
| p90 | 3.72 |
| **p95 (= `T_fast`)** | **4.80** |
| p99 | 8.20 |
| p99.9 | 18.60 |

(log-bulk fit: mu=0.594, sigma=0.536, over log values trimmed to [p1, p99];
n_tiles=83,323, n_valid_bins=3,351,762.)
