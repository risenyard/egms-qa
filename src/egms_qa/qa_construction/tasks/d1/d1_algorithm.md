# D1: Temporal trend geometry

## Task overview

| task | description |
|---|---|
| **D1 group** | Describe curvature, changepoint strength, and trend shape in the tile-median displacement history. |
| D11 | Trend-shape class obtained from the strong-curvature and strong-changepoint flags. |
| D12 | Curvature strength combining quadratic-fit improvement with the normalized curvature effect. |
| D13 | Changepoint strength combining piecewise-linear fit improvement with the slope-change effect. |
| D14 | Time of the selected changepoint, reported when the changepoint-strength criterion is satisfied. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv) · [Implementation](d1_compute.py)

## Key concepts

D1 describes the geometry of each tile's median displacement history through
curvature, changepoint strength, trend shape, and changepoint time.

### Inputs

For each tile, read the stored model-ready EGMS displacement interval `[0,294)`
and take the median displacement over all points at each epoch. This stored
interval corresponds exactly to `[8,302)` on the original 304-step prepared
axis. The epoch cadence is 6 days and the original index offset is retained in
the data config for physical-time calculations.

## Algorithm steps

### Fitted Geometry

All fits include intercept, normalized time, and annual plus semiannual sine/cosine terms.

Time is normalized to the interval [-1,1]. Fits use four Huber reweighting
iterations with NumPy least squares, a median-absolute-deviation scale factor
of 1.4826, Huber tuning constant 1.345, and numerical epsilon 1e-10. At least
200 valid median epochs are required. The breakpoint scan uses 52 rounded,
evenly spaced positions from stored index 36 through 257, inclusive. The
fixed iteration count and score definitions are part of the data contract;
a generic change-point detector would produce different target quantities.

- Linear baseline: seasonal linear model.
- D12 candidate: add a quadratic term.
- D13/D14 candidate: scan piecewise-linear hinge breakpoints in the central time window.
- Complex diagnostic: quadratic plus scanned hinge, kept only as a diagnostic.

The primary scalar primitives are:

- `D12_curvature_strength`: quadratic improvement times normalized curvature effect.
- `D13_changepoint_strength`: piecewise-linear improvement times normalized slope-change effect.
- `D14_dominant_changepoint_time_year`: fractional-year time of the D13 changepoint, only for D13-strong tiles.

For each alternative model, the gain is `max(0, 1 - SSE_alternative/SSE_linear)`.
Curvature strength multiplies this gain by `log(1 + effect)`, where the effect
is the absolute quadratic coefficient times the standard deviation of squared
normalized time, divided by the linear-fit residual scale. Changepoint
strength uses the best hinge coefficient and hinge standard deviation in the
same calculation. The linear residual scale is the square root of linear SSE
divided by the valid epoch count minus the six baseline coefficients.

### Train-Fitted Thresholds

Thresholds are corpus-relative and fitted on train only:

```json
{
  "threshold_mode": "train_p85_primitives",
  "d12_strong_quantile": 0.85,
  "d12_strong_threshold": 0.20298744933908217,
  "d13_strong_quantile": 0.85,
  "d13_strong_threshold": 0.6056297951274243,
  "d14_time_source": "D14_candidate_changepoint_time_year",
  "cp_bins": 8
}
```

D12/D13 strong flags:

```text
D11_is_curved = D12_curvature_strength >= train p85
D11_has_break = D13_changepoint_strength >= train p85
```

D11 class rule:

| has strong break | has strong curve | D11_long_term_trend_shape |
|---|---|---|
| no | no | `linear_trend` |
| no | yes | `curved_trend` |
| yes | no | `stage_change` |
| yes | yes | `complex_trend` |

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.d1.d1_compute \
    --out-dir outputs/tasks-rebuilt/d1
```

The new table is written to `outputs/tasks-rebuilt/d1/d1_final_table.csv`.
The installed reference remains at `outputs/tasks/d1/d1_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| NPZ source tiles | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/tree/main/artifacts/source_tiles) | `data/tiles/` |
| Split manifest | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/split_manifest.parquet) | `data/encoder/manifest/split.parquet` |
| Data configuration | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/metadata/data_config.json) | `data/encoder/manifest/data_config.json` |

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

The published NPZ values and split remain fixed. Recomputing the table
preserves its categorical targets, missingness, and numerical values up to
floating-point rounding across linear-algebra implementations.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

### Numeric targets

| task | defined | missing | p05 | median | p95 |
|---|---:|---:|---:|---:|---:|
| D12 | 10,000 | 0 | 0 | 0.0113259 | 0.457342 |
| D13 | 10,000 | 0 | 0.00238516 | 0.0824703 | 0.940769 |
| D14 | 1,509 | 8,491 | 2019.94 | 2021.43 | 2022.78 |

### D11 label distribution

| label | count | share |
|---|---:|---:|
| `linear_trend` | 8,218 | 82.18% |
| `complex_trend` | 1,229 | 12.29% |
| `stage_change` | 280 | 2.80% |
| `curved_trend` | 273 | 2.73% |
