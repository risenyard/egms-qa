# D1: Temporal trend geometry

## Task overview

D1 describes the shape of a tile’s displacement history. At each observation time, it takes the median displacement across points. It then checks whether that curve bends gradually or contains a breakpoint where its slope changes.

| Task | Type | Output and relationship |
|---|---|---|
| D11 | Classification | Combines D12 curvature and D13 changepoint strength into linear, curved, stage-change, or complex trend shape. |
| D12 | Numeric score | Measures how much a curved trend improves on a linear fit, accounting for the size of the curvature. |
| D13 | Numeric score | Measures how much a change in slope improves on a linear fit, accounting for the size of that change. |
| D14 | Numeric time (year) | Reports the D13 breakpoint time as a fractional year, only when D13 meets the strong-change criterion. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv) · [Implementation](d1_compute.py) · [Run the task system](../README.md#run-the-task-system)

For each tile, read the stored model-ready EGMS displacement interval `[0,294)`
and take the median displacement over all points at each epoch. This stored
interval corresponds exactly to `[8,302)` on the original 304-step prepared
axis. The epoch cadence is 6 days and the original index offset is retained in
the data config for physical-time calculations.

Threshold fitting requires at least 100 training tiles with sufficient valid
epochs and finite D12/D13 scores.

## Algorithm steps

### Fitted geometry

All fits include an intercept (constant offset), normalized time, and annual plus semiannual sine/cosine terms to account for seasonality. A quadratic term allows a smooth bend; a hinge term allows the slope to change at a candidate breakpoint. SSE below means the sum of squared fitting errors.

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

### Training thresholds and D11 classes

Thresholds are corpus-relative and fitted on train only:

| Score | Strong threshold (training 85th percentile) |
|---|---:|
| D12 curvature strength | 0.20298744933908217 |
| D13 changepoint strength | 0.6056297951274243 |

D14 reports the fractional-year time of the best D13 breakpoint only when
D13 reaches its strong threshold; otherwise D14 is missing.

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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately. In numeric tables, p05 and p95
are the 5th and 95th percentiles.

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
