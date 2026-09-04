# D1 Temporal Trend Geometry Algorithm

This folder is the canonical EGMS-QA D1 delivery: a deterministic temporal-trend
geometry table.

## Inputs

For each tile, read the stored model-ready EGMS displacement interval `[0,294)`
and take the median displacement over all points at each epoch. This stored
interval corresponds exactly to `[8,302)` on the original 304-step prepared
axis. The epoch cadence is 6 days and the original index offset is retained in
the data config for physical-time calculations.

## Fitted Geometry

All fits include intercept, normalized time, and annual plus semiannual sine/cosine terms.

- Linear baseline: seasonal linear model.
- D12 candidate: add a quadratic term.
- D13/D14 candidate: scan piecewise-linear hinge breakpoints in the central time window.
- Complex diagnostic: quadratic plus scanned hinge, kept only as a diagnostic.

The primary scalar primitives are:

- `D12_curvature_strength`: quadratic improvement times normalized curvature effect.
- `D13_changepoint_strength`: piecewise-linear improvement times normalized slope-change effect.
- `D14_dominant_changepoint_time_year`: fractional-year time of the D13 changepoint, only for D13-strong tiles.

## Train-Fitted Thresholds

Thresholds are corpus-relative and fitted on train only:

```json
{
  "threshold_mode": "train_p85_primitives",
  "d12_strong_quantile": 0.85,
  "d12_strong_threshold": 0.20298744933908217,
  "d13_strong_quantile": 0.85,
  "d13_strong_threshold": 0.6056297951274243,
  "d14_time_source": "D13_exp_dominant_changepoint_time_year",
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

## Current All-10k Counts

D11:

| value | count |
|---|---:|
| `linear_trend` | 8218 |
| `complex_trend` | 1229 |
| `stage_change` | 280 |
| `curved_trend` | 273 |

D11_has_break:

| value | count |
|---|---:|
| `no_strong_break` | 8491 |
| `has_break` | 1509 |

D11_is_curved:

| value | count |
|---|---:|
| `not_curved` | 8498 |
| `curved` | 1502 |

## File Inventory

- `d1_final_table.csv`: canonical D11-D14 targets plus diagnostics needed to reproduce them.
- `d1_summary.json`: thresholds, class counts, and score summaries.
- `d1_algorithm.md`: this algorithm note.
- `d1_compute.py`: canonicalizes/promotes the deterministic geometry D1 table.
