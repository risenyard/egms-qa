# D4 Temporal Composition Algorithm

## Current Scope

This folder is the D4 temporal composition family. The delivered target is:

- `D41_temporal_dominant_process`
- `D42_temporal_evolution_archetype`

D41 summarizes which broad temporal process dominates a tile:

```text
low_activity / trend_dominant / seasonal_dominant / acceleration_dominant / mixed
```

D41 is a composite summary. It does not introduce a new time-series model. It
uses already delivered B-family primitive strengths.

D42 is a readable temporal archetype. It does not introduce a new measurement,
model, or threshold. It uses D41 as the routing label and then attaches the most
relevant already delivered temporal detail from D11, D21, or D31.

## Inputs

| process | input scalar | meaning |
|---|---|---|
| trend | `B33_vel_abs_p90_mm_yr` | long-term motion strength |
| seasonal | `B51_seasonality_p90` | annual seasonal strength |
| acceleration | `B41_acc_abs_p90` | recent acceleration strength |

The three inputs have different physical units, so D41 does not compare raw
values directly. It first converts each input to a train-split empirical
percentile rank and applies those train-fitted ranks to all 10k tiles.

## Formula

For each process:

```text
D41_trend_rank        = train_empirical_percentile_rank(B33_vel_abs_p90_mm_yr)
D41_seasonal_rank     = train_empirical_percentile_rank(B51_seasonality_p90)
D41_acceleration_rank = train_empirical_percentile_rank(B41_acc_abs_p90)
```

Then:

```text
D41_top_rank = max(D41_trend_rank, D41_seasonal_rank, D41_acceleration_rank)
D41_second_rank = second_largest(rank values)
D41_dominance_margin = D41_top_rank - D41_second_rank
```

Class rule:

| class | rule |
|---|---|
| `low_activity` | `D41_top_rank < 0.30` |
| `trend_dominant` | trend is top process and `D41_dominance_margin >= 0.15` |
| `seasonal_dominant` | seasonal is top process and `D41_dominance_margin >= 0.15` |
| `acceleration_dominant` | acceleration is top process and `D41_dominance_margin >= 0.15` |
| `mixed` | not low activity, but no process leads by at least 0.15 rank |

The `0.30` and `0.15` thresholds are corpus-relative design choices selected
after inspecting the top-rank and margin distributions and class proportions.
They are not physical thresholds.

## Interpretation

- `low_activity`: all three process strengths are weak relative to the 10k corpus.
- `trend_dominant`: long-term motion strength clearly leads.
- `seasonal_dominant`: seasonal strength clearly leads.
- `acceleration_dominant`: acceleration strength clearly leads.
- `mixed`: multiple temporal processes are comparable.

## D42 Temporal Evolution Archetype

D42 converts the D-family temporal story into one answerable class:

```text
D42_temporal_evolution_archetype
```

Input columns:

| source | input | role |
|---|---|---|
| D41 | `D41_temporal_dominant_process` | route to low/trend/seasonal/acceleration/mixed story |
| D11 | `D11_long_term_trend_shape` | trend-dominant subtype |
| D21 | `D21_dominant_seasonal_peak` | seasonal-dominant clear vs unclear phase |
| D31 | `D31_motion_intensification_mm_yr2` | acceleration-dominant direction |
| D41 ranks | `D41_trend_rank`, `D41_seasonal_rank`, `D41_acceleration_rank` | mixed top-two process pair |

Class rule:

| route | D42 class |
|---|---|
| `D41 = low_activity` | `low_activity` |
| `D41 = trend_dominant` and `D11 = linear_trend` | `linear_trend_dominated` |
| `D41 = trend_dominant` and `D11 = curved_trend` | `curved_trend_dominated` |
| `D41 = trend_dominant` and `D11 = stage_change / complex_trend` | `regime_change_trend_dominated` |
| `D41 = seasonal_dominant` and D21 has a clear seasonal peak | `coherent_seasonal_dominated` |
| `D41 = seasonal_dominant` and `D21 = no_clear_seasonal_peak` | `incoherent_seasonal_dominated` |
| `D41 = acceleration_dominant` and `D31 > 0` | `intensifying_acceleration_dominated` |
| `D41 = acceleration_dominant` and `D31 < 0` | `weakening_acceleration_dominated` |
| `D41 = acceleration_dominant` and D31 is missing or zero | `uncertain_direction_acceleration_dominated` |
| `D41 = mixed` | unordered top-two rank pair: `trend_seasonal_mixed`, `trend_acceleration_mixed`, or `seasonal_acceleration_mixed` |

## Intentional Exclusions

- D41 does not replace D1, D2, or D3. It summarizes their broad process context.
- D41 uses B33/B51/B41 as base strengths, not D11/D21/D31 labels, because it is
  a composition comparison across process magnitudes.
- D42 does not use D12/D13/D14/D22/D23/D24/D32/D33/D34/D35. Those columns are
  important diagnostics, but adding them to D42 would make the class a mixed
  heuristic rather than a readable story label.
- D42 does not add any new threshold. Its only thresholded input is D41, whose
  thresholds are already labeled corpus-relative.

## Current All-10k Result

| class | count |
|---|---:|
| `low_activity` | 864 |
| `trend_dominant` | 1529 |
| `seasonal_dominant` | 1825 |
| `acceleration_dominant` | 1093 |
| `mixed` | 4689 |

D42 result:

| class | count |
|---|---:|
| `low_activity` | 864 |
| `linear_trend_dominated` | 765 |
| `curved_trend_dominated` | 389 |
| `regime_change_trend_dominated` | 375 |
| `coherent_seasonal_dominated` | 1429 |
| `incoherent_seasonal_dominated` | 396 |
| `intensifying_acceleration_dominated` | 487 |
| `weakening_acceleration_dominated` | 290 |
| `uncertain_direction_acceleration_dominated` | 316 |
| `trend_seasonal_mixed` | 1321 |
| `trend_acceleration_mixed` | 1905 |
| `seasonal_acceleration_mixed` | 1463 |

Rank profile:

| class | trend rank mean | seasonal rank mean | acceleration rank mean | top rank mean | margin mean |
|---|---:|---:|---:|---:|---:|
| `low_activity` | 0.129 | 0.151 | 0.105 | 0.207 | 0.091 |
| `trend_dominant` | 0.687 | 0.254 | 0.362 | 0.687 | 0.294 |
| `seasonal_dominant` | 0.286 | 0.701 | 0.302 | 0.701 | 0.335 |
| `acceleration_dominant` | 0.358 | 0.382 | 0.745 | 0.745 | 0.288 |
| `mixed` | 0.624 | 0.591 | 0.641 | 0.734 | 0.065 |

## File Inventory

- `d4_final_table.csv`: canonical D41/D42 target table plus base inputs, ranks,
  and the upstream D11/D21/D31 columns needed to reproduce D42.
- `d4_final_summary.json`: class counts, rule, and rank summaries.
- `d4_final_distribution.png`: D41/D42 class counts and rank/margin distributions.
- `d4_compute.py`: recomputes D41/D42 from B3/B4/B5 and D1/D2/D3 final tables.
