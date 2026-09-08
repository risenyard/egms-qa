# D4: Temporal composition

## Task overview

D4 combines existing motion indicators into a dominant temporal process and a more specific evolution category. It uses [B33 absolute velocity strength](../b3/b3_algorithm.md), [B51 seasonal strength](../b5/b5_algorithm.md), and [B41 absolute acceleration strength](../b4/b4_algorithm.md). The added detail comes from [D11 trend shape](../d1/d1_algorithm.md), [D21 seasonal peak](../d2/d2_algorithm.md), and [D31 motion intensification](../d3/d3_algorithm.md).

| Task | Type | Output and relationship |
|---|---|---|
| D41 | Combined classification | Compares training-reference ranks of velocity, seasonality, and acceleration strength to name the dominant process, mixed activity, or low activity. |
| D42 | Combined classification | Adds trend shape, seasonal agreement, intensification direction, or the leading process pair to the D41 result. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d4/d4_final_table.csv) · [Implementation](d4_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

The three strengths have different units. Convert each to an empirical percentile rank fitted on the training split, then apply that same mapping to every split. Higher ranks mean stronger values relative to the training population.

### Formula

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

### D42 Temporal Evolution Archetype

D42 refines D41 with the most relevant temporal detail:

```text
D42_temporal_evolution_archetype
```

Input columns:

| source | input | role |
|---|---|---|
| D41 | `D41_temporal_dominant_process` | select low activity, trend, seasonal, acceleration, or mixed detail |
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

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d4/d4_final_table.csv).

### D41 label distribution

| label | count | share |
|---|---:|---:|
| `mixed` | 4,689 | 46.89% |
| `seasonal_dominant` | 1,825 | 18.25% |
| `trend_dominant` | 1,529 | 15.29% |
| `acceleration_dominant` | 1,093 | 10.93% |
| `low_activity` | 864 | 8.64% |

### D42 label distribution

| label | count | share |
|---|---:|---:|
| `trend_acceleration_mixed` | 1,905 | 19.05% |
| `seasonal_acceleration_mixed` | 1,463 | 14.63% |
| `coherent_seasonal_dominated` | 1,429 | 14.29% |
| `linear_trend_dominated` | 1,420 | 14.20% |
| `trend_seasonal_mixed` | 1,321 | 13.21% |
| `low_activity` | 864 | 8.64% |
| `intensifying_acceleration_dominated` | 487 | 4.87% |
| `incoherent_seasonal_dominated` | 396 | 3.96% |
| `uncertain_direction_acceleration_dominated` | 316 | 3.16% |
| `weakening_acceleration_dominated` | 290 | 2.90% |
| `regime_change_trend_dominated` | 80 | 0.80% |
| `curved_trend_dominated` | 29 | 0.29% |

Trend-dominated subtypes by split:

| label | train | validation | test |
|---|---:|---:|---:|
| `linear_trend_dominated` | 1,147 | 135 | 138 |
| `curved_trend_dominated` | 28 | 1 | 0 |
| `regime_change_trend_dominated` | 66 | 8 | 6 |
