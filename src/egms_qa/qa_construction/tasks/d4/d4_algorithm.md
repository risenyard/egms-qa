# D4: Temporal composition

## Task overview

| task | description |
|---|---|
| **D4 group** | Combine trend, seasonality, and acceleration into a dominant process and evolution archetype. |
| D41 | Dominant temporal process selected by comparing train-reference ranks of the prescribed process indicators. |
| D42 | Evolution archetype combining the dominant process with trend shape, seasonal phase, and intensification. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d4/d4_final_table.csv) · [Implementation](d4_compute.py)

## Key concepts

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

### Inputs

| process | input scalar | meaning |
|---|---|---|
| trend | `B33_vel_abs_p90_mm_yr` | long-term motion strength |
| seasonal | `B51_seasonality_p90` | annual seasonal strength |
| acceleration | `B41_acc_abs_p90` | recent acceleration strength |

The three inputs have different physical units, so D41 does not compare raw
values directly. It first converts each input to a train-split empirical
percentile rank and applies those train-fitted ranks to all 10k tiles.

### Interpretation

- `low_activity`: all three process strengths are weak relative to the 10k corpus.
- `trend_dominant`: long-term motion strength clearly leads.
- `seasonal_dominant`: seasonal strength clearly leads.
- `acceleration_dominant`: acceleration strength clearly leads.
- `mixed`: multiple temporal processes are comparable.

### Intentional Exclusions

- D41 does not replace D1, D2, or D3. It summarizes their broad process context.
- D41 uses B33/B51/B41 as base strengths, not D11/D21/D31 labels, because it is
  a composition comparison across process magnitudes.
- D42 does not use D12/D13/D14/D22/D23/D24/D32/D33/D34/D35. Those columns are
  important diagnostics, but adding them to D42 would make the class a mixed
  heuristic rather than a readable story label.
- D42 does not add any new threshold. Its only thresholded input is D41, whose
  thresholds are already labeled corpus-relative.

## Algorithm steps

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

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.d4.d4_compute \
    --out-dir outputs/tasks-rebuilt/d4
```

The new table is written to `outputs/tasks-rebuilt/d4/d4_final_table.csv`.
The installed reference remains at `outputs/tasks/d4/d4_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

| required input | published source | installed path |
|---|---|---|
| B3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b3/b3_final_table.csv) | `outputs/tasks/b3/b3_final_table.csv` |
| B4 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b4/b4_final_table.csv) | `outputs/tasks/b4/b4_final_table.csv` |
| B5 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/b5/b5_final_table.csv) | `outputs/tasks/b5/b5_final_table.csv` |
| D1 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d1/d1_final_table.csv) | `outputs/tasks/d1/d1_final_table.csv` |
| D2 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d2/d2_final_table.csv) | `outputs/tasks/d2/d2_final_table.csv` |
| D3 reference table | [HF file](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d3/d3_final_table.csv) | `outputs/tasks/d3/d3_final_table.csv` |

The computation may also write local summaries or diagnostics next to its
new table. Their filenames and options are defined in the linked script;
they are not part of the published reference-table inventory unless linked
explicitly above.

## Results

The following summaries use all 10,000 rows of the
[published reference table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/d4/d4_final_table.csv). Numeric summaries use finite
values. Missing targets are reported separately.

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
| `trend_seasonal_mixed` | 1,321 | 13.21% |
| `low_activity` | 864 | 8.64% |
| `linear_trend_dominated` | 765 | 7.65% |
| `intensifying_acceleration_dominated` | 487 | 4.87% |
| `incoherent_seasonal_dominated` | 396 | 3.96% |
| `curved_trend_dominated` | 389 | 3.89% |
| `regime_change_trend_dominated` | 375 | 3.75% |
| `uncertain_direction_acceleration_dominated` | 316 | 3.16% |
| `weakening_acceleration_dominated` | 290 | 2.90% |
