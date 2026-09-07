# Temporal task summary

## Scope

The optional summary assembles the released D1-D4 family targets into one
local analysis table:

```text
d_final_table.csv
```

The D group story is:

1. D1 asks what the long-term trend/regime looks like.
2. D2 asks whether the seasonal phase is coherent and when it peaks.
3. D3 asks whether motion is intensifying, weakening, spatially supported, or
   locally concentrated.
4. D4 summarizes which temporal process dominates and converts the D-family
   story into a readable archetype.

D4 uses train-split empirical percentile ranks for its corpus-relative
comparison, then applies those ranks to all 10k tiles before this D-group table
is assembled.

## Delivered Tasks

| family | delivered tasks | source table |
|---|---|---|
| D1 | D11-D14 | `outputs/tasks/d1/d1_final_table.csv` |
| D2 | D21-D24 | `outputs/tasks/d2/d2_final_table.csv` |
| D3 | D31-D35 | `outputs/tasks/d3/d3_final_table.csv` |
| D4 | D41-D42 | `outputs/tasks/d4/d4_final_table.csv` |

The group table includes only task target columns plus the D41 rank columns
needed to interpret D41/D42.

## Run

After installing the Dataset as described in the [QA guide](README.md), run:

```bash
python -m egms_qa.qa_construction.summarize_temporal
```

Inputs default to `outputs/tasks/`; generated files go to
`outputs/summaries/temporal/`. Both locations can be overridden with
`--tasks-root` and `--out-dir`. Add `--plot` after installing `.[tasks]` to
generate the distribution plot. The summary is optional and is not an input
to label aggregation or QA generation. D12 and D13 use the target columns
defined in [task_specs.py](task_specs.py).

## Output Files

- `d_final_table.csv`: merged 10k D-group target table.
- `d_final_summary.json`: row count, delivered task list, class counts, and
  scalar summaries.
- `d_final_distribution.png`: optional distribution plot (`--plot`).


## Verification

The table is valid when:

- `d_final_table.csv` has 10,000 rows.
- D1-D4 source tables merge one-to-one by `tile_id, split`.
- Python scripts compile and generated caches/logs are removed after checking.
