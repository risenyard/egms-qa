# X2: Unavailable information

## Task overview

X2 defines answers for questions requiring unavailable data or a finer scope than the supported outputs. EGMS-QA reports tile summaries, 8×8 cell locations, and comparisons against specified reference populations. Each task is a separate refusal category.

| Task | Type | Output and relationship |
|---|---|---|
| X21 | Refusal | Declines conclusions about a named address, building, road segment, parcel, or asset. |
| X22 | Refusal | Declines point-level, sub-cell, pixel-level, or exact-coordinate answers beyond the supported output scale. |
| X23 | Refusal | Declines displacement components or directions absent from the input data. |
| X24 | Refusal | Declines claims requiring unavailable imagery, land use, geology, or asset inventories. |
| X25 | Refusal | Declines claims about live conditions or times outside the observation window. |
| X26 | Refusal | Declines rankings such as “worst” or “most severe” without a defined comparison population. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) · [Implementation](x2_compute.py) · [Run the task system](../README.md#run-the-task-system)

## Algorithm steps

### Rule

When a question asks for exact assets, point/sub-cell conclusions, unsupported
displacement components, external imagery/land-use/geology context, live status,
or open-world rankings/superlatives, return a refusal-style answer:

```text
1. State that the requested scope is not supported.
2. Name the missing data channel, scale, time status, or ranking universe.
3. Redirect to supported tile-level, bin-level, or corpus-relative tasks.
```

Europe-defined and corpus-relative classes are allowed when the task explicitly
defines that reference distribution. X26 only rejects undefined rank, worst,
most severe, or highest-risk claims.

### Answer Guidance

Every X2 catalog row includes:

- `answer_policy`: the group-level refusal policy.
- `response_template`: a task-specific answer template for question–answer generation.
- `supported_redirect_tasks`: task IDs from the [task index](../README.md#task-groups) that can answer the
  nearest supported tile-level, bin-level, or corpus-relative question.

Generic pattern:

```text
Cannot answer the requested scale, data channel, live status, or undefined
ranking from the current inputs. State the missing scale/channel/time/reference
universe, then redirect to supported tile-level, 8x8 bin-level, or predefined
Europe/corpus-relative tasks.
```

## Results

The [published catalog](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) contains 6 task rows, all
with `target_value=refusal`. These are task-level policies; QA rendering
controls how they are sampled across tiles.
