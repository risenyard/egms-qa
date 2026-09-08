# X2: Unavailable information

## Task overview

| task | description |
|---|---|
| **X2 group** | Define refusals for information, components, scales, and comparisons outside the available data. |
| X21 | Refuse conclusions about a specific address, building, road segment, parcel, or named asset. |
| X22 | Refuse point-level, sub-cell, pixel-level, or exact-coordinate conclusions. |
| X23 | Refuse displacement components or directions not provided by the input data. |
| X24 | Refuse inferences requiring external imagery, land use, geology, or asset inventories. |
| X25 | Refuse claims about live conditions or times beyond the observation window. |
| X26 | Refuse undefined open-world rankings and superlatives. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) · [Implementation](x2_compute.py)

## Key concepts

### Role

X2 is a boundary/refusal task group for questions that exceed the available input
channels, spatial scale, temporal status, or reference universe.

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
- `response_template`: a task-specific answer template suitable for VQA
  generation.
- `supported_redirect_tasks`: concrete A/B/C/D/S task IDs that can answer the
  nearest supported tile-level, bin-level, or corpus-relative question.

Generic pattern:

```text
Cannot answer the requested scale, data channel, live status, or undefined
ranking from the current inputs. State the missing scale/channel/time/reference
universe, then redirect to supported tile-level, 8x8 bin-level, or predefined
Europe/corpus-relative tasks.
```

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.x2.x2_compute \
    --out-dir outputs/tasks-rebuilt/x2
```

The new table is written to `outputs/tasks-rebuilt/x2/x2_final_table.csv`.
The installed reference remains at `outputs/tasks/x2/x2_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

This static catalog is generated from the task definitions in the Python
script. It does not require tile measurements or encoder tokens.

## Results

The [published catalog](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x2/x2_final_table.csv) contains 6 task rows, all
with `target_value=refusal`. These are task-level policies; QA rendering
controls how they are sampled across tiles.
