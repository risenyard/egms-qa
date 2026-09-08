# X1: Unsupported inference

## Task overview

| task | description |
|---|---|
| **X1 group** | Define refusals for causal, predictive, safety, economic, and intervention claims that the observations cannot establish. |
| X11 | Refuse causal attribution of the observed deformation to external processes. |
| X12 | Refuse forecasts of future deformation or claims about whether motion will continue. |
| X13 | Refuse structural-safety or habitability conclusions about buildings and infrastructure. |
| X14 | Refuse estimates of economic loss, insurance impact, asset-value loss, or compensation. |
| X15 | Refuse prescriptions for engineering intervention, evacuation, repair, or operational action. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) · [Implementation](x1_compute.py)

## Key concepts

### Role

X1 is a boundary/refusal task group. It does not compute tile-level scalar targets
and does not require probe. It defines question types that cannot be inferred
from EGMS tile observations or encoder representations.

## Algorithm steps

### Rule

When a question asks for cause, future forecast, structural safety, monetary
loss, or intervention advice, return a refusal-style answer:

```text
1. State that the requested inference is not supported.
2. Name the missing evidence or unsupported task objective.
3. Redirect to supported A/B/C/D/S monitoring facts.
```

### Answer Guidance

Every X1 catalog row includes:

- `answer_policy`: the group-level refusal policy.
- `response_template`: a task-specific answer template suitable for VQA
  generation.
- `supported_redirect_tasks`: concrete A/B/C/D/S task IDs that can answer the
  nearest supported monitoring question.

Generic pattern:

```text
Cannot infer the requested cause, future outcome, safety state, loss, or
intervention from the current EGMS/VQA evidence. The missing evidence is
causal, predictive, engineering, or economic. Answer supported historical
monitoring facts instead.
```

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.x1.x1_compute \
    --out-dir outputs/tasks-rebuilt/x1
```

The new table is written to `outputs/tasks-rebuilt/x1/x1_final_table.csv`.
The installed reference remains at `outputs/tasks/x1/x1_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

This static catalog is generated from the task definitions in the Python
script. It does not require tile measurements or encoder tokens.

## Results

The [published catalog](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) contains 5 task rows, all
with `target_value=refusal`. These are task-level policies; QA rendering
controls how they are sampled across tiles.
