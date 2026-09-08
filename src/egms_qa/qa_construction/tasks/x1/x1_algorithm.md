# X1: Unsupported inference

## Task overview

X1 defines answers for questions whose requested conclusions cannot be established from historical EGMS displacement observations. Each task supplies a refusal policy and response template; the tasks are parallel categories, with no tile-level numeric calculation.

| Task | Type | Output and relationship |
|---|---|---|
| X11 | Refusal | Declines attribution of deformation to a real-world cause without causal evidence. |
| X12 | Refusal | Declines forecasts of future deformation or whether motion will continue. |
| X13 | Refusal | Declines conclusions about structural safety or habitability. |
| X14 | Refusal | Declines estimates of economic loss, insurance impact, asset-value loss, or compensation. |
| X15 | Refusal | Declines prescriptions for engineering work, evacuation, repair, or operational action. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) · [Implementation](x1_compute.py) · [Run and files](../README.md#run-x1)

## Algorithm steps

### Rule

When a question asks for cause, future forecast, structural safety, monetary
loss, or intervention advice, return a refusal-style answer:

```text
1. State that the requested inference is not supported.
2. Name the missing evidence or unsupported task objective.
3. Redirect to supported facts about measurement quality, motion, spatial patterns, temporal behavior, or representation properties.
```

### Answer Guidance

Every X1 catalog row includes:

- `answer_policy`: the group-level refusal policy.
- `response_template`: a task-specific answer template for question–answer generation.
- `supported_redirect_tasks`: task IDs from the [task index](../README.md#task-groups) that can answer the
  nearest supported monitoring question.

Generic pattern:

```text
Cannot infer the requested cause, future outcome, safety state, loss, or
intervention from the current EGMS-QA observations. The missing evidence is
causal, predictive, engineering, or economic. Answer supported historical
monitoring facts instead.
```

## Results

The [published catalog](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x1/x1_final_table.csv) contains 5 task rows, all
with `target_value=refusal`. These are task-level policies; QA rendering
controls how they are sampled across tiles.
