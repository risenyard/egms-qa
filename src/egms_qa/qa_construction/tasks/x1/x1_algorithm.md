# X1 Algorithm: Unsupported Inference Boundary

## Delivered Tasks

- `X11_causal_attribution_refusal`
- `X12_future_forecast_refusal`
- `X13_building_safety_refusal`
- `X14_economic_loss_refusal`
- `X15_intervention_recommendation_refusal`

## Role

X1 is a boundary/refusal family. It does not compute tile-level scalar targets
and does not require probe. It defines question types that cannot be inferred
from EGMS tile observations or encoder representations.

## Rule

When a question asks for cause, future forecast, structural safety, monetary
loss, or intervention advice, return a refusal-style answer:

```text
1. State that the requested inference is not supported.
2. Name the missing evidence or unsupported task objective.
3. Redirect to supported A/B/C/D/S monitoring facts.
```

## Answer Guidance

Every X1 catalog row includes:

- `answer_policy`: the family-level refusal policy.
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

## File Inventory

- `x1_final_table.csv`: static boundary catalog, one row per X1 task, including answer guidance and redirect task IDs.
- `x1_compute.py`: reproducible catalog generator.
