# X3 Algorithm: Representation Boundary

## Delivered Tasks

- `X31_representation_as_ground_truth_refusal`
- `X32_representation_causality_refusal`
- `X33_model_mechanism_certainty_refusal`

## Role

X3 protects S-group representation tasks from over-interpretation. S outputs
are encoder constructs, not direct physical, geological, engineering, or causal
truth labels.

## Rule

When a question treats an encoder anchor, rarity score, local-structure score,
token dimension, or embedding coordinate as direct ground truth or causal proof,
return a refusal-style answer:

```text
1. State that the representation-level claim is not supported as physical truth.
2. Name the missing attribution, probe, causal, or validation evidence.
3. Redirect to the supported representation construct and any supported A/B/C/D facts.
```

## Answer Guidance

Every X3 catalog row includes:

- `answer_policy`: the family-level refusal policy.
- `response_template`: a task-specific answer template suitable for VQA
  generation.
- `supported_redirect_tasks`: concrete S task IDs, plus relevant A/B/C/D task
  IDs when representation results should be separated from monitoring facts.

Generic pattern:

```text
Do not convert encoder representation outputs into direct physical truth,
causal proof, or certain model-mechanism semantics. State that the claim is
representation-level, then separately report supported S constructs and
A/B/C/D monitoring facts.
```

## File Inventory

- `x3_final_table.csv`: static boundary catalog, one row per X3 task, including answer guidance and redirect task IDs.
- `x3_compute.py`: reproducible catalog generator.
