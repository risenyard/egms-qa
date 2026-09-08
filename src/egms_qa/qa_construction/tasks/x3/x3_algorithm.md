# X3: Representation boundary

## Task overview

X3 defines answers for unsupported interpretations of encoder vectors. These vectors summarize tile observations; reference anchors are representative training vectors, and rarity measures isolation from other training vectors. Each task covers a different interpretation boundary.

| Task | Type | Output and relationship |
|---|---|---|
| X31 | Refusal | Declines treating a representation profile or rarity score as a physical or engineering truth label. |
| X32 | Refusal | Declines treating similarity in the representation as proof of a real-world cause. |
| X33 | Refusal | Declines assigning a certain meaning to a vector dimension or model mechanism without attribution evidence. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) · [Implementation](x3_compute.py) · [Run and files](../README.md#run-x3)

## Algorithm steps

### Rule

When a question treats an encoder anchor, rarity score, local-structure score,
token dimension, or embedding coordinate as direct ground truth or causal proof,
return a refusal-style answer:

```text
1. State that the representation-level claim is not supported as physical truth.
2. Name the missing model-attribution, causal, or physical-validation evidence.
3. Redirect to the supported representation property and measured motion facts.
```

### Answer Guidance

Every X3 catalog row includes:

- `answer_policy`: the group-level refusal policy.
- `response_template`: a task-specific answer template for question–answer generation.
- `supported_redirect_tasks`: representation and measurement task IDs from the [task index](../README.md#task-groups) when representation results should be separated from monitoring facts.

Generic pattern:

```text
Do not convert encoder representation outputs into direct physical truth,
causal proof, or certain model-mechanism semantics. State that the claim is
representation-level, then separately report supported representation properties and measured
monitoring facts.
```

## Results

The [published catalog](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) contains 3 task rows, all
with `target_value=refusal`. These are task-level policies; QA rendering
controls how they are sampled across tiles.
