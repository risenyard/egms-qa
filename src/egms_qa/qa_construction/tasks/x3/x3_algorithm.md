# X3: Representation boundary

## Task overview

| task | description |
|---|---|
| **X3 group** | Define refusals for unsupported interpretations of the encoder representation. |
| X31 | Refuse treating a representation anchor or rarity score as a direct physical or engineering truth label. |
| X32 | Refuse interpreting an embedding pattern or reference-anchor assignment as proof of a real-world cause. |
| X33 | Refuse assigning a certain meaning to a token dimension or model mechanism without attribution evidence. |

[Task index](../README.md) · [Published table](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) · [Implementation](x3_compute.py)

## Key concepts

### Role

X3 protects S-group representation tasks from over-interpretation. S outputs
are encoder constructs, not direct physical, geological, engineering, or causal
truth labels.

## Algorithm steps

### Rule

When a question treats an encoder anchor, rarity score, local-structure score,
token dimension, or embedding coordinate as direct ground truth or causal proof,
return a refusal-style answer:

```text
1. State that the representation-level claim is not supported as physical truth.
2. Name the missing attribution, probe, causal, or validation evidence.
3. Redirect to the supported representation construct and any supported A/B/C/D facts.
```

### Answer Guidance

Every X3 catalog row includes:

- `answer_policy`: the group-level refusal policy.
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

## Run and files

Complete the [task setup](../README.md#setup) first. Run these commands from
the repository root:

```bash
python -m egms_qa.qa_construction.tasks.x3.x3_compute \
    --out-dir outputs/tasks-rebuilt/x3
```

The new table is written to `outputs/tasks-rebuilt/x3/x3_final_table.csv`.
The installed reference remains at `outputs/tasks/x3/x3_final_table.csv`.
See the [path conventions](../README.md#paths) for the relationship to Hugging Face.

This static catalog is generated from the task definitions in the Python
script. It does not require tile measurements or encoder tokens.

## Results

The [published catalog](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/artifacts/reference_tables/x3/x3_final_table.csv) contains 3 task rows, all
with `target_value=refusal`. These are task-level policies; QA rendering
controls how they are sampled across tiles.
