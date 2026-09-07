# EGMS-QA Translator

The translator answers questions from frozen EGMS tile tokens. A two-layer
projector maps the 256-dimensional tokens to the host model's embedding width.
The projected tokens precede the question, and a LoRA adapter tunes the
language model while its base weights remain frozen.

[Models and recipes](https://huggingface.co/risenyard/egms-qa-translator) ·
[Encoder](https://huggingface.co/risenyard/egms-qa-encoder) ·
[Dataset](https://huggingface.co/datasets/risenyard/egms-qa-dataset)

## Evaluate a released model

Run the following commands from the repository root. Evaluation requires CUDA
and GPU memory for the host model in bfloat16 plus generation state.

```bash
pip install -e '.[translator]'
hf download risenyard/egms-qa-translator \
    --include 'qwen/*' --include 'evaluation_config.json' \
    --local-dir outputs/runs
hf download risenyard/egms-qa-dataset --repo-type dataset \
    --local-dir release/egms-qa-dataset
python -m egms_qa.release install \
    --release-dir release/egms-qa-dataset --target-root .
python -m egms_qa.reproduce evaluate \
    --variant-dir outputs/runs/qwen \
    --evaluation-config outputs/runs/evaluation_config.json \
    --output-dir outputs/evaluation/qwen
```

Add `--dry-run` to inspect the evaluation command without loading the model.
The output directory contains `answers.jsonl` and `metrics.json`. The
`reporting_summary` field separates numeric, categorical, and boundary metrics.

Replace `qwen` in the download and runtime paths to use another variant:

| variant | pinned host model |
|---|---|
| `qwen` | Qwen/Qwen3.5-9B |
| `gemma` | unsloth/gemma-3-12b-it |
| `llama` | unsloth/Meta-Llama-3.1-8B-Instruct |
| `mistral` | unsloth/Mistral-Nemo-Instruct-2407 |

Each `translator_config.json` records the host-model revision, projector
dimensions, adapter path, and prompt format. The input contains 65 tokens of
width 256 and a 65-element validity mask. Token 0 summarizes the tile, followed
by 64 spatial-cell tokens in row-major order.

## Reproduce training

With the Dataset and variant files installed, run the complete training recipe:

```bash
python -m egms_qa.reproduce translator \
    --variant-dir outputs/runs/qwen --output-dir outputs/training/qwen
```

The runner reads `training_args.json` and starts with the pinned base model, a
fresh LoRA adapter, and a randomly initialized projector. Each subsequent stage
loads the preceding stage's best adapter and projector, then initializes a new
optimizer and scheduler. Add `--dry-run` to inspect all stage commands and task
lists.

Training uses model-specific schedules recorded in the recipes. Checkpoints
appear under `outputs/training/<variant>/<stage>/best/`. To evaluate a newly
trained checkpoint, supply that directory as `--variant-dir` to the evaluation
command. The test split is reserved for evaluation.

## Evaluation protocol

Evaluation measures numerical answers, categorical answers, and refusals to
questions outside the supported scope. The
[EGMS-QA task catalog](../qa_construction/README.md) defines the question targets
and scoring rules for these three answer types.

The reported evaluation uses 71 tasks on 1,000 held-out tiles, yielding
71,000 answers per model. Results average R² over 29 numeric tasks and balanced
accuracy over 28 categorical and 14 boundary tasks.

The [evaluation configuration](https://huggingface.co/risenyard/egms-qa-translator/blob/main/evaluation_config.json)
records the reporting subset, question-phrasing pool, model-specific seeds,
and greedy-decoding settings. Parse counts and per-task scores remain in the
outputs. After evaluating all four variants, combine the results:

```bash
python -m egms_qa.translator.summarize_results \
    --evaluation-root outputs/evaluation
```

## Code reference

| module | purpose |
|---|---|
| `train.py` | sampling, optimization, and checkpoint selection |
| `checkpoint.py` | configuration validation and projector loading |
| `modeling.py` | projector, batching, and training loss |
| `evaluate.py` | free generation and task-level scoring |
| `answer_extractor.py` | numeric and categorical answer extraction |
| `compute_ci.py` | per-task bootstrap confidence intervals |
| `summarize_results.py` | four-model summary tables |

The lower-level training and evaluation entry points support custom
experiments. Their generic defaults differ from the published recipes and
reporting protocol.

## Scope

The translators require the released encoder's token representation and
questions within the EGMS-QA task definitions. Answers describe measured
vertical displacement histories. They do not establish causes, forecast
motion, or certify structural safety. Host-model weights are downloaded
separately and remain subject to their respective licenses.
