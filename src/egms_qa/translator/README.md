# EGMS-QA Translator

The translator adapts a host language model to answer questions from frozen
EGMS tile tokens using a token projector and a LoRA adapter. The projector maps tile tokens to the host model's embedding width. The
projected tokens form a prefix before the question. Training updates the
projector and LoRA adapter. The encoder and host-model base weights remain
frozen.

## Workflows

This guide covers running and training the [EGMS-QA](../../../README.md)
translator. The [Hugging Face model card](https://huggingface.co/risenyard/egms-qa-translator)
documents the four variants, released files, input contract, and results.
The [Dataset](https://huggingface.co/datasets/risenyard/egms-qa-dataset) provides
the encoder tokens, labels, and task tables used by these workflows.

| goal | where to start |
|---|---|
| Evaluate released weights | [Install the code](#installation), [prepare the data and model](#prepare-data-and-model), then [evaluate](#evaluate-a-released-model) |
| Train a translator | Complete the same setup, then [reproduce training](#reproduce-training) |
| Inspect model files or results | Hugging Face [files](https://huggingface.co/risenyard/egms-qa-translator#files) and [evaluation](https://huggingface.co/risenyard/egms-qa-translator#evaluation) |

## Installation

Python 3.10 or later is required. Training and answer generation require CUDA
and enough GPU memory for the host model in bfloat16 plus its runtime state.
Command previews with `--dry-run` can run on CPU.

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e '.[translator]'
```

Run all commands below from the `egms-qa` repository root.

## Prepare data and model

The example uses `qwen`. Replace it in the download and runtime paths with
`gemma`, `llama`, or `mistral` to use another variant. The
[model card](https://huggingface.co/risenyard/egms-qa-translator) lists the
corresponding host models and recipes.

```bash
python -m egms_qa.release install --download --components qa tokens
hf download risenyard/egms-qa-translator \
    --include 'qwen/*' --include 'evaluation_config.json' \
    --local-dir outputs/runs
```

This installs QA artifacts and precomputed tokens. Source tiles are a separate
optional download. Each variant's configuration
identifies the host model and revision to download when execution starts.

## Evaluate a released model

After completing [setup](#prepare-data-and-model), preview the evaluation plan:

```bash
python -m egms_qa.reproduce evaluate \
    --variant-dir outputs/runs/qwen \
    --evaluation-config outputs/runs/evaluation_config.json \
    --output-dir outputs/evaluation/qwen \
    --dry-run
```

Run the same command without `--dry-run` to load the host model and evaluate the
released translator. Generated answers are written to `answers.jsonl` and
scores to `metrics.json` in the output directory. The `reporting_summary` field
separates numeric, categorical, and boundary metrics.

For new tile representations, use the [Encoder guide](../../egms_encoder/README.md)
and follow the [translator input contract](https://huggingface.co/risenyard/egms-qa-translator#input-and-output).

## Reproduce training

With the Dataset and variant files from [setup](#prepare-data-and-model),
preview the complete training recipe:

```bash
python -m egms_qa.reproduce translator \
    --variant-dir outputs/runs/qwen \
    --output-dir outputs/training/qwen \
    --dry-run
```

Run the same command without `--dry-run` to start training. The runner reads
`training_args.json`. Its first stage initializes a fresh LoRA adapter and
projector on the pinned host model. Each subsequent stage loads the preceding
stage's best adapter and projector, then initializes a new optimizer and
scheduler.

Checkpoints appear under `outputs/training/<variant>/<stage>/best/`. To evaluate
a newly trained checkpoint, supply that directory as `--variant-dir` in the
evaluation command. The test split is reserved for evaluation.

## Evaluation protocol

The [HF evaluation section](https://huggingface.co/risenyard/egms-qa-translator#evaluation)
explains the reported metrics. The
[evaluation configuration](https://huggingface.co/risenyard/egms-qa-translator/blob/main/evaluation_config.json)
defines the reporting subset and generation settings used by the reproduction
command. After evaluating all four variants, combine their outputs:

```bash
python -m egms_qa.translator.summarize_results \
    --evaluation-root outputs/evaluation
```

The lower-level training and evaluation entry points support custom
experiments. Their defaults differ from the published recipes and reporting
protocol.

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

## Scope

Use the released encoder's token representation and questions within the
[EGMS-QA task definitions](../qa_construction/README.md). The
[model card](https://huggingface.co/risenyard/egms-qa-translator#scope-and-license)
describes application limits and model licensing.
