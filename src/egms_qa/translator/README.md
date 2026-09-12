# EGMS-QA Translator

## Architecture

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
| Ask one question | [Install the code](#installation), then [run the minimal example](#ask-one-question) |
| Evaluate released weights | [Install the code](#installation), [prepare the data and model](#prepare-data-and-model), then [evaluate](#evaluate-a-released-model) |
| Train a translator | Complete the same setup, then [reproduce training](#reproduce-training) |
| Inspect model files or results | Hugging Face [files](https://huggingface.co/risenyard/egms-qa-translator#files) and [evaluation](https://huggingface.co/risenyard/egms-qa-translator#evaluation) |

## Installation

Follow the shared [installation instructions](../../../README.md#installation)
to clone the repository and create a Python environment, then install the
translator dependencies from the repository root:

```bash
python -m pip install -e '.[translator]'
```

Training and answer generation require CUDA, bfloat16 support, and enough GPU
memory for the host model plus its runtime state. Complete the
[CUDA check](../../../docs/environment.md#check-the-environment) before running
them. Command previews with `--dry-run` can run on CPU. Run all commands below
from the `egms-qa` repository root with the environment activated.

## Ask one question

After [installation](#installation), run this example from the repository root.
It downloads the released encoder and one source tile, extracts that tile's
tokens on CPU, and uses the Qwen translator to generate one answer on GPU.

```bash
python -m egms_encoder.extract_tokens \
    --max-tiles 1 --device cpu --output-dir outputs/example-tokens
hf download risenyard/egms-qa-translator \
    --include 'qwen/*' --local-dir outputs/runs
python -m egms_qa.translator.ask \
    --variant-dir outputs/runs/qwen \
    --token-cache outputs/example-tokens/egms_tokens_1.pt \
    --question "What is the mean vertical velocity in this tile?" \
    --output outputs/example-answer.json
```

The last command prints the tile ID, question, and generated answer, and saves
those three fields in `outputs/example-answer.json`. The pinned host-model
weights are downloaded on the first run; answer generation requires CUDA.
This example does not need the full tile collection, QA labels, or reference
tables.

One run of this example produced:

```json
{
  "tile_id": "E30N33_x3006050_y3355750",
  "question": "What is the mean vertical velocity in this tile?",
  "answer": "The mean vertical ground velocity is -1.33 mm/yr."
}
```

The answer is a model prediction; exact wording and values may vary across
runtime environments.

To ask another question, change `--question` within the
[supported task definitions](../qa_construction/tasks/README.md#task-groups).
With a multi-tile token cache, use `--tile-id` to select a tile; the default is
the first tile in the cache. Use `--token-cache` with your own encoder output
to ask about a new tile, following the [Encoder guide](../../egms_encoder/README.md).

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
| [ask.py](ask.py) | one question and one generated answer for a tile |
| `train.py` | sampling, optimization, and checkpoint selection |
| `checkpoint.py` | configuration validation and projector loading |
| `modeling.py` | projector, batching, and training loss |
| `evaluate.py` | free generation and task-level scoring |
| `answer_extractor.py` | numeric and categorical answer extraction |
| `summarize_results.py` | four-model summary tables |

## Scope

Use the released encoder's token representation and questions within the
[EGMS-QA task definitions](../qa_construction/tasks/README.md#task-groups). The
[model card](https://huggingface.co/risenyard/egms-qa-translator#scope-and-license)
describes application limits and model licensing.
