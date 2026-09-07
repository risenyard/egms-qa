# EGMS-QA

[English](README.md) · [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data terms](https://img.shields.io/badge/data-CC%20BY%204.0%20%2B%20CLMS-blue.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](pyproject.toml)
[![CI](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/release-v1.0.0-green.svg)](CHANGELOG.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EGMS--QA-yellow)](https://huggingface.co/collections/risenyard/egms-qa)

EGMS-QA answers monitoring questions from European Ground Motion Service
(EGMS) displacement time series. A frozen encoder represents each 7 km tile
as 65 tokens. A projector and a LoRA-adapted language model use those tokens
to produce numerical answers, categorical answers, or refusals for questions
outside the supported scope.

![EGMS-QA framework](docs/assets/egms-framework.png)

## Start here

| component | guide | released artifacts |
|---|---|---|
| Encoder | [Encode tiles and reproduce pretraining](src/egms_encoder/README.md) | [Weights, normalization, and training recipe](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [Generate questions and inspect the task catalog](src/egms_qa/qa_construction/README.md) | [Tiles, tokens, labels, QA splits, and reference tables](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [Train and evaluate a host language model](src/egms_qa/translator/README.md) | [Four projector and LoRA bundles](https://huggingface.co/risenyard/egms-qa-translator) |

GitHub provides the code. Hugging Face provides the data, weights, and model
recipes, grouped in the [EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa).

## Quick start

Install the package and encode one released tile:

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --max-tiles 1 --output-dir outputs/tokens
```

The extractor downloads the required artifacts into the HF cache and writes
tokens and metadata to `outputs/tokens/`. Remove `--max-tiles 1` to encode the
full collection. Python 3.10 or later is required. CPU execution supports small
encoder checks, while GPU execution is recommended for the full collection.
Translator training and evaluation require CUDA.

Install optional dependencies for the workflow being used:

```bash
pip install -e '.[translator]'   # host-model training and evaluation
pip install -e '.[tasks]'        # task reference-value computation
```

## Evaluate a released translator

After installing `.[translator]`, download the Dataset and one translator
variant, then install the Dataset into the runtime paths used by the code:

```bash
hf download risenyard/egms-qa-dataset --repo-type dataset \
    --local-dir release/egms-qa-dataset
python -m egms_qa.release audit --release-dir release/egms-qa-dataset
python -m egms_qa.release install \
    --release-dir release/egms-qa-dataset --target-root .
hf download risenyard/egms-qa-translator \
    --include 'qwen/*' --include 'evaluation_config.json' \
    --local-dir outputs/runs
python -m egms_qa.reproduce evaluate \
    --variant-dir outputs/runs/qwen \
    --evaluation-config outputs/runs/evaluation_config.json \
    --output-dir outputs/evaluation/qwen
```

The Dataset includes the precomputed token cache, so evaluation can use the
released representations directly. Replace `qwen` with `gemma`, `llama`, or
`mistral` in the download
and runtime paths to select another model. Add `--dry-run` to inspect the
evaluation command before loading the host model.

The installer creates `data/` and `outputs/` links without duplicating the
tile store. Run commands from the checkout root. The
[Encoder guide](src/egms_encoder/README.md) also provides an installer limited
to encoder inputs.

## Reproduce training and QA construction

With the Dataset installed, obtain the encoder recipe and start pretraining:

```bash
hf download risenyard/egms-qa-encoder --include '*.json' \
    --local-dir data/encoder/checkpoint
python -m egms_encoder.pretrain \
    --output-dir outputs/encoder_pretrain --device cuda:0
```

To train a translator from its pinned host model, use the downloaded variant's
complete recipe:

```bash
python -m egms_qa.reproduce translator \
    --variant-dir outputs/runs/qwen --output-dir outputs/training/qwen
```

Each recipe defines its training stages and checkpoint transitions. Add
`--dry-run` to inspect the translator commands. To evaluate newly trained
weights, set `--variant-dir` to the final stage's `best/` directory.

The [QA construction guide](src/egms_qa/qa_construction/README.md) explains how
reference tables become labels and natural-language records, with commands
that write generated outputs to a separate directory.

## Evaluation results

Encoder evaluation measures reconstruction of masked observations.
Translator evaluation measures numerical answers, categorical answers, and
refusals for out-of-scope questions.

| evaluation | reported result |
|---|---|
| Encoder masked reconstruction | RMSE 1.510 mm over 1,000 held-out tiles |
| Translator numerical answers | highest model-level mean R² 0.778 |
| Translator categorical answers | highest model-level mean balanced accuracy 0.777 |

The translator reporting protocol covers 71 tasks on 1,000 test tiles.
It averages R² over 29 numeric tasks and balanced accuracy separately over
28 categorical and 14 boundary tasks. The full 78-task catalog is retained
for training. Model-specific results and protocol settings are available in
the [Encoder](https://huggingface.co/risenyard/egms-qa-encoder) and
[Translator](https://huggingface.co/risenyard/egms-qa-translator) model cards.

After evaluating all four translator variants, combine the outputs:

```bash
python -m egms_qa.translator.summarize_results \
    --evaluation-root outputs/evaluation
```

## Data requirements and scope

The Dataset contains 10,000 model-ready tiles with `[N,294]` displacement
arrays and a fixed 8,000/1,000/1,000 train/validation/test split. Stored
`[0,294)` corresponds to source-preparation indices `[8,302)`. Source-axis
offsets and preprocessing are defined in the data config.

A new collection must satisfy the encoder's input contract, including units,
displacement component, temporal sampling, and coordinate geometry. Use the
released normalization with the released frozen checkpoint. Training a new
encoder on another corpus requires normalization fitted on its training split.
Official-product downloading, source-format conversion, and time-window
selection require a separate preparation workflow.

EGMS-QA describes measured deformation histories. Its answers do not establish
causes, predict future motion, or certify structural safety.

## License and provenance

Code is distributed under the [MIT License](LICENSE). EGMS-QA-created data and
model artifacts use CC-BY-4.0. The source tiles are a selected and repacked
derivative of the EGMS Level-3 Ortho Vertical product and retain the Copernicus
Land Monitoring Service attribution and modification requirements. See
[DATA_LICENSE](DATA_LICENSE) and the
[source provenance](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md).
