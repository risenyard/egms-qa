# EGMS-QA

[English](README.md) · [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data terms](https://img.shields.io/badge/data-CC%20BY%204.0%20%2B%20CLMS-blue.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](pyproject.toml)
[![CI](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/risenyard/egms-qa/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/release-v1.0.0-green.svg)](CHANGELOG.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EGMS--QA-yellow)](https://huggingface.co/collections/risenyard/egms-qa)

EGMS-QA supports natural-language question answering over European Ground
Motion Service (EGMS) displacement time series. The encoder extracts point
representations from each 7 km tile and pools them into 65 tokens. QA
construction defines monitoring tasks, derives their reference values, and
renders question–answer records. The translator adapts a host language model
with a projector and LoRA to answer questions from the frozen tile
representations.

![EGMS-QA framework](docs/assets/egms-framework.png)

## Code and Models

| component | guide | released artifacts |
|---|---|---|
| Encoder | [Encode tiles and reproduce pretraining](src/egms_encoder/README.md) | [Weights, normalization, and training recipe](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [Generate questions and inspect the task catalog](src/egms_qa/qa_construction/README.md) | [Tiles, tokens, labels, QA splits, and reference tables](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [Train and evaluate a host language model](src/egms_qa/translator/README.md) | [Four projector and LoRA bundles](https://huggingface.co/risenyard/egms-qa-translator) |

GitHub provides the code. Hugging Face provides the data, weights, and training
recipes in the [EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa).
The module guides cover data installation, training, evaluation, and output
files.

## Installation

Python 3.10 or later is required. Install the shared codebase for the encoder,
QA construction, and translator from the repository root:

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
```

The core installation supports encoder workflows, label aggregation, and QA
rendering. Add the optional dependencies needed for reference-value computation
or host-model training and evaluation:

```bash
pip install -e '.[tasks]'        # QA task reference-value computation
pip install -e '.[translator]'   # Translator training and evaluation
```

The [Encoder guide](src/egms_encoder/README.md) covers pretraining and token
extraction. The [QA construction guide](src/egms_qa/qa_construction/README.md)
covers task reference values, labels, and question–answer records. The
[Translator guide](src/egms_qa/translator/README.md) covers language-model
adaptation and evaluation. Each guide specifies the required HF artifacts
and commands.

Label aggregation, QA rendering, and small encoder checks can run on CPU.
GPU execution is recommended for encoder training and full-collection token
extraction. Translator training and evaluation require CUDA. The released
Dataset includes reference tables, labels, QA records, and precomputed tokens
for workflows that use the published artifacts directly.

## Evaluation

Encoder evaluation measures reconstruction of masked observations.
Translator evaluation measures numerical answers, categorical answers, and
refusals for out-of-scope questions.

| model | evaluation | reported result |
|---|---|---|
| EGMS-QA Encoder | masked reconstruction | RMSE 1.510 mm over 1,000 held-out tiles |
| Mistral translator | numerical answers | mean R² 0.778 |
| Llama translator | categorical answers | mean balanced accuracy 0.777 |

The translator protocol covers 71 tasks on 1,000 test tiles. It averages R²
over 29 numeric tasks and balanced accuracy separately over 28 categorical
and 14 boundary tasks. The Mistral and Llama entries show the highest reported
means for the two answer types across the four variants. Full results and
protocol settings are available in the
[model card](https://huggingface.co/risenyard/egms-qa-translator).

## Dataset

The Dataset contains 10,000 model-ready tiles with `[N,294]` displacement
arrays and a fixed 8,000/1,000/1,000 train/validation/test split. Training uses
the full 78-task catalog. The
[QA construction guide](src/egms_qa/qa_construction/README.md) describes the
task definitions and records.

## Scope

New collections must satisfy the
[encoder input requirements](src/egms_encoder/README.md#input-requirements).
Use the released normalization with the released frozen checkpoint. Training
a new encoder on another corpus requires normalization fitted on its training
split. Official-product downloading and data preparation require a separate
workflow.

EGMS-QA describes measured deformation histories. Its answers do not establish
causes, predict future motion, or certify structural safety.

## Citation

The software citation is provided in [CITATION.cff](CITATION.cff). Research
using the source measurements should also cite the EGMS product identified in
the [source provenance](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md).

## License

Code uses the [MIT License](LICENSE). EGMS-QA-created data and model artifacts
use CC-BY-4.0. The repacked EGMS Level-3 Ortho Vertical measurements retain the
Copernicus Land Monitoring Service attribution and modification requirements
described in [DATA_LICENSE](DATA_LICENSE).
