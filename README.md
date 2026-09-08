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


## Components

The three module guides provide data setup, training, and evaluation commands.
The [task implementation index](src/egms_qa/qa_construction/tasks/README.md)
lists task dependencies and reconstruction scope. The associated data,
weights, and recipes are available through the
[EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa).

| component | code and guide | Hugging Face release |
|---|---|---|
| Encoder | [Pretraining and token extraction](src/egms_encoder/README.md) | [Encoder weights, configuration, and normalization](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [QA construction and reproduction](src/egms_qa/qa_construction/README.md) | [Source tiles, tokens, labels, reference tables, and QA records](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [Language-model adaptation and evaluation](src/egms_qa/translator/README.md) | [Qwen, Gemma, Llama, and Mistral variants](https://huggingface.co/risenyard/egms-qa-translator) |

## Installation

Python 3.10 or later is required. Install the shared codebase from source:

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
```

The core package supports encoder workflows, label aggregation, and QA
rendering. Install the optional dependencies for the remaining workflows:

```bash
pip install -e '.[tasks]'        # QA task reference-value computation
pip install -e '.[translator]'   # Translator training and evaluation
```

Label aggregation, QA rendering, and small encoder checks can run on CPU.
GPU execution is recommended for encoder training and full-collection token
extraction. Translator training and evaluation require CUDA. Follow the
component guides above to download the required artifacts and run each
workflow from the repository root.

## Dataset

The release contains 10,000 overlapping 7 km tiles from the EGMS Level-3
Ortho Vertical product for 2019–2023. Each tile stores displacement histories
as `[N,294]` arrays. The fixed tile-level split contains 8,000 training,
1,000 validation, and 1,000 test tiles.

QA construction defines 78 tasks across observation quality, motion,
spatial organization, temporal dynamics, representation properties, and
refusal boundaries. Reference tables and labels provide the targets used to
render the question–answer records.

The published Dataset includes the source tiles, precomputed encoder tokens,
reference tables, labels, and QA splits. These artifacts support direct use
and reproduction of the three components. The
[Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset)
documents the file layout and data contract.

## Evaluation

Encoder evaluation measures reconstruction of masked observations.
Translator evaluation measures numerical answers, categorical answers, and
refusals for out-of-scope questions.

The translator reporting protocol covers 71 tasks on 1,000 test tiles.
It averages R² over 29 numeric tasks and balanced accuracy separately over
28 categorical and 14 boundary tasks.

| model | evaluation | reported result |
|---|---|---|
| EGMS-QA Encoder | masked reconstruction | RMSE 1.510 mm over 1,000 held-out tiles |
| Mistral translator | numerical answers | mean R² 0.778 |
| Llama translator | categorical answers | mean balanced accuracy 0.777 |

The Mistral and Llama entries show the highest reported means for their
respective answer types across the four variants. Full results and protocol
settings are available in the
[Encoder](https://huggingface.co/risenyard/egms-qa-encoder) and
[Translator](https://huggingface.co/risenyard/egms-qa-translator) model cards.

## Scope

The code operates on prepared EGMS-QA tiles. New collections must satisfy the
[encoder input requirements](src/egms_encoder/README.md#input-requirements),
including displacement units, component, temporal sampling, and coordinate
geometry. Use the released normalization with the released frozen checkpoint.
When training a new encoder on another corpus, fit normalization on that
corpus's training split. Official-product downloading and data preparation
require a separate workflow.

EGMS-QA describes measured deformation histories. Its answers do not establish
causes, predict future motion, or certify structural safety.

## Citation

Use [CITATION.cff](CITATION.cff) for the software citation. Research using the
source measurements should also cite the EGMS product identified in the
[source provenance](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md).

## License

Code uses the [MIT License](LICENSE). EGMS-QA-created data and model artifacts
use CC-BY-4.0. The repacked EGMS measurements retain the Copernicus Land
Monitoring Service attribution and modification requirements described in
[DATA_LICENSE](DATA_LICENSE).
