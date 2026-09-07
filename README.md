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

## Example question and reference answer

The [released test split](https://huggingface.co/datasets/risenyard/egms-qa-dataset/viewer/default/test?row=157)
contains this B21 reference record for tile
`E52N20_x5270550_y2000050`:

> **Question:** For a ground-motion screening report, what is the average vertical ground velocity?
>
> **Reference answer:** The mean vertical ground velocity is -0.450 mm/yr.

![EGMS-QA framework](docs/assets/egms-framework.png)

## Guides and artifacts

| component | guide | released artifacts |
|---|---|---|
| Encoder | [Encode tiles and reproduce pretraining](src/egms_encoder/README.md) | [Weights, normalization, and training recipe](https://huggingface.co/risenyard/egms-qa-encoder) |
| QA construction | [Generate questions and inspect the task catalog](src/egms_qa/qa_construction/README.md) | [Tiles, tokens, labels, QA splits, and reference tables](https://huggingface.co/datasets/risenyard/egms-qa-dataset) |
| Translator | [Train and evaluate a host language model](src/egms_qa/translator/README.md) | [Four projector and LoRA bundles](https://huggingface.co/risenyard/egms-qa-translator) |

GitHub provides the code. Hugging Face provides the data, weights, and training
recipes in the [EGMS-QA Collection](https://huggingface.co/collections/risenyard/egms-qa).
The module guides cover data installation, training, evaluation, and output
files.

## Install and check the encoder

Encode one released tile to check the installation:

```bash
git clone https://github.com/risenyard/egms-qa
cd egms-qa
pip install -e .
python -m egms_encoder.extract_tokens \
    --encoder-repo risenyard/egms-qa-encoder \
    --dataset-repo risenyard/egms-qa-dataset \
    --max-tiles 1 --output-dir outputs/tokens
```

This encoder smoke test downloads the required artifacts and writes tokens
and metadata to `outputs/tokens/`. Python 3.10 or later is required. CPU
execution supports small encoder checks, while GPU execution is recommended
for the full collection.

For question answering, install `pip install -e '.[translator]'` and follow
the [released-model evaluation guide](src/egms_qa/translator/README.md#evaluate-a-released-model).
Translator training and evaluation require CUDA. The Dataset includes
precomputed tokens, so those workflows can use the released representations
directly.

## Selected evaluation results

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

## Dataset and scope

The Dataset contains 10,000 model-ready tiles with `[N,294]` displacement
arrays and a fixed 8,000/1,000/1,000 train/validation/test split. Training uses
the full 78-task catalog. The
[QA construction guide](src/egms_qa/qa_construction/README.md) describes the
task definitions and records.

New collections must satisfy the
[encoder input requirements](src/egms_encoder/README.md#input-requirements).
Use the released normalization with the released frozen checkpoint. Training
a new encoder on another corpus requires normalization fitted on its training
split. Official-product downloading and data preparation require a separate
workflow.

EGMS-QA describes measured deformation histories. Its answers do not establish
causes, predict future motion, or certify structural safety.

## Citation and license

The software citation is provided in [CITATION.cff](CITATION.cff). Research
using the source measurements should also cite the EGMS product identified in
the [source provenance](https://huggingface.co/datasets/risenyard/egms-qa-dataset/blob/main/SOURCE_PROVENANCE.md).

Code uses the [MIT License](LICENSE). EGMS-QA-created data and model artifacts
use CC-BY-4.0. The repacked EGMS Level-3 Ortho Vertical measurements retain the
Copernicus Land Monitoring Service attribution and modification requirements
described in [DATA_LICENSE](DATA_LICENSE).
