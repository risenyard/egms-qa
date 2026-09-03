```text
 _____ _____ ___  ___ _____        _____  ___
|  ___|  __ \|  \/  |/  ___|      |  _  |/ _ \
| |__ | |  \/| .  . |\ `--. ______| | | / /_\ \
|  __|| | __ | |\/| | `--. \______| | | |  _  |
| |___| |_\ \| |  | |/\__/ /      \ \/' / | | |
\____/ \____/\_|  |_/\____/        \_/\_\_| |_/
```

# EGMS-QA

*[English](README.md) · [中文](README.zh-CN.md)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-blue.svg)](DATA_LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue.svg)](pyproject.toml)
[![Stars](https://img.shields.io/github/stars/risenyard/egms-qa?style=flat)](https://github.com/risenyard/egms-qa/stargazers)
[![Forks](https://img.shields.io/github/forks/risenyard/egms-qa?style=flat)](https://github.com/risenyard/egms-qa/network/members)

Natural-language question answering over persistent-scatterer displacement time
series, for the [European Ground Motion Service](https://egms.land.copernicus.eu/)
(EGMS).

EGMS-QA reads each 7 km tile of EGMS point histories, represents it as a fixed
set of tokens, and lets a host language model answer monitoring questions in
plain language — with calibrated refusal for out-of-scope questions.

```
tile (variable # points, 294-step histories)
   → frozen EGMS encoder + 8×8 pooling → 65 × 256 tokens
   → 2-layer projector → prefix ; "Question: …\nAnswer:" → frozen host LLM + LoRA → answer
```

## The three blocks

| block | directory | what it is |
|---|---|---|
| **encoder** | [`src/egms_encoder/`](src/egms_encoder/) | the frozen tile representation and token extraction |
| **qa_construction** | [`src/egms_qa/qa_construction/`](src/egms_qa/qa_construction/) | the 78-task definitions and the question–answer records |
| **translator** | [`src/egms_qa/translator/`](src/egms_qa/translator/) | projector + LoRA training and evaluation on host LLMs |

Each block has its own README. The task system (78 tasks, families A/B/C/D/S/X)
and the dataset datasheet are documented in
[`src/egms_qa/qa_construction/README.md`](src/egms_qa/qa_construction/README.md).

## Results (held-out test)

The frozen encoder reconstructs masked intervals at 1.510 mm RMSE, near the
source residual noise. Across four host models (Qwen, Gemma, Llama, Mistral)
trained with the identical recipe, the adapted system reaches mean R² up to
0.778 on numeric tasks and balanced accuracy up to 0.777 on categorical tasks,
refuses out-of-scope questions at near-ceiling rates, and collapses toward chance
under a shuffled-token control — so the answers depend on the supplied tile.
Regenerate the full table with `python -m egms_qa.translator.summarize_results`.

## Install

```bash
pip install -e .                 # core (representation + QA rendering)
pip install -e ".[translator]"   # + host-LLM training / evaluation
pip install -e ".[tasks]"        # + task reference-value computation
```

Requires Python ≥ 3.10. GPU is needed for encoder token extraction and
translator training/evaluation.

## Data

Code lives here; heavy artifacts are in the data release:

- encoder checkpoint and 10k-tile token cache,
- the question–answer records and per-family reference tables,
- the four trained translator adapters (one per host model).

**Download:** _<data release URL — to be added>_. Place files as:

```
data/encoder/checkpoint/encoder.pt
data/encoder/tokens/encoder_tokens_10k.pt
outputs/qa/…            # QA records + labels
outputs/tasks/…         # per-family reference tables
outputs/runs/<key>/best/   # trained adapters (qwen | gemma | llama | mistral)
```

Paths are overridable via `EGMS_QA_ROOT`, `EGMS_QA_DATA`, `EGMS_QA_OUTPUTS`,
`EGMS_ENCODER_HOME` (see [`src/egms_qa/paths.py`](src/egms_qa/paths.py)). The raw
EGMS Level-3 product is Copernicus data and is not redistributed; see
[`data/METADATA.md`](data/METADATA.md).

## Reproduce

```bash
# 1. tokens: either download the cache, or extract from the raw tile store
python -m egms_encoder.extract_tokens --output-dir outputs/tokens

# 2. task labels + QA records (or download them)
python -m egms_qa.qa_construction.build_probe_labels
python -m egms_qa.qa_construction.generate_qa --out-dir outputs/qa

# 3. train and evaluate a host model
python -m egms_qa.translator.train --qwen-path Qwen/Qwen3.5-9B \
    --token-cache data/encoder/tokens/encoder_tokens_10k.pt --output-dir outputs/runs/qwen
python -m egms_qa.translator.evaluate --adapter-dir outputs/runs/qwen/best --split test

# 4. combined four-model report
python -m egms_qa.translator.summarize_results
```

`pytest` runs the answer-extractor tests.

## Licence

Code is released under the MIT Licence ([`LICENSE`](LICENSE)); the dataset and
model weights under CC-BY-4.0 ([`DATA_LICENSE`](DATA_LICENSE)).
