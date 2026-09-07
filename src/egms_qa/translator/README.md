# Translator (language-model adaptation)

> 🤗 Released adapters + projectors (4 host models): [`risenyard/egms-qa-translator`](https://huggingface.co/risenyard/egms-qa-translator)

The translator adapts a host language model to answer EGMS-QA questions from the
frozen tile tokens alone. A two-layer projector maps each 256-d token to the host
model's embedding width; the projected tokens form a prefix before the tokenized
question, and a LoRA adapter is trained on the answer tokens with the base
weights frozen. Training uses bf16, AdamW, and cross-entropy on answer tokens
only.

Files:

- `train.py` — sampling and the training loop (entry point).
- `modeling.py` — projector, batch construction, loss, evaluation building blocks.
- `generation.py` — decoding utilities (prompt builder, greedy decode).
- `evaluate.py` — free-generation evaluation on the test split, scoring generated
  answers against the canonical task labels; supports a shuffled-token control.
- `answer_extractor.py` — deterministic extraction of the canonical value from a
  visible natural-language answer (numeric via quantulum3; categorical via label
  aliases).
- `compute_ci.py` — bootstrap 95% confidence intervals per task.
- `summarize_results.py` — aggregate the four host-model test summaries into one
  report (JSON + per-task CSV + Markdown table).

## Host models

EGMS-QA provides four variants with the same projector-plus-LoRA architecture
and model-specific training configurations (ids in `../paths.py`):

| key | base LLM |
|---|---|
| qwen | Qwen/Qwen3.5-9B |
| gemma | unsloth/gemma-3-12b-it |
| llama | unsloth/Meta-Llama-3.1-8B-Instruct |
| mistral | unsloth/Mistral-Nemo-Instruct-2407 |

## Reproduce the published training and evaluation

From the checkout root, install `pip install -e '.[translator]'` and follow the
top-level README to install the Dataset and download the translator bundle.
The complete recipes live in each HF variant's `training_args.json`;
`evaluation_config.json` defines the reported 71-task protocol. CUDA is required.

```bash
python -m egms_qa.reproduce translator \
    --variant-dir outputs/runs/qwen --output-dir outputs/training/qwen
python -m egms_qa.reproduce evaluate --variant-dir outputs/runs/qwen \
    --evaluation-config outputs/runs/evaluation_config.json \
    --output-dir outputs/evaluation/qwen
```

Use `--dry-run` to inspect the resolved commands. The runner starts from the
pinned host model, then connects all required training stages. Evaluation
selects 71 reported tasks and samples one phrasing for each of 1,000 test tiles
per task. Numeric, categorical, and boundary means are reported separately.

## Custom training and sampled evaluation

These lower-level examples use generic defaults, not the published recipe:

```bash
# train (GPU); --host-model selects the frozen language model
python -m egms_qa.translator.train \
    --host-model Qwen/Qwen3.5-9B \
    --token-cache data/encoder/tokens/egms_tokens_10k.pt \
    --output-dir outputs/runs/qwen

# evaluate a trained checkpoint on the test split
python -m egms_qa.translator.evaluate \
    --adapter-dir outputs/runs/qwen/best \
    --token-cache data/encoder/tokens/egms_tokens_10k.pt \
    --split test

# combine the four host-model summaries into the results report
python -m egms_qa.translator.summarize_results
```

Training writes compatible checkpoints under `outputs/runs/<key>/best/`.
Released variants download directly to `outputs/runs/<key>/` and contain
`projector.safetensors`, `translator_config.json`, and `adapter/`; see the
top-level README for the download link.
