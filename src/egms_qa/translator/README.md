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

EGMS-QA trains four host models with the identical projector + LoRA recipe
(ids in `../paths.py`):

| key | base LLM |
|---|---|
| qwen | Qwen/Qwen3.5-9B |
| gemma | unsloth/gemma-3-12b-it |
| llama | unsloth/Meta-Llama-3.1-8B-Instruct |
| mistral | unsloth/Mistral-Nemo-Instruct-2407 |

## Train / evaluate

```bash
# train (GPU); --host-model selects the frozen language model
python -m egms_qa.translator.train \
    --host-model Qwen/Qwen3.5-9B \
    --token-cache data/encoder/tokens/encoder_tokens_10k.pt \
    --output-dir outputs/runs/qwen

# evaluate a trained checkpoint on the test split
python -m egms_qa.translator.evaluate \
    --adapter-dir outputs/runs/qwen/best \
    --token-cache data/encoder/tokens/encoder_tokens_10k.pt \
    --split test

# combine the four host-model summaries into the results report
python -m egms_qa.translator.summarize_results
```

Released adapters (one per host model) can be placed under
`outputs/runs/<key>/best/` to evaluate without retraining; see the top-level
README for the download link.
