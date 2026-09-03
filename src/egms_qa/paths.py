"""Central path and host-model configuration for EGMS-QA.

All locations are overridable through environment variables so the pipeline can
run against a local checkout, a downloaded data release, or a cluster scratch
directory without editing source.

  EGMS_QA_ROOT       repository / working root (default: current directory)
  EGMS_QA_DATA       lightweight + downloaded heavy data   (default: <root>/data)
  EGMS_QA_OUTPUTS    generated artifacts (labels, QA, runs) (default: <root>/outputs)

Heavy artifacts (encoder checkpoint, token cache, QA records, adapters) are not
in the git repository; download them from the data release and place them under
EGMS_QA_DATA / EGMS_QA_OUTPUTS as described in the top-level README.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


ROOT = _env_path("EGMS_QA_ROOT", Path.cwd())
DATA_DIR = _env_path("EGMS_QA_DATA", ROOT / "data")
OUTPUTS_DIR = _env_path("EGMS_QA_OUTPUTS", ROOT / "outputs")

# --- Encoder representation (heavy; from data release) ---
ENCODER_CKPT = DATA_DIR / "encoder" / "checkpoint" / "encoder.pt"
ENCODER_TOKENS = DATA_DIR / "encoder" / "tokens" / "encoder_tokens_10k.pt"
SPLIT_MANIFEST = DATA_DIR / "encoder" / "manifest" / "split.parquet"

# --- Task reference values and QA records (generated / from data release) ---
TASKS_DIR = OUTPUTS_DIR / "tasks"          # per-family reference tables
QA_DIR = OUTPUTS_DIR / "qa"
LABELS = QA_DIR / "labels.parquet"
LABELS_META = QA_DIR / "labels_meta.json"
QA_AUDIT = QA_DIR / "qa_audit.json"

# --- Host language models (Hugging Face ids; override to a local path if cached) ---
HOST_MODELS = {
    "qwen": "Qwen/Qwen3.5-9B",
    "gemma": "unsloth/gemma-3-12b-it",
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral": "unsloth/Mistral-Nemo-Instruct-2407",
}
DEFAULT_HOST_MODEL = os.environ.get("EGMS_QA_HOST_MODEL", HOST_MODELS["qwen"])
