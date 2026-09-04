#!/bin/bash
#SBATCH --job-name=egms-encoder-release-test
#SBATCH --partition=itc-gpu,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

: "${CHECKOUT:?CHECKOUT is required}"
: "${ENCODER_RELEASE:?ENCODER_RELEASE is required}"
: "${DATASET_RELEASE:?DATASET_RELEASE is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${CHECKOUT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export EGMS_QA_HF_ENCODER_DIR="${ENCODER_RELEASE}"
export EGMS_QA_HF_DATASET_DIR="${DATASET_RELEASE}"
export EGMS_QA_REQUIRE_HF_INTEGRATION=1
export EGMS_QA_REQUIRE_FULL_HF_DATASET=1

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cd "${CHECKOUT}"
"${PYTHON_BIN}" -m egms_qa.release audit \
  --release-dir "${DATASET_RELEASE}" \
  --verify-hashes \
  --workers "${SLURM_CPUS_PER_TASK}"
"${PYTHON_BIN}" -m pytest -q -m "not hf_integration"
"${PYTHON_BIN}" -m pytest -q -m hf_integration

TEST_ROOT="${TMPDIR:-/tmp}/egms-qa-encoder-release-${SLURM_JOB_ID}"
RUNTIME_ROOT="${TEST_ROOT}/runtime"
TOKEN_ROOT="${TEST_ROOT}/tokens"
mkdir -p "${RUNTIME_ROOT}"
"${PYTHON_BIN}" -m egms_qa.release install \
  --release-dir "${DATASET_RELEASE}" \
  --target-root "${RUNTIME_ROOT}"
cd "${RUNTIME_ROOT}"
"${PYTHON_BIN}" -m egms_encoder.extract_tokens \
  --checkpoint "${ENCODER_RELEASE}/encoder.safetensors" \
  --model-config "${ENCODER_RELEASE}/config.json" \
  --normalization "${ENCODER_RELEASE}/normalization.json" \
  --manifest data/encoder/manifest/split.parquet \
  --data-config data/encoder/manifest/data_config.json \
  --device cuda:0 \
  --max-tiles 1 \
  --output-dir "${TOKEN_ROOT}"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

metadata = json.loads(
    Path("${TOKEN_ROOT}/egms_tokens_1_metadata.json").read_text(encoding="utf-8")
)
assert metadata["schema_version"] == "egms-tokens-1.1"
assert metadata["input_contract"]["stored_window"] == "[0,294)"
assert metadata["input_contract"]["source_window"] == "[8,302)"
assert metadata["input_contract"]["token_count"] == 65
assert metadata["input_contract"]["token_width"] == 256
assert metadata["statistics"]["tiles"] == 1
assert set(metadata["reproduction_input_sha256"]) == {
    "encoder_weights",
    "encoder_config",
    "normalization",
    "split_manifest",
    "data_config",
}
print("encoder_release_cli_smoke_passed")
PY
