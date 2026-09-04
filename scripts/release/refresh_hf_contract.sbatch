#!/bin/bash
#SBATCH --job-name=egms-hf-contract
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.out
#SBATCH --error=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.err

set -euo pipefail

: "${TARGET_RELEASE:?TARGET_RELEASE is required}"
: "${CHECKOUT:?CHECKOUT is required}"
: "${ENCODER_RELEASE:?ENCODER_RELEASE is required}"
: "${UPLOAD_RECORD:?UPLOAD_RECORD is required}"

PYTHON_BIN="/home/lis2/miniconda3/envs/egms_vqa/bin/python"
HF_BIN="/home/lis2/miniconda3/envs/egms_vqa/bin/hf"
DATASET_REPO="risenyard/egms-qa-dataset"
ENCODER_REPO="risenyard/egms-qa-encoder"
export PYTHONPATH="${CHECKOUT}/src${PYTHONPATH:+:${PYTHONPATH}}"

${PYTHON_BIN} - <<'PY'
import json
import os
from pathlib import Path
from egms_qa.release import sha256_file

release = Path(os.environ["TARGET_RELEASE"])
token = release / "artifacts/representations/egms_tokens_10k.pt"
metadata = release / "artifacts/representations/egms_tokens_10k_metadata.json"
config = release / "metadata/data_config.json"
token_sha = sha256_file(token)
if token_sha != "9eb517b37834cc1bb971aea0df86a46018cee2566956b51236aea33e4bd1d991":
    raise SystemExit(f"canonical token cache changed: {token_sha}")
payload = json.loads(metadata.read_text(encoding="utf-8"))
config_sha = sha256_file(config)
if payload["current_release"]["data_config_sha256"] != config_sha:
    raise SystemExit("token metadata data_config SHA does not match the release")
if payload["input_contract"]["stored_window"] != "[0,294)":
    raise SystemExit("token metadata does not declare stored [0,294)")
print(f"canonical token cache unchanged: {token_sha}")
print(f"token metadata data_config SHA verified: {config_sha}")
PY

${PYTHON_BIN} -m egms_qa.release manifest \
  --release-dir "${TARGET_RELEASE}" \
  --workers "${SLURM_CPUS_PER_TASK}"
${PYTHON_BIN} -m egms_qa.release audit \
  --release-dir "${TARGET_RELEASE}" \
  --verify-hashes \
  --workers "${SLURM_CPUS_PER_TASK}"

dataset_output="$(${HF_BIN} upload "${DATASET_REPO}" "${TARGET_RELEASE}" . \
  --repo-type dataset \
  --include 'README.md' \
  --include 'artifacts/representations/egms_tokens_10k_metadata.json' \
  --include 'metadata/files.sha256' \
  --include 'metadata/release_manifest.json' \
  --include 'metadata/tile_manifest.parquet' \
  --delete 'artifacts/representations/encoder_tokens_10k*' \
  --commit-message 'Align metadata with the 294-step release contract')"
printf '%s\n' "${dataset_output}"

encoder_output="$(${HF_BIN} upload "${ENCODER_REPO}" \
  "${ENCODER_RELEASE}/README.md" README.md \
  --commit-message 'Clarify the released 294-step input contract')"
printf '%s\n' "${encoder_output}"

dataset_revision="$(${PYTHON_BIN} -c \
  'from huggingface_hub import HfApi; print(HfApi().repo_info("risenyard/egms-qa-dataset", repo_type="dataset").sha)')"
encoder_revision="$(${PYTHON_BIN} -c \
  'from huggingface_hub import HfApi; print(HfApi().repo_info("risenyard/egms-qa-encoder", repo_type="model").sha)')"

{
  printf 'dataset_revision=%s\n' "${dataset_revision}"
  printf 'encoder_revision=%s\n' "${encoder_revision}"
  printf 'dataset_upload=%s\n' "${dataset_output}"
  printf 'encoder_upload=%s\n' "${encoder_output}"
} > "${UPLOAD_RECORD}"
