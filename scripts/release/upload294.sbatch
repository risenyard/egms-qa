#!/bin/bash
#SBATCH --job-name=egms-upload294
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.out
#SBATCH --error=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.err

set -euo pipefail

: "${TARGET_RELEASE:?TARGET_RELEASE is required}"
: "${ENCODER_RELEASE:?ENCODER_RELEASE is required}"
: "${UPLOAD_RECORD:?UPLOAD_RECORD is required}"

DATASET_REPO="risenyard/egms-qa-dataset"
ENCODER_REPO="risenyard/egms-qa-encoder"
HF_BIN="/home/lis2/miniconda3/envs/egms_vqa/bin/hf"

${HF_BIN} upload-large-folder "${DATASET_REPO}" "${TARGET_RELEASE}" \
  --repo-type dataset \
  --revision main \
  --include 'artifacts/source_tiles/**' \
  --include 'metadata/**' \
  --include 'README.md' \
  --num-workers "${SLURM_CPUS_PER_TASK}"

encoder_output="$(${HF_BIN} upload "${ENCODER_REPO}" "${ENCODER_RELEASE}/README.md" README.md \
  --commit-message "Document direct 294-step encoder input")"
printf '%s\n' "${encoder_output}"

dataset_revision="$(/home/lis2/miniconda3/envs/egms_vqa/bin/python -c \
  'from huggingface_hub import HfApi; print(HfApi().repo_info("risenyard/egms-qa-dataset", repo_type="dataset").sha)')"
encoder_revision="$(/home/lis2/miniconda3/envs/egms_vqa/bin/python -c \
  'from huggingface_hub import HfApi; print(HfApi().repo_info("risenyard/egms-qa-encoder", repo_type="model").sha)')"

{
  printf 'dataset_revision=%s\n' "${dataset_revision}"
  printf 'dataset_rollback_revision=%s\n' 'b370ae937baf541067a9a84d424d1130e0d48a7a'
  printf 'encoder_upload=%s\n' "${encoder_output}"
  printf 'encoder_revision=%s\n' "${encoder_revision}"
} > "${UPLOAD_RECORD}"
