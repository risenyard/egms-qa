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

dataset_output="$(${HF_BIN} upload "${DATASET_REPO}" "${TARGET_RELEASE}" . \
  --repo-type dataset --commit-message "Store model-ready 294-step EGMS tiles")"
printf '%s\n' "${dataset_output}"

encoder_output="$(${HF_BIN} upload "${ENCODER_REPO}" "${ENCODER_RELEASE}/README.md" README.md \
  --commit-message "Document direct 294-step encoder input")"
printf '%s\n' "${encoder_output}"

{
  printf 'dataset_upload=%s\n' "${dataset_output}"
  printf 'encoder_upload=%s\n' "${encoder_output}"
} > "${UPLOAD_RECORD}"
