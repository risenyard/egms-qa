#!/bin/bash
set -euo pipefail

CHECKOUT="/home/lis2/EGMS_VLM/egms-qa-migration294"
SOURCE_RELEASE="/home/lis2/EGMS_VLM/egms-qa-hf/dataset"
TARGET_RELEASE="/home/lis2/EGMS_VLM/egms-qa-hf/dataset-294-staging"
ENCODER_RELEASE="/home/lis2/EGMS_VLM/egms-qa-hf/encoder"
ENCODER_CHECKPOINT="${ENCODER_RELEASE}/encoder.safetensors"
ENCODER_CONFIG="${ENCODER_RELEASE}/config.json"
AUDIT_BASE="/home/lis2/EGMS_VLM/egms-qa-hf/migration294-audit"
UPLOAD_RECORD="/home/lis2/EGMS_VLM/egms-qa-hf/logs/egms-upload294-final.txt"

mkdir -p /home/lis2/EGMS_VLM/egms-qa-hf/logs "${AUDIT_BASE}"

crop_job_id="$(sbatch --parsable \
  --export=ALL,SOURCE_RELEASE="${SOURCE_RELEASE}",TARGET_RELEASE="${TARGET_RELEASE}",CHECKOUT="${CHECKOUT}" \
  "${CHECKOUT}/scripts/release/crop294.sbatch")"

audit_job_id="$(sbatch --parsable --dependency="afterok:${crop_job_id}" \
  --export=ALL,SOURCE_RELEASE="${SOURCE_RELEASE}",TARGET_RELEASE="${TARGET_RELEASE}",CHECKOUT="${CHECKOUT}",ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT}",ENCODER_CONFIG="${ENCODER_CONFIG}",AUDIT_BASE="${AUDIT_BASE}" \
  "${CHECKOUT}/scripts/release/audit_crop294.sbatch")"

upload_job_id="$(sbatch --parsable --dependency="afterok:${audit_job_id}" \
  --export=ALL,TARGET_RELEASE="${TARGET_RELEASE}",ENCODER_RELEASE="${ENCODER_RELEASE}",UPLOAD_RECORD="${UPLOAD_RECORD}" \
  "${CHECKOUT}/scripts/release/upload294.sbatch")"

printf 'crop_job_id=%s\n' "${crop_job_id}"
printf 'audit_job_id=%s\n' "${audit_job_id}"
printf 'upload_job_id=%s\n' "${upload_job_id}"
