#!/bin/bash
#SBATCH --job-name=egms-crop294-audit
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.out
#SBATCH --error=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.err

set -euo pipefail

: "${SOURCE_RELEASE:?SOURCE_RELEASE is required}"
: "${TARGET_RELEASE:?TARGET_RELEASE is required}"
: "${CHECKOUT:?CHECKOUT is required}"
: "${ENCODER_CHECKPOINT:?ENCODER_CHECKPOINT is required}"
: "${AUDIT_BASE:?AUDIT_BASE is required}"

export PYTHONPATH="${CHECKOUT}/src${PYTHONPATH:+:${PYTHONPATH}}"
/home/lis2/miniconda3/envs/egms_vqa/bin/python \
  "${CHECKOUT}/scripts/release/audit_crop294.py" \
  --source-release "${SOURCE_RELEASE}" \
  --target-release "${TARGET_RELEASE}" \
  --checkout "${CHECKOUT}" \
  --audit-root "${AUDIT_BASE}/${SLURM_JOB_ID}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --workers "${SLURM_CPUS_PER_TASK}"
