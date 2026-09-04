#!/bin/bash
#SBATCH --job-name=egms-crop294
#SBATCH --partition=itc-cpu,main-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.out
#SBATCH --error=/home/lis2/EGMS_VLM/egms-qa-hf/logs/%x-%j.err

set -euo pipefail

: "${SOURCE_RELEASE:?SOURCE_RELEASE is required}"
: "${TARGET_RELEASE:?TARGET_RELEASE is required}"
: "${CHECKOUT:?CHECKOUT is required}"

export PYTHONPATH="${CHECKOUT}/src${PYTHONPATH:+:${PYTHONPATH}}"
/home/lis2/miniconda3/envs/egms_vqa/bin/python -m egms_qa.migrate_tiles_294 migrate \
  --source "${SOURCE_RELEASE}" \
  --target "${TARGET_RELEASE}" \
  --workers "${SLURM_CPUS_PER_TASK}"
