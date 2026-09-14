#!/usr/bin/env bash

#SBATCH --job-name=rq2-seed-prep
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/rq2-seed-prep-%A_%a.out
#SBATCH --error=logs/rq2-seed-prep-%A_%a.err

set -uxo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-tswm}"

REPO_DIR="${REPO_DIR:?set REPO_DIR to the repository root}"
cd "$REPO_DIR"

export TMPDIR="$REPO_DIR/.tmp"
mkdir -p "$TMPDIR"

export DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"

DATASET_SWEEP=(${DATASET_SWEEP:-greenhouse vitaldb cgmacros shanghai_diabetes pleiadata predist wastewater_nutrient})
D_IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( D_IDX >= ${#DATASET_SWEEP[@]} )); then
  echo "ERROR: array index $D_IDX out of range for ${#DATASET_SWEEP[@]} datasets" >&2; exit 1; fi
export DATASET="${DATASET_SWEEP[$D_IDX]}"

echo "node=${SLURMD_NODENAME:-$(hostname -s)} job=${SLURM_JOB_ID:-<none>} task=$D_IDX -> dataset=$DATASET data_root=$DATA_ROOT"

export DEVICE=cpu
export WANDB=0
export LATENT_SCOPE=target
export PREPARE_ONLY=1
export WARM_CACHE=1
export CODEC=ae
export DATA_SEED=0
export SEED=0
export AE_DIR="results/latent_space/target/ae"
export OUT_ROOT="results/latent_space/target"

bash rq2_latent_vs_ob/latent_space/scripts/latent_train_eval.sh \
  || { echo "STAGE0 FAILED: dataset=$DATASET"; exit 1; }

echo "DONE stage0 (task=$D_IDX dataset=$DATASET): window cache warm at data_seed=0"
