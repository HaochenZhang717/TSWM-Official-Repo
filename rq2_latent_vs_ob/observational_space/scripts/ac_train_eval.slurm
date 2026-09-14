#!/usr/bin/env bash

#SBATCH --job-name=ac-bench
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/ac-bench-%A_%a.out
#SBATCH --error=logs/ac-bench-%A_%a.err

set -uxo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-tswm}"

{ set +x; } 2>/dev/null
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
_mask() { local v="${1:-}"; if [[ -n "$v" ]]; then echo "set(${#v}ch ...${v: -4})"; else echo "UNSET"; fi; }
echo "creds: WANDB_API_KEY=$(_mask "$WANDB_API_KEY")  HF_TOKEN=$(_mask "$HF_TOKEN")"
set -x

REPO_DIR="${REPO_DIR:?set REPO_DIR to the repository root}"
PROJECT_DIR="$REPO_DIR"
cd "$PROJECT_DIR"

export DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"

DATASET_SWEEP=(${DATASET_SWEEP:-greenhouse vitaldb cgmacros shanghai_diabetes pleiadata predist wastewater_nutrient})
N_DS="${#DATASET_SWEEP[@]}"
D_IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( D_IDX < 0 || D_IDX >= N_DS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$D_IDX out of range [0,$N_DS)" >&2; exit 1; fi
export DATASET="${DATASET_SWEEP[$D_IDX]}"

export DEVICE="${DEVICE:-cuda}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-tswm-obs}"
export EPOCHS="${EPOCHS:-100}"

echo "node=${SLURMD_NODENAME:-$(hostname -s)} job=${SLURM_JOB_ID:-<none>} array_task=$D_IDX -> dataset=$DATASET data_root=$DATA_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if bash rq2_latent_vs_ob/observational_space/scripts/ac_train_eval.sh; then
  echo "DONE (array_task=$D_IDX dataset=$DATASET): train+eval finished"
else
  echo "DONE WITH FAILURES (array_task=$D_IDX dataset=$DATASET)"; exit 1
fi
