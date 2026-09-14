#!/usr/bin/env bash

#SBATCH --job-name=rq2-seed
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq2-seed-%A_%a.out
#SBATCH --error=logs/rq2-seed-%A_%a.err

set -uxo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-tswm}"

{ set +x; } 2>/dev/null
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
set -x

REPO_DIR="${REPO_DIR:?set REPO_DIR to the repository root}"
cd "$REPO_DIR"

export TMPDIR="$REPO_DIR/.tmp"
mkdir -p "$TMPDIR"

export DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"

MANIFEST="${MANIFEST:?set MANIFEST=<path> (see gen_seed_manifest.py)}"
T_IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((T_IDX + 1))p" "$MANIFEST")"
if [[ -z "$LINE" ]]; then echo "ERROR: no manifest line for array index $T_IDX" >&2; exit 1; fi
read -r ARM DATASET MODEL SEED <<< "$LINE"
if [[ -z "${SEED:-}" ]]; then
  echo "ERROR: manifest line '$LINE' has no SEED field (expected 'ARM DATASET MODEL SEED')" >&2; exit 1; fi
echo "node=${SLURMD_NODENAME:-$(hostname -s)} job=${SLURM_JOB_ID:-<none>} task=$T_IDX -> arm=$ARM dataset=$DATASET model=$MODEL seed=$SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

export DEVICE="${DEVICE:-cuda}"
export WANDB="${WANDB:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DATASET MODEL SEED
export MODELS="$MODEL"

export DATA_SEED="${DATA_SEED:-0}"

if [[ "$SEED" == "0" ]]; then OBS_ROOT="results/action_conditioning"; LAT_ROOT="results/latent_space/target"
else                          OBS_ROOT="results/action_conditioning_seed${SEED}"; LAT_ROOT="results/latent_space/target_seed${SEED}"; fi

case "$ARM" in
  obs)
    export WANDB_PROJECT="${WANDB_PROJECT:-tswm-obs}"
    export OUT_ROOT="$OBS_ROOT"
    bash rq2_latent_vs_ob/observational_space/scripts/ac_train_eval.sh
    ;;
  ae|vae|jepa)
    export WANDB_PROJECT="${WANDB_PROJECT:-tswm-latent}"
    export LATENT_SCOPE=target
    export AE_MODE="${AE_MODE:-frozen}"
    export CODEC="$ARM"
    export JEPA_ALPHA="${JEPA_ALPHA:-0.1}"
    export OUT_ROOT="$LAT_ROOT"
    export AE_DIR="${AE_DIR:-results/latent_space/target/ae}"
    if [[ "$ARM" != "jepa" ]]; then
      _suffix=""; [[ "$ARM" == "vae" ]] && _suffix="__vae"
      _codec_ckpt="${AE_DIR}/${DATASET}__target${_suffix}.pt"
      [[ "$_codec_ckpt" != /* ]] && _codec_ckpt="rq2_latent_vs_ob/${_codec_ckpt}"
      if [[ ! -f "$_codec_ckpt" ]]; then
        echo "ERROR: shared codec missing: $_codec_ckpt -- train it once, serially, before submitting" >&2
        exit 1
      fi
    fi
    bash rq2_latent_vs_ob/latent_space/scripts/latent_train_eval.sh
    ;;
  *)
    echo "ERROR: unknown ARM '$ARM' (expected obs|ae|vae|jepa)" >&2; exit 1 ;;
esac

case "$ARM" in
  obs)  _eval_json="${OBS_ROOT}/eval/${DATASET}__${MODEL}.json" ;;
  ae)   _eval_json="${LAT_ROOT}/eval/${DATASET}__${MODEL}.json" ;;
  *)    _eval_json="${LAT_ROOT}/eval/${DATASET}__${MODEL}__${ARM}.json" ;;
esac
if [[ ! -s "rq2_latent_vs_ob/${_eval_json}" ]]; then
  echo "FAILED (task=$T_IDX arm=$ARM dataset=$DATASET model=$MODEL seed=$SEED): no eval JSON at $_eval_json" >&2
  exit 1
fi

echo "DONE (task=$T_IDX arm=$ARM dataset=$DATASET model=$MODEL seed=$SEED) -> $_eval_json"
