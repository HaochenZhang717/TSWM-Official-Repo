#!/usr/bin/env bash
set -uxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RQ2_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$RQ2_DIR"

PY=${PYTHON:-python}
DEVICE=${DEVICE:-cpu}
DATASET=${DATASET:?set DATASET (one of: greenhouse vitaldb cgmacros shanghai_diabetes pleiadata predist wastewater_nutrient)}

: "${DATA_ROOT:?set DATA_ROOT=/path/to/datasets}"

CTX=${CTX:-256}
HORIZON=${HORIZON:-16}
STRIDE=${STRIDE:-80}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}
LR_PATIENCE=${LR_PATIENCE:-5}
BATCH=${BATCH:-128}
LR=${LR:-1e-3}
LOSS=${LOSS:-mse}
GRAD_CLIP=${GRAD_CLIP:-1.0}
SEED=${SEED:-0}
DATA_SEED=${DATA_SEED:-$SEED}
VAL_RATIO=${VAL_RATIO:-0.2}
EVAL_BATCH=${EVAL_BATCH:-256}
NUM_WORKERS=${NUM_WORKERS:-8}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-tswm-obs}

OUT_ROOT=${OUT_ROOT:-results/action_conditioning}
CKPT_DIR=${CKPT_DIR:-${OUT_ROOT}/ckpts}
EVAL_DIR=${EVAL_DIR:-${OUT_ROOT}/eval}
mkdir -p "$CKPT_DIR" "$EVAL_DIR"

MODELS=${MODELS:-"TimeXer TiDE DUET PatchTST TimeKAN CrossLinear Amplifier"}

dataset_root() {
  case "$1" in
    greenhouse)          echo "$DATA_ROOT/greenhouse3/TimeSeries" ;;
    vitaldb)             echo "$DATA_ROOT/vital_db" ;;
    cgmacros)            echo "$DATA_ROOT/diabetes_datasets/cgmacros" ;;
    shanghai_diabetes)   echo "$DATA_ROOT/diabetes_datasets/Shanghai_T1DM_T2DM" ;;
    pleiadata)           echo "$DATA_ROOT/PLEIAData" ;;
    predist)             echo "$DATA_ROOT/PreDist/predist_dataset/manufacturer_2" ;;
    wastewater_nutrient) echo "$DATA_ROOT/Wastewater_Treatment_Plant_Data_for_Nutrient_Removal_System/IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv" ;;
    mimic_cardio)        echo "$DATA_ROOT/mimic_cardio" ;;
    *) echo "ERROR: unknown dataset $1" >&2; return 1 ;;
  esac
}
dataset_opts() {
  case "$1" in
    vitaldb)           echo "--split all --download" ;;
    cgmacros)          echo "--split all" ;;
    shanghai_diabetes) echo "--split T2DM" ;;
    *)                 echo "" ;;
  esac
}

ROOT=$(dataset_root "$DATASET")
OPTS=$(dataset_opts "$DATASET")
echo "================ Action-Conditioned :: dataset=$DATASET  root=$ROOT  device=$DEVICE ================"

wandb_flags=()
[[ "$WANDB" == "1" ]] && wandb_flags=(--wandb --wandb-project "$WANDB_PROJECT")
cache_dir_flag=()
[[ -n "${CACHE_DIR:-}" ]] && cache_dir_flag=(--cache-dir "$CACHE_DIR")

for MODEL in $MODELS; do
  echo "---------------- $DATASET :: $MODEL ----------------"

  echo "=== [1/2] TRAIN dataset=$DATASET model=$MODEL ==="
  "$PY" -m observational_space.train \
    --dataset "$DATASET" --model "$MODEL" \
    --context-length "$CTX" --horizon "$HORIZON" --stride "$STRIDE" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --lr-patience "$LR_PATIENCE" --batch-size "$BATCH" \
    --lr "$LR" --loss "$LOSS" --grad-clip "$GRAD_CLIP" \
    --seed "$SEED" --data-seed "$DATA_SEED" --val-ratio "$VAL_RATIO" --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --root "$ROOT" $OPTS \
    --ckpt-dir "$CKPT_DIR" \
    "${wandb_flags[@]}" \
    "${cache_dir_flag[@]}" \
    || { echo "TRAIN FAILED: $DATASET/$MODEL"; continue; }

  echo "=== [2/2] EVAL  dataset=$DATASET model=$MODEL ==="
  "$PY" -m observational_space.eval \
    --dataset "$DATASET" --model "$MODEL" \
    --ckpt-dir "$CKPT_DIR" \
    --device "$DEVICE" --batch-size "$EVAL_BATCH" \
    --out "${EVAL_DIR}/${DATASET}__${MODEL}.json" \
    || { echo "EVAL FAILED: $DATASET/$MODEL"; continue; }
done

echo "================ DONE dataset=$DATASET (eval JSONs in $EVAL_DIR) ================"
