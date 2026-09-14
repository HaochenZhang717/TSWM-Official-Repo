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

LATENT_SCOPE=${LATENT_SCOPE:-target}
AE_MODE=${AE_MODE:-frozen}
CODEC=${CODEC:-ae}
AE_DMODEL=${AE_DMODEL:-32}
AE_DCOV=${AE_DCOV:-8}
AE_EPOCHS=${AE_EPOCHS:-100}
AE_PATIENCE=${AE_PATIENCE:-15}
AE_BATCH=${AE_BATCH:-128}
AE_REVIN=${AE_REVIN:-0}

JEPA_ALPHA=${JEPA_ALPHA:-0.02}
JEPA_EMA=${JEPA_EMA:-0.996}
JEPA_EMA_SCHEDULE=${JEPA_EMA_SCHEDULE:-cosine}
JEPA_INIT=${JEPA_INIT:-scratch}

WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-tswm-latent}

OUT_ROOT=${OUT_ROOT:-results/latent_space}
AE_DIR=${AE_DIR:-${OUT_ROOT}/ae}
CKPT_DIR=${CKPT_DIR:-${OUT_ROOT}/ckpts}
EVAL_DIR=${EVAL_DIR:-${OUT_ROOT}/eval}
mkdir -p "$AE_DIR" "$CKPT_DIR" "$EVAL_DIR"

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
echo "============ Latent :: dataset=$DATASET scope=$LATENT_SCOPE ae_mode=$AE_MODE root=$ROOT device=$DEVICE ============"

PREPARE_ONLY=${PREPARE_ONLY:-0}
WARM_CACHE=${WARM_CACHE:-0}

revin_flag=()
[[ "$AE_REVIN" == "1" ]] && revin_flag=(--revin)
wandb_flags=()
[[ "$WANDB" == "1" ]] && wandb_flags=(--wandb --wandb-project "$WANDB_PROJECT")
cache_dir_flag=()
[[ -n "${CACHE_DIR:-}" ]] && cache_dir_flag=(--cache-dir "$CACHE_DIR")

pretrain_ae() {
  local _want="${3:-$CODEC}"
  local _codec="ae" _suffix=""
  if [[ "$1" == "target" && "$_want" != "ae" ]]; then
    _codec="$_want"; _suffix="__${_want}"
  fi
  local _ae_ckpt="${AE_DIR}/${DATASET}__${1}${_suffix}.pt"
  if [[ "${AE_FORCE:-0}" != "1" && -f "$_ae_ckpt" ]]; then
    echo "=== [AE] reuse existing codec signal=$1 codec=$_codec -> $_ae_ckpt (AE_FORCE=1 to retrain) ==="
    return 0
  fi
  echo "=== [AE] pretrain signal=$1 codec=$_codec d_model=$2 dataset=$DATASET ==="
  "$PY" -m latent_space.train_codec --codec "$_codec" \
    --dataset "$DATASET" --signal "$1" --d-model "$2" \
    --context-length "$CTX" --horizon "$HORIZON" \
    --epochs "$AE_EPOCHS" --patience "$AE_PATIENCE" --batch-size "$AE_BATCH" \
    --seed "$DATA_SEED" --val-ratio "$VAL_RATIO" --device "$DEVICE" \
    --root "$ROOT" $OPTS "${revin_flag[@]}" \
    --ckpt "$_ae_ckpt"
}

if [[ "$WARM_CACHE" == "1" ]]; then
  echo "=== [cache] warm window cache dataset=$DATASET data_seed=$DATA_SEED ==="
  "$PY" "$RQ2_DIR/scripts/warm_window_cache.py" \
    --dataset "$DATASET" --seed "$DATA_SEED" \
    --context-length "$CTX" --horizon "$HORIZON" --stride "$STRIDE" \
    --val-ratio "$VAL_RATIO" --root "$ROOT" $OPTS "${cache_dir_flag[@]}" \
    || { echo "CACHE WARM FAILED: $DATASET data_seed=$DATA_SEED"; exit 1; }
fi

jepa_flags=()
case "$CODEC" in
  jepa|jepa0)
    jepa_flags=(--jepa-alpha "$JEPA_ALPHA" --jepa-ema "$JEPA_EMA"
                --jepa-ema-schedule "$JEPA_EMA_SCHEDULE" --jepa-init "$JEPA_INIT"
                --jepa-d-model "$AE_DMODEL")
    if [[ "$JEPA_INIT" == "ae" ]]; then
      pretrain_ae target "$AE_DMODEL" ae \
        || { echo "WARM-START AE FAILED: $DATASET"; exit 1; }
    else
      echo "=== [AE] skip target codec pretraining (CODEC=$CODEC trains it jointly) ==="
    fi
    ;;
  *)
    pretrain_ae target "$AE_DMODEL" || { echo "TARGET AE FAILED: $DATASET"; exit 1; }
    ;;
esac
if [[ "$LATENT_SCOPE" == "target_cov" ]]; then
  pretrain_ae covariate "$AE_DCOV" || echo "skip covariate AE (no continuous/exog channels?)"
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "============ PREPARE ONLY: cache + codecs ready for dataset=$DATASET seed=$SEED codec=$CODEC ============"
  exit 0
fi

for MODEL in $MODELS; do
  echo "---------------- $DATASET :: $MODEL (latent/$LATENT_SCOPE) ----------------"

  echo "=== [1/2] TRAIN dataset=$DATASET model=$MODEL codec=$CODEC ==="
  "$PY" -m latent_space.train \
    --dataset "$DATASET" --model "$MODEL" --codec "$CODEC" \
    --ae-dir "$AE_DIR" --latent-scope "$LATENT_SCOPE" --ae-mode "$AE_MODE" \
    --context-length "$CTX" --horizon "$HORIZON" --stride "$STRIDE" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --lr-patience "$LR_PATIENCE" --batch-size "$BATCH" \
    --lr "$LR" --loss "$LOSS" --grad-clip "$GRAD_CLIP" \
    --seed "$SEED" --data-seed "$DATA_SEED" --val-ratio "$VAL_RATIO" --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --root "$ROOT" $OPTS \
    --ckpt-dir "$CKPT_DIR" \
    "${jepa_flags[@]}" \
    "${wandb_flags[@]}" \
    "${cache_dir_flag[@]}" \
    || { echo "TRAIN FAILED: $DATASET/$MODEL"; continue; }

  CODEC_SUFFIX=""
  [[ "$CODEC" != "ae" ]] && CODEC_SUFFIX="__${CODEC}"
  echo "=== [2/2] EVAL  dataset=$DATASET model=$MODEL codec=$CODEC ==="
  "$PY" -m latent_space.eval \
    --dataset "$DATASET" --model "$MODEL" --codec "$CODEC" \
    --ckpt-dir "$CKPT_DIR" \
    --device "$DEVICE" --batch-size "$EVAL_BATCH" \
    --out "${EVAL_DIR}/${DATASET}__${MODEL}${CODEC_SUFFIX}.json" \
    || { echo "EVAL FAILED: $DATASET/$MODEL"; continue; }
done

echo "============ DONE dataset=$DATASET (eval JSONs in $EVAL_DIR) ============"
