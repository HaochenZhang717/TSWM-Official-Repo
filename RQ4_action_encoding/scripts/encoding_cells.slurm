#!/usr/bin/env bash

#SBATCH --job-name=rq4-encoding
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq4-encoding-%A_%a.out
#SBATCH --error=logs/rq4-encoding-%A_%a.err

set -uxo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-tswm}"

REPO_DIR="${REPO_DIR:?set REPO_DIR to the repository root}"

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"

export TMPDIR="$REPO_DIR/.tmp"; mkdir -p "$TMPDIR"

MANIFEST="${MANIFEST:?set MANIFEST=<path> (see gen_encoding_manifest.py)}"
[[ "$MANIFEST" != /* ]] && MANIFEST="$REPO_DIR/$MANIFEST"
T_IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((T_IDX + 1))p" "$MANIFEST")"
if [[ -z "$LINE" ]]; then echo "ERROR: no manifest line for array index $T_IDX" >&2; exit 1; fi
read -r ENC FUS DATASET MODEL SEED <<< "$LINE"
if [[ -z "${SEED:-}" ]]; then
  echo "ERROR: manifest line '$LINE' incomplete (expected 'ENC FUSION DATASET MODEL SEED')" >&2
  exit 1; fi
echo "node=${SLURMD_NODENAME:-$(hostname -s)} job=${SLURM_JOB_ID:-<none>} task=$T_IDX -> enc=$ENC fusion=$FUS dataset=$DATASET model=$MODEL seed=$SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$REPO_DIR/RQ4_action_encoding"

CODEC="${CODEC:-ae}"
DEVICE="${DEVICE:-cuda}"
CONV_K="${CONV_K:-8}"
CONV_L="${CONV_L:-6}"
ATTN_DH="${ATTN_DH:-64}"
ATTN_HEADS="${ATTN_HEADS:-4}"
DATA_SEED="${DATA_SEED:-0}"

if [[ "$SEED" == "0" ]]; then OUT_ROOT="results/${ENC}__${FUS}"
else                          OUT_ROOT="results/${ENC}__${FUS}_seed${SEED}"; fi
mkdir -p "$OUT_ROOT/cells" "$OUT_ROOT/ckpts"

CACHE_DIR="${CACHE_DIR:-$REPO_DIR/rq2_latent_vs_ob/.gpu_window_cache}"

AE_DIR="${AE_DIR:-$REPO_DIR/rq2_latent_vs_ob/results/latent_space/target/ae}"
_suffix=""; [[ "$CODEC" != "ae" ]] && _suffix="__${CODEC}"
_codec_ckpt="${AE_DIR}/${DATASET}__target${_suffix}.pt"
if [[ ! -s "$_codec_ckpt" ]]; then
  echo "ERROR: shared codec missing: $_codec_ckpt -- train it once, serially, before submitting" >&2
  exit 1
fi

_cell_json="${OUT_ROOT}/cells/${DATASET}__${MODEL}__${ENC}__${FUS}.json"
_ckpt="${OUT_ROOT}/ckpts/${DATASET}__${MODEL}__${ENC}__${FUS}.pt"

python encode_train.py \
  --datasets "$DATASET" --models "$MODEL" \
  --encodings "$ENC" --fusion-arms "$FUS" --codec "$CODEC" \
  --ae-dir "$AE_DIR" --data-root "$DATA_ROOT" --device "$DEVICE" \
  --context-length 256 --horizon 16 --stride 80 --epochs 100 --patience 20 \
  --batch-size 128 --lr 1e-3 --lr-patience 5 --loss mse --grad-clip 1.0 --val-ratio 0.2 \
  --seed "$SEED" --data-seed "$DATA_SEED" \
  --conv-kernel "$CONV_K" --conv-layers "$CONV_L" --attn-dh "$ATTN_DH" --attn-heads "$ATTN_HEADS" \
  --ckpt-dir "$OUT_ROOT/ckpts" --cache-dir "$CACHE_DIR" \
  --out "$_cell_json"

if [[ ! -s "$_cell_json" ]]; then
  echo "FAILED (task=$T_IDX enc=$ENC fusion=$FUS dataset=$DATASET model=$MODEL seed=$SEED): no result JSON at $_cell_json" >&2
  exit 1
fi
python - "$_cell_json" "$ENC" "$FUS" <<'PY' || exit 1
import json, sys
path, enc, fus = sys.argv[1], sys.argv[2], sys.argv[3]
rows = json.load(open(path)).get("rows", [])
ok = [r for r in rows if r.get("encoding") == enc and r.get("fusion_arm") == fus
      and r.get("MAE") is not None]
if not ok:
    print(f"no usable MAE row for {enc}/{fus} in {path}", file=sys.stderr)
    sys.exit(1)
PY
if [[ ! -s "$_ckpt" ]]; then
  echo "FAILED (task=$T_IDX): no checkpoint at $_ckpt" >&2; exit 1
fi

echo "DONE (task=$T_IDX enc=$ENC fusion=$FUS dataset=$DATASET model=$MODEL seed=$SEED) -> $_cell_json"
