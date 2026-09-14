#!/bin/bash
#SBATCH --job-name=rq7-soft
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=24:00:00
#SBATCH --output=logs/rq7-soft-%A_%a.out
#SBATCH --error=logs/rq7-soft-%A_%a.err
set -uxo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-tswm}"

REPO_DIR="${REPO_DIR:?set REPO_DIR to the repository root}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
export TMPDIR="$REPO_DIR/.tmp"; mkdir -p "$TMPDIR"

MANIFEST="${MANIFEST:?set MANIFEST=<path>}"
[[ "$MANIFEST" != /* ]] && MANIFEST="$REPO_DIR/$MANIFEST"
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$MANIFEST")
[[ -z "$LINE" ]] && { echo "ERROR: empty manifest line" >&2; exit 1; }
read -r DATASET MODEL SEED RHO <<<"$LINE"

cd "$REPO_DIR"
TAG="rho$(echo "$RHO" | tr '.' 'p')"
PRIOR_SPLIT="${PRIOR_SPLIT:-default}"
if [[ "$PRIOR_SPLIT" == "default" ]]; then ARM_DIR="gate_soft_${TAG}"
else                                       ARM_DIR="gate_soft_${PRIOR_SPLIT}_${TAG}"; fi
if [[ "$SEED" == "0" ]]; then OUT_ROOT="RQ7_causal_fusion/results/${ARM_DIR}"
else                          OUT_ROOT="RQ7_causal_fusion/results/${ARM_DIR}_seed${SEED}"; fi
mkdir -p "$OUT_ROOT/cells" "$OUT_ROOT/ckpts"
AE_DIR="$REPO_DIR/rq2_latent_vs_ob/results/latent_space/target/ae"
[[ -s "${AE_DIR}/${DATASET}__target.pt" ]] || { echo "ERROR: codec missing" >&2; exit 1; }
CELL="$OUT_ROOT/cells/${DATASET}__${MODEL}__instant__gate_soft.json"

python RQ7_causal_fusion/soft_cli.py \
  --dataset "$DATASET" --model "$MODEL" --seed "$SEED" --rho "$RHO" \
  --prior-split "$PRIOR_SPLIT" \
  --ae-dir "$AE_DIR" --data-root "$DATA_ROOT" \
  --cache-dir "$REPO_DIR/rq2_latent_vs_ob/.gpu_window_cache" \
  --ckpt-dir "$OUT_ROOT/ckpts" --out "$CELL" --device cuda

[[ -s "$CELL" ]] || { echo "FAILED: no JSON" >&2; exit 1; }
python - "$CELL" <<'PY' || exit 1
import json, sys
b = json.load(open(sys.argv[1]))
ok = [r for r in b.get("rows", []) if r.get("MAE") is not None]
if not ok:
    print(f"no usable row; failures={b.get('failures')}", file=sys.stderr); sys.exit(1)
r = ok[0]
if r.get("n_constraints", 0) == 0 and r.get("dataset") != "wastewater_nutrient":
    print("n_constraints=0: no prior was installed", file=sys.stderr); sys.exit(1)
print(f"OK MAE={r['MAE']:.5f} MAE_auto={r['MAE_auto']:.5f} n_cons={r.get('n_constraints')}")
PY
echo "DONE ($DATASET/$MODEL/s$SEED/rho=$RHO)"
