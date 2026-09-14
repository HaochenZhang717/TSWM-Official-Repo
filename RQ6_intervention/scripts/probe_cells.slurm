#!/bin/bash

#SBATCH --job-name=rq6-probe
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq6-probe-%A_%a.out
#SBATCH --error=logs/rq6-probe-%A_%a.err

set -uxo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-tswm}"

REPO_DIR="${REPO_DIR:?set REPO_DIR to the repository root}"
cd "$REPO_DIR"

export TMPDIR="$REPO_DIR/.tmp"; mkdir -p "$TMPDIR"

MANIFEST="${MANIFEST:?set MANIFEST=<path> (see gen_probe_manifest.py)}"
[[ "$MANIFEST" != /* ]] && MANIFEST="$REPO_DIR/$MANIFEST"
T_IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((T_IDX + 1))p" "$MANIFEST")
[[ -z "$LINE" ]] && { echo "ERROR: manifest line $((T_IDX + 1)) is empty ($MANIFEST)" >&2; exit 1; }
read -r DATASET ARM EXTRA <<<"$LINE"
[[ -z "${DATASET:-}" || -z "${ARM:-}" ]] && { echo "ERROR: bad manifest line: $LINE" >&2; exit 1; }

OUT_JSON="$REPO_DIR/RQ6_intervention/results/probe_intervention__${DATASET}__${ARM}.json"
mkdir -p "$(dirname "$OUT_JSON")"

BANK="$REPO_DIR/RQ6_intervention/.probe_val_cache/${DATASET}__L256__H16__s80.pt"
RANGES="$REPO_DIR/RQ6_intervention/results/action_ranges.json"
for f in "$BANK" "$RANGES"; do
  [[ -s "$f" ]] || { echo "ERROR: missing prerequisite $f -- run build_probe_cache.py / extract_action_ranges.py on the login node first" >&2; exit 1; }
done

python RQ6_intervention/scripts/probe_intervention.py \
  --dataset "$DATASET" --arms "$ARM" --out "$OUT_JSON" $EXTRA

[[ -s "$OUT_JSON" ]] || { echo "FAILED (task=$T_IDX $DATASET/$ARM): no JSON at $OUT_JSON" >&2; exit 1; }
python - "$OUT_JSON" "$ARM" <<'PY' || exit 1
import json, sys
blob = json.load(open(sys.argv[1])); arm = sys.argv[2]
rows = [r for r in blob.get("rows", []) if r.get("fusion_arm") == arm]
if not rows:
    print(f"no rows for arm {arm} in {sys.argv[1]}", file=sys.stderr); sys.exit(1)
bad = [r for r in rows if r.get("MAE") is None or r.get("R_perm") is None]
if bad:
    print(f"{len(bad)} rows missing MAE/R_perm", file=sys.stderr); sys.exit(1)
worst = max(r.get("R_same_max", 0.0) for r in rows)
if worst > 0.0:
    print(f"A_same NOT bitwise identical: max|delta|={worst:.3e}", file=sys.stderr); sys.exit(1)
print(f"OK {len(rows)} cells, {len(blob.get('failures', {}))} missing ckpts")
PY

echo "DONE (task=$T_IDX $DATASET/$ARM) -> $OUT_JSON"
