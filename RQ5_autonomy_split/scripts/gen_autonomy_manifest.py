#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RQ5 = Path(__file__).resolve().parents[1]
_ROOT = _RQ5.parent
_RQ2_SCRIPTS = _ROOT / "rq2_latent_vs_ob" / "scripts"
if str(_RQ2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_RQ2_SCRIPTS))

from gen_seed_manifest import COST_H, DATASETS, MODELS, SEEDS

RESULTS = _RQ5 / "results"

CODEC, FUSION, ENCODING = "ae", "gate", "instant"
LAMBDAS = (0.1, 0.3)

COST_CALIBRATION = 1.0 / 2.1


def lam_tag(lam: float) -> str:
    return f"lam{lam:g}".replace(".", "p")


def out_root(lam: float, seed: int) -> Path:
    stem = lam_tag(lam)
    return RESULTS / (stem if seed == 0 else f"{stem}_seed{seed}")


def cell_json(lam: float, dataset: str, model: str, seed: int) -> Path:
    return (out_root(lam, seed) / "cells"
            / f"{dataset}__{model}__{ENCODING}__{FUSION}__{lam_tag(lam)}.json")


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text()).get("rows", [])
    except (json.JSONDecodeError, OSError):
        return []


def is_done(lam: float, dataset: str, model: str, seed: int) -> bool:
    for r in _rows(cell_json(lam, dataset, model, seed)):
        if (r.get("dataset") == dataset and r.get("model") == model
                and r.get("fusion_arm") == FUSION and r.get("encoding") == ENCODING
                and abs(float(r.get("aux_auto", -1)) - lam) < 1e-9
                and r.get("MAE") is not None):
            return True
    return False


def _next_free(path: Path) -> Path:
    n = 2
    while (cand := path.with_name(f"{path.stem}_v{n}{path.suffix}")).exists():
        n += 1
    return cand


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="manifest path to write")
    p.add_argument("--report", action="store_true", help="print coverage only, write nothing")
    p.add_argument("--lambdas", nargs="*", type=float, default=list(LAMBDAS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--only-dataset", nargs="*", default=None)
    p.add_argument("--exclude-dataset", nargs="*", default=[])
    p.add_argument("--force", action="store_true", help="emit every cell, even finished ones")
    args = p.parse_args()

    datasets = [d for d in (args.only_dataset or DATASETS) if d not in args.exclude_dataset]

    lines, per_ds, todo_h, total = [], {}, 0.0, 0
    for ds in sorted(datasets, key=lambda d: sum(COST_H.get(d, {}).values())):
        for lam in args.lambdas:
            for model in args.models:
                for seed in args.seeds:
                    total += 1
                    if not args.force and is_done(lam, ds, model, seed):
                        continue
                    lines.append(f"{lam:g} {ds} {model} {seed}")
                    h = COST_H.get(ds, {}).get(model, 0.0)
                    todo_h += h
                    per_ds.setdefault(ds, [0, 0.0])
                    per_ds[ds][0] += 1
                    per_ds[ds][1] += h

    print(f"grid: {len(args.lambdas)} lambdas x {len(datasets)} datasets x "
          f"{len(args.models)} models x {len(args.seeds)} seeds = {total} cells   "
          f"(codec={CODEC}, fusion={FUSION}, encoding={ENCODING})")
    print(f"done: {total - len(lines)}   todo: {len(lines)}")
    print(f"estimated GPU-h remaining: ~{todo_h * COST_CALIBRATION:.0f} "
          f"(COST_H upper bound {todo_h:.0f})")
    for ds, (n, h) in sorted(per_ds.items(), key=lambda kv: -kv[1][1]):
        print(f"  {ds:<22} {n:>5} cells  ~{h * COST_CALIBRATION:>7.1f} GPU-h")
    print(f"\nlambda=0 baseline is NOT in this manifest: it is the gate/ae arm of the fusion sweep "
          f"({len(datasets) * len(args.models) * len(args.seeds)} cells).")

    if args.report or not args.out:
        if not args.out and not args.report:
            print("\n(no --out given; nothing written)")
        return

    out = Path(args.out)
    if out.exists():
        out = _next_free(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"\nwrote {out}  ({len(lines)} lines)")
    if lines:
        rel = out.relative_to(_ROOT) if out.is_absolute() and _ROOT in out.parents else out
        print(f"submit with:  MANIFEST={rel} sbatch --export=ALL "
              f"--array=0-{len(lines) - 1}%16 RQ5_autonomy_split/scripts/autonomy_cells.slurm")


if __name__ == "__main__":
    main()
