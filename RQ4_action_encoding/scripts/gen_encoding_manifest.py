#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RQ4 = Path(__file__).resolve().parents[1]
_ROOT = _RQ4.parent
_RQ2_SCRIPTS = _ROOT / "rq2_latent_vs_ob" / "scripts"
if str(_RQ2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_RQ2_SCRIPTS))

from gen_seed_manifest import COST_H, DATASETS, MODELS, SEEDS

RESULTS = _RQ4 / "results"
RQ3_RESULTS = _ROOT / "RQ3_fusion_benchmark" / "results"

CODEC = "ae"
ENCODINGS = ("decay", "conv", "attn")
FUSION_ARMS = ("gate", "film_zero")

COST_CALIBRATION = 1.0 / 2.1


def out_root(encoding: str, fusion: str, seed: int) -> Path:
    stem = f"{encoding}__{fusion}"
    return RESULTS / (stem if seed == 0 else f"{stem}_seed{seed}")


def cell_json(encoding: str, fusion: str, dataset: str, model: str, seed: int) -> Path:
    return (out_root(encoding, fusion, seed) / "cells"
            / f"{dataset}__{model}__{encoding}__{fusion}.json")


def ckpt_path(encoding: str, fusion: str, dataset: str, model: str, seed: int) -> Path:
    return (out_root(encoding, fusion, seed) / "ckpts"
            / f"{dataset}__{model}__{encoding}__{fusion}.pt")


def rq3_instant_json(fusion: str, dataset: str, model: str, seed: int) -> Path:
    root = RQ3_RESULTS / (CODEC if seed == 0 else f"{CODEC}_seed{seed}")
    return root / "cells" / f"{dataset}__{model}__{fusion}__{CODEC}.json"


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text()).get("rows", [])
    except (json.JSONDecodeError, OSError):
        return []


def is_done(encoding: str, fusion: str, dataset: str, model: str, seed: int) -> bool:
    for r in _rows(cell_json(encoding, fusion, dataset, model, seed)):
        if (r.get("encoding") == encoding and r.get("fusion_arm") == fusion
                and r.get("dataset") == dataset and r.get("model") == model
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
    p.add_argument("--encodings", nargs="*", default=list(ENCODINGS))
    p.add_argument("--fusion-arms", nargs="*", default=list(FUSION_ARMS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--only-dataset", nargs="*", default=None)
    p.add_argument("--exclude-dataset", nargs="*", default=[])
    p.add_argument("--force", action="store_true", help="emit every cell, even finished ones")
    args = p.parse_args()

    datasets = [d for d in (args.only_dataset or DATASETS) if d not in args.exclude_dataset]

    lines, per_ds, todo_h = [], {}, 0.0
    total = 0
    for ds in sorted(datasets, key=lambda d: sum(COST_H.get(d, {}).values())):
        for enc in args.encodings:
            for fus in args.fusion_arms:
                for model in args.models:
                    for seed in args.seeds:
                        total += 1
                        if not args.force and is_done(enc, fus, ds, model, seed):
                            continue
                        lines.append(f"{enc} {fus} {ds} {model} {seed}")
                        h = COST_H.get(ds, {}).get(model, 0.0)
                        todo_h += h
                        per_ds[ds] = per_ds.get(ds, [0, 0.0])
                        per_ds[ds][0] += 1
                        per_ds[ds][1] += h

    print(f"grid: {len(args.encodings)} encodings x {len(args.fusion_arms)} fusion x "
          f"{len(datasets)} datasets x {len(args.models)} models x {len(args.seeds)} seeds "
          f"= {total} cells   (codec={CODEC})")
    print(f"done: {total - len(lines)}   todo: {len(lines)}")
    print(f"estimated GPU-h remaining: ~{todo_h * COST_CALIBRATION:.0f} "
          f"(COST_H upper bound {todo_h:.0f}; see COST_CALIBRATION)")
    for ds, (n, h) in sorted(per_ds.items(), key=lambda kv: -kv[1][1]):
        print(f"  {ds:<22} {n:>5} cells  ~{h * COST_CALIBRATION:>7.1f} GPU-h")
    if "attn" in args.encodings:
        print("  note: `attn` is O(T^2) over 272 steps and will run somewhat above these.")

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
        print(f"submit with:  MANIFEST={out} sbatch --export=ALL "
              f"--array=0-{len(lines) - 1}%16 RQ4_action_encoding/scripts/encoding_cells.slurm")


if __name__ == "__main__":
    main()
