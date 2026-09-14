#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RQ3 = Path(__file__).resolve().parents[1]
RESULTS = RQ3 / "results"

sys.path.insert(0, str(RQ3.parent / "rq2_latent_vs_ob" / "scripts"))
from gen_seed_manifest import COST_H, DATASETS, MODELS, SEEDS

CODEC = "ae"
ARMS = ("none", "res", "res_zero", "film_zero", "xattn", "gate")


def out_root(codec: str, seed: int) -> Path:
    return RESULTS / (codec if seed == 0 else f"{codec}_seed{seed}")


def cell_json(arm: str, dataset: str, model: str, seed: int, codec: str = CODEC) -> Path:
    return out_root(codec, seed) / "cells" / f"{dataset}__{model}__{arm}__{codec}.json"


def ckpt_path(arm: str, dataset: str, model: str, seed: int, codec: str = CODEC) -> Path:
    return out_root(codec, seed) / "ckpts" / f"{dataset}__{model}__{arm}__{codec}.pt"


def legacy_json(dataset: str, model: str, seed: int, codec: str = CODEC) -> Path:
    return out_root(codec, seed) / f"{dataset}__{model}__{codec}.json"


def _rows(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text()).get("rows", [])
    except (json.JSONDecodeError, OSError):
        return []


def find_row(arm: str, dataset: str, model: str, seed: int, codec: str = CODEC) -> dict | None:
    for r in _rows(cell_json(arm, dataset, model, seed, codec)):
        if r.get("arm") == arm and r.get("MAE") is not None:
            return r
    if seed == 0:
        for r in _rows(legacy_json(dataset, model, seed, codec)):
            if r.get("arm") == arm and r.get("MAE") is not None:
                return r
    return None


def is_done(arm: str, dataset: str, model: str, seed: int, codec: str = CODEC) -> bool:
    if find_row(arm, dataset, model, seed, codec) is None:
        return False
    p = ckpt_path(arm, dataset, model, seed, codec)
    return p.is_file() and p.stat().st_size > 0


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
    p.add_argument("--codec", default=CODEC)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--only-dataset", nargs="*", default=None)
    p.add_argument("--exclude-dataset", nargs="*", default=[])
    p.add_argument("--force", action="store_true",
                   help="overwrite --out in place instead of picking a fresh _vN name (unsafe)")
    args = p.parse_args()

    datasets = [d for d in (args.only_dataset or DATASETS) if d not in args.exclude_dataset]

    todo, done_n = [], 0
    for ds in datasets:
        for arm in args.arms:
            for seed in args.seeds:
                for model in args.models:
                    if is_done(arm, ds, model, seed, args.codec):
                        done_n += 1
                    else:
                        todo.append((arm, ds, model, seed, COST_H[ds][model]))

    todo.sort(key=lambda r: (r[4], r[1], r[0], r[3]))

    total = done_n + len(todo)
    hours = sum(r[4] for r in todo)
    print(f"grid: {len(args.arms)} arms x {len(datasets)} datasets x {len(args.models)} models "
          f"x {len(args.seeds)} seeds = {total} cells   (codec={args.codec})")
    print(f"done: {done_n}   todo: {len(todo)}   estimated {hours:.0f} GPU-h remaining")

    by_ds: dict[str, float] = {}
    for arm, ds, model, seed, h in todo:
        by_ds[ds] = by_ds.get(ds, 0.0) + h
    for ds, h in sorted(by_ds.items(), key=lambda kv: -kv[1]):
        n = sum(1 for r in todo if r[1] == ds)
        print(f"  {ds:<22}{n:>5} cells{h:>9.1f} GPU-h")

    if args.report:
        return
    if not args.out:
        raise SystemExit("pass --out PATH to write the manifest (or --report to only summarize)")
    if not todo:
        print("nothing to do; manifest not written")
        return
    out = Path(args.out)
    if out.exists() and not args.force:
        alt = _next_free(out)
        print(f"\n{out} already exists; writing {alt} instead "
              f"(manifests are read by line number, so reusing a name can misindex a live array).")
        out = alt

    lines = [f"{arm} {ds} {model} {seed}" for arm, ds, model, seed, _ in todo]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}  ({len(lines)} lines)")
    print(f"submit with:  MANIFEST={out} sbatch --export=ALL "
          f"--array=0-{len(lines) - 1}%12 RQ3_fusion_benchmark/scripts/fusion_cells.slurm")
    if len(lines) > 1000:
        print(f"WARNING: {len(lines)} > MaxArraySize=1001; split with "
              f"--only-dataset / --exclude-dataset or submit in chunks")


if __name__ == "__main__":
    main()
