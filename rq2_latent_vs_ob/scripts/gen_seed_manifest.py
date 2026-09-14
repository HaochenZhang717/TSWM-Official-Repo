#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

RQ2 = Path(__file__).resolve().parents[1]
RESULTS = RQ2 / "results"

ARMS = ("obs", "ae", "vae", "jepa")
DATASETS = ("greenhouse", "vitaldb", "cgmacros", "shanghai_diabetes",
            "pleiadata", "predist", "wastewater_nutrient", "mimic_cardio")
MODELS = ("TimeXer", "TiDE", "DUET", "PatchTST", "TimeKAN", "CrossLinear", "Amplifier")
SEEDS = (0, 1, 2, 3, 4)

COST_H = {
    "greenhouse":          {"TimeXer": 0.03, "TiDE": 0.03, "DUET": 0.03, "PatchTST": 0.08, "TimeKAN": 0.06, "CrossLinear": 0.02, "Amplifier": 0.02},
    "vitaldb":             {"TimeXer": 6.68, "TiDE": 2.05, "DUET": 4.86, "PatchTST": 10.07, "TimeKAN": 3.81, "CrossLinear": 1.34, "Amplifier": 1.37},
    "cgmacros":            {"TimeXer": 0.17, "TiDE": 0.28, "DUET": 0.13, "PatchTST": 0.36, "TimeKAN": 0.31, "CrossLinear": 0.10, "Amplifier": 0.08},
    "shanghai_diabetes":   {"TimeXer": 0.01, "TiDE": 0.02, "DUET": 0.01, "PatchTST": 0.03, "TimeKAN": 0.05, "CrossLinear": 0.01, "Amplifier": 0.01},
    "pleiadata":           {"TimeXer": 0.77, "TiDE": 0.72, "DUET": 0.44, "PatchTST": 2.71, "TimeKAN": 1.89, "CrossLinear": 0.50, "Amplifier": 0.32},
    "predist":             {"TimeXer": 1.70, "TiDE": 0.45, "DUET": 0.22, "PatchTST": 0.72, "TimeKAN": 0.65, "CrossLinear": 0.16, "Amplifier": 0.09},
    "wastewater_nutrient": {"TimeXer": 0.14, "TiDE": 0.06, "DUET": 0.11, "PatchTST": 0.27, "TimeKAN": 0.22, "CrossLinear": 0.03, "Amplifier": 0.02},
    "mimic_cardio":        {"TimeXer": 0.42, "TiDE": 0.31, "DUET": 0.24, "PatchTST": 0.80, "TimeKAN": 0.84, "CrossLinear": 0.19, "Amplifier": 0.17},
}


def out_root(arm: str, seed: int) -> Path:
    if arm == "obs":
        return RESULTS / ("action_conditioning" if seed == 0 else f"action_conditioning_seed{seed}")
    return RESULTS / "latent_space" / ("target" if seed == 0 else f"target_seed{seed}")


def eval_path(arm: str, dataset: str, model: str, seed: int) -> Path:
    suffix = "" if arm in ("obs", "ae") else f"__{arm}"
    return out_root(arm, seed) / "eval" / f"{dataset}__{model}{suffix}.json"


def is_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("MAE") is not None
    except (json.JSONDecodeError, OSError):
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
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
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
                for model in MODELS:
                    if is_done(eval_path(arm, ds, model, seed)):
                        done_n += 1
                    else:
                        todo.append((arm, ds, model, seed, COST_H[ds][model]))

    todo.sort(key=lambda r: (r[4], r[1], r[0], r[3]))

    total = done_n + len(todo)
    hours = sum(r[4] for r in todo)
    print(f"grid: {len(args.arms)} arms x {len(datasets)} datasets x {len(MODELS)} models "
          f"x {len(args.seeds)} seeds = {total} cells")
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
          f"--array=0-{len(lines) - 1}%12 rq2_latent_vs_ob/scripts/seeds_stage1.slurm")
    if len(lines) > 1000:
        print(f"WARNING: {len(lines)} > MaxArraySize=1001; split with "
              f"--only-dataset / --exclude-dataset or submit in chunks")


if __name__ == "__main__":
    main()
