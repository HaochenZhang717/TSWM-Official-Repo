#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RQ2_ROOT = Path(__file__).resolve().parents[1]
for _p in (_RQ2_ROOT, _RQ2_ROOT / "common", _RQ2_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.gpu_dataset import prepare_gpu


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--root", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    options = {}
    if args.split:
        options["split"] = args.split
    if args.download:
        options["download"] = True

    data = prepare_gpu(
        args.dataset,
        context_length=args.context_length,
        horizon=args.horizon,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        root=args.root,
        options=options or None,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    n_train = len(data.train)
    n_val = len(data.val)
    print(f"[warm] {args.dataset} seed={args.seed}: {n_train} train / {n_val} val windows cached")


if __name__ == "__main__":
    main()
