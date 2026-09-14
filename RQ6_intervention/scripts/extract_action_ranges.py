from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "rq2_latent_vs_ob", REPO / "rq2_latent_vs_ob" / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.datasets import build_train_val

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_probe_cache import H, L, OPTIONS, ROOTS, STRIDE, VAL_RATIO, DATA_SEED
from units import window_units

from RQ6_intervention.priors import channels

OUT = REPO / "RQ6_intervention/results/action_ranges.json"

KIND = {
    "greenhouse": "minmax01",
    "vitaldb": "log1p01",
    "mimic_cardio": "minmax01",
    "pleiadata": "minmax11",
    "predist": "minmax11",
    "wastewater_nutrient": "minmax11",
    "cgmacros": "none",
    "shanghai_diabetes": "none",
}


def _global(name: str) -> dict | None:
    if name == "greenhouse":
        from dataset_utils.greenhouse import dataset as m
        return {"scope": "global",
                "ranges": [[float(lo), float(hi)] for _n, lo, hi in m.ACTION_CHANNELS]}
    if name == "vitaldb":
        from dataset_utils.vital_db import dataset as m
        return {"scope": "global",
                "ranges": [[0.0, float(scale)] for _n, scale, _lbl in m.ACTION_TS_CHANNELS]}
    if name in ("cgmacros", "shanghai_diabetes"):
        return {"scope": "global", "ranges": []}
    return None


def extract(name: str) -> dict:
    g = _global(name)
    if g is not None:
        return {"kind": KIND[name], "channels": list(channels(name).continuous), **g}

    train_raw, val_raw = build_train_val(
        name, seq_len=L + H, stride=STRIDE, val_ratio=VAL_RATIO, seed=DATA_SEED,
        root=ROOTS[name], options=OPTIONS.get(name, {}),
    )
    cont = list(channels(name).continuous)

    if name == "pleiadata":
        lo, hi = train_raw.setpoint_range
        return {"kind": KIND[name], "channels": cont, "scope": "global",
                "ranges": [[float(lo), float(hi)]]}

    if name == "wastewater_nutrient":
        return {"kind": KIND[name], "channels": cont, "scope": "global",
                "ranges": [[float(lo), float(hi)] for lo, hi in train_raw.action_ranges]}

    if name == "mimic_cardio":
        ar = train_raw.action_range
        return {"kind": KIND[name], "channels": cont, "scope": "global",
                "ranges": [[float(ar[c][0]), float(ar[c][1])] for c in cont]}

    if name == "predist":
        per_unit: dict[str, list] = {}
        for ds in (train_raw, val_raw):
            units = window_units(name, ds)
            for i, u in enumerate(units):
                key = str(u)
                if key in per_unit:
                    continue
                s_idx = ds.windows[i][0]
                rng = ds.sessions[s_idx]["ranges"]["action"]
                per_unit[key] = [[float(lo), float(hi)] for lo, hi in rng]
        return {"kind": KIND[name], "channels": cont, "scope": "per_unit",
                "ranges": per_unit}

    raise KeyError(name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(KIND))
    a = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ds in (a.datasets or list(KIND)):
        out[ds] = extract(ds)
        scope = out[ds]["scope"]
        n = len(out[ds]["ranges"])
        print(f"  [ranges] {ds:22s} kind={out[ds]['kind']:10s} scope={scope:9s} "
              f"{n} {'units' if scope == 'per_unit' else 'channels'}", flush=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True))
    tmp.replace(OUT)
    print(f"wrote {OUT}")
