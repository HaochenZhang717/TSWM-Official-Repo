from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "rq2_latent_vs_ob", REPO / "rq2_latent_vs_ob" / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.datasets import build_train_val
from common.gpu_dataset import FIELD_DTYPES
from common.windowing import Schema, unified_window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import window_starts, window_units

from RQ6_intervention.priors import channels, check_against_schema

CACHE16 = REPO / "rq2_latent_vs_ob/.gpu_window_cache"
OUT = REPO / "RQ6_intervention/.probe_val_cache"
L, H, STRIDE, VAL_RATIO, DATA_SEED = 256, 16, 80, 0.2, 0

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

_REL = {
    "greenhouse": "greenhouse3/TimeSeries",
    "vitaldb": "vital_db",
    "cgmacros": "diabetes_datasets/cgmacros",
    "shanghai_diabetes": "diabetes_datasets/Shanghai_T1DM_T2DM",
    "pleiadata": "PLEIAData",
    "predist": "PreDist/predist_dataset/manufacturer_2",
    "wastewater_nutrient": ("Wastewater_Treatment_Plant_Data_for_Nutrient_Removal_System/"
                            "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"),
    "mimic_cardio": "mimic_cardio",
}
ROOTS = {k: DATA_ROOT / v for k, v in _REL.items()}

OPTIONS = {"cgmacros": {"split": "all"}, "shanghai_diabetes": {"split": "T2DM"},
           "vitaldb": {"split": "all"}}

DATASETS = tuple(_REL)


def schema_and_vocab(name: str) -> tuple[Schema, dict | None]:
    for p in sorted(CACHE16.glob(f"{name}__*.pt")):
        blob = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        kp = blob["key_payload"]
        if kp["context_length"] == L and kp["horizon"] == H and kp["stride"] == STRIDE:
            return Schema(**blob["schema"]), blob["combo_to_id"]
    raise FileNotFoundError(f"{name}: no cache for L={L}/H={H}/s={STRIDE} under {CACHE16}")


def cache_path(name: str) -> Path:
    return OUT / f"{name}__L{L}__H{H}__s{STRIDE}.pt"


def build(name: str, force: bool = False) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = cache_path(name)
    if path.exists() and not force:
        print(f"  [probe-val] {name}: cached ({path.name})", flush=True)
        return path

    schema, combo_to_id = schema_and_vocab(name)
    check_against_schema(name, schema)

    _, val_raw = build_train_val(
        name, seq_len=L + H, stride=STRIDE, val_ratio=VAL_RATIO, seed=DATA_SEED,
        root=ROOTS[name], options=OPTIONS.get(name, {}),
    )
    n = len(val_raw)
    print(f"  [probe-val] {name}: materializing {n} val windows of {L + H} steps", flush=True)

    units = window_units(name, val_raw)
    starts = window_starts(name, val_raw)

    first = unified_window(val_raw[0], L, H, schema, combo_to_id)
    buf = {k: torch.empty((n, *a.shape), dtype=FIELD_DTYPES[k]) for k, a in first.items()}
    for k, a in first.items():
        buf[k][0] = torch.from_numpy(np.ascontiguousarray(a))
    for i in range(1, n):
        w = unified_window(val_raw[i], L, H, schema, combo_to_id)
        for k, a in w.items():
            buf[k][i] = torch.from_numpy(np.ascontiguousarray(a))

    ch = channels(name)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "tensors": buf, "n": n, "schema": asdict(schema), "combo_to_id": combo_to_id,
        "unit_ids": units, "starts": starts, "channel_names": asdict(ch),
        "context_length": L, "horizon": H, "stride": STRIDE,
        "val_ratio": VAL_RATIO, "data_seed": DATA_SEED, "root": str(ROOTS[name]),
        "options": OPTIONS.get(name, {}),
    }, tmp)
    tmp.replace(path)
    print(f"  [probe-val] {name}: wrote {path.name}  "
          f"(n={n}, units={len(set(units))})", flush=True)
    return path


def load(name: str) -> dict:
    blob = torch.load(cache_path(name), map_location="cpu", weights_only=False)
    blob["schema"] = Schema(**blob["schema"])
    return blob


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"DATA_ROOT = {DATA_ROOT}")
    for ds in (a.datasets or list(DATASETS)):
        build(ds, a.force)
