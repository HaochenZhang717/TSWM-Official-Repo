from __future__ import annotations

from typing import Any, Hashable

_WW_BUCKET_STEPS = 720


def _index_of(ds: Any) -> list:
    idx = getattr(ds, "_windows", None)
    if idx is None:
        idx = getattr(ds, "windows", None)
    if idx is None:
        raise AttributeError(f"{type(ds).__name__} has neither _windows nor windows")
    return idx


def window_units(name: str, ds: Any) -> list[Hashable]:
    idx = _index_of(ds)
    if len(idx) != len(ds):
        raise AssertionError(f"{name}: window index length {len(idx)} != len(ds) {len(ds)}")

    if name == "greenhouse":
        out = [w[0] for w in idx]
    elif name == "vitaldb":
        out = [int(w[0]) for w in idx]
    elif name == "cgmacros":
        out = [w[1] for w in idx]
    elif name == "shanghai_diabetes":
        out = [w[1] for w in idx]
    elif name == "pleiadata":
        out = [tuple(w[0]) if isinstance(w[0], (list, tuple)) else w[0] for w in idx]
    elif name in ("predist", "mimic_cardio"):
        sessions = ds.sessions
        out = [sessions[w[0]]["sid"] for w in idx]
    elif name == "wastewater_nutrient":
        out = [(int(w[0]), int(w[1]) // _WW_BUCKET_STEPS) for w in idx]
    else:
        raise KeyError(name)

    if len(out) != len(ds):
        raise AssertionError(f"{name}: unit list length {len(out)} != len(ds) {len(ds)}")
    return out


def window_starts(name: str, ds: Any) -> list[int]:
    idx = _index_of(ds)
    if name in ("greenhouse", "vitaldb", "predist", "wastewater_nutrient", "mimic_cardio"):
        return [int(w[-1]) for w in idx]
    if name in ("cgmacros", "shanghai_diabetes", "pleiadata"):
        return [int(w[-1]) for w in idx]
    raise KeyError(name)
