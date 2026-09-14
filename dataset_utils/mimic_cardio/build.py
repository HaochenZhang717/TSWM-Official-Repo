from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_ITEMS = {
    220045: "HR", 220210: "RR", 220277: "SpO2",
    220179: "NBPs", 220180: "NBPd", 220181: "NBPm",
}
TARGET_CHANNELS = ("HR", "RR", "SpO2", "NBPs", "NBPd", "NBPm")

TARGET_RANGE = {
    "HR": (10.0, 300.0), "RR": (0.0, 80.0), "SpO2": (20.0, 100.0),
    "NBPs": (30.0, 300.0), "NBPd": (5.0, 225.0), "NBPm": (20.0, 250.0),
}
TARGET_FILL = {"HR": 80.0, "RR": 16.0, "SpO2": 97.0,
               "NBPs": 120.0, "NBPd": 70.0, "NBPm": 85.0}

DRUG_ITEMS = {
    221906: "norepinephrine",
    221289: "epinephrine", 229617: "epinephrine",
    221749: "phenylephrine", 229630: "phenylephrine", 229632: "phenylephrine",
    222315: "vasopressin",
    221662: "dopamine",
    221653: "dobutamine",
    221986: "milrinone",
    221794: "furosemide", 228340: "furosemide",
    225158: "fluid_bolus", 220955: "fluid_bolus", 220953: "fluid_bolus",
}
INFUSION_CHANNELS = ("norepinephrine", "epinephrine", "phenylephrine", "vasopressin",
                     "dopamine", "dobutamine", "milrinone", "furosemide", "fluid_bolus")
INFUSION_RATE_UNIT = {
    "norepinephrine": "mcg/kg/min", "epinephrine": "mcg/kg/min",
    "phenylephrine": "mcg/kg/min", "dopamine": "mcg/kg/min",
    "dobutamine": "mcg/kg/min", "milrinone": "mcg/kg/min",
    "vasopressin": "units/hour", "furosemide": "mg/hour", "fluid_bolus": "mL/hour",
}
BOLUS_CHANNELS = ("furosemide", "fluid_bolus")
BOLUS_AMOUNT_UNIT = {"furosemide": "mg", "fluid_bolus": "mL"}
BOLUS_CLINICAL_REF = {"furosemide": 200.0, "fluid_bolus": 1000.0}

KIND_INFUSION = {"Continuous Med", "Continuous IV"}
KIND_BOLUS = {"Drug Push", "Bolus"}

INF_COLS = tuple(f"inf_{c}" for c in INFUSION_CHANNELS)
BOL_COLS = tuple(f"bol_{c}" for c in BOLUS_CHANNELS)

VENT_ITEMS = {223835: "FiO2", 220339: "PEEP", 223834: "O2Flow",
              224684: "SetTidalVolume", 224688: "SetRespRate"}
VENT_CHANNELS = ("FiO2", "PEEP", "O2Flow", "SetTidalVolume", "SetRespRate")
VENT_RANGE = {"FiO2": (21.0, 100.0), "PEEP": (0.0, 40.0), "O2Flow": (0.0, 100.0),
              "SetTidalVolume": (0.0, 2000.0), "SetRespRate": (0.0, 60.0)}
VENT_BASELINE = {"FiO2": 21.0, "PEEP": 0.0, "O2Flow": 0.0,
                 "SetTidalVolume": 0.0, "SetRespRate": 0.0}

ACTION_CONT_COLS = INF_COLS + BOL_COLS + VENT_CHANNELS

MIN_LOS_HOURS = 12.0
MIN_TARGET_POINTS = 3
ACTION_HI_PCT = 99.5
CHUNK = 4_000_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_cohort(root: Path, max_stays: int | None) -> pd.DataFrame:
    icu = pd.read_csv(root / "icu" / "icustays.csv",
                      usecols=["subject_id", "hadm_id", "stay_id", "first_careunit",
                               "intime", "outtime", "los"],
                      parse_dates=["intime", "outtime"])
    dx = pd.read_csv(root / "hosp" / "diagnoses_icd.csv",
                     usecols=["hadm_id", "icd_code", "icd_version"],
                     dtype={"icd_code": "string"})
    code = dx["icd_code"].str.strip()
    p3 = pd.to_numeric(code.str.slice(0, 3), errors="coerce")
    is_cardio = (((dx["icd_version"] == 9) & p3.between(390, 459))
                 | ((dx["icd_version"] == 10) & code.str.upper().str.startswith("I")))
    cardio = set(dx.loc[is_cardio, "hadm_id"].dropna().astype("int64").unique())

    icu["los_hours"] = icu["los"] * 24.0
    icu.loc[icu["los_hours"].isna(), "los_hours"] = (
        (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0)
    coh = icu[icu["hadm_id"].isin(cardio) & (icu["los_hours"] >= MIN_LOS_HOURS)].copy()
    coh = coh.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    if max_stays:
        coh = coh.head(max_stays).copy()
    log(f"cohort: {len(coh):,} stays / {coh['subject_id'].nunique():,} subjects")
    return coh


def scan_chartevents(root: Path, stay_set: set, max_subj: int | None) -> pd.DataFrame:
    keep = set(TARGET_ITEMS) | set(VENT_ITEMS)
    parts, seen = [], 0
    reader = pd.read_csv(root / "icu" / "chartevents.csv",
                         usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
                         dtype={"subject_id": "int64", "stay_id": "Int64",
                                "itemid": "Int64", "valuenum": "float64",
                                "charttime": "string"},
                         chunksize=CHUNK)
    for chunk in reader:
        seen += len(chunk)
        if max_subj is not None and int(chunk["subject_id"].iloc[0]) > max_subj:
            break
        m = chunk["itemid"].isin(keep) & chunk["stay_id"].isin(stay_set)
        if m.any():
            parts.append(chunk.loc[m, ["stay_id", "charttime", "itemid", "valuenum"]])
        if seen % (25 * CHUNK) == 0:
            log(f"  chartevents ~{seen:,} rows, kept {sum(len(p) for p in parts):,}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["stay_id", "charttime", "itemid", "valuenum"])
    log(f"  chartevents done: kept {len(df):,} rows")
    return df.dropna(subset=["valuenum", "stay_id", "charttime"])


def scan_inputevents(root: Path, stay_set: set, max_subj: int | None) -> pd.DataFrame:
    parts, seen = [], 0
    reader = pd.read_csv(root / "icu" / "inputevents.csv",
                         usecols=["subject_id", "stay_id", "starttime", "endtime", "itemid",
                                  "amount", "rate", "ordercategorydescription"],
                         dtype={"subject_id": "int64", "stay_id": "Int64", "itemid": "Int64",
                                "amount": "float64", "rate": "float64",
                                "starttime": "string", "endtime": "string",
                                "ordercategorydescription": "string"},
                         chunksize=CHUNK)
    for chunk in reader:
        seen += len(chunk)
        if max_subj is not None and int(chunk["subject_id"].iloc[0]) > max_subj:
            break
        m = chunk["itemid"].isin(DRUG_ITEMS) & chunk["stay_id"].isin(stay_set)
        if m.any():
            parts.append(chunk.loc[m])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    log(f"  inputevents done: kept {len(df):,} administration intervals")
    return df


def _chart_grid(chart: pd.DataFrame, items: dict, clip: dict, channels: tuple,
                coh: pd.DataFrame, dt_h: float, n_steps: pd.Series,
                agg: str) -> pd.DataFrame:
    sub = chart[chart["itemid"].isin(items)].copy()
    sub["channel"] = sub["itemid"].map(items)
    sub["t"] = pd.to_datetime(sub["charttime"], errors="coerce")
    sub = sub.dropna(subset=["t"])
    sub["t_rel"] = ((sub["t"] - sub["stay_id"].map(coh.set_index("stay_id")["intime"]))
                    .dt.total_seconds() / 3600.0)
    sub = sub[sub["t_rel"] >= 0]
    keep = np.ones(len(sub), dtype=bool)
    ch, val = sub["channel"].to_numpy(), sub["valuenum"].to_numpy()
    for c, (lo, hi) in clip.items():
        sel = ch == c
        keep &= ~(sel & ((val < lo) | (val > hi)))
    sub = sub[keep]
    sub["step"] = np.floor(sub["t_rel"] / dt_h).astype(np.int64)
    sub = sub[sub["step"] < sub["stay_id"].map(n_steps).fillna(0).astype(np.int64)]
    g = sub.groupby(["stay_id", "step", "channel"], observed=True)["valuenum"].agg(agg)
    return g.unstack("channel").reindex(columns=list(channels))


def _prep_inputs(inp: pd.DataFrame, coh: pd.DataFrame, dt_h: float,
                 n_steps: pd.Series) -> pd.DataFrame:
    d = inp[inp["stay_id"].notna()].copy()
    d["stay_id"] = d["stay_id"].astype("int64")
    d["drug"] = d["itemid"].map(DRUG_ITEMS)
    d = d.dropna(subset=["drug"])
    d["t0"] = pd.to_datetime(d["starttime"], errors="coerce")
    d["t1"] = pd.to_datetime(d["endtime"], errors="coerce")
    d = d.dropna(subset=["t0"])
    d["t1"] = d["t1"].fillna(d["t0"])
    intime = coh.set_index("stay_id")["intime"]
    d["h0"] = (d["t0"] - d["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    d["h1"] = (d["t1"] - d["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    d["h1"] = np.maximum(d["h1"], d["h0"])
    d = d[d["h1"] >= 0]
    d["h0"] = np.maximum(d["h0"], 0.0)
    d["nmax"] = d["stay_id"].map(n_steps).fillna(0).astype(np.int64)
    d = d[(d["nmax"] > 0) & (d["h0"] < d["nmax"] * dt_h)]
    d["h1"] = np.minimum(d["h1"], d["nmax"] * dt_h)
    d["kind"] = d["ordercategorydescription"].fillna("")
    return d


def grid_infusions(d: pd.DataFrame, dt_h: float, index: pd.MultiIndex) -> pd.DataFrame:
    out = np.zeros((len(index), len(INFUSION_CHANNELS)), dtype=np.float64)
    sel = d["kind"].isin(KIND_INFUSION) & d["rate"].notna() & (d["rate"] > 0)
    inf = d[sel]
    if not len(inf):
        return pd.DataFrame(out, index=index, columns=list(INF_COLS))

    pos = {c: i for i, c in enumerate(INFUSION_CHANNELS)}
    row_of = pd.Series(np.arange(len(index)), index=index)
    ci = inf["drug"].map(pos).to_numpy()
    s0 = np.floor(inf["h0"].to_numpy() / dt_h).astype(np.int64)
    s1 = np.minimum(np.ceil(inf["h1"].to_numpy() / dt_h).astype(np.int64),
                    inf["nmax"].to_numpy())
    s1 = np.maximum(s1, s0 + 1)
    rate = inf["rate"].to_numpy()
    sids = inf["stay_id"].to_numpy()
    h0, h1 = inf["h0"].to_numpy(), inf["h1"].to_numpy()
    for k in range(len(inf)):
        steps = np.arange(s0[k], s1[k])
        lo = np.maximum(h0[k], steps * dt_h)
        hi = np.minimum(h1[k], (steps + 1) * dt_h)
        w = np.clip(hi - lo, 0.0, None) / dt_h
        ridx = row_of.reindex(list(zip([sids[k]] * len(steps), steps))).to_numpy()
        ok = ~np.isnan(ridx) & (w > 0)
        if ok.any():
            np.add.at(out, (ridx[ok].astype(np.int64), ci[k]), rate[k] * w[ok])
    return pd.DataFrame(out, index=index, columns=list(INF_COLS))


def grid_boluses(d: pd.DataFrame, dt_h: float, index: pd.MultiIndex) -> pd.DataFrame:
    out = np.zeros((len(index), len(BOLUS_CHANNELS)), dtype=np.float64)
    sel = (d["kind"].isin(KIND_BOLUS) & d["drug"].isin(BOLUS_CHANNELS)
           & d["amount"].notna() & (d["amount"] > 0))
    bol = d[sel]
    if not len(bol):
        return pd.DataFrame(out, index=index, columns=list(BOL_COLS))

    pos = {c: i for i, c in enumerate(BOLUS_CHANNELS)}
    row_of = pd.Series(np.arange(len(index)), index=index)
    cb = bol["drug"].map(pos).to_numpy()
    sb = np.floor(bol["h0"].to_numpy() / dt_h).astype(np.int64)
    ridx = row_of.reindex(list(zip(bol["stay_id"].to_numpy(), sb))).to_numpy()
    ok = ~np.isnan(ridx)
    if ok.any():
        np.add.at(out, (ridx[ok].astype(np.int64), cb[ok]),
                  bol["amount"].to_numpy()[ok])
    return pd.DataFrame(out, index=index, columns=list(BOL_COLS))


def build(mimic_root: Path, out_root: Path, grid_min: int, max_stays: int | None,
          min_steps: int) -> None:
    dt_h = grid_min / 60.0
    out_dir = out_root / f"grid_{grid_min}min"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("[1/6] build cohort")
    coh = build_cohort(mimic_root, max_stays)
    max_subj = int(coh["subject_id"].max()) if max_stays else None
    coh["n_steps"] = np.floor(coh["los_hours"] / dt_h).astype(np.int64)
    coh = coh[coh["n_steps"] >= min_steps].reset_index(drop=True)
    stay_set = set(coh["stay_id"].astype(int))
    n_steps = coh.set_index("stay_id")["n_steps"]
    log(f"      grid {grid_min}min, median steps per stay {coh['n_steps'].median():.0f}, "
        f"stays with >= {min_steps} steps: {len(coh):,}")

    log("[2/6] scan chartevents (vitals + ventilator settings)")
    chart = scan_chartevents(mimic_root, stay_set, max_subj)
    chart["stay_id"] = chart["stay_id"].astype("int64")

    log("[3/6] grid the vitals and require the core channels")
    tgt = _chart_grid(chart, TARGET_ITEMS, TARGET_RANGE, TARGET_CHANNELS,
                      coh, dt_h, n_steps, agg="mean")
    cnt = tgt.notna().groupby(level="stay_id").sum()
    good = set(cnt.index[(cnt >= MIN_TARGET_POINTS).all(axis=1)])
    coh = coh[coh["stay_id"].isin(good)].reset_index(drop=True)
    stay_set = set(coh["stay_id"])
    n_steps = coh.set_index("stay_id")["n_steps"]
    log(f"      stays with all {len(TARGET_CHANNELS)} core channels: {len(coh):,}")

    log("[4/6] assemble the grid skeleton and forward-fill causally")
    idx = pd.MultiIndex.from_arrays(
        [np.repeat(coh["stay_id"].to_numpy(), coh["n_steps"].to_numpy()),
         np.concatenate([np.arange(n) for n in coh["n_steps"].to_numpy()])],
        names=["stay_id", "step"])
    tgt = tgt.reindex(idx).groupby(level="stay_id").ffill()
    for c in TARGET_CHANNELS:
        tgt[c] = tgt[c].fillna(TARGET_FILL[c])

    vent = _chart_grid(chart, VENT_ITEMS, VENT_RANGE, VENT_CHANNELS,
                       coh, dt_h, n_steps, agg="last")
    vent = vent.reindex(idx).groupby(level="stay_id").ffill()
    for c in VENT_CHANNELS:
        vent[c] = vent[c].fillna(VENT_BASELINE[c])
    del chart

    log("[5/6] scan inputevents and grid them by administration route")
    inp = scan_inputevents(mimic_root, stay_set, max_subj)
    d = _prep_inputs(inp, coh, dt_h, n_steps) if len(inp) else pd.DataFrame(
        columns=["stay_id", "drug", "h0", "h1", "nmax", "kind", "rate", "amount"])
    del inp
    n_inf = int((d["kind"].isin(KIND_INFUSION)).sum()) if len(d) else 0
    n_bol = int((d["kind"].isin(KIND_BOLUS) & d["drug"].isin(BOLUS_CHANNELS)).sum()) if len(d) else 0
    log(f"      continuous infusions {n_inf:,} segments / boluses {n_bol:,}")
    inf = grid_infusions(d, dt_h, idx)
    bol = grid_boluses(d, dt_h, idx)
    del d

    log("[6/6] write to disk")
    out = pd.DataFrame(index=idx).reset_index()
    intime = coh.set_index("stay_id")["intime"]
    out["t_rel_hours"] = out["step"] * dt_h
    out["timestamp"] = (out["stay_id"].map(intime)
                        + pd.to_timedelta(out["t_rel_hours"], unit="h")
                        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in TARGET_CHANNELS:
        out[c] = tgt[c].to_numpy(dtype=np.float32)
    for c in INF_COLS:
        out[c] = inf[c].to_numpy(dtype=np.float32)
    for c in BOL_COLS:
        out[c] = bol[c].to_numpy(dtype=np.float32)
    for c in VENT_CHANNELS:
        out[c] = vent[c].to_numpy(dtype=np.float32)

    out.to_parquet(out_dir / "grid.parquet", index=False)
    coh[["subject_id", "hadm_id", "stay_id", "first_careunit",
         "los_hours", "n_steps"]].to_parquet(out_dir / "cohort.parquet", index=False)

    action_hi, clip_frac, nonzero = {}, {}, {}
    for c in INF_COLS + BOL_COLS:
        v = out[c].to_numpy()
        nz = v[v > 0]
        hi = float(np.percentile(nz, ACTION_HI_PCT)) if nz.size else 1.0
        action_hi[c] = max(hi, 1e-6)
        clip_frac[c] = float((nz > action_hi[c]).mean()) if nz.size else 0.0
        nonzero[c] = float((v > 0).mean())

    summary = {
        "grid_minutes": grid_min,
        "n_stays": int(len(coh)),
        "n_subjects": int(coh["subject_id"].nunique()),
        "n_rows": int(len(out)),
        "median_steps_per_stay": float(coh["n_steps"].median()),
        "target_channels": list(TARGET_CHANNELS),
        "action_cont_channels": list(ACTION_CONT_COLS),
        "bolus_event_channels": list(BOLUS_CHANNELS),
        "infusion_rate_units": INFUSION_RATE_UNIT,
        "bolus_amount_units": BOLUS_AMOUNT_UNIT,
        "action_hi": action_hi,
        "action_hi_percentile": ACTION_HI_PCT,
        "action_clip_frac": clip_frac,
        "action_nonzero_frac": nonzero,
        "bolus_clinical_reference": {f"bol_{k}": v for k, v in BOLUS_CLINICAL_REF.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {time.time()-t0:.0f}s -> {out_dir}")
    log(f"  {len(coh):,} stays, {len(out):,} grid rows, "
        f"{len(TARGET_CHANNELS)} target + {len(ACTION_CONT_COLS)} action channels")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mimic-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--grid-min", type=int, default=30,
                   help="grid resolution in minutes (vitals are native q1h, ventilator settings q4h)")
    p.add_argument("--max-stays", type=int, default=None, help="smoke test; None = all stays")
    p.add_argument("--min-steps", type=int, default=272,
                   help="keep a stay only if it has at least this many steps (default = 256 + 16)")
    a = p.parse_args()
    build(a.mimic_root, a.out_root, a.grid_min, a.max_stays, a.min_steps)


if __name__ == "__main__":
    main()
