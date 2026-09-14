import re
from pathlib import Path
from typing import Literal
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


NO_ACTION_ID  = -1
MEAL_ID       = 0
CSII_BOLUS_ID = 15
INSULIN_IV_ID = 16

ACTION_NAMES: dict[int, str] = {
    NO_ACTION_ID:  "no action",
    MEAL_ID:       "meal",
    1:  "insulin aspart 70/30",
    2:  "insulin aspart",
    3:  "insulin glargine",
    4:  "insulin degludec",
    5:  "insulin detemir",
    6:  "insulin glulisine",
    7:  "novolin 30R",
    8:  "novolin 50R",
    9:  "novolin R",
    10: "humulin 70/30",
    11: "humulin R",
    12: "gansulin 40R",
    13: "gansulin R",
    14: "scilin M30",
    15: "CSII bolus (novolin R)",
    16: "insulin IV",
    17: "metformin",
    18: "acarbose",
    19: "voglibose",
    20: "gliclazide",
    21: "sitagliptin",
    22: "repaglinide",
    23: "dapagliflozin",
    24: "liraglutide",
    25: "glimepiride",
    26: "pioglitazone",
    27: "linagliptin",
    28: "canagliflozin",
    29: "empagliflozin",
    30: "gliquidone",
}

_DRUG_PATTERNS: list[tuple[str, int]] = [
    ("insulin aspart 70/30",  1),
    ("insulin aspart",        2),
    ("insulin glarg",         3),
    ("insulin degludec",      4),
    ("insulin detemir",       5),
    ("insulin glulisine",     6),
    ("novolin 30r",           7),
    ("novolin 50r",           8),
    ("novolin r",             9),
    ("humulin 70/30",        10),
    ("humulin r",            11),
    ("gansulin 40r",         12),
    ("gansulin r",           13),
    ("scilin m30",           14),
    ("metformin",            17),
    ("acarbose",             18),
    ("voglibose",            19),
    ("gliclazide",           20),
    ("sitagliptin",          21),
    ("repaglinide",          22),
    ("dapagliflozin",        23),
    ("liraglutide",          24),
    ("glimepiride",          25),
    ("pioglitazone",         26),
    ("linagliptin",          27),
    ("canagliflozin",        28),
    ("empagliflozin",        29),
    ("gliquidone",           30),
]

_MAX_GAP_MINUTES = 30


def _find_col(df: pd.DataFrame, *substrings: str) -> str | None:
    for sub in substrings:
        for col in df.columns:
            if sub.lower() in col.strip().lower():
                return col
    return None


def _interp_nan_1d(col: np.ndarray) -> np.ndarray:
    col = np.asarray(col, dtype=np.float32).copy()
    good = ~np.isnan(col)
    if good.all():
        return col
    if not good.any():
        return np.zeros_like(col)
    idx = np.arange(col.shape[0])
    return np.interp(idx, idx[good], col[good]).astype(np.float32)


def _window_1d_ok(col: np.ndarray, max_nan_ratio: float, max_nan_gap) -> bool:
    m = np.isnan(np.asarray(col, dtype=np.float32))
    if float(m.mean()) > max_nan_ratio:
        return False
    if max_nan_gap is not None and m.any():
        d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
        runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        if int(runs.max()) > max_nan_gap:
            return False
    return True


def _extract_iu(val) -> float:
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower()
    if "suspend" in s or "stop" in s:
        return 0.0
    m = re.search(r"([\d.]+)\s*iu", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else 0.0


def _text_to_all_drug_ids(val) -> list[int]:
    if pd.isna(val):
        return []
    s = str(val).lower().strip()
    if not s or "suspend" in s or "stop" in s:
        return []
    found = []
    for keyword, drug_id in _DRUG_PATTERNS:
        if keyword in s:
            found.append(drug_id)
            s = s.replace(keyword, " ")
    return found


def _build_action_col(
    df: pd.DataFrame,
    sc_col: str | None,
    bolus_col: str | None,
    iv_col: str | None,
    oral_col: str | None,
    meal_col: str | None,
) -> pd.Series:
    n       = len(df)
    actions: list[list[int]] = [[] for _ in range(n)]

    if meal_col is not None:
        for i in df.index[~df[meal_col].isna()]:
            actions[i].append(MEAL_ID)

    if oral_col is not None:
        for i, val in enumerate(df[oral_col]):
            actions[i].extend(_text_to_all_drug_ids(val))

    if iv_col is not None:
        for i, val in enumerate(df[iv_col]):
            if _extract_iu(val) > 0:
                actions[i].append(INSULIN_IV_ID)

    if bolus_col is not None:
        for i, val in enumerate(df[bolus_col]):
            if _extract_iu(val) > 0:
                actions[i].append(CSII_BOLUS_ID)

    if sc_col is not None:
        for i, val in enumerate(df[sc_col]):
            actions[i].extend(_text_to_all_drug_ids(val))

    return pd.Series(actions, index=df.index, dtype=object)


def _load_patient_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=0)

    cgm_col   = _find_col(df, "cgm")
    if cgm_col is None:
        raise KeyError("CGM column not found")
    df[cgm_col] = pd.to_numeric(df[cgm_col], errors="coerce")

    meal_col  = _find_col(df, "dietary intake")
    oral_col  = _find_col(df, "non-insulin")
    sc_col    = _find_col(df, "insulin dose - s.c.")
    bolus_col = _find_col(df, "bolus insulin")
    iv_col    = _find_col(df, "insulin dose - i.v.")

    df = df.sort_values("Date").reset_index(drop=True)

    out = pd.DataFrame()
    out["Date"]   = pd.to_datetime(df["Date"])
    out["cgm"]    = df[cgm_col].astype("float32")
    out["action"] = _build_action_col(df, sc_col, bolus_col, iv_col, oral_col, meal_col)

    return out


def _split_sessions(df: pd.DataFrame) -> list[pd.DataFrame]:
    if len(df) == 0:
        return []
    gaps = df["Date"].diff().dt.total_seconds().div(60).fillna(0)
    break_points = [0] + list((gaps > _MAX_GAP_MINUTES).to_numpy().nonzero()[0]) + [len(df)]
    return [
        df.iloc[s:e].reset_index(drop=True)
        for s, e in zip(break_points, break_points[1:])
        if e - s > 0
    ]


class ShanghaiDiabetesDataset(Dataset):

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["T1DM", "T2DM", "all"] = "all",
        subjects: list[str] | None = None,
        seq_len: int = 96,
        stride: int = 12,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int | None = 6,
        target_range: tuple[float, float] | None = (40.0, 400.0),
    ):
        self.seq_len      = seq_len
        self.stride       = stride
        self.max_nan_ratio = max_nan_ratio
        self.max_nan_gap  = max_nan_gap
        self.target_range = target_range

        subject_set = set(subjects) if subjects is not None else None
        data_root   = Path(data_root)

        dirs: list[tuple[Path, str]] = []
        if split in ("T1DM", "all"):
            dirs.append((data_root / "Shanghai_T1DM", "T1DM"))
        if split in ("T2DM", "all"):
            dirs.append((data_root / "Shanghai_T2DM", "T2DM"))

        self._windows: list[tuple[pd.DataFrame, str, str, int]] = []

        for folder, dtype in dirs:
            for path in sorted(folder.iterdir()):
                if path.suffix not in (".xls", ".xlsx"):
                    continue
                subject_id = path.stem.split("_")[0]
                if subject_set is not None and subject_id not in subject_set:
                    continue
                try:
                    df = _load_patient_df(path)
                except Exception as e:
                    print(f"[warn] skipping {path.name}: {e}")
                    continue

                for session in _split_sessions(df):
                    n = len(session)
                    for start in range(0, n - seq_len + 1, stride):
                        cgm_win = session["cgm"].iloc[start : start + seq_len].to_numpy()
                        if _window_1d_ok(cgm_win, max_nan_ratio, max_nan_gap):
                            self._windows.append((session, subject_id, dtype, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        session, subject_id, diabetes_type, start = self._windows[idx]
        w = session.iloc[start : start + self.seq_len]

        cgm = torch.from_numpy(_interp_nan_1d(w["cgm"].to_numpy()))
        if self.target_range is not None:
            lo, hi = self.target_range
            cgm = 2.0 * (cgm - lo) / (hi - lo) - 1.0

        action_id: dict[int, list[int]] = {
            int(i): lst
            for i, lst in enumerate(w["action"].tolist())
            if lst
        }
        action_type: dict[int, list[str]] = {
            i: [ACTION_NAMES.get(a, f"unknown({a})") for a in ids]
            for i, ids in action_id.items()
        }

        timestamp: list[str] = (
            w["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        )

        return {
            "target_ts":   cgm,
            "timestamp":   timestamp,
            "action_id":   action_id,
            "action_type": action_type,
        }

    @classmethod
    def subject_split(
        cls,
        data_root: str | Path,
        test_ratio: float = 0.2,
        seed: int = 42,
        split: Literal["T1DM", "T2DM", "all"] = "all",
        **kwargs,
    ) -> tuple["ShanghaiDiabetesDataset", "ShanghaiDiabetesDataset"]:
        data_root = Path(data_root)

        def collect_subjects(folder: Path) -> list[str]:
            return sorted({p.stem.split("_")[0]
                           for p in folder.iterdir()
                           if p.suffix in (".xls", ".xlsx")})

        rng = random.Random(seed)
        train_subjects, test_subjects = [], []

        for subdir, dtype in [("Shanghai_T1DM", "T1DM"), ("Shanghai_T2DM", "T2DM")]:
            if split != "all" and split != dtype:
                continue
            subjects = collect_subjects(data_root / subdir)
            rng.shuffle(subjects)
            n_test = max(1, round(len(subjects) * test_ratio))
            test_subjects  += subjects[:n_test]
            train_subjects += subjects[n_test:]

        return (
            cls(data_root, split=split, subjects=train_subjects, **kwargs),
            cls(data_root, split=split, subjects=test_subjects,  **kwargs),
        )
