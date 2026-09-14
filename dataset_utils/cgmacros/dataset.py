from pathlib import Path
from typing import Literal
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


NO_ACTION_ID = -1

ACTION_NAMES: dict[int, str] = {
    NO_ACTION_ID: "no action",
    0: "breakfast",
    1: "lunch",
    2: "dinner",
    3: "snacks",
}

_MEAL_PATTERNS: list[tuple[str, int]] = [
    ("breakfast", 0),
    ("lunch",     1),
    ("dinner",    2),
    ("snack",     3),
]

_NATIVE_INTERVAL_MIN = 1


def _find_col(df: pd.DataFrame, *substrings: str) -> str | None:
    for sub in substrings:
        for col in df.columns:
            if sub.lower() in col.strip().lower():
                return col
    return None


def _find_exact_col(df: pd.DataFrame, name: str) -> str | None:
    for col in df.columns:
        if col.strip().lower() == name.lower():
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


_NUTRITION_FIELDS: list[tuple[str, str, str, bool]] = [
    ("cal",     "cal",     "kcal", True),
    ("carbs",   "carb",    "g",    False),
    ("protein", "protein", "g",    False),
    ("fat",     "fat",     "g",    False),
    ("fiber",   "fiber",   "g",    False),
]


def _format_meal(rec: dict) -> str:
    name  = ACTION_NAMES.get(rec["id"], f"unknown({rec['id']})")
    parts = []
    for key, label, unit, _ in _NUTRITION_FIELDS:
        v = rec.get(key)
        if v is not None and not pd.isna(v):
            parts.append(f"{label}={v:g}{unit}")
    return f"{name} ({', '.join(parts)})" if parts else name


def _meal_to_ids(val) -> list[int]:
    if pd.isna(val):
        return []
    s = str(val).lower().strip()
    if not s:
        return []
    for keyword, mid in _MEAL_PATTERNS:
        if keyword in s:
            return [mid]
    return []


def _load_patient_df(path: Path, resample_minutes: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col   = _find_col(df, "timestamp")
    cgm_col  = _find_col(df, "libre")
    meal_col = _find_col(df, "meal type")
    if ts_col is None or cgm_col is None:
        raise KeyError(f"missing timestamp/Libre column: {list(df.columns)}")

    nut_cols: dict[str, str | None] = {}
    for key, _label, _unit, exact in _NUTRITION_FIELDS:
        src = "calories" if key == "cal" else key
        nut_cols[key] = _find_exact_col(df, src) if exact else _find_col(df, src)

    actions: list[list[dict]] = []
    if meal_col is None:
        actions = [[] for _ in range(len(df))]
    else:
        for _, row in df.iterrows():
            recs = []
            for mid in _meal_to_ids(row[meal_col]):
                rec = {"id": mid}
                for key, col in nut_cols.items():
                    rec[key] = (pd.to_numeric(row[col], errors="coerce")
                                if col is not None else None)
                recs.append(rec)
            actions.append(recs)

    out = pd.DataFrame()
    out["Date"]   = pd.to_datetime(df[ts_col])
    out["cgm"]    = pd.to_numeric(df[cgm_col], errors="coerce").astype("float32")
    out["action"] = pd.Series(actions, index=df.index, dtype=object)

    out = out.sort_values("Date").reset_index(drop=True)

    if resample_minutes is not None:
        out = _resample(out, resample_minutes)

    return out


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    binned = df.copy()
    binned["Date"] = binned["Date"].dt.floor(f"{minutes}min")
    g = binned.groupby("Date", sort=True)
    cgm    = g["cgm"].mean()
    action = g["action"].apply(lambda lists: [a for lst in lists for a in lst])
    return pd.DataFrame({
        "Date":   cgm.index,
        "cgm":    cgm.to_numpy(dtype="float32"),
        "action": action.to_numpy(),
    }).reset_index(drop=True)


def _split_sessions(df: pd.DataFrame, gap_minutes: float) -> list[pd.DataFrame]:
    if len(df) == 0:
        return []
    gaps = df["Date"].diff().dt.total_seconds().div(60).fillna(0)
    break_points = [0] + list((gaps > gap_minutes).to_numpy().nonzero()[0]) + [len(df)]
    return [
        df.iloc[s:e].reset_index(drop=True)
        for s, e in zip(break_points, break_points[1:])
        if e - s > 0
    ]


def _classify_a1c(a1c: float) -> str:
    if pd.isna(a1c):
        return "unknown"
    if a1c < 5.7:
        return "healthy"
    if a1c < 6.5:
        return "pre"
    return "t2d"


def _load_subject_classes(cgm_dir: Path) -> dict[str, str]:
    bio_path = cgm_dir / "bio.csv"
    if not bio_path.exists():
        return {}
    bio = pd.read_csv(bio_path)
    subj_col = _find_col(bio, "subject")
    a1c_col  = _find_col(bio, "a1c")
    if subj_col is None or a1c_col is None:
        return {}
    classes: dict[str, str] = {}
    for _, row in bio.iterrows():
        try:
            sid = f"{int(row[subj_col]):03d}"
        except (ValueError, TypeError):
            continue
        classes[sid] = _classify_a1c(pd.to_numeric(row[a1c_col], errors="coerce"))
    return classes


class CGMacrosDataset(Dataset):

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["all", "healthy", "pre", "t2d"] = "all",
        subjects: list[str] | None = None,
        seq_len: int = 1440,
        stride: int = 60,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int | None = 6,
        target_range: tuple[float, float] | None = (40.0, 400.0),
        resample_minutes: int | None = None,
    ):
        self.seq_len          = seq_len
        self.stride           = stride
        self.max_nan_ratio    = max_nan_ratio
        self.max_nan_gap      = max_nan_gap
        self.target_range     = target_range
        self.resample_minutes = resample_minutes

        interval    = resample_minutes or _NATIVE_INTERVAL_MIN
        gap_minutes = max(30.0, interval * 2.0)

        cgm_dir     = Path(data_root) / "CGMacros"
        subject_set = set(subjects) if subjects is not None else None
        classes     = _load_subject_classes(cgm_dir) if split != "all" else {}

        self._windows: list[tuple[pd.DataFrame, str, int]] = []

        for folder in sorted(cgm_dir.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("CGMacros-"):
                continue
            subject_id = folder.name.split("-")[-1]
            csv_path   = folder / f"{folder.name}.csv"
            if not csv_path.exists():
                continue
            if subject_set is not None and subject_id not in subject_set:
                continue
            if split != "all" and classes.get(subject_id) != split:
                continue

            try:
                df = _load_patient_df(csv_path, resample_minutes)
            except Exception as e:
                print(f"[warn] skipping {csv_path.name}: {e}")
                continue

            for session in _split_sessions(df, gap_minutes):
                n = len(session)
                for start in range(0, n - seq_len + 1, stride):
                    cgm_win = session["cgm"].iloc[start : start + seq_len].to_numpy()
                    if _window_1d_ok(cgm_win, max_nan_ratio, max_nan_gap):
                        self._windows.append((session, subject_id, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        session, _subject_id, start = self._windows[idx]
        w = session.iloc[start : start + self.seq_len]

        cgm = torch.from_numpy(_interp_nan_1d(w["cgm"].to_numpy()))
        if self.target_range is not None:
            lo, hi = self.target_range
            cgm = 2.0 * (cgm - lo) / (hi - lo) - 1.0

        records: dict[int, list[dict]] = {
            int(i): recs
            for i, recs in enumerate(w["action"].tolist())
            if recs
        }
        action_id: dict[int, list[int]] = {
            i: [r["id"] for r in recs] for i, recs in records.items()
        }
        action_type: dict[int, list[str]] = {
            i: [_format_meal(r) for r in recs] for i, recs in records.items()
        }

        timestamp: list[str] = w["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

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
        split: Literal["all", "healthy", "pre", "t2d"] = "all",
        **kwargs,
    ) -> tuple["CGMacrosDataset", "CGMacrosDataset"]:
        cgm_dir = Path(data_root) / "CGMacros"
        classes = _load_subject_classes(cgm_dir) if split != "all" else {}

        subjects = sorted(
            folder.name.split("-")[-1]
            for folder in cgm_dir.iterdir()
            if folder.is_dir() and folder.name.startswith("CGMacros-")
            and (folder / f"{folder.name}.csv").exists()
            and (split == "all" or classes.get(folder.name.split("-")[-1]) == split)
        )

        rng = random.Random(seed)
        rng.shuffle(subjects)
        n_test = max(1, round(len(subjects) * test_ratio))
        test_subjects  = subjects[:n_test]
        train_subjects = subjects[n_test:]

        return (
            cls(data_root, split=split, subjects=train_subjects, **kwargs),
            cls(data_root, split=split, subjects=test_subjects,  **kwargs),
        )
