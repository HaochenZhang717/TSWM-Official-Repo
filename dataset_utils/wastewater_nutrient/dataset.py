from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_CHANNELS = ("T1_O2", "T1_NH4", "T1_PO4")
TARGET_NAMES    = ("do_tank1", "nh4", "po4")

CONT_ACTION_CHANNELS = ("METAL_Q", "IN_METAL_Q", "MAX_CF")
CONT_ACTION_NAMES    = ("metal_dosing", "inlet_metal_dosing", "max_chem_factor")

CATEG_ACTION_CHANNELS = ("PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET")
CATEG_ACTION_NAMES    = ("inlet_phase", "outlet_phase")
CATEG_N_CLASSES = (2, 2)
CATEG_BASE      = (1, 1)
CATEG_CARDINALITIES = tuple(n + 1 for n in CATEG_N_CLASSES)

ALL_ACTION_CHANNELS = CONT_ACTION_CHANNELS + CATEG_ACTION_CHANNELS
PHASE_CHANNELS = CATEG_ACTION_CHANNELS

EXO_CHANNELS = ("IN_Q", "TEMPERATURE")
EXO_NAMES    = ("inflow", "temperature")


def _minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-9:
        hi = lo + 1.0
    y = 2.0 * (x - lo) / (hi - lo) - 1.0
    return np.clip(y, -1.0, 1.0)


def _percentile_range(x: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> tuple:
    v = x[~np.isnan(x)]
    if v.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(v, lo_pct))
    hi = float(np.percentile(v, hi_pct))
    if hi - lo < 1e-9:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
    return (lo, hi)


def _find_col(df: pd.DataFrame, *substrings: str):
    for sub in substrings:
        for col in df.columns:
            if sub.lower() == col.strip().lower():
                return col
    for sub in substrings:
        for col in df.columns:
            if sub.lower() in col.strip().lower():
                return col
    return None


def _encode_cat_column(col: np.ndarray, n_classes: int, base: int = 0) -> np.ndarray:
    out = np.full(col.shape[0], n_classes, dtype=np.int64)
    valid = ~np.isnan(col)
    if valid.any():
        k = np.rint(col[valid]).astype(np.int64) - base
        inb = (k >= 0) & (k < n_classes)
        vi = np.nonzero(valid)[0]
        out[vi[inb]] = k[inb]
    return out


def _encode_cat_ts(arr: np.ndarray, n_classes_list, bases=None) -> np.ndarray:
    C = arr.shape[1]
    if C == 0:
        return np.zeros((arr.shape[0], 0), dtype=np.int64)
    bases = bases if bases is not None else [0] * C
    return np.stack([_encode_cat_column(arr[:, c], n_classes_list[c], bases[c])
                     for c in range(C)], axis=1)


def _window_continuous_ok(arr: np.ndarray, max_nan_ratio: float, max_nan_gap) -> bool:
    if arr.shape[1] == 0:
        return True
    nan = np.isnan(arr)
    if float(nan.mean(axis=0).max()) > max_nan_ratio:
        return False
    if max_nan_gap is not None:
        for c in range(arr.shape[1]):
            m = nan[:, c]
            if m.any():
                d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
                runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
                if int(runs.max()) > max_nan_gap:
                    return False
    return True


def _interp_nan_columns(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32).copy()
    if out.shape[1] == 0:
        return out
    idx = np.arange(out.shape[0])
    for c in range(out.shape[1]):
        col = out[:, c]
        good = ~np.isnan(col)
        if good.all():
            continue
        out[:, c] = 0.0 if not good.any() else np.interp(idx, idx[good], col[good])
    return out


class WastewaterNutrientDataset(Dataset):

    _FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        data_path,
        split: str = "all",
        seq_len: int = 720,
        stride: int = 180,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int | None = 6,
        train_ratio: float = 0.8,
        target_ranges=None,
        action_ranges=None,
        exo_ranges=None,
        max_gap_min: float = 10.0,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.max_nan_ratio = max_nan_ratio
        self.max_nan_gap = max_nan_gap
        self.split = split
        self.train_ratio = train_ratio
        self.max_gap_min = max_gap_min
        self.n_target = len(TARGET_CHANNELS)
        self.n_cont_action = len(CONT_ACTION_CHANNELS)
        self.n_categ_action = len(CATEG_ACTION_CHANNELS)
        self.n_exo = len(EXO_CHANNELS)

        path = Path(data_path)
        if path.is_dir():
            cands = sorted(path.glob("*.csv"))
            if not cands:
                raise FileNotFoundError("no CSV in directory: %s" % path)
            path = cands[0]
        self.data_path = path

        df = self._load_df(path)

        n_all = len(df)
        cut = int(round(n_all * train_ratio))
        train_df = df.iloc[:cut]

        self.target_ranges = tuple(
            target_ranges[c] if target_ranges else _percentile_range(train_df[TARGET_CHANNELS[c]].to_numpy())
            for c in range(self.n_target)
        )
        self.action_ranges = tuple(
            action_ranges[c] if action_ranges
            else _percentile_range(train_df[CONT_ACTION_CHANNELS[c]].to_numpy())
            for c in range(self.n_cont_action)
        )
        self.exo_ranges = tuple(
            exo_ranges[c] if exo_ranges else _percentile_range(train_df[EXO_CHANNELS[c]].to_numpy())
            for c in range(self.n_exo)
        )

        if split == "train":
            sub = df.iloc[:cut]
        elif split == "test":
            sub = df.iloc[cut:]
        elif split == "all":
            sub = df
        else:
            raise ValueError("split must be 'train'/'test'/'all', got %r" % split)
        sub = sub.reset_index(drop=True)

        self.sessions = self._build_sessions(sub)

        self.windows = []
        for s_idx, sess in enumerate(self.sessions):
            L = sess["target"].shape[0]
            aux_full = np.concatenate(
                [sess["action"][:, :self.n_cont_action], sess["exo"]], axis=1)
            aux_valid = ~np.isnan(aux_full).all(axis=0)
            for start in range(0, L - seq_len + 1, stride):
                chk = np.concatenate(
                    [sess["target"][start:start + seq_len],
                     aux_full[start:start + seq_len][:, aux_valid]], axis=1)
                if _window_continuous_ok(chk, max_nan_ratio, self.max_nan_gap):
                    self.windows.append((s_idx, start))

    def _load_df(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        date_col = _find_col(raw, "date")
        out = pd.DataFrame()
        out["Date"] = pd.to_datetime(raw[date_col], utc=True)
        for cols in (TARGET_CHANNELS, ALL_ACTION_CHANNELS, EXO_CHANNELS):
            for c in cols:
                col = _find_col(raw, c)
                if col is None:
                    raise KeyError("missing required column %s in %s" % (c, path.name))
                series = pd.to_numeric(raw[col], errors="coerce").astype("float32")
                if c in PHASE_CHANNELS:
                    series = series.round().clip(1, 2)
                out[c] = series
        return out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    def _build_sessions(self, df: pd.DataFrame) -> list:
        if len(df) == 0:
            return []
        gap = df["Date"].diff().dt.total_seconds().div(60).fillna(0)
        breaks = [0] + list(np.where(gap.to_numpy() > self.max_gap_min)[0]) + [len(df)]

        target = np.stack([df[c].to_numpy() for c in TARGET_CHANNELS], axis=1)
        action = np.stack([df[c].to_numpy() for c in ALL_ACTION_CHANNELS], axis=1)
        exo    = np.stack([df[c].to_numpy() for c in EXO_CHANNELS], axis=1)
        ts = df["Date"].dt.strftime(self._FMT).to_numpy()

        sessions = []
        for s, e in zip(breaks, breaks[1:]):
            if e - s < 2:
                continue
            sl = slice(s, e)
            sessions.append({
                "target": target[sl].copy(),
                "action": action[sl].copy(),
                "exo": exo[sl].copy(),
                "timestamp": list(ts[sl]),
            })
        return sessions

    def _norm(self, arr: np.ndarray, ranges) -> np.ndarray:
        out = np.empty_like(arr, dtype=np.float32)
        for c, (lo, hi) in enumerate(ranges):
            out[:, c] = _minmax(arr[:, c], lo, hi)
        return out

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        s_idx, start = self.windows[idx]
        sess = self.sessions[s_idx]
        end = start + self.seq_len

        act = sess["action"][start:end]
        target_ts    = torch.from_numpy(
            self._norm(_interp_nan_columns(sess["target"][start:end]), self.target_ranges))
        continuous_action_ts = torch.from_numpy(
            self._norm(_interp_nan_columns(act[:, :self.n_cont_action]), self.action_ranges))
        categorical_action_ts = torch.from_numpy(
            _encode_cat_ts(act[:, self.n_cont_action:], CATEG_N_CLASSES, CATEG_BASE))
        exogenous_ts = torch.from_numpy(
            self._norm(_interp_nan_columns(sess["exo"][start:end]), self.exo_ranges))
        timestamp    = list(sess["timestamp"][start:end])

        return {
            "target_ts": target_ts,
            "timestamp": timestamp,
            "continuous_action_ts": continuous_action_ts,
            "continuous_action_names": list(CONT_ACTION_NAMES),
            "categorical_action_ts": categorical_action_ts,
            "categorical_action_names": list(CATEG_ACTION_NAMES),
            "categorical_cardinalities": list(CATEG_CARDINALITIES),
            "exogenous_ts": exogenous_ts,
            "exogenous_names": list(EXO_NAMES),
        }

    @property
    def target_names(self):
        return list(TARGET_NAMES)

    @property
    def exogenous_names(self):
        return list(EXO_NAMES)

    @classmethod
    def subject_split(cls, data_path, train_ratio: float = 0.8, **kwargs):
        kwargs.pop("split", None)
        train = cls(data_path, split="train", train_ratio=train_ratio, **kwargs)
        test  = cls(data_path, split="test",  train_ratio=train_ratio, **kwargs)
        return train, test


def collate_fn(batch):
    return {
        "target_ts": torch.stack([b["target_ts"] for b in batch]),
        "continuous_action_ts": torch.stack([b["continuous_action_ts"] for b in batch]),
        "categorical_action_ts": torch.stack([b["categorical_action_ts"] for b in batch]),
        "exogenous_ts": torch.stack([b["exogenous_ts"] for b in batch]),
        "timestamp": [b["timestamp"] for b in batch],
        "continuous_action_names": batch[0]["continuous_action_names"],
        "categorical_action_names": batch[0]["categorical_action_names"],
        "categorical_cardinalities": batch[0]["categorical_cardinalities"],
        "exogenous_names": batch[0]["exogenous_names"],
    }
