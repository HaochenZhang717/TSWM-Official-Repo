from __future__ import annotations

import io
import gzip
import urllib.request
from pathlib import Path
from typing import Literal
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


INTERVAL_SEC = 2
_SYNTHETIC_EPOCH = pd.Timestamp("2000-01-01 00:00:00")


TARGET_CHANNELS: list[tuple[str, float, float]] = [
    ("Solar8000/ART_MBP",    20.0, 200.0),
    ("Solar8000/HR",         30.0, 200.0),
    ("Solar8000/PLETH_SPO2", 70.0, 100.0),
    ("BIS/BIS",               0.0, 100.0),
]

ACTION_TS_CHANNELS: list[tuple[str, float, str]] = [
    ("Orchestra/PPF20_RATE",  80.0,  "propofol"),
    ("Orchestra/RFTN20_RATE", 150.0, "remifentanil"),
]

_LEGACY_CACHE_ACTION_COUNT = 7

_MAX_GAP_SEC = 120.0


_API = "https://api.vitaldb.net"


def _fetch_csv(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return pd.read_csv(io.BytesIO(raw))


def load_listings(cache_dir: str | Path, download: bool = True
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trks_p, cases_p = cache_dir / "trks.parquet", cache_dir / "cases.parquet"

    if trks_p.exists() and cases_p.exists():
        return pd.read_parquet(trks_p), pd.read_parquet(cases_p)
    if not download:
        raise FileNotFoundError(f"listings missing and download=False: {trks_p} / {cases_p}")

    trks = _fetch_csv(f"{_API}/trks")
    cases = _fetch_csv(f"{_API}/cases")
    trks.to_parquet(trks_p)
    cases.to_parquet(cases_p)
    return trks, cases


def select_caseids(
    trks: pd.DataFrame,
    target_tracks: list[str],
    require_all_targets: bool = True,
) -> list[int]:
    sets = [set(trks.loc[trks["tname"] == t, "caseid"]) for t in target_tracks]
    if require_all_targets:
        keep = set.intersection(*sets) if sets else set()
    else:
        keep = set.union(*sets) if sets else set()
    return sorted(int(c) for c in keep)


def _load_case_array(
    caseid: int,
    target_tracks: list[str],
    action_tracks: list[str],
    interval: int,
    cache_dir: Path,
    download: bool,
) -> np.ndarray | None:
    n_keep = len(target_tracks) + len(action_tracks)
    cache_root = cache_dir / "cases_cache"
    key = f"case{caseid}_int{interval}_t{len(target_tracks)}_a{len(action_tracks)}.npz"
    fp = cache_root / key
    if fp.exists():
        return np.load(fp)["arr"]

    legacy_fp = (cache_root /
                 f"case{caseid}_int{interval}_t{len(target_tracks)}_a{_LEGACY_CACHE_ACTION_COUNT}.npz")
    if legacy_fp.exists():
        return np.load(legacy_fp)["arr"][:, :n_keep]

    if not download:
        return None

    import vitaldb
    tnames = target_tracks + action_tracks
    arr = vitaldb.load_case(caseid, tnames, interval=interval)
    if arr is None or arr.size == 0:
        return None
    arr = arr.astype("float32")

    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fp, arr=arr)
    return arr


def _split_sessions(target_arr: np.ndarray, interval: int) -> list[tuple[int, int]]:
    n = len(target_arr)
    if n == 0:
        return []
    all_nan = np.isnan(target_arr).all(axis=1)
    gap_len = int(round(_MAX_GAP_SEC / interval))

    segments: list[tuple[int, int]] = []
    seg_start = 0
    i = 0
    while i < n:
        if all_nan[i]:
            j = i
            while j < n and all_nan[j]:
                j += 1
            if j - i >= gap_len:
                if i > seg_start:
                    segments.append((seg_start, i))
                seg_start = j
            i = j
        else:
            i += 1
    if seg_start < n:
        segments.append((seg_start, n))
    return segments


def _window_continuous_ok(arr: np.ndarray, max_nan_ratio: float, max_nan_gap: int | None) -> bool:
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
    out = np.asarray(a, dtype="float32").copy()
    idx = np.arange(out.shape[0])
    for c in range(out.shape[1]):
        col = out[:, c]
        good = ~np.isnan(col)
        if good.all():
            continue
        out[:, c] = 0.0 if not good.any() else np.interp(idx, idx[good], col[good])
    return out


class VitalDBDataset(Dataset):

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["all", "general", "thoracic", "urology", "gynecology"] = "all",
        caseids: list[int] | None = None,
        seq_len: int = 900,
        stride: int = 150,
        interval: int = INTERVAL_SEC,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int | None = 6,
        require_any_action: bool = True,
        max_cases: int | None = None,
        download: bool = True,
        target_channels: list[tuple[str, float, float]] = TARGET_CHANNELS,
        action_ts_channels: list[tuple[str, float, str]] = ACTION_TS_CHANNELS,
    ):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.interval = interval
        self.max_nan_ratio = max_nan_ratio
        self.max_nan_gap = max_nan_gap
        self.require_any_action = require_any_action
        self.download = download
        self.target_channels = target_channels
        self.action_ts_channels = action_ts_channels

        self._target_tracks = [c[0] for c in target_channels]
        self._action_tracks = [c[0] for c in action_ts_channels]
        self._action_ts_names = [c[2] for c in action_ts_channels]
        self._n_tgt = len(target_channels)
        self._n_ats = len(action_ts_channels)
        self._array_cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._cache_max = 32

        trks, cases = load_listings(self.data_root, download=download)
        self._cases_df = cases.set_index("caseid")

        if caseids is None:
            caseids = select_caseids(trks, self._target_tracks, require_all_targets=True)
        if split != "all":
            dept = self._cases_df["department"]
            keep_dept = {
                "general": "General surgery", "thoracic": "Thoracic surgery",
                "urology": "Urology", "gynecology": "Gynecology",
            }[split]
            caseids = [c for c in caseids if dept.get(c) == keep_dept]
        if max_cases is not None:
            caseids = caseids[:max_cases]
        self.caseids = caseids

        self._windows: list[tuple[int, int]] = []
        for cid in caseids:
            arr = _load_case_array(cid, self._target_tracks, self._action_tracks,
                                   interval, self.data_root, download)
            if arr is None or len(arr) < seq_len:
                continue
            if self.require_any_action and self._n_ats > 0 and \
                    np.isnan(arr[:, self._n_tgt:self._n_tgt + self._n_ats]).all():
                continue
            tgt = arr[:, :self._n_tgt]
            for s, e in _split_sessions(tgt, interval):
                for start in range(s, e - seq_len + 1, stride):
                    win = tgt[start:start + seq_len]
                    if _window_continuous_ok(win, max_nan_ratio, self.max_nan_gap):
                        self._windows.append((cid, start))

    def _case_array(self, caseid: int) -> np.ndarray:
        cached = self._array_cache.get(caseid)
        if cached is not None:
            return cached
        arr = _load_case_array(caseid, self._target_tracks, self._action_tracks,
                               self.interval, self.data_root, self.download)
        assert arr is not None, f"case {caseid} failed to load"
        self._array_cache[caseid] = arr
        self._cache_order.append(caseid)
        if len(self._cache_order) > self._cache_max:
            evict = self._cache_order.pop(0)
            self._array_cache.pop(evict, None)
        return arr

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        caseid, start = self._windows[idx]
        arr = self._case_array(caseid)
        w = arr[start:start + self.seq_len]
        tgt = w[:, :self._n_tgt]

        tgt_norm = np.empty_like(tgt, dtype="float32")
        for i, (_name, lo, hi) in enumerate(self.target_channels):
            col = np.clip(tgt[:, i], lo, hi)
            tgt_norm[:, i] = 2.0 * (col - lo) / (hi - lo) - 1.0
        tgt_norm = _interp_nan_columns(tgt_norm)
        target_ts = torch.from_numpy(tgt_norm)

        ats = np.nan_to_num(w[:, self._n_tgt:self._n_tgt + self._n_ats], nan=0.0)
        ats_norm = np.empty_like(ats, dtype="float32")
        for i, (_name, scale, _label) in enumerate(self.action_ts_channels):
            ats_norm[:, i] = np.clip(np.log1p(np.maximum(ats[:, i], 0.0)) /
                                     np.log1p(scale), 0.0, 1.0)
        action_ts = torch.from_numpy(ats_norm)

        base = _SYNTHETIC_EPOCH + pd.Timedelta(seconds=int(start * self.interval))
        timestamp = [
            (base + pd.Timedelta(seconds=int(k * self.interval)))
            .strftime("%Y-%m-%d %H:%M:%S")
            for k in range(self.seq_len)
        ]

        return {
            "target_ts":               target_ts,
            "continuous_action_ts":     action_ts,
            "continuous_action_names":  self._action_ts_names,
            "timestamp":               timestamp,
            "meta":                    self._case_meta(caseid),
        }

    def _case_meta(self, caseid: int) -> dict:
        row = self._cases_df.loc[caseid]
        keys = ["subjectid", "age", "sex", "bmi", "asa", "department", "optype",
                "ane_type", "icu_days", "death_inhosp", "intraop_ebl"]
        meta = {"caseid": int(caseid)}
        for k in keys:
            if k in row.index:
                v = row[k]
                meta[k] = (None if pd.isna(v) else
                           (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v))
        return meta

    @classmethod
    def case_split(
        cls,
        data_root: str | Path,
        test_ratio: float = 0.2,
        seed: int = 42,
        split: str = "all",
        download: bool = True,
        max_cases: int | None = None,
        **kwargs,
    ) -> tuple["VitalDBDataset", "VitalDBDataset"]:
        data_root = Path(data_root)
        trks, cases = load_listings(data_root, download=download)
        cases_idx = cases.set_index("caseid")

        target_tracks = [c[0] for c in kwargs.get("target_channels", TARGET_CHANNELS)]
        caseids = select_caseids(trks, target_tracks, require_all_targets=True)
        if split != "all":
            keep_dept = {
                "general": "General surgery", "thoracic": "Thoracic surgery",
                "urology": "Urology", "gynecology": "Gynecology",
            }[split]
            caseids = [c for c in caseids if cases_idx["department"].get(c) == keep_dept]
        if max_cases is not None:
            caseids = caseids[:max_cases]

        subj_of = {c: int(cases_idx.loc[c, "subjectid"]) for c in caseids}
        subjects = sorted(set(subj_of.values()))
        rng = random.Random(seed)
        rng.shuffle(subjects)
        n_test = max(1, round(len(subjects) * test_ratio))
        test_subj = set(subjects[:n_test])

        train_ids = [c for c in caseids if subj_of[c] not in test_subj]
        test_ids = [c for c in caseids if subj_of[c] in test_subj]

        return (
            cls(data_root, split=split, caseids=train_ids, download=download, **kwargs),
            cls(data_root, split=split, caseids=test_ids, download=download, **kwargs),
        )


def collate_fn(batch: list[dict]) -> dict:
    return {
        "target_ts":               torch.stack([b["target_ts"] for b in batch]),
        "continuous_action_ts":    torch.stack([b["continuous_action_ts"] for b in batch]),
        "continuous_action_names": batch[0]["continuous_action_names"],
        "timestamp":               [b["timestamp"] for b in batch],
        "meta":                    [b["meta"] for b in batch],
    }
