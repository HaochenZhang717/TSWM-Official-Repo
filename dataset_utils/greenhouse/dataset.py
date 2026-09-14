from __future__ import annotations

from pathlib import Path
from typing import Literal
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_CHANNELS: list[tuple[str, float, float]] = [
    ("Tair",   5.0,   35.0),
    ("Rhair",  20.0,  100.0),
    ("CO2air", 350.0, 1800.0),
    ("PARin",  0.0,   600.0),
]

ACTION_CHANNELS: list[tuple[str, float, float]] = [
    ("Tpipe",      0.0, 90.0),
    ("VentLee",    0.0, 100.0),
    ("VentWind",   0.0, 100.0),
    ("AssimLight", 0.0, 100.0),
    ("EnScr",      0.0, 100.0),
    ("BlckScr",    0.0, 100.0),
    ("CO2dosing",  0.0, 1.0),
]

EXOG_CHANNELS: list[tuple[str, float, float]] = [
    ("Tout",      -5.0,  20.0),
    ("Rhout",     20.0,  100.0),
    ("Iglob",     0.0,   700.0),
    ("Windsp",    0.0,   30.0),
    ("Winddir",   0.0,   360.0),
    ("Rain",      0.0,   1.0),
    ("PARout",    0.0,   1400.0),
    ("AbsHumOut", 0.0,   12.0),
]

_MAX_GAP_MINUTES = 30.0
_INTERVAL_MINUTES = 5


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    target = name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    return None


def _read_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.iloc[1:].reset_index(drop=True)
    return df


def _load_team_df(team_dir: Path, weather_path: Path) -> pd.DataFrame:
    clim = _read_sheet(team_dir / "GreenhouseClimate.xlsx")
    wth = _read_sheet(weather_path)

    clim["Date"] = pd.to_datetime(clim["Date"], errors="coerce")
    wth["Date"] = pd.to_datetime(wth["Date"], errors="coerce")
    clim = clim.dropna(subset=["Date"])
    wth = wth.dropna(subset=["Date"])

    out = pd.DataFrame({"Date": clim["Date"].to_numpy()})

    for name, _lo, _hi in TARGET_CHANNELS:
        col = _find_col(clim, name)
        out[name] = pd.to_numeric(clim[col], errors="coerce").to_numpy() if col else np.nan
    for name, _lo, _hi in ACTION_CHANNELS:
        if name == "CO2dosing":
            col = _find_col(clim, "CO2reg")
            reg = pd.to_numeric(clim[col], errors="coerce").to_numpy() if col else np.nan
            out["CO2dosing"] = 2.0 - reg
        else:
            col = _find_col(clim, name)
            out[name] = pd.to_numeric(clim[col], errors="coerce").to_numpy() if col else np.nan

    wmap = {"Date": wth["Date"]}
    for name, _lo, _hi in EXOG_CHANNELS:
        col = _find_col(wth, name)
        wmap[name] = pd.to_numeric(wth[col], errors="coerce") if col else np.nan
    wdf = pd.DataFrame(wmap)

    out = out.merge(wdf, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    return out


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


def _split_sessions(df: pd.DataFrame) -> list[tuple[int, int]]:
    if len(df) == 0:
        return []
    gaps = df["Date"].diff().dt.total_seconds().div(60).fillna(0)
    bp = [0] + list((gaps > _MAX_GAP_MINUTES).to_numpy().nonzero()[0]) + [len(df)]
    return [(s, e) for s, e in zip(bp, bp[1:]) if e - s > 0]


def _load_crop_meta(team_dir: Path) -> dict:
    try:
        raw = pd.read_excel(team_dir / "GreenhouseCrop.xlsx", header=1)
        raw = raw.iloc[1:]
        def colmean(key):
            for c in raw.columns:
                if key.lower() in str(c).lower():
                    v = pd.to_numeric(raw[c], errors="coerce")
                    return float(v.mean()) if v.notna().any() else None
            return None
        return {
            "fresh_weight_mean": colmean("Fresh Weight"),
            "dry_weight_mean":   colmean("Dry Weight"),
            "height_mean":       colmean("Height"),
            "n_plants":          int(raw.shape[0]),
        }
    except Exception:
        return {}


class GreenhouseDataset(Dataset):

    def __init__(
        self,
        data_root: str | Path,
        teams: list[str] | None = None,
        seq_len: int = 288,
        stride: int = 36,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int | None = 6,
        cache: bool = True,
        target_channels: list[tuple[str, float, float]] = TARGET_CHANNELS,
        action_channels: list[tuple[str, float, float]] = ACTION_CHANNELS,
        exog_channels: list[tuple[str, float, float]] = EXOG_CHANNELS,
    ):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.max_nan_ratio = max_nan_ratio
        self.max_nan_gap = max_nan_gap
        self.target_channels = target_channels
        self.action_channels = action_channels
        self.exog_channels = exog_channels

        self._tgt_names = [c[0] for c in target_channels]
        self._act_names = [c[0] for c in action_channels]
        self._exo_names = [c[0] for c in exog_channels]

        weather_path = self.data_root / "Weather.xlsx"
        if teams is None:
            teams = sorted(
                p.name for p in self.data_root.iterdir()
                if p.is_dir() and (p / "GreenhouseClimate.xlsx").exists()
            )
        self.teams = teams

        self._team_df: dict[str, pd.DataFrame] = {}
        self._team_meta: dict[str, dict] = {}
        self._windows: list[tuple[str, int]] = []

        cache_dir = self.data_root / "_cache"
        for team in teams:
            team_dir = self.data_root / team
            cache_fp = cache_dir / f"{team}.parquet"
            if cache and cache_fp.exists():
                df = pd.read_parquet(cache_fp)
            else:
                df = _load_team_df(team_dir, weather_path)
                if cache:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_fp)

            self._team_df[team] = df
            self._team_meta[team] = {"team": team, **_load_crop_meta(team_dir)}

            tgt = df[self._tgt_names].to_numpy(dtype="float32")
            exo_all = df[self._exo_names].to_numpy(dtype="float32")
            for s, e in _split_sessions(df):
                exo_valid = ~np.isnan(exo_all[s:e]).all(axis=0)
                for start in range(s, e - seq_len + 1, stride):
                    chk = np.concatenate(
                        [tgt[start:start + seq_len],
                         exo_all[start:start + seq_len][:, exo_valid]], axis=1)
                    if _window_continuous_ok(chk, max_nan_ratio, max_nan_gap):
                        self._windows.append((team, start))

    def __len__(self) -> int:
        return len(self._windows)

    @staticmethod
    def _norm(arr: np.ndarray, chans: list[tuple[str, float, float]],
              lo_out: float, hi_out: float) -> np.ndarray:
        out = np.empty_like(arr, dtype="float32")
        for i, (_n, lo, hi) in enumerate(chans):
            col = np.clip(arr[:, i], lo, hi)
            out[:, i] = (col - lo) / (hi - lo) * (hi_out - lo_out) + lo_out
        return out

    def __getitem__(self, idx: int) -> dict:
        team, start = self._windows[idx]
        df = self._team_df[team]
        w = df.iloc[start:start + self.seq_len]

        tgt = w[self._tgt_names].to_numpy(dtype="float32")
        act = w[self._act_names].to_numpy(dtype="float32")
        exo = w[self._exo_names].to_numpy(dtype="float32")

        tgt = _interp_nan_columns(tgt)
        exo = _interp_nan_columns(exo)
        target_ts = torch.from_numpy(self._norm(tgt, self.target_channels, -1.0, 1.0))
        act = np.nan_to_num(act, nan=0.0)
        action_ts = torch.from_numpy(self._norm(act, self.action_channels, 0.0, 1.0))
        exog_ts = torch.from_numpy(self._norm(exo, self.exog_channels, -1.0, 1.0))

        timestamp = w["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        return {
            "target_ts": target_ts,
            "continuous_action_ts": action_ts,
            "continuous_action_names": list(self._act_names),
            "exog_ts":   exog_ts,
            "timestamp": timestamp,
            "meta": self._team_meta[team],
        }

    @classmethod
    def subject_split(
        cls,
        data_root: str | Path,
        test_ratio: float = 0.34,
        seed: int = 42,
        **kwargs,
    ) -> tuple["GreenhouseDataset", "GreenhouseDataset"]:
        data_root = Path(data_root)
        teams = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and (p / "GreenhouseClimate.xlsx").exists()
        )
        rng = random.Random(seed)
        rng.shuffle(teams)
        n_test = max(1, round(len(teams) * test_ratio))
        test_teams, train_teams = teams[:n_test], teams[n_test:]
        return (
            cls(data_root, teams=train_teams, **kwargs),
            cls(data_root, teams=test_teams, **kwargs),
        )


def collate_fn(batch: list[dict]) -> dict:
    return {
        "target_ts": torch.stack([b["target_ts"] for b in batch]),
        "continuous_action_ts": torch.stack([b["continuous_action_ts"] for b in batch]),
        "continuous_action_names": batch[0]["continuous_action_names"],
        "exog_ts":   torch.stack([b["exog_ts"] for b in batch]),
        "timestamp": [b["timestamp"] for b in batch],
        "meta":      [b["meta"] for b in batch],
    }
