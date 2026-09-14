from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_CHANNELS = (
    "s_hc1_supply_temperature",
    "p_hc1_return_temperature",
    "p_net_return_temperature",
    "p_net_meter_heat_power",
    "p_net_meter_flow",
)
TARGET_NAMES = ("s_hc1_supply_temp", "p_hc1_return_temp", "p_net_return_temp",
                "heat_power", "flow")

CONT_ACTION_CHANNELS = (
    "p_hc1_control_valve_position_setpoint",
    "s_hc1_supply_temperature_setpoint",
)
CONT_ACTION_NAMES = ("valve_pos", "supply_setpoint")

CATEG_ACTION_CHANNELS = (
    "s_hc1_heating_pump_status_setpoint",
    "s_hc1_control_unit_mode",
    "s_dhw_3-way_valve_status",
)
CATEG_ACTION_NAMES = ("pump_status", "control_mode", "dhw_3way")
CATEG_N_CLASSES = (2, 3, 2)
CATEG_CARDINALITIES = tuple(n + 1 for n in CATEG_N_CLASSES)

ALL_ACTION_CHANNELS = CONT_ACTION_CHANNELS + CATEG_ACTION_CHANNELS

CATEG_TEXT_MAPS = {
    "s_hc1_heating_pump_status_setpoint": {"aus": 0.0, "ein": 1.0},
    "s_dhw_3-way_valve_status":           {"aus": 0.0, "ein": 1.0},
    "s_hc1_control_unit_mode":            {"standby": 0.0, "nacht": 1.0, "tag": 2.0},
}
OPTIONAL_CHANNELS = {"s_hc1_control_unit_mode", "s_dhw_3-way_valve_status", "outdoor_temperature"}

EXO_CHANNELS = ("outdoor_temperature", "p_net_supply_temperature")
EXO_NAMES = ("outdoor_temp", "p_net_supply")


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


def _find_col(df: pd.DataFrame, name: str):
    for col in df.columns:
        if name.lower() == col.strip().lower():
            return col
    for col in df.columns:
        if name.lower() in col.strip().lower():
            return col
    return None


def _encode_categorical(series: pd.Series, mapping: dict) -> np.ndarray:
    out = series.astype(str).str.strip().str.lower().map(mapping)
    return pd.to_numeric(out, errors="coerce").to_numpy(dtype=np.float32)


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


class PreDistSubstationDataset(Dataset):

    _FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        data_root,
        split: str = "all",
        subjects=None,
        seq_len: int = 144,
        stride: int = 36,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int | None = 6,
        drop_first_days: float = 30.0,
        max_gap_min: float = 30.0,
        target_ranges=None,
        action_ranges=None,
        exo_ranges=None,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.max_nan_ratio = max_nan_ratio
        self.max_nan_gap = max_nan_gap
        self.drop_first_days = drop_first_days
        self.max_gap_min = max_gap_min
        self.split = split
        self.n_target = len(TARGET_CHANNELS)
        self.n_cont_action = len(CONT_ACTION_CHANNELS)
        self.n_categ_action = len(CATEG_ACTION_CHANNELS)
        self.n_exo = len(EXO_CHANNELS)

        op_dir = self._resolve_paths(Path(data_root))

        subject_set = set(subjects) if subjects is not None else None

        self.sessions = []
        self.windows = []

        for path in sorted(op_dir.glob("substation_*.csv")):
            sid = path.stem.split("_", 1)[1]
            if subject_set is not None and sid not in subject_set:
                continue
            try:
                df = self._load_station_df(path)
            except Exception as e:
                print(f"[warn] skipping {path.name}: {e}")
                continue
            if len(df) < 2:
                continue

            ranges = self._entity_ranges(df, target_ranges, action_ranges, exo_ranges)

            for sess in self._build_sessions(df, sid, ranges):
                s_idx = len(self.sessions)
                self.sessions.append(sess)
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

    @staticmethod
    def _resolve_paths(root: Path):
        if (root / "operational_data").is_dir():
            return root / "operational_data"
        if root.name == "operational_data":
            return root
        raise FileNotFoundError(f"operational_data not found under {root}")

    def _load_station_df(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, sep=";", low_memory=False)
        date_col = _find_col(raw, "timestamp") or raw.columns[0]
        out = pd.DataFrame()
        out["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
        out = out.dropna(subset=["Date"]).reset_index(drop=True)
        raw = raw.loc[out.index] if len(out) == len(raw) else raw.iloc[: len(out)]

        for c in TARGET_CHANNELS + EXO_CHANNELS:
            col = _find_col(raw, c)
            out[c] = (pd.to_numeric(raw[col], errors="coerce").astype("float32").to_numpy()
                      if col is not None else np.full(len(out), np.nan, dtype=np.float32))

        for c in ALL_ACTION_CHANNELS:
            col = _find_col(raw, c)
            if c in CATEG_TEXT_MAPS:
                mapping = CATEG_TEXT_MAPS[c]
                out[c] = (_encode_categorical(raw[col], mapping)
                          if col is not None else np.full(len(out), np.nan, dtype=np.float32))
            else:
                out[c] = (pd.to_numeric(raw[col], errors="coerce").astype("float32").to_numpy()
                          if col is not None else np.full(len(out), np.nan, dtype=np.float32))

        out = out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

        if self.drop_first_days > 0 and len(out) > 0:
            t0 = out["Date"].iloc[0] + pd.Timedelta(days=self.drop_first_days)
            out = out[out["Date"] >= t0].reset_index(drop=True)
        return out

    def _entity_ranges(self, df, target_ranges, action_ranges, exo_ranges):
        tr = tuple(
            target_ranges[c] if target_ranges else _percentile_range(df[TARGET_CHANNELS[c]].to_numpy())
            for c in range(self.n_target)
        )
        ar = tuple(
            action_ranges[c] if action_ranges
            else _percentile_range(df[CONT_ACTION_CHANNELS[c]].to_numpy())
            for c in range(self.n_cont_action)
        )
        er = tuple(
            exo_ranges[c] if exo_ranges else _percentile_range(df[EXO_CHANNELS[c]].to_numpy())
            for c in range(self.n_exo)
        )
        return {"target": tr, "action": ar, "exo": er}

    def _build_sessions(self, df, sid, ranges) -> list:
        gap = df["Date"].diff().dt.total_seconds().div(60).fillna(0)
        breaks = [0] + list(np.where(gap.to_numpy() > self.max_gap_min)[0]) + [len(df)]

        target = np.stack([df[c].to_numpy(np.float32) for c in TARGET_CHANNELS], axis=1)
        action = np.stack([df[c].to_numpy(np.float32) for c in ALL_ACTION_CHANNELS], axis=1)
        exo = np.stack([df[c].to_numpy(np.float32) for c in EXO_CHANNELS], axis=1)
        ts_str = df["Date"].dt.strftime(self._FMT).to_numpy()

        sessions = []
        for s, e in zip(breaks, breaks[1:]):
            if e - s < 2:
                continue
            sl = slice(s, e)
            sessions.append({
                "sid": sid,
                "target": target[sl].copy(),
                "action": action[sl].copy(),
                "exo": exo[sl].copy(),
                "ts_str": list(ts_str[sl]),
                "ranges": ranges,
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
        rg = sess["ranges"]

        act = sess["action"][start:end]
        target_ts = torch.from_numpy(
            self._norm(_interp_nan_columns(sess["target"][start:end]), rg["target"]))
        continuous_action_ts = torch.from_numpy(
            self._norm(_interp_nan_columns(act[:, :self.n_cont_action]), rg["action"]))
        categorical_action_ts = torch.from_numpy(
            _encode_cat_ts(act[:, self.n_cont_action:], CATEG_N_CLASSES))
        exogenous_ts = torch.from_numpy(
            self._norm(_interp_nan_columns(sess["exo"][start:end]), rg["exo"]))
        timestamp = list(sess["ts_str"][start:end])

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
    def list_subjects(cls, data_root) -> list[str]:
        op_dir = cls._resolve_paths(Path(data_root))
        return sorted(
            (p.stem.split("_", 1)[1] for p in op_dir.glob("substation_*.csv")),
            key=lambda s: int(s) if s.isdigit() else s,
        )

    @classmethod
    def subject_split(cls, data_root, test_ratio: float = 0.2, seed: int = 42, **kwargs):
        import random

        subjects = cls.list_subjects(data_root)
        rng = random.Random(seed)
        rng.shuffle(subjects)
        n_test = max(1, round(len(subjects) * test_ratio))
        test_subjects = subjects[:n_test]
        train_subjects = subjects[n_test:]
        kwargs.pop("subjects", None)
        return (
            cls(data_root, subjects=train_subjects, **kwargs),
            cls(data_root, subjects=test_subjects, **kwargs),
        )


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
