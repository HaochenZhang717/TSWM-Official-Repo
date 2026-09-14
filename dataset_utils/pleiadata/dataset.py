import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SP_LO, SP_HI = 10, 32
_SP_BASE = 300
_ONOFF_BASE = 360
_MODE_BASE = 370
_MODE_NAMES = {0: "off", 1: "heat", 2: "cool", 3: "dry", 4: "fan", 5: "auto"}


def _level_setpoint_name(t):
    return "setpoint_%dC" % t


def _build_action_names():
    names = {}
    for t in range(SP_LO, SP_HI + 1):
        names[_SP_BASE + (t - SP_LO)] = _level_setpoint_name(t)
    names[_ONOFF_BASE + 0] = "hvac_off"
    names[_ONOFF_BASE + 1] = "hvac_on"
    for code, nm in _MODE_NAMES.items():
        names[_MODE_BASE + code] = "mode_%s" % nm
    return names


ACTION_NAMES = _build_action_names()
NAME_TO_ID = {v: k for k, v in ACTION_NAMES.items()}


def encode_setpoint(temp_c):
    t = int(round(float(temp_c)))
    t = max(SP_LO, min(SP_HI, t))
    return _SP_BASE + (t - SP_LO)


def encode_onoff(v4):
    return _ONOFF_BASE + (1 if float(v4) >= 0.5 else 0)


def encode_mode(code):
    code = int(code)
    if (_MODE_BASE + code) not in ACTION_NAMES:
        ACTION_NAMES[_MODE_BASE + code] = "mode_%d" % code
    return _MODE_BASE + code


def action_category(action_id):
    if _SP_BASE <= action_id < _ONOFF_BASE:
        return "setpoint"
    if _ONOFF_BASE <= action_id < _MODE_BASE:
        return "onoff"
    return "mode"


TARGET_CHANNELS = ("V2", "dif_cons")
TARGET_NAMES = ("indoor_temp", "step_energy")
DEFAULT_TARGET_RANGES = ((10.0, 35.0),
                         (0.0, 3.1))

CONT_ACTION_NAMES  = ("setpoint_norm",)
CATEG_ACTION_NAMES = ("onoff", "mode")
CATEG_N_CLASSES    = (2, len(_MODE_NAMES))
CATEG_CARDINALITIES = tuple(n + 1 for n in CATEG_N_CLASSES)
DEFAULT_SETPOINT_RANGE = (21.0, 30.0)

EXO_CHANNELS = ("tmed", "hrmed", "radmed", "vvmed", "dvmed", "prec", "dewpt", "dpv")
DEFAULT_EXO_RANGES = ((-5.0, 46.0),
                      (0.0, 100.0),
                      (0.0, 1000.0),
                      (0.0, 12.0),
                      (0.0, 360.0),
                      (0.0, 10.0),
                      (-10.0, 30.0),
                      (0.0, 8.0))


def _minmax(x, lo, hi):
    y = 2.0 * (x - lo) / (hi - lo) - 1.0
    return np.clip(y, -1.0, 1.0)


def _encode_cat_column(col, n_classes, base=0):
    out = np.full(col.shape[0], n_classes, dtype=np.int64)
    valid = ~np.isnan(col)
    if valid.any():
        k = np.rint(col[valid]).astype(np.int64) - base
        inb = (k >= 0) & (k < n_classes)
        vi = np.nonzero(valid)[0]
        out[vi[inb]] = k[inb]
    return out


def _encode_cat_ts(arr, n_classes_list, bases=None):
    C = arr.shape[1]
    if C == 0:
        return np.zeros((arr.shape[0], 0), dtype=np.int64)
    bases = bases if bases is not None else [0] * C
    return np.stack([_encode_cat_column(arr[:, c], n_classes_list[c], bases[c])
                     for c in range(C)], axis=1)


def _window_continuous_ok(arr, max_nan_ratio, max_nan_gap):
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


def _interp_nan_columns(a):
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


class PLEIADataHVACDataset(Dataset):

    _GROUND_TZ = "%Y-%m-%d %H:%M:%S"
    _MAX_GAP_MIN = 30

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 144,
        stride: int = 36,
        max_nan_ratio: float = 0.1,
        max_nan_gap: int = 6,
        target_ranges: tuple = DEFAULT_TARGET_RANGES,
        exo_ranges: tuple = DEFAULT_EXO_RANGES,
        setpoint_range: tuple = DEFAULT_SETPOINT_RANGE,
        split: str = "train",
        subjects: list = None,
        blocks: tuple = ("A", "B", "C"),
        cache_dir: str = None,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.stride = stride
        self.max_nan_ratio = max_nan_ratio
        self.max_nan_gap = max_nan_gap
        self.target_ranges = tuple(target_ranges)
        self.exo_ranges = tuple(exo_ranges)
        self.setpoint_range = tuple(setpoint_range)
        self.split = split
        self.test_ratio = test_ratio
        self.seed = seed
        self.blocks = tuple(blocks)
        self.n_target = len(TARGET_CHANNELS)
        self.n_exo = len(EXO_CHANNELS)
        self.n_cont_action = len(CONT_ACTION_NAMES)
        self.n_categ_action = len(CATEG_ACTION_NAMES)
        self.cache_dir = cache_dir or os.path.join(data_dir, "_cache_dataset_mc2")

        self._ensure_cache()
        all_subjects = self._list_cached_subjects()

        if subjects is not None:
            keep = set(subjects)
            self.subjects = [s for s in all_subjects if s in keep]
        else:
            self.subjects = self.subject_split(all_subjects, split, test_ratio=test_ratio, seed=seed)

        self.sessions = {}
        self.windows = []
        for subj in self.subjects:
            sess_list = self._load_subject_sessions(subj)
            self.sessions[subj] = sess_list
            for s_idx, sess in enumerate(sess_list):
                L = sess["target"].shape[0]
                if L < seq_len:
                    continue
                exo_valid = ~np.isnan(sess["exo"]).all(axis=0)
                for start in range(0, L - seq_len + 1, stride):
                    chk = np.concatenate(
                        [sess["target"][start:start + seq_len],
                         sess["exo"][start:start + seq_len][:, exo_valid]], axis=1)
                    if _window_continuous_ok(chk, max_nan_ratio, self.max_nan_gap):
                        self.windows.append((subj, s_idx, start))

    def _ensure_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        done = os.path.join(self.cache_dir, "_DONE")
        if os.path.exists(done):
            return
        for blk in self.blocks:
            path = os.path.join(self.data_dir, "processed_data", "data-room%s-10T.csv" % blk)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=";")
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            v5cols = sorted([c for c in df.columns if c.startswith("V5_")],
                            key=lambda c: int(c.split("_")[1]))
            mode_codes = np.array([int(c.split("_")[1]) for c in v5cols])
            for (sid, hid), g in df.groupby(["IDsensor", "IDhvac"]):
                g = g.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
                room = g["room"].iloc[0]
                subj = "%s/room%s/s%s/h%s" % (blk, room, sid, hid)
                sessions = self._build_sessions(g, v5cols, mode_codes)
                if sessions:
                    cf = os.path.join(self.cache_dir, subj.replace("/", "__") + ".pkl")
                    with open(cf, "wb") as f:
                        pickle.dump({"subject": subj, "block": blk, "sessions": sessions}, f)
            del df
        with open(done, "w") as f:
            f.write("ok\n")

    def _build_sessions(self, g, v5cols, mode_codes):
        dates = g["Date"]
        gap = dates.diff().dt.total_seconds().div(60).fillna(0)
        breaks = [0] + list(np.where(gap.values > self._MAX_GAP_MIN)[0]) + [len(g)]

        target = np.stack([pd.to_numeric(g[c], errors="coerce").values
                           for c in TARGET_CHANNELS], axis=1).astype(np.float32)
        exo = np.stack([pd.to_numeric(g[c], errors="coerce").values
                        for c in EXO_CHANNELS], axis=1).astype(np.float32)
        sp = g["V12"].values
        onoff = g["V4"].values
        mode = mode_codes[g[v5cols].values.argmax(1)] if v5cols else np.zeros(len(g), int)
        ts = dates.dt.strftime(self._GROUND_TZ).values
        sp_ff = pd.to_numeric(g["V12"], errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)
        onoff_ff = np.nan_to_num(pd.to_numeric(g["V4"], errors="coerce").values.astype(np.float32), nan=0.0)
        mode_ff = mode.astype(np.float32)

        sessions = []
        for s, e in zip(breaks, breaks[1:]):
            if e - s < 2:
                continue
            sl = slice(s, e)
            spw, onw, mdw = sp[sl], onoff[sl], mode[sl]
            actions = {}

            def add(i, aid):
                actions.setdefault(int(i), [])
                if aid not in actions[i]:
                    actions[i].append(aid)

            n = e - s
            for i in range(n):
                if i == 0:
                    if not np.isnan(spw[i]):
                        add(i, encode_setpoint(spw[i]))
                    if not np.isnan(onw[i]):
                        add(i, encode_onoff(onw[i]))
                    add(i, encode_mode(mdw[i]))
                else:
                    if not np.isnan(spw[i]) and not np.isnan(spw[i - 1]) and \
                       encode_setpoint(spw[i]) != encode_setpoint(spw[i - 1]):
                        add(i, encode_setpoint(spw[i]))
                    if not np.isnan(onw[i]) and onw[i] != onw[i - 1]:
                        add(i, encode_onoff(onw[i]))
                    if mdw[i] != mdw[i - 1]:
                        add(i, encode_mode(mdw[i]))

            sessions.append({
                "target": target[sl].copy(),
                "exo": exo[sl].copy(),
                "timestamp": list(ts[sl]),
                "actions": actions,
                "setpoint": sp_ff[sl].copy(),
                "onoff": onoff_ff[sl].copy(),
                "mode": mode_ff[sl].copy(),
            })
        return sessions

    def _list_cached_subjects(self):
        out = []
        for f in glob.glob(os.path.join(self.cache_dir, "*.pkl")):
            out.append(os.path.splitext(os.path.basename(f))[0].replace("__", "/"))
        return sorted(out)

    def _load_subject_sessions(self, subj):
        cf = os.path.join(self.cache_dir, subj.replace("/", "__") + ".pkl")
        with open(cf, "rb") as f:
            return pickle.load(f)["sessions"]

    def _norm_target(self, arr):
        out = np.empty_like(arr, dtype=np.float32)
        for c, (lo, hi) in enumerate(self.target_ranges):
            out[:, c] = _minmax(arr[:, c], lo, hi)
        return out

    def _norm_exo(self, arr):
        out = np.empty_like(arr, dtype=np.float32)
        for c, (lo, hi) in enumerate(self.exo_ranges):
            out[:, c] = _minmax(arr[:, c], lo, hi)
        return out

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        subj, s_idx, start = self.windows[idx]
        sess = self.sessions[subj][s_idx]
        end = start + self.seq_len

        target_ts = torch.from_numpy(
            self._norm_target(_interp_nan_columns(sess["target"][start:end])))
        exogenous_ts = torch.from_numpy(
            self._norm_exo(_interp_nan_columns(sess["exo"][start:end])))
        timestamp = list(sess["timestamp"][start:end])


        sp_n = _minmax(sess["setpoint"][start:end], *self.setpoint_range)
        continuous_action_ts = torch.from_numpy(sp_n.reshape(-1, 1).astype(np.float32))
        categ_raw = np.stack([sess["onoff"][start:end], sess["mode"][start:end]], axis=1)
        categorical_action_ts = torch.from_numpy(_encode_cat_ts(categ_raw, CATEG_N_CLASSES))

        return {
            "target_ts": target_ts,
            "timestamp": timestamp,
            "continuous_action_ts": continuous_action_ts,
            "continuous_action_names": list(CONT_ACTION_NAMES),
            "categorical_action_ts": categorical_action_ts,
            "categorical_action_names": list(CATEG_ACTION_NAMES),
            "categorical_cardinalities": list(CATEG_CARDINALITIES),
            "exogenous_ts": exogenous_ts,
            "exogenous_names": list(EXO_CHANNELS),
        }

    @property
    def target_names(self):
        return list(TARGET_NAMES)

    @property
    def exogenous_names(self):
        return list(EXO_CHANNELS)

    @staticmethod
    def subject_split(all_subjects, split, test_ratio=0.2, seed=42):
        rng = np.random.RandomState(seed)
        subs = sorted(all_subjects)
        rng.shuffle(subs)
        n = len(subs)
        n_test = max(1, int(round(n * test_ratio)))
        n_tr = n - n_test
        if split == "train":
            return sorted(subs[:n_tr])
        if split in ("val", "test"):
            return sorted(subs[n_tr:])
        return sorted(subs)


def collate_fn(batch):
    return {
        "target_ts": torch.stack([b["target_ts"] for b in batch]),
        "exogenous_ts": torch.stack([b["exogenous_ts"] for b in batch]),
        "continuous_action_ts": torch.stack([b["continuous_action_ts"] for b in batch]),
        "categorical_action_ts": torch.stack([b["categorical_action_ts"] for b in batch]),
        "timestamp": [b["timestamp"] for b in batch],
        "continuous_action_names": batch[0]["continuous_action_names"],
        "categorical_action_names": batch[0]["categorical_action_names"],
        "categorical_cardinalities": batch[0]["categorical_cardinalities"],
        "exogenous_names": batch[0]["exogenous_names"],
    }
