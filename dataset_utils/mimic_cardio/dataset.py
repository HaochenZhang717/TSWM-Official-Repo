from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from dataset_utils.mimic_cardio.build import (
    ACTION_CONT_COLS,
    BOL_COLS,
    BOLUS_CHANNELS,
    TARGET_CHANNELS,
    TARGET_RANGE,
    VENT_BASELINE,
    VENT_CHANNELS,
    VENT_RANGE,
)

TARGET_NAMES = tuple(TARGET_CHANNELS)
CONT_ACTION_NAMES = tuple(ACTION_CONT_COLS)

EVENT_NAMES: dict[int, str] = {i: f"{c}_push" for i, c in enumerate(BOLUS_CHANNELS)}

_FALLBACK_RANGE: dict[str, tuple[float, float]] = {
    **{c: (0.0, 1.0) for c in ACTION_CONT_COLS},
    **{c: (float(VENT_BASELINE[c]), float(VENT_RANGE[c][1])) for c in VENT_CHANNELS},
}


def _minmax_sym(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def _minmax_unit(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _load_ranges(root: Path) -> dict[str, tuple[float, float]]:
    ranges = dict(_FALLBACK_RANGE)
    summary = root / "build_summary.json"
    if summary.is_file():
        hi = json.loads(summary.read_text()).get("action_hi") or {}
        for c, v in hi.items():
            if c in ranges and float(v) > 0:
                ranges[c] = (0.0, float(v))
    return ranges


class MimicCardioDataset(Dataset):

    def __init__(
        self,
        data_root,
        grid_minutes: int = 30,
        subjects=None,
        seq_len: int = 272,
        stride: int = 80,
        max_stays: int | None = None,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.stride = stride
        root = self._resolve_root(Path(data_root), grid_minutes)
        self.action_range = _load_ranges(root)

        coh = pq.read_table(str(root / "cohort.parquet")).to_pandas()
        if subjects is not None:
            coh = coh[coh["subject_id"].isin(set(subjects))]
        coh = coh[coh["n_steps"] >= seq_len]
        if max_stays is not None and len(coh) > max_stays:
            rng = np.random.default_rng(seed)
            keep = rng.choice(coh["stay_id"].to_numpy(), size=max_stays, replace=False)
            coh = coh[coh["stay_id"].isin(set(keep.tolist()))]
        stay_ids = [int(s) for s in coh["stay_id"]]

        self.sessions: list[dict] = []
        self.windows: list[tuple[int, int]] = []
        if not stay_ids:
            return

        cols = (["stay_id", "step", "timestamp"]
                + list(TARGET_CHANNELS) + list(ACTION_CONT_COLS))
        grid = pq.read_table(str(root / "grid.parquet"), columns=cols,
                             filters=[("stay_id", "in", stay_ids)]).to_pandas()
        grid = grid.sort_values(["stay_id", "step"])
        self._bol_pos = [ACTION_CONT_COLS.index(c) for c in BOL_COLS]

        for sid, g in grid.groupby("stay_id", sort=False):
            target = np.stack([g[c].to_numpy(np.float32) for c in TARGET_CHANNELS], axis=1)
            action = np.stack([g[c].to_numpy(np.float32) for c in ACTION_CONT_COLS], axis=1)
            s_idx = len(self.sessions)
            self.sessions.append({
                "sid": int(sid),
                "target": target,
                "action": action,
                "ts_str": g["timestamp"].astype(str).tolist(),
            })
            L = target.shape[0]
            for start in range(0, L - seq_len + 1, stride):
                self.windows.append((s_idx, start))

    @staticmethod
    def _resolve_root(root: Path, grid_minutes: int) -> Path:
        if (root / "grid.parquet").is_file():
            return root
        sub = root / f"grid_{grid_minutes}min"
        if (sub / "grid.parquet").is_file():
            return sub
        raise FileNotFoundError(
            f"grid.parquet not found: tried {root} and {sub}. "
            f"Run `python -m dataset_utils.mimic_cardio.build` first.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        s_idx, start = self.windows[idx]
        sess = self.sessions[s_idx]
        end = start + self.seq_len

        tgt = sess["target"][start:end]
        out_t = np.empty_like(tgt)
        for c, name in enumerate(TARGET_CHANNELS):
            lo, hi = TARGET_RANGE[name]
            out_t[:, c] = _minmax_sym(tgt[:, c], lo, hi)

        act = sess["action"][start:end]
        out_a = np.empty_like(act)
        for c, name in enumerate(ACTION_CONT_COLS):
            lo, hi = self.action_range[name]
            out_a[:, c] = _minmax_unit(act[:, c], lo, hi)

        action_id: dict[int, list[int]] = {}
        action_type: dict[int, list[str]] = {}
        bol = act[:, self._bol_pos]
        steps, evs = np.nonzero(bol > 0)
        for s, e in zip(steps.tolist(), evs.tolist()):
            action_id.setdefault(s, []).append(e)
            action_type.setdefault(s, []).append(EVENT_NAMES[e])

        return {
            "target_ts": torch.from_numpy(out_t),
            "timestamp": list(sess["ts_str"][start:end]),
            "continuous_action_ts": torch.from_numpy(out_a),
            "continuous_action_names": list(CONT_ACTION_NAMES),
            "action_id": action_id,
            "action_type": action_type,
        }

    @property
    def target_names(self) -> list[str]:
        return list(TARGET_NAMES)

    @property
    def action_names(self) -> list[str]:
        return list(CONT_ACTION_NAMES)

    @classmethod
    def list_subjects(cls, data_root, grid_minutes: int = 30) -> list[int]:
        root = cls._resolve_root(Path(data_root), grid_minutes)
        coh = pq.read_table(str(root / "cohort.parquet"), columns=["subject_id"]).to_pandas()
        return sorted(int(s) for s in coh["subject_id"].unique())

    @classmethod
    def subject_split(cls, data_root, test_ratio: float = 0.2, seed: int = 42, **kwargs):
        grid_minutes = kwargs.get("grid_minutes", 30)
        subjects = cls.list_subjects(data_root, grid_minutes)
        rng = random.Random(seed)
        rng.shuffle(subjects)
        n_test = max(1, round(len(subjects) * test_ratio))
        test_subjects, train_subjects = subjects[:n_test], subjects[n_test:]
        kwargs.pop("subjects", None)
        max_stays = kwargs.pop("max_stays", None)
        tr_cap = int(max_stays * (1 - test_ratio)) if max_stays else None
        te_cap = max(1, int(max_stays * test_ratio)) if max_stays else None
        return (
            cls(data_root, subjects=train_subjects, max_stays=tr_cap, seed=seed, **kwargs),
            cls(data_root, subjects=test_subjects, max_stays=te_cap, seed=seed, **kwargs),
        )


def collate_fn(batch):
    return {
        "target_ts": torch.stack([b["target_ts"] for b in batch]),
        "continuous_action_ts": torch.stack([b["continuous_action_ts"] for b in batch]),
        "timestamp": [b["timestamp"] for b in batch],
        "action_id": [b["action_id"] for b in batch],
        "action_type": [b["action_type"] for b in batch],
        "continuous_action_names": batch[0]["continuous_action_names"],
    }
