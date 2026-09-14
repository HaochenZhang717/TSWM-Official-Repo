from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from dataset_utils.common.splits import DEFAULT_ROOTS
from dataset_utils.common.windowing import build_action_combo_mapping

from common.windowing import Schema, discover_schema, unified_window

SPARSE_ACTION_DATASETS = {"cgmacros", "shanghai_diabetes", "mimic_cardio"}


def build_train_val(
    name: str,
    *,
    seq_len: int,
    stride: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    root: str | Path | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[Dataset, Dataset]:
    options = options or {}
    kw = dict(seq_len=seq_len, stride=stride)
    data_root = Path(root) if root is not None else DEFAULT_ROOTS[name]

    if name == "greenhouse":
        from dataset_utils.greenhouse.dataset import GreenhouseDataset

        return GreenhouseDataset.subject_split(data_root, test_ratio=val_ratio, seed=seed, **kw)

    if name == "vitaldb":
        from dataset_utils.vital_db.dataset import VitalDBDataset

        return VitalDBDataset.case_split(
            data_root, test_ratio=val_ratio, seed=seed,
            split=options.get("split", "all"), download=options.get("download", False), **kw,
        )

    if name == "cgmacros":
        from dataset_utils.cgmacros.dataset import CGMacrosDataset

        return CGMacrosDataset.subject_split(
            data_root, test_ratio=val_ratio, seed=seed,
            split=options.get("split", "all"),
            resample_minutes=options.get("resample_minutes", None), **kw,
        )

    if name == "shanghai_diabetes":
        from dataset_utils.shanghai_diabetes.dataset import ShanghaiDiabetesDataset

        return ShanghaiDiabetesDataset.subject_split(
            data_root, test_ratio=val_ratio, seed=seed, split=options.get("split", "all"), **kw,
        )

    if name == "pleiadata":
        from dataset_utils.pleiadata.dataset import PLEIADataHVACDataset

        common = dict(test_ratio=val_ratio, seed=seed, **kw)
        train = PLEIADataHVACDataset(str(data_root), split="train", **common)
        val = PLEIADataHVACDataset(str(data_root), split="val", **common)
        return train, val

    if name == "predist":
        from dataset_utils.predist.dataset import PreDistSubstationDataset

        return PreDistSubstationDataset.subject_split(data_root, test_ratio=val_ratio, seed=seed, **kw)

    if name == "wastewater_nutrient":
        from dataset_utils.wastewater_nutrient.dataset import WastewaterNutrientDataset

        return WastewaterNutrientDataset.subject_split(
            str(data_root), train_ratio=1.0 - val_ratio, **kw,
        )

    if name == "mimic_cardio":
        from dataset_utils.mimic_cardio.dataset import MimicCardioDataset

        return MimicCardioDataset.subject_split(
            data_root, test_ratio=val_ratio, seed=seed,
            grid_minutes=options.get("grid_minutes", 30),
            max_stays=options.get("max_stays", None), **kw,
        )

    raise ValueError(f"Unknown dataset: {name!r}")


class UnifiedWindowDataset(Dataset):

    def __init__(self, base: Dataset, schema: Schema, context_length: int, horizon: int, combo_to_id=None):
        self.base = base
        self.schema = schema
        self.L = context_length
        self.H = horizon
        self.combo_to_id = combo_to_id

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w = unified_window(self.base[idx], self.L, self.H, self.schema, self.combo_to_id)
        out = {}
        for key, arr in w.items():
            dtype = torch.long if key.startswith("categorical") else torch.float32
            out[key] = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
        return out


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def prepare(
    name: str,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    root: str | Path | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[UnifiedWindowDataset, UnifiedWindowDataset, Schema]:
    seq_len = context_length + horizon
    stride = stride if stride is not None else seq_len
    train_raw, val_raw = build_train_val(
        name, seq_len=seq_len, stride=stride, val_ratio=val_ratio, seed=seed, root=root, options=options,
    )

    combo_to_id = None
    if name in SPARSE_ACTION_DATASETS:
        combo_to_id = build_action_combo_mapping(train_raw)

    schema = discover_schema(train_raw[0], combo_to_id)
    train = UnifiedWindowDataset(train_raw, schema, context_length, horizon, combo_to_id)
    val = UnifiedWindowDataset(val_raw, schema, context_length, horizon, combo_to_id)
    return train, val, schema
