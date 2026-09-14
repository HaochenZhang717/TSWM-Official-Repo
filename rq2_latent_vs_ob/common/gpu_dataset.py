from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from common.datasets import SPARSE_ACTION_DATASETS, build_train_val
from common.windowing import Schema, discover_schema, unified_window
from dataset_utils.common.windowing import build_action_combo_mapping

FORMAT_VERSION = 1

FIELD_DTYPES = {
    "target_history": torch.float32, "target_future": torch.float32,
    "continuous_history": torch.float32, "continuous_future": torch.float32,
    "categorical_history": torch.long, "categorical_future": torch.long,
    "exog_history": torch.float32, "exog_future": torch.float32,
    "mark_history": torch.float32, "mark_future": torch.float32,
}


@dataclass
class GPUWindowBank:

    tensors: dict[str, torch.Tensor]
    n: int

    def __len__(self) -> int:
        return self.n


@dataclass
class GPUResidentData:
    train: GPUWindowBank
    val: GPUWindowBank
    schema: Schema
    combo_to_id: dict | None


def _default_cache_dir() -> Path:
    env = os.environ.get("WINDOW_CACHE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / ".gpu_window_cache"


def _cache_key(name, root, context_length, horizon, stride, val_ratio, seed, options):
    payload = {
        "name": name, "root": str(root), "context_length": context_length,
        "horizon": horizon, "stride": stride, "val_ratio": val_ratio, "seed": seed,
        "options": sorted((options or {}).items()),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return payload, digest


def _materialize(raw_ds, schema: Schema, context_length: int, horizon: int,
                  combo_to_id: dict | None) -> dict[str, "torch.Tensor"]:
    import numpy as np

    n = len(raw_ds)
    if n == 0:
        return {key: torch.empty((0,), dtype=dtype) for key, dtype in FIELD_DTYPES.items()}

    first = unified_window(raw_ds[0], context_length, horizon, schema, combo_to_id)
    buffers = {key: torch.empty((n, *arr.shape), dtype=FIELD_DTYPES[key]) for key, arr in first.items()}
    for key, arr in first.items():
        buffers[key][0] = torch.from_numpy(np.ascontiguousarray(arr))
    for idx in range(1, n):
        w = unified_window(raw_ds[idx], context_length, horizon, schema, combo_to_id)
        for key, arr in w.items():
            buffers[key][idx] = torch.from_numpy(np.ascontiguousarray(arr))
    return buffers


def prepare_gpu(
    name: str,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    root: str | Path | None = None,
    options: dict[str, Any] | None = None,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
) -> GPUResidentData:
    from dataset_utils.common.splits import DEFAULT_ROOTS

    seq_len = context_length + horizon
    stride = stride if stride is not None else seq_len
    resolved_root = Path(root) if root is not None else DEFAULT_ROOTS[name]
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_payload, digest = _cache_key(
        name, resolved_root, context_length, horizon, stride, val_ratio, seed, options)
    cache_path = cache_dir / f"{name}__v{FORMAT_VERSION}__{digest}.pt"

    blob = None
    if cache_path.exists():
        loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
        if loaded.get("format_version") == FORMAT_VERSION and loaded.get("key_payload") == key_payload:
            blob = loaded
        else:
            print(f"  [gpu_dataset] stale cache at {cache_path}, rebuilding")

    if blob is None:
        train_raw, val_raw = build_train_val(
            name, seq_len=seq_len, stride=stride, val_ratio=val_ratio, seed=seed,
            root=resolved_root, options=options,
        )
        combo_to_id = None
        if name in SPARSE_ACTION_DATASETS:
            combo_to_id = build_action_combo_mapping(train_raw)
        schema = discover_schema(train_raw[0], combo_to_id)

        print(f"  [gpu_dataset] {name}: materializing {len(train_raw)} train / "
              f"{len(val_raw)} val windows (one-time cost, cached at {cache_path})")
        train_tensors = _materialize(train_raw, schema, context_length, horizon, combo_to_id)
        val_tensors = _materialize(val_raw, schema, context_length, horizon, combo_to_id)

        blob = {
            "format_version": FORMAT_VERSION, "key_payload": key_payload,
            "schema": asdict(schema), "combo_to_id": combo_to_id,
            "train": train_tensors, "val": val_tensors,
            "train_n": len(train_raw), "val_n": len(val_raw),
        }
        tmp_path = cache_path.with_suffix(".pt.tmp")
        torch.save(blob, tmp_path)
        os.replace(tmp_path, cache_path)

    schema = Schema(**blob["schema"])
    combo_to_id = blob["combo_to_id"]
    train_bank = GPUWindowBank(
        tensors={k: v.to(device) for k, v in blob["train"].items()}, n=blob["train_n"])
    val_bank = GPUWindowBank(
        tensors={k: v.to(device) for k, v in blob["val"].items()}, n=blob["val_n"])
    return GPUResidentData(train=train_bank, val=val_bank, schema=schema, combo_to_id=combo_to_id)


class GPUEpochLoader:

    def __init__(self, bank: GPUWindowBank, *, batch_size: int, shuffle: bool, drop_last: bool):
        self.bank = bank
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._device = next(iter(bank.tensors.values())).device if bank.tensors else torch.device("cpu")

    def __len__(self) -> int:
        n = self.bank.n
        if self.drop_last:
            return n // self.batch_size
        return -(-n // self.batch_size)

    def __iter__(self):
        n = self.bank.n
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        idx = idx.to(self._device, non_blocking=True)
        usable = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, usable, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield {key: t.index_select(0, b) for key, t in self.bank.tensors.items()}
