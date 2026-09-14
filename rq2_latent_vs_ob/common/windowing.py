from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TIME_FEAT = 4


def to_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


@dataclass
class Schema:

    n_target: int
    target_key: str = "target_ts"
    n_continuous: int = 0
    continuous_key: str | None = None
    cat_mode: str = "none"
    n_cat_channels: int = 0
    cardinalities: list[int] = field(default_factory=list)
    n_exog: int = 0
    exog_key: str | None = None

    @property
    def has_continuous(self) -> bool:
        return self.n_continuous > 0

    @property
    def has_exog(self) -> bool:
        return self.n_exog > 0

    def summary(self) -> str:
        return (
            f"target={self.n_target}, continuous={self.n_continuous}, "
            f"categorical[{self.cat_mode}]={self.cardinalities}, exog={self.n_exog}"
        )


def discover_schema(sample: dict[str, Any], combo_to_id: dict | None = None) -> Schema:
    target = to_numpy(sample["target_ts"])
    n_target = 1 if target.ndim == 1 else target.shape[-1]

    n_continuous, continuous_key = 0, None
    if "continuous_action_ts" in sample:
        continuous_key = "continuous_action_ts"
        n_continuous = to_numpy(sample[continuous_key]).shape[-1]

    cat_mode, n_cat_channels, cardinalities = "none", 0, []
    if "categorical_action_ts" in sample:
        cat_mode = "dense"
        cardinalities = list(sample["categorical_cardinalities"])
        n_cat_channels = len(cardinalities)
    elif "action_id" in sample:
        if combo_to_id is None:
            raise ValueError("combo_to_id is required for sparse-action datasets")
        cat_mode = "sparse"
        n_cat_channels = 1
        cardinalities = [len(combo_to_id) + 1]

    n_exog, exog_key = 0, None
    for key in ("exogenous_ts", "exog_ts"):
        if key in sample:
            exog_key = key
            n_exog = to_numpy(sample[key]).shape[-1]
            break

    return Schema(
        n_target=n_target,
        n_continuous=n_continuous,
        continuous_key=continuous_key,
        cat_mode=cat_mode,
        n_cat_channels=n_cat_channels,
        cardinalities=cardinalities,
        n_exog=n_exog,
        exog_key=exog_key,
    )


def _sparse_to_dense_index(
    action_dict: dict[int, list[int]], length: int, combo_to_id: dict[tuple[int, ...], int]
) -> np.ndarray:
    dense = np.zeros((length, 1), dtype=np.int64)
    for step, ids in action_dict.items():
        step = int(step)
        if step >= length:
            continue
        combo = tuple(sorted(int(a) for a in ids))
        dense[step, 0] = combo_to_id.get(combo, -1) + 1
    return dense


def _time_marks(timestamps: list[str], length: int) -> np.ndarray:
    marks = np.zeros((length, TIME_FEAT), dtype=np.float32)
    if not timestamps:
        return marks
    dt = pd.to_datetime(pd.Series(timestamps[:length]), errors="coerce")
    n = min(length, len(dt))
    if n == 0:
        return marks
    marks[:n, 0] = (dt.dt.month.to_numpy()[:n] - 1) / 11.0 - 0.5
    marks[:n, 1] = (dt.dt.day.to_numpy()[:n] - 1) / 30.0 - 0.5
    marks[:n, 2] = dt.dt.weekday.to_numpy()[:n] / 6.0 - 0.5
    marks[:n, 3] = dt.dt.hour.to_numpy()[:n] / 23.0 - 0.5
    return np.nan_to_num(marks).astype(np.float32)


def unified_window(
    sample: dict[str, Any],
    context_length: int,
    horizon: int,
    schema: Schema,
    combo_to_id: dict | None = None,
) -> dict[str, np.ndarray]:
    end = context_length + horizon
    L = context_length

    target = to_numpy(sample[schema.target_key])
    if target.ndim == 1:
        target = target[:, None]
    target = target[:end]
    if len(target) < end:
        raise ValueError(f"sample shorter than context+horizon: {len(target)} < {end}")

    if schema.has_continuous:
        continuous = to_numpy(sample[schema.continuous_key])[:end]
    else:
        continuous = np.zeros((end, 0), dtype=np.float32)

    if schema.cat_mode == "dense":
        categorical = to_numpy(sample["categorical_action_ts"], dtype=np.int64)[:end]
    elif schema.cat_mode == "sparse":
        categorical = _sparse_to_dense_index(sample["action_id"], end, combo_to_id)
    else:
        categorical = np.zeros((end, 0), dtype=np.int64)

    if schema.has_exog:
        exog = to_numpy(sample[schema.exog_key])[:end]
    else:
        exog = np.zeros((end, 0), dtype=np.float32)

    marks = _time_marks(sample.get("timestamp", []), end)

    return {
        "target_history": target[:L],
        "target_future": target[L:end],
        "continuous_history": continuous[:L],
        "continuous_future": continuous[L:end],
        "categorical_history": categorical[:L],
        "categorical_future": categorical[L:end],
        "exog_history": exog[:L],
        "exog_future": exog[L:end],
        "mark_history": marks[:L],
        "mark_future": marks[L:end],
    }
