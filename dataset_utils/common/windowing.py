from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def build_action_combo_mapping(dataset: Any) -> dict[tuple[int, ...], int]:
    combo_counts: dict[tuple[int, ...], int] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        action_dict = sample.get("action_id")
        if not action_dict:
            continue
        for ids in action_dict.values():
            combo = tuple(sorted(int(action_id) for action_id in ids))
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

    ordered = sorted(combo_counts.items(), key=lambda item: (-item[1], item[0]))
    return {combo: index for index, (combo, _count) in enumerate(ordered)}


def sparse_action_dict_to_dense_series(
    action_dict: dict[int, list[int]],
    length: int,
    combo_to_id: dict[tuple[int, ...], int],
) -> np.ndarray:
    dense = np.full((length, 1), -1.0, dtype=np.float32)
    for step, ids in action_dict.items():
        if step >= length:
            continue
        combo = tuple(sorted(int(action_id) for action_id in ids))
        dense[step, 0] = float(combo_to_id.get(combo, -1.0))
    return dense


def split_window(
    sample: dict[str, Any],
    context_length: int,
    horizon: int,
    *,
    combo_to_id: dict[tuple[int, ...], int] | None = None,
) -> dict[str, Any]:
    end = context_length + horizon
    target = to_numpy(sample["target_ts"])[:end]
    if target.ndim == 1:
        target = target[:, None]

    action = None
    action_names = None
    if "continuous_action_ts" in sample or "categorical_action_ts" in sample:
        action_parts: list[np.ndarray] = []
        name_parts: list[str] = []
        if "continuous_action_ts" in sample:
            action_parts.append(to_numpy(sample["continuous_action_ts"])[:end])
            name_parts.extend(sample.get("continuous_action_names") or [])
        if "categorical_action_ts" in sample:
            action_parts.append(to_numpy(sample["categorical_action_ts"])[:end])
            name_parts.extend(sample.get("categorical_action_names") or [])
        action = np.concatenate(action_parts, axis=1)
        action_names = name_parts or None
    elif "action_id" in sample:
        if combo_to_id is None:
            raise ValueError("combo_to_id is required for sparse action_id datasets")
        action = sparse_action_dict_to_dense_series(sample["action_id"], end, combo_to_id)
        action_names = ["action_combo_id"]

    if "exogenous_ts" in sample:
        exogenous = to_numpy(sample["exogenous_ts"])[:end]
        exogenous_names = sample.get("exogenous_names")
    elif "exog_ts" in sample:
        exogenous = to_numpy(sample["exog_ts"])[:end]
        exogenous_names = sample.get("exog_names")
    else:
        exogenous = None
        exogenous_names = None
    if len(target) < end:
        raise ValueError(f"Sample is shorter than context+horizon: {len(target)} < {end}")

    return {
        "target_history": target[:context_length],
        "target_future": target[context_length:end],
        "action_history": None if action is None else action[:context_length],
        "future_action_ts": None if action is None else action[context_length:end],
        "exogenous_history": None if exogenous is None else exogenous[:context_length],
        "future_exogenous_ts": None if exogenous is None else exogenous[context_length:end],
        "action_names": action_names,
        "exogenous_names": exogenous_names,
    }


def target_channel_names(dataset: Any, n_targets: int) -> list[str]:
    raw = getattr(dataset, "_tgt_names", None)
    if raw is None:
        raw = getattr(dataset, "target_names", None)
    if raw is None:
        channels = getattr(dataset, "target_channels", None)
        if channels is not None:
            raw = [channel[0] if isinstance(channel, (tuple, list)) else channel for channel in channels]
    if not raw:
        return [f"target_{i}" for i in range(n_targets)]
    return [str(name).split("/")[-1] for name in raw]
