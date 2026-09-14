from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RQ2_ROOT = Path(__file__).resolve().parents[1]
if str(_RQ2_ROOT) not in sys.path:
    sys.path.insert(0, str(_RQ2_ROOT))

from ts_benchmark.baselines.timexer.timexer import TimeXer
from ts_benchmark.baselines.duet.duet import DUET
from ts_benchmark.baselines.crosslinear.crosslinear import CrossLinear
from ts_benchmark.baselines.amplifier.amplifier import Amplifier
from ts_benchmark.baselines.timekan.timekan import TimeKAN
from ts_benchmark.baselines.time_series_library import (
    PatchTST,
    TiDE,
)
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import (
    TransformerAdapter,
)
from ts_benchmark.baselines.utils import MLP

INTERNAL_EXOG = {"TimeXer", "TiDE"}
PURE_SERIES = {"DUET", "PatchTST", "TimeKAN", "CrossLinear", "Amplifier"}
ALL_MODELS = [
    "TimeXer", "TiDE", "DUET", "PatchTST", "TimeKAN", "CrossLinear", "Amplifier",
]


def is_pure_series(name: str) -> bool:
    return name in PURE_SERIES


def _construct(name: str, seq_len: int, horizon: int, cov_width: int, mlp_hidden_dims: int):
    common = dict(seq_len=seq_len, horizon=horizon, norm=True)
    if name == "TimeXer":
        return TimeXer(**common)
    if name == "TiDE":
        return TransformerAdapter("TiDE", TiDE, **common, use_future_exog=1, covariate_dim=cov_width)
    if name == "DUET":
        return DUET(**common, fusion_method="mlp", mlp_hidden_dims=mlp_hidden_dims)
    if name == "CrossLinear":
        return CrossLinear(**common, alpha=1.0, beta=1.0, fusion_method="mlp", mlp_hidden_dims=mlp_hidden_dims)
    if name == "Amplifier":
        return Amplifier(**common, fusion_method="mlp", mlp_hidden_dims=mlp_hidden_dims)
    if name == "TimeKAN":
        return TimeKAN(**common, fusion_method="mlp", mlp_hidden_dims=mlp_hidden_dims)
    if name == "PatchTST":
        return TransformerAdapter("PatchTST", PatchTST, **common, fusion_method="mlp", mlp_hidden_dims=mlp_hidden_dims)
    raise ValueError(f"Unknown model: {name!r}")


def configure_adapter(name, schema, cov_width, seq_len, horizon, *, mlp_hidden_dims=256):
    n_target = schema.n_target
    input_dim = n_target + cov_width
    adapter = _construct(name, seq_len, horizon, cov_width, mlp_hidden_dims)

    idx = pd.date_range("2020-01-01", periods=seq_len + horizon, freq="h")
    df = pd.DataFrame(np.zeros((len(idx), input_dim)), index=idx)
    if input_dim == 1:
        adapter.single_forecasting_hyper_param_tune(df)
    else:
        adapter.multi_forecasting_hyper_param_tune(df)

    adapter.config.series_dim = n_target
    adapter.config.input_dim = input_dim
    adapter.config.output_dim = n_target

    if name == "TiDE":
        adapter.config.covariate_dim = cov_width

    adapter._init_criterion()
    adapter.model = adapter._init_model()

    if name in PURE_SERIES and cov_width > 0:
        adapter.config.mlp_hidden_dims = mlp_hidden_dims
        adapter.CovariateFusion = MLP(adapter.config)
    else:
        adapter.CovariateFusion = None

    adapter.model.train()
    return adapter
