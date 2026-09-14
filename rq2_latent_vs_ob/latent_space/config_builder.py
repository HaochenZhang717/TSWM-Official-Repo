from __future__ import annotations

import numpy as np
import pandas as pd

from ts_benchmark.baselines.utils import MLP

from observational_space.config_builder import (
    ALL_MODELS,
    INTERNAL_EXOG,
    PURE_SERIES,
    _construct,
    is_pure_series,
)


def configure_latent_adapter(name, schema, cov_width, seq_len, horizon, n_latent_target, *,
                             mlp_hidden_dims=256):
    input_dim = n_latent_target + cov_width
    adapter = _construct(name, seq_len, horizon, cov_width, mlp_hidden_dims)

    idx = pd.date_range("2020-01-01", periods=seq_len + horizon, freq="h")
    df = pd.DataFrame(np.zeros((len(idx), input_dim)), index=idx)
    if input_dim == 1:
        adapter.single_forecasting_hyper_param_tune(df)
    else:
        adapter.multi_forecasting_hyper_param_tune(df)

    adapter.config.series_dim = n_latent_target
    adapter.config.input_dim = input_dim
    adapter.config.output_dim = n_latent_target

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
