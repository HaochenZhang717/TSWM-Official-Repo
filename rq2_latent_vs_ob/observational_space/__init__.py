import sys
from pathlib import Path

_RQ2_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _RQ2_ROOT / "common"
_CODE_ROOT = _RQ2_ROOT.parent
for _p in (_CODE_ROOT, _RQ2_ROOT, _COMMON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.action_embedder import ActionEmbedder, default_emb_dim
from common.datasets import build_train_val, prepare
from common.windowing import Schema, discover_schema, unified_window
from observational_space.config_builder import (
    ALL_MODELS,
    INTERNAL_EXOG,
    PURE_SERIES,
    configure_adapter,
    is_pure_series,
)
from observational_space.train import TrainConfig, Trainer, run_benchmark, train_one

__all__ = [
    "ActionEmbedder",
    "default_emb_dim",
    "ALL_MODELS",
    "INTERNAL_EXOG",
    "PURE_SERIES",
    "configure_adapter",
    "is_pure_series",
    "build_train_val",
    "prepare",
    "TrainConfig",
    "Trainer",
    "train_one",
    "run_benchmark",
    "Schema",
    "discover_schema",
    "unified_window",
]
