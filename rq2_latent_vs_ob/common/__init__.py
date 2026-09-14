import sys
from pathlib import Path

_RQ2_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _RQ2_ROOT / "common"
_CODE_ROOT = _RQ2_ROOT.parent
for _p in (_CODE_ROOT, _RQ2_ROOT, _COMMON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.action_embedder import ActionEmbedder, default_emb_dim
from common.datasets import build_train_val, prepare, collate
from common.windowing import Schema, discover_schema, unified_window

__all__ = [
    "ActionEmbedder",
    "default_emb_dim",
    "build_train_val",
    "prepare",
    "collate",
    "Schema",
    "discover_schema",
    "unified_window",
]
