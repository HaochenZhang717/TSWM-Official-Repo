import sys
from pathlib import Path

_RQ2_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _RQ2_ROOT / "common"
_CODE_ROOT = _RQ2_ROOT.parent
for _p in (_CODE_ROOT, _RQ2_ROOT, _COMMON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from latent_space.autoencoder import (
    AE_MODES,
    RevIN,
    StepwiseAutoEncoder,
    build_autoencoder,
    load_autoencoder,
)
from latent_space.embedder import LatentActionEmbedder

__all__ = [
    "AE_MODES",
    "RevIN",
    "StepwiseAutoEncoder",
    "build_autoencoder",
    "load_autoencoder",
    "LatentActionEmbedder",
]
