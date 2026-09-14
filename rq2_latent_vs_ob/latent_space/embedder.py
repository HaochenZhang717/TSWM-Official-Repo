from __future__ import annotations

import torch
import torch.nn as nn

from common.action_embedder import default_emb_dim


class LatentActionEmbedder(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_continuous: int,
        n_exog: int,
        *,
        covariate_ae: nn.Module | None = None,
        emb_dim_fn=default_emb_dim,
    ):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.n_continuous = n_continuous
        self.n_exog = n_exog
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim_fn(card)) for card in self.cardinalities]
        )
        self.emb_width = sum(emb_dim_fn(card) for card in self.cardinalities)

        self.n_real_cov = n_continuous + n_exog
        self.covariate_ae = covariate_ae if self.n_real_cov > 0 else None

        proj_width = self.covariate_ae.d_model if self.covariate_ae is not None else self.n_real_cov
        self.cov_width = proj_width + self.emb_width

    def embed_categorical(self, categorical: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0 or categorical.shape[-1] == 0:
            return categorical.new_zeros((*categorical.shape[:-1], 0), dtype=torch.float32)
        parts = [emb(categorical[..., i].long()) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _cat_real(continuous: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        parts = [t.float() for t in (continuous, exog) if t.shape[-1] > 0]
        if not parts:
            return continuous.new_zeros((*continuous.shape[:-1], 0), dtype=torch.float32)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def build_covariate(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        exog: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embed_categorical(categorical)
        if self.covariate_ae is not None:
            proj = self.covariate_ae.encode(self._cat_real(continuous, exog))
            parts = [proj, embedded]
        else:
            parts = [continuous.float(), embedded, exog.float()]
        parts = [t for t in parts if t.shape[-1] > 0]
        if not parts:
            return continuous.new_zeros((*continuous.shape[:-1], 0), dtype=torch.float32)
        return torch.cat(parts, dim=-1)
