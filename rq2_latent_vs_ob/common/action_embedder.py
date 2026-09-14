from __future__ import annotations

import torch
import torch.nn as nn


def default_emb_dim(cardinality: int) -> int:
    return min(16, max(2, cardinality))


class ActionEmbedder(nn.Module):
    def __init__(self, cardinalities: list[int], n_continuous: int, n_exog: int, emb_dim_fn=default_emb_dim):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.n_continuous = n_continuous
        self.n_exog = n_exog
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim_fn(card)) for card in self.cardinalities]
        )
        self.emb_width = sum(emb_dim_fn(card) for card in self.cardinalities)
        self.cov_width = n_continuous + self.emb_width + n_exog

    def embed_categorical(self, categorical: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0 or categorical.shape[-1] == 0:
            return categorical.new_zeros((*categorical.shape[:-1], 0), dtype=torch.float32)
        parts = [emb(categorical[..., i].long()) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)

    def build_covariate(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        exog: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embed_categorical(categorical)
        parts = [t for t in (continuous.float(), embedded, exog.float()) if t.shape[-1] > 0]
        if not parts:
            return continuous.new_zeros((*continuous.shape[:-1], 0), dtype=torch.float32)
        return torch.cat(parts, dim=-1)
