from __future__ import annotations

import torch
import torch.nn as nn

FUSION_ARMS = ("none", "concat", "res", "res_zero", "film_zero", "xattn", "gate")

ALL_ARMS = FUSION_ARMS


def _zero_init(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _CtxEncoder(nn.Module):

    def __init__(self, n_latent: int, cov_width: int, d_h: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * n_latent + 2 * cov_width, d_h), nn.ReLU())

    def forward(self, z_hist: torch.Tensor, cov_hist: torch.Tensor) -> torch.Tensor:
        parts = [z_hist[:, -1], z_hist.mean(dim=1)]
        if cov_hist.shape[-1] > 0:
            parts += [cov_hist[:, -1], cov_hist.mean(dim=1)]
        return self.proj(torch.cat(parts, dim=-1))


class _StepwiseFusion(nn.Module):

    def __init__(self, n_latent: int, cov_width: int, horizon: int, d_h: int, out_dim: int,
                 *, zero_last: bool):
        super().__init__()
        self.ctx = _CtxEncoder(n_latent, cov_width, d_h)
        self.pos = nn.Parameter(torch.zeros(horizon, d_h // 4))
        nn.init.normal_(self.pos, std=0.02)
        last = nn.Linear(d_h, out_dim)
        if zero_last:
            _zero_init(last)
        self.head = nn.Sequential(
            nn.Linear(d_h + cov_width + d_h // 4, d_h), nn.ReLU(), last)

    def step_features(self, z_hist, cov_hist, cov_fut):
        B, H = z_hist.shape[0], self.pos.shape[0]
        ctx = self.ctx(z_hist, cov_hist)
        ctx = ctx.unsqueeze(1).expand(B, H, -1)
        pos = self.pos.unsqueeze(0).expand(B, -1, -1)
        feats = [ctx, pos]
        if cov_fut.shape[-1] > 0:
            feats.insert(1, cov_fut)
        return torch.cat(feats, dim=-1)

    def head_out(self, z_hist, cov_hist, cov_fut) -> torch.Tensor:
        return self.head(self.step_features(z_hist, cov_hist, cov_fut))


class ResidualFusion(_StepwiseFusion):

    def __init__(self, n_latent, cov_width, horizon, d_h=128, *, zero_last=False):
        super().__init__(n_latent, cov_width, horizon, d_h, n_latent, zero_last=zero_last)

    def forward(self, z_hist, cov_hist, cov_fut, z_pred):
        return z_pred + self.head_out(z_hist, cov_hist, cov_fut)


class FiLMFusion(_StepwiseFusion):

    def __init__(self, n_latent, cov_width, horizon, d_h=128):
        super().__init__(n_latent, cov_width, horizon, d_h, 2 * n_latent, zero_last=True)
        self.n_latent = n_latent

    def forward(self, z_hist, cov_hist, cov_fut, z_pred):
        gb = self.head_out(z_hist, cov_hist, cov_fut)
        gamma, beta = gb[..., : self.n_latent], gb[..., self.n_latent:]
        return (1.0 + gamma) * z_pred + beta


class GatedFusion(_StepwiseFusion):

    def __init__(self, n_latent, cov_width, horizon, d_h=128):
        super().__init__(n_latent, cov_width, horizon, d_h, 2 * n_latent, zero_last=False)
        self.head = nn.Sequential(
            nn.Linear(d_h + cov_width + d_h // 4 + n_latent, d_h), nn.ReLU(),
            nn.Linear(d_h, 2 * n_latent))
        self.n_latent = n_latent

    def forward(self, z_hist, cov_hist, cov_fut, z_pred):
        feats = torch.cat([self.step_features(z_hist, cov_hist, cov_fut), z_pred], dim=-1)
        ab = self.head(feats)
        a, b = ab[..., : self.n_latent], ab[..., self.n_latent:]
        return z_pred + a * torch.sigmoid(b)


class CrossAttnFusion(nn.Module):

    def __init__(self, n_latent, cov_width, horizon, context_length, d_h=128, n_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(n_latent, d_h)
        self.kv_proj = nn.Linear(cov_width, d_h)
        self.pos_q = nn.Parameter(torch.zeros(horizon, d_h))
        self.pos_kv = nn.Parameter(torch.zeros(context_length + horizon, d_h))
        nn.init.normal_(self.pos_q, std=0.02)
        nn.init.normal_(self.pos_kv, std=0.02)
        self.attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.out = nn.Linear(d_h, n_latent)

    def forward(self, z_hist, cov_hist, cov_fut, z_pred):
        if cov_fut.shape[-1] == 0:
            return z_pred
        q = self.q_proj(z_pred) + self.pos_q
        cov_seq = torch.cat([cov_hist, cov_fut], dim=1)
        kv = self.kv_proj(cov_seq) + self.pos_kv[: cov_seq.shape[1]]
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return z_pred + self.out(attn_out)


def build_fusion(arm: str, n_latent: int, cov_width: int, horizon: int,
                 context_length: int, d_h: int = 128) -> nn.Module | None:
    if arm not in FUSION_ARMS:
        raise ValueError(f"unknown fusion arm {arm!r}; choose from {ALL_ARMS}")
    if arm in ("none", "concat"):
        return None
    if cov_width == 0:
        raise ValueError(f"arm {arm!r} needs covariates but cov_width=0 for this dataset")
    if arm == "res":
        return ResidualFusion(n_latent, cov_width, horizon, d_h, zero_last=False)
    if arm == "res_zero":
        return ResidualFusion(n_latent, cov_width, horizon, d_h, zero_last=True)
    if arm == "film_zero":
        return FiLMFusion(n_latent, cov_width, horizon, d_h)
    if arm == "gate":
        return GatedFusion(n_latent, cov_width, horizon, d_h)
    if arm == "xattn":
        return CrossAttnFusion(n_latent, cov_width, horizon, context_length, d_h)
    raise AssertionError(arm)
