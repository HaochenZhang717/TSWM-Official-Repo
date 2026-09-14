from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

JEPA_ARMS = ("jepa", "jepa0")
EMA_SCHEDULES = ("cosine", "const")


class EMATargetEncoder(nn.Module):

    def __init__(self, codec: nn.Module, *, momentum: float = 0.996,
                 schedule: str = "cosine", total_steps: int | None = None):
        super().__init__()
        if schedule not in EMA_SCHEDULES:
            raise ValueError(f"unknown ema schedule {schedule!r}; choose from {EMA_SCHEDULES}")
        self.target = copy.deepcopy(codec)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.momentum = momentum
        self.schedule = schedule
        self.total_steps = total_steps
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def current_momentum(self) -> float:
        if self.schedule == "const" or not self.total_steps:
            return self.momentum
        t = min(int(self.num_updates.item()), self.total_steps)
        ramp = (math.cos(math.pi * t / self.total_steps) + 1.0) / 2.0
        return 1.0 - (1.0 - self.momentum) * ramp

    @torch.no_grad()
    def update(self, codec: nn.Module) -> float:
        m = self.current_momentum()
        for p_t, p_o in zip(self.target.parameters(), codec.parameters()):
            p_t.mul_(m).add_(p_o.detach(), alpha=1.0 - m)
        for b_t, b_o in zip(self.target.buffers(), codec.buffers()):
            b_t.copy_(b_o)
        self.num_updates += 1
        return m

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.target.encode(x)


def jepa_loss(z_pred: torch.Tensor, target_future_obs: torch.Tensor,
              ema: EMATargetEncoder) -> torch.Tensor:
    return F.mse_loss(z_pred, ema.encode(target_future_obs.float()))


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor) -> dict:
    flat = z.reshape(-1, z.shape[-1]).float()
    std = flat.std(dim=0).mean().item()
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eig.sum()
    if total <= 0:
        return {"emb_std": std, "eff_rank": 0.0}
    p = (eig / total).clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * p.log()).sum()).item())
    return {"emb_std": std, "eff_rank": eff_rank}
