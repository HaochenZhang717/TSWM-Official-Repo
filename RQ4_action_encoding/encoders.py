from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODING_ARMS = ("instant", "decay", "conv", "attn")

DEFAULT_CONV_K = 8
DEFAULT_CONV_LAYERS = 6
DEFAULT_ATTN_DH = 64
DEFAULT_ATTN_HEADS = 4


def _causal_depthwise(x: torch.Tensor, weight: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    c = x.shape[-1]
    K = weight.shape[-1]
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (dilation * (K - 1), 0))
    out = F.conv1d(xt, weight, groups=c, dilation=dilation)
    return out.transpose(1, 2)


class DecayEncoder(nn.Module):

    def __init__(self, cov_width: int, seq_len: int, init_theta: float = 0.0):
        super().__init__()
        self.cov_width = cov_width
        self.seq_len = seq_len
        self.theta = nn.Parameter(torch.full((cov_width,), float(init_theta)))
        self.gate = nn.Parameter(torch.zeros(cov_width))

    def kernel(self) -> torch.Tensor:
        alpha = torch.sigmoid(self.theta).clamp(0.0, 1.0 - 1e-6)
        lags = torch.arange(self.seq_len - 1, -1, -1,
                            device=alpha.device, dtype=alpha.dtype)
        w = (1.0 - alpha).unsqueeze(1) * alpha.unsqueeze(1) ** lags.unsqueeze(0)
        return w.unsqueeze(1)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.shape[-1] == 0:
            return cov
        ema = _causal_depthwise(cov, self.kernel().to(cov.dtype))
        return cov + self.gate.to(cov.dtype) * (ema - cov)

    @torch.no_grad()
    def readout(self) -> dict:
        return {"alpha": torch.sigmoid(self.theta).tolist(),
                "gate": self.gate.tolist()}


class ConvEncoder(nn.Module):

    def __init__(self, cov_width: int, kernel_size: int = DEFAULT_CONV_K,
                 n_layers: int = DEFAULT_CONV_LAYERS, dilations: list[int] | None = None):
        super().__init__()
        self.cov_width = cov_width
        self.kernel_size = kernel_size
        self.dilations = dilations if dilations is not None else [2 ** i for i in range(n_layers)]
        self.weights = nn.ParameterList(
            nn.Parameter(torch.zeros(cov_width, 1, kernel_size)) for _ in self.dilations)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.shape[-1] == 0:
            return cov
        u = cov
        for w, d in zip(self.weights, self.dilations):
            u = u + F.gelu(_causal_depthwise(u, w, dilation=d))
        return u


class AttnEncoder(nn.Module):

    def __init__(self, cov_width: int, seq_len: int,
                 d_h: int = DEFAULT_ATTN_DH, n_heads: int = DEFAULT_ATTN_HEADS):
        super().__init__()
        self.cov_width = cov_width
        self.seq_len = seq_len
        self.in_proj = nn.Linear(cov_width, d_h)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_h))
        nn.init.normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_h, cov_width)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.shape[-1] == 0:
            return cov
        t = cov.shape[1]
        h = self.in_proj(cov) + self.pos[:t].unsqueeze(0)
        mask = torch.triu(torch.ones(t, t, device=cov.device, dtype=torch.bool),
                          diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        return cov + self.out_proj(a)


def build_encoder(arm: str, cov_width: int, seq_len: int, *,
                  conv_kernel: int = DEFAULT_CONV_K,
                  conv_layers: int = DEFAULT_CONV_LAYERS,
                  attn_dh: int = DEFAULT_ATTN_DH,
                  attn_heads: int = DEFAULT_ATTN_HEADS) -> nn.Module | None:
    if arm not in ENCODING_ARMS:
        raise ValueError(f"unknown encoding arm {arm!r}; choose from {ENCODING_ARMS}")
    if arm == "instant" or cov_width == 0:
        return None
    if arm == "decay":
        return DecayEncoder(cov_width, seq_len)
    if arm == "conv":
        enc = ConvEncoder(cov_width, conv_kernel, conv_layers)
        if enc.receptive_field < seq_len:
            print(f"[encoders] WARNING: conv receptive field {enc.receptive_field} "
                  f"< seq_len {seq_len}; decay can reach further, which confounds "
                  f"the conv-vs-decay comparison. Raise --conv-layers/--conv-kernel.")
        return enc
    if arm == "attn":
        return AttnEncoder(cov_width, seq_len, attn_dh, attn_heads)
    raise AssertionError(arm)
