from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _normalize(self, x):
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x


class StepwiseAutoEncoder(nn.Module):

    def __init__(self, enc_in: int, d_model: int, d_ff: int, *, use_revin: bool = False,
                 revin_affine: bool = True):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_revin = use_revin
        self.revin = RevIN(enc_in, affine=revin_affine) if use_revin else None

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, enc_in),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.decoder(latent)
        if self.revin is not None:
            x = self.revin(x, "denorm")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def set_trainable_mode(self, mode: str) -> list[nn.Parameter]:
        if mode not in AE_MODES:
            raise ValueError(f"unknown ae_mode {mode!r}; choose from {AE_MODES}")
        train_encoder = mode == "finetune_all"
        train_decoder = mode in ("finetune_decoder", "finetune_all")
        for p in self.encoder.parameters():
            p.requires_grad_(train_encoder)
        for p in self.decoder.parameters():
            p.requires_grad_(train_decoder)
        if self.revin is not None:
            for p in self.revin.parameters():
                p.requires_grad_(train_encoder)
        return [p for p in self.parameters() if p.requires_grad]


AE_MODES = ("frozen", "finetune_decoder", "finetune_all")


def build_autoencoder(enc_in: int, d_model: int, d_ff: int | None = None, *,
                      use_revin: bool = False, revin_affine: bool = True) -> StepwiseAutoEncoder:
    d_ff = d_ff if d_ff is not None else 2 * d_model
    return StepwiseAutoEncoder(
        enc_in=enc_in, d_model=d_model, d_ff=d_ff,
        use_revin=use_revin, revin_affine=revin_affine,
    )


def load_autoencoder(ckpt_path, map_location="cpu") -> tuple[StepwiseAutoEncoder, dict]:
    import torch

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    model = build_autoencoder(
        meta["enc_in"], meta["d_model"], meta["d_ff"], use_revin=meta["use_revin"])
    model.load_state_dict(ckpt["state"])
    return model, meta
