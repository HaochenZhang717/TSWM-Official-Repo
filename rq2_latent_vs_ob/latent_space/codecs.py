from __future__ import annotations

import torch
import torch.nn as nn

from latent_space.autoencoder import build_autoencoder

CODECS = ("ae", "vae")


class StepwiseVAE(nn.Module):

    def __init__(self, enc_in: int, d_model: int, d_ff: int):
        super().__init__()
        self.enc_in, self.d_model, self.d_ff = enc_in, d_model, d_ff
        self.trunk = nn.Sequential(nn.Linear(enc_in, d_ff), nn.ReLU())
        self.mu_head = nn.Linear(d_ff, d_model)
        self.logvar_head = nn.Linear(d_ff, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, enc_in))

    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_dist(x)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_dist(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return self.decode(z), kl

    def set_trainable_mode(self, mode: str) -> list[nn.Parameter]:
        train_encoder = mode == "finetune_all"
        train_decoder = mode in ("finetune_decoder", "finetune_all")
        for m in (self.trunk, self.mu_head, self.logvar_head):
            for p in m.parameters():
                p.requires_grad_(train_encoder)
        for p in self.decoder.parameters():
            p.requires_grad_(train_decoder)
        return [p for p in self.parameters() if p.requires_grad]


def build_codec(codec: str, enc_in: int, d_model: int, d_ff: int | None = None,
                *, use_revin: bool = False):
    d_ff = d_ff if d_ff is not None else 2 * d_model
    if codec == "ae":
        return build_autoencoder(enc_in, d_model, d_ff, use_revin=use_revin)
    if codec == "vae":
        return StepwiseVAE(enc_in, d_model, d_ff)
    raise ValueError(f"unknown codec {codec!r}; choose from {CODECS}")


def load_codec(ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    codec = meta.get("codec", "ae")
    model = build_codec(codec, meta["enc_in"], meta["d_model"], meta.get("d_ff"),
                        use_revin=meta.get("use_revin", False))
    model.load_state_dict(ckpt["state"])
    return model, meta


def codec_ckpt_name(dataset: str, signal: str, codec: str) -> str:
    return (f"{dataset}__{signal}.pt" if codec == "ae"
            else f"{dataset}__{signal}__{codec}.pt")
