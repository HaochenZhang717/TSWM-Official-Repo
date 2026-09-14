from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "RQ4_action_encoding", _REPO / "RQ3_fusion_benchmark",
           _REPO / "rq2_latent_vs_ob", _REPO / "rq2_latent_vs_ob" / "common",
           _REPO / "RQ7_causal_fusion"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encode_train import EncodingTrainer
from latent_space.train import TrainConfig

from causal_fusion import build_spec
from prior_split import SPLITS, train_priors

RHO_DEFAULT = 1.0
DELTA_LO, DELTA_HI = 0.05, 0.30
EPS = 1e-6


class SoftConstraintTrainer(EncodingTrainer):

    def __init__(self, cfg: TrainConfig, *, prior_split: str = "default",
                 rho: float = RHO_DEFAULT, seed: int = 0):
        if prior_split not in SPLITS:
            raise ValueError(f"unknown prior_split {prior_split!r}")
        self.prior_split, self.rho = prior_split, rho
        super().__init__(cfg, "gate", "instant",
                         conv_kernel=8, conv_layers=6, attn_dh=64, attn_heads=4)
        self._install_constraints()
        self._gen = torch.Generator(device="cpu").manual_seed(seed + 9173)

    def _install_constraints(self) -> None:
        prs = train_priors(self.cfg.dataset, self.prior_split)
        slots, cons = build_spec(self.cfg.dataset, self.schema, prs)
        self.slots, self.constraints = slots, cons
        self._marg = {}
        for c in cons:
            s = slots[c.slot_idx]
            if s.kind != "continuous":
                continue
            if s.col not in self._marg:
                x = self.train_ds.tensors["continuous_future"][..., s.col].reshape(-1).float()
                self._marg[s.col] = torch.sort(x).values.to(self.device)
        print(f"  [gate_soft] {self.cfg.dataset}: {len(cons)} constraints "
              f"{[c.key for c in cons]}  rho={self.rho}",
              flush=True)

    def _shift(self, cont: torch.Tensor, col: int, delta: float) -> torch.Tensor:
        v = self._marg[col]
        n = v.numel()
        x = cont[..., col].float().contiguous()
        if delta >= 0:
            rank = torch.searchsorted(v, x, right=True)
            tgt = torch.clamp(rank + int(delta * n), max=n - 1)
        else:
            rank = torch.searchsorted(v, x, right=False)
            tgt = torch.clamp(rank + int(delta * n), min=0)
        out = cont.clone()
        out[..., col] = v[tgt].to(cont.dtype)
        return out

    def _fwd(self, batch):
        cfg, dev = self.cfg, self.device
        batch = {k: v.to(dev) for k, v in batch.items()}
        th, tf = batch["target_history"], batch["target_future"]
        L, H = th.shape[1], tf.shape[1]
        ll = self.adapter.config.label_len

        z_hist = self.target_ae.encode(th.float())
        cov_hist = self.embedder.build_covariate(
            batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
        cov_fut0 = self.embedder.build_covariate(
            batch["continuous_future"], batch["categorical_future"], batch["exog_future"])
        B = z_hist.shape[0]
        dec_in = z_hist.new_zeros((B, ll + H, self.n_latent))
        dec_in[:, :ll] = z_hist[:, L - ll:]
        mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
        out = self.adapter._process(z_hist, dec_in, batch["mark_history"],
                                    mark_full[:, L - ll:], z_hist.new_zeros((B, H, 0)))
        z_pred = out["output"][:, -H:, :self.n_latent]

        cont, cat, ex = (batch["continuous_future"], batch["categorical_future"],
                         batch["exog_future"])

        def decode(c, cf=None):
            if cf is None:
                cf = self.embedder.build_covariate(c, cat, ex)
            return self.target_ae.decode(self.fusion(z_hist, cov_hist, cf, z_pred))

        pred = decode(cont, cov_fut0)
        main = self.criterion(pred, tf.float())

        aux_bb = z_hist.new_zeros(())
        extra = out.get("additional_loss")
        if extra is not None and torch.is_tensor(extra):
            aux_bb = extra.mean()

        l_dir = z_hist.new_zeros(())
        if self.fusion.training and self.constraints:
            delta = float(torch.rand(1, generator=self._gen).item()) * (DELTA_HI - DELTA_LO) + DELTA_LO
            deltas = (delta,)
            terms = []
            for c in self.constraints:
                s = self.slots[c.slot_idx]
                if s.kind != "continuous":
                    continue
                for dl in deltas:
                    shifted = self._shift(cont, s.col, dl)
                    d = (decode(shifted) - pred)[..., c.tgt]
                    keep = torch.ones_like(d, dtype=torch.bool)
                    if c.conditional:
                        keep = (cat[..., c.cond_col] == c.cond_cls)
                    if not bool(keep.any()):
                        continue
                    dk = d[keep]
                    scale = dk.abs().mean().detach() + EPS
                    want = c.sign if dl > 0 else -c.sign
                    terms.append((torch.relu(-want * dk) / scale).mean())
            if terms:
                l_dir = torch.stack(terms).mean()

        loss = main + aux_bb + self.rho * l_dir
        return loss, pred, tf.float(), {"main": float(main.item()),
                                        "aux": float(aux_bb.item()) + float(l_dir.item())}

    def fit(self, ckpt_path: Path | None = None, verbose: bool = True) -> dict:
        row = super().fit(ckpt_path=ckpt_path, verbose=verbose)
        row["fusion_arm"] = "gate_soft"
        row["rho"] = self.rho
        row["prior_split"] = self.prior_split
        row["n_constraints"] = len(self.constraints)
        row["constraints"] = [c.key for c in self.constraints]
        row.update(self.autonomous_metrics())
        row["auto_over_full"] = row["MAE_auto"] / max(row["MAE"], 1e-12)
        return row

    @torch.no_grad()
    def autonomous_metrics(self) -> dict:
        self._set_train(False)
        abs_sum = cnt = 0
        for batch in self.val_loader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            th, tf = b["target_history"], b["target_future"]
            L, H = th.shape[1], tf.shape[1]
            ll = self.adapter.config.label_len
            z_hist = self.target_ae.encode(th.float())
            B = z_hist.shape[0]
            dec_in = z_hist.new_zeros((B, ll + H, self.n_latent))
            dec_in[:, :ll] = z_hist[:, L - ll:]
            mf = torch.cat([b["mark_history"], b["mark_future"]], dim=1)
            o = self.adapter._process(z_hist, dec_in, b["mark_history"], mf[:, L - ll:],
                                      z_hist.new_zeros((B, H, 0)))
            pred = self.target_ae.decode(o["output"][:, -H:, :self.n_latent])
            abs_sum += (pred - tf.float()).abs().sum().item()
            cnt += tf.numel()
        return {"MAE_auto": abs_sum / max(cnt, 1)}
