from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "RQ4_action_encoding", REPO / "RQ3_fusion_benchmark",
           REPO / "rq2_latent_vs_ob", REPO / "rq2_latent_vs_ob" / "common", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_train as _ft
from common.gpu_dataset import GPUResidentData, GPUWindowBank
from common.windowing import Schema

from action_space import ActionSpace
from build_probe_cache import load as load_bank
from RQ6_intervention.priors import channels, priors_for

L, H = 256, 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AE_DIR = REPO / "rq2_latent_vs_ob/results/latent_space/target/ae"
MODELS = ["Amplifier", "CrossLinear", "DUET", "PatchTST", "TiDE", "TimeKAN", "TimeXer"]
MODULE_ARMS = ("none", "res", "res_zero", "film_zero", "xattn", "gate")
ALL_ARMS = MODULE_ARMS + ("concat",)
STEPWISE_ARMS = ("res", "res_zero", "film_zero", "gate")
QSHIFTS = (0.10, 0.25, 0.50)
QSHIFTS_DN = (-0.10, -0.25, -0.50)
_FWD_SEED = 20260825
N_PERM = 3
SCALES = (0.0, 0.25, 0.5, 2.0, 4.0)
DOSE_QS = (0.25, 0.5, 0.75)
TOL = 1e-6


PSPACE_ARMS = ("obs", "vae", "jepa")
ENCODINGS = ("decay", "conv", "attn")


def split_encoding_arm(arm: str) -> tuple[str | None, str | None]:
    if "__" in arm:
        enc, fus = arm.split("__", 1)
        if enc in ENCODINGS:
            return enc, fus
    return None, None


def route(arm: str) -> str:
    if arm == "concat" or arm in PSPACE_ARMS:
        return "concat"
    return "module"


def ckpt_path(ds: str, model: str, arm: str, seed: int, lam: float) -> Path:
    sfx = "" if seed == 0 else f"_seed{seed}"
    if arm.startswith("gate_soft"):
        assert lam == 0.0, "gate_soft only has lambda=0"
        return (REPO / f"RQ7_causal_fusion/results/{arm}{sfx}/ckpts/"
                       f"{ds}__{model}__instant__gate_soft.pt")
    if arm == "obs":
        assert lam == 0.0, "the obs arm only has lambda=0"
        return REPO / f"rq2_latent_vs_ob/results/action_conditioning{sfx}/ckpts/{ds}__{model}.pt"
    if arm in ("vae", "jepa"):
        assert lam == 0.0, f"the {arm} arm only has lambda=0"
        return (REPO / f"rq2_latent_vs_ob/results/latent_space/target{sfx}/ckpts/"
                       f"{ds}__{model}__{arm}.pt")
    _enc, _fus = split_encoding_arm(arm)
    if _enc is not None:
        assert lam == 0.0, "encoding arms only have lambda=0"
        return (REPO / f"RQ4_action_encoding/results/{_enc}__{_fus}{sfx}/ckpts/"
                       f"{ds}__{model}__{_enc}__{_fus}.pt")
    if arm == "concat":
        assert lam == 0.0, "the concat arm only has lambda=0"
        return REPO / f"rq2_latent_vs_ob/results/latent_space/target{sfx}/ckpts/{ds}__{model}.pt"
    if lam == 0.0:
        return REPO / f"RQ3_fusion_benchmark/results/ae{sfx}/ckpts/{ds}__{model}__{arm}__ae.pt"
    assert arm == "gate", "lambda>0 was only trained on the gate arm"
    tag = f"lam{lam:g}".replace(".", "p")
    return (REPO / f"RQ5_autonomy_split/results/{tag}{sfx}/ckpts/"
                   f"{ds}__{model}__instant__gate__{tag}.pt")


def _stub_prepare(bank: dict, schema: Schema):
    one = {}
    for k, v in bank["tensors"].items():
        x = v[:1]
        one[k] = (x[:, :H] if k.endswith("_future") else x).to(DEVICE)
    b = GPUWindowBank(tensors=one, n=1)
    return lambda *a, **kw: GPUResidentData(train=b, val=b, schema=schema,
                                            combo_to_id=bank.get("combo_to_id"))


class IdentityCodec(torch.nn.Module):

    def __init__(self, n_target: int):
        super().__init__()
        self.d_model = n_target

    def encode(self, x):
        return x

    def decode(self, z):
        return z


class Cell:

    def __init__(self, target_ae, adapter, fusion, embedder, n_latent, recorded,
                 encoder=None):
        self.target_ae, self.adapter = target_ae, adapter
        self.fusion, self.embedder = fusion, embedder
        self.n_latent, self.recorded = n_latent, recorded
        self.encoder = encoder


def load_cell(ds, model, arm, seed, lam, bank, schema) -> Cell | str:
    ck = ckpt_path(ds, model, arm, seed, lam)
    if not ck.exists():
        return f"missing {ck.name}"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    recorded0 = (blob.get("metrics") or {}).get("MAE")

    if arm == "obs":
        from common.action_embedder import ActionEmbedder
        from observational_space.config_builder import configure_adapter
        st = blob["state"]
        cfgd = blob["config"]
        embedder = ActionEmbedder(schema.cardinalities, schema.n_continuous, schema.n_exog)
        adapter = configure_adapter(model, schema, embedder.cov_width,
                                    cfgd["context_length"], cfgd["horizon"])
        adapter.model.load_state_dict(st["model"])
        embedder.load_state_dict(st["embedder"])
        if "fusion" in st and adapter.CovariateFusion is not None:
            adapter.CovariateFusion.load_state_dict(st["fusion"])
        for m in (adapter.model, embedder):
            m.to(DEVICE).eval()
        if adapter.CovariateFusion is not None:
            adapter.CovariateFusion.to(DEVICE).eval()
        return Cell(IdentityCodec(schema.n_target).to(DEVICE), adapter, None, embedder,
                    schema.n_target, recorded0)

    from latent_space.train import TrainConfig
    cfg = TrainConfig(**blob["config"])
    cfg.device, cfg.ae_dir = DEVICE, str(AE_DIR)
    recorded = (blob.get("metrics") or {}).get("MAE")
    state = blob["state"]

    if route(arm) == "concat":
        from latent_space.config_builder import configure_latent_adapter
        from latent_space.embedder import LatentActionEmbedder
        from latent_space.train import _load_codecs
        target_ae, covariate_ae = _load_codecs(cfg, schema)
        n_latent = target_ae.d_model
        embedder = LatentActionEmbedder(schema.cardinalities, schema.n_continuous,
                                        schema.n_exog, covariate_ae=covariate_ae)
        adapter = configure_latent_adapter(model, schema, embedder.cov_width,
                                           cfg.context_length, cfg.horizon, n_latent)
        adapter.model.load_state_dict(state["model"])
        embedder.load_state_dict(state["embedder"])
        target_ae.load_state_dict(state["target_ae"])
        if "fusion" in state and adapter.CovariateFusion is not None:
            adapter.CovariateFusion.load_state_dict(state["fusion"])
        for m in (adapter.model, embedder, target_ae):
            m.to(DEVICE).eval()
        if adapter.CovariateFusion is not None:
            adapter.CovariateFusion.to(DEVICE).eval()
        return Cell(target_ae, adapter, None, embedder, n_latent, recorded)

    orig = _ft.prepare_gpu
    _ft.prepare_gpu = _stub_prepare(bank, schema)
    try:
        _enc, _fus = split_encoding_arm(arm)
        if _enc is not None:
            from encode_train import EncodingTrainer
            tr = EncodingTrainer(cfg, _fus, _enc,
                                 conv_kernel=8, conv_layers=6, attn_dh=64, attn_heads=4)
        elif arm.startswith("gate_soft"):
            from encode_train import EncodingTrainer
            tr = EncodingTrainer(cfg, "gate", "instant",
                                 conv_kernel=8, conv_layers=6, attn_dh=64, attn_heads=4)
        elif lam == 0.0:
            tr = _ft.FusionTrainer(cfg, arm)
        else:
            from encode_train import EncodingTrainer
            tr = EncodingTrainer(cfg, arm, "instant",
                                 conv_kernel=8, conv_layers=6, attn_dh=64, attn_heads=4)
        tr._load_state(state)
    finally:
        _ft.prepare_gpu = orig
    tr._set_train(False)
    enc = getattr(tr, "encoder", None)
    if enc is not None:
        enc.to(DEVICE).eval()
    return Cell(tr.target_ae, tr.adapter, tr.fusion, tr.embedder, tr.n_latent, recorded,
                encoder=enc)


def _seeded(fn, *a, **kw):
    torch.manual_seed(_FWD_SEED)
    return fn(*a, **kw)


class Forward:

    def __init__(self, cell: Cell, batch: dict, arm: str):
        self.cell, self.batch, self.arm = cell, batch, arm
        self.route = route(arm)
        ll = cell.adapter.config.label_len
        th = batch["target_history"].float()
        self.B = th.shape[0]
        self.z_hist = cell.target_ae.encode(th)
        self.cov_hist_raw = cell.embedder.build_covariate(
            batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
        self.cov_hist = self.cov_hist_raw
        mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
        self.mark_enc = batch["mark_history"]
        self.mark_dec = mark_full[:, L - ll:]
        self.ll = ll

        if self.route == "concat":
            self.model_input = torch.cat([self.z_hist, self.cov_hist], dim=-1)
            self.series_dec = self.z_hist.new_zeros((self.B, ll + H, cell.n_latent))
            self.series_dec[:, :ll] = self.z_hist[:, L - ll:]
        else:
            dec = self.z_hist.new_zeros((self.B, ll + H, cell.n_latent))
            dec[:, :ll] = self.z_hist[:, L - ll:]
            empty = self.z_hist.new_zeros((self.B, H, 0))
            out = _seeded(cell.adapter._process, self.z_hist, dec,
                          self.mark_enc, self.mark_dec, empty)
            self.z_raw = out["output"][:, -H:, :cell.n_latent]

    def cov_fut(self, cont, cat, exog) -> torch.Tensor:
        return self.cov_pair(cont, cat, exog)[1]

    def cov_pair(self, cont, cat, exog) -> tuple[torch.Tensor, torch.Tensor]:
        cf = self.cell.embedder.build_covariate(cont, cat, exog)
        enc = self.cell.encoder
        if enc is None or self.cov_hist_raw.shape[-1] == 0:
            return self.cov_hist_raw, cf
        u = enc(torch.cat([self.cov_hist_raw, cf], dim=1))
        return u[:, :L], u[:, L:]

    def latent(self, cov_hist: torch.Tensor, cov_fut: torch.Tensor) -> torch.Tensor:
        c = self.cell
        if self.route == "concat":
            cov_full = torch.cat([cov_hist, cov_fut], dim=1)
            target = torch.cat([self.series_dec, cov_full[:, L - self.ll:]], dim=-1)
            out = _seeded(c.adapter._process, self.model_input, target,
                          self.mark_enc, self.mark_dec, cov_fut)
            z = out["output"][:, -H:, :c.n_latent]
            if c.adapter.CovariateFusion is not None:
                z = c.adapter.CovariateFusion(cov_fut, z)
            return z
        if c.fusion is None:
            return self.z_raw
        return c.fusion(self.z_hist, cov_hist, cov_fut, self.z_raw)

    def predict(self, cont, cat, exog) -> torch.Tensor:
        c = self.cell
        cov_hist, cov_fut = self.cov_pair(cont, cat, exog)
        return c.target_ae.decode(self.latent(cov_hist, cov_fut))


def _mad(x: torch.Tensor) -> torch.Tensor:
    med = x.median(dim=0).values
    return (x - med).abs().median(dim=0).values.clamp(min=1e-3)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def rank(t):
        idx = t.argsort(dim=-1)
        r = torch.empty_like(t)
        ar = torch.arange(t.shape[-1], dtype=t.dtype, device=t.device)
        r.scatter_(-1, idx, ar.expand_as(t))
        return r
    a, b = rank(x), rank(y)
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(den > 1e-12, (a * b).sum(dim=-1) / den, torch.zeros_like(den))


class Acc:

    def __init__(self):
        self.s: dict[str, float] = {}
        self.n: dict[str, float] = {}

    def add(self, key: str, total: float, count: float):
        self.s[key] = self.s.get(key, 0.0) + float(total)
        self.n[key] = self.n.get(key, 0.0) + float(count)

    def mean(self, key: str):
        c = self.n.get(key, 0.0)
        return None if c == 0 else self.s[key] / c

    def keys(self):
        return self.s.keys()


def _event_dense_id(combo_to_id: dict, event_classes) -> int | None:
    want = set(event_classes)
    hit = [cid for combo, cid in (combo_to_id or {}).items() if want & set(combo)]
    return (min(hit) + 1) if hit else None


@torch.no_grad()
def run_cell(cell: Cell, bank: dict, aspace: ActionSpace, prs, ch, arm: str,
             batch_size: int, ladders: str, max_windows: int | None) -> dict:
    n = bank["n"] if max_windows is None else min(bank["n"], max_windows)
    T = bank["tensors"]
    y_true_all = T["target_future"][:n].float()
    scale = _mad(y_true_all.reshape(-1, y_true_all.shape[-1])).to(DEVICE)

    g = torch.Generator().manual_seed(_FWD_SEED)
    perms = [torch.randperm(bank["n"], generator=g) for _ in range(N_PERM)]

    n_cont = T["continuous_future"].shape[-1]
    n_cat = T["categorical_future"].shape[-1]
    ev_col = n_cat - 1
    doses = {p.action: aspace.dose_quantiles(bank, ch.continuous.index(p.action))
             for p in prs if p.kind == "continuous"}
    marg = {p.action: aspace.build_marginal(bank, ch.continuous.index(p.action))
            for p in prs if p.kind == "continuous"}

    acc = Acc()
    same_max = 0.0
    for s0 in range(0, n, batch_size):
        sl = slice(s0, min(s0 + batch_size, n))
        batch = {k: v[sl].to(DEVICE) for k, v in T.items()}
        cont, cat, exog = (batch["continuous_future"], batch["categorical_future"],
                           batch["exog_future"])
        y_true = batch["target_future"].float()
        B = y_true.shape[0]
        fw = Forward(cell, batch, arm)

        y0 = fw.predict(cont, cat, exog)
        acc.add("MAE", (y0 - y_true).abs().sum().item(), y_true.numel())
        acc.add("MSE", (y0 - y_true).pow(2).sum().item(), y_true.numel())

        same_max = max(same_max, (fw.predict(cont.clone(), cat.clone(), exog) - y0)
                       .abs().max().item())

        def resp(y: torch.Tensor) -> float:
            return ((y - y0).abs() / scale).sum().item()

        cont_null = aspace.null(cont, sl)
        cat_null = cat.clone()
        for cname, cls in aspace.cat_null.items():
            col = ev_col if cname == "action_combo" else ch.categorical.index(cname)
            cat_null[..., col] = cls
        y_null = fw.predict(cont_null, cat_null, exog)
        acc.add("R_null", resp(y_null), y0.numel())
        acc.add("MAE_null", (y_null - y_true).abs().sum().item(), y_true.numel())

        for d, perm in enumerate(perms):
            rows = perm[sl]
            cp = T["continuous_future"][rows].to(DEVICE)
            kp = T["categorical_future"][rows].to(DEVICE)
            acc.add("R_perm", resp(fw.predict(cp, kp, exog)), y0.numel())
            if d == 0:
                cont_perm, cat_perm = cp, kp

        if n_cont:
            c2, clip2 = aspace.scale(cont, sl, 2.0)
            acc.add("R_2x", resp(fw.predict(c2, cat, exog)), y0.numel())
            acc.add("frac_clipped_2x", clip2 * y0.shape[0], y0.shape[0])

        k = H - 1
        ck, kk = cont.clone(), cat.clone()
        if n_cont:
            ck[:, k] = cont_perm[:, k]
        if n_cat:
            kk[:, k] = cat_perm[:, k]
        d_leak = (fw.predict(ck, kk, exog) - y0).abs().sum(dim=-1)
        tot = d_leak.sum(dim=1)
        acc.add("L", (d_leak[:, :k].sum(dim=1) / tot.clamp(min=1e-12)).sum().item(), B)
        acc.add("leak_total", tot.sum().item(), B)

        if cell.fusion is not None:
            z_a = fw.latent(*fw.cov_pair(cont, cat, exog))
            z_0 = fw.latent(*fw.cov_pair(cont_null, cat_null, exog))
            y_fix = cell.target_ae.decode(fw.z_raw + (z_a - z_0))
            acc.add("MAE_difffix", (y_fix - y_true).abs().sum().item(), y_true.numel())
            acc.add("MAE_zraw", (cell.target_ae.decode(fw.z_raw) - y_true).abs().sum().item(),
                    y_true.numel())

        for p in prs:
            tcol = ch.target.index(p.target)
            keep = torch.ones(B, dtype=torch.bool, device=DEVICE)
            if p.condition is not None:
                ccol = ch.categorical.index(p.condition[0])
                hist_cat = batch["categorical_history"]
                allcat = torch.cat([hist_cat, cat], dim=1)
                keep = (allcat[..., ccol] == p.condition[1]).float().mean(dim=1) >= 0.5
            if int(keep.sum()) == 0:
                continue

            if p.kind == "continuous":
                col = ch.continuous.index(p.action)
                q = doses[p.action]
                up = aspace.set_dose(cont, sl, col, q[0.75])
                _qs = (QSHIFTS + QSHIFTS_DN) if ladders == "full" \
                    else (QSHIFTS[:1] + QSHIFTS_DN[:1])
                up_q = {d: aspace.quantile_shift(cont, col, marg[p.action], d)
                        for d in _qs}
                inc_old = (up[..., col] - cont[..., col]).sum(dim=1) > 1e-6
                active = aspace.active_mask(cont, sl, col)
                ladder_vals, ladder_cont = [], []
                if ladders != "none":
                    base = aspace.null_norm[sl, col]
                    for qq in DOSE_QS:
                        ladder_cont.append(aspace.set_dose(cont, sl, col, q[qq]))
                        ladder_vals.append(torch.full((B,), q[qq], device=DEVICE))
                    ladder_cont.insert(0, aspace.set_dose(cont, sl, col, base.unsqueeze(1)
                                                          .expand(B, H)))
                    ladder_vals.insert(0, base)
            elif p.kind == "categorical":
                col = ch.categorical.index(p.action)
                up, dn = cat.clone(), cat.clone()
                up[..., col], dn[..., col] = p.levels[1], p.levels[0]
                active = torch.ones(B, dtype=torch.bool, device=DEVICE)
                ladder_vals, ladder_cont = [], []
            else:
                eid = _event_dense_id(bank.get("combo_to_id"), p.event_classes)
                if eid is None:
                    continue
                col = ev_col
                up, dn = cat.clone(), cat.clone()
                dn[..., col] = 0
                up[..., col] = 0
                up[:, 0, col] = eid
                active = torch.ones(B, dtype=torch.bool, device=DEVICE)
                ladder_vals, ladder_cont = [], []

            if p.kind == "continuous":
                delta = (fw.predict(up, cat, exog) - y0)[..., tcol].mean(dim=1)
            else:
                delta = (fw.predict(cont, up, exog)
                         - fw.predict(cont, dn, exog))[..., tcol].mean(dim=1)

            agree = (torch.sign(delta) == p.sign)
            thr = 1e-3 * float(scale[tcol])
            big = delta.abs() > thr
            kk_ = keep
            for tag, m in (("all", kk_), ("tol", kk_ & big), ("active", kk_ & active & big)):
                if int(m.sum()):
                    acc.add(f"sign_agree_{tag}::{p.key}", float(agree[m].sum()), int(m.sum()))

            if p.kind == "continuous":
                mf = kk_ & big & inc_old
                if int(mf.sum()):
                    acc.add(f"sign_agree_filt::{p.key}",
                            float(agree[mf].sum()), int(mf.sum()))
                acc.add(f"frac_up_is_down::{p.key}",
                        float((kk_ & ~inc_old).sum()), int(kk_.sum()))
                for d, up_d in up_q.items():
                    dq = (fw.predict(up_d, cat, exog) - y0)[..., tcol].mean(dim=1)
                    moved = (up_d[..., col] - cont[..., col]).sum(dim=1)
                    inc = moved > 1e-6 if d > 0 else moved < -1e-6
                    want = p.sign if d > 0 else -p.sign
                    mq = kk_ & inc & (dq.abs() > thr)
                    if int(mq.sum()):
                        acc.add(f"sign_agree_q{d:g}::{p.key}",
                                float((torch.sign(dq[mq]) == want).sum()), int(mq.sum()))
                    if d == QSHIFTS[0]:
                        acc.add(f"frac_can_increase::{p.key}",
                                float(inc[kk_].sum()), int(kk_.sum()))
            acc.add(f"frac_active::{p.key}", float((kk_ & active).sum()), int(kk_.sum()))
            acc.add(f"delta_mean::{p.key}", float(delta[kk_].sum()), int(kk_.sum()))

            if ladder_cont:
                deltas = torch.stack(
                    [(fw.predict(c, cat, exog) - y0)[..., tcol].mean(dim=1)
                     for c in ladder_cont], dim=-1)
                rho = _spearman(torch.stack(ladder_vals, dim=-1), deltas) * p.sign
                acc.add(f"mono_dose::{p.key}", float(rho[kk_].sum()), int(kk_.sum()))

            if ladders == "full" and p.kind == "continuous":
                sc, dd, clips = [], [], 0.0
                for c in SCALES:
                    cc, fr = aspace.scale(cont, sl, c, only=col)
                    clips += fr
                    sc.append(torch.full((B,), c, device=DEVICE))
                    dd.append((fw.predict(cc, cat, exog) - y0)[..., tcol].mean(dim=1))
                rho = _spearman(torch.stack(sc, -1), torch.stack(dd, -1)) * p.sign
                acc.add(f"mono_scale::{p.key}", float(rho[kk_].sum()), int(kk_.sum()))
                acc.add(f"frac_clipped::{p.key}", clips / len(SCALES) * B, B)

        if aspace.dataset == "pleiadata":
            mcol = ch.categorical.index("mode")
            spcol = ch.continuous.index("setpoint_norm")
            q75 = doses["setpoint_norm"][0.75]
            hot, cold = cat.clone(), cat.clone()
            hot[..., mcol], cold[..., mcol] = 1, 2
            sp = aspace.set_dose(cont, sl, spcol, q75)
            tcol = ch.target.index("indoor_temp")
            d = (fw.predict(sp, hot, exog) - fw.predict(sp, cold, exog))[..., tcol].mean(dim=1)
            acc.add("modeflip_heat_minus_cool", float(d.sum()), B)
            acc.add("modeflip_frac_positive", float((d > 0).sum()), B)

    out = {k: acc.mean(k) for k in acc.keys()}
    out["R_same_max"] = same_max
    out["n_val"] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Action-intervention probe (no training)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", nargs="+", default=["gate"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ladders", choices=["full", "dose", "none"], default=None,
                    help="default: full for module arms, dose for the concat arm (it re-runs the backbone per counterfactual)")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if DEVICE == "cpu" and os.environ.get("PROBE_ALLOW_CPU") != "1":
        raise SystemExit(
            "CUDA is not available (torch.cuda.is_available() == False). The probe is two orders "
            "of magnitude slower on CPU and does not fail, so it exits here. Check the driver; "
            "set PROBE_ALLOW_CPU=1 to run on CPU anyway.")

    ds = a.dataset
    bank = load_bank(ds)
    schema = bank["schema"]
    ch = channels(ds)
    prs = priors_for(ds)
    aspace = ActionSpace.from_cache(ds, bank, device=DEVICE)
    print(f"[probe] {ds}: n={bank['n']} units={len(set(bank['unit_ids']))} "
          f"priors={len(prs)} device={DEVICE}", flush=True)

    rows, failures = [], {}
    for arm in a.arms:
        ladders = a.ladders or ("dose" if route(arm) == "concat" else "full")
        lams = a.lambdas if arm == "gate" else [0.0]
        for model in a.models:
            for seed in a.seeds:
                for lam in lams:
                    tag = f"{ds}/{arm}/{model}/s{seed}/lam{lam:g}"
                    t0 = time.time()
                    cell = load_cell(ds, model, arm, seed, lam, bank, schema)
                    if isinstance(cell, str):
                        failures[tag] = cell
                        print(f"  ! {tag}: {cell}", flush=True)
                        continue
                    r = run_cell(cell, bank, aspace, prs, ch, arm,
                                 a.batch_size, ladders, a.max_windows)
                    enc_, fus_ = split_encoding_arm(arm)
                    r.update({"dataset": ds, "model": model, "fusion_arm": arm,
                              "seed": seed, "aux_auto": lam,
                              "recorded_MAE": cell.recorded, "ladders": ladders,
                              "seconds": round(time.time() - t0, 1),
                              "encoding": enc_ or "instant",
                              "fusion": fus_ or ("concat" if route(arm) == "concat" else arm),
                              "pred_space": ("obs" if arm == "obs" else
                                             arm if arm in ("vae", "jepa") else "ae")})

                    if (fus_ or arm) in STEPWISE_ARMS or arm == "none":
                        assert (r.get("L") or 0.0) < 1e-6, f"{tag}: L={r['L']} but this arm should give 0 by construction"
                    if arm == "none":
                        assert (r.get("R_perm") or 0.0) < 1e-9, \
                            f"{tag}: none arm R_perm={r['R_perm']} but covariates never enter the model"

                    rows.append(r)
                    dm = "" if cell.recorded is None else f" (ckpt {cell.recorded:.5f})"
                    print(f"  {tag}: MAE={r['MAE']:.5f}{dm} R_perm={r['R_perm']:.4f} "
                          f"same={r['R_same_max']:.1e} {r['seconds']}s", flush=True)

                    out = Path(a.out or (REPO / "RQ6_intervention/results"
                                         / f"probe_intervention__{ds}__{arm}.json"))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps({"rows": rows, "failures": failures}, indent=1))
    print(f"[probe] {ds}: {len(rows)} cells, {len(failures)} missing", flush=True)


if __name__ == "__main__":
    main()
