from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_RQ2_ROOT = Path(__file__).resolve().parents[1]
for _p in (_RQ2_ROOT, _RQ2_ROOT / "common", _RQ2_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.gpu_dataset import GPUEpochLoader, prepare_gpu
from latent_space.autoencoder import AE_MODES, load_autoencoder
from latent_space.config_builder import ALL_MODELS, configure_latent_adapter, is_pure_series
from latent_space.embedder import LatentActionEmbedder
from latent_space.engine import (
    evaluate,
    forward_loss,
    set_train_mode,
    to_device,
    trainable_params,
)

DATASETS_WITH_DATA = [
    "greenhouse", "vitaldb", "cgmacros", "shanghai_diabetes", "pleiadata", "predist",
    "wastewater_nutrient", "mimic_cardio",
]
LATENT_SCOPES = ("target", "target_cov")

LATENT_ARMS = ("ae", "vae", "jepa", "jepa0")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_seed(cfg) -> int:
    if isinstance(cfg, dict):
        return cfg["seed"] if cfg.get("data_seed") is None else cfg["data_seed"]
    return cfg.seed if cfg.data_seed is None else cfg.data_seed


@dataclass
class TrainConfig:
    dataset: str
    model: str
    ae_dir: str
    latent_scope: str = "target"
    ae_mode: str = "frozen"
    codec: str = "ae"
    jepa_alpha: float = 0.02
    jepa_ema: float = 0.996
    jepa_ema_schedule: str = "cosine"
    jepa_init: str = "scratch"
    jepa_d_model: int = 32
    context_length: int = 96
    horizon: int = 24
    stride: int | None = None
    batch_size: int = 32
    epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    val_ratio: float = 0.2
    seed: int = 42
    data_seed: int | None = None
    num_workers: int = 0
    cache_dir: str | None = None
    device: str = "cpu"
    lr_scheduler: bool = True
    lr_patience: int = 5
    loss: str = "huber"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    root: str | None = None
    options: dict = field(default_factory=dict)
    wandb: bool = False
    wandb_project: str = "tswm-latent"
    wandb_run_name: str | None = None


def _make_criterion(name: str) -> nn.Module:
    return {"huber": nn.HuberLoss(), "mse": nn.MSELoss(), "mae": nn.L1Loss()}[name]


def _ae_path(ae_dir: Path, dataset: str, signal: str, codec: str = "ae") -> Path:
    from latent_space.codecs import codec_ckpt_name

    return ae_dir / codec_ckpt_name(dataset, signal, codec)


def _build_jepa_codec(cfg: TrainConfig, schema):
    from latent_space.autoencoder import build_autoencoder

    if cfg.latent_scope != "target":
        raise ValueError(
            f"the JEPA arm is defined for latent_scope='target' only (raw covariates, "
            f"like the observational baseline), got {cfg.latent_scope!r}")
    if cfg.jepa_init == "ae":
        codec, meta = load_autoencoder(_ae_path(Path(cfg.ae_dir), cfg.dataset, "target"))
        if meta["enc_in"] != schema.n_target:
            raise ValueError(
                f"warm-start AE enc_in={meta['enc_in']} != schema.n_target={schema.n_target}")
    elif cfg.jepa_init == "scratch":
        codec = build_autoencoder(schema.n_target, cfg.jepa_d_model, use_revin=False)
    else:
        raise ValueError(f"jepa_init must be 'scratch' or 'ae', got {cfg.jepa_init!r}")
    codec.set_trainable_mode("finetune_all")
    return codec


def _load_codecs(cfg: TrainConfig, schema):
    from latent_space.codecs import load_codec
    from latent_space.jepa import JEPA_ARMS

    if cfg.codec in JEPA_ARMS:
        return _build_jepa_codec(cfg, schema), None

    ae_dir = Path(cfg.ae_dir)
    target_ae, meta = load_codec(_ae_path(ae_dir, cfg.dataset, "target", getattr(cfg, "codec", "ae")))
    if meta["enc_in"] != schema.n_target:
        raise ValueError(
            f"target AE enc_in={meta['enc_in']} != schema.n_target={schema.n_target} "
            f"(AE trained on a different dataset/contract?)")
    target_ae.set_trainable_mode(cfg.ae_mode)

    covariate_ae = None
    n_real_cov = schema.n_continuous + schema.n_exog
    if cfg.latent_scope == "target_cov" and n_real_cov > 0:
        covariate_ae, cmeta = load_autoencoder(_ae_path(ae_dir, cfg.dataset, "covariate"))
        if cmeta["enc_in"] != n_real_cov:
            raise ValueError(
                f"covariate AE enc_in={cmeta['enc_in']} != n_continuous+n_exog={n_real_cov}")
        covariate_ae.set_trainable_mode(cfg.ae_mode)
    return target_ae, covariate_ae


class Trainer:
    def __init__(self, cfg: TrainConfig):
        from latent_space.jepa import JEPA_ARMS

        if cfg.latent_scope not in LATENT_SCOPES:
            raise ValueError(f"latent_scope must be one of {LATENT_SCOPES}, got {cfg.latent_scope!r}")
        if cfg.codec not in LATENT_ARMS:
            raise ValueError(f"codec must be one of {LATENT_ARMS}, got {cfg.codec!r}")
        if cfg.codec in JEPA_ARMS:
            cfg.ae_mode = "finetune_all"
            if cfg.codec == "jepa0":
                cfg.jepa_alpha = 0.0
        if cfg.ae_mode not in AE_MODES:
            raise ValueError(f"ae_mode must be one of {AE_MODES}, got {cfg.ae_mode!r}")
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = torch.device(cfg.device)

        data = prepare_gpu(
            cfg.dataset, context_length=cfg.context_length, horizon=cfg.horizon,
            stride=cfg.stride, val_ratio=cfg.val_ratio, seed=split_seed(cfg),
            root=cfg.root, options=cfg.options or None,
            device=cfg.device, cache_dir=cfg.cache_dir,
        )
        self.train_ds, self.val_ds, self.schema = data.train, data.val, data.schema

        self.target_ae, covariate_ae = _load_codecs(cfg, self.schema)
        self.n_latent = self.target_ae.d_model
        self.embedder = LatentActionEmbedder(
            self.schema.cardinalities, self.schema.n_continuous, self.schema.n_exog,
            covariate_ae=covariate_ae)
        self.cov_width = self.embedder.cov_width

        self.adapter = configure_latent_adapter(
            cfg.model, self.schema, self.cov_width, cfg.context_length, cfg.horizon, self.n_latent)
        to_device(self.adapter, self.embedder, self.target_ae, self.device)

        self.train_loader = GPUEpochLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.val_loader = GPUEpochLoader(
            self.val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        self.jepa = None
        if cfg.codec in JEPA_ARMS:
            from latent_space.jepa import EMATargetEncoder

            self.jepa = EMATargetEncoder(
                self.target_ae, momentum=cfg.jepa_ema, schedule=cfg.jepa_ema_schedule,
                total_steps=cfg.epochs * max(len(self.train_loader), 1))
            self.jepa.to(self.device)

        self.criterion = _make_criterion(cfg.loss)
        params = trainable_params(self.adapter, self.embedder, self.target_ae)
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=cfg.lr_patience)
            if cfg.lr_scheduler else None
        )
        self.n_params = sum(p.numel() for p in params)

    def _fwd(self, batch):
        return forward_loss(self.adapter, self.embedder, self.target_ae, batch,
                            self.n_latent, self.schema.n_target, self.device, self.criterion,
                            jepa=self.jepa, jepa_alpha=self.cfg.jepa_alpha)

    def train_epoch(self) -> dict:
        set_train_mode(self.adapter, self.embedder, self.target_ae, True)
        total = main = aux = jepa = 0.0
        seen = 0
        for i, batch in enumerate(self.train_loader):
            if self.cfg.max_train_batches is not None and i >= self.cfg.max_train_batches:
                break
            loss, _, _, stats = self._fwd(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(
                    trainable_params(self.adapter, self.embedder, self.target_ae), self.cfg.grad_clip)
            self.optimizer.step()
            if self.jepa is not None:
                self.jepa.update(self.target_ae)
            total += loss.item()
            main += stats["main"]
            aux += stats["aux"]
            jepa += stats["jepa"]
            seen += 1
        n = max(seen, 1)
        return {"total": total / n, "main": main / n, "aux": aux / n, "jepa": jepa / n}

    def val_loss(self) -> dict:
        set_train_mode(self.adapter, self.embedder, self.target_ae, False)
        total = main = aux = jepa = 0.0
        seen = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if self.cfg.max_eval_batches is not None and i >= self.cfg.max_eval_batches:
                    break
                loss, _, _, stats = self._fwd(batch)
                total += loss.item()
                main += stats["main"]
                aux += stats["aux"]
                jepa += stats["jepa"]
                seen += 1
        set_train_mode(self.adapter, self.embedder, self.target_ae, True)
        n = max(seen, 1)
        return {"total": total / n, "main": main / n, "aux": aux / n, "jepa": jepa / n}

    @staticmethod
    def _selection_loss(vl: dict) -> float:
        return vl["main"] + vl["aux"]

    def state_dict(self) -> dict:
        state = {"model": self.adapter.model.state_dict(),
                 "embedder": self.embedder.state_dict(),
                 "target_ae": self.target_ae.state_dict()}
        if self.adapter.CovariateFusion is not None:
            state["fusion"] = self.adapter.CovariateFusion.state_dict()
        if self.jepa is not None:
            state["jepa_target"] = self.jepa.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.adapter.model.load_state_dict(state["model"])
        self.embedder.load_state_dict(state["embedder"])
        self.target_ae.load_state_dict(state["target_ae"])
        if "fusion" in state and self.adapter.CovariateFusion is not None:
            self.adapter.CovariateFusion.load_state_dict(state["fusion"])
        if "jepa_target" in state and self.jepa is not None:
            self.jepa.load_state_dict(state["jepa_target"])

    def _wandb_init(self):
        cfg = self.cfg
        if not cfg.wandb:
            return None
        try:
            import wandb
        except ImportError:
            print("  [wandb] not installed; skipping logging")
            return None
        suffix = "" if cfg.codec == "ae" else f"__{cfg.codec}"
        run_name = cfg.wandb_run_name or f"{cfg.dataset}__{cfg.model}__{cfg.latent_scope}{suffix}"
        run = wandb.init(
            project=cfg.wandb_project, name=run_name, reinit=True,
            config={**asdict(cfg), "schema": self.schema.summary(),
                    "n_latent": self.n_latent, "cov_width": self.cov_width,
                    "n_params": self.n_params, "pure_series": is_pure_series(cfg.model)},
        )
        return run

    def fit(self, ckpt_path: Path | None = None, verbose: bool = True) -> dict:
        cfg = self.cfg
        run = self._wandb_init()
        best_val = float("inf")
        best_state = self.state_dict()
        best_epoch = -1
        bad = 0
        t0 = time.time()
        for epoch in range(cfg.epochs):
            tr = self.train_epoch()
            vl = self.val_loss()
            vl_total = self._selection_loss(vl)
            if self.scheduler is not None:
                self.scheduler.step(vl_total)
            improved = vl_total < best_val - 1e-6
            if improved:
                best_val, best_epoch, bad = vl_total, epoch, 0
                best_state = {k: {kk: vv.detach().cpu().clone() for kk, vv in sd.items()}
                              for k, sd in self.state_dict().items()}
            else:
                bad += 1
            lr = self.optimizer.param_groups[0]["lr"]
            if run is not None:
                log = {"epoch": epoch + 1,
                       "train_loss": tr["total"], "train_main_loss": tr["main"], "train_aux_loss": tr["aux"],
                       "val_loss": vl_total, "val_main_loss": vl["main"], "val_aux_loss": vl["aux"],
                       "lr": lr, "best_val_loss": best_val}
                if self.jepa is not None:
                    log.update({"train_jepa_loss": tr["jepa"], "val_jepa_loss": vl["jepa"],
                                "ema_momentum": self.jepa.current_momentum()})
                run.log(log, step=epoch + 1)
            if verbose:
                flag = " *" if improved else ""
                aux_str = f" aux={tr['aux']:.4f}" if tr["aux"] else ""
                jepa_str = f" jepa={tr['jepa']:.4f}" if self.jepa is not None else ""
                print(f"  [{cfg.model}/{cfg.dataset}/{cfg.latent_scope}] epoch {epoch + 1}/{cfg.epochs} "
                      f"train={tr['total']:.4f} val={vl_total:.4f}{aux_str}{jepa_str} lr={lr:.2e}{flag}")
            if bad >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} (best epoch {best_epoch + 1}, val={best_val:.4f})")
                break

        self.load_state_dict(best_state)
        metrics = evaluate(self.adapter, self.embedder, self.target_ae, self.val_loader,
                           self.n_latent, self.schema.n_target, self.device, max_batches=cfg.max_eval_batches)
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": best_state, "config": asdict(cfg),
                        "schema": self.schema.summary(), "metrics": metrics,
                        "n_latent": self.n_latent, "cov_width": self.cov_width,
                        "best_epoch": best_epoch + 1, "best_val_loss": best_val}, ckpt_path)
        if run is not None:
            run.summary.update({"best_epoch": best_epoch + 1, "best_val_loss": best_val,
                                "val_MAE": metrics["MAE"], "val_MSE": metrics["MSE"],
                                "val_RMSE": metrics["RMSE"]})
            run.finish()

        row_jepa = ({"jepa_alpha": cfg.jepa_alpha, "jepa_init": cfg.jepa_init}
                    if self.jepa is not None else {})
        return {
            "dataset": cfg.dataset, "model": cfg.model, "latent_scope": cfg.latent_scope,
            "codec": cfg.codec, **row_jepa,
            "ae_mode": cfg.ae_mode, "n_latent": self.n_latent, "n_target": self.schema.n_target,
            "cov_width": self.cov_width, "input_dim": self.n_latent + self.cov_width,
            "pure_series": is_pure_series(cfg.model),
            "fusion": self.adapter.CovariateFusion is not None,
            "params_M": round(self.n_params / 1e6, 3),
            "best_epoch": best_epoch + 1, "best_val_loss": round(best_val, 5),
            "MAE": round(metrics["MAE"], 5), "MSE": round(metrics["MSE"], 5), "RMSE": round(metrics["RMSE"], 5),
            "train_windows": len(self.train_ds), "val_windows": len(self.val_ds),
            "seconds": round(time.time() - t0, 1),
        }


def train_one(cfg: TrainConfig, ckpt_dir: Path | None = None, verbose: bool = True) -> dict:
    trainer = Trainer(cfg)
    suffix = "" if cfg.codec == "ae" else f"__{cfg.codec}"
    ckpt = (ckpt_dir / f"{cfg.dataset}__{cfg.model}{suffix}.pt") if ckpt_dir else None
    return trainer.fit(ckpt_path=ckpt, verbose=verbose)


def run_benchmark(datasets, models, base_cfg_kwargs, out_path, ckpt_dir, verbose=True):
    rows, failures = [], {}
    for dataset in datasets:
        for model in models:
            cfg = TrainConfig(dataset=dataset, model=model, **base_cfg_kwargs)
            print(f"\n>>> training {model} on {dataset} (latent_scope={cfg.latent_scope})")
            try:
                row = train_one(cfg, ckpt_dir=ckpt_dir, verbose=verbose)
                rows.append(row)
                print(f"<<< {model}/{dataset}: MAE={row['MAE']} MSE={row['MSE']} "
                      f"(best epoch {row['best_epoch']}, {row['seconds']}s)")
            except Exception as exc:
                failures[f"{dataset}/{model}"] = repr(exc)
                print(f"<<< {model}/{dataset} FAILED: {exc!r}")
                traceback.print_exc()

    result = {"rows": rows, "failures": failures}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out_path}")

    _print_table(rows)
    if failures:
        print("FAILED:", failures)
    return result


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no results)")
        return
    print("\n" + "=" * 86)
    print(f"{'dataset':<20}{'model':<13}{'scope':<11}{'MAE':>9}{'MSE':>9}{'RMSE':>9}{'epoch':>7}{'fus':>5}")
    print("-" * 86)
    for r in sorted(rows, key=lambda x: (x["dataset"], x["MAE"])):
        print(f"{r['dataset']:<20}{r['model']:<13}{r['latent_scope']:<11}{r['MAE']:>9.4f}{r['MSE']:>9.4f}"
              f"{r['RMSE']:>9.4f}{r['best_epoch']:>7}{'Y' if r['fusion'] else '-':>5}")


def main():
    p = argparse.ArgumentParser(description="Train action-conditioned backbones in AE latent space")
    p.add_argument("--dataset", default=None, help="single dataset (shortcut for --datasets X)")
    p.add_argument("--datasets", nargs="+", default=None, help="datasets or 'all'")
    p.add_argument("--model", default=None, help="single model (shortcut for --models X)")
    p.add_argument("--models", nargs="+", default=None, help="models or 'all'")
    p.add_argument("--ae-dir", required=True, help="dir holding {dataset}__{signal}.pt AE checkpoints")
    p.add_argument("--latent-scope", default="target", choices=list(LATENT_SCOPES))
    p.add_argument("--ae-mode", default="frozen", choices=list(AE_MODES))
    p.add_argument("--codec", default="ae", choices=list(LATENT_ARMS),
                   help="latent arm: ae|vae = pretrained codec {dataset}__target[__{codec}].pt; "
                        "jepa = joint-trained + EMA target encoder; jepa0 = same with alpha=0")
    p.add_argument("--jepa-alpha", type=float, default=0.02, help="JEPA arm: latent loss weight")
    p.add_argument("--jepa-ema", type=float, default=0.996, help="JEPA arm: EMA momentum")
    p.add_argument("--jepa-ema-schedule", default="cosine", choices=["cosine", "const"],
                   help="JEPA arm: cosine ramps the momentum to 1.0 (V-JEPA recipe)")
    p.add_argument("--jepa-init", default="scratch", choices=["scratch", "ae"],
                   help="JEPA arm: train E_c/D from scratch, or warm-start from the AE ckpt")
    p.add_argument("--jepa-d-model", type=int, default=32,
                   help="JEPA arm latent width (keep at 32 to match the AE codec width)")
    p.add_argument("--context-length", type=int, default=96)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42,
                   help="training randomness (init, shuffling, dropout)")
    p.add_argument("--data-seed", type=int, default=None,
                   help="seed for the train/val split only; default follows --seed. "
                        "Pin it across a multi-seed sweep to hold the split fixed.")
    p.add_argument("--num-workers", type=int, default=0, help="unused (kept for script compat): no DataLoader workers, batches are a GPU index_select")
    p.add_argument("--cache-dir", default=None, help="windowed-tensor cache dir (default: <repo>/.gpu_window_cache)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--loss", default="huber", choices=["huber", "mse", "mae"])
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--out", default=None, help="results JSON path (sweep mode)")
    p.add_argument("--ckpt-dir", default=None, help="dir to save best checkpoints")
    p.add_argument("--root", default=None, help="dataset root path (single-dataset runs)")
    p.add_argument("--split", default=None, help="loader split (vitaldb=all, shanghai=T2DM, ...)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--wandb", action="store_true", help="log train/val loss to W&B")
    p.add_argument("--wandb-project", default="tswm-latent")
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    datasets = args.datasets or ([args.dataset] if args.dataset else ["predist"])
    if datasets == ["all"]:
        datasets = DATASETS_WITH_DATA
    models = args.models or ([args.model] if args.model else ["TimeXer"])
    if models == ["all"]:
        models = ALL_MODELS

    options = {}
    if args.split is not None:
        options["split"] = args.split
    if args.download:
        options["download"] = True

    base_cfg_kwargs = dict(
        ae_dir=args.ae_dir, latent_scope=args.latent_scope, ae_mode=args.ae_mode, codec=args.codec,
        jepa_alpha=args.jepa_alpha, jepa_ema=args.jepa_ema,
        jepa_ema_schedule=args.jepa_ema_schedule, jepa_init=args.jepa_init,
        jepa_d_model=args.jepa_d_model,
        context_length=args.context_length, horizon=args.horizon, stride=args.stride,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, val_ratio=args.val_ratio,
        seed=args.seed, data_seed=args.data_seed,
        num_workers=args.num_workers, cache_dir=args.cache_dir, device=args.device, loss=args.loss,
        lr_scheduler=not args.no_scheduler, lr_patience=args.lr_patience,
        max_train_batches=args.max_train_batches, max_eval_batches=args.max_eval_batches,
        root=args.root, options=options,
        wandb=args.wandb, wandb_project=args.wandb_project, wandb_run_name=args.wandb_run_name,
    )
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None

    run_benchmark(datasets, models, base_cfg_kwargs, args.out, ckpt_dir, verbose=True)


if __name__ == "__main__":
    main()
