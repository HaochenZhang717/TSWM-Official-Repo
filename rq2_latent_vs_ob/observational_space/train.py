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
if str(_RQ2_ROOT) not in sys.path:
    sys.path.insert(0, str(_RQ2_ROOT))

from common.action_embedder import ActionEmbedder
from observational_space.config_builder import ALL_MODELS, configure_adapter, is_pure_series
from common.gpu_dataset import GPUEpochLoader, prepare_gpu
from observational_space.engine import (
    evaluate,
    forward_loss,
    set_train_mode,
    to_device,
    trainable_params,
)

DATASETS_WITH_DATA = [
    "greenhouse", "vitaldb", "cgmacros", "shanghai_diabetes", "pleiadata", "predist",
    "wastewater_nutrient",
]


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
    wandb_project: str = "tswm-obs"
    wandb_run_name: str | None = None


def _make_criterion(name: str) -> nn.Module:
    return {"huber": nn.HuberLoss(), "mse": nn.MSELoss(), "mae": nn.L1Loss()}[name]


class Trainer:
    def __init__(self, cfg: TrainConfig):
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
        self.embedder = ActionEmbedder(self.schema.cardinalities, self.schema.n_continuous, self.schema.n_exog)
        self.cov_width = self.embedder.cov_width
        self.adapter = configure_adapter(
            cfg.model, self.schema, self.cov_width, cfg.context_length, cfg.horizon)
        to_device(self.adapter, self.embedder, self.device)

        self.train_loader = GPUEpochLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.val_loader = GPUEpochLoader(
            self.val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        self.criterion = _make_criterion(cfg.loss)
        self.optimizer = torch.optim.Adam(
            trainable_params(self.adapter, self.embedder), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=cfg.lr_patience)
            if cfg.lr_scheduler else None
        )
        self.n_params = sum(p.numel() for p in trainable_params(self.adapter, self.embedder))

    def train_epoch(self) -> dict:
        set_train_mode(self.adapter, self.embedder, True)
        total = main = aux = 0.0
        seen = 0
        for i, batch in enumerate(self.train_loader):
            if self.cfg.max_train_batches is not None and i >= self.cfg.max_train_batches:
                break
            loss, _, _, stats = forward_loss(
                self.adapter, self.embedder, batch, self.schema.n_target, self.device, self.criterion)
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(trainable_params(self.adapter, self.embedder), self.cfg.grad_clip)
            self.optimizer.step()
            total += loss.item()
            main += stats["main"]
            aux += stats["aux"]
            seen += 1
        n = max(seen, 1)
        return {"total": total / n, "main": main / n, "aux": aux / n}

    def val_loss(self) -> dict:
        set_train_mode(self.adapter, self.embedder, False)
        crit = self.criterion
        total = main = aux = 0.0
        seen = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if self.cfg.max_eval_batches is not None and i >= self.cfg.max_eval_batches:
                    break
                loss, _, _, stats = forward_loss(
                    self.adapter, self.embedder, batch, self.schema.n_target, self.device, crit)
                total += loss.item()
                main += stats["main"]
                aux += stats["aux"]
                seen += 1
        set_train_mode(self.adapter, self.embedder, True)
        n = max(seen, 1)
        return {"total": total / n, "main": main / n, "aux": aux / n}

    def state_dict(self) -> dict:
        state = {"model": self.adapter.model.state_dict(), "embedder": self.embedder.state_dict()}
        if self.adapter.CovariateFusion is not None:
            state["fusion"] = self.adapter.CovariateFusion.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.adapter.model.load_state_dict(state["model"])
        self.embedder.load_state_dict(state["embedder"])
        if "fusion" in state and self.adapter.CovariateFusion is not None:
            self.adapter.CovariateFusion.load_state_dict(state["fusion"])

    def _wandb_init(self):
        cfg = self.cfg
        if not cfg.wandb:
            return None
        try:
            import wandb
        except ImportError:
            print("  [wandb] not installed; skipping logging")
            return None
        run_name = cfg.wandb_run_name or f"{cfg.dataset}__{cfg.model}"
        run = wandb.init(
            project=cfg.wandb_project, name=run_name, reinit=True,
            config={**asdict(cfg), "schema": self.schema.summary(),
                    "cov_width": self.cov_width, "n_params": self.n_params,
                    "pure_series": is_pure_series(cfg.model)},
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
            _te = time.time()
            tr = self.train_epoch()
            vl = self.val_loss()
            epoch_secs = time.time() - _te
            vl_total = vl["total"]
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
                run.log({"epoch": epoch + 1,
                         "train_loss": tr["total"], "train_main_loss": tr["main"], "train_aux_loss": tr["aux"],
                         "val_loss": vl_total, "val_main_loss": vl["main"], "val_aux_loss": vl["aux"],
                         "lr": lr, "best_val_loss": best_val}, step=epoch + 1)
            if verbose:
                flag = " *" if improved else ""
                aux_str = f" aux={tr['aux']:.4f}" if tr["aux"] else ""
                print(f"  [{cfg.model}/{cfg.dataset}] epoch {epoch + 1}/{cfg.epochs} "
                      f"train={tr['total']:.4f} val={vl_total:.4f}{aux_str} lr={lr:.2e} "
                      f"({epoch_secs:.1f}s){flag}", flush=True)
            if bad >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} (best epoch {best_epoch + 1}, val={best_val:.4f})")
                break

        self.load_state_dict(best_state)
        metrics = evaluate(self.adapter, self.embedder, self.val_loader,
                           self.schema.n_target, self.device, max_batches=cfg.max_eval_batches)
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": best_state, "config": asdict(cfg),
                        "schema": self.schema.summary(), "metrics": metrics,
                        "best_epoch": best_epoch + 1, "best_val_loss": best_val}, ckpt_path)
        if run is not None:
            run.summary.update({"best_epoch": best_epoch + 1, "best_val_loss": best_val,
                                "val_MAE": metrics["MAE"], "val_MSE": metrics["MSE"],
                                "val_RMSE": metrics["RMSE"]})
            run.finish()

        return {
            "dataset": cfg.dataset, "model": cfg.model,
            "n_target": self.schema.n_target, "cov_width": self.cov_width,
            "input_dim": self.schema.n_target + self.cov_width,
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
    ckpt = (ckpt_dir / f"{cfg.dataset}__{cfg.model}.pt") if ckpt_dir else None
    return trainer.fit(ckpt_path=ckpt, verbose=verbose)


def run_benchmark(datasets, models, base_cfg_kwargs, out_path, ckpt_dir, verbose=True):
    rows, failures = [], {}
    for dataset in datasets:
        for model in models:
            cfg = TrainConfig(dataset=dataset, model=model, **base_cfg_kwargs)
            print(f"\n>>> training {model} on {dataset}")
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
    print("\n" + "=" * 78)
    print(f"{'dataset':<20}{'model':<13}{'MAE':>9}{'MSE':>9}{'RMSE':>9}{'epoch':>7}{'fus':>5}")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: (x["dataset"], x["MAE"])):
        print(f"{r['dataset']:<20}{r['model']:<13}{r['MAE']:>9.4f}{r['MSE']:>9.4f}"
              f"{r['RMSE']:>9.4f}{r['best_epoch']:>7}{'Y' if r['fusion'] else '-':>5}")


def main():
    p = argparse.ArgumentParser(description="Train action-conditioned forecasting backbones")
    p.add_argument("--dataset", default=None, help="single dataset (shortcut for --datasets X)")
    p.add_argument("--datasets", nargs="+", default=None, help="datasets or 'all'")
    p.add_argument("--model", default=None, help="single model (shortcut for --models X)")
    p.add_argument("--models", nargs="+", default=None, help="models or 'all'")
    p.add_argument("--context-length", type=int, default=96)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--stride", type=int, default=None, help="window stride; default seq_len")
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
    p.add_argument("--lr-patience", type=int, default=5, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--out", default=None, help="results JSON path (sweep mode)")
    p.add_argument("--ckpt-dir", default=None, help="dir to save best checkpoints")
    p.add_argument("--root", default=None, help="dataset root path (single-dataset runs)")
    p.add_argument("--split", default=None, help="loader split (vitaldb=all, shanghai=T2DM, ...)")
    p.add_argument("--download", action="store_true", help="download dataset if missing (vitaldb)")
    p.add_argument("--wandb", action="store_true", help="log train/val loss to W&B")
    p.add_argument("--wandb-project", default="tswm-obs")
    p.add_argument("--wandb-run-name", default=None, help="default f'{dataset}__{model}'")
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
        context_length=args.context_length, horizon=args.horizon, stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, val_ratio=args.val_ratio,
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
