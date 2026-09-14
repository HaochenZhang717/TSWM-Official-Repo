from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

_RQ3_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _RQ3_DIR.parent
_RQ2_ROOT = _CODE_ROOT / "rq2_latent_vs_ob"
for _p in (_RQ3_DIR, _RQ2_ROOT, _RQ2_ROOT / "common", _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.gpu_dataset import GPUEpochLoader, prepare_gpu
from fusion import ALL_ARMS, FUSION_ARMS, build_fusion
from latent_space import engine
from latent_space.codecs import CODECS, codec_ckpt_name, load_codec
from latent_space.config_builder import configure_latent_adapter
from latent_space.embedder import LatentActionEmbedder
from latent_space.train import (DATASETS_WITH_DATA, TrainConfig, _make_criterion,
                                seed_everything, split_seed)

FUSION_MODELS_DEFAULT = ["PatchTST", "TiDE", "CrossLinear"]


def fusion_forward_loss(adapter, embedder, target_ae, fusion, batch, n_latent, device, criterion):
    label_len = adapter.config.label_len
    batch = {k: v.to(device) for k, v in batch.items()}
    th, tf = batch["target_history"], batch["target_future"]
    L, H = th.shape[1], tf.shape[1]

    z_hist = target_ae.encode(th.float())
    cov_hist = embedder.build_covariate(
        batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
    cov_fut = embedder.build_covariate(
        batch["continuous_future"], batch["categorical_future"], batch["exog_future"])

    B = z_hist.shape[0]
    series_dec = z_hist.new_zeros((B, label_len + H, n_latent))
    series_dec[:, :label_len] = z_hist[:, L - label_len:]
    mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
    empty_cov = z_hist.new_zeros((B, H, 0))

    out = adapter._process(z_hist, series_dec, batch["mark_history"],
                           mark_full[:, L - label_len:], empty_cov)
    z_pred = out["output"][:, -H:, :n_latent]
    if fusion is not None:
        z_pred = fusion(z_hist, cov_hist, cov_fut, z_pred)

    pred_obs = target_ae.decode(z_pred)
    target_future = tf.float()
    main = criterion(pred_obs, target_future)
    loss = main
    aux_val = 0.0
    extra = out.get("additional_loss")
    if extra is not None and torch.is_tensor(extra):
        aux = extra.mean()
        loss = main + aux
        aux_val = aux.item()
    return loss, pred_obs, target_future, {"main": main.item(), "aux": aux_val}


class FusionTrainer:

    def __init__(self, cfg: TrainConfig, arm: str):
        if arm not in ALL_ARMS:
            raise ValueError(f"arm must be one of {ALL_ARMS}, got {arm!r}")
        if cfg.codec not in CODECS:
            raise ValueError(f"codec must be one of {CODECS}, got {cfg.codec!r}")
        self.cfg, self.arm = cfg, arm
        seed_everything(cfg.seed)
        self.device = torch.device(cfg.device)

        data = prepare_gpu(
            cfg.dataset, context_length=cfg.context_length, horizon=cfg.horizon,
            stride=cfg.stride, val_ratio=cfg.val_ratio, seed=split_seed(cfg),
            root=cfg.root, options=cfg.options or None,
            device=cfg.device, cache_dir=cfg.cache_dir,
        )
        self.train_ds, self.val_ds, self.schema = data.train, data.val, data.schema

        codec_path = Path(cfg.ae_dir) / codec_ckpt_name(cfg.dataset, "target", cfg.codec)
        self.target_ae, meta = load_codec(codec_path)
        if meta["enc_in"] != self.schema.n_target:
            raise ValueError(f"codec enc_in={meta['enc_in']} != n_target={self.schema.n_target}")
        self.target_ae.set_trainable_mode(cfg.ae_mode)
        self.n_latent = self.target_ae.d_model

        self.embedder = LatentActionEmbedder(
            self.schema.cardinalities, self.schema.n_continuous, self.schema.n_exog,
            covariate_ae=None)
        self.cov_width = self.embedder.cov_width

        backbone_cov = self.cov_width if arm == "concat" else 0
        self.adapter = configure_latent_adapter(
            cfg.model, self.schema, backbone_cov, cfg.context_length, cfg.horizon, self.n_latent)
        self.fusion = build_fusion(arm, self.n_latent, self.cov_width, cfg.horizon,
                                   cfg.context_length)

        engine.to_device(self.adapter, self.embedder, self.target_ae, self.device)
        if self.fusion is not None:
            self.fusion.to(self.device)

        self.train_loader = GPUEpochLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.val_loader = GPUEpochLoader(
            self.val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        self.criterion = _make_criterion(cfg.loss)
        params = self._trainable_params()
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=cfg.lr_patience)
            if cfg.lr_scheduler else None)
        self.n_params = sum(p.numel() for p in params)
        self.fusion_params = (sum(p.numel() for p in self.fusion.parameters())
                              if self.fusion is not None else 0)

    def _trainable_params(self):
        params = engine.trainable_params(self.adapter, self.embedder, self.target_ae)
        if self.fusion is not None:
            params += [p for p in self.fusion.parameters() if p.requires_grad]
        return params

    def _set_train(self, train: bool):
        engine.set_train_mode(self.adapter, self.embedder, self.target_ae, train)
        if self.fusion is not None:
            self.fusion.train(train)

    def _fwd(self, batch):
        if self.arm == "concat":
            return engine.forward_loss(self.adapter, self.embedder, self.target_ae, batch,
                                       self.n_latent, self.schema.n_target, self.device,
                                       self.criterion)
        return fusion_forward_loss(self.adapter, self.embedder, self.target_ae, self.fusion,
                                   batch, self.n_latent, self.device, self.criterion)

    def _epoch(self, loader, train: bool) -> float:
        self._set_train(train)
        total, seen = 0.0, 0
        limit = self.cfg.max_train_batches if train else self.cfg.max_eval_batches
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i, batch in enumerate(loader):
                if limit is not None and i >= limit:
                    break
                loss, _, _, _ = self._fwd(batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.cfg.grad_clip:
                        nn.utils.clip_grad_norm_(self._trainable_params(), self.cfg.grad_clip)
                    self.optimizer.step()
                total += loss.item()
                seen += 1
        return total / max(seen, 1)

    @torch.no_grad()
    def _metrics(self) -> dict:
        self._set_train(False)
        abs_sum = sq_sum = count = 0.0
        for i, batch in enumerate(self.val_loader):
            if self.cfg.max_eval_batches is not None and i >= self.cfg.max_eval_batches:
                break
            _, pred, target, _ = self._fwd(batch)
            abs_sum += (pred - target).abs().sum().item()
            sq_sum += ((pred - target) ** 2).sum().item()
            count += target.numel()
        if count == 0:
            return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan")}
        mse = sq_sum / count
        return {"MAE": abs_sum / count, "MSE": mse, "RMSE": mse ** 0.5}

    def _state_dict(self) -> dict:
        state = {"model": self.adapter.model.state_dict(),
                 "embedder": self.embedder.state_dict(),
                 "target_ae": self.target_ae.state_dict()}
        if self.adapter.CovariateFusion is not None:
            state["fusion_mlp"] = self.adapter.CovariateFusion.state_dict()
        if self.fusion is not None:
            state["fusion"] = self.fusion.state_dict()
        return state

    def _load_state(self, state: dict) -> None:
        self.adapter.model.load_state_dict(state["model"])
        self.embedder.load_state_dict(state["embedder"])
        self.target_ae.load_state_dict(state["target_ae"])
        if "fusion_mlp" in state and self.adapter.CovariateFusion is not None:
            self.adapter.CovariateFusion.load_state_dict(state["fusion_mlp"])
        if "fusion" in state and self.fusion is not None:
            self.fusion.load_state_dict(state["fusion"])

    def fit(self, ckpt_path: Path | None = None, verbose: bool = True) -> dict:
        cfg = self.cfg
        best_val, best_epoch, bad = float("inf"), -1, 0
        best_state = self._state_dict()
        t0 = time.time()
        for epoch in range(cfg.epochs):
            tr = self._epoch(self.train_loader, train=True)
            vl = self._epoch(self.val_loader, train=False)
            if self.scheduler is not None:
                self.scheduler.step(vl)
            if vl < best_val - 1e-6:
                best_val, best_epoch, bad = vl, epoch, 0
                best_state = {k: {kk: vv.detach().cpu().clone() for kk, vv in sd.items()}
                              for k, sd in self._state_dict().items()}
            else:
                bad += 1
            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                flag = " *" if best_epoch == epoch else ""
                print(f"  [{cfg.model}/{cfg.dataset}/{self.arm}/{cfg.codec}] "
                      f"epoch {epoch + 1}/{cfg.epochs} train={tr:.5f} val={vl:.5f} "
                      f"lr={lr:.2e}{flag}")
            if bad >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} (best {best_epoch + 1}, val={best_val:.5f})")
                break

        self._load_state(best_state)
        metrics = self._metrics()
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": best_state, "config": asdict(cfg), "arm": self.arm,
                        "schema": self.schema.summary(), "metrics": metrics,
                        "n_latent": self.n_latent, "cov_width": self.cov_width,
                        "best_epoch": best_epoch + 1, "best_val_loss": best_val}, ckpt_path)
        return {
            "dataset": cfg.dataset, "model": cfg.model, "arm": self.arm, "codec": cfg.codec,
            "n_latent": self.n_latent, "cov_width": self.cov_width,
            "params_M": round(self.n_params / 1e6, 3),
            "fusion_params": self.fusion_params,
            "best_epoch": best_epoch + 1, "best_val_loss": round(best_val, 6),
            "MAE": round(metrics["MAE"], 5), "MSE": round(metrics["MSE"], 5),
            "RMSE": round(metrics["RMSE"], 5),
            "train_windows": len(self.train_ds), "val_windows": len(self.val_ds),
            "seed": cfg.seed, "data_seed": split_seed(cfg),
            "seconds": round(time.time() - t0, 1),
        }


def main():
    p = argparse.ArgumentParser(description="RQ3 fusion-mechanism benchmark")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=FUSION_MODELS_DEFAULT)
    p.add_argument("--arms", nargs="+", default=["all"])
    p.add_argument("--codec", default="ae", choices=list(CODECS))
    p.add_argument("--ae-dir", required=True,
                   help="dir holding {dataset}__target[__{codec}].pt codec checkpoints")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--loss", default="mse", choices=["huber", "mse", "mae"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="training randomness (init, shuffling, dropout)")
    p.add_argument("--data-seed", type=int, default=None,
                   help="seed for the train/val split; defaults to --seed. Pin it (e.g. 0) "
                        "across a multi-seed sweep so every seed is scored on the same "
                        "held-out subjects.")
    p.add_argument("--num-workers", type=int, default=0, help="unused (kept for script compat): no DataLoader workers, batches are a GPU index_select")
    p.add_argument("--cache-dir", default=None, help="windowed-tensor cache dir (default: <repo>/.gpu_window_cache)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--root", default=None, help="dataset root override (single-dataset runs)")
    p.add_argument("--data-root", default=None, help="dataset root (multi-dataset)")
    p.add_argument("--split", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt-dir", default=None)
    args = p.parse_args()

    datasets = DATASETS_WITH_DATA if args.datasets == ["all"] else args.datasets
    arms = list(FUSION_ARMS) if args.arms == ["all"] else args.arms

    def dataset_root(name: str) -> str | None:
        if args.root:
            return args.root
        if not args.data_root:
            return None
        r = Path(args.data_root)
        return str({
            "greenhouse": r / "greenhouse3" / "TimeSeries",
            "vitaldb": r / "vital_db",
            "cgmacros": r / "diabetes_datasets" / "cgmacros",
            "shanghai_diabetes": r / "diabetes_datasets" / "Shanghai_T1DM_T2DM",
            "pleiadata": r / "PLEIAData",
            "predist": r / "PreDist" / "predist_dataset" / "manufacturer_2",
            "wastewater_nutrient": r / "Wastewater_Treatment_Plant_Data_for_Nutrient_Removal_System"
                                     / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv",
            "mimic_cardio": r / "mimic_cardio",
        }[name])

    dataset_opts = {"vitaldb": {"split": "all"}, "cgmacros": {"split": "all"},
                    "shanghai_diabetes": {"split": "T2DM"}}

    rows, failures = [], {}
    out_path = Path(args.out) if args.out else None
    for dataset in datasets:
        for model in args.models:
            for arm in arms:
                cfg = TrainConfig(
                    dataset=dataset, model=model, ae_dir=args.ae_dir,
                    latent_scope="target", ae_mode="frozen", codec=args.codec,
                    context_length=args.context_length, horizon=args.horizon, stride=args.stride,
                    batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                    lr=args.lr, grad_clip=args.grad_clip, val_ratio=args.val_ratio,
                    seed=args.seed, data_seed=args.data_seed,
                    num_workers=args.num_workers, cache_dir=args.cache_dir, device=args.device,
                    lr_scheduler=True, lr_patience=args.lr_patience, loss=args.loss,
                    max_train_batches=args.max_train_batches,
                    max_eval_batches=args.max_eval_batches,
                    root=dataset_root(dataset),
                    options={**dataset_opts.get(dataset, {}),
                             **({"split": args.split} if args.split else {})},
                )
                print(f"\n>>> fusion arm={arm} model={model} dataset={dataset} codec={args.codec}")
                try:
                    trainer = FusionTrainer(cfg, arm)
                    ckpt = (Path(args.ckpt_dir) / f"{dataset}__{model}__{arm}__{args.codec}.pt"
                            if args.ckpt_dir else None)
                    row = trainer.fit(ckpt_path=ckpt, verbose=True)
                    rows.append(row)
                    print(f"<<< {dataset}/{model}/{arm}/{args.codec}: MAE={row['MAE']} "
                          f"MSE={row['MSE']} (epoch {row['best_epoch']}, {row['seconds']}s, "
                          f"fusion_params={row['fusion_params']})")
                except Exception as exc:
                    failures[f"{dataset}/{model}/{arm}/{args.codec}"] = repr(exc)
                    print(f"<<< {dataset}/{model}/{arm}/{args.codec} FAILED: {exc!r}")
                    traceback.print_exc()
                if out_path is not None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))

    print("\n" + "=" * 100)
    print(f"{'dataset':<20}{'model':<13}{'arm':<11}{'codec':<7}{'MAE':>9}{'MSE':>9}{'RMSE':>9}"
          f"{'epoch':>7}{'sec':>8}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: (x["dataset"], x["model"], x["MSE"])):
        print(f"{r['dataset']:<20}{r['model']:<13}{r['arm']:<11}{r['codec']:<7}{r['MAE']:>9.4f}"
              f"{r['MSE']:>9.4f}{r['RMSE']:>9.4f}{r['best_epoch']:>7}{r['seconds']:>8.0f}")
    if failures:
        print("FAILED:", json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
