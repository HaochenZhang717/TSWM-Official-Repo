from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

_RQ5_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _RQ5_DIR.parent
for _p in (_CODE_ROOT / "RQ4_action_encoding", _CODE_ROOT / "RQ3_fusion_benchmark",
           _CODE_ROOT / "rq2_latent_vs_ob", _CODE_ROOT / "rq2_latent_vs_ob" / "common",
           _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encode_train import (ALLOWED_FUSION, DATASET_OPTS, EncodingTrainer,
                          dataset_root)
from encoders import ENCODING_ARMS
from latent_space.train import DATASETS_WITH_DATA, TrainConfig

AUX_SPACES = ("obs", "latent")
SELECT_ON = ("full", "total")


def lam_tag(lam: float) -> str:
    return f"lam{lam:g}".replace(".", "p")


class AutonomyTrainer(EncodingTrainer):

    def __init__(self, cfg: TrainConfig, fusion_arm: str, encoding_arm: str, *,
                 aux_auto: float, aux_space: str, select_on: str = "full", **kw):
        if aux_space not in AUX_SPACES:
            raise ValueError(f"aux_space must be one of {AUX_SPACES}, got {aux_space!r}")
        if select_on not in SELECT_ON:
            raise ValueError(f"select_on must be one of {SELECT_ON}, got {select_on!r}")
        if aux_auto < 0:
            raise ValueError(f"aux_auto must be >= 0, got {aux_auto}")
        self.aux_auto, self.aux_space, self.select_on = aux_auto, aux_space, select_on
        super().__init__(cfg, fusion_arm, encoding_arm, **kw)

    def _epoch(self, loader, train: bool) -> float:
        self._set_train(train)
        total, seen = 0.0, 0
        limit = self.cfg.max_train_batches if train else self.cfg.max_eval_batches
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i, batch in enumerate(loader):
                if limit is not None and i >= limit:
                    break
                loss, _, _, info = self._fwd(batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.cfg.grad_clip:
                        nn.utils.clip_grad_norm_(self._trainable_params(), self.cfg.grad_clip)
                    self.optimizer.step()
                score = loss.item()
                if not train and self.select_on == "full":
                    score -= self.aux_auto * info["auto"]
                total += score
                seen += 1
        return total / max(seen, 1)

    def _split_forward(self, batch):
        tr = self
        batch = {k: v.to(tr.device) for k, v in batch.items()}
        th, tf = batch["target_history"], batch["target_future"]
        L, H = th.shape[1], tf.shape[1]

        z_hist = tr.target_ae.encode(th.float())
        cov_hist = tr.embedder.build_covariate(
            batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
        cov_fut = tr.embedder.build_covariate(
            batch["continuous_future"], batch["categorical_future"], batch["exog_future"])
        if tr.encoder is not None and cov_hist.shape[-1] > 0:
            u_full = tr.encoder(torch.cat([cov_hist, cov_fut], dim=1))
            cov_hist, cov_fut = u_full[:, :L], u_full[:, L:]

        B = z_hist.shape[0]
        label_len = tr.adapter.config.label_len
        series_dec = z_hist.new_zeros((B, label_len + H, tr.n_latent))
        series_dec[:, :label_len] = z_hist[:, L - label_len:]
        mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
        out = tr.adapter._process(z_hist, series_dec, batch["mark_history"],
                                  mark_full[:, L - label_len:], z_hist.new_zeros((B, H, 0)))

        z_raw = out["output"][:, -H:, :tr.n_latent]
        z_pred = tr.fusion(z_hist, cov_hist, cov_fut, z_raw)
        return (tr.target_ae.decode(z_pred), tr.target_ae.decode(z_raw),
                z_raw, tf.float(), out.get("additional_loss"))

    def _fwd(self, batch):
        pred_full, pred_auto, z_raw, y, extra = self._split_forward(batch)

        main = self.criterion(pred_full, y)
        if self.aux_space == "obs":
            l_auto = self.criterion(pred_auto, y)
        else:
            with torch.no_grad():
                z_true = self.target_ae.encode(y)
            l_auto = self.criterion(z_raw, z_true)

        loss = main + self.aux_auto * l_auto
        aux_val = 0.0
        if extra is not None and torch.is_tensor(extra):
            aux = extra.mean()
            loss = loss + aux
            aux_val = aux.item()
        info = {"main": main.item(), "aux": aux_val, "auto": l_auto.item(),
                "pred_auto": pred_auto}
        return loss, pred_full, y, info

    @torch.no_grad()
    def _metrics(self) -> dict:
        self._set_train(False)
        acc = {k: [0.0, 0.0] for k in ("full", "auto")}
        count = 0.0
        for i, batch in enumerate(self.val_loader):
            if self.cfg.max_eval_batches is not None and i >= self.cfg.max_eval_batches:
                break
            _, pred, target, info = self._fwd(batch)
            for key, p in (("full", pred), ("auto", info["pred_auto"])):
                acc[key][0] += (p - target).abs().sum().item()
                acc[key][1] += ((p - target) ** 2).sum().item()
            count += target.numel()
        if count == 0:
            nan = float("nan")
            return {k: nan for k in ("MAE", "MSE", "RMSE", "MAE_auto", "MSE_auto")}
        (a_f, s_f), (a_a, s_a) = acc["full"], acc["auto"]
        mse = s_f / count
        return {"MAE": a_f / count, "MSE": mse, "RMSE": mse ** 0.5,
                "MAE_auto": a_a / count, "MSE_auto": s_a / count}

    def fit(self, ckpt_path: Path | None = None, verbose: bool = True) -> dict:
        row = super().fit(ckpt_path=ckpt_path, verbose=verbose)
        m = self._metrics()
        row["aux_auto"] = self.aux_auto
        row["aux_space"] = self.aux_space
        row["select_on"] = self.select_on
        row["MAE_auto"] = round(m["MAE_auto"], 5)
        row["MSE_auto"] = round(m["MSE_auto"], 5)
        row["auto_over_full"] = round(m["MAE_auto"] / m["MAE"], 4) if m["MAE"] else None
        return row


def main() -> None:
    p = argparse.ArgumentParser(description="RQ5 autonomy-split: aux loss on the pre-fusion latent")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["CrossLinear", "TimeKAN", "TiDE"])
    p.add_argument("--fusion-arms", nargs="+", default=["gate"], choices=list(ALLOWED_FUSION))
    p.add_argument("--encodings", nargs="+", default=["conv"], choices=list(ENCODING_ARMS))
    p.add_argument("--aux-auto", nargs="+", type=float, default=[0.0],
                   help="lambda on the autonomous term; 0 reproduces the RQ4 cell exactly")
    p.add_argument("--aux-space", default="obs", choices=list(AUX_SPACES))
    p.add_argument("--select-on", default="full", choices=list(SELECT_ON),
                   help="val score for early stop / checkpoint pick. 'full' excludes the "
                        "aux term so every lambda arm is selected on the same criterion as "
                        "the reused RQ3 lambda=0 rows.")
    p.add_argument("--codec", default="ae")
    p.add_argument("--ae-dir", required=True)
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--conv-kernel", type=int, default=8)
    p.add_argument("--conv-layers", type=int, default=6)
    p.add_argument("--attn-dh", type=int, default=64)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt-dir", default=None)
    args = p.parse_args()

    datasets = DATASETS_WITH_DATA if args.datasets == ["all"] else args.datasets
    rows, failures = [], {}
    out_path = Path(args.out) if args.out else None

    for dataset in datasets:
        for model in args.models:
            for enc in args.encodings:
                for fus in args.fusion_arms:
                    for lam in args.aux_auto:
                        cfg = TrainConfig(
                            dataset=dataset, model=model, ae_dir=args.ae_dir,
                            latent_scope="target", ae_mode="frozen", codec=args.codec,
                            context_length=args.context_length, horizon=args.horizon,
                            stride=args.stride, batch_size=args.batch_size,
                            epochs=args.epochs, patience=args.patience, lr=args.lr,
                            grad_clip=args.grad_clip, val_ratio=args.val_ratio,
                            seed=args.seed, data_seed=args.data_seed,
                            cache_dir=args.cache_dir, device=args.device,
                            lr_scheduler=True, lr_patience=args.lr_patience, loss=args.loss,
                            max_train_batches=args.max_train_batches,
                            max_eval_batches=args.max_eval_batches,
                            root=dataset_root(args.data_root, args.root, dataset),
                            options=DATASET_OPTS.get(dataset, {}),
                        )
                        tag = f"{dataset}/{model}/{enc}/{fus}/lam={lam}"
                        print(f"\n>>> {tag}", flush=True)
                        try:
                            trainer = AutonomyTrainer(
                                cfg, fus, enc, aux_auto=lam, aux_space=args.aux_space,
                                select_on=args.select_on,
                                conv_kernel=args.conv_kernel, conv_layers=args.conv_layers,
                                attn_dh=args.attn_dh, attn_heads=args.attn_heads)
                            ckpt = (Path(args.ckpt_dir) /
                                    f"{dataset}__{model}__{enc}__{fus}__{lam_tag(lam)}.pt"
                                    if args.ckpt_dir else None)
                            row = trainer.fit(ckpt_path=ckpt, verbose=True)
                            rows.append(row)
                            print(f"<<< {tag}: MAE={row['MAE']} MAE_auto={row['MAE_auto']} "
                                  f"ratio={row['auto_over_full']}", flush=True)
                        except Exception as exc:
                            failures[tag] = repr(exc)
                            print(f"<<< {tag} FAILED: {exc!r}", flush=True)
                            traceback.print_exc()
                        if out_path is not None:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_text(
                                json.dumps({"rows": rows, "failures": failures}, indent=2))

    print("\n" + "=" * 96)
    print(f"{'dataset':<20}{'model':<13}{'fus':<11}{'lam':>6}{'MAE':>10}{'MAE_auto':>11}{'ratio':>8}")
    print("-" * 96)
    for r in sorted(rows, key=lambda x: (x["dataset"], x["model"], x["aux_auto"])):
        print(f"{r['dataset']:<20}{r['model']:<13}{r['fusion_arm']:<11}{r['aux_auto']:>6}"
              f"{r['MAE']:>10.5f}{r['MAE_auto']:>11.5f}{r['auto_over_full']:>8.2f}")
    if failures:
        print("FAILED:", json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
