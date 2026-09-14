from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_RQ4 = Path(__file__).resolve().parent
_ROOT = _RQ4.parent
_RQ2 = _ROOT / "rq2_latent_vs_ob"
_RQ3 = _ROOT / "RQ3_fusion_benchmark"
for _p in (_RQ4, _RQ3, _RQ2, _RQ2 / "common", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encoders import ENCODING_ARMS, build_encoder

from fusion_train import FusionTrainer

from latent_space.train import DATASETS_WITH_DATA, TrainConfig

ALLOWED_FUSION = ("gate", "film_zero", "res", "res_zero", "xattn")

MODELS_DEFAULT = ["TimeXer", "TiDE", "DUET", "PatchTST", "TimeKAN", "CrossLinear", "Amplifier"]


def encoding_forward_loss(adapter, embedder, target_ae, encoder, fusion, batch,
                          n_latent, device, criterion):
    label_len = adapter.config.label_len
    batch = {k: v.to(device) for k, v in batch.items()}
    th, tf = batch["target_history"], batch["target_future"]
    L, H = th.shape[1], tf.shape[1]

    z_hist = target_ae.encode(th.float())
    cov_hist = embedder.build_covariate(
        batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
    cov_fut = embedder.build_covariate(
        batch["continuous_future"], batch["categorical_future"], batch["exog_future"])

    if encoder is not None and cov_hist.shape[-1] > 0:
        u_full = encoder(torch.cat([cov_hist, cov_fut], dim=1))
        cov_hist, cov_fut = u_full[:, :L], u_full[:, L:]

    B = z_hist.shape[0]
    series_dec = z_hist.new_zeros((B, label_len + H, n_latent))
    series_dec[:, :label_len] = z_hist[:, L - label_len:]
    mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
    empty_cov = z_hist.new_zeros((B, H, 0))

    out = adapter._process(z_hist, series_dec, batch["mark_history"],
                           mark_full[:, L - label_len:], empty_cov)
    z_pred = out["output"][:, -H:, :n_latent]
    z_pred = fusion(z_hist, cov_hist, cov_fut, z_pred)

    pred_obs = target_ae.decode(z_pred)
    target_future = tf.float()
    main = criterion(pred_obs, target_future)
    loss, aux_val = main, 0.0
    extra = out.get("additional_loss")
    if extra is not None and torch.is_tensor(extra):
        aux = extra.mean()
        loss = main + aux
        aux_val = aux.item()
    return loss, pred_obs, target_future, {"main": main.item(), "aux": aux_val}


class EncodingTrainer(FusionTrainer):

    encoder = None

    def __init__(self, cfg: TrainConfig, fusion_arm: str, encoding_arm: str, *,
                 conv_kernel: int, conv_layers: int, attn_dh: int, attn_heads: int):
        if fusion_arm not in ALLOWED_FUSION:
            raise ValueError(f"fusion arm must be one of {ALLOWED_FUSION}, got {fusion_arm!r} "
                             f"(`none` ignores cov; `concat` bypasses the fusion module)")
        if encoding_arm not in ENCODING_ARMS:
            raise ValueError(f"encoding must be one of {ENCODING_ARMS}, got {encoding_arm!r}")
        self.encoding_arm = encoding_arm
        super().__init__(cfg, fusion_arm)

        self.encoder = build_encoder(
            encoding_arm, self.cov_width, cfg.context_length + cfg.horizon,
            conv_kernel=conv_kernel, conv_layers=conv_layers,
            attn_dh=attn_dh, attn_heads=attn_heads)
        if self.encoder is not None:
            self.encoder.to(self.device)

        params = self._trainable_params()
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=cfg.lr_patience)
            if cfg.lr_scheduler else None)
        self.n_params = sum(p.numel() for p in params)
        self.encoder_params = (sum(p.numel() for p in self.encoder.parameters())
                               if self.encoder is not None else 0)

    def _trainable_params(self):
        params = super()._trainable_params()
        if self.encoder is not None:
            params += [p for p in self.encoder.parameters() if p.requires_grad]
        return params

    def _set_train(self, train: bool):
        super()._set_train(train)
        if self.encoder is not None:
            self.encoder.train(train)

    def _fwd(self, batch):
        return encoding_forward_loss(
            self.adapter, self.embedder, self.target_ae, self.encoder, self.fusion,
            batch, self.n_latent, self.device, self.criterion)

    def _state_dict(self) -> dict:
        state = super()._state_dict()
        if self.encoder is not None:
            state["encoder"] = self.encoder.state_dict()
        return state

    def _load_state(self, state: dict) -> None:
        super()._load_state(state)
        if "encoder" in state and self.encoder is not None:
            self.encoder.load_state_dict(state["encoder"])

    def fit(self, ckpt_path: Path | None = None, verbose: bool = True) -> dict:
        row = super().fit(ckpt_path=ckpt_path, verbose=verbose)
        row["fusion_arm"] = row.pop("arm")
        row["encoding"] = self.encoding_arm
        row["encoder_params"] = self.encoder_params
        row["conv_kernel"] = getattr(self.encoder, "kernel_size", None)
        row["conv_dilations"] = getattr(self.encoder, "dilations", None)
        row["receptive_field"] = getattr(self.encoder, "receptive_field", None)
        if hasattr(self.encoder, "readout"):
            row["decay_readout"] = self.encoder.readout()
        return row


def dataset_root(data_root: str | None, root: str | None, name: str) -> str | None:
    if root:
        return root
    if not data_root:
        return None
    r = Path(data_root)
    return str({
        "greenhouse": r / "greenhouse3" / "TimeSeries",
        "vitaldb": r / "vital_db",
        "cgmacros": r / "diabetes_datasets" / "cgmacros",
        "shanghai_diabetes": r / "diabetes_datasets" / "Shanghai_T1DM_T2DM",
        "pleiadata": r / "PLEIAData",
        "predist": r / "PreDist" / "predist_dataset" / "manufacturer_2",
        "wastewater_nutrient": (r / "Wastewater_Treatment_Plant_Data_for_Nutrient_Removal_System"
                                / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"),
        "mimic_cardio": r / "mimic_cardio",
    }[name])


DATASET_OPTS = {"vitaldb": {"split": "all"}, "cgmacros": {"split": "all"},
                "shanghai_diabetes": {"split": "T2DM"}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    p.add_argument("--encodings", nargs="+", default=["all"])
    p.add_argument("--fusion-arms", nargs="+", default=["gate", "film_zero"])
    p.add_argument("--codec", default="ae")
    p.add_argument("--ae-dir", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="mse")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--conv-kernel", type=int, default=8, help="taps per conv layer")
    p.add_argument("--conv-layers", type=int, default=6,
                   help="dilated layers; dilations are 1,2,4,... Receptive field is\n"
                        "1+(K-1)*(2^L-1) and must cover context+horizon.")
    p.add_argument("--attn-dh", type=int, default=64)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    args = p.parse_args()

    datasets = DATASETS_WITH_DATA if args.datasets == ["all"] else args.datasets
    encodings = list(ENCODING_ARMS) if args.encodings == ["all"] else args.encodings

    rows, failures = [], {}
    out_path = Path(args.out) if args.out else None
    for dataset in datasets:
        for model in args.models:
            for enc in encodings:
                for fus in args.fusion_arms:
                    cfg = TrainConfig(
                        dataset=dataset, model=model, ae_dir=args.ae_dir,
                        latent_scope="target", ae_mode="frozen", codec=args.codec,
                        context_length=args.context_length, horizon=args.horizon,
                        stride=args.stride, batch_size=args.batch_size, epochs=args.epochs,
                        patience=args.patience, lr=args.lr, grad_clip=args.grad_clip,
                        val_ratio=args.val_ratio, seed=args.seed, data_seed=args.data_seed,
                        num_workers=args.num_workers, cache_dir=args.cache_dir,
                        device=args.device, lr_scheduler=True, lr_patience=args.lr_patience,
                        loss=args.loss, max_train_batches=args.max_train_batches,
                        max_eval_batches=args.max_eval_batches,
                        root=dataset_root(args.data_root, args.root, dataset),
                        options=DATASET_OPTS.get(dataset, {}),
                    )
                    key = f"{dataset}/{model}/{enc}/{fus}"
                    print(f"\n>>> encoding={enc} fusion={fus} model={model} dataset={dataset}")
                    try:
                        trainer = EncodingTrainer(
                            cfg, fus, enc, conv_kernel=args.conv_kernel,
                            conv_layers=args.conv_layers,
                            attn_dh=args.attn_dh, attn_heads=args.attn_heads)
                        ckpt = (Path(args.ckpt_dir) /
                                f"{dataset}__{model}__{enc}__{fus}.pt") if args.ckpt_dir else None
                        row = trainer.fit(ckpt_path=ckpt)
                        rows.append(row)
                        print(f"<<< {key}: MAE={row['MAE']} MSE={row['MSE']} "
                              f"(epoch {row['best_epoch']}, {row['seconds']}s, "
                              f"encoder_params={row['encoder_params']})")
                    except Exception as exc:
                        failures[key] = repr(exc)
                        print(f"!!! FAILED {key}: {exc!r}", file=sys.stderr)
                    if out_path is not None:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(
                            {"rows": rows, "failures": failures}, indent=1))

    if rows:
        print("\n" + "=" * 100)
        hdr = f"{'dataset':<20}{'model':<12}{'enc':<10}{'fusion':<11}{'MAE':>9}{'MSE':>9}{'sec':>8}"
        print(hdr); print("-" * 100)
        for r in rows:
            print(f"{r['dataset']:<20}{r['model']:<12}{r['encoding']:<10}{r['fusion_arm']:<11}"
                  f"{r['MAE']:>9.4f}{r['MSE']:>9.4f}{r['seconds']:>8.0f}")
    if failures:
        print(f"\n{len(failures)} FAILURES: {list(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
