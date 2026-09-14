from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_RQ2_ROOT = Path(__file__).resolve().parents[1]
for _p in (_RQ2_ROOT, _RQ2_ROOT / "common", _RQ2_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.datasets import collate, prepare
from latent_space.config_builder import configure_latent_adapter, is_pure_series
from latent_space.embedder import LatentActionEmbedder
from latent_space.engine import evaluate, to_device
from latent_space.train import LATENT_ARMS, _load_codecs, TrainConfig, split_seed


def _load_ckpt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def eval_one(
    dataset: str,
    model: str,
    ckpt_path: Path,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    root: str | None = None,
    options: dict | None = None,
    out_path: Path | None = None,
) -> dict:
    ckpt = _load_ckpt(ckpt_path)
    cfg_d = ckpt["config"]
    cfg = TrainConfig(**cfg_d)

    root = root if root is not None else cfg.root
    options = options if options is not None else (cfg.options or None)
    _, val_ds, schema = prepare(
        dataset, context_length=cfg.context_length, horizon=cfg.horizon,
        stride=cfg.stride, val_ratio=cfg.val_ratio, seed=split_seed(cfg),
        root=root, options=options,
    )

    target_ae, covariate_ae = _load_codecs(cfg, schema)
    n_latent = target_ae.d_model
    embedder = LatentActionEmbedder(
        schema.cardinalities, schema.n_continuous, schema.n_exog,
        covariate_ae=covariate_ae)
    adapter = configure_latent_adapter(
        model, schema, embedder.cov_width, cfg.context_length, cfg.horizon, n_latent)

    state = ckpt["state"]
    adapter.model.load_state_dict(state["model"])
    embedder.load_state_dict(state["embedder"])
    target_ae.load_state_dict(state["target_ae"])
    if "fusion" in state and adapter.CovariateFusion is not None:
        adapter.CovariateFusion.load_state_dict(state["fusion"])

    dev = torch.device(device)
    to_device(adapter, embedder, target_ae, dev)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    metrics = evaluate(adapter, embedder, target_ae, val_loader, n_latent, schema.n_target, dev)

    from latent_space.jepa import JEPA_ARMS

    row = {
        "dataset": dataset, "model": model,
        "latent_scope": cfg.latent_scope, "codec": cfg.codec, "ae_mode": cfg.ae_mode,
        **({"jepa_alpha": cfg.jepa_alpha, "jepa_init": cfg.jepa_init}
           if cfg.codec in JEPA_ARMS else {}),
        "n_latent": n_latent, "n_target": schema.n_target, "cov_width": embedder.cov_width,
        "pure_series": is_pure_series(model),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "MAE": round(metrics["MAE"], 6),
        "MSE": round(metrics["MSE"], 6),
        "RMSE": round(metrics["RMSE"], 6),
        "val_windows": len(val_ds),
        "metric_space": "normalized[-1,1]",
        "checkpoint": str(ckpt_path),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"wrote {out_path}")
    return row


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained latent-space checkpoint")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ckpt-dir", default=None, help="dir holding {dataset}__{model}[__{codec}].pt")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --ckpt-dir)")
    p.add_argument("--codec", default="ae", choices=list(LATENT_ARMS),
                   help="latent arm; picks the __{codec} checkpoint suffix under --ckpt-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--root", default=None, help="override dataset root (default: ckpt's own)")
    p.add_argument("--split", default=None, help="override loader split")
    p.add_argument("--download", action="store_true")
    p.add_argument("--out", default=None, help="eval JSON output path")
    args = p.parse_args()

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif args.ckpt_dir:
        suffix = "" if args.codec == "ae" else f"__{args.codec}"
        ckpt_path = Path(args.ckpt_dir) / f"{args.dataset}__{args.model}{suffix}.pt"
    else:
        raise SystemExit("provide --ckpt or --ckpt-dir")

    options = None
    if args.split is not None or args.download:
        options = {}
        if args.split is not None:
            options["split"] = args.split
        if args.download:
            options["download"] = True

    row = eval_one(
        args.dataset, args.model, ckpt_path,
        device=args.device, batch_size=args.batch_size,
        root=args.root, options=options,
        out_path=Path(args.out) if args.out else None,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
