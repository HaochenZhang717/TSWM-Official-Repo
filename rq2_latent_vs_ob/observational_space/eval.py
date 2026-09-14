from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_RQ2_ROOT = Path(__file__).resolve().parents[1]
if str(_RQ2_ROOT) not in sys.path:
    sys.path.insert(0, str(_RQ2_ROOT))

from common.action_embedder import ActionEmbedder
from observational_space.config_builder import configure_adapter, is_pure_series
from common.datasets import collate, prepare
from observational_space.engine import evaluate, to_device
from observational_space.train import split_seed


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
    cfg = ckpt["config"]

    root = root if root is not None else cfg.get("root")
    options = options if options is not None else (cfg.get("options") or None)
    _, val_ds, schema = prepare(
        dataset,
        context_length=cfg["context_length"], horizon=cfg["horizon"],
        stride=cfg.get("stride"), val_ratio=cfg["val_ratio"], seed=split_seed(cfg),
        root=root, options=options,
    )

    embedder = ActionEmbedder(schema.cardinalities, schema.n_continuous, schema.n_exog)
    adapter = configure_adapter(
        model, schema, embedder.cov_width, cfg["context_length"], cfg["horizon"])

    state = ckpt["state"]
    adapter.model.load_state_dict(state["model"])
    embedder.load_state_dict(state["embedder"])
    if "fusion" in state and adapter.CovariateFusion is not None:
        adapter.CovariateFusion.load_state_dict(state["fusion"])

    dev = torch.device(device)
    to_device(adapter, embedder, dev)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    metrics = evaluate(adapter, embedder, val_loader, schema.n_target, dev)

    row = {
        "dataset": dataset, "model": model,
        "n_target": schema.n_target, "cov_width": embedder.cov_width,
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
    p = argparse.ArgumentParser(description="Evaluate a trained action-conditioned checkpoint")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ckpt-dir", default=None, help="dir holding {dataset}__{model}.pt")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --ckpt-dir)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--root", default=None, help="override dataset root (default: ckpt's own)")
    p.add_argument("--split", default=None, help="override loader split")
    p.add_argument("--download", action="store_true", help="download dataset if missing (vitaldb)")
    p.add_argument("--out", default=None, help="eval JSON output path")
    args = p.parse_args()

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif args.ckpt_dir:
        ckpt_path = Path(args.ckpt_dir) / f"{args.dataset}__{args.model}.pt"
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
