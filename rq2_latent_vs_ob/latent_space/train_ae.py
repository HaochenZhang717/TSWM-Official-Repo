from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_RQ2_ROOT = Path(__file__).resolve().parents[1]
for _p in (_RQ2_ROOT, _RQ2_ROOT / "common", _RQ2_ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.datasets import collate, prepare
from latent_space.autoencoder import build_autoencoder

SIGNALS = {
    "target":     [("target_history", "target_future", "n_target")],
    "continuous": [("continuous_history", "continuous_future", "n_continuous")],
    "exog":       [("exog_history", "exog_future", "n_exog")],
    "covariate":  [("continuous_history", "continuous_future", "n_continuous"),
                   ("exog_history", "exog_future", "n_exog")],
}


def _signal_enc_in(schema, signal: str) -> int:
    return sum(getattr(schema, attr) for _, _, attr in SIGNALS[signal])


def _signal_window(batch: dict, signal: str, device: torch.device) -> torch.Tensor:
    groups = []
    for h_key, f_key, _ in SIGNALS[signal]:
        h = batch[h_key].to(device)
        f = batch[f_key].to(device)
        groups.append(torch.cat([h, f], dim=1))
    return torch.cat(groups, dim=-1) if len(groups) > 1 else groups[0]


def _make_criterion(name: str) -> nn.Module:
    return {"mse": nn.MSELoss(), "mae": nn.L1Loss()}[name]


@torch.no_grad()
def _eval(model, loader, signal, criterion, device) -> float:
    model.eval()
    total, seen = 0.0, 0
    for batch in loader:
        x = _signal_window(batch, signal, device)
        total += criterion(model(x), x).item()
        seen += 1
    model.train()
    return total / max(seen, 1)


def train_ae(
    dataset: str, *, signal: str, context_length: int, horizon: int, d_model: int,
    d_ff: int | None, use_revin: bool, epochs: int, patience: int, batch_size: int,
    lr: float, loss: str, val_ratio: float, seed: int, device: str, root: str | None,
    options: dict | None, ckpt: Path | None,
) -> dict:
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose from {list(SIGNALS)}")
    torch.manual_seed(seed)
    dev = torch.device(device)

    train_ds, val_ds, schema = prepare(
        dataset, context_length=context_length, horizon=horizon,
        val_ratio=val_ratio, seed=seed, root=root, options=options or None,
    )
    enc_in = _signal_enc_in(schema, signal)
    if enc_in == 0:
        raise ValueError(f"dataset {dataset!r} has no '{signal}' channels (n={enc_in}); nothing to train")

    model = build_autoencoder(enc_in, d_model, d_ff, use_revin=use_revin).to(dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    criterion = _make_criterion(loss)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            x = _signal_window(batch, signal, dev)
            loss_val = criterion(model(x), x)
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            running += loss_val.item()
            seen += 1
        val = _eval(model, val_loader, signal, criterion, dev)
        sched.step(val)
        improved = val < best_val - 1e-7
        if improved:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [AE/{dataset}/{signal}] epoch {epoch + 1}/{epochs} "
              f"train={running / max(seen, 1):.6f} val={val:.6f}{' *' if improved else ''}")
        if bad >= patience:
            print(f"  early stop at epoch {epoch + 1} (best epoch {best_epoch + 1}, val={best_val:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "dataset": dataset, "signal": signal, "enc_in": enc_in, "d_model": d_model,
        "d_ff": d_ff if d_ff is not None else 2 * d_model, "use_revin": use_revin,
        "context_length": context_length, "horizon": horizon,
        "best_epoch": best_epoch + 1, "best_val": round(best_val, 6),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    if ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "meta": meta}, ckpt)
        print(f"wrote AE checkpoint -> {ckpt}")
    print("AE meta:", meta)
    return meta


def main():
    p = argparse.ArgumentParser(description="Pre-train a step-wise AutoEncoder on one signal")
    p.add_argument("--dataset", required=True)
    p.add_argument("--signal", default="target", choices=list(SIGNALS))
    p.add_argument("--context-length", type=int, default=96)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--d-model", type=int, default=32, help="latent channel width")
    p.add_argument("--d-ff", type=int, default=None, help="hidden width (default 2*d_model)")
    p.add_argument("--revin", action="store_true", help="use RevIN inside the AE")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="mse", choices=["mse", "mae"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default=None, help="dataset root path (overrides DEFAULT_ROOTS)")
    p.add_argument("--split", default=None, help="loader split (vitaldb=all, shanghai=T2DM, ...)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--ckpt", default=None, help="output checkpoint path")
    args = p.parse_args()

    options = {}
    if args.split is not None:
        options["split"] = args.split
    if args.download:
        options["download"] = True

    train_ae(
        args.dataset, signal=args.signal, context_length=args.context_length, horizon=args.horizon,
        d_model=args.d_model, d_ff=args.d_ff, use_revin=args.revin,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, loss=args.loss, val_ratio=args.val_ratio, seed=args.seed,
        device=args.device, root=args.root, options=options,
        ckpt=Path(args.ckpt) if args.ckpt else None,
    )


if __name__ == "__main__":
    main()
