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
from latent_space.codecs import CODECS, StepwiseVAE
from latent_space.train_ae import SIGNALS, _signal_enc_in, _signal_window, train_ae


def _loaders(dataset, context_length, horizon, batch_size, val_ratio, seed, root, options):
    train_ds, val_ds, schema = prepare(
        dataset, context_length=context_length, horizon=horizon,
        val_ratio=val_ratio, seed=seed, root=root, options=options or None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, schema


def train_vae(dataset, *, signal, context_length, horizon, d_model, d_ff, beta,
              epochs, patience, batch_size, lr, val_ratio, seed, device, root,
              options, ckpt: Path | None) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_loader, val_loader, schema = _loaders(
        dataset, context_length, horizon, batch_size, val_ratio, seed, root, options)
    enc_in = _signal_enc_in(schema, signal)
    if enc_in == 0:
        raise ValueError(f"dataset {dataset!r} has no '{signal}' channels")
    d_ff = d_ff if d_ff is not None else 2 * d_model
    model = StepwiseVAE(enc_in, d_model, d_ff).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)

    def _batch_loss(batch):
        x = _signal_window(batch, signal, dev)
        recon, kl = model(x)
        rec = nn.functional.mse_loss(recon, x)
        return rec + beta * kl, rec, kl

    best_val = best_recon = float("inf")
    best_state, best_epoch, bad = None, -1, 0
    for epoch in range(epochs):
        model.train()
        run, seen = 0.0, 0
        for batch in train_loader:
            loss, _, _ = _batch_loss(batch)
            optim.zero_grad(); loss.backward(); optim.step()
            run += loss.item(); seen += 1
        model.eval()
        v_tot = v_rec = v_seen = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, rec, _ = _batch_loss(batch)
                v_tot += loss.item(); v_rec += rec.item(); v_seen += 1
        v_tot /= max(v_seen, 1); v_rec /= max(v_seen, 1)
        sched.step(v_tot)
        improved = v_tot < best_val - 1e-7
        if improved:
            best_val, best_recon, best_epoch, bad = v_tot, v_rec, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [VAE/{dataset}/{signal}] epoch {epoch + 1}/{epochs} "
              f"train={run / max(seen, 1):.6f} val={v_tot:.6f} (recon {v_rec:.6f})"
              f"{' *' if improved else ''}")
        if bad >= patience:
            print(f"  early stop at epoch {epoch + 1} (best {best_epoch + 1}, val={best_val:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "codec": "vae", "dataset": dataset, "signal": signal, "enc_in": enc_in,
        "d_model": d_model, "d_ff": d_ff, "use_revin": False, "beta": beta,
        "context_length": context_length, "horizon": horizon,
        "best_epoch": best_epoch + 1, "best_val": round(best_val, 6),
        "best_recon_val": round(best_recon, 6),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    if ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "meta": meta}, ckpt)
        print(f"wrote VAE checkpoint -> {ckpt}")
    print("VAE meta:", meta)
    return meta


def main():
    p = argparse.ArgumentParser(description="Pre-train a latent-target codec (ae|vae)")
    p.add_argument("--codec", default="ae", choices=list(CODECS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--signal", default="target", choices=list(SIGNALS))
    p.add_argument("--context-length", type=int, default=96)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--revin", action="store_true", help="AE only")
    p.add_argument("--beta", type=float, default=1e-3, help="VAE KL weight")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="mse", choices=["mse", "mae"], help="AE only")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()

    options = {}
    if args.split is not None:
        options["split"] = args.split
    if args.download:
        options["download"] = True
    ckpt = Path(args.ckpt) if args.ckpt else None
    common = dict(signal=args.signal, context_length=args.context_length,
                  horizon=args.horizon, d_model=args.d_model, d_ff=args.d_ff,
                  epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                  lr=args.lr, val_ratio=args.val_ratio, seed=args.seed,
                  device=args.device, root=args.root, options=options, ckpt=ckpt)

    if args.codec == "ae":
        train_ae(args.dataset, use_revin=args.revin, loss=args.loss, **common)
    else:
        train_vae(args.dataset, beta=args.beta, **common)


if __name__ == "__main__":
    main()
