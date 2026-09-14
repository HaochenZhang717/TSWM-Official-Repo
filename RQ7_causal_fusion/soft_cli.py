from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "RQ4_action_encoding", _REPO / "RQ3_fusion_benchmark",
           _REPO / "rq2_latent_vs_ob", _REPO / "rq2_latent_vs_ob" / "common",
           _REPO / "RQ7_causal_fusion"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from encode_train import DATASET_OPTS, dataset_root
from latent_space.train import TrainConfig

from soft_train import RHO_DEFAULT, SoftConstraintTrainer

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--rho", type=float, default=RHO_DEFAULT)
    a.add_argument("--prior-split", default="default")
    a.add_argument("--codec", default="ae")
    a.add_argument("--ae-dir", required=True)
    a.add_argument("--data-root", required=True)
    a.add_argument("--cache-dir", required=True)
    a.add_argument("--ckpt-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cuda")
    a.add_argument("--context-length", type=int, default=256)
    a.add_argument("--horizon", type=int, default=16)
    a.add_argument("--stride", type=int, default=80)
    a.add_argument("--epochs", type=int, default=100)
    a.add_argument("--patience", type=int, default=20)
    a.add_argument("--batch-size", type=int, default=128)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--lr-patience", type=int, default=5)
    a.add_argument("--loss", default="mse")
    a.add_argument("--grad-clip", type=float, default=1.0)
    a.add_argument("--val-ratio", type=float, default=0.2)
    a.add_argument("--data-seed", type=int, default=0)
    args = a.parse_args()

    cfg = TrainConfig(
        dataset=args.dataset, model=args.model, codec=args.codec,
        ae_dir=args.ae_dir, ae_mode="frozen", latent_scope="target",
        context_length=args.context_length, horizon=args.horizon, stride=args.stride,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        lr=args.lr, lr_patience=args.lr_patience, loss=args.loss,
        grad_clip=args.grad_clip, val_ratio=args.val_ratio,
        seed=args.seed, data_seed=args.data_seed, device=args.device,
        root=dataset_root(args.data_root, None, args.dataset),
        options=DATASET_OPTS.get(args.dataset), cache_dir=args.cache_dir)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    _tag = "gate_soft"
    ck = Path(args.ckpt_dir) / f"{args.dataset}__{args.model}__instant__{_tag}.pt"
    blob = {"rows": [], "failures": {}}
    try:
        tr = SoftConstraintTrainer(cfg, prior_split=args.prior_split, rho=args.rho,
                                   seed=args.seed)
        blob["rows"].append(tr.fit(ckpt_path=ck))
    except Exception as e:
        blob["failures"][f"{args.dataset}/{args.model}/{_tag}"] = repr(e)
        traceback.print_exc()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(blob, indent=1, default=float))
    print(f"wrote {args.out}")
