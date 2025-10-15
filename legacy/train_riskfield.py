#!/usr/bin/env python3
# train_riskfield.py
import argparse
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import wandb
from open_world_risk.utils.model import RiskField
# ----------------------- Dataset -----------------------
class PairTensorDataset(Dataset):
    """
    Expects a tensor with layout:
      [ featA(FA) | featB(FB) | bins(100) | coeffs(K) ]
    We supervise ONLY on coeffs(K). Inputs are (featA, featB).
    """
    def __init__(self, tensor: torch.Tensor, featA: int, featB: int, bins: int):
        super().__init__()
        self.T = tensor
        self.FA = featA
        self.FB = featB
        self.BINS = bins

        N, M = tensor.shape
        expected_prefix = featA + featB + bins
        if M <= expected_prefix:
            raise ValueError(f"Tensor has {M} dims, need > {expected_prefix} to include coeffs(K).")

        self.K = M - expected_prefix
        self.idxA0, self.idxA1 = 0, featA
        self.idxB0, self.idxB1 = self.idxA1, self.idxA1 + featB
        self.idxP0, self.idxP1 = self.idxB1, self.idxB1 + bins
        self.idxC0, self.idxC1 = self.idxP1, M

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, i):
        row = self.T[i]
        featA = row[self.idxA0:self.idxA1]
        featB = row[self.idxB0:self.idxB1]
        coeffs = row[self.idxC0:self.idxC1]
        # Return float32 tensors
        return featA.to(torch.float32), featB.to(torch.float32), coeffs.to(torch.float32)

# --------------------- Config & Utils -------------------
@dataclass
class TrainConfig:
    pair_tensor_path: str
    out_dir: str = "./checkpoints_riskfield"
    project: str = "riskfield"
    run_name: str = "rf_bezier_supervision"
    entity: str | None = None

    # tensor layout
    featA: int = 1024
    featB: int = 1024
    bins: int = 100

    # model hyperparams (must match RiskField signature)
    projection_dim: int = 256
    num_layers: int = 2
    num_neurons: int = 512
    num_layers_trans: int = 2
    num_heads: int = 8
    attn_dropout: float = 0.1
    monotonic: bool = True
    compute_covariances: bool = False

    # training
    device: str = "cuda"
    seed: int = 1337
    batch_size: int = 128
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_split: float = 0.1
    num_workers: int = 4
    amp: bool = True

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------- Training Loop -------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load pair tensor
    full_T = torch.load(cfg.pair_tensor_path, map_location="cpu")
    if not (isinstance(full_T, torch.Tensor) and full_T.ndim == 2):
        raise RuntimeError("pair_tensor_with_bezier.pt must be a 2D torch.Tensor")

    dataset = PairTensorDataset(full_T, cfg.featA, cfg.featB, cfg.bins)
    K = dataset.K  # number of Bezier coefficients
    print(f"Loaded tensor: {tuple(full_T.shape)} | Inferred K (coeff dims) = {K}")

    # Split
    N = len(dataset)
    n_val = max(1, int(math.floor(cfg.val_split * N)))
    n_train = N - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    dl_train = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    dl_val = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # Instantiate model
    model = RiskField(
        clip_dim=cfg.featA,                 # DINO feat dim per object
        num_layers=cfg.num_layers,
        num_neurons=cfg.num_neurons,
        projection_dim=cfg.projection_dim,
        alphas_dim=K,                       # supervise to K Bezier coeffs
        monotonic=cfg.monotonic,
        compute_covariances=cfg.compute_covariances,
        num_layers_trans=cfg.num_layers_trans,
        num_heads=cfg.num_heads,
        attn_dropout=cfg.attn_dropout
    )

    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu" and cfg.amp))

    # ----- Weights & Biases -----
    wandb.init(project=cfg.project, name=cfg.run_name, entity=cfg.entity, config=asdict(cfg))
    wandb.watch(model, log="all", log_freq=200)

    best_val = float("inf")
    ckpt_best = os.path.join(cfg.out_dir, "riskfield_best.pt")
    ckpt_last = os.path.join(cfg.out_dir, "riskfield_last.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for featA, featB, coeffs in dl_train:
            featA = featA.to(device, non_blocking=True)
            featB = featB.to(device, non_blocking=True)
            y = coeffs.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device != "cpu" and cfg.amp)):
                pred, _ = model(featA, featB)  # (B, K)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            train_loss += loss.item() * featA.size(0)

        train_loss /= n_train
        sched.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for featA, featB, coeffs in dl_val:
                featA = featA.to(device, non_blocking=True)
                featB = featB.to(device, non_blocking=True)
                y = coeffs.to(device, non_blocking=True)

                pred, _ = model(featA, featB)
                loss = criterion(pred, y)
                val_loss += loss.item() * featA.size(0)
        val_loss /= n_val

        wandb.log({
            "epoch": epoch,
            "train/loss_mse": train_loss,
            "val/loss_mse": val_loss,
            "lr": sched.get_last_lr()[0],
        })

        print(f"[{epoch:03d}/{cfg.epochs}] train={train_loss:.6f} | val={val_loss:.6f} | lr={sched.get_last_lr()[0]:.3e}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "cfg": asdict(cfg),
                        "epoch": epoch,
                        "best_val": best_val,
                        "K": K},
                       ckpt_best)

        # Always save last
        torch.save({"model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "K": K},
                   ckpt_last)

    print(f"Done. Best val loss: {best_val:.6f}")
    wandb.finish()

def parse_args():
    p = argparse.ArgumentParser("Train RiskField to predict Bezier coefficients from DINO features.")
    p.add_argument("--pair_tensor", required=True, help="Path to pair_tensor_with_bezier.pt")
    p.add_argument("--out_dir", default="./checkpoints_riskfield")
    p.add_argument("--project", default="riskfield")
    p.add_argument("--run_name", default="rf_bezier_supervision")
    p.add_argument("--entity", default=None)

    # layout
    p.add_argument("--featA", type=int, default=1024)
    p.add_argument("--featB", type=int, default=1024)
    p.add_argument("--bins", type=int, default=100)

    # model hparams
    p.add_argument("--projection_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_neurons", type=int, default=512)
    p.add_argument("--num_layers_trans", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--attn_dropout", type=float, default=0.1)
    p.add_argument("--no_monotonic", action="store_true")
    p.add_argument("--compute_covariances", action="store_true")

    # training
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        pair_tensor_path=args.pair_tensor,
        out_dir=args.out_dir,
        project=args.project,
        run_name=args.run_name,
        entity=args.entity,
        featA=args.featA,
        featB=args.featB,
        bins=args.bins,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        num_neurons=args.num_neurons,
        num_layers_trans=args.num_layers_trans,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        monotonic=not args.no_monotonic,
        compute_covariances=args.compute_covariances,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
