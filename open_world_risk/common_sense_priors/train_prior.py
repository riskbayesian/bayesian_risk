#!/usr/bin/env python3
# train_riskfield.py
import glob
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
import shutil
from sqlite3.dbapi2 import Timestamp
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import matplotlib.pyplot as plt
import petname as PetnamePkg
import wandb

from open_world_risk.common_sense_priors.model_prior import PriorModel
from open_world_risk.common_sense_priors.dataset_prior import PriorDataset


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dataset(tensor_dir, distribution_stats_path, train_cfg):
    tensor_pattern = os.path.join(tensor_dir, "*.pt")
    tensor_paths = glob.glob(tensor_pattern)    
    if not tensor_paths:
        raise FileNotFoundError(f"No .pt files found in directory: {tensor_dir}")
    
    print(f"Found {len(tensor_paths)} tensor files:")
    for path in tensor_paths:
        print(f"  - {path}")
    
    dataset = PriorDataset(tensor_paths, distribution_stats_path, feat_size=train_cfg["feat_size"])

    N = len(dataset)
    n_val = max(1, int(math.floor(train_cfg["val_split"] * N)))
    n_train = N - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(train_cfg["seed"]))

    dl_train = dataset.get_balanced_dataloader(
        batch_size=train_cfg["batch_size"],
        subset_indices=train_set.indices,
        num_workers=train_cfg["num_workers"], 
        pin_memory=True, 
        drop_last=False
    )

    dl_val = dataset.get_balanced_dataloader(
        batch_size=train_cfg["batch_size"],
        subset_indices=val_set.indices,
        num_workers=train_cfg["num_workers"], 
        pin_memory=True, 
        drop_last=False
    )

    return dl_train, dl_val

def setup_weights_and_biases(model, petname, wandb_cfg, dl_train, dl_val):
    wandb_run_name = f"{wandb_cfg['run_name']}-{petname}" if wandb_cfg['run_name'] else petname
    wandb.init(project=wandb_cfg["project"], name=wandb_run_name, entity=wandb_cfg["entity"])
    wandb.watch(model, log="all", log_freq=200)
    
    # Log dataset information once at the beginning
    wandb.log({
        "dataset/total_size": len(dl_train.dataset) + len(dl_val.dataset),
        "dataset/train_size": len(dl_train.dataset),
        "dataset/val_size": len(dl_val.dataset),
    })

    return wandb_run_name

def setup_model(train_cfg, model_cfg):
    model = PriorModel(
        clip_dim=train_cfg["feat_size"],
        num_layers=model_cfg["num_layers"],
        num_neurons=model_cfg["num_neurons"],
        projection_dim=model_cfg["projection_dim"],
        alphas_dim=model_cfg["ctrl_pts"],                   
        monotonic=model_cfg["monotonic"],
        num_layers_trans=model_cfg["num_layers_trans"],
        num_heads=model_cfg["num_heads"],
        attn_dropout=model_cfg["attn_dropout"]
    )
    device = "cuda"
    model = model.to(device)

    return model 

def train(dl_train, dl_val, model, train_cfg, petname, run_name, run_dir, device="cuda"):
    n_total = len(dl_train.dataset) + len(dl_val.dataset)
    n_train = len(dl_train.dataset)
    n_val   = len(dl_val.dataset)

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg["epochs"])
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu" and train_cfg["amp"]))

    best_val = float("inf")
    ckpt_best = os.path.join(run_dir, f"riskfield_best_{run_name}.pt")
    ckpt_last = os.path.join(run_dir, f"riskfield_last_{run_name}.pt")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in dl_train:
            featA = batch_inputs['dino1'].to(device, non_blocking=True)
            featB = batch_inputs['dino2'].to(device, non_blocking=True)
            y = batch_targets['prior'].to(device, non_blocking=True)  # (B, 1)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device != "cpu" and train_cfg["amp"])):
                pred = model(featA, featB)  # (B, K)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            if train_cfg["grad_clip"] is not None and train_cfg["grad_clip"] > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optim)
            scaler.update()

            train_loss += loss.item() * featA.size(0)

        train_loss /= n_train
        sched.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in dl_val:
                featA = batch_inputs['dino1'].to(device, non_blocking=True)
                featB = batch_inputs['dino2'].to(device, non_blocking=True)
                y = batch_targets['prior'].to(device, non_blocking=True)  # (B, 1)

                pred = model(featA, featB)
                loss = criterion(pred, y)
                val_loss += loss.item() * featA.size(0)
        val_loss /= n_val

        wandb.log({
            "epoch": epoch,
            "train/loss_mse": train_loss,
            "val/loss_mse": val_loss,
            "lr": sched.get_last_lr()[0],
        })

        print(f"[{epoch:03d}/{train_cfg['epochs']}] train={train_loss:.6f} | val={val_loss:.6f} | lr={sched.get_last_lr()[0]:.3e}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "cfg": train_cfg,
                        "epoch": epoch,
                        "best_val": best_val},
                       ckpt_best)

        # Always save last
        torch.save({"model": model.state_dict(),
                    "cfg": train_cfg,
                    "epoch": epoch,
                    "val_loss": val_loss},
                   ckpt_last)

    print(f"Done. Best val loss: {best_val:.6f}")
    wandb.finish()

if __name__ == "__main__":
    cfg_path = "train_prior_config.yaml"
    if not os.path.isabs(cfg_path):
        script_dir = Path(__file__).parent
        cfg_path = script_dir / cfg_path
    cfg = load_config(cfg_path)

    train_cfg = cfg["training"]
    wandb_cfg = cfg["wandb"]
    model_cfg = cfg["model"]

    set_seed(cfg["training"]["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    now = datetime.now()
    date = now.strftime("%m_%d_%Y")
    timestamp_str = str(int(now.timestamp()))
    petname = PetnamePkg.Generate()
    run_dir = os.path.join(cfg["out_dir"], f"{date}_{timestamp_str}_{petname}")
    os.makedirs(run_dir, exist_ok=True)
    run_name = f"{date}_{timestamp_str}_{petname}"
    with open(os.path.join(run_dir, "model_run"), "w") as f:
        f.write(run_name + "\n")

    try:
        shutil.copy2(cfg_path, os.path.join(run_dir, "train_prior_config.yaml"))
        print(f"✅ Copied train_prior_config.yaml to run directory: {run_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to copy config file to run directory: {e}")

    dl_train, dl_val = create_dataset(cfg["tensor_dir"], cfg["distribution_stats_path"], train_cfg)
    model = setup_model(train_cfg, model_cfg)
    setup_weights_and_biases(model, petname, wandb_cfg, dl_train, dl_val)
    train(dl_train, dl_val, model, train_cfg, petname, run_name, run_dir)