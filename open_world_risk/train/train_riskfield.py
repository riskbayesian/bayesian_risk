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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import matplotlib.pyplot as plt
import petname as PetnamePkg
import wandb

from open_world_risk.utils.model import RiskField
from open_world_risk.src.dataset_loader import HistogramDataset

# --------------------- Config & Utils -------------------
@dataclass
class TrainConfig:
    tensor_dir: str
    out_dir: str

    # wandb
    project: str
    run_name: str
    entity: str | None

    # tensor layout
    feat_size: int
    hist_size: int
    ctrl_pts: int

    # model hyperparams (must match RiskField signature)
    projection_dim: int
    num_layers: int
    num_neurons: int
    num_layers_trans: int
    num_heads: int
    attn_dropout: float
    monotonic: bool
    compute_covariances: bool

    # training
    device: str
    seed: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    val_split: float
    num_workers: int
    amp: bool

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: dict) -> None:
    """
    Validate that required configuration parameters are present.
    """
    # required top-level keys
    required_top = ['tensor_dir', 'out_dir', 'wandb', 'tensor_layout', 'model', 'training']
    missing_top = [k for k in required_top if k not in config]
    if missing_top:
        raise ValueError(f"Missing required top-level keys in config: {missing_top}")

    # nested requirements
    required_wandb = ['project', 'run_name']  # 'entity' optional
    missing_wandb = [k for k in required_wandb if k not in config['wandb']]
    if missing_wandb:
        raise ValueError(f"Missing wandb keys: {missing_wandb}")

    required_layout = ['feat_size', 'hist_size', 'ctrl_pts']
    missing_layout = [k for k in required_layout if k not in config['tensor_layout']]
    if missing_layout:
        raise ValueError(f"Missing tensor_layout keys: {missing_layout}")

    required_model = ['projection_dim', 'num_layers', 'num_neurons', 'num_layers_trans', 'num_heads', 'attn_dropout', 'monotonic', 'compute_covariances']
    missing_model = [k for k in required_model if k not in config['model']]
    if missing_model:
        raise ValueError(f"Missing model keys: {missing_model}")

    required_training = ['device', 'seed', 'batch_size', 'epochs', 'lr', 'weight_decay', 'grad_clip', 'val_split', 'num_workers', 'amp']
    missing_training = [k for k in required_training if k not in config['training']]
    if missing_training:
        raise ValueError(f"Missing training keys: {missing_training}")

    # Validate tensor_dir exists
    if not os.path.exists(config['tensor_dir']):
        raise FileNotFoundError(f"Tensor directory does not exist: {config['tensor_dir']}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_hist(batch_targets_hist: torch.Tensor, output_dir: str = ".", prefix: str = "histogram"):
    """
    Plot histogram data from batch targets and save to local directory.
    
    Args:
        batch_targets_hist: Tensor of shape (batch_size, hist_size) containing histogram data
        output_dir: Directory to save the plot (default: current directory)
        prefix: Prefix for the saved plot filename (default: "histogram")
    """
    # Convert to numpy for plotting
    hist_data = batch_targets_hist.detach().cpu().numpy()
    
    # Create figure with subplots for each sample in the batch
    batch_size, hist_size = hist_data.shape
    n_cols = min(4, batch_size)  # Max 4 columns
    n_rows = int(np.ceil(batch_size / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle different cases for axes indexing
    if batch_size == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each histogram in the batch
    for i in range(batch_size):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Create bin centers (assuming 0 to 0.5 meters range with hist_size bins)
        max_distance = 0.5
        bin_edges = np.linspace(0, max_distance, hist_size + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = max_distance / hist_size
        
        # Plot histogram
        ax.bar(bin_centers, hist_data[i], width=bin_width, alpha=0.7, color='steelblue')
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_distance)
    
    # Hide empty subplots
    for i in range(batch_size, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    plt.suptitle(f'Histogram Batch (Size: {batch_size})', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{prefix}_batch_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Saved histogram plot to {plot_path}")

def load_training_config(config_path: str = "train_config.yaml") -> Tuple[TrainConfig, str]:
    """
    Load training configuration from YAML file and create TrainConfig object.
    
    Args:
        config_path: Path to the YAML configuration file (relative to script directory)
        
    Returns:
        Tuple[TrainConfig, str]: (TrainConfig object, resolved config file path)
    """
    # Resolve config path relative to script directory
    if not os.path.isabs(config_path):
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path
    
    # Load and validate configuration
    config = load_config(str(config_path))
    validate_config(config)
    
    # Extract nested config values
    wandb_config = config['wandb']
    tensor_layout = config['tensor_layout']
    model_config = config['model']
    training_config = config['training']
    
    # Create TrainConfig from YAML configuration
    cfg = TrainConfig(
        tensor_dir=config['tensor_dir'],
        out_dir=config['out_dir'],
        project=wandb_config['project'],
        run_name=wandb_config['run_name'],
        entity=wandb_config.get('entity'),
        feat_size=tensor_layout['feat_size'],
        hist_size=tensor_layout['hist_size'],
        ctrl_pts=tensor_layout['ctrl_pts'],
        projection_dim=model_config['projection_dim'],
        num_layers=model_config['num_layers'],
        num_neurons=model_config['num_neurons'],
        num_layers_trans=model_config['num_layers_trans'],
        num_heads=model_config['num_heads'],
        attn_dropout=model_config['attn_dropout'],
        monotonic=model_config['monotonic'],
        compute_covariances=model_config['compute_covariances'],
        device=training_config['device'],
        seed=training_config['seed'],
        batch_size=training_config['batch_size'],
        epochs=training_config['epochs'],
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        grad_clip=training_config['grad_clip'],
        val_split=training_config['val_split'],
        num_workers=training_config['num_workers'],
        amp=training_config['amp'],
    )
    return cfg, str(config_path)

# ---------------------- Training Loop -------------------
def train(cfg: TrainConfig, config_file_path: str):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    now = datetime.now()
    date = now.strftime("%m_%d_%Y")
    epoch = str(int(now.timestamp()))
    petname = PetnamePkg.Generate()
    run_dir = os.path.join(cfg.out_dir, f"{date}_{epoch}_{petname}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        shutil.copy2(config_file_path, os.path.join(run_dir, "train_config.yaml"))
        print(f"✅ Copied train_config.yaml to run directory: {run_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to copy config file to run directory: {e}")

    run_name = f"{date}_{epoch}_{petname}"
    with open(os.path.join(run_dir, "model_run"), "w") as f:
        f.write(run_name + "\n")

    tensor_pattern = os.path.join(cfg.tensor_dir, "histogram_tensor*.pt")
    tensor_paths = glob.glob(tensor_pattern)    
    if not tensor_paths:
        raise FileNotFoundError(f"No .pt files found in directory: {cfg.tensor_dir}")
    
    print(f"Found {len(tensor_paths)} tensor files:")
    for path in tensor_paths:
        print(f"  - {path}")
    
    dataset = HistogramDataset(tensor_paths, ctrl_pts=cfg.ctrl_pts, feat_size=cfg.feat_size)
    K = cfg.ctrl_pts  # number of Bezier coefficients
    print(f"Loaded dataset: {len(dataset)} samples | Control points = {K}")

    N = len(dataset)
    n_val = max(1, int(math.floor(cfg.val_split * N)))
    n_train = N - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    dl_train = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                          collate_fn=dataset.collate_fn)
    dl_val = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                        collate_fn=dataset.collate_fn)

    model = RiskField(
        clip_dim=cfg.feat_size,             # DINO feat dim per object
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
    wandb_run_name = f"{cfg.run_name}-{petname}" if cfg.run_name else petname
    wandb.init(project=cfg.project, name=wandb_run_name, entity=cfg.entity, config=asdict(cfg))
    wandb.watch(model, log="all", log_freq=200)
    
    # Log dataset information once at the beginning
    wandb.log({
        "dataset/total_size": N,
        "dataset/train_size": n_train,
        "dataset/val_size": n_val,
        "dataset/ctrl_pts": K,
    })

    best_val = float("inf")
    ckpt_best = os.path.join(run_dir, f"riskfield_best_{date}_{epoch}_{petname}.pt")
    ckpt_last = os.path.join(run_dir, f"riskfield_last_{date}_{epoch}_{petname}.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in dl_train:
            featA = batch_inputs['dino1'].to(device, non_blocking=True)
            featB = batch_inputs['dino2'].to(device, non_blocking=True)
            y = batch_targets['control_points'].to(device, non_blocking=True)

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
            for batch_inputs, batch_targets in dl_val:
                featA = batch_inputs['dino1'].to(device, non_blocking=True)
                featB = batch_inputs['dino2'].to(device, non_blocking=True)
                y = batch_targets['control_points'].to(device, non_blocking=True)

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

if __name__ == "__main__":
    cfg, cfg_path = load_training_config()
    train(cfg, cfg_path)
