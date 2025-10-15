#!/usr/bin/env python3
"""
posterior_from_folders.py

Given two folders of grayscale PNGs (same filenames):
  - prior_gray/       (values 0..255 → normalized to [0,1])
  - likelihood_gray/  (values 0..255 → normalized to [0,1])

Compute per-pixel product: posterior = prior * likelihood (in [0,1]).
Also compute a colored visualization of 1 - posterior and save it.

Outputs (at the same level as the inputs by default):
  - posterior_gray/      (uint8 grayscale, product*255)
  - posterior_colored/   (RGB colored, colormap applied to (1 - product))

Usage:
  python posterior_from_folders.py                # reads CONFIG env or ./posterior_config.yaml
  CONFIG=/path/to/posterior_config.yaml python posterior_from_folders.py
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import argparse

# Optional: matplotlib for colormap
try:
    import matplotlib.cm as cm
except Exception:
    cm = None

def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(explicit: str | None = None) -> Tuple[dict, Path]:
    """
    Load YAML config. If explicit is provided, load that path.
    Else load from CONFIG env or ./posterior_config.yaml (next to script).
    Returns (cfg, cfg_dir).
    """
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
    else:
        envp = os.environ.get("CONFIG")
        cfg_path = Path(envp).expanduser().resolve() if envp else (Path(__file__).parent / "posterior_config.yaml").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    return load_yaml(cfg_path), cfg_path.parent

def list_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"], key=lambda p: p.name)

def to_float01(img_u8: np.ndarray) -> np.ndarray:
    # img_u8: uint8 HxW or HxW(,1)
    if img_u8.ndim == 3 and img_u8.shape[2] == 1:
        img_u8 = img_u8[..., 0]
    return img_u8.astype(np.float32) / 255.0


def save_u8_gray(arr01: np.ndarray, path: Path) -> None:
    # arr01 in [0,1], save as uint8 PNG
    u8 = np.clip(arr01 * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(u8).save(path)


def colorize_01(arr01: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Apply matplotlib colormap (H,W) -> (H,W,3) uint8.
    """
    if cm is None:
        raise RuntimeError("matplotlib is required for colormap. Install matplotlib or set visualization.colormap to '' to skip.")
    m = cm.get_cmap(cmap_name)
    rgb = (m(arr01)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser(description="Compute posterior = prior * likelihood from two grayscale folders.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config. Alternatively set CONFIG env.")
    args = ap.parse_args()

    cfg, cfg_dir = load_config(args.config)

    paths = cfg.get("paths", {})
    vis   = cfg.get("visualization", {})
    save  = cfg.get("save", {})

    # Resolve folders (relative to config directory if not absolute)
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (cfg_dir / p).resolve()

    prior_dir = _resolve(paths["prior_gray"])
    like_dir  = _resolve(paths["likelihood_gray"])

    if not prior_dir.exists() or not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_gray not found or not a directory: {prior_dir}")
    if not like_dir.exists() or not like_dir.is_dir():
        raise FileNotFoundError(f"likelihood_gray not found or not a directory: {like_dir}")

    # Output root: default to common parent of the two input dirs
    out_root = _resolve(paths.get("output_root", "")) if paths.get("output_root") else prior_dir.parent
    posterior_gray_dir    = out_root / "posterior_gray"
    posterior_colored_dir = out_root / "posterior_colored"

    posterior_gray_dir.mkdir(parents=True, exist_ok=True)
    posterior_colored_dir.mkdir(parents=True, exist_ok=True)

    # Colormap name (for 1 - product)
    cmap_name = vis.get("colormap", "viridis")
    save_colored = bool(vis.get("save_colored", True))

    # Strict matching set of filenames
    prior_files = list_pngs(prior_dir)
    like_files  = list_pngs(like_dir)

    prior_names = {p.name for p in prior_files}
    like_names  = {p.name for p in like_files}

    inter = sorted(prior_names & like_names)
    missing_prior = sorted(like_names - prior_names)
    missing_like  = sorted(prior_names - like_names)

    if missing_prior:
        print(f"[WARN] {len(missing_prior)} files present in likelihood but missing in prior (skipping):")
        print("       " + ", ".join(missing_prior[:10]) + (" ..." if len(missing_prior) > 10 else ""))
    if missing_like:
        print(f"[WARN] {len(missing_like)} files present in prior but missing in likelihood (skipping):")
        print("       " + ", ".join(missing_like[:10]) + (" ..." if len(missing_like) > 10 else ""))

    if not inter:
        raise RuntimeError("No matching filenames between prior_gray and likelihood_gray.")

    overwrite = bool(save.get("overwrite", True))

    # Process each pair
    for name in inter:
        p_path = prior_dir / name
        l_path = like_dir / name

        # Load as grayscale
        p_img = Image.open(p_path).convert("L")
        l_img = Image.open(l_path).convert("L")

        # Ensure same size; if not, resize likelihood to prior
        if p_img.size != l_img.size:
            l_img = l_img.resize(p_img.size, resample=Image.BILINEAR)

        p_np = np.array(p_img, dtype=np.uint8)
        l_np = np.array(l_img, dtype=np.uint8)

        p01 = to_float01(p_np)
        l01 = to_float01(l_np)

        # Posterior = product
        post01 = np.clip(p01 * l01, 0.0, 1.0)

        # Save grayscale posterior
        out_gray = posterior_gray_dir / name
        if overwrite or not out_gray.exists():
            save_u8_gray(post01, out_gray)

        # Save colored visualization of (1 - posterior)
        if save_colored:
            inv01 = 1.0 - post01
            colored = colorize_01(inv01, cmap_name)
            out_col = posterior_colored_dir / name
            if overwrite or not out_col.exists():
                Image.fromarray(colored).save(out_col)

    print(f"[OK] Wrote posterior to:\n  {posterior_gray_dir}\n  {posterior_colored_dir if save_colored else '(colored disabled)'}")

if __name__ == "__main__":
    main()
