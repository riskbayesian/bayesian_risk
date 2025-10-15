#!/usr/bin/env python3
"""
combine_likelihood_and_prior_tree.py

Iterates a parent directory for multiple objects. For each object name in the YAML,
expects sibling folders:

  likelihood_{object}/greyscale/*.png
  prior_{object}/risk_gray/*.png

Computes posterior = prior * likelihood (in [0,1]) for filename matches and writes:

  posterior_{object}/posterior_gray/<file>.png
  posterior_{object}/posterior_colored/<file>.png   (colormap on 1 - posterior)

Notes:
- If sizes differ, likelihood is resized to the prior image size.
- Object folder names use spaces->underscores (config list can have natural names).
- Overwrite behavior & colormap are configurable in YAML.

Usage:
  python combine_likelihood_and_prior_tree.py --config /path/to/config.yaml
  # or set CONFIG=/path/to/config.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

# Optional colormap via matplotlib
try:
    import matplotlib.cm as cm
except Exception:
    cm = None

# ---------------- YAML ----------------
def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(explicit: str | None = None) -> Tuple[dict, Path]:
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
    else:
        envp = os.environ.get("CONFIG")
        cfg_path = Path(envp).expanduser().resolve() if envp else (Path(__file__).parent / "combine_tree_config.yaml").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    return load_yaml(cfg_path), cfg_path.parent


# ---------------- Utils ----------------
def sanitize_object_name(name: str) -> str:
    return name.strip().replace(" ", "_")

def list_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"], key=lambda p: p.name)

def to_float01(u8: np.ndarray) -> np.ndarray:
    if u8.ndim == 3 and u8.shape[2] == 1:
        u8 = u8[..., 0]
    return u8.astype(np.float32) / 255.0

def save_u8_gray(arr01: np.ndarray, path: Path) -> None:
    u8 = np.clip(arr01 * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(u8).save(path)

def colorize(arr01: np.ndarray, cmap_name: str) -> np.ndarray:
    if cm is None:
        raise RuntimeError("matplotlib is required for colored output. Install matplotlib or set visualization.save_colored=false.")
    m = cm.get_cmap(cmap_name)
    rgb = (m(arr01)[..., :3] * 255.0).astype(np.uint8)
    return rgb


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch combine likelihood & prior across objects under a parent directory.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config (or set CONFIG env).")
    args = ap.parse_args()

    cfg, cfg_dir = load_config(args.config)
    paths = cfg.get("paths", {})
    vis   = cfg.get("visualization", {})
    save  = cfg.get("save", {})

    # Resolve parent dir
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (cfg_dir / p).resolve()

    parent_dir = _resolve(paths["parent_dir"])
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise FileNotFoundError(f"parent_dir not found or not a directory: {parent_dir}")

    objects: List[str] = cfg.get("objects", [])
    if not objects:
        raise ValueError("Config must include a non-empty 'objects' list.")

    like_prefix = paths.get("likelihood_prefix", "likelihood_")
    prior_prefix = paths.get("prior_prefix", "prior_")
    like_gray_sub = paths.get("likelihood_gray_subdir", "greyscale")
    prior_gray_sub = paths.get("prior_gray_subdir", "risk_gray")

    # Output dirs: posterior_{object}/posterior_gray + posterior_colored
    post_prefix = paths.get("posterior_prefix", "posterior_")
    save_colored = bool(vis.get("save_colored", True))
    cmap_name = vis.get("colormap", "viridis")
    overwrite = bool(save.get("overwrite", True))

    for obj in objects:
        obj_sanitized = sanitize_object_name(obj)

        like_dir = parent_dir / f"{like_prefix}{obj_sanitized}" / like_gray_sub
        prior_dir = parent_dir / f"{prior_prefix}{obj_sanitized}" / prior_gray_sub

        if not like_dir.exists() or not like_dir.is_dir():
            print(f"[WARN] Likelihood dir missing for '{obj}': {like_dir}")
            continue
        if not prior_dir.exists() or not prior_dir.is_dir():
            print(f"[WARN] Prior dir missing for '{obj}': {prior_dir}")
            continue

        post_root = parent_dir / f"{post_prefix}{obj_sanitized}"
        post_gray_dir = post_root / "posterior_gray"
        post_col_dir  = post_root / "posterior_colored"
        post_gray_dir.mkdir(parents=True, exist_ok=True)
        if save_colored:
            post_col_dir.mkdir(parents=True, exist_ok=True)

        # Index files
        like_files = list_pngs(like_dir)
        prior_files = list_pngs(prior_dir)

        like_names = {p.name for p in like_files}
        prior_names = {p.name for p in prior_files}
        inter = sorted(like_names & prior_names)

        if not inter:
            print(f"[WARN] No matching PNG filenames for '{obj}'. (likelihood={like_dir}, prior={prior_dir})")
            continue

        missing_like = sorted(prior_names - like_names)
        missing_prior = sorted(like_names - prior_names)
        if missing_like:
            print(f"[INFO] {len(missing_like)} images present in prior but missing in likelihood for '{obj}'. Sample: {missing_like[:5]}")
        if missing_prior:
            print(f"[INFO] {len(missing_prior)} images present in likelihood but missing in prior for '{obj}'. Sample: {missing_prior[:5]}")

        print(f"[RUN] Object '{obj}'  | matches: {len(inter)}")
        for name in inter:
            p_path = prior_dir / name
            l_path = like_dir / name

            # Load grayscale
            p_img = Image.open(p_path).convert("L")
            l_img = Image.open(l_path).convert("L")

            # Resize likelihood to prior size if needed
            if p_img.size != l_img.size:
                l_img = l_img.resize(p_img.size, resample=Image.BILINEAR)

            p_np = np.array(p_img, dtype=np.uint8)
            l_np = np.array(l_img, dtype=np.uint8)

            p01 = to_float01(p_np)
            l01 = to_float01(l_np)

            # Posterior = product in [0,1]
            post01 = np.clip(p01 * l01, 0.0, 1.0)

            # Save grayscale posterior
            out_gray = post_gray_dir / name
            if overwrite or not out_gray.exists():
                save_u8_gray(post01, out_gray)

            # Save colored visualization of (1 - posterior)
            if save_colored:
                inv01 = 1.0 - post01
                colored = colorize(inv01, cmap_name)
                out_col = post_col_dir / name
                if overwrite or not out_col.exists():
                    Image.fromarray(colored).save(out_col)

        print(f"[DONE] '{obj}' → {post_root}")

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
