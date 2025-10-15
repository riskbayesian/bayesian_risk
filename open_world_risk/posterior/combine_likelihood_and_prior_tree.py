#!/usr/bin/env python3
"""
combine_likelihood_and_prior_tree.py

Iterates a parent directory for multiple objects. For each object name in the YAML,
expects sibling folders (per object):

  likelihood_{object}/greyscale/*.png
  prior_{object}/risk_gray/*.png

Additionally, this version also expects scene-level depth & mask folders:

  depth_png_mm/*.png           # depth in millimeters (default)
  depth_valid_mask/*.png       # >127 == valid

For each matching filename (e.g., 000003.png), it computes:

  posterior = likelihood * ( 1 - exp(-(depth * depth_scale)^2) * (1 - prior) )

Then it forces invalid-depth pixels to 0 (black) in both grayscale and colored outputs.

Outputs (per object):
  posterior_{object}/posterior_gray/<file>.png
  posterior_{object}/posterior_colored/<file>.png   (colormap on 1 - posterior, with invalid pixels black)

If sizes differ, likelihood/depth/mask are resized to the PRIOR size (bilinear for images, nearest for mask).

YAML example at bottom.
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
    """uint8 [0..255] -> float [0..1]"""
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

def resize_to(img: Image.Image, size_hw: tuple[int, int], mode: str = "bilinear") -> Image.Image:
    # size_hw is (H, W) but PIL expects (W, H)
    interp = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    return img.resize((size_hw[1], size_hw[0]), resample=interp)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch combine likelihood & prior across objects under a parent directory with depth modulation and masking.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config (or set CONFIG env).")
    args = ap.parse_args()

    cfg, cfg_dir = load_config(args.config)
    paths = cfg.get("paths", {})
    vis   = cfg.get("visualization", {})
    save  = cfg.get("save", {})
    parms = cfg.get("params", {})

    # Resolve base
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

    # Scene-level depth folders (one set shared per scene/parent)
    depth_sub = paths.get("depth_subdir", "depth_png_mm")
    mask_sub  = paths.get("depth_mask_subdir", "depth_valid_mask")
    depth_dir = parent_dir / depth_sub
    mask_dir  = parent_dir / mask_sub
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Depth valid mask directory not found: {mask_dir}")

    # Output dirs: posterior_{object}/posterior_gray + posterior_colored
    post_prefix = paths.get("posterior_prefix", "posterior_")
    save_colored = bool(vis.get("save_colored", True))
    cmap_name = vis.get("colormap", "viridis")
    overwrite = bool(save.get("overwrite", True))

    # Depth scale: convert mm to meters by default (0.001). If your depth already meters, set to 1.0
    depth_scale = float(parms.get("depth_scale", 0.001))

    # Index depth & mask files by name for quick lookup
    depth_map = {p.name: p for p in list_pngs(depth_dir)}
    mask_map  = {p.name: p for p in list_pngs(mask_dir)}

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

        like_names  = {p.name for p in like_files}
        prior_names = {p.name for p in prior_files}
        inter = sorted(like_names & prior_names & depth_map.keys() & mask_map.keys())

        if not inter:
            print(f"[WARN] No matching PNG filenames for '{obj}'. "
                  f"(likelihood={like_dir}, prior={prior_dir}, depth={depth_dir}, mask={mask_dir})")
            continue

        # Inform about mismatches
        missing_like  = sorted((prior_names & depth_map.keys() & mask_map.keys()) - like_names)
        missing_prior = sorted((like_names  & depth_map.keys() & mask_map.keys()) - prior_names)
        missing_depth = sorted((like_names & prior_names & mask_map.keys()) - depth_map.keys())
        missing_mask  = sorted((like_names & prior_names & depth_map.keys()) - mask_map.keys())
        if missing_like:
            print(f"[INFO] {len(missing_like)} images present in prior/depth/mask but missing in likelihood for '{obj}'. Sample: {missing_like[:5]}")
        if missing_prior:
            print(f"[INFO] {len(missing_prior)} images present in likelihood/depth/mask but missing in prior for '{obj}'. Sample: {missing_prior[:5]}")
        if missing_depth:
            print(f"[INFO] {len(missing_depth)} images present in prior/likelihood/mask but missing in depth for '{obj}'. Sample: {missing_depth[:5]}")
        if missing_mask:
            print(f"[INFO] {len(missing_mask)} images present in prior/likelihood/depth but missing in mask for '{obj}'. Sample: {missing_mask[:5]}")

        print(f"[RUN] Object '{obj}'  | matches: {len(inter)}")
        for name in inter:
            p_path = prior_dir / name
            l_path = like_dir / name
            d_path = depth_map[name]
            m_path = mask_map[name]

            # Load grayscale prior/likelihood
            p_img = Image.open(p_path).convert("L")
            l_img = Image.open(l_path).convert("L")

            # Depth & mask
            d_img = Image.open(d_path).convert("F")  # float32 depth, assumed mm values
            m_img = Image.open(m_path).convert("L")  # 0..255, >127 == valid

            # Resize likelihood/depth/mask to PRIOR size if needed
            H, W = p_img.size[1], p_img.size[0]
            if p_img.size != l_img.size:
                l_img = resize_to(l_img, (H, W), mode="bilinear")
            if p_img.size != d_img.size:
                d_img = resize_to(d_img, (H, W), mode="bilinear")
            if p_img.size != m_img.size:
                m_img = resize_to(m_img, (H, W), mode="nearest")

            # To numpy
            p_np = np.array(p_img, dtype=np.uint8)
            l_np = np.array(l_img, dtype=np.uint8)
            d_np = np.array(d_img, dtype=np.float32)    # depth (mm as float if original png_mm)
            m_np = np.array(m_img, dtype=np.uint8)

            # Normalize prior/likelihood to [0,1]
            p01 = to_float01(p_np)
            l01 = to_float01(l_np)

            # Convert depth mm -> meters by depth_scale, then compute modulation:
            # mod = 1 - exp(-(d_scaled)^2) * (1 - prior)
            d_scaled = d_np * depth_scale
            mod = 1.0 - np.exp(-(d_scaled ** 2.0)) * (1.0 - p01)
            mod = np.clip(mod, 0.0, 1.0)

            # Posterior
            post01 = np.clip(l01 * mod, 0.0, 1.0)

            # Mask invalid depth to black
            valid = (m_np > 127)
            post01[~valid] = 0.0

            # Save grayscale posterior
            out_gray = post_gray_dir / name
            if overwrite or not out_gray.exists():
                save_u8_gray(post01, out_gray)

            # Save colored visualization of (1 - posterior) with invalid pixels black
            if save_colored:
                inv01 = 1.0 - post01
                colored = colorize(inv01, cmap_name)
                colored[~valid] = 0  # black out invalid
                out_col = post_col_dir / name
                if overwrite or not out_col.exists():
                    Image.fromarray(colored).save(out_col)

        print(f"[DONE] '{obj}' → {post_root}")

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
