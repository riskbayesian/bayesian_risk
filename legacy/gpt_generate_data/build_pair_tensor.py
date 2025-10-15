#!/usr/bin/env python3
"""
Build an N x M tensor from a safety-CSV and per-object images using DINOv3.

Row i corresponds to CSV row i:
  [ featA_1024 | featB_1024 | probs_100 ]  ->  2148 dims total

- One 1024-D vector per image via masked global average of the last DINO block features.
- Probabilities come directly from the CSV columns p@{distance}m (100 bins from 0..0.5m).
"""

import argparse, os, sys, math
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# --------------------------- Settings / Constants ---------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
EXPECTED_DINO_DIM = 1024  # ViT-L/16 in DINOv3
PATCH_DEFAULT = 16

# --------------------------- Utilities (from your style) --------------------

def strip_prefixes(state):
    prefixes = ("model.", "module.", "state_dict.", "backbone.")
    out = {}
    for k, v in state.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def load_local_state_dict(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file as safetensors_load
        except Exception:
            raise RuntimeError("Install safetensors to load .safetensors weights: pip install safetensors")
        state = safetensors_load(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    return strip_prefixes(state)

def resize_to_multiple_of(img: Image.Image, multiple: int, target_h: Optional[int] = None):
    """Resize to target_h (if provided), then snap H,W down to multiples of `multiple`."""
    w, h = img.size
    if target_h is not None:
        s = target_h / h
        H = max(multiple, (int((h * s) // multiple) * multiple))
        W = max(multiple, (int((w * s) // multiple) * multiple))
    else:
        H = max(multiple, (h // multiple) * multiple)
        W = max(multiple, (w // multiple) * multiple)
    img_resized = img.resize((W, H), Image.BILINEAR)
    return img_resized, (H, W)

def nonwhite_nontransparent_mask(img_rgba: torch.Tensor, tol_white=0.98, tol_alpha=0.01):
    """
    img_rgba: [4,H,W] in [0,1]
    Returns: [H,W] boolean mask (True = keep pixel)
    """
    rgb = img_rgba[:3]
    alpha = img_rgba[3]
    nonwhite = (rgb < tol_white).any(dim=0)
    nontrans = alpha > tol_alpha
    return nonwhite & nontrans

# --------------------------- DINO helpers -----------------------------------

class _GetLast(nn.Module):
    """Wrapper to fetch only the last block reshaped to [H,W,C]."""
    def __init__(self, model, norm=True):
        super().__init__()
        self.model = model
        self.norm = norm
    def forward(self, x):
        t = self.model.get_intermediate_layers(
            x, n=[len(self.model.blocks)-1], reshape=True, norm=self.norm
        )[0]
        # t: [B,Ht,Wt,C] or [Ht,Wt,C]
        if t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)
        # enforce [H,W,C]
        s = t.shape  # len=3
        c_axis = max(range(3), key=lambda i: s[i])
        if c_axis == 0:        # [C,H,W] -> [H,W,C]
            t = t.permute(1, 2, 0)
        elif c_axis == 1:      # [H,C,W] -> [H,W,C]
            t = t.permute(0, 2, 1)
        return t.contiguous()  # [H,W,C] on same device

@torch.inference_mode()
def dino_single_vector(model, img_path: str, *,
                       device="cuda", target_h=512, patch=16, norm=True,
                       mask_white=0.98, mask_alpha=0.01) -> torch.Tensor:
    """
    Load image, resize, get last-block features [H,W,C], masked global average -> [C].
    """
    # Try RGBA (to enable transparency mask); fall back to RGB
    try:
        img = Image.open(img_path).convert("RGBA")
        rgba_used = True
    except Exception:
        img = Image.open(img_path).convert("RGB")
        rgba_used = False

    img_rz, (H, W) = resize_to_multiple_of(img, multiple=patch, target_h=target_h)

    if rgba_used:
        t_rgba = TF.to_tensor(img_rz)  # [4,H,W]
        t_rgb  = t_rgba[:3]
    else:
        t_rgb  = TF.to_tensor(img_rz)  # [3,H,W]

    # Normalize on device
    x = t_rgb.to(device, dtype=torch.float32, non_blocking=True)
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  dtype=x.dtype, device=device)[:, None, None]
    x = ((x - mean) / std).unsqueeze(0)  # [1,3,H,W]

    # Forward last block
    get_last = _GetLast(model, norm=norm).eval()
    feats = get_last(x)  # [H,W,C] on device
    Hf, Wf, C = feats.shape
    if C != EXPECTED_DINO_DIM:
        raise RuntimeError(f"Unexpected DINO feature dim {C}. Expected {EXPECTED_DINO_DIM} (ViT-L/16).")

    # Masked average over spatial map
    if rgba_used:
        m = nonwhite_nontransparent_mask(t_rgba.to(device), tol_white=mask_white, tol_alpha=mask_alpha)  # [H,W]
        # If downsample from image grid to token grid is needed, interpolate mask
        if (Hf, Wf) != (t_rgba.shape[1], t_rgba.shape[2]):
            m = m.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            m = F.interpolate(m, size=(Hf, Wf), mode="nearest").squeeze(0).squeeze(0).bool()
    else:
        # No alpha; still suppress white background if any (build a synthetic mask)
        rgb = t_rgb.to(device)
        m = (rgb < 0.98).any(dim=0)  # [H,W]
        if (Hf, Wf) != (rgb.shape[1], rgb.shape[2]):
            m = m.float().unsqueeze(0).unsqueeze(0)
            m = F.interpolate(m, size=(Hf, Wf), mode="nearest").squeeze(0).squeeze(0).bool()

    if not m.any():
        # Fallback to full average
        vec = feats.view(-1, C).mean(dim=0)
    else:
        vec = feats[m].view(-1, C).mean(dim=0)  # [C]

    vec = F.normalize(vec, dim=0)  # L2 normalize
    return vec.to(torch.float32).detach()

def resolve_image_path(images_dir: str, name: str) -> str:
    """
    Try a few filename variants:
      - exact 'name'.png / .jpg
      - spaces replaced with underscores
    """
    candidates = [
        os.path.join(images_dir, f"{name}.png"),
        os.path.join(images_dir, f"{name}.jpg"),
        os.path.join(images_dir, f"{name}.jpeg"),
        os.path.join(images_dir, f"{name.replace(' ', '_')}.png"),
        os.path.join(images_dir, f"{name.replace(' ', '_')}.jpg"),
        os.path.join(images_dir, f"{name.replace(' ', '_')}.jpeg"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Image for '{name}' not found in {images_dir}. Tried: {candidates}")

# --------------------------- Main build script ------------------------------

def parse_args():
    ap = argparse.ArgumentParser("Build N x 2148 tensor from CSV + images using DINOv3 (ViT-L/16).")
    ap.add_argument("--csv", required=True, help="Path to the safety CSV (100 p@... columns).")
    ap.add_argument("--images-dir", required=True, help="Directory containing object images.")
    ap.add_argument("--weights", required=True, help="Path to DINOv3 ViT-L/16 weights (.pth or .safetensors).")
    ap.add_argument("--out", required=True, help="Output .pt file for the N x 2148 tensor.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference.")
    ap.add_argument("--target-h", type=int, default=512, help="Resize height (snapped to multiple of patch).")
    ap.add_argument("--patch", type=int, default=PATCH_DEFAULT, help="Patch size (16 for ViT-L/16).")
    ap.add_argument("--norm", action="store_true", help="Use DINO layer norm in get_intermediate_layers.")
    ap.add_argument("--mask-white", type=float, default=0.98, help="RGB >= threshold counts as white.")
    ap.add_argument("--mask-alpha", type=float, default=0.01, help="Alpha <= threshold is transparent.")
    return ap.parse_args()

def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # 1) Load DINOv3 ViT-L/16 backbone and weights (same approach as your script)
    model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16",
                           source="github", pretrained=False).to(device).eval()
    state = load_local_state_dict(args.weights)
    model.load_state_dict(state, strict=False)

    # 2) Read CSV and identify probability columns (ensure correct distance order)
    df = pd.read_csv(args.csv)
    if "manipulated_object" not in df.columns or "around_object" not in df.columns:
        raise RuntimeError("CSV must have 'manipulated_object' and 'around_object' columns.")
    prob_cols = [c for c in df.columns if c.startswith("p@") and c.endswith("m")]
    if len(prob_cols) != 100:
        raise RuntimeError(f"Expected 100 probability columns; found {len(prob_cols)}.")

    # Parse distances and sort columns by distance (just in case)
    def parse_d(col):
        # c like 'p@0.123m' -> 0.123
        return float(col[2:-1])
    prob_cols = sorted(prob_cols, key=parse_d)

    # 3) Precompute a single DINO feature per unique object
    names = sorted(set(df["manipulated_object"].tolist()) | set(df["around_object"].tolist()))
    feat_cache: Dict[str, torch.Tensor] = {}

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    for name in names:
        img_path = resolve_image_path(args.images_dir, name)
        vec = dino_single_vector(model, img_path,
                                 device=device, target_h=args.target_h, patch=args.patch,
                                 norm=args.norm, mask_white=args.mask_white, mask_alpha=args.mask_alpha)
        if vec.numel() != EXPECTED_DINO_DIM:
            raise RuntimeError(f"Got feature dim {vec.numel()} for {name}, expected {EXPECTED_DINO_DIM}.")
        feat_cache[name] = vec

    # 4) Build N x 2148 tensor in CSV order
    N = len(df)
    M = EXPECTED_DINO_DIM + EXPECTED_DINO_DIM + len(prob_cols)  # 1024 + 1024 + 100
    out_tensor = torch.empty((N, M), dtype=torch.float32)

    # Extract probs to a contiguous numpy array in the sorted column order
    probs_mat = df[prob_cols].to_numpy(dtype=np.float32)  # shape (N,100)

    for i, row in df.iterrows():
        a = row["manipulated_object"]
        b = row["around_object"]
        fa = feat_cache[a]
        fb = feat_cache[b]
        p  = torch.from_numpy(probs_mat[i])  # [100]
        out_tensor[i, :EXPECTED_DINO_DIM] = fa
        out_tensor[i, EXPECTED_DINO_DIM:2*EXPECTED_DINO_DIM] = fb
        out_tensor[i, 2*EXPECTED_DINO_DIM:] = p

    # 5) Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out_tensor, args.out)
    print(f"Saved tensor {tuple(out_tensor.shape)} to {args.out}")

if __name__ == "__main__":
    main()
