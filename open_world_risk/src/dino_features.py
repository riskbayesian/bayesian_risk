#!/usr/bin/env python3
# DINOv3 dense per-pixel click-to-similarity between two images (local weights).
# Middle panel shows cosine similarity of the *non-white, non-transparent* average of Image A to Image B.

import argparse, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from torch import nn
import gc


# Optional safetensors support
try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class DinoFeatures:
    def __init__(self, stride=4, norm=True, patch_size=16, device="cuda"):
        
        self.model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16",
                        source="github", pretrained=False).to(device).eval()
        self.state = load_local_state_dict("/workspace/open_world_risk/src/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        self.model.load_state_dict(self.state, strict=False)

        self.device = device
        self.stride = stride
        self.norm = norm
        self.patch_size = patch_size
        assert self.patch_size % self.stride == 0, "stride must divide patch_size"
        
    def get_dino_features_rgb(self, rgb_tensor, norm=True):
           
        # Optional compilation for speed
        compiled_get_last = _GetLast(self.model, norm)

        with torch.inference_mode():
            featsB = self.dense_embeddings(rgb_tensor,
                                    dtype_out=torch.bfloat16,
                                    compiled_infer=compiled_get_last)

            featsB_n = normalize_feat_map(featsB)

        return featsB_n

    @torch.inference_mode()
    def dense_embeddings(self, img_rgb, dtype_out=torch.bfloat16, compiled_infer=None) -> torch.Tensor:
        """
        Faster dense per-pixel embeddings via multi-shift trick.
        - Stays on GPU end-to-end
        - Only asks DINO for the **last** block
        - Uses index_copy_ to assemble [H,W,C] on device
        - Returns [C,H,W] in dtype_out (bf16/half recommended)
        """
        assert self.patch_size % self.stride == 0, "stride must divide patch_size"
        Cimg, H, W = img_rgb.shape

        # Normalize on device
        mean = torch.tensor(IMAGENET_MEAN, dtype=img_rgb.dtype, device=self.device)[:, None, None]
        std  = torch.tensor(IMAGENET_STD,  dtype=img_rgb.dtype, device=self.device)[:, None, None]
        x0 = ((img_rgb.to(self.device, non_blocking=True) - mean) / std).unsqueeze(0)  # [1,3,H,W]

        # Wrapper to fetch only the last block, reshaped to [Ht,Wt,C]
        # Replace your infer_last() inside dense_embeddings with this:
        def infer_last(x):
            # Call compiled or plain
            t = (compiled_infer(x) if compiled_infer is not None
                else self.model.get_intermediate_layers(
                        x, n=[len(self.model.blocks)-1], reshape=True, norm=self.norm
                    )[0])

            # Remove batch dim if present: [1,H,W,C] or [1,C,H,W] -> [H,W,C]/[C,H,W]
            if t.dim() == 4 and t.size(0) == 1:
                t = t.squeeze(0)

            # Ensure HWC ordering on **GPU**
            # Pick the largest axis as channels (usually 768/1024), then permute to [H,W,C]
            s = t.shape  # length-3
            c_axis = max(range(3), key=lambda i: s[i])
            if c_axis == 0:        # [C,H,W] -> [H,W,C]
                t = t.permute(1, 2, 0)
            elif c_axis == 1:      # [H,C,W] -> [H,W,C]
                t = t.permute(0, 2, 1)
            # else already [H,W,C]

            return t.contiguous()  # [H,W,C] on device

        shifts = [(dy, dx) for dy in range(0, self.patch_size, self.stride)
                            for dx in range(0, self.patch_size, self.stride)]

        dense = None  # will allocate after first infer

        for (dy, dx) in shifts:
            Hp = math.ceil((H + dy) / self.patch_size) * self.patch_size
            Wp = math.ceil((W + dx) / self.patch_size) * self.patch_size
            pad_bottom = Hp - (H + dy)
            pad_right  = Wp - (W + dx)

            # Pad on device (no syncs)
            x = F.pad(x0, (dx, pad_right, dy, pad_bottom))

            feats = infer_last(x)  # [Ht,Wt,C], on GPU

            Ht, Wt, C = feats.shape
            if dense is None:
                # Lazy alloc on GPU; use fp32 for accumulating, cast at the end
                dense = feats.new_full((H, W, C), float('nan'))

            ys = (torch.arange(Ht, device=feats.device) * self.patch_size - dy)
            xs = (torch.arange(Wt, device=feats.device) * self.patch_size - dx)
            vy = (ys >= 0) & (ys < H)
            vx = (xs >= 0) & (xs < W)
            if not (vy.any() and vx.any()):
                continue

            ys = ys[vy]; xs = xs[vx]
            block = feats[vy][:, vx, :].reshape(-1, C)       # [n, C]
            yy = ys[:, None].repeat(1, xs.numel()).reshape(-1)
            xx = xs[None, :].repeat(ys.numel(), 1).reshape(-1)
            flat_idx = yy * W + xx                           # [n]
            dense.view(-1, C).index_copy_(0, flat_idx, block)

            block, vy, vx, ys, xs, flat_idx = None, None, None, None, None, None
            del block, vy, vx, ys, xs, flat_idx
            #gc.collect()

        # If any positions somehow remained NaN (shouldn't, but be safe), fill from the self.stride grid
        if torch.isnan(dense).any():
            step = self.stride
            small = dense[::step, ::step, :].permute(2, 0, 1).unsqueeze(0)
            up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            mask = ~torch.isnan(dense).any(dim=2, keepdim=True)
            dense = torch.where(mask, dense, up)

            # clean memory
            mask, up, small = None, None, None
            del mask, up, small   
            #gc.collect()

        return dense.permute(2, 0, 1).to(dtype_out).contiguous()  # [C,H,W]

    def rgb_image_preprocessing(self, rgb_im, target_h):
        assert target_h is not None, "target_h passed into rgb_image_preprocessing() None"

        rgb_im_resized, (height, width) = resize_to_multiple_of_numpy(rgb_im, self.patch_size, target_h)
        rgb_tensor  = TF.to_tensor(rgb_im_resized)  # [3,H,W]

        return rgb_tensor

# ----------------------------- Utilities -----------------------------

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
        if safetensors_load is None:
            raise RuntimeError("Install safetensors: pip install safetensors")
        state = safetensors_load(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    return strip_prefixes(state)

def resize_to_multiple_of(img: Image.Image, multiple: int, target_h: int = None):
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

def resize_to_multiple_of_numpy(img_np: np.ndarray, multiple: int, target_h: int = None):
    """Resize numpy array to target_h (if provided), then snap H,W down to multiples of `multiple`."""
    h, w = img_np.shape[:2]  # numpy arrays are [H, W, C]
    if target_h is not None:
        # Calculate scale factor to reach target_h
        scale = target_h / h
        #print(f"Debug: original={h}x{w}, target_h={target_h}, scale={scale}")
        # Apply scale to both dimensions to maintain aspect ratio
        new_h = int(h * scale)
        new_w = int(w * scale)
        #print(f"Debug: scaled={new_h}x{new_w}")
        # Snap down to multiples of `multiple`
        H = max(multiple, (new_h // multiple) * multiple)
        W = max(multiple, (new_w // multiple) * multiple)
        #print(f"Debug: snapped={H}x{W}")
    else:
        H = max(multiple, (h // multiple) * multiple)
        W = max(multiple, (w // multiple) * multiple)
    
    # Convert to PIL, resize, convert back to numpy
    img_pil = Image.fromarray(img_np)
    img_resized_pil = img_pil.resize((W, H), Image.BILINEAR)
    img_resized_np = np.array(img_resized_pil)
    
    return img_resized_np, (H, W)

def _to_hw_c(feats):
    """Robustly convert get_intermediate_layers(..., reshape=True) output to [H,W,C]."""
    t = feats
    if t.dim() == 4:
        t = t[0]
    dims = list(t.shape)
    if len(dims) != 3:
        raise RuntimeError(f"Unexpected feats shape: {tuple(dims)}")
    c_axis = int(torch.tensor(dims).argmax().item())  # channel is largest (≈1024)
    if c_axis == 2:
        out = t
    elif c_axis == 0:
        out = t.permute(1, 2, 0)
    elif c_axis == 1:
        out = t.permute(0, 2, 1)
    else:
        out = t
    return out.detach().cpu().contiguous()

# Optional: compile just the "get last block" path to amortize Python overhead
class _GetLast(nn.Module):
    def __init__(self, model, norm):
        super().__init__()
        self.model = model
        self.norm = norm
    def forward(self, x):
        # BEFORE: return ...[0][None, ...]
        return self.model.get_intermediate_layers(
            x, n=[len(self.model.blocks)-1], reshape=True, norm=self.norm
        )[0]   # -> [Ht, Wt, C]

def normalize_feat_map(feats_chw):
    return F.normalize(feats_chw, dim=0)  # [C,H,W]

def cosine_similarity_map(query_vec, feats_norm):
    return (feats_norm * query_vec[:, None, None]).sum(dim=0)  # [H,W]

def nonwhite_nontransparent_mask(img_rgba, tol_white=0.98, tol_alpha=0.01):
    """
    img_rgba: [4,H,W] in [0,1]
    tol_white: threshold to decide "white" (all channels >= tol_white)
    tol_alpha: alpha <= tol_alpha is considered transparent
    Returns: [H,W] boolean mask (True = keep pixel)
    """
    rgb = img_rgba[:3]
    alpha = img_rgba[3]
    nonwhite = (rgb < tol_white).any(dim=0)
    nontrans = alpha > tol_alpha
    return nonwhite & nontrans

