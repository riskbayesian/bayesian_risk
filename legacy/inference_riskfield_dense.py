#!/usr/bin/env python3
# inference_riskfield_dense_multishift.py
"""
Dense RiskField inference using the multi-shift dense DINO trick (same as dinov3_cosine_sim.py):
1) Image A -> dense [C,H,W] DINO grid via patch/stride assembly on GPU.
2) Image B -> dense grid -> masked average to a single 1024-D vector.
3) RiskField(featA_token, featB_avg) -> Bezier coeffs per token.
4) Evaluate Bezier at 100 bins (0..0.5m); compute UNNORMALIZED expected distance sum_i p(d_i)*d_i.

Outputs:
- coeffs_hwK.pt
- expected_distance_unnormalized_hw.npy / .png
- probs_hwS.pt (optional)
"""

import argparse, os, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import torchvision.transforms.functional as TF
from matplotlib import cm  # for colormap lookups (no plotting windows)

# ---- Make `open_world_risk.utils.model` importable (run from workspace root or adjust) ----
THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.insert(0, str(ROOT))

from open_world_risk.utils.model import RiskField  # noqa

# ----------------- DINO helpers (multi-shift dense) -----------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
EXPECTED_DINO_DIM = 1024

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

def resize_to_multiple_of(img: Image.Image, multiple: int, target_h: int | None):
    w, h = img.size
    if target_h is not None:
        s = target_h / h
        H = max(multiple, (int((h * s) // multiple) * multiple))
        W = max(multiple, (int((w * s) // multiple) * multiple))
    else:
        H = max(multiple, (h // multiple) * multiple)
        W = max(multiple, (w // multiple) * multiple)
    return img.resize((W, H), Image.BILINEAR), (H, W)

def nonwhite_nontransparent_mask(img_rgba: torch.Tensor, tol_white=0.98, tol_alpha=0.01):
    """
    img_rgba: [4,H,W] in [0,1]
    Returns [H,W] boolean: True = keep pixel (not white, not transparent)
    """
    rgb = img_rgba[:3]
    alpha = img_rgba[3]
    nonwhite = (rgb < tol_white).any(dim=0)
    nontrans = alpha > tol_alpha
    return nonwhite & nontrans

class _GetLast(nn.Module):
    """Fetch only last block (reshape=True)."""
    def __init__(self, model, norm):
        super().__init__()
        self.model = model
        self.norm = norm
    def forward(self, x):
        t = self.model.get_intermediate_layers(
            x, n=[len(self.model.blocks)-1], reshape=True, norm=self.norm
        )[0]  # [1,Ht,Wt,C] or [Ht,Wt,C]
        if t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)
        # ensure [H,W,C] on device; pick largest axis as channels
        s = t.shape
        c_axis = max(range(3), key=lambda i: s[i])
        if c_axis == 0: t = t.permute(1,2,0)
        elif c_axis == 1: t = t.permute(0,2,1)
        return t.contiguous()

@torch.inference_mode()
def dense_embeddings(model, img_rgb, *, patch_size=16, stride=4, norm=True,
                     device="cuda", dtype_out=torch.float32, compiled_infer=None):
    """
    Multi-shift dense embeddings (as in dinov3_cosine_sim.py):
    - Normalize -> pad per shift -> get last block -> assemble [H,W,C] with index_copy_.
    Returns [C,H,W] in dtype_out.
    """
    assert patch_size % stride == 0, "stride must divide patch_size"
    Cimg, H, W = img_rgb.shape

    mean = torch.tensor(IMAGENET_MEAN, dtype=img_rgb.dtype, device=device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  dtype=img_rgb.dtype, device=device)[:, None, None]
    x0 = ((img_rgb.to(device, non_blocking=True) - mean) / std).unsqueeze(0)  # [1,3,H,W]

    def infer_last(x):
        t = (compiled_infer(x) if compiled_infer is not None
             else model.get_intermediate_layers(
                    x, n=[len(model.blocks)-1], reshape=True, norm=norm
                  )[0])
        if t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)
        s = t.shape
        c_axis = max(range(3), key=lambda i: s[i])
        if c_axis == 0: t = t.permute(1,2,0)
        elif c_axis == 1: t = t.permute(0,2,1)
        return t.contiguous()  # [H,W,C]

    shifts = [(dy, dx) for dy in range(0, patch_size, stride)
                        for dx in range(0, patch_size, stride)]

    dense = None
    for (dy, dx) in shifts:
        Hp = math.ceil((H + dy) / patch_size) * patch_size
        Wp = math.ceil((W + dx) / patch_size) * patch_size
        pad_bottom = Hp - (H + dy)
        pad_right  = Wp - (W + dx)
        x = F.pad(x0, (dx, pad_right, dy, pad_bottom))
        feats = infer_last(x)  # [Ht,Wt,C] on device

        Ht, Wt, C = feats.shape
        if dense is None:
            dense = feats.new_full((H, W, C), float('nan'))

        ys = (torch.arange(Ht, device=feats.device) * patch_size - dy)
        xs = (torch.arange(Wt, device=feats.device) * patch_size - dx)
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

    # fill any NaNs via simple upsample from the stride grid (rare)
    if torch.isnan(dense).any():
        step = stride
        small = dense[::step, ::step, :].permute(2,0,1).unsqueeze(0)
        up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1,2,0)
        mask = ~torch.isnan(dense).any(dim=2, keepdim=True)
        dense = torch.where(mask, dense, up)

    return dense.permute(2,0,1).to(dtype_out).contiguous()  # [C,H,W]

# ----------------- Bezier / Bernstein -----------------
def bernstein_B(num_ctrl: int, S: int = 100, device: str = "cuda", dtype=torch.float32):
    n = num_ctrl - 1
    t = torch.linspace(0, 1, S, device=device, dtype=dtype)
    comb = [math.comb(n, k) for k in range(num_ctrl)]
    B = []
    for k in range(num_ctrl):
        Bk = comb[k] * (t**k) * ((1.0 - t)**(n - k))
        B.append(Bk)
    return torch.stack(B, dim=1), t  # [S,K], [S]

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser("Dense RiskField inference with multi-shift DINO.")
    ap.add_argument("--image_a", required=True, help="Arbitrary image path for dense map (object A).")
    ap.add_argument("--image_b", required=True, help="Reference image path for average vector (object B).")
    ap.add_argument("--riskfield_ckpt", required=True, help="Path to trained RiskField checkpoint.")
    ap.add_argument("--dino_weights", required=True, help="Path to dinov3 ViT-L/16 weights (.pth/.safetensors).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--target_h", type=int, default=768)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--mask_white", type=float, default=0.98)
    ap.add_argument("--mask_alpha", type=float, default=0.01)
    ap.add_argument("--norm", action="store_true", help="Use DINO layer norm in get_intermediate_layers.")
    ap.add_argument("--save_probs", action="store_true", help="Save per-bin probabilities [H,W,100].")
    ap.add_argument("--upsample_to_image", action="store_true", help="Upsample outputs to resized image size.")
    ap.add_argument("--cmap", default="turbo",
                help="Matplotlib colormap name (e.g., turbo, viridis, magma, plasma).")
    ap.add_argument("--overlay_alpha", type=float, default=0.5,
                    help="Alpha for heatmap overlay onto scene RGB (0..1).")
    ap.add_argument("--scale", choices=["fixed","auto"], default="fixed",
                    help="Color scaling: 'fixed' uses sum(d) (comparable across runs); 'auto' scales per image.")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Load DINO backbone + local weights
    dino = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16",
                          source="github", pretrained=False).to(device).eval()
    state = torch.load(args.dino_weights, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    dino.load_state_dict(strip_prefixes(state), strict=False)

    get_last = _GetLast(dino, norm=args.norm).eval()   # compiled path wrapper

    # Load & resize images
    imgA = Image.open(args.image_a).convert("RGBA")
    imgB = Image.open(args.image_b).convert("RGBA")
    a_resized, (Ha, Wa) = resize_to_multiple_of(imgA, args.patch, args.target_h)
    b_resized, (Hb, Wb) = resize_to_multiple_of(imgB, args.patch, args.target_h)
    a_rgba = TF.to_tensor(a_resized)  # [4,Ha,Wa]
    b_rgba = TF.to_tensor(b_resized)  # [4,Hb,Wb]
    a_rgb = a_rgba[:3]; b_rgb = b_rgba[:3]

    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

    # ---- Dense embeddings via multi-shift (matches your script) ----
    featsA = dense_embeddings(dino, a_rgb, patch_size=args.patch, stride=args.stride,
                              norm=args.norm, device=device, dtype_out=torch.float32,
                              compiled_infer=get_last)
    featsB = dense_embeddings(dino, b_rgb, patch_size=args.patch, stride=args.stride,
                              norm=args.norm, device=device, dtype_out=torch.float32,
                              compiled_infer=get_last)
    # L2-normalize per token (channel dim)
    featsA = F.normalize(featsA, dim=0)   # [C,Ha,Wa]
    featsB = F.normalize(featsB, dim=0)   # [C,Hb,Wb]

    # ---- Average vector for B using its own nonwhite/nontransparent mask ----
    maskB = nonwhite_nontransparent_mask(b_rgba.to(device), tol_white=args.mask_white,
                                         tol_alpha=args.mask_alpha)  # [Hb,Wb]
    flatB = featsB.permute(1,2,0).reshape(-1, featsB.shape[0])       # [Hb*Wb,C]
    if maskB.any():
        vecB = flatB[maskB.view(-1)].mean(dim=0)
    else:
        vecB = flatB.mean(dim=0)
    vecB = F.normalize(vecB, dim=0)  # [1024]

    # ---- Load RiskField ----
    ckpt = torch.load(args.riskfield_ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", None); K = ckpt.get("K", None)
    if cfg is None or K is None:
        raise RuntimeError("Checkpoint missing 'cfg' or 'K' (retrain with the provided training script).")
    model = RiskField(
        clip_dim=cfg.get("featA", EXPECTED_DINO_DIM),
        num_layers=cfg.get("num_layers", 2),
        num_neurons=cfg.get("num_neurons", 512),
        projection_dim=cfg.get("projection_dim", 256),
        alphas_dim=K,
        monotonic=cfg.get("monotonic", True),
        compute_covariances=cfg.get("compute_covariances", False),
        num_layers_trans=cfg.get("num_layers_trans", 2),
        num_heads=cfg.get("num_heads", 8),
        attn_dropout=cfg.get("attn_dropout", 0.1),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)

    # ---- Forward per token ----
    Hf, Wf = featsA.shape[1:]
    A = featsA.permute(1,2,0).reshape(-1, featsA.shape[0])                   # [Ntok,1024]
    B = vecB.unsqueeze(0).expand(A.size(0), -1).contiguous()                  # [Ntok,1024]

    outs = []
    with torch.inference_mode():
        bs = 8192
        for i in range(0, A.size(0), bs):
            fa = A[i:i+bs].to(device, non_blocking=True)
            fb = B[i:i+bs].to(device, non_blocking=True)
            pred = model(fa, fb)
            pred = pred[0] if isinstance(pred, tuple) else pred
            outs.append(pred)
    coeffs = torch.cat(outs, dim=0).view(Hf, Wf, K)                           # [Hf,Wf,K]

    # ---- Bezier eval at 100 bins; UNNORMALIZED expected distance ----
    Bbern, t = bernstein_B(K, S=100, device=device, dtype=torch.float32)      # [100,K]
    d = 0.5 * t                                                                # [100]
    probs = 1. - torch.einsum("hwk,sk->hws", coeffs.to(device), Bbern).clamp(0, 1) # [Hf,Wf,100]
    exp_un = torch.einsum("hws,s->hw", probs, d).contiguous()                  # [Hf,Wf], no division

    # Optional upsample to resized image grid
    if args.upsample_to_image:
        exp_img = F.interpolate(exp_un.unsqueeze(0).unsqueeze(0), size=(Ha, Wa),
                                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        coeffs_img = F.interpolate(coeffs.permute(2,0,1).unsqueeze(0), size=(Ha, Wa),
                                   mode="nearest").squeeze(0).permute(1,2,0)
        probs_img = (F.interpolate(probs.permute(2,0,1).unsqueeze(0), size=(Ha, Wa),
                                   mode="bilinear", align_corners=False)
                     .squeeze(0).permute(1,2,0)) if args.save_probs else None
    else:
        exp_img, coeffs_img, probs_img = exp_un, coeffs, (probs if args.save_probs else None)

    # ---- Save (PNG heatmap + overlay), no plotting ----
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Save coeffs and raw array
    torch.save(coeffs_img.cpu(), os.path.join(args.out_dir, "coeffs_hwK.pt"))

    exp_arr = exp_img.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(args.out_dir, "expected_distance_unnormalized_hw.npy"), exp_arr)

    # 2) Ensure expectation map matches scene resize for overlay
    HA_img, WA_img = a_resized.size[1], a_resized.size[0]   # PIL: (W,H)
    if exp_img.shape[-2:] != (HA_img, WA_img):
        exp_img = torch.nn.functional.interpolate(
            exp_img.unsqueeze(0).unsqueeze(0),
            size=(HA_img, WA_img), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)
        exp_arr = exp_img.detach().cpu().numpy().astype(np.float32)

    # 3) Normalize for colormap (fixed uses sum(d); auto uses image max)
    if args.scale == "fixed":
        max_val = 100
        norm = exp_arr / max_val
    else:
        norm = (exp_arr - exp_arr.min()) / (exp_arr.max() - exp_arr.min())

    # 4) Apply colormap -> uint8 RGB heatmap
    cmap = cm.get_cmap(args.cmap)
    heat_rgba = cmap(norm)                          # [H,W,4] float in [0,1]
    heat_rgb  = (heat_rgba[..., :3] * 255).astype(np.uint8)  # [H,W,3] uint8

    # 5) Overlay on scene RGB (alpha blend)
    scene_rgb = np.array(a_resized.convert("RGB"), dtype=np.uint8)  # [H,W,3]
    alpha = float(np.clip(args.overlay_alpha, 0.0, 1.0))
    overlay = ( (1.0 - alpha) * scene_rgb.astype(np.float32)
            + alpha * heat_rgb.astype(np.float32) )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # 6) Write PNGs
    Image.fromarray(heat_rgb, mode="RGB").save(
        os.path.join(args.out_dir, "expected_distance_unnormalized_colormap.png")
    )
    Image.fromarray(overlay, mode="RGB").save(
        os.path.join(args.out_dir, "expected_distance_unnormalized_overlay.png")
    )

    # 7) Optional per-bin probabilities
    if probs_img is not None:
        torch.save(probs_img.cpu(), os.path.join(args.out_dir, "probs_hwS.pt"))

    print(f"Saved PNG heatmap + overlay to {args.out_dir}")

if __name__ == "__main__":
    main()
