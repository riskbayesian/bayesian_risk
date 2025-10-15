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

# Optional safetensors support
try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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

@torch.inference_mode()
def dense_embeddings(model, img_rgb, *, patch_size=16, stride=4, norm=True,
                     device="cuda", dtype_out=torch.bfloat16, compiled_infer=None):
    """
    Faster dense per-pixel embeddings via multi-shift trick.
    - Stays on GPU end-to-end
    - Only asks DINO for the **last** block
    - Uses index_copy_ to assemble [H,W,C] on device
    - Returns [C,H,W] in dtype_out (bf16/half recommended)
    """
    assert patch_size % stride == 0, "stride must divide patch_size"
    Cimg, H, W = img_rgb.shape

    # Normalize on device
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_rgb.dtype, device=device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  dtype=img_rgb.dtype, device=device)[:, None, None]
    x0 = ((img_rgb.to(device, non_blocking=True) - mean) / std).unsqueeze(0)  # [1,3,H,W]

    # Wrapper to fetch only the last block, reshaped to [Ht,Wt,C]
    # Replace your infer_last() inside dense_embeddings with this:
    def infer_last(x):
        # Call compiled or plain
        t = (compiled_infer(x) if compiled_infer is not None
            else model.get_intermediate_layers(
                    x, n=[len(model.blocks)-1], reshape=True, norm=norm
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

    shifts = [(dy, dx) for dy in range(0, patch_size, stride)
                        for dx in range(0, patch_size, stride)]

    dense = None  # will allocate after first infer

    for (dy, dx) in shifts:
        Hp = math.ceil((H + dy) / patch_size) * patch_size
        Wp = math.ceil((W + dx) / patch_size) * patch_size
        pad_bottom = Hp - (H + dy)
        pad_right  = Wp - (W + dx)

        # Pad on device (no syncs)
        x = F.pad(x0, (dx, pad_right, dy, pad_bottom))

        feats = infer_last(x)  # [Ht,Wt,C], on GPU

        Ht, Wt, C = feats.shape
        if dense is None:
            # Lazy alloc on GPU; use fp32 for accumulating, cast at the end
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

    # If any positions somehow remained NaN (shouldn't, but be safe), fill from the stride grid
    if torch.isnan(dense).any():
        step = stride
        small = dense[::step, ::step, :].permute(2, 0, 1).unsqueeze(0)
        up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        mask = ~torch.isnan(dense).any(dim=2, keepdim=True)
        dense = torch.where(mask, dense, up)

    return dense.permute(2, 0, 1).to(dtype_out).contiguous()  # [C,H,W]

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

# ----------------------------- Main -----------------------------

def parse_args():
    ap = argparse.ArgumentParser("Click A; similarity on B. Middle = similarity of A's non-white/non-transparent average.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image_a", required=True)
    ap.add_argument("--image_b", required=True)
    ap.add_argument("--target_h", type=int, default=768)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--mask-threshold", type=float, default=0.98,
                    help="RGB >= threshold counts as white")
    ap.add_argument("--alpha-threshold", type=float, default=0.01,
                    help="Alpha <= threshold counts as transparent")
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return ap.parse_args()

def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # 1) Load backbone
    model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16",
                           source="github", pretrained=False).to(device).eval()

    # 2) Load weights
    state = load_local_state_dict(args.weights)
    model.load_state_dict(state, strict=False)

    # 3) Load & resize
    imgA = Image.open(args.image_a).convert("RGBA")
    imgB = Image.open(args.image_b).convert("RGB")
    a_resized, (Ha, Wa) = resize_to_multiple_of(imgA, multiple=args.patch, target_h=args.target_h)
    b_resized, (Hb, Wb) = resize_to_multiple_of(imgB, multiple=args.patch, target_h=args.target_h)

    a_rgba = TF.to_tensor(a_resized)  # [4,H,W]
    a_rgb  = a_rgba[:3]               # [3,H,W]
    b_rgb  = TF.to_tensor(b_resized)  # [3,H,W]

    # 4) Dense embeddings
    with torch.no_grad():
        with torch.inference_mode():
            # tnow = time.time()
            # torch.cuda.synchronize()
            # featsA = dense_embeddings(model, a_rgb, patch_size=args.patch,
            #                           stride=args.stride, norm=args.norm, device=device)  # [C,Ha,Wa]
            # torch.cuda.synchronize()
            # print(f"Dense embeddings A: {time.time() - tnow}s")

            # tnow = time.time()
            # torch.cuda.synchronize()
            # featsB = dense_embeddings(model, b_rgb, patch_size=args.patch,
            #                           stride=args.stride, norm=args.norm, device=device)  # [C,Hb,Wb]
            # torch.cuda.synchronize()
            # print(f"Dense embeddings B: {time.time() - tnow}s")

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            get_last = _GetLast(model, args.norm).eval()
            compiled_get_last = get_last

            tnow = time.time()
            torch.cuda.synchronize()
            featsA = dense_embeddings(model, a_rgb, patch_size=args.patch, stride=args.stride,
                                    norm=args.norm, device=device, dtype_out=torch.bfloat16,
                                    compiled_infer=compiled_get_last)

            torch.cuda.synchronize()
            print(f"Dense embeddings A: {time.time() - tnow}s")
            print(featsA.shape)
            print(a_rgb.shape)

            tnow = time.time()
            torch.cuda.synchronize()
            featsB = dense_embeddings(model, b_rgb, patch_size=args.patch, stride=args.stride,
                                    norm=args.norm, device=device, dtype_out=torch.bfloat16,
                                    compiled_infer=compiled_get_last)
            torch.cuda.synchronize()
            print(f"Dense embeddings B: {time.time() - tnow}s")
            print(featsB.shape)
            print(b_rgb.shape)

            featsB_n = normalize_feat_map(featsB)

            # --- Global avg of non-white/non-transparent in A ---
            mask = nonwhite_nontransparent_mask(a_rgba, tol_white=args.mask_threshold,
                                                tol_alpha=args.alpha_threshold)
            flat_feats = featsA.permute(1, 2, 0).reshape(-1, featsA.shape[0])  # [Ha*Wa,C]
            flat_mask = mask.view(-1)
            if flat_mask.any():
                avgA = flat_feats[flat_mask].mean(dim=0)
            else:
                avgA = flat_feats.mean(dim=0)
            avgA = F.normalize(avgA, dim=0)

            sim_avg = cosine_similarity_map(avgA, featsB_n)
            sim_avg01 = (sim_avg - sim_avg.min()) / (sim_avg.max() - sim_avg.min() + 1e-8)

            cmap = plt.get_cmap("jet")
            sim_avg_rgb = cmap(sim_avg01.to(torch.float32).cpu().numpy())[..., :3]

            baseB = b_rgb.permute(1, 2, 0).numpy()
            overlay_avg = 0.55 * baseB + 0.45 * sim_avg_rgb

    # 5) Plot
    fig, (axA, axM, axB) = plt.subplots(1, 3, figsize=(16, 6), dpi=140)
    axA.imshow(np.array(a_resized)); axA.set_title("Image A (click)"); axA.axis("off")
    axM.imshow(overlay_avg); axM.set_title("Similarity: avg(non-white/non-transparent A) → B"); axM.axis("off")
    imB = axB.imshow(baseB); axB.set_title("Similarity: click(A) → B"); axB.axis("off")
    hm = [None]

    def onclick(event):
        if event.inaxes != axA:
            return
        x = int(round(event.xdata)); y = int(round(event.ydata))
        if x < 0 or x >= Wa or y < 0 or y >= Ha: return
        with torch.inference_mode():
            q = featsA[:, y, x].to(featsB_n.dtype)
            q = F.normalize(q, dim=0)
            sim = cosine_similarity_map(q, featsB_n)
            sim01 = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            sim_rgb = cmap(sim01.to(torch.float32).cpu().numpy())[..., :3]
            overlay = 0.55 * baseB + 0.45 * sim_rgb
        axA.scatter([x], [y], s=30, c="yellow", marker="x")
        if hm[0] is None:
            hm[0] = axB.imshow(overlay, interpolation="nearest")
        else:
            hm[0].set_data(overlay)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
