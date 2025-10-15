#!/usr/bin/env python3
# Minimal DINOv3 ViT-L/16 per-pixel PCA using a *local* checkpoint.

import argparse, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
import urllib
import torch.nn.functional as F

# Optional safetensors support (only if your file is .safetensors)
try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def parse_args():
    ap = argparse.ArgumentParser(description="DINOv3 per-pixel PCA (local weights)")
    ap.add_argument("--weights", required=True, help="Path to local weights (.pth or .safetensors)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--target_h", type=int, default=768, help="Resize height (multiples of 16)")
    ap.add_argument("--patch", type=int, default=16, help="Patch size (ViT-L/16 -> 16)")
    ap.add_argument("--norm", action="store_true", help="Apply LayerNorm in get_intermediate_layers")
    return ap.parse_args()

def resize_to_patch_grid(img: Image.Image, target_h: int, patch: int):
    """Resize keeping aspect ratio; snap H,W down to multiples of patch."""
    w, h = img.size
    s = target_h / h
    H = max(patch, (int((h * s) // patch) * patch))
    W = max(patch, (int((w * s) // patch) * patch))
    tens = TF.to_tensor(TF.resize(img, (H, W)))
    return tens, H // patch, W // patch

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
        raw = torch.load(path, map_location="cpu")
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    return strip_prefixes(state)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load backbone *code only* (no downloading)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model="dinov3_vitl16",
        source="github",
        pretrained=False,
    ).to(device).eval()

    # 2) Load local weights
    state = load_local_state_dict(args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    PATCH_SIZE = 16
    IMAGE_SIZE = 768

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    image_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"

    def load_image_from_url(url: str) -> Image:
        with urllib.request.urlopen(url) as f:
            return Image.open(f).convert("RGB")
            
    # image resize transform to dimensions divisible by patch size
    def resize_transform(
        mask_image: Image,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
    ) -> torch.Tensor:
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


    # image = load_image_from_url(image_uri)
    image = Image.open("color_0049.png")
    # image_resized = resize_transform(image)
    # # image_resized = TF.to_tensor(image)
    # image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # n_layers = 24

    # with torch.inference_mode():
    #     with torch.autocast(device_type='cuda', dtype=torch.float32):
    #         feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
    #         x = feats[-1].squeeze().detach().cpu()
    #         dim = x.shape[0]
    #         x = x.view(dim, -1).permute(1, 0)

    # pca = PCA(n_components=3, whiten=True)
    # pca.fit(x)

    # h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]

    # # apply the PCA, and then reshape
    # projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)

    # # multiply by 2.0 and pass through a sigmoid to get vibrant colors 
    # projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

    @torch.inference_mode()
    def dense_pca_image(
        model,
        img_rgb_tensor,              # [3,H,W] in [0,1], unnormalized
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225),
        patch_size=16,
        stride=4,                    # must divide patch_size; try 8/4/2/1
        norm=True,
        device="cuda",
    ):
        assert patch_size % stride == 0, "stride must divide patch_size"
        Cimg, H, W = img_rgb_tensor.shape

        # normalize
        mean = torch.tensor(mean, dtype=img_rgb_tensor.dtype)[:, None, None]
        std  = torch.tensor(std,  dtype=img_rgb_tensor.dtype)[:, None, None]
        img = (img_rgb_tensor.to(device) - mean.to(device)) / std.to(device)
        img = img.unsqueeze(0)  # [1,3,H,W]

        # helper to ensure feats -> [H_p, W_p, C] regardless of backend ordering
        def to_hw_c(feats_reshaped):
            """
            feats_reshaped is the output of get_intermediate_layers(..., reshape=True)[-1]
            Typical shapes seen:
            - [1, H_p, W_p, C]
            - [1, C, H_p, W_p]
            - [H_p, W_p, C]
            - [C, H_p, W_p]
            Return as [H_p, W_p, C] on CPU.
            """
            t = feats_reshaped
            if t.dim() == 4:
                # remove batch if present
                t = t[0]
            # Now t is 3D
            h, w, c = t.shape                 # optimistic
            # If the last dim looks too small to be channels, auto-detect:
            dims = list(t.shape)
            # pick channel axis as the argmax dimension (ViT-L/16 has C=1024, larger than H_p,W_p)
            c_axis = int(torch.tensor(dims).argmax())
            if c_axis == 2:
                out = t                       # already [H,W,C]
            elif c_axis == 0:
                out = t.permute(1, 2, 0)      # [C,H,W] -> [H,W,C]
            elif c_axis == 1:
                out = t.permute(0, 2, 1)      # [H,C,W] -> [H,W,C]
            else:
                out = t
            return out.detach().cpu().contiguous()

        # allocate later when we know C
        dense_feats = None

        shifts = [(dy, dx) for dy in range(0, patch_size, stride)
                            for dx in range(0, patch_size, stride)]

        for (dy, dx) in shifts:
            # pad so patch grid anchors align to this phase
            pad_top, pad_left = dy, dx
            Hp = (H + pad_top + patch_size - 1) // patch_size * patch_size
            Wp = (W + pad_left + patch_size - 1) // patch_size * patch_size
            pad_bottom = Hp - (H + pad_top)
            pad_right  = Wp - (W + pad_left)

            x = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
            feats = model.get_intermediate_layers(x, n=1, reshape=True, norm=norm)[-1]

            # -> [H_t, W_t, C]
            feats = to_hw_c(feats)
            Ht, Wt, C = feats.shape

            if dense_feats is None:
                dense_feats = torch.zeros(H, W, C, dtype=feats.dtype)

            # anchor positions for each token
            ys = torch.arange(Ht) * patch_size - pad_top
            xs = torch.arange(Wt) * patch_size - pad_left

            valid_y = (ys >= 0) & (ys < H)
            valid_x = (xs >= 0) & (xs < W)
            ys = ys[valid_y]
            xs = xs[valid_x]
            if ys.numel() == 0 or xs.numel() == 0:
                continue

            feats_valid = feats[valid_y][:, valid_x, :]  # [Ht_v, Wt_v, C]

            # scatter into dense grid at anchor pixels
            yy = ys.view(-1, 1).repeat(1, xs.numel()).reshape(-1)
            xx = xs.view(1, -1).repeat(ys.numel(), 1).reshape(-1)
            feats_flat = feats_valid.reshape(-1, C)

            dense_feats[yy, xx, :] = feats_flat

        # If stride > 1, fill any untouched pixels by upsampling from the filled lattice
        if stride > 1:
            step = stride
            small = dense_feats[::step, ::step, :].permute(2, 0, 1).unsqueeze(0)  # [1,C,h',w']
            up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            mask = (dense_feats.abs().sum(dim=2) > 0).unsqueeze(2).float()
            dense_feats = mask * dense_feats + (1 - mask) * up

        # PCA → 3 channels at full res
        C = dense_feats.shape[2]
        X = dense_feats.view(-1, C).numpy()
        proj = PCA(n_components=3, whiten=True, random_state=0).fit_transform(X)
        proj = torch.from_numpy(proj).view(H, W, 3)
        vis = torch.sigmoid(proj * 2.0).permute(2, 0, 1)  # [3,H,W] in [0,1]
        return vis

    img_rgb = TF.to_tensor(image)  # [3,H,W] in [0,1]
    dense = dense_pca_image(model, img_rgb, patch_size=16, stride=4, norm=True, device="cuda")  # try stride=4 first

    # enjoy
    fig, ax = plt.subplots(1, 2, dpi=300)
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(dense.permute(1, 2, 0))
    ax[1].set_title('Projected Image')
    ax[1].axis('off')
    plt.show()

if __name__ == "__main__":
    main()
