#!/usr/bin/env python3
"""
Lite RiskField inference with foldered outputs.

Changes vs original:
- Saves ALL outputs into dedicated subfolders under cfg["out_dir"].
- Filenames match the input RGB basename (e.g., 000003.png / 000003.npy).
- No more concatenated names like riskfield_at_depth_risk_depth.png.

Folders created inside out_dir:
  - expectation/       : colored risk expectation image
  - depth_risk/        : colored risk-at-depth image
  - npy/               : raw numpy array for risk-at-depth (and optional extras)
  - (optional) probs/  : if cfg["save_probs"] is True, dump a small debug PNG of PDF/CDF per frame

Config keys remain the same; out_path is no longer required (ignored if present).
"""
import os, sys, math, glob, shutil, time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import yaml

# Local imports (same as original)
THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.insert(0, str(ROOT))

from open_world_risk.utils.model import RiskField  # noqa
from open_world_risk.src.dino_features import (
    DinoFeatures, resize_to_multiple_of, nonwhite_nontransparent_mask
)  # noqa
from open_world_risk.data_deriv_likelihood.inference_riskfield_video_utils import (
    plot_coefficents,
    plot_risk_image,
    plot_depth_risk_image,
    timeFunction,
    learn_pca_from_features,
    vis_dino_pca,
    bernstein_B,
)

EXPECTED_DINO_DIM = 1024

# ----------------- DINO helpers -----------------

def extract_dino_inputs_and_features(dino_extractor: DinoFeatures, image_array: np.ndarray,
                                      patch_size: int, target_height: int, norm_features: bool = True):
    assert image_array.ndim == 3 and image_array.shape[2] == 3, f"Expected [H,W,3], got {image_array.shape}"
    alpha = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=np.uint8)
    img_rgba = np.concatenate([image_array, alpha], axis=2)
    img = Image.fromarray(img_rgba, 'RGBA')
    resized_image, (H, W) = resize_to_multiple_of(img, patch_size, target_height)

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    rgba_tensor = dino_extractor.rgb_image_preprocessing(np.array(resized_image), target_h=target_height)
    rgb_tensor = rgba_tensor[:3]
    dino_features = dino_extractor.get_dino_features_rgb(rgb_tensor, norm=norm_features).float()
    return resized_image, rgba_tensor, dino_features


def compute_average_vector_dino(dinoFeats, rgba_im, cfg, device):
    mask = nonwhite_nontransparent_mask(rgba_im.to(device), tol_white=cfg["mask_white"], tol_alpha=cfg["mask_alpha"])  # [H,W]
    flat = dinoFeats.permute(1, 2, 0).reshape(-1, dinoFeats.shape[0])
    vec = flat[mask.view(-1)].mean(dim=0) if mask.any() else flat.mean(dim=0)
    return F.normalize(vec, dim=0)

# ----------------- RiskField core -----------------

def load_model_riskfield(cfg, device):
    with open(cfg["riskfield_cfg"], "r") as f:
        model_config = yaml.safe_load(f)
    tensor_layout = model_config["tensor_layout"]
    model_params = model_config["model"]
    num_ctrl_pts = tensor_layout["ctrl_pts"]

    ckpt = torch.load(cfg["riskfield_ckpt"], map_location="cpu")
    model = RiskField(
        clip_dim=tensor_layout["feat_size"],
        num_layers=model_params["num_layers"],
        num_neurons=model_params["num_neurons"],
        projection_dim=model_params["projection_dim"],
        alphas_dim=num_ctrl_pts,
        monotonic=model_params["monotonic"],
        compute_covariances=model_params["compute_covariances"],
        num_layers_trans=model_params["num_layers_trans"],
        num_heads=model_params["num_heads"],
        attn_dropout=model_params["attn_dropout"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    return model, num_ctrl_pts


def evaluate_riskfield_depth(coeffs: torch.Tensor, depth_frame: torch.Tensor, depth_mask_frame: torch.Tensor,
                             dists: torch.Tensor, max_dist: float) -> np.ndarray:
    H, W = coeffs.shape[:2]
    num_ctrl_pts = coeffs.shape[2]
    device = coeffs.device

    depth_frame = depth_frame.to(device=device, dtype=torch.float32)
    depth_mask_frame = depth_mask_frame.to(device=device)
    coeffs = coeffs.to(device=device, dtype=torch.float32)

    risk_values = torch.zeros_like(depth_frame, dtype=torch.float32)
    valid_mask = depth_mask_frame.bool()
    assert valid_mask.any(), "Invalid Mask!"

    valid_depths = torch.clamp(depth_frame[valid_mask], 0.0, float(max_dist))
    t_values = valid_depths / float(max_dist)

    valid_coeffs = coeffs[valid_mask]

    i_idx = torch.arange(num_ctrl_pts, device=device, dtype=torch.float32)
    binom = torch.tensor([math.comb(num_ctrl_pts - 1, int(i)) for i in range(num_ctrl_pts)],
                         device=device, dtype=torch.float32)
    t_col = t_values.unsqueeze(1)
    basis = binom * (t_col ** i_idx) * ((1.0 - t_col) ** (num_ctrl_pts - 1 - i_idx))

    risk_at_depths = (valid_coeffs * basis).sum(dim=1)
    risk_values[valid_mask] = risk_at_depths
    return risk_values.detach().cpu().numpy()


def forward_riskfield_dense(model, featsA, vecB, num_ctrl_pts, device, max_distance):
    Hf, Wf = featsA.shape[1:]
    A = featsA.permute(1, 2, 0).reshape(-1, featsA.shape[0])
    B = vecB.unsqueeze(0).expand(A.size(0), -1).contiguous()

    outs = []
    with torch.inference_mode():
        bs = 8192
        for i in range(0, A.size(0), bs):
            fa = A[i:i+bs].to(device, non_blocking=True)
            fb = B[i:i+bs].to(device, non_blocking=True)
            pred = model(fa, fb)
            pred = pred[0] if isinstance(pred, tuple) else pred
            outs.append(pred)
    coeffs = torch.cat(outs, dim=0).view(Hf, Wf, num_ctrl_pts).contiguous()

    Bbern, t_values = bernstein_B(num_ctrl_pts, S=100, device=device, dtype=torch.float32)
    dists = max_distance * t_values

    # CDF -> PDF and expectation
    probs_cdf = torch.einsum("hwk,sk->hws", coeffs.to(device), Bbern).clamp(0, 1)
    dt = torch.diff(t_values, prepend=t_values[:1]).to(probs_cdf.device)
    F_diff = torch.diff(probs_cdf, dim=-1)
    dt_diff = dt[1:]
    pdf = torch.zeros_like(probs_cdf)
    pdf[..., 1:] = F_diff / dt_diff.view(1, 1, -1)
    pdf[..., 0] = probs_cdf[..., 0]
    pdf = pdf.clamp_min(0)
    s = pdf.sum(dim=-1)
    pdf = pdf / s.unsqueeze(-1).clamp(min=1e-8)
    expectation = torch.einsum("hws,s->hw", pdf, dists).contiguous()
    return coeffs, pdf, expectation, dists


def generate_risk_image(exp_un: torch.Tensor, a_resized: Image.Image, cfg: dict) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    Ha, Wa = a_resized.size[1], a_resized.size[0]
    if cfg["upsample_to_image"]:
        exp_img = F.interpolate(exp_un.unsqueeze(0).unsqueeze(0), size=(Ha, Wa),
                                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    else:
        exp_img = exp_un
    if exp_img.shape[-2:] != (Ha, Wa):
        exp_img = F.interpolate(exp_img.unsqueeze(0).unsqueeze(0), size=(Ha, Wa),
                                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    exp_arr = exp_img.detach().cpu().numpy().astype(np.float32)
    if cfg["scale"] == "fixed":
        max_val = 100.0
        norm = exp_arr / max_val
    else:
        denom = (exp_arr.max() - exp_arr.min()) if (exp_arr.max() > exp_arr.min()) else 1.0
        norm = (exp_arr - exp_arr.min()) / denom
    return exp_img, exp_arr, norm

# ----------------- Config & I/O -----------------

def load_infer_cfg(config_path: str = "infer_risk_field_vid_lite_cfg.yaml") -> Tuple[dict, Path]:
    # Resolve config path relative to script directory
    if not os.path.isabs(config_path):
        config_path = ROOT / config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    required = [
        "env_rgb", "env_depth", "env_depth_mask", "image_b",
        "riskfield_ckpt", "riskfield_cfg", "dino_weights",
        "out_dir", "device", "target_h", "patch", "stride",
        "mask_white", "mask_alpha", "norm", "save_probs",
        "upsample_to_image", "cmap", "overlay_alpha", "scale", "max_distance"
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg, Path(config_path)


def _make_out_dirs(root: Path) -> dict:
    sub = {
        "expectation": root / "expectation",
        "depth_risk": root / "depth_risk",
        "npy": root / "npy",
        "probs": root / "probs",  # optional
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub

# ----------------- Runner -----------------

def runner(frame_rgb: np.ndarray, frame_depth: np.ndarray, frame_mask: np.ndarray,
           manip_dino: torch.Tensor, risk_model: torch.nn.Module, dino_extractor: DinoFeatures,
           num_ctrl_pts: int, max_dist: float, device: str, out_dirs: dict,
           cfg: dict, base_name: str, verbose: bool = False) -> np.ndarray:

    frame_resized, frame_rgba, frame_dino = extract_dino_inputs_and_features(
        dino_extractor, frame_rgb, cfg["patch"], cfg["target_h"], bool(cfg["norm"]) )

    # Resize depth & mask to resized frame size
    H_env, W_env = frame_resized.size[1], frame_resized.size[0]
    target_size = (W_env, H_env)

    depth_t = torch.tensor(frame_depth, device=device, dtype=torch.float32)
    if frame_mask is not None:
        mask_t = torch.tensor(frame_mask, device=device, dtype=torch.bool)
    else:
        mask_t = torch.ones_like(depth_t, device=device, dtype=torch.bool)

    depth_resz = F.interpolate(depth_t.unsqueeze(0).unsqueeze(0), size=(target_size[1], target_size[0]),
                               mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    mask_resz = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0).float(), size=(target_size[1], target_size[0]),
                              mode="nearest").squeeze(0).squeeze(0).bool()

    coeffs, probs_pdf, expectation, dists = forward_riskfield_dense(
        risk_model, frame_dino, manip_dino, num_ctrl_pts, device, max_dist)

    _, exp_arr, _ = generate_risk_image(expectation, frame_resized, cfg)
    risk_at_depth_np = evaluate_riskfield_depth(coeffs, depth_resz, mask_resz, dists, max_dist)

    # ---- Save outputs with foldered structure ----
    # npy
    np.save(out_dirs["npy"] / f"{base_name}.npy", coeffs.cpu().numpy())

    # images (colored) using existing helpers
    plot_risk_image(exp_arr, cfg, str(out_dirs["expectation"] / f"{base_name}.png"),
                    max_dist, label="risk_expectation")
    plot_depth_risk_image(risk_at_depth_np, cfg, str(out_dirs["depth_risk"] / f"{base_name}.png"), save_npy=False)

    # Optional: small debug image of probs (first pixel histogram) if save_probs
    if bool(cfg.get("save_probs", False)):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm as _cm
            Hf, Wf, S = probs_pdf.shape
            arr = probs_pdf[min(0, Hf-1), min(0, Wf-1)].detach().cpu().numpy()
            plt.figure(figsize=(4, 3))
            plt.plot(arr)
            plt.title("PDF (debug pixel)")
            plt.tight_layout()
            plt.savefig(out_dirs["probs"] / f"{base_name}.png", dpi=120)
            plt.close()
        except Exception:
            pass

    return risk_at_depth_np

# ----------------- Main -----------------

def main():
    cfg, cfg_path = load_infer_cfg()
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"

    out_root = Path(cfg["out_dir"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # Save a copy of the config used
    shutil.copy2(str(cfg_path), str(out_root / "infer_risk_field_vid_lite_cfg.yaml"))

    # Make subfolders
    out_dirs = _make_out_dirs(out_root)

    # Prepare models
    dino_extractor = DinoFeatures(stride=cfg["stride"], norm=bool(cfg["norm"]), patch_size=cfg["patch"], device=device)
    risk_model, num_ctrl_pts = load_model_riskfield(cfg, device)

    # Manipulated object B
    manip_img = np.array(Image.open(cfg["image_b"]).convert("RGB"), dtype=np.uint8)
    manip_resized, manip_rgba, featsManip = extract_dino_inputs_and_features(
        dino_extractor, manip_img, cfg["patch"], cfg["target_h"], bool(cfg["norm"]))
    manip_dino = compute_average_vector_dino(featsManip, manip_rgba, cfg, device)

    # Load inputs
    frame_rgb = np.array(Image.open(cfg["env_rgb"]).convert("RGB"), dtype=np.uint8)
    frame_depth_img = Image.open(cfg["env_depth"])  # 16-bit
    frame_depth = np.array(frame_depth_img, dtype=np.float16) / 1000.0
    if cfg["env_depth_mask"] is not None:
        print(f"Loading mask from {cfg['env_depth_mask']}")
        mask_img = Image.open(cfg["env_depth_mask"])  # white=True
        mask_arr = np.array(mask_img)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        frame_mask = (mask_arr > 127)
    else:
        frame_mask = None

    base_name = Path(cfg["env_rgb"]).stem  # e.g., 000003

    # Run
    _ = runner(frame_rgb, frame_depth, frame_mask, manip_dino, risk_model, dino_extractor,
               num_ctrl_pts, cfg["max_distance"], device, out_dirs, cfg, base_name, verbose=False)

    print(f"[DONE] Saved to {out_root} (expectation/, depth_risk/, npy/)")


if __name__ == "__main__":
    main()
