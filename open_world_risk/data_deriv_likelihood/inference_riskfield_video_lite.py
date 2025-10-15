from re import L
import os, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib import cm  # for colormap lookups (no plotting windows)
import yaml
from sklearn.decomposition import PCA
from datetime import datetime
import glob
from tqdm import tqdm
import cv2
import functools
import time
import subprocess
import shutil
from pprint import pprint
import petname as PetnamePkg

# ---- Make `open_world_risk.utils.model` importable (run from workspace root or adjust) ----
THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.insert(0, str(ROOT))

from open_world_risk.utils.model import RiskField  # noqa
from open_world_risk.src.dino_features import DinoFeatures, resize_to_multiple_of, nonwhite_nontransparent_mask  # noqa
from open_world_risk.data_deriv_likelihood.inference_riskfield_video_utils import (
    plot_coefficents,
    plot_risk_image,
    plot_depth_risk_image,
    timeFunction,
    learn_pca_from_features,
    vis_dino_pca,
    bernstein_B,
    timeFunction,
)

EXPECTED_DINO_DIM = 1024

def extract_dino_inputs_and_features(dino_extractor: DinoFeatures, image_array: np.ndarray, patch_size: int, target_height: int, norm_features: bool = True):
    # Numpy array input - assume RGB format [H,W,3]
    assert len(image_array.shape) == 3 and image_array.shape[2] == 3, f"Expected numpy array with shape [H,W,3], got {image_array.shape}"
    
    # Convert RGB to RGBA by adding alpha channel
    alpha_channel = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=np.uint8)
    img_array = np.concatenate([image_array, alpha_channel], axis=2)
    img = Image.fromarray(img_array, 'RGBA')
    img_array_rgba = np.array(img)  # [H,W,4] RGBA format
    resized_image, (H, W) = resize_to_multiple_of(img, patch_size, target_height)

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass

    rgba_tensor = dino_extractor.rgb_image_preprocessing(img_array_rgba, target_h=target_height)
    rgb_tensor = rgba_tensor[:3]  # [3,H,W]
    dino_features = dino_extractor.get_dino_features_rgb(rgb_tensor, norm=norm_features)  # [C,H,W]
    dino_features = dino_features.float()

    return resized_image, rgba_tensor, dino_features

def compute_average_vector_dino(dinoFeats, rgba_im, cfg, device):
    """
    Compute the average vector for image B using its nonwhite/nontransparent mask.

    Args:
        featsB: DINO features for image B [C,Hb,Wb]
        b_rgba: RGBA tensor for image B [4,Hb,Wb]
        cfg: Configuration dict
        device: PyTorch device

    Returns:
        vecB: Normalized average vector [1024]
    """
    mask = nonwhite_nontransparent_mask(rgba_im.to(device), tol_white=cfg["mask_white"],
                                         tol_alpha=cfg["mask_alpha"])  # [Hb,Wb]
    flatDino = dinoFeats.permute(1,2,0).reshape(-1, dinoFeats.shape[0])       # [Hb*Wb,C]
    if mask.any():
        vecDino = flatDino[mask.view(-1)].mean(dim=0)
    else:
        vecDino = flatDino.mean(dim=0)
    vecDino = F.normalize(vecDino, dim=0)  # [1024]
    return vecDino

def load_model_riskfield(cfg, device):
    """
    Load model configuration and initialize either RiskField or Prior model.

    Args:
        cfg: Configuration dict
        device: PyTorch device
        model_type: "riskfield" or "prior"

    Returns:
        model: Loaded model
        num_ctrl_pts: Number of control points
    """

    model_config_path = cfg["riskfield_cfg"]
    checkpoint_path = cfg["riskfield_ckpt"]

    # Load model configuration from external YAML file
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    tensor_layout = model_config["tensor_layout"]
    model_params = model_config["model"]
    num_ctrl_pts = tensor_layout["ctrl_pts"]
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    model = RiskField(
        clip_dim=tensor_layout["feat_size"],  # feat_size from tensor_layout
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
                                dists: torch.Tensor, max_dist: float) -> torch.Tensor:
    """
    Evaluate risk field at specific depth values using Bezier coefficients.
    
    Args:
        coeffs: Bezier coefficients [H, W, num_ctrl_pts] - pixelwise values for Bezier curve
        depth_frame: Depth image [H, W] in meters
        depth_mask_frame: Depth validity mask [H, W] where 1=valid, 0=invalid
        dists: Distance bins [S] used for Bezier curve evaluation
        
    Returns:
        risk_values: Risk field values at depth locations [H, W]
    """
    import math
    H, W = coeffs.shape[:2]
    num_ctrl_pts = coeffs.shape[2]
    device = coeffs.device

    # Ensure computation in float32 for numerical stability
    depth_frame = depth_frame.to(device=device, dtype=torch.float32)
    depth_mask_frame = depth_mask_frame.to(device=device)
    coeffs = coeffs.to(device=device, dtype=torch.float32)

    risk_values = torch.zeros_like(depth_frame, dtype=torch.float32)
    valid_mask = depth_mask_frame.bool()
    assert valid_mask.any(), "Invalid Mask!"

    # Gather valid depths and clamp to [0, max_dist]
    valid_depths = depth_frame[valid_mask]
    valid_depths = torch.clamp(valid_depths, 0.0, float(max_dist))

    # Normalize to t in [0, 1]
    t_values = valid_depths / float(max_dist)  # [N_valid]

    # Coefficients corresponding to valid pixels: [N_valid, num_ctrl_pts]
    valid_coeffs = coeffs[valid_mask]

    # Vectorized Bernstein basis computation
    # basis[:, i] = C(n-1, i) * t^i * (1-t)^(n-1-i)
    i_idx = torch.arange(num_ctrl_pts, device=device, dtype=torch.float32)
    binom = torch.tensor([math.comb(num_ctrl_pts - 1, int(i)) for i in range(num_ctrl_pts)],
                         device=device, dtype=torch.float32)
    t_col = t_values.unsqueeze(1)  # [N_valid, 1]
    basis = binom * (t_col ** i_idx) * ((1.0 - t_col) ** (num_ctrl_pts - 1 - i_idx))  # [N_valid, num_ctrl_pts]

    # Weighted sum across control points
    risk_at_depths = (valid_coeffs * basis).sum(dim=1)

    # Scatter back to full image
    risk_values[valid_mask] = risk_at_depths

    risk_values_np = risk_values.detach().cpu().numpy()
    return risk_values_np

def forward_riskfield_dense(model, featsA, vecB, num_ctrl_pts, device, max_distance, verbose=False):
    """
    Run batched forward passes over dense tokens and compute:
      - coeffs: [Hf,Wf,num_ctrl_pts]
      - probs:  [Hf,Wf,S] with S=100
      - exp_un: [Hf,Wf] unnormalized expected distance

    Args:
        model: RiskField model
        featsA: [C,Hf,Wf]
        vecB: [C]
        num_ctrl_pts: number of Bezier control points
        device: torch device
        max_distance: max distance
    """
    Hf, Wf = featsA.shape[1:]
    A = featsA.permute(1,2,0).reshape(-1, featsA.shape[0])                   # [Ntok,1024]
    B = vecB.unsqueeze(0).expand(A.size(0), -1).contiguous()                 # [Ntok,1024]

    outs = []
    with torch.inference_mode():
        bs = 8192
        for i in range(0, A.size(0), bs):
            fa = A[i:i+bs].to(device, non_blocking=True)
            fb = B[i:i+bs].to(device, non_blocking=True)
            pred = model(fa, fb)
            pred = pred[0] if isinstance(pred, tuple) else pred
            outs.append(pred)
    
    coeffs = torch.cat(outs, dim=0).view(Hf, Wf, num_ctrl_pts).contiguous()             # [Hf,Wf,num_ctrl_pts]
    Bbern, t_values = bernstein_B(num_ctrl_pts, S=100, device=device, dtype=torch.float32)     # [100,num_ctrl_pts]
    dists = max_distance * t_values                                             # [100]
    probs = torch.einsum("hwk,sk->hws", coeffs.to(device), Bbern).clamp(0, 1)  # [Hf,Wf,S] (S=100)
    #exp_un = torch.einsum("hws,s->hw", probs, dists).contiguous()                 # [Hf,Wf]

    # Get PDF from CDF using Finite Difference Method
    dt = torch.diff(t_values, prepend=t_values[:1]).to(probs.device)        # [S]
    F_diff = torch.diff(probs, dim=-1)  # [H, W, S-1]
    dt_diff = dt[1:]  # [S-1] - skip first dt since we don't have F[-1]
    
    # Pad to match original size
    pdf_padded = torch.zeros_like(probs)
    pdf_padded[..., 1:] = F_diff / dt_diff.view(1, 1, -1)
    pdf_padded[..., 0] = probs[..., 0]
    
    probs_pdf = pdf_padded.clamp_min(0).clamp_max(1e6)
    probs_pdf_sum = probs_pdf.sum(dim=-1)  # [Hf, Wf] Compute sum of probs_pdf for each pixel (sum over distance bins)
    probs_pdf = probs_pdf / probs_pdf_sum.unsqueeze(-1).clamp(min=1e-8)  # [Hf, Wf, S] # Normalize
    expectation = torch.einsum("hws,s->hw", probs_pdf, dists).contiguous()             # [Hf,Wf]

    # Quick matplotlib plot of probs_pdf[0,0] and probs[0,0] for debugging
    # Compute empirical CDF from PDF with numerical safeguards
    if verbose:
        pdf_safe = probs_pdf[0, 0].clamp(min=0)  # Ensure non-negative
        dt_safe = dt.clamp(min=1e-8)  # Avoid division by zero
        cdf_empirical = torch.cumsum(pdf_safe * dt_safe, dim=0) * 100 # 100 for scaling since we approx divide by 100 when norming
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_values.cpu().numpy(), probs_pdf[0, 0].cpu().numpy(), 'b-', linewidth=2, label='PDF (derivative)')
        plt.plot(t_values.cpu().numpy(), probs[0, 0].cpu().numpy(), 'r-', linewidth=2, label='CDF (original)')
        plt.plot(t_values.cpu().numpy(), cdf_empirical.cpu().numpy(), 'g--', linewidth=2, label='CDF (empirical from PDF)')
        plt.xlabel('t (normalized distance)')
        plt.ylabel('Value')
        plt.title('Probability Functions at pixel (0,0)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('probs_comparison_pixel_0_0.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved probability comparison plot to: probs_comparison_pixel_0_0.png check your current working dir")

    return coeffs, probs_pdf, expectation, dists

def generate_risk_image(exp_un: torch.Tensor, a_resized: Image.Image, cfg: dict) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Generate the risk image tensor aligned to the resized image and return
    both the raw values and a [0,1] normalized array for visualization.

    Returns:
        exp_img: torch.Tensor [Ha,Wa]
        exp_arr: np.ndarray float32 [Ha,Wa]
        norm:    np.ndarray float32 [Ha,Wa] in [0,1]
    """
    # Extract dimensions from PIL image (width, height) -> (height, width)
    Ha, Wa = a_resized.size[1], a_resized.size[0]
    
    # Upsample to resized image grid if requested
    if cfg["upsample_to_image"]:
        exp_img = F.interpolate(exp_un.unsqueeze(0).unsqueeze(0), size=(Ha, Wa),
                                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    else:
        exp_img = exp_un

    # Ensure match with actual resized PIL dims for overlays
    HA_img, WA_img = a_resized.size[1], a_resized.size[0]
    if exp_img.shape[-2:] != (HA_img, WA_img):
        exp_img = F.interpolate(exp_img.unsqueeze(0).unsqueeze(0), size=(HA_img, WA_img),
                                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    exp_arr = exp_img.detach().cpu().numpy().astype(np.float32)
    if cfg["scale"] == "fixed":
        max_val = 100
        norm = exp_arr / max_val
    else:
        denom = (exp_arr.max() - exp_arr.min()) if (exp_arr.max() > exp_arr.min()) else 1.0
        norm = (exp_arr - exp_arr.min()) / denom
    return exp_img, exp_arr, norm

def load_infer_cfg(config_path: str = "infer_risk_field_vid_lite_cfg.yaml") -> dict:
    # Resolve config path relative to script directory
    if not os.path.isabs(config_path):
        script_dir = ROOT
        config_path = script_dir / config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    required = [
        "image_b", "riskfield_ckpt", "riskfield_cfg", "dino_weights", "out_path", "device",
        "target_h", "patch", "stride", "mask_white", "mask_alpha", "norm",
        "save_probs", "upsample_to_image", "cmap", "overlay_alpha", "scale"
    ]
    
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # videos_range is optional; if missing, leave as None
    cfg.setdefault("videos_range", None)
    return cfg, config_path

def runner(frame_rgb, frame_depth, frame_mask, manip_dino, risk_model, 
            dino_extractor, num_ctrl_pts, max_dist, device, output_path_np, cfg, verbose=False):

    frame_resized, frame_rgba, frame_dino = extract_dino_inputs_and_features(
            dino_extractor, frame_rgb, cfg["patch"], cfg["target_h"], bool(cfg["norm"])
        )

    H_env, W_env = frame_resized.size[1], frame_resized.size[0]
    target_size = (W_env, H_env)  # (width, height)

    print(f"frame_depth_shape {frame_depth.shape}")
    print(f"frame_mask.shape {frame_mask.shape}")

    # Convert numpy arrays to PyTorch tensors
    frame_depth_tensor = torch.tensor(frame_depth, device=device, dtype=torch.float32)
    frame_mask_tensor = torch.tensor(frame_mask, device=device, dtype=torch.bool)

    frame_depth_resized = F.interpolate(
        frame_depth_tensor.unsqueeze(0).unsqueeze(0), 
        size=(target_size[1], target_size[0]),  # (height, width) for F.interpolate
        mode="bilinear", 
        align_corners=False
    ).squeeze(0).squeeze(0)
    
    frame_mask_resized = F.interpolate(
        frame_mask_tensor.unsqueeze(0).unsqueeze(0).float(), 
        size=(target_size[1], target_size[0]),  # (height, width) for F.interpolate
        mode="nearest"
    ).squeeze(0).squeeze(0).bool()

    coeffs, probs_pdf, expectation, dists = forward_riskfield_dense(risk_model, frame_dino, manip_dino, num_ctrl_pts, device, max_dist)
    exp_img, exp_arr, norm = generate_risk_image(expectation, frame_resized, cfg)
    risk_at_depth_np = evaluate_riskfield_depth(coeffs, frame_depth_resized, frame_mask_resized, dists, max_dist)

    assert output_path_np.endswith('.npy'), f"output_path_np must end with .npy, got: {output_path_np}"
    np.save(output_path_np, risk_at_depth_np)

    # Generate output paths based on output_path_np
    base_path = os.path.splitext(output_path_np)[0]  # Remove .npy extension
    output_path_expect = f"{base_path}_expectation.png"
    output_path_risk_d = f"{base_path}_risk_depth.png"
    
    plot_risk_image(exp_arr, cfg, output_path_expect, max_dist, label="risk_expectation")
    plot_depth_risk_image(risk_at_depth_np, cfg, output_path_risk_d, save_npy=False)
    return risk_at_depth_np
    
def get_manip_tensor_label(manip_path, dino_extractor, device, cfg):

    manip_img_array = np.array(Image.open(manip_path).convert("RGB"), dtype=np.uint8)
    manip_resized, manip_rgba, featsManip = extract_dino_inputs_and_features(
        dino_extractor, manip_img_array, cfg["patch"], cfg["target_h"], bool(cfg["norm"])
    )
    manip_dino = compute_average_vector_dino(featsManip, manip_rgba, cfg, device)
    manip_label = os.path.splitext(os.path.basename(manip_path))[0]

    return manip_dino, manip_label

def compute_target_size_from_h(root_dir: str, target_h: int, make_multiple_of: int | None = None) -> tuple[int, int]:
    """
    Compute (target_w, target_h) from the first frame's aspect ratio under root_dir/*/*/rgb_png.
    Optionally snap both dimensions to a multiple (e.g., patch size).
    """
    rgb_dirs = sorted(glob.glob(os.path.join(root_dir, "*", "*", "rgb_png")))
    if not rgb_dirs:
        raise FileNotFoundError(f"No rgb_png dirs under: {root_dir}")
    first_pngs = sorted(glob.glob(os.path.join(rgb_dirs[0], "*.png")))
    if not first_pngs:
        raise FileNotFoundError(f"No PNGs in: {rgb_dirs[0]}")
    w0, h0 = Image.open(first_pngs[0]).size
    target_w = int(round(w0 * (target_h / h0)))
    if make_multiple_of and make_multiple_of > 1:
        target_w = (target_w // make_multiple_of) * make_multiple_of
        target_h = (target_h // make_multiple_of) * make_multiple_of

    print(f"Target size found: width {target_w} height {target_h}")
    print(f"Original size: width {w0} height {h0}")
    return (target_w, target_h)

def load_img_png(image_path, target_size=None):
    img = Image.open(image_path).convert("RGB")

    if target_size is None:
        target_size = img.size  # (width, height)
    else:
        target_size = tuple(target_size)
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.uint8)  # [H, W, 3]d
    filename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Loaded: {filename} - Shape: {img_array.shape}")

    return img_array, filename

def load_image_depth(depth_path, target_size=None):
    """
    Load a single depth image.
    
    Args:
        depth_path: Path to depth image file
        target_size: Target size for image (width, height) or None to keep original
        
    Returns:
        depth_array: numpy array of depth frame in meters [H, W]
        filename: filename
    """
    
    assert os.path.exists(depth_path), f"Depth image not found: {depth_path}"
    
    # Load 16-bit depth image
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img, dtype=np.float16)
    
    # Convert from mm to meters (divide by 1000)
    depth_array = depth_array / 1000.0

    # Resize if requested (target_size is (width, height))
    if target_size is not None:
        target_w, target_h = tuple(target_size)
        # Use linear interpolation for depth values
        depth_array = cv2.resize(depth_array.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    filename = os.path.basename(depth_path)
    return depth_array, filename

def load_image_mask(mask_path, target_size=None):
    """
    Load a single depth mask image.
    
    Args:
        mask_path: Path to mask image file
        target_size: Target size for image (width, height) or None to keep original
        
    Returns:
        mask_array: numpy array of boolean mask [H, W] where white=True, black=False
        filename: filename
    """
    
    assert os.path.exists(mask_path), f"Mask image not found: {mask_path}"
    
    # Load mask image
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)
    
    # Convert to boolean: white (255) = True, black (0) = False
    # Handle both grayscale and RGB images
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    
    # Resize if requested (target_size is (width, height))
    if target_size is not None:
        target_w, target_h = tuple(target_size)
        mask_array = cv2.resize(mask_array, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    mask_bool = mask_array > 127  # Threshold at 127 (middle of 0-255)
    
    filename = os.path.basename(mask_path)
    return mask_bool, filename

def main():
    cfg, config_path = load_infer_cfg()
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    timestamp = datetime.now().strftime("%H%M_%m%d")
    
    os.makedirs(cfg["out_dir"], exist_ok=True)
    config_dest_path = os.path.join(cfg["out_dir"], "infer_risk_field_vid_cfg.yaml")
    shutil.copy2(config_path, config_dest_path)
    print(f"Copied config to output directory: {config_dest_path}")
    
    max_dist = cfg["max_distance"]
    output_path_np = os.path.join(cfg["out_dir"], cfg["out_path"])

    dino_extractor = DinoFeatures(stride=cfg["stride"], norm=bool(cfg["norm"]), patch_size=cfg["patch"], device=device)
    risk_model, num_ctrl_pts = load_model_riskfield(cfg, device)
    manip_dino, manip_label = get_manip_tensor_label(cfg["image_b"], dino_extractor, device, cfg)
    
    frame_rgb, _ = load_img_png(cfg["env_rgb"])
    frame_depth, _ = load_image_depth(cfg["env_depth"])
    frame_mask, _ = load_image_mask(cfg["env_depth_mask"])

    runner(frame_rgb, frame_depth, frame_mask, manip_dino, risk_model, 
            dino_extractor, num_ctrl_pts, max_dist, device, output_path_np, cfg, verbose=False)

if __name__ == "__main__":
    main()