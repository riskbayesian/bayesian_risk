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
    create_video_output, 
    load_img_png,
    load_dir_img,
    load_mov_file,  
    bernstein_B,
    load_dir_img_data_09_22_2025_wrapper,
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

def evaluate_riskfield_depth(coeffs, depth_frame, depth_mask_frame, dists, max_dist):
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

def load_infer_cfg(config_path: str = "infer_risk_field_vid_cfg.yaml") -> dict:
    # Resolve config path relative to script directory
    if not os.path.isabs(config_path):
        script_dir = ROOT
        config_path = script_dir / config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    required = [
        "image_a", "image_b", "riskfield_ckpt", "riskfield_cfg", "dino_weights", "out_dir", "device",
        "target_h", "patch", "stride", "mask_white", "mask_alpha", "norm",
        "save_probs", "upsample_to_image", "cmap", "overlay_alpha", "scale"
    ]
    
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # videos_range is optional; if missing, leave as None
    cfg.setdefault("videos_range", None)
    return cfg, config_path

def runner(rgb_frames, depth_frames, depth_mask_frames, vid_label, manips_tensor, manips_labels, risk_model, 
            dino_extractor, num_ctrl_pts, max_dist, device, output_dict, cfg, batch_size, create_vid=False, verbose=False):

    assert len(rgb_frames) == len(depth_frames), "Mismatch in rgb_frames and depth_frames length"
    assert len(rgb_frames) == len(depth_mask_frames), "Mismatch in rgb_frames and depth_mask_frames length"

    H_env, W_env = None, None # assume const
    label_set = set()

    for batch_start in tqdm(range(0, len(rgb_frames), batch_size), desc=f"Processing {vid_label} batches"):
        batch_end = min(batch_start + batch_size, len(rgb_frames))
        batch_rgb_frames        = rgb_frames[batch_start:batch_end]
        batch_depth_frames      = depth_frames[batch_start:batch_end]
        batch_depth_mask_frames = depth_mask_frames[batch_start:batch_end]
        
        # Key: manipulated object label; Value: List of values wrt each frame
        # These are lists of dictionaries
        coeffs_data = []
        probs_pdf_data = []
        frame_resized_lst = []
        norm_data = []
        expect_arr_data = []
        risk_d_data = []
        frame_lst = [i for i in range(batch_start, batch_end)]

        for batch_idx in range(batch_end - batch_start):
            frame_idx = batch_start + batch_idx  # Absolute frame index
            frame_rgb = batch_rgb_frames[batch_idx]
            frame_depth = torch.tensor(batch_depth_frames[batch_idx], device=device, dtype=torch.float16)
            frame_mask = torch.tensor(batch_depth_mask_frames[batch_idx], device=device, dtype=torch.bool)
            frame_resized, frame_rgba, frame_dino = extract_dino_inputs_and_features(
                    dino_extractor, frame_rgb, cfg["patch"], cfg["target_h"], bool(cfg["norm"])
                )

            assert frame_resized.size[1] == frame_rgb.shape[0], "size changed Height"
            assert frame_resized.size[0] == frame_rgb.shape[1], "size changed Width"
            if H_env is None and W_env is None:
                H_env, W_env = frame_resized.size[1], frame_resized.size[0]

            frame_resized_lst.append(frame_resized)
            coeffs_dict = dict()
            probs_pdf_dict = dict()
            expect_arr_dict = dict()
            norm_dict = dict()
            risk_d_dict = dict()

            for manips_idx, manip_dino in enumerate(manips_tensor):
                coeffs, probs_pdf, expectation, dists = forward_riskfield_dense(risk_model, frame_dino, manip_dino, num_ctrl_pts, device, max_dist)
                exp_img, exp_arr, norm = generate_risk_image(expectation, frame_resized, cfg)
                risk_at_depth = evaluate_riskfield_depth(coeffs, frame_depth, frame_mask, dists, max_dist)

                key = manips_labels[manips_idx]
                coeffs_dict[key] = coeffs
                probs_pdf_dict[key] = probs_pdf
                expect_arr_dict[key] = exp_arr
                norm_dict[key] = norm
                risk_d_dict[key] = risk_at_depth

                expectation = None
                exp_img = None
                del expectation, exp_img

            coeffs_data.append(coeffs_dict)
            probs_pdf_data.append(probs_pdf_dict)
            expect_arr_data.append(expect_arr_dict)
            norm_data.append(norm_dict)
            risk_d_data.append(risk_d_dict)

        assert H_env is not None, "Improper H_env init in runner()"
        assert W_env is not None, "Improper W_env init in runner()"
        
        for idx, frame_idx in enumerate(frame_lst):
            coeffs_dict = coeffs_data[idx]
            probs_pdf_dict = probs_pdf_data[idx]
            norm_dict = norm_data[idx]
            expect_arr_dict = expect_arr_data[idx]
            risk_d_dict = risk_d_data[idx]

            for manips_idx, manip_label in enumerate(manips_labels):
                label = f"_{vid_label}_{manip_label}"
                label_set.add(label)

                key = manip_label
                manip_dir = output_dict[key]
                output_path_expect = os.path.join(manip_dir, f"{manip_label}_expectation_frame_{frame_idx:06d}.png")
                output_path_risk_d = os.path.join(manip_dir, f"{manip_label}_risk_wrt_depth_frame_{frame_idx:06d}.png")
                
                plot_risk_image(expect_arr_dict[key], cfg, output_path_expect, max_dist, label=label)
                plot_depth_risk_image(risk_d_dict[key], cfg, output_path_risk_d, save_npy=False)
                #plot_coefficents(coeffs_dict[key], device, num_ctrl_pts, output_dict[manip_label], max_dist, frame_idx=frame_idx)
                
                if verbose:
                    print(f"Risk Expectation saved: {output_path_expect}")
                    print(f"Risk wrt Depth saved:   {output_path_risk_d}")

        frame_depth = None
        frame_mask = None 
        del frame_depth, frame_mask

        coeffs_data = None
        probs_pdf_data = None
        expect_arr_data = None
        frame_resized_lst = None
        norm_data = None
        del coeffs_data
        del probs_pdf_data
        del expect_arr_data
        del frame_resized_lst
        del norm_data
        
    print(f"Completed video {vid_label}")
    if create_vid:
        for label in label_set:
            # Extract vid_label and manip_label from the label
            # label format: _vid_label_manip_label
            parts = label.split('_')
            if len(parts) >= 3:
                vid_label = parts[1]
                manip_label = '_'.join(parts[2:])  # In case manip_label has underscores
                
                # Get the output directory for this video and manipulation
                if vid_label in output_dict and manip_label in output_dict[vid_label]:
                    manip_dir = output_dict[vid_label][manip_label]
                    risk_video_path = create_video_output(
                        filename_prefix=f"{manip_label}_expectation_frame",
                        output_dir=manip_dir,
                        fps=10,
                        video_name=f"{manip_label}_expectation_video.mp4"
                    )
                    print(f"Risk field video created: {risk_video_path}")
                else:
                    print(f"Warning: Could not find output directory for {vid_label}/{manip_label}")
        print("Video creation completed!")
    

def get_manip_tensor_labels(manip_cfg_paths, dino_extractor, device, cfg):
    manip_paths_lst = manip_cfg_paths if isinstance(manip_cfg_paths, list) else [manip_cfg_paths]
    manip_paths_lst = sorted(manip_paths_lst)
    manip_dino_lst = []

    for i, manip_path in tqdm(enumerate(manip_paths_lst), desc="Processing manipulation images", total=len(manip_paths_lst)):
        manip_img_array = np.array(Image.open(manip_path).convert("RGB"), dtype=np.uint8)
        manip_resized, manip_rgba, featsManip = extract_dino_inputs_and_features(
            dino_extractor, manip_img_array, cfg["patch"], cfg["target_h"], bool(cfg["norm"])
        )
        manip_dino = compute_average_vector_dino(featsManip, manip_rgba, cfg, device)
        manip_dino_lst.append(manip_dino)

    manips_tensor = torch.stack(manip_dino_lst)  # [num_manips, 1024]
    manips_labels = [os.path.splitext(os.path.basename(path))[0] for path in manip_paths_lst]

    return manips_tensor, manips_labels

def get_image_or_video(image_a_input: str, target_size=None, frame_range=None, video_range=None, folder_specified=""):

    if not (os.path.isdir(image_a_input) and "data_09_22_2025" in image_a_input):
        raise ValueError(f"Unsupported file format for image_a: {image_a_input}. Only data_09_22_2025 directory supported.")
    
    # Use the wrapper function for data_09_22_2025 directory
    print(f"Loading all rgb_png directories from: {image_a_input}")
    results, rgb_frames, depth_frames, depth_mask_frames, dir_names = load_dir_img_data_09_22_2025_wrapper(image_a_input, 
                                                                target_size=target_size, folder_specified=folder_specified, 
                                                                frame_range=frame_range, video_range=video_range)

    assert len(rgb_frames) > 0, "No valid rgb_png directories found in data_09_22_2025"
    
    return results, rgb_frames, depth_frames, depth_mask_frames, dir_names

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

def process_videos_in_batches(image_a_input, videos_range, video_batch_size, manips_tensor, manips_labels, 
                            risk_model, dino_extractor, num_ctrl_pts, max_dist, device, cfg, target_size,
                            frame_batch_size, frame_range=None, folder_specified=""):
    """
    Process videos in batches using get_image_or_video.
    
    Args:
        image_a_input: Path to the data directory
        num_videos: Total number of videos to process
        video_batch_size: Number of videos to process in each batch
        manips_tensor: Manipulation tensor
        manips_labels: Manipulation labels
        risk_model: Risk field model
        dino_extractor: DINO feature extractor
        num_ctrl_pts: Number of control points
        max_dist: Maximum distance
        device: PyTorch device
        cfg: Configuration dictionary
        frame_range: Frame range for each video
        folder_specified: Specific folder to process
        
    Returns:
        all_dir_names: Combined directory names from all batches
    """
    all_dir_names = []
    
    if videos_range is None:
        # Use all videos found by the wrapper
        start_idx, end_idx = 0, None
    else:
        start_idx, end_idx = videos_range

    # Iterate batches over the specified index range
    current = start_idx
    batch_num = 1
    while True:
        if end_idx is None:
            batch_end = None if current is None else current + video_batch_size
        else:
            batch_end = min(current + video_batch_size, end_idx)

        video_range = (current, batch_end)
        
        print(f"\n=== Processing batch {current//video_batch_size + 1}: videos {current}-{batch_end-1 if batch_end else 'end'} ===")
        
        # Get videos for this batch
        results, rgb_frames, depth_frames, depth_mask_frames, dir_names = get_image_or_video(
            image_a_input, 
            target_size=target_size,
            frame_range=frame_range, 
            video_range=video_range,
            folder_specified=folder_specified
        )

        output_dict = create_output_directories(results, dir_names, manips_labels)

        for i, (rgb_frames_vid, depth_frames_vid, depth_mask_frames_vid) in enumerate(zip(rgb_frames, depth_frames, depth_mask_frames)):
            vid_label = dir_names[i]
            runner(rgb_frames_vid, depth_frames_vid, depth_mask_frames_vid, vid_label, manips_tensor, manips_labels, 
                        risk_model, dino_extractor, num_ctrl_pts, max_dist, device, output_dict[vid_label], cfg, 
                        batch_size=frame_batch_size, create_vid=False)

        all_dir_names.extend(dir_names)
        
        print(f"Batch completed: {len(rgb_frames)} videos processed")

        if end_idx is None:
            # We don't know total; stop after first batch to avoid reprocessing all
            break
        current = batch_end
        if current >= end_idx:
            break

    print(f"\n=== All batches completed: {len(all_dir_names)} total videos processed ===")
    return all_dir_names

def create_output_directories(results, dir_names, manips_labels):
    """
    Create output directories for each video and manipulation based on data structure.
    
    Args:
        results: Dictionary containing video metadata
        dir_names: List of video directory names
        manips_labels: List of manipulation labels
        
    Returns:
        output_dict: Nested dictionary [vid_label][manip_label] = output_path
    """
    output_dict = {}
    timestamp = datetime.now().strftime("%H%M%S_%m%d")
    petname = PetnamePkg.Generate()
    
    for i, vid_label in enumerate(dir_names):
        # Extract the parent directory path from the results
        rgb_png_dir = results[vid_label]['directory']  # e.g., data/data_09_22_2025/kitchen_full/kitchen_full/rgb_png
        parent_dir = os.path.dirname(rgb_png_dir)  # e.g., data/data_09_22_2025/kitchen_full/kitchen_full
        base_output_path = os.path.join(parent_dir, f"likelihood_out_{timestamp}_{petname}")
        
        # Create nested dictionary: output_dict[vid_label][manip_label] = output_path
        output_dict[vid_label] = {}
        print(f"Created output directories for {vid_label}")
        for manip_label in manips_labels:
            manip_dir = os.path.join(base_output_path, manip_label)
            output_dict[vid_label][manip_label] = manip_dir
            os.makedirs(manip_dir, exist_ok=True)
            print(f"\t{manip_dir}")
    
    return output_dict

def main():
    cfg, config_path = load_infer_cfg()
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    timestamp = datetime.now().strftime("%H%M_%m%d")
    cfg["out_dir"] = os.path.join(cfg["out_dir"], f"inference_{timestamp}")
    
    os.makedirs(cfg["out_dir"], exist_ok=True)
    config_dest_path = os.path.join(cfg["out_dir"], "infer_risk_field_vid_cfg.yaml")
    shutil.copy2(config_path, config_dest_path)
    print(f"Copied config to output directory: {config_dest_path}")
    
    max_dist = cfg["max_distance"]
    output_dir = cfg["out_dir"]
    video_batch_size = cfg["video_batch_size"]
    videos_range = cfg.get("videos_range", None)
    frame_range = cfg["frame_range"]
    frame_batch_size = cfg["frame_batch_size"]

    dino_extractor = DinoFeatures(stride=cfg["stride"], norm=bool(cfg["norm"]), patch_size=cfg["patch"], device=device)
    risk_model, num_ctrl_pts = load_model_riskfield(cfg, device)
    manips_tensor, manips_labels = get_manip_tensor_labels(cfg["image_b"], dino_extractor, device, cfg)


    # Compute target_size from cfg["target_h"] and align to patch size
    target_size = compute_target_size_from_h(cfg["image_a"], int(cfg["target_h"]), make_multiple_of=int(cfg["patch"]))

    dir_names = process_videos_in_batches(
        cfg["image_a"], 
        videos_range=videos_range, 
        video_batch_size=video_batch_size,
        manips_tensor=manips_tensor,
        manips_labels=manips_labels,
        risk_model=risk_model,
        dino_extractor=dino_extractor,
        num_ctrl_pts=num_ctrl_pts,
        max_dist=max_dist,
        device=device,
        cfg=cfg,
        target_size=target_size,
        frame_batch_size=frame_batch_size,
        frame_range=frame_range,
        folder_specified=""
    )

if __name__ == "__main__":
    main()