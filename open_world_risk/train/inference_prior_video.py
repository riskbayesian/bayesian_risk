# inference_prior_video.py  — prior-only risk image/video export

import os, sys, shutil
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# --- project imports (keep your package layout) ---
THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.insert(0, str(ROOT))

from open_world_risk.common_sense_priors.model_prior import PriorModel
from open_world_risk.src.dino_features import (
    DinoFeatures,
    resize_to_multiple_of,
    nonwhite_nontransparent_mask,
)
from open_world_risk.train.inference_riskfield_video_utils import (
    plot_prior_image,
    create_video_output,
    load_img_png,
    load_mov_file,
)

EXPECTED_DINO_DIM = 1024


# -----------------------
# DINO helpers (unchanged)
# -----------------------
def extract_dino_inputs_and_features(dino_extractor, image_array, patch_size, target_height, norm_features=True):
    assert image_array.ndim == 3 and image_array.shape[2] == 3, f"Expected [H,W,3] RGB, got {image_array.shape}"
    alpha = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=np.uint8)
    img_array_rgba = np.concatenate([image_array, alpha], axis=2)
    img = Image.fromarray(img_array_rgba, "RGBA")
    resized_image, _ = resize_to_multiple_of(img, patch_size, target_height)

    # speed flags
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    rgba_tensor = dino_extractor.rgb_image_preprocessing(np.array(img), target_h=target_height)
    rgb_tensor = rgba_tensor[:3]
    dino_features = dino_extractor.get_dino_features_rgb(rgb_tensor, norm=norm_features).float()  # [C,H,W]
    return resized_image, rgba_tensor, dino_features


def compute_average_vector_dino(dinoFeats, rgba_im, cfg, device):
    mask = nonwhite_nontransparent_mask(
        rgba_im.to(device), tol_white=cfg["mask_white"], tol_alpha=cfg["mask_alpha"]
    )  # [H,W]
    flat = dinoFeats.permute(1, 2, 0).reshape(-1, dinoFeats.shape[0])  # [HW,C]
    vec = flat[mask.view(-1)].mean(0) if mask.any() else flat.mean(0)
    return F.normalize(vec, dim=0)  # [1024]


# -----------------------
# Prior model only
# -----------------------
def load_model_prior(cfg, device):
    with open(cfg["prior_cfg"], "r") as f:
        model_config = yaml.safe_load(f)

    train_cfg = model_config["training"]
    model_cfg = model_config["model"]
    ckpt = torch.load(cfg["prior_ckpt"], map_location="cpu")

    model = PriorModel(
        clip_dim=train_cfg["feat_size"],
        num_layers=model_cfg["num_layers"],
        num_neurons=model_cfg["num_neurons"],
        projection_dim=model_cfg["projection_dim"],
        alphas_dim=model_cfg["ctrl_pts"],
        monotonic=model_cfg["monotonic"],
        num_layers_trans=model_cfg["num_layers_trans"],
        num_heads=model_cfg["num_heads"],
        attn_dropout=model_cfg["attn_dropout"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    return model


@torch.inference_mode()
def generate_prior_image(prior_model, featsA, vecB, device):
    """Dense prior over tokens → [H,W]"""
    Hf, Wf = featsA.shape[1:]
    A = featsA.permute(1, 2, 0).reshape(-1, featsA.shape[0])         # [N, C]
    B = vecB.unsqueeze(0).expand(A.size(0), -1).contiguous()          # [N, C]

    outs = []
    bs = 8192
    for i in range(0, A.size(0), bs):
        fa = A[i:i+bs].to(device, non_blocking=True)
        fb = B[i:i+bs].to(device, non_blocking=True)
        pred = prior_model(fa, fb)
        pred = pred[0] if isinstance(pred, tuple) else pred           # allow (alphas, ...) tuples
        outs.append(pred)

    prior_vals = torch.cat(outs, dim=0).view(Hf, Wf, -1)              # [H,W,K or 1]
    if prior_vals.shape[-1] == 1:
        prior_vals = prior_vals.squeeze(-1)                           # [H,W]
    else:
        # if model outputs 5 control points in [0,1], take a scalar prior proxy (e.g., mean)
        raise
        prior_vals = prior_vals.argmax(dim=-1).to(torch.float32)                          # [H,W]
    return prior_vals


def upsample_to_image(grid_tensor_hw, target_pil_img, align_corners=False):
    Ha, Wa = target_pil_img.size[1], target_pil_img.size[0]
    if grid_tensor_hw.shape[-2:] != (Ha, Wa):
        grid_tensor_hw = F.interpolate(
            grid_tensor_hw.unsqueeze(0).unsqueeze(0),
            size=(Ha, Wa),
            mode="bilinear",
            align_corners=align_corners,
        ).squeeze(0).squeeze(0)
    return grid_tensor_hw


# -----------------------
# Config & runner
# -----------------------
def load_infer_cfg(config_path: str = "infer_risk_field_vid_cfg.yaml"):
    # You can keep your existing YAML; only these fields are required now.
    if not os.path.isabs(config_path):
        config_path = ROOT / config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    required = [
        "image_a", "image_b", "prior_ckpt", "prior_cfg",
        "dino_weights", "out_dir", "device",
        "target_h", "patch", "stride",
        "mask_white", "mask_alpha", "norm",
        "upsample_to_image", "cmap", "overlay_alpha", "scale",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # optional
    cfg.setdefault("num_frames", None)
    return cfg, config_path


def runner(env_frames_array, manip_img_array, prior_model, dino_extractor, device, cfg, label=""):
    # Manipulation (object B) → average DINO feature
    manip_resized, manip_rgba, featsManip = extract_dino_inputs_and_features(
        dino_extractor, manip_img_array, cfg["patch"], cfg["target_h"], bool(cfg["norm"])
    )
    vecManip = compute_average_vector_dino(featsManip, manip_rgba, cfg, device)

    # Process environment frames
    N = env_frames_array.shape[0]
    print(f"Found {N} environment frames to process")

    prior_imgs = []
    env_resized_list = []

    for i in tqdm(range(N), desc="Prior-only: processing frames"):
        env_rgb = env_frames_array[i]
        env_resized, env_rgba, featsEnv = extract_dino_inputs_and_features(
            dino_extractor, env_rgb, cfg["patch"], cfg["target_h"], bool(cfg["norm"])
        )

        prior_hw = generate_prior_image(prior_model, featsEnv, vecManip, device)

        # optional normalization / upsample to full frame for viz
        if cfg["upsample_to_image"]:
            prior_hw = upsample_to_image(prior_hw, env_resized)

        prior_imgs.append(prior_hw.detach().cpu())
        env_resized_list.append(env_resized)

    # Plot frames (prior only) and assemble video
    for i in tqdm(range(N), desc="Prior-only: plotting frames"):
        H_env, W_env = env_resized_list[i].size[1], env_resized_list[i].size[0]
        plot_prior_image(prior_imgs[i], env_resized_list[i], cfg, H_env, W_env, frame_idx=i, label=label)

    print("\nCreating prior video from frames...")
    prior_video_path = create_video_output(
        filename_prefix="prior_model_matplotlib_",
        output_dir=cfg["out_dir"],
        fps=10,
        video_name=f"prior_model_video_{label}.mp4",
    )
    print(f"Prior video created: {prior_video_path}")
    print("Done.")


def main():
    cfg, config_path = load_infer_cfg()
    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"

    # output dir
    timestamp = datetime.now().strftime("%H%M_%m%d")
    cfg["out_dir"] = os.path.join(cfg["out_dir"], f"inference_prior_{timestamp}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    shutil.copy2(config_path, os.path.join(cfg["out_dir"], "infer_cfg_used.yaml"))

    manip_label = os.path.splitext(os.path.basename(cfg["image_b"]))[0]

    # models & features
    dino_extractor = DinoFeatures(stride=cfg["stride"], norm=bool(cfg["norm"]), patch_size=cfg["patch"], device=device)
    prior_model = load_model_prior(cfg, device)

    # load A as frames
    if cfg["image_a"].lower().endswith(".mov"):
        env_frames_array, meta = load_mov_file(cfg["image_a"])
        print(f"Video metadata: {meta}")
    elif cfg["image_a"].lower().endswith(".png"):
        env_frames_array, filenames = load_img_png(cfg["image_a"])
        print(f"Loaded PNG image: {filenames[0]}")
    else:
        raise ValueError(f"Unsupported file format for image_a: {cfg['image_a']} (use .mov or .png)")

    # subsample frames if requested
    if cfg["num_frames"] is not None:
        env_frames_array = env_frames_array[: cfg["num_frames"]]

    manip_img_array = np.array(Image.open(cfg["image_b"]).convert("RGB"), dtype=np.uint8)
    runner(env_frames_array, manip_img_array, prior_model, dino_extractor, device, cfg, label=manip_label)


if __name__ == "__main__":
    main()
