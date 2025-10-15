#!/usr/bin/env python3
"""
Batch prior inference over all rgb_png/ folders and multiple object_B values.

- Scans a root directory recursively for subfolders named `rgb_png` that contain images.
- For each found folder and each object_B in the config list, runs LUT prior inference.
- Saves outputs next to each `rgb_png/` as a sibling folder: `prior_{object_B}` (spaces -> underscores).
- Reuses your LUT pipeline (DINO, backends, saving layout, legends) from lookup_risk_lut.py.

Usage:
  python batch_prior_over_folders.py                     # reads CONFIG env or ./batch_prior.yaml
  CONFIG=/path/to/batch_prior.yaml python batch_prior_over_folders.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np
from PIL import Image
import time

# ---- Import your LUT pipeline ----
# Adjust the import below if your module path differs.
from open_world_risk.common_sense_priors.lookup_risk_lut import (
    load_yaml_config as _load_yaml_lut,   # not used here; we have our own loader
    ensure_out_dirs as lut_ensure_out_dirs,
    LookupRiskModel,
    _ensure_dino,
    _faiss_topk_chunked,
    _torch_topk_chunked,
    _prepare_backend,
)

import yaml
import torch

# Optional: matplotlib for colored maps (same as LUT script)
try:
    import matplotlib.cm as cm
    _HAVE_CM = True
except Exception:
    cm = None
    _HAVE_CM = False


# ------------------ Config ------------------

def load_config(explicit: str | None = None, default_name="batch_prior.yaml") -> tuple[dict, Path]:
    """
    If explicit provided, load that path (absolute/relative); else use $CONFIG or ./batch_prior.yaml
    Returns (cfg, cfg_dir)
    """
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
    else:
        envp = os.environ.get("CONFIG")
        cfg_path = Path(envp).expanduser().resolve() if envp else (Path(__file__).parent / default_name).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f), cfg_path.parent


# ------------------ IO helpers ------------------

def iter_rgb_png_roots(root_dir: Path, image_subdir: str = "rgb_png") -> List[Path]:
    """
    Recursively find subdirectories named `image_subdir` that contain images.
    """
    results: List[Path] = []
    for p in root_dir.rglob(image_subdir):
        if p.is_dir():
            has_img = any(f.suffix.lower() in {".png", ".jpg", ".jpeg"} for f in p.iterdir() if f.is_file())
            if has_img:
                results.append(p.resolve())
    results.sort()
    return results


def iter_folder_images(folder: Path, patterns=(".jpg", ".jpeg", ".png")) -> Iterable[tuple[str, np.ndarray]]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in patterns:
            try:
                rgb = np.array(Image.open(p).convert("RGB"))
                yield p.stem, rgb
            except Exception:
                continue


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_")


# ------------------ per-frame (prior) using LUT internals ------------------

def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def process_frame_prior(model: LookupRiskModel, dino, rgb_full, cfg: dict,
                        reason_cache=None, backend: str = "faiss") -> dict:
    params = cfg.get("params", {})
    infer = cfg.get("inference", {})
    vis_cfg = cfg.get("visualization", {})

    # DINO feats
    x = dino.rgb_image_preprocessing(rgb_full, params.get("target_h_query", 560))
    feats = dino.get_dino_features_rgb(x)  # [C,H,W]

    # Flatten & normalize
    C, Hf, Wf = feats.shape
    Xf = feats.permute(1, 2, 0).reshape(-1, C)
    Xf = l2_normalize_rows(Xf)

    # Top-k search
    Kq = int(infer.get("topk", 5))
    if backend == "torch":
        gemm_chunk = int(params.get("gemm_chunk", 20000))
        scores, obj_ids_k = _torch_topk_chunked(model, Xf, k=Kq, chunk_rows=gemm_chunk, dtype=torch.float16)
    else:
        chunk = int(params.get("faiss_chunk", 100_000))
        scores, obj_ids_k = _faiss_topk_chunked(model.obj_lut, Xf, k=Kq, chunk=chunk)

    max_sim = scores.max(dim=1).values
    top1_ids = obj_ids_k[:, 0]

    # Per-B cache
    if reason_cache is None:
        Bname = infer["object_B"]
        if Bname not in model.obj_lut.name2id:
            raise KeyError(f"Unknown object_B '{Bname}'")
        B_id = model.obj_lut.name2id[Bname]
        default_unknown = float(infer.get("default_unknown", 0.5))
        reason_cache = model._pair_vectors_for_B(B_id, default_unknown)
    risk_vec, reason_idx_vec, reason_palette, reason_labels = reason_cache

    # Risk aggregation
    if infer.get("soft", True):
        logits = scores * float(infer.get("temperature", 20.0))
        weights = torch.softmax(logits, dim=-1)
        risk_vals = risk_vec[obj_ids_k]
        risk_flat = (weights * risk_vals).sum(dim=-1)
    else:
        risk_flat = risk_vec[top1_ids]

    # Unknown gating
    tau = float(model.tau)
    default_unknown = float(infer.get("default_unknown", 0.5))
    risk_flat = torch.where(max_sim >= tau, risk_flat, torch.tensor(default_unknown, dtype=risk_flat.dtype))

    # Reshape & (optionally) upsample to original resolution
    H0, W0 = rgb_full.shape[:2]
    risk_map = risk_flat.view(Hf, Wf)
    if bool(vis_cfg.get("resize_to_original", True)):
        t = risk_map[None, None, ...]
        risk_up = torch.nn.functional.interpolate(t, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
        risk_map = risk_up

    # ID & Reason maps
    id_map_feat = top1_ids.view(Hf, Wf).cpu().numpy()
    unknown_mask_feat = (max_sim.view(Hf, Wf).cpu().numpy() < tau)
    id_map_feat[unknown_mask_feat] = -1

    reason_idx_per_obj = reason_idx_vec.numpy()
    reason_idx_img = reason_idx_per_obj[top1_ids.cpu().numpy()].reshape(Hf, Wf)
    RIDX_UNKNOWN, RIDX_MISSING = 0, 1
    has_pair = (risk_vec[top1_ids] != default_unknown).cpu().numpy().reshape(Hf, Wf)
    reason_idx_img[unknown_mask_feat] = RIDX_UNKNOWN
    reason_idx_img[~unknown_mask_feat & (~has_pair)] = RIDX_MISSING

    if bool(vis_cfg.get("resize_to_original", True)):
        id_t = torch.from_numpy(id_map_feat)[None, None, ...].float()
        id_up = torch.nn.functional.interpolate(id_t, size=(H0, W0), mode="nearest")[0, 0].long().numpy()
        reason_t = torch.from_numpy(reason_idx_img)[None, None, ...].float()
        reason_up = torch.nn.functional.interpolate(reason_t, size=(H0, W0), mode="nearest")[0, 0].long().numpy()
    else:
        id_up = id_map_feat
        reason_up = reason_idx_img

    # Colorize reason + inline legend (simple overlay here; we’ll reuse palette)
    reason_palette = reason_cache[2]
    reason_labels = reason_cache[3]
    reason_rgb = reason_palette[reason_up]

    risk_np = np.clip(risk_map.detach().cpu().numpy(), 0.0, 1.0)
    risk_gray_u8 = (risk_np * 255.0).round().astype(np.uint8)

    return {
        "risk_gray": risk_gray_u8,
        "risk_float": risk_np,
        "id_map": id_up,
        "reason_rgb": reason_rgb,
        "reason_palette": reason_palette,
        "reason_labels": reason_labels,
    }


# ------------------ Main ------------------

def main():
    cfg, cfg_dir = load_config()

    # read sections
    paths = cfg["paths"]
    params = cfg.get("params", {})
    infer_base = cfg.get("inference", {})
    vis_cfg = cfg.get("visualization", {})

    # --- add near other pipeline knobs in main() ---
    pipeline = cfg.get("pipeline", {})
    frame_stride = int(pipeline.get("frame_stride", 1))
    if frame_stride < 1:
        frame_stride = 1

    # 1) Build/load model once
    if "model_dir" in paths:
        model = LookupRiskModel.load(paths["model_dir"])  # type: ignore
    else:
        print("[INFO] No model_dir; building LUT from raw…")
        dino_tmp = _ensure_dino(params.get("device"))
        model = LookupRiskModel.build_from_raw(
            objects_root=paths["objects_root"],
            objects_csv=paths["objects_csv"],
            pairs_csv=paths["pairs_csv"],
            dino=dino_tmp,
            K=params.get("K", 32),
            target_h=params.get("target_h_build", 224),
            per_image_samples=params.get("per_image_samples", 512),
            exclude_patterns=params.get("exclude_patterns", ["_pca"]),
            open_set_threshold=infer_base.get("open_set_threshold", 0.65),
            temperature=infer_base.get("temperature", 20.0),
            symmetric=bool(params.get("pairs_symmetric", True)),
            conflict_policy=params.get("pairs_conflict_policy", "mean"),
        )

    backend = _prepare_backend(model, params)
    dino = _ensure_dino(params.get("device"))

    # 2) Find all rgb_png folders under root_dir
    root_dir = Path(paths["root_dir"] if Path(paths["root_dir"]).is_absolute() else (cfg_dir / paths["root_dir"])).resolve()
    rgb_subdir = paths.get("image_subdir", "rgb_png")
    rgb_folders = iter_rgb_png_roots(root_dir, rgb_subdir)
    if not rgb_folders:
        raise RuntimeError(f"No '{rgb_subdir}' folders with images found under: {root_dir}")

    # object_B list
    object_list: List[str] = cfg.get("object_B_list", [])
    if not object_list:
        raise ValueError("Config must include a non-empty 'object_B_list' (list of object names).")

    # ID palette for id visualization
    max_id = max(model.obj_lut.id2name.keys()) if model.obj_lut.id2name else 0
    id_palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    import colorsys
    for oid in range(max_id + 1):
        h = ((oid * 0.61803398875) % 1.0); s, v = 0.65, 1.0
        r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
        id_palette[oid] = (int(r_ * 255), int(g_ * 255), int(b_ * 255))
    id_unknown_color = tuple(vis_cfg.get("id_unknown_color", [0, 0, 0]))

    # colormap
    use_colored = bool(vis_cfg.get("colored", True)) and _HAVE_CM
    cmap = cm.get_cmap(vis_cfg.get("cmap", "viridis")) if use_colored else None
    overlay_on = bool(vis_cfg.get("overlay", True))
    alpha = float(vis_cfg.get("alpha", 0.5))

    # 3) Batch over (folder, object_B)
    for rgb_dir in rgb_folders:
        frame_list = list(iter_folder_images(rgb_dir))
        if not frame_list:
            continue

        for objB in object_list:
            if objB not in model.obj_lut.name2id:
                print(f"[WARN] object_B '{objB}' not in LUT; skipping for {rgb_dir}")
                continue

            # Set per-run inference options
            infer = dict(infer_base)
            infer["object_B"] = objB

            # Cache per-B vectors once per (folder, B)
            B_id = model.obj_lut.name2id[objB]
            default_unknown = float(infer.get("default_unknown", 0.5))
            reason_cache = model._pair_vectors_for_B(B_id, default_unknown)

            # Output root next to this rgb_png folder, named prior_{objectB}
            out_root = rgb_dir.parent / f"prior_{sanitize_name(objB)}"
            out_dirs = lut_ensure_out_dirs(out_root)

            print(f"[RUN] {rgb_dir}  ×  B='{objB}'  →  {out_root}")
            t_folder = time.time()

            # Per-frame
            # for stem, rgb in frame_list:
            #     t0 = time.time()
            #     out = process_frame_prior(model, dino, rgb,
            #                               cfg={"params": params, "inference": infer, "visualization": vis_cfg},
            #                               reason_cache=reason_cache,
            #                               backend=backend)
            #     risk_gray = out["risk_gray"]
            #     risk_np = out["risk_float"]
            #     id_map = out["id_map"]
            #     reason_rgb = out["reason_rgb"]

            #     # Save risk gray
            #     Image.fromarray(risk_gray).save(out_dirs["risk_gray"] / f"{stem}.png")

            #     # Save colored risk
            #     if use_colored and cmap is not None:
            #         colored = (cmap(risk_np)[..., :3] * 255).astype(np.uint8)
            #         Image.fromarray(colored).save(out_dirs["risk_colored"] / f"{stem}.png")

            #     # Save overlay (if colored)
            #     if overlay_on and use_colored and cmap is not None:
            #         overlay = (rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
            #         overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            #         Image.fromarray(overlay).save(out_dirs["overlays"] / f"{stem}.png")

            #     # Save ID image
            #     id_rgb = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
            #     known_mask = id_map >= 0
            #     if known_mask.any():
            #         id_rgb[known_mask] = id_palette[id_map[known_mask]]
            #     if (~known_mask).any():
            #         id_rgb[~known_mask] = np.array(id_unknown_color, dtype=np.uint8)
            #     Image.fromarray(id_rgb).save(out_dirs["ids"] / f"{stem}.png")

            #     # Save reasoning (already with legend-style colors if provided)
            #     Image.fromarray(reason_rgb).save(out_dirs["reasons"] / f"{stem}.png")

            #     if bool(cfg.get("debug", {}).get("save_numpy", False)):
            #         np.save(out_dirs["npy"] / f"{stem}.risk.npy", risk_np)

            #     print(f"  [OK] {stem} in {time.time()-t0:.2f}s")

            for idx, (stem, rgb) in enumerate(frame_list):
                if (idx % frame_stride) != 0:
                    continue  # skip this frame, no outputs written, filenames remain 1:1 for processed frames

                t0 = time.time()
                out = process_frame_prior(
                    model, dino, rgb,
                    cfg={"params": params, "inference": infer, "visualization": vis_cfg},
                    reason_cache=reason_cache,
                    backend=backend
                )
                risk_gray = out["risk_gray"]
                risk_np   = out["risk_float"]
                id_map    = out["id_map"]
                reason_rgb= out["reason_rgb"]

                # Save risk gray
                Image.fromarray(risk_gray).save(out_dirs["risk_gray"] / f"{stem}.png")

                # Save colored risk
                if use_colored and cmap is not None:
                    colored = (cmap(risk_np)[..., :3] * 255).astype(np.uint8)
                    Image.fromarray(colored).save(out_dirs["risk_colored"] / f"{stem}.png")

                # Save overlay (if colored)
                if overlay_on and use_colored and cmap is not None:
                    overlay = (rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
                    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                    Image.fromarray(overlay).save(out_dirs["overlays"] / f"{stem}.png")

                # Save ID image
                id_rgb = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
                known_mask = id_map >= 0
                if known_mask.any():
                    id_rgb[known_mask] = id_palette[id_map[known_mask]]
                if (~known_mask).any():
                    id_rgb[~known_mask] = np.array(id_unknown_color, dtype=np.uint8)
                Image.fromarray(id_rgb).save(out_dirs["ids"] / f"{stem}.png")

                # Save reasoning
                Image.fromarray(reason_rgb).save(out_dirs["reasons"] / f"{stem}.png")

                if bool(cfg.get("debug", {}).get("save_numpy", False)):
                    np.save(out_dirs["npy"] / f"{stem}.risk.npy", risk_np)

                print(f"  [OK] {stem} in {time.time()-t0:.2f}s")

            print(f"[DONE] {rgb_dir.name} × {objB} in {time.time()-t_folder:.1f}s  →  {out_root}")

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
