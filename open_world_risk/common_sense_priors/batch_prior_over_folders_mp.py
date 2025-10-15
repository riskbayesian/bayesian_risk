#!/usr/bin/env python3
"""
Batch LUT prior over all rgb_png/ folders & multiple object_B values
with a single GPU worker + threaded I/O prefetch to keep the GPU busy.

- Finds <root>/**/rgb_png with images
- For each folder × object_B, writes outputs to sibling: prior_{object_B}
- Reuses lookup_risk_lut.py logic for feature extraction, NN search, and saving

Config:
  Use CONFIG env or --config path (see YAML example at bottom)

Author: you + ChatGPT
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import queue
import threading
import numpy as np
from PIL import Image

import yaml
import torch

# ---- Import your LUT pipeline helpers ----
from open_world_risk.common_sense_priors.lookup_risk_lut import (
    LookupRiskModel,
    ensure_out_dirs as lut_ensure_out_dirs,
    _ensure_dino,
    _faiss_topk_chunked,
    _torch_topk_chunked,
)

# Optional colormap for colored/overlay outputs
try:
    import matplotlib.cm as cm
    HAVE_CM = True
except Exception:
    cm = None
    HAVE_CM = False


# ---------------- Config ----------------
def load_config(explicit: Optional[str] = None, default="batch_prior.yaml") -> tuple[dict, Path]:
    """
    Load YAML config. If explicit is provided, use it; else $CONFIG or ./batch_prior.yaml
    Returns (cfg, cfg_dir)
    """
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
    else:
        envp = os.environ.get("CONFIG")
        cfg_path = Path(envp).expanduser().resolve() if envp else (Path(__file__).parent / default).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f), cfg_path.parent


# ---------------- Discovery ----------------
def find_rgb_png_folders(root_dir: Path, subdir_name: str = "rgb_png") -> List[Path]:
    """recursively find all subdirs named subdir_name that contain images"""
    found: List[Path] = []
    for p in root_dir.rglob(subdir_name):
        if p.is_dir():
            if any(fp.suffix.lower() in {".png", ".jpg", ".jpeg"} for fp in p.iterdir() if fp.is_file()):
                found.append(p.resolve())
    found.sort()
    return found


def iter_folder_images(folder: Path, patterns=(".jpg", ".jpeg", ".png")) -> Iterable[tuple[str, Path]]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in patterns:
            yield p.stem, p


def sanitize(name: str) -> str:
    return name.lower().replace(" ", "_")


# ---------------- GPU worker task ----------------
def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def process_frame_prior(model: LookupRiskModel,
                        dino,
                        rgb_np: np.ndarray,
                        params: dict,
                        infer: dict,
                        vis_cfg: dict,
                        backend: str,
                        reason_cache):
    # preprocess & features
    x = dino.rgb_image_preprocessing(rgb_np, params.get("target_h_query", 560))
    feats = dino.get_dino_features_rgb(x)  # [C,H,W]
    C, Hf, Wf = feats.shape
    Xf = feats.permute(1, 2, 0).reshape(-1, C)
    Xf = l2_normalize_rows(Xf)

    # top-k nearest prototypes
    Kq = int(infer.get("topk", 5))
    if backend == "torch":
        scores, obj_ids_k = _torch_topk_chunked(model, Xf, k=Kq,
                                                chunk_rows=int(params.get("gemm_chunk", 20000)),
                                                dtype=torch.float16)
    else:
        scores, obj_ids_k = _faiss_topk_chunked(model.obj_lut, Xf, k=Kq,
                                                chunk=int(params.get("faiss_chunk", 100_000)))

    max_sim = scores.max(dim=1).values
    top1_ids = obj_ids_k[:, 0]

    # reason/risk vectors for object_B
    risk_vec, reason_idx_vec, reason_palette, reason_labels = reason_cache

    # aggregate risk per pixel
    if infer.get("soft", True):
        logits = scores * float(infer.get("temperature", 20.0))
        weights = torch.softmax(logits, dim=-1)
        risk_vals = risk_vec[obj_ids_k]
        risk_flat = (weights * risk_vals).sum(dim=-1)
    else:
        risk_flat = risk_vec[top1_ids]

    # unknown gating
    tau = float(model.tau)
    default_unknown = float(infer.get("default_unknown", 0.5))
    risk_flat = torch.where(max_sim >= tau, risk_flat, torch.tensor(default_unknown, dtype=risk_flat.dtype))

    # reshape / upsample to original size
    H0, W0 = rgb_np.shape[:2]
    risk_map = risk_flat.view(Hf, Wf)
    if bool(vis_cfg.get("resize_to_original", True)):
        risk_map = torch.nn.functional.interpolate(risk_map[None, None, ...],
                                                   size=(H0, W0),
                                                   mode="bilinear",
                                                   align_corners=False)[0, 0]

    # id & reason maps
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
        id_map = torch.nn.functional.interpolate(torch.from_numpy(id_map_feat)[None, None, ...].float(),
                                                 size=(H0, W0), mode="nearest")[0, 0].long().numpy()
        reason_up = torch.nn.functional.interpolate(torch.from_numpy(reason_idx_img)[None, None, ...].float(),
                                                    size=(H0, W0), mode="nearest")[0, 0].long().numpy()
    else:
        id_map = id_map_feat
        reason_up = reason_idx_img

    reason_rgb = reason_palette[reason_up]
    risk_np = np.clip(risk_map.detach().cpu().numpy(), 0.0, 1.0)
    risk_gray = (risk_np * 255.0).round().astype(np.uint8)

    return risk_gray, risk_np, id_map, reason_rgb


class GPUWorker:
    """
    Runs on the main process; processes tasks but overlaps I/O using thread prefetchers.
    """
    def __init__(self, model: LookupRiskModel, params: dict, infer_base: dict, vis_cfg: dict, backend: str):
        self.model = model
        self.params = params
        self.infer_base = infer_base
        self.vis_cfg = vis_cfg
        self.backend = backend
        self.dino = _ensure_dino(params.get("device"))
        # id palette for visualization
        max_id = max(self.model.obj_lut.id2name.keys()) if self.model.obj_lut.id2name else 0
        self.id_palette = self._make_id_palette(max_id, vis_cfg.get("id_unknown_color", [0, 0, 0]))
        # colormap / overlay
        self.use_colored = bool(vis_cfg.get("colored", True)) and HAVE_CM
        self.cmap = cm.get_cmap(vis_cfg.get("cmap", "viridis")) if self.use_colored else None
        self.overlay_on = bool(vis_cfg.get("overlay", True))
        self.alpha = float(vis_cfg.get("alpha", 0.5))

    @staticmethod
    def _make_id_palette(max_id: int, unk_color: List[int]) -> dict:
        import colorsys
        pal = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for oid in range(max_id + 1):
            h = ((oid * 0.61803398875) % 1.0); s, v = 0.65, 1.0
            r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
            pal[oid] = (int(r_ * 255), int(g_ * 255), int(b_ * 255))
        return {"pal": pal, "unk": np.array(unk_color, dtype=np.uint8)}

    def _save_outputs(self, out_dirs: Dict[str, Path], stem: str,
                      risk_gray: np.ndarray, risk_np: np.ndarray,
                      id_map: np.ndarray, reason_rgb: np.ndarray, rgb_np: np.ndarray,
                      save_numpy: bool):
        Image.fromarray(risk_gray).save(out_dirs["risk_gray"] / f"{stem}.png")

        if self.use_colored and self.cmap is not None:
            colored = (self.cmap(risk_np)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(colored).save(out_dirs["risk_colored"] / f"{stem}.png")
            if self.overlay_on:
                overlay = (rgb_np.astype(np.float32) * (1 - self.alpha) + colored.astype(np.float32) * self.alpha)
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                Image.fromarray(overlay).save(out_dirs["overlays"] / f"{stem}.png")

        # ID image
        pal = self.id_palette["pal"]; unk = self.id_palette["unk"]
        id_rgb = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
        known = id_map >= 0
        if known.any():
            id_rgb[known] = pal[id_map[known]]
        if (~known).any():
            id_rgb[~known] = unk
        Image.fromarray(id_rgb).save(out_dirs["ids"] / f"{stem}.png")

        # reasoning
        Image.fromarray(reason_rgb).save(out_dirs["reasons"] / f"{stem}.png")

        if save_numpy:
            np.save(out_dirs["npy"] / f"{stem}.risk.npy", risk_np)

    def run_task(self, rgb_dir: Path, object_B: str, out_root: Path, save_numpy: bool,
                 io_threads: int, io_queue_size: int):
        infer = dict(self.infer_base)
        infer["object_B"] = object_B

        # prepare per-B reason cache
        if object_B not in self.model.obj_lut.name2id:
            print(f"[WARN] object_B '{object_B}' not in LUT; skipping: {rgb_dir}")
            return
        B_id = self.model.obj_lut.name2id[object_B]
        default_unknown = float(infer.get("default_unknown", 0.5))
        reason_cache = self.model._pair_vectors_for_B(B_id, default_unknown)

        out_dirs = lut_ensure_out_dirs(out_root)

        # --- threaded prefetch of images into a queue ---
        q: queue.Queue[tuple[str, Optional[np.ndarray]]] = queue.Queue(maxsize=io_queue_size)
        stop_token = object()

        def loader(paths: List[tuple[str, Path]]):
            for stem, pth in paths:
                try:
                    rgb = np.array(Image.open(pth).convert("RGB"))
                except Exception:
                    rgb = None
                q.put((stem, rgb))
            q.put(("__STOP__", None))

        frames = list(iter_folder_images(rgb_dir))
        t0 = time.time()

        # start IO threads
        per_thread = (len(frames) + io_threads - 1) // io_threads
        threads = []
        for i in range(io_threads):
            chunk = frames[i * per_thread:(i + 1) * per_thread]
            if not chunk:
                continue
            th = threading.Thread(target=loader, args=(chunk,), daemon=True)
            th.start()
            threads.append(th)

        done_threads = 0
        processed = 0
        while True:
            stem, rgb = q.get()
            if stem == "__STOP__":
                done_threads += 1
                if done_threads >= len(threads):
                    break
                continue
            if rgb is None:
                continue
            # GPU work
            risk_gray, risk_np, id_map, reason_rgb = process_frame_prior(
                model=self.model,
                dino=self.dino,
                rgb_np=rgb,
                params=self.params,
                infer=infer,
                vis_cfg=self.vis_cfg,
                backend=("torch" if hasattr(self.model, "_protos_T") else "faiss"),
                reason_cache=reason_cache,
            )
            self._save_outputs(out_dirs, stem, risk_gray, risk_np, id_map, reason_rgb, rgb, save_numpy)
            processed += 1

        for th in threads:
            th.join()

        print(f"  [OK] {rgb_dir} × {object_B} : {processed} frames in {time.time() - t0:.1f}s → {out_root}")


# ---------------- Main ----------------
def main():
    cfg, cfg_dir = load_config()

    paths = cfg["paths"]
    params = cfg.get("params", {})
    infer_base = cfg.get("inference", {})
    vis_cfg = cfg.get("visualization", {})
    debug_cfg = cfg.get("debug", {})
    pool_cfg = cfg.get("pipeline", {})

    # resolve root
    root_dir = Path(paths["root_dir"]) if Path(paths["root_dir"]).is_absolute() else (cfg_dir / paths["root_dir"])
    root_dir = root_dir.resolve()
    rgb_subdir = paths.get("image_subdir", "rgb_png")

    # build or load LUT once
    if "model_dir" in paths and paths["model_dir"]:
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

    # prepare GEMM backend cache if selected
    backend = (params.get("nn_backend") or "faiss").lower()
    if params.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    if backend == "torch" and not hasattr(model, "_protos_T"):
        device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        P = torch.from_numpy(model.obj_lut.prototypes).to(device=device, dtype=torch.float16)
        P = torch.nn.functional.normalize(P, dim=1)
        model._protos_T = P.t().contiguous()
        model._proto_obj_ids_t = torch.from_numpy(model.obj_lut.proto_obj_ids).to(device)

    # find all rgb folders
    rgb_folders = find_rgb_png_folders(root_dir, rgb_subdir)
    if not rgb_folders:
        raise RuntimeError(f"No '{rgb_subdir}' folders with images found under: {root_dir}")

    object_B_list: List[str] = cfg.get("object_B_list", [])
    if not object_B_list:
        raise ValueError("Config needs non-empty 'object_B_list'")

    # pipeline knobs
    io_threads = int(pool_cfg.get("io_threads", 4))
    io_queue_size = int(pool_cfg.get("io_queue_size", 32))
    save_numpy = bool(debug_cfg.get("save_numpy", False))

    # GPU worker (single process) with threaded I/O per task
    worker = GPUWorker(model, params, infer_base, vis_cfg, backend)

    # Schedule tasks (sequential on GPU, but each task overlaps I/O via threads)
    t_all = time.time()
    for rgb_dir in rgb_folders:
        for objB in object_B_list:
            out_root = rgb_dir.parent / f"prior_{sanitize(objB)}"
            worker.run_task(rgb_dir=rgb_dir,
                            object_B=objB,
                            out_root=out_root,
                            save_numpy=save_numpy,
                            io_threads=io_threads,
                            io_queue_size=io_queue_size)

    print(f"[ALL DONE] {len(rgb_folders)} folders × {len(object_B_list)} objects in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    # Local import (used by GPUWorker._save_outputs)
    from PIL import Image  # noqa: E402
    main()
