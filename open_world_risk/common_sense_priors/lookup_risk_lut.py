#!/usr/bin/env python3
"""
YAML-driven lookup-table (LUT) risk model — vectorized & batched — now for
• single image, • folders of images, or • videos (constant object_B across frames).

What’s new:
- Batch processor for folders/videos with per-type output subfolders (risk_gray/, risk_colored/, overlays/, ids/, reasons/, debug/)
- Reason **legend is embedded** directly into the reasoning image (only labels that appear)
- Torch GEMM Top‑K backend behind YAML flag (faster than FAISS FlatIP for many cases)
- Reuses DINO, per‑B vectors, and prototype cache across frames

Config: `lookup_lut_config.yaml` or env `LOOKUP_LUT_CONFIG`.
"""
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml
import time

import faiss
import faiss.contrib.torch_utils

# Optional: OpenCV for video IO
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# Try to import your DINO util
try:
    from open_world_risk.src.dino_features import DinoFeatures
except Exception:  # pragma: no cover
    DinoFeatures = None

# --------------------------- Utilities ---------------------------

def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

def to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    if x.is_cuda:
        x = x.detach().cpu()
    return x.contiguous().to(torch.float32).numpy()


def load_yaml_config(default_name: str = "lookup_lut_config.yaml") -> dict:
    cfg_path = os.environ.get("LOOKUP_LUT_CONFIG")
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), default_name)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def ensure_out_dirs(root: Path) -> Dict[str, Path]:
    sub = {
        "risk_gray": root / "risk_gray",
        "risk_colored": root / "risk_colored",
        "overlays": root / "overlays",
        "ids": root / "ids",
        "reasons": root / "reasons",
        "debug": root / "debug",
        "npy": root / "npy",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub

# --------------------------- Pair LUT ---------------------------

@dataclass
class PairInfo:
    risk: float  # 0..1 (1=safe, 0=unsafe)
    reason: str


class PairLUT:
    """(A_id, B_id) → (risk, reason). Order-invariant support and a reason palette."""
    def __init__(self, name2id: Dict[str, int], symmetric: bool = True, conflict_policy: str = "mean"):
        self.name2id = dict(name2id)
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.table: Dict[Tuple[int, int], PairInfo] = {}
        self.symmetric = bool(symmetric)
        self.conflict_policy = conflict_policy  # 'mean' | 'first' | 'max'
        # Reason handling
        self.reasons: List[str] = []
        self.reason_palette: Dict[str, List[int]] = {}

    @staticmethod
    def _palette_from_reasons(reasons: List[str]) -> Dict[str, List[int]]:
        import colorsys
        uniq = [r for r in sorted(set([r.strip() for r in reasons if r and r.strip()]))]
        n = max(1, len(uniq))
        palette: Dict[str, List[int]] = {}
        for i, r in enumerate(uniq):
            h = (i / n) % 1.0
            s, v = 0.65, 1.0
            r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
            palette[r] = [int(r_ * 255), int(g_ * 255), int(b_ * 255)]
        return palette

    @classmethod
    def from_csv(cls, csv_path: str, object_list_csv: Optional[str] = None,
                 name2id: Optional[Dict[str, int]] = None,
                 symmetric: bool = True,
                 conflict_policy: str = "mean") -> "PairLUT":
        import csv
        if name2id is None:
            if object_list_csv is None:
                raise ValueError("Need object_list_csv or name2id to build PairLUT")
            names = []
            with open(object_list_csv, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        names.append(row[0].strip())
            name2id = {name: i for i, name in enumerate(sorted(set(names)))}
        lut = cls(name2id, symmetric=symmetric, conflict_policy=conflict_policy)
        seen_reasons: List[str] = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = row["object_a"].strip()
                b = row["object_b"].strip()
                rating15 = float(row["rating_1_unsafe_5_safe"])  # 1..5
                reason = (row.get("reason", "") or "").strip()
                risk01 = (rating15 - 1.0) / 4.0  # map to [0,1]
                if a in lut.name2id and b in lut.name2id:
                    aid, bid = lut.name2id[a], lut.name2id[b]
                    def _insert(xid, yid):
                        key = (xid, yid)
                        if key in lut.table:
                            old = lut.table[key]
                            if conflict_policy == "mean":
                                lut.table[key] = PairInfo(risk=(old.risk + risk01) / 2.0, reason=old.reason or reason)
                            elif conflict_policy == "max":
                                lut.table[key] = PairInfo(risk=max(old.risk, risk01), reason=old.reason or reason)
                        else:
                            lut.table[key] = PairInfo(risk=risk01, reason=reason)
                    _insert(aid, bid)
                    if symmetric:
                        _insert(bid, aid)
                    if reason:
                        seen_reasons.append(reason)
        lut.reasons = sorted(set(seen_reasons))
        lut.reason_palette = lut._palette_from_reasons(lut.reasons)
        return lut

    def lookup(self, a_id: int, b_id: int, default: float = 0.5) -> PairInfo:
        info = self.table.get((a_id, b_id))
        if info is None and self.symmetric:
            info = self.table.get((b_id, a_id))
        if info is None:
            return PairInfo(risk=default, reason="")
        return info

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "name2id.json", "w") as f:
            json.dump(self.name2id, f, indent=2)
        with open(out / "meta.json", "w") as f:
            json.dump({
                "symmetric": self.symmetric,
                "conflict_policy": self.conflict_policy,
                "reasons": self.reasons,
                "reason_palette": self.reason_palette,
            }, f, indent=2)
        serial = {f"{a},{b}": {"risk": v.risk, "reason": v.reason}
                  for (a, b), v in self.table.items()}
        with open(out / "pair_lut.json", "w") as f:
            json.dump(serial, f)

    @classmethod
    def load(cls, in_dir: str) -> "PairLUT":
        p = Path(in_dir)
        with open(p / "name2id.json", "r") as f:
            name2id = json.load(f)
        symmetric = True
        conflict_policy = "mean"
        reasons = []
        reason_palette = {}
        meta_path = p / "meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path, "r"))
            symmetric = bool(meta.get("symmetric", True))
            conflict_policy = meta.get("conflict_policy", "mean")
            reasons = meta.get("reasons", [])
            reason_palette = meta.get("reason_palette", {})
        with open(p / "pair_lut.json", "r") as f:
            serial = json.load(f)
        lut = cls(name2id, symmetric=symmetric, conflict_policy=conflict_policy)
        lut.reasons = reasons
        lut.reason_palette = reason_palette
        for k, v in serial.items():
            a, b = map(int, k.split(","))
            lut.table[(a, b)] = PairInfo(risk=float(v["risk"]), reason=v.get("reason", ""))
        return lut

# --------------------------- Object LUT ---------------------------

class ObjectLUT:
    """Prototype bank + FAISS index for pixel→object classification with open-set handling."""
    def __init__(self, name2id: Dict[str, int], metric: str = "cosine"):
        assert metric in {"cosine", "ip"}
        self.metric = metric
        self.name2id = dict(name2id)
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.index: Optional[faiss.Index] = None
        self.prototypes: Optional[np.ndarray] = None
        self.proto_obj_ids: Optional[np.ndarray] = None

    @staticmethod
    def _features_from_images(dino: DinoFeatures, image_paths: List[str], target_h: int = 224,
                              per_image_samples: int = 512) -> torch.Tensor:
        feats = []
        for p in image_paths:
            rgb = np.array(Image.open(p).convert("RGB"))
            x = dino.rgb_image_preprocessing(rgb, target_h)
            f = dino.get_dino_features_rgb(x)  # [C,H,W]
            C, H, W = f.shape
            flat = f.permute(1, 2, 0).reshape(-1, C)
            if per_image_samples > 0 and flat.shape[0] > per_image_samples:
                idx = torch.randperm(flat.shape[0])[:per_image_samples]
                flat = flat[idx]
            feats.append(flat)
        X = torch.cat(feats, dim=0)
        X = l2_normalize_rows(X)
        return X

    def build_from_folders(self, objects_root: str, name_list: List[str], dino: DinoFeatures,
                            K: int = 32, target_h: int = 224, per_image_samples: int = 512,
                            exclude_patterns: Optional[List[str]] = None):
        exclude_patterns = exclude_patterns or ["_pca"]
        all_centroids = []
        all_ids = []
        for name in tqdm(name_list, desc="Build ObjectLUT"):
            obj_dir = Path(objects_root) / name.replace(" ", "_")
            if not obj_dir.exists():
                continue
            ims = [str(p) for p in obj_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and p.parent.name.lower() != "pca"
                   and not any(pat.lower() in p.name.lower() for pat in exclude_patterns)]
            if not ims:
                continue
            X = self._features_from_images(dino, ims, target_h, per_image_samples)
            N, C = X.shape
            k_use = min(K, max(2, min(N, K)))
            km = faiss.Kmeans(d=C, k=k_use, niter=20, verbose=False, gpu=True)
            km.train(to_numpy_f32(X))
            centroids = km.centroids
            centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-6)
            all_centroids.append(centroids)
            all_ids.append(np.full((centroids.shape[0],), self.name2id[name], dtype=np.int32))
        if not all_centroids:
            raise RuntimeError("No centroids built; check inputs.")
        self.prototypes = np.concatenate(all_centroids, axis=0).astype("float32")
        self.proto_obj_ids = np.concatenate(all_ids, axis=0).astype("int32")
        d = self.prototypes.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.prototypes)

    def build_from_feature_dict(self, feature_dict: Dict[str, List[torch.Tensor]], K: int = 32):
        all_centroids = []
        all_ids = []
        C = None
        for name, vecs in feature_dict.items():
            if not vecs:
                continue
            X = torch.stack(vecs, dim=0)
            X = l2_normalize_rows(X)
            N, C_ = X.shape
            C = C_ if C is None else C
            arr = to_numpy_f32(X)
            if N == 1:
                centroids = arr
            else:
                k_use = min(K, max(1, min(N, K)))
                km = faiss.Kmeans(d=C_, k=k_use, niter=20, verbose=False, gpu=True)
                km.train(arr)
                centroids = km.centroids
            centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-6)
            all_centroids.append(centroids)
            all_ids.append(np.full((centroids.shape[0],), self.name2id[name], dtype=np.int32))
        if not all_centroids:
            raise RuntimeError("No centroids built; check feature_dict.")
        self.prototypes = np.concatenate(all_centroids, axis=0).astype("float32")
        self.proto_obj_ids = np.concatenate(all_ids, axis=0).astype("int32")
        d = self.prototypes.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.prototypes)

    def topk(self, feats: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.index is not None
        X = to_numpy_f32(l2_normalize_rows(feats))
        D, I = self.index.search(X, k)
        obj_ids = self.proto_obj_ids[I]
        return torch.from_numpy(D), torch.from_numpy(I), torch.from_numpy(obj_ids)

    # --- persistence helpers ---
    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.prototypes is None or self.proto_obj_ids is None or self.index is None:
            raise RuntimeError("ObjectLUT.save(): index/prototypes not built")
        np.save(out / "prototypes.npy", self.prototypes)
        np.save(out / "proto_obj_ids.npy", self.proto_obj_ids)
        with open(out / "name2id.json", "w") as f:
            json.dump(self.name2id, f, indent=2)
        faiss.write_index(self.index, str(out / "index.faiss"))

    @classmethod
    def load(cls, in_dir: str) -> "ObjectLUT":
        p = Path(in_dir)
        with open(p / "name2id.json", "r") as f:
            name2id = json.load(f)
        lut = cls(name2id)
        lut.prototypes = np.load(p / "prototypes.npy")
        lut.proto_obj_ids = np.load(p / "proto_obj_ids.npy")
        lut.index = faiss.read_index(str(p / "index.faiss"))
        return lut

# --------------------------- Combined model ---------------------------

class LookupRiskModel:
    def __init__(self, object_lut: ObjectLUT, pair_lut: PairLUT,
                 open_set_threshold: float = 0.65, temperature: float = 20.0):
        self.obj_lut = object_lut
        self.pair_lut = pair_lut
        self.tau = open_set_threshold
        self.temp = temperature

    @classmethod
    def build_from_raw(
        cls,
        objects_root: str,
        objects_csv: str,
        pairs_csv: str,
        dino: DinoFeatures,
        K: int = 32,
        target_h: int = 224,
        per_image_samples: int = 512,
        exclude_patterns: Optional[List[str]] = None,
        open_set_threshold: float = 0.65,
        temperature: float = 20.0,
        symmetric: bool = True,
        conflict_policy: str = "mean",
    ) -> "LookupRiskModel":
        import csv
        names = []
        with open(objects_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    names.append(row[0].strip())
        name2id = {name: i for i, name in enumerate(sorted(set(names)))}
        pair_lut = PairLUT.from_csv(
            pairs_csv,
            object_list_csv=objects_csv,
            name2id=name2id,
            symmetric=symmetric,
            conflict_policy=conflict_policy,
        )
        obj_lut = ObjectLUT(name2id)
        obj_lut.build_from_folders(objects_root, names, dino,
                                   K=K, target_h=target_h, per_image_samples=per_image_samples,
                                   exclude_patterns=exclude_patterns)
        return cls(obj_lut, pair_lut, open_set_threshold=open_set_threshold, temperature=temperature)

    @classmethod
    def build_from_feature_dict(
        cls,
        feature_dict: Dict[str, List[torch.Tensor]],
        pairs_csv: str,
        objects_csv: Optional[str] = None,
        K: int = 16,
        open_set_threshold: float = 0.65,
        temperature: float = 20.0,
        symmetric: bool = True,
        conflict_policy: str = "mean",
    ) -> "LookupRiskModel":
        names = sorted(feature_dict.keys())
        name2id = {name: i for i, name in enumerate(names)}
        pair_lut = PairLUT.from_csv(pairs_csv, object_list_csv=objects_csv, name2id=name2id,
                                    symmetric=symmetric, conflict_policy=conflict_policy)
        obj_lut = ObjectLUT(name2id)
        obj_lut.build_from_feature_dict(feature_dict, K=K)
        return cls(obj_lut, pair_lut, open_set_threshold=open_set_threshold, temperature=temperature)

    # --- Fast per-B vectors for risk & reasons ---
    def _pair_vectors_for_B(self, B_id: int, default_unknown: float = 0.5):
        """Return:
           risk_vec:  [num_objects] float32 in [0,1] (1=safe)
           reason_idx_vec: [num_objects] int32
           reason_palette_list: [num_reasons+specials, 3] uint8
           reason_labels: list[str] aligned with palette rows
        """
        num_obj = max(self.obj_lut.id2name.keys()) + 1
        risk_vec = torch.full((num_obj,), float(default_unknown), dtype=torch.float32)
        specials = ["__unknown__", "__missing_pair__", "__no_reason__"]
        real = getattr(self.pair_lut, "reasons", [])
        reason_labels = specials + real
        palette = {
            "__unknown__": [0, 0, 0],
            "__missing_pair__": [255, 255, 255],
            "__no_reason__": [128, 128, 128],
        }
        palette.update(getattr(self.pair_lut, "reason_palette", {}))
        reason_palette_list = np.array([palette[name] for name in reason_labels], dtype=np.uint8)
        reason2idx = {name: i for i, name in enumerate(reason_labels)}
        reason_idx_vec = torch.full((num_obj,), reason2idx["__no_reason__"], dtype=torch.int32)
        tbl = self.pair_lut.table
        for (a, b), info in tbl.items():
            if b == B_id:
                risk_vec[a] = float(info.risk)
                ridx = reason2idx.get(info.reason, reason2idx["__no_reason__"]) 
                reason_idx_vec[a] = int(ridx)
        return risk_vec, reason_idx_vec, reason_palette_list, reason_labels

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.obj_lut.save(str(out / "object_lut"))
        self.pair_lut.save(str(out / "pair_lut"))
        with open(out / "config.json", "w") as f:
            json.dump({"tau": self.tau, "temperature": self.temp}, f, indent=2)

    @classmethod
    def load(cls, in_dir: str) -> "LookupRiskModel":
        p = Path(in_dir)
        obj_lut = ObjectLUT.load(str(p / "object_lut"))
        pair_lut = PairLUT.load(str(p / "pair_lut"))
        with open(p / "config.json", "r") as f:
            cfg = json.load(f)
        return cls(obj_lut, pair_lut, open_set_threshold=cfg.get("tau", 0.65), temperature=cfg.get("temperature", 20.0))

# --------------------------- Backends ---------------------------

def _faiss_topk_chunked(obj_lut: ObjectLUT, feats_flat: torch.Tensor, k: int, chunk: int = 100_000):
    N = feats_flat.shape[0]
    Ds, OIDs = [], []
    for i in range(0, N, chunk):
        x = feats_flat[i:i+chunk]
        D, _, O = obj_lut.topk(x, k=k)
        Ds.append(D)
        OIDs.append(O)
    return torch.cat(Ds, 0), torch.cat(OIDs, 0)


def _torch_topk_chunked(model, Xf_cpu: torch.Tensor, k: int, chunk_rows: int = 20000, dtype=torch.float16):
    device = model._protos_T.device
    D_all, O_all = [], []
    for i in range(0, Xf_cpu.shape[0], chunk_rows):
        x = Xf_cpu[i:i+chunk_rows].to(device=device, dtype=dtype)
        x = torch.nn.functional.normalize(x, dim=1)
        S = x @ model._protos_T
        vals, idxs = torch.topk(S, k=k, dim=1)
        D_all.append(vals.to(dtype=torch.float32).cpu())
        O_all.append(model._proto_obj_ids_t[idxs].cpu())
        del x, S, vals, idxs
        torch.cuda.synchronize()
    return torch.cat(D_all, 0), torch.cat(O_all, 0)

# --------------------------- Orchestration (YAML-driven) ---------------------------

def _ensure_dino(device: Optional[str] = None) -> DinoFeatures:
    if DinoFeatures is None:
        raise RuntimeError("DinoFeatures not importable; please ensure your env has it.")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return DinoFeatures(device=dev)


def run_build(cfg: dict):
    paths = cfg["paths"]
    params = cfg.get("params", {})
    infer = cfg.get("inference", {})

    dino = _ensure_dino(params.get("device"))

    model = LookupRiskModel.build_from_raw(
        objects_root=paths["objects_root"],
        objects_csv=paths["objects_csv"],
        pairs_csv=paths["pairs_csv"],
        dino=dino,
        K=params.get("K", 32),
        target_h=params.get("target_h_build", 224),
        per_image_samples=params.get("per_image_samples", 512),
        exclude_patterns=params.get("exclude_patterns", ["_pca"]),
        open_set_threshold=infer.get("open_set_threshold", 0.65),
        temperature=infer.get("temperature", 20.0),
        symmetric=bool(params.get("pairs_symmetric", True)),
        conflict_policy=params.get("pairs_conflict_policy", "mean"),
    )
    out_dir = Path(paths["out_dir"]) if isinstance(paths.get("out_dir"), str) else Path(paths["out_dir"]) if isinstance(paths.get("out_dir"), Path) else Path(paths["out_dir"])  # tolerant
    model.save(str(out_dir))
    print(f"[BUILD] Saved LUT model to {out_dir}")


# ---- Frame iterator helpers ----

def _iter_folder_images(folder: Path, patterns=(".jpg", ".jpeg", ".png")) -> Iterable[Tuple[str, np.ndarray]]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in patterns:
            try:
                rgb = np.array(Image.open(p).convert("RGB"))
                yield p.stem, rgb
            except Exception:
                continue


def _iter_video_frames(video_path: Path, step: int = 1) -> Iterable[Tuple[str, np.ndarray]]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available for video decoding.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield f"f{idx:06d}", rgb
        idx += 1
    cap.release()


def _draw_inline_legend(reason_rgb: np.ndarray, reason_idx_img: np.ndarray, reason_palette: np.ndarray,
                        reason_labels: List[str]) -> np.ndarray:
    """Overlay a compact legend in the bottom-left of the reasoning image, only for classes present."""
    H, W, _ = reason_rgb.shape
    used = np.unique(reason_idx_img)
    entries = []
    for ridx in used:
        label = reason_labels[int(ridx)] if int(ridx) < len(reason_labels) else str(ridx)
        color = tuple(map(int, reason_palette[int(ridx)].tolist()))
        entries.append((color, label))
    # layout
    sw, sh = 24, 24
    pad, gap = 10, 8
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_w = max((len(t[1]) for t in entries), default=10) * 7
    box_w = pad * 2 + sw + 8 + text_w
    box_h = pad * 2 + len(entries) * (sh + gap) - gap
    # draw on PIL canvas over the image
    canvas = Image.fromarray(reason_rgb.copy())
    draw = ImageDraw.Draw(canvas, "RGBA")
    # semi-transparent bg
    draw.rectangle([5, H - box_h - 5, 5 + box_w, H - 5], fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))
    y = H - box_h - 5 + pad
    x = 5 + pad
    for color, label in entries:
        draw.rectangle([x, y, x + sw, y + sh], fill=tuple(color)+(255,), outline=(0, 0, 0, 255))
        draw.text((x + sw + 8, y + 4), label, fill=(0, 0, 0, 255), font=font)
        y += sh + gap
    return np.array(canvas)


def _prepare_backend(model: 'LookupRiskModel', params: dict):
    backend = (params.get("nn_backend") or "faiss").lower()
    if params.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    if backend == "torch":
        device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(model, "_protos_T"):
            P = torch.from_numpy(model.obj_lut.prototypes).to(device=device, dtype=torch.float16)
            P = torch.nn.functional.normalize(P, dim=1)
            model._protos_T = P.t().contiguous()
            model._proto_obj_ids_t = torch.from_numpy(model.obj_lut.proto_obj_ids).to(device)
    return backend

def _process_frame(model: 'LookupRiskModel', dino: DinoFeatures, rgb_full: np.ndarray, cfg: dict,
                   reason_cache: Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str]]] = None,
                   backend: str = "faiss"):
    params = cfg.get("params", {})
    infer = cfg.get("inference", {})
    vis_cfg = cfg.get("visualization", {})

    # DINO
    x = dino.rgb_image_preprocessing(rgb_full, params.get("target_h_query", 560))
    feats = dino.get_dino_features_rgb(x)  # [C,H,W]

    # Flatten + norm
    C, Hf, Wf = feats.shape
    Xf = feats.permute(1, 2, 0).reshape(-1, C)
    Xf = l2_normalize_rows(Xf)

    # Backend search
    Kq = int(infer.get("topk", 5))
    if backend == "torch":
        gemm_chunk = int(params.get("gemm_chunk", 20000))
        scores, obj_ids_k = _torch_topk_chunked(model, Xf, k=Kq, chunk_rows=gemm_chunk, dtype=torch.float16)
    else:
        chunk = int(params.get("faiss_chunk", 100_000))
        scores, obj_ids_k = _faiss_topk_chunked(model.obj_lut, Xf, k=Kq, chunk=chunk)

    max_sim = scores.max(dim=1).values
    top1_ids = obj_ids_k[:, 0]

    # Per-B vectors (cache across frames)
    if reason_cache is None:
        Bname = infer["object_B"]
        if Bname not in model.obj_lut.name2id:
            raise KeyError(f"Unknown object_B '{Bname}'")
        B_id = model.obj_lut.name2id[Bname]
        default_unknown = float(infer.get("default_unknown", 0.5))
        reason_cache = model._pair_vectors_for_B(B_id, default_unknown)
    risk_vec, reason_idx_vec, reason_palette, reason_labels = reason_cache

    # Risk vectorized
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

    # Reshape & upsample to original
    H0, W0 = rgb_full.shape[:2]
    risk_map = risk_flat.view(Hf, Wf)
    if bool(vis_cfg.get("resize_to_original", True)):
        t = risk_map[None, None, ...]
        risk_up = torch.nn.functional.interpolate(t, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
        risk_map = risk_up

    # ID map & Reason map (top-1)
    id_map_feat = top1_ids.view(Hf, Wf).cpu().numpy()
    unknown_mask_feat = (max_sim.view(Hf, Wf).cpu().numpy() < tau)
    id_map_feat[unknown_mask_feat] = -1
    # Reason indices per-pixel
    reason_idx_per_obj = reason_idx_vec.numpy()
    reason_idx_img = reason_idx_per_obj[top1_ids.cpu().numpy()].reshape(Hf, Wf)
    RIDX_UNKNOWN, RIDX_MISSING = 0, 1
    has_pair = (risk_vec[top1_ids] != default_unknown).cpu().numpy().reshape(Hf, Wf)
    reason_idx_img[unknown_mask_feat] = RIDX_UNKNOWN
    reason_idx_img[~unknown_mask_feat & (~has_pair)] = RIDX_MISSING

    # Upsample ID/Reason if needed
    if bool(vis_cfg.get("resize_to_original", True)):
        id_t = torch.from_numpy(id_map_feat)[None, None, ...].float()
        id_up = torch.nn.functional.interpolate(id_t, size=(H0, W0), mode="nearest")[0, 0].long().numpy()
        reason_t = torch.from_numpy(reason_idx_img)[None, None, ...].float()
        reason_up = torch.nn.functional.interpolate(reason_t, size=(H0, W0), mode="nearest")[0, 0].long().numpy()
    else:
        id_up = id_map_feat
        reason_up = reason_idx_img

    # Colorize reason & embed legend
    reason_rgb = reason_palette[reason_up]
    reason_rgb = _draw_inline_legend(reason_rgb, reason_up, reason_palette, reason_labels)

    # Colored risk for readability
    risk_np = np.clip(risk_map.detach().cpu().numpy(), 0.0, 1.0)
    risk_gray_u8 = (risk_np * 255.0).round().astype(np.uint8)

    # Overlay (optional) — use matplotlib-like colormap via Pillow? keep gray overlay simple here
    return {
        "risk_gray": risk_gray_u8,
        "risk_float": risk_np,
        "id_map": id_up,
        "reason_rgb": reason_rgb,
    }


def run_query(cfg: dict):
    paths = cfg["paths"]
    infer = cfg.get("inference", {})
    params = cfg.get("params", {})
    vis_cfg = cfg.get("visualization", {})

    # Load or build model
    if "model_dir" in paths:
        model = LookupRiskModel.load(paths["model_dir"])  # type: ignore
    else:
        print("[QUERY] No model_dir provided; building a temporary model from raw paths...")
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
            open_set_threshold=infer.get("open_set_threshold", 0.65),
            temperature=infer.get("temperature", 20.0),
            symmetric=bool(params.get("pairs_symmetric", True)),
            conflict_policy=params.get("pairs_conflict_policy", "mean"),
        )

    # Prepare backend & DINO once
    backend = _prepare_backend(model, params)
    dino = _ensure_dino(params.get("device"))

    # Input handling
    in_type = (paths.get("input_type") or "image").lower()  # image | folder | video
    out_root = paths.get("output_root")
    if not out_root:
        # default: sibling of input
        if in_type == "image":
            in_path = Path(paths["image"]).resolve()
            out_root = str(in_path.parent / (in_path.stem + "_out"))
        elif in_type == "folder":
            in_path = Path(paths["image_folder"]).resolve()
            out_root = str(in_path.parent / (in_path.name + "_out"))
        else:
            in_path = Path(paths["video"]).resolve()
            out_root = str(in_path.parent / (in_path.stem + "_out"))
    out_dirs = ensure_out_dirs(Path(out_root))

    # Iterate frames
    if in_type == "image":
        frames = [(Path(paths["image"]).resolve().stem, np.array(Image.open(paths["image"]).convert("RGB")))]
    elif in_type == "folder":
        frames = list(_iter_folder_images(Path(paths["image_folder"]).resolve()))
    elif in_type == "video":
        step = int(paths.get("video_frame_step", 1))
        frames = list(_iter_video_frames(Path(paths["video"]).resolve(), step=step))
    else:
        raise ValueError(f"Unknown input_type: {in_type}")

    # Cache per-B vectors across frames
    Bname = infer["object_B"]
    if Bname not in model.obj_lut.name2id:
        raise KeyError(f"Unknown object_B '{Bname}'")
    B_id = model.obj_lut.name2id[Bname]
    default_unknown = float(infer.get("default_unknown", 0.5))
    reason_cache = model._pair_vectors_for_B(B_id, default_unknown)

    # Color palette for ID visualization
    max_id = max(model.obj_lut.id2name.keys()) if model.obj_lut.id2name else 0
    id_palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    import colorsys
    for oid in range(max_id + 1):
        h = ((oid * 0.61803398875) % 1.0)
        s, v = 0.65, 1.0
        r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
        id_palette[oid] = (int(r_ * 255), int(g_ * 255), int(b_ * 255))
    id_unknown_color = tuple(vis_cfg.get("id_unknown_color", [0, 0, 0]))

    # Colormap for colored risk via matplotlib if available
    use_colored = bool(vis_cfg.get("colored", True))
    cmap_name = vis_cfg.get("cmap", "viridis")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
    except Exception:
        use_colored = False
        cmap = None

    # Process
    for stem, rgb in tqdm(frames, desc=f"Processing {in_type}"):
        t0 = time.time()
        out = _process_frame(model, dino, rgb, cfg, reason_cache=reason_cache, backend=backend)
        risk_gray = out["risk_gray"]
        risk_np = out["risk_float"]
        id_map = out["id_map"]
        reason_rgb = out["reason_rgb"]
        # Save risk gray
        Image.fromarray(risk_gray).save(out_dirs["risk_gray"] / f"{stem}.png")
        # Save colored risk
        if use_colored and cmap is not None:
            colored = (cmap(risk_np)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(colored).save(out_dirs["risk_colored"] / f"{stem}.png")
        # Save overlay
        if bool(vis_cfg.get("overlay", True)) and use_colored and cmap is not None:
            alpha = float(vis_cfg.get("alpha", 0.5))
            overlay = (rgb.copy()).astype(np.float32)
            overlay = np.clip(overlay * (1 - alpha) + (colored.astype(np.float32)) * alpha, 0, 255).astype(np.uint8)
            Image.fromarray(overlay).save(out_dirs["overlays"] / f"{stem}.png")
        # Save ID image
        id_rgb = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
        known_mask = id_map >= 0
        if known_mask.any():
            id_rgb[known_mask] = id_palette[id_map[known_mask]]
        if (~known_mask).any():
            id_rgb[~known_mask] = np.array(id_unknown_color, dtype=np.uint8)
        Image.fromarray(id_rgb).save(out_dirs["ids"] / f"{stem}.png")
        # Save reasoning with inline legend
        Image.fromarray(reason_rgb).save(out_dirs["reasons"] / f"{stem}.png")
        # Debug maps
        if bool(cfg.get("debug", {}).get("save_numpy", False)):
            np.save(out_dirs["npy"] / f"{stem}.risk.npy", risk_np)
        t1 = time.time()
        print(f"[QUERY] {stem}: {t1 - t0:.2f}s")

    print(f"[QUERY] Saved outputs to {Path(out_root)}")

# --------------------------- Entry ---------------------------

def main():
    cfg = load_yaml_config()
    mode = cfg.get("mode", "build")
    if mode == "build":
        run_build(cfg)
    elif mode == "query":
        run_query(cfg)
    elif mode == "both":
        run_build(cfg)
        run_query(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
