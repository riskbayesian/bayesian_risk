#!/usr/bin/env python3
"""
YAML-driven lookup-table (LUT) risk model builder & query runner — **Vectorized & Fast**.

What's new in this version (speed-ups + features):
- **Single FAISS search** reused for risk, ID, debug, and reason outputs
- **Fully vectorized** risk (soft/hard) and reason image generation (no per-pixel Python loops)
- **Chunked FAISS** option for big frames (configurable `params.faiss_chunk`)
- Reason image + legend, symmetric pair LUT with conflict policy
- Optional `.npy` saving, color heatmaps, overlay, ID map, debug maps, upsampling to original size
- No stray `params` lookups inside model methods; everything passed explicitly

Config: `lookup_lut_config.yaml` next to this file, or set env `LOOKUP_LUT_CONFIG`.
"""
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml
import time

import faiss
import faiss.contrib.torch_utils

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
                # map to [0,1] where 1=safe, 0=unsafe
                risk01 = (rating15 - 1.0) / 4.0
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
                            else:  # 'first'
                                pass
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
    out_dir = paths["out_dir"]
    model.save(out_dir)
    print(f"[BUILD] Saved LUT model to {out_dir}")


def _faiss_topk_chunked(obj_lut: ObjectLUT, feats_flat: torch.Tensor, k: int, chunk: int = 100_000):
    """Chunked FAISS search to control RAM usage for large images."""
    N = feats_flat.shape[0]
    all_D = []
    all_I_obj = []
    for i in range(0, N, chunk):
        x = feats_flat[i:i+chunk]
        D, _, O = obj_lut.topk(x, k=k)
        all_D.append(D)
        all_I_obj.append(O)
    return torch.cat(all_D, dim=0), torch.cat(all_I_obj, dim=0)

def _torch_topk_chunked(model, Xf_cpu: torch.Tensor, k: int, chunk_rows: int = 20000, dtype=torch.float16):
    """Matrix-multiply search on GPU: Xf @ prototypes^T, chunked to control memory."""
    device = model._protos_T.device
    D_all, O_all = [], []
    for i in range(0, Xf_cpu.shape[0], chunk_rows):
        x = Xf_cpu[i:i+chunk_rows].to(device=device, dtype=dtype)     # [n,C]
        # Xf_cpu was L2-normalized already; re-normalize for numerical safety
        x = torch.nn.functional.normalize(x, dim=1)
        S = x @ model._protos_T                                        # [n,M]
        vals, idxs = torch.topk(S, k=k, dim=1)                         # [n,k]
        D_all.append(vals.to(dtype=torch.float32).cpu())
        O_all.append(model._proto_obj_ids_t[idxs].cpu())               # map proto->obj id
        del x, S, vals, idxs
        torch.cuda.synchronize()
    return torch.cat(D_all, 0), torch.cat(O_all, 0)

def run_query(cfg: dict):
    paths = cfg["paths"]
    infer = cfg.get("inference", {})
    params = cfg.get("params", {})

    model = LookupRiskModel.load(paths["model_dir"]) if "model_dir" in paths else None

    # pick backend
    backend = (params.get("nn_backend") or "faiss").lower()

    # optional TF32 speedup
    if params.get("allow_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # prepare Torch GEMM cache if selected
    if backend == "torch":
        device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(model, "_protos_T"):
            P = torch.from_numpy(model.obj_lut.prototypes).to(device=device, dtype=torch.float16)  # [M,C]
            P = torch.nn.functional.normalize(P, dim=1)  # safety: L2 norm 1
            model._protos_T = P.t().contiguous()         # [C,M]
            model._proto_obj_ids_t = torch.from_numpy(model.obj_lut.proto_obj_ids).to(device)

    if model is None:
        print("[QUERY] No model_dir provided; building a temporary model from raw paths...")
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

    # DINO features for the scene image
    dino = _ensure_dino(params.get("device"))
    rgb_full = np.array(Image.open(paths["image"]).convert("RGB"))


    tnow = time.time()
    torch.cuda.synchronize()
    x = dino.rgb_image_preprocessing(rgb_full, params.get("target_h_query", 560))
    feats = dino.get_dino_features_rgb(x)  # [C,H,W]
    torch.cuda.synchronize()
    dino_time = time.time() - tnow
    print(f"[QUERY] DINO features time: {dino_time:.2f} seconds")

    tnow = time.time()
    torch.cuda.synchronize()

    # Flatten feature map and L2-normalize
    C, Hf, Wf = feats.shape
    Xf = feats.permute(1, 2, 0).reshape(-1, C)
    Xf = l2_normalize_rows(Xf)

    Kq = int(infer.get("topk", 5))
    backend = (params.get("nn_backend") or "faiss").lower()

    if backend == "torch":
        gemm_chunk = int(params.get("gemm_chunk", 20000))
        scores, obj_ids_k = _torch_topk_chunked(model, Xf, k=Kq, chunk_rows=gemm_chunk, dtype=torch.float16)
    else:
        chunk = int(params.get("faiss_chunk", 100_000))
        scores, obj_ids_k = _faiss_topk_chunked(model.obj_lut, Xf, k=Kq, chunk=chunk)

    max_sim = scores.max(dim=1).values              # [N]
    top1_ids = obj_ids_k[:, 0]                      # [N]

    # Prepare per-B vectors once
    Bname = infer["object_B"]
    if Bname not in model.obj_lut.name2id:
        raise KeyError(f"Unknown object_B '{Bname}'")
    B_id = model.obj_lut.name2id[Bname]
    default_unknown = float(infer.get("default_unknown", 0.5))
    risk_vec, reason_idx_vec, reason_palette, reason_labels = model._pair_vectors_for_B(B_id, default_unknown)

    # Vectorized risk (soft or hard)
    if infer.get("soft", True):
        logits = scores * float(infer.get("temperature", 20.0))
        weights = torch.softmax(logits, dim=-1)               # [N,K]
        risk_vals = risk_vec[obj_ids_k]                       # [N,K]
        risk_flat = (weights * risk_vals).sum(dim=-1)         # [N]
    else:
        risk_flat = risk_vec[top1_ids]

    torch.cuda.synchronize()
    faiss_time = time.time() - tnow
    print(f"[QUERY] {backend.upper()} search time: {faiss_time:.2f} seconds")

    # Open-set mask → default_unknown
    tau = float(model.tau)
    mask_known = (max_sim >= tau)
    if not isinstance(risk_flat, torch.Tensor):
        risk_flat = torch.tensor(risk_flat, dtype=torch.float32)
    risk_flat = torch.where(mask_known, risk_flat, torch.tensor(default_unknown, dtype=risk_flat.dtype))

    # Reshape + (optional) upsample to original
    vis_cfg = cfg.get("visualization", {})
    resize_to_original = bool(vis_cfg.get("resize_to_original", True))
    H0, W0 = rgb_full.shape[:2]
    risk_map = risk_flat.view(Hf, Wf)
    if resize_to_original:
        t = risk_map[None, None, ...]
        risk_up = torch.nn.functional.interpolate(t, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
        risk_map = risk_up

    # --- Save outputs (optionally .npy) ---
    save_numpy = bool(vis_cfg.get("save_numpy", True))
    risk_np = risk_map.detach().cpu().numpy()
    if save_numpy:
        out_path = paths.get("risk_out") or str(Path(paths["image"]).with_suffix(".risk.npy"))
        np.save(out_path, risk_np)
        print(f"[QUERY] Saved risk map to {out_path}")

    # --- Risk PNG ---
    risk_u8 = (np.clip(risk_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    risk_png = paths.get("risk_png") or str(Path(paths["image"]).with_suffix(".risk.png"))
    Image.fromarray(risk_u8).save(risk_png)
    print(f"[QUERY] Saved risk PNG to {risk_png}")

    # Optional colored/overlay
    if vis_cfg.get("colored", False):
        import matplotlib.pyplot as plt
        cmap_name = vis_cfg.get("cmap", "viridis")
        plt.figure()
        plt.imshow(risk_np, cmap=cmap_name, vmin=0.0, vmax=1.0)
        if vis_cfg.get("colorbar", True):
            plt.colorbar(label="Risk (0-1)")
        plt.axis("off")
        colored_path = vis_cfg.get("colored_out") or str(Path(risk_png).with_suffix(".colored.png"))
        plt.savefig(colored_path, dpi=vis_cfg.get("dpi", 150), bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[QUERY] Saved colored risk to {colored_path}")

    if vis_cfg.get("overlay", False):
        import matplotlib.pyplot as plt
        alpha = float(vis_cfg.get("alpha", 0.5))
        plt.figure()
        plt.imshow(rgb_full)
        plt.imshow(risk_np, cmap=vis_cfg.get("cmap", "viridis"), vmin=0.0, vmax=1.0, alpha=alpha)
        plt.axis("off")
        overlay_path = vis_cfg.get("overlay_out") or str(Path(risk_png).with_suffix(".overlay.png"))
        plt.savefig(overlay_path, dpi=vis_cfg.get("dpi", 150), bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[QUERY] Saved overlay to {overlay_path}")

    # === ID PNG (top-1) using same FAISS results ===
    if vis_cfg.get("save_id_png", True):
        id_map = top1_ids.view(Hf, Wf).cpu().numpy()
        unknown_mask = (max_sim.view(Hf, Wf).cpu().numpy() < tau)
        id_map[unknown_mask] = -1
        if resize_to_original:
            id_t = torch.from_numpy(id_map)[None, None, ...].float()
            id_up = torch.nn.functional.interpolate(id_t, size=(H0, W0), mode="nearest")[0, 0].long().numpy()
        else:
            id_up = id_map
        import colorsys
        max_id = max(model.obj_lut.id2name.keys()) if model.obj_lut.id2name else 0
        palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for oid in range(max_id + 1):
            h = ((oid * 0.61803398875) % 1.0)
            s, v = 0.65, 1.0
            r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
            palette[oid] = (int(r_ * 255), int(g_ * 255), int(b_ * 255))
        unknown_color = tuple(vis_cfg.get("id_unknown_color", [0, 0, 0]))
        id_rgb = np.zeros((id_up.shape[0], id_up.shape[1], 3), dtype=np.uint8)
        known_mask_im = id_up >= 0
        if known_mask_im.any():
            id_rgb[known_mask_im] = palette[id_up[known_mask_im]]
        if (~known_mask_im).any():
            id_rgb[~known_mask_im] = np.array(unknown_color, dtype=np.uint8)
        id_png = paths.get("id_png") or str(Path(paths["image"]).with_suffix(".id.png"))
        Image.fromarray(id_rgb).save(id_png)
        print(f"[QUERY] Saved object-id PNG to {id_png}")
        legend_path = paths.get("id_legend") or str(Path(id_png).with_suffix(".legend.json"))
        with open(legend_path, "w") as f:
            json.dump(model.obj_lut.id2name, f, indent=2)
        print(f"[QUERY] Saved id legend to {legend_path}")

    # === DEBUG maps (reuse scores/max_sim) ===
    debug_cfg = cfg.get("debug", {})
    if debug_cfg.get("save_debug_maps", True):
        sim_map = np.clip(max_sim.view(Hf, Wf).cpu().numpy(), 0.0, 1.0)
        sim_u8 = (sim_map * 255.0).round().astype(np.uint8)
        unk_mask = (max_sim < tau).view(Hf, Wf).cpu().numpy().astype(np.uint8) * 255
        base = Path(paths["image"]).with_suffix("")
        sim_png = str(base) + ".maxsim.png"
        unk_png = str(base) + ".unknown.png"
        Image.fromarray(sim_u8).save(sim_png)
        Image.fromarray(unk_mask).save(unk_png)
        unknown_ratio = float(unk_mask.mean() / 255.0)
        print(f"[QUERY][DEBUG] Saved max-sim to {sim_png}, unknown mask to {unk_png} (unknown ratio={unknown_ratio:.3f})")
        if debug_cfg.get("save_pair_miss_map", True):
            has_pair = (risk_vec[top1_ids] != default_unknown).cpu().numpy().reshape(Hf, Wf)
            miss = (~has_pair & (unk_mask == 0)).astype(np.uint8) * 255
            miss_png = str(base) + ".pair_miss.png"
            Image.fromarray(miss).save(miss_png)
            miss_ratio = float(miss.mean() / 255.0)
            print(f"[QUERY][DEBUG] Saved pair-miss map to {miss_png} (miss ratio={miss_ratio:.3f})")

    # === Reason PNG (top-1) using same FAISS results ===
    if vis_cfg.get("save_reason_png", True):
        # Indices per object → per-pixel
        reason_idx_per_obj = reason_idx_vec.numpy()  # [num_objects]
        reason_idx_img = reason_idx_per_obj[top1_ids.cpu().numpy()].reshape(Hf, Wf)
        # Overrides for unknown & missing-pair
        RIDX_UNKNOWN = 0  # order in _pair_vectors_for_B: ["__unknown__", "__missing_pair__", "__no_reason__", ...]
        RIDX_MISSING = 1
        unknown_mask = (max_sim.view(Hf, Wf).cpu().numpy() < tau)
        has_pair = (risk_vec[top1_ids] != default_unknown).cpu().numpy().reshape(Hf, Wf)
        reason_idx_img[unknown_mask] = RIDX_UNKNOWN
        reason_idx_img[~unknown_mask & (~has_pair)] = RIDX_MISSING
        reason_rgb_feat = reason_palette[reason_idx_img]  # [Hf,Wf,3]
        if resize_to_original:
            t = torch.from_numpy(reason_rgb_feat).permute(2, 0, 1).float()[None]
            up = torch.nn.functional.interpolate(t, size=(H0, W0), mode="nearest")[0].permute(1, 2, 0).byte().numpy()
            reason_rgb = up
        else:
            reason_rgb = reason_rgb_feat
        reason_png = paths.get("reason_png") or str(Path(paths["image"]).with_suffix(".reason.png"))
        Image.fromarray(reason_rgb).save(reason_png)
        print(f"[QUERY] Saved reason PNG to {reason_png}")
        # Legend PNG
        legend_png = paths.get("reason_legend_png") or str(Path(reason_png).with_suffix(".legend.png"))
        try:
            from PIL import ImageDraw, ImageFont
            entries = []
            # specials first
            specials = [(tuple(reason_palette[0]), "unknown (below τ)"),
                        (tuple(reason_palette[1]), "missing pair (A,B)"),
                        (tuple(reason_palette[2]), "no reason in CSV")]
            entries.extend(specials)
            # real reasons
            real_labels = reason_labels[3:]
            for idx, label in enumerate(real_labels, start=3):
                entries.append((tuple(reason_palette[idx]), label if label else "(empty)"))
            sw, sh = 24, 24
            pad, gap = 10, 8
            font = ImageFont.load_default()
            max_text_w = 480
            W = pad * 2 + sw + 8 + max_text_w
            H = pad * 2 + len(entries) * (sh + gap) - gap
            canvas = Image.new("RGB", (W, H), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            y = pad
            for color, label in entries:
                draw.rectangle([pad, y, pad + sw, y + sh], fill=color, outline=(0, 0, 0))
                draw.text((pad + sw + 8, y + 4), label, fill=(0, 0, 0), font=font)
                y += sh + gap
            canvas.save(legend_png)
            print(f"[QUERY] Saved reason legend to {legend_png}")
        except Exception as e:
            print(f"[QUERY][WARN] Could not render reason legend: {e}")

    print("[QUERY] Done.")

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
