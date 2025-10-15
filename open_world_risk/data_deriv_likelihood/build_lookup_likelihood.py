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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml
import time
import csv

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

def load_yaml_config(default_name: str = "lookup_lut_likelihood_config.yaml") -> dict:
    cfg_path = os.environ.get("LOOKUP_LUT_CONFIG")
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), default_name)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def bernstein_B(num_ctrl: int, S: int = 100, device: str = "cuda", dtype=torch.float32):
    n = num_ctrl - 1
    t = torch.linspace(0, 1, S, device=device, dtype=dtype)
    comb = [math.comb(n, k) for k in range(num_ctrl)]
    B = []
    for k in range(num_ctrl):
        Bk = comb[k] * (t**k) * ((1.0 - t)**(n - k))
        B.append(Bk)
    return torch.stack(B, dim=1), t  # [S,K], [S]

# --------------------------- Pair LUT ---------------------------

@dataclass
class PairInfo:
    ctrl_pts: List[float]

class PairLUT:
    """(A_id, B_id) → (risk, reason). Order-invariant support and a reason palette."""
    def __init__(self, name2id: Dict[str, int], symmetric: bool = True, conflict_policy: str = "mean"):
        self.name2id = dict(name2id)
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.table: Dict[Tuple[int, int], PairInfo] = {}
        self.symmetric = bool(symmetric)
        self.conflict_policy = conflict_policy  # 'mean' | 'first' | 'max'

    @classmethod
    def from_dataset(cls, dataset_files: List[str], label_to_id_filename: str, 
                     id_to_dino_feat_filenames: List[str], symmetric: bool = True, 
                     conflict_policy: str = "mean") -> "PairLUT":
        """
        Build PairLUT from dataset files produced by dataset_creator.py and dataset_loader.py.
        
        Args:
            dataset_files: List of .pt files containing [dino_feat_a, dino_feat_b, control_pts]
            label_to_id_filename: CSV file with columns [label, id] produced by dino_to_label_write
            id_to_dino_feat_filenames: List of PyTorch file paths containing {"ids": tensor, "data": tensor}
            symmetric: Whether to make the lookup table symmetric (default: True)
            conflict_policy: How to handle conflicts - only "mean" supported for now (default: "mean")
            
        Returns:
            PairLUT: Lookup table populated with control points from dataset
        """
        
        # Load label-to-id mapping from CSV
        name2id = {}
        with open(label_to_id_filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].strip()
                id_val = int(row["id"])
                name2id[label] = id_val

        # Debug: Print all entries in name2id
        print(f"name2id mapping contains {len(name2id)} entries:")
        for label, id_val in name2id.items():
            print(f"  '{label}' -> {id_val}")
        
        # Load DINO features from PyTorch files (same as build_from_dataset_creator)
        all_ids = []
        all_features = []
        for filename in id_to_dino_feat_filenames:
            data = torch.load(filename, map_location='cpu')
            ids = data['ids']  # tensor of label indices
            features = data['data']  # tensor of dino features [N, D]
            all_ids.append(ids)
            all_features.append(features)
        
        # Concatenate all data
        all_ids = torch.cat(all_ids, dim=0)
        all_features = torch.cat(all_features, dim=0)
        
        # Group DINO features by label (same as build_from_dataset_creator)
        feature_dict = {}  # label -> list of dino_feat tensors
        for i, label_idx in enumerate(all_ids):
            # Get label name from the CSV mapping
            label_name = None
            for label, id_val in name2id.items():
                if label_idx.item() == id_val:
                    label_name = label
                    break
            
            if label_name is None:
                raise ValueError(f"Label index {label_idx.item()} not found in CSV mapping")
                
            if label_name not in feature_dict:
                feature_dict[label_name] = []
            feature_dict[label_name].append(all_features[i])
        
        # Create PairLUT instance
        lut = cls(name2id, symmetric=symmetric, conflict_policy=conflict_policy)
        
        # Load and process dataset files
        print("loading and processing dataset_files:")
        for df in dataset_files:
            print(f"  {df}")
        # Sort dataset files for consistent processing order
        dataset_files = sorted(dataset_files)

        for dataset_file in dataset_files:
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
                
            # Load dataset tensor: [N, 2048 + ctrl_pts] where 2048 = 1024 + 1024 (dino_feat_a + dino_feat_b)
            dataset = torch.load(dataset_file, map_location='cpu')
            if not isinstance(dataset, torch.Tensor) or dataset.ndim != 2:
                raise ValueError(f"Invalid dataset format in {dataset_file}: expected 2D torch.Tensor, got {type(dataset)} with shape {getattr(dataset, 'shape', 'N/A')}")
            
            # Extract features and control points
            dino_feat_a = dataset[:, :1024]  # First 1024 dimensions
            dino_feat_b = dataset[:, 1024:2048]  # Next 1024 dimensions  
            control_points = dataset[:, 2048:]  # Remaining dimensions are control points
            
            print(f"Processing {dataset_file}: {dataset.shape[0]} samples, {control_points.shape[1]} control points")
            
            # For each sample, match DINO features to labels using cosine similarity
            #TESTING_MAX_ONLY = 50
            for sample_idx in tqdm(range(dataset.shape[0]), desc=f"Processing {os.path.basename(dataset_file)}"):
            #for sample_idx in tqdm(range(TESTING_MAX_ONLY), desc=f"Processing {os.path.basename(dataset_file)}"):
                feat_a = dino_feat_a[sample_idx]
                feat_b = dino_feat_b[sample_idx]
                ctrl_pts = control_points[sample_idx]
                
                # Find best matching labels for both features using cosine similarity
                def find_best_label_match(feat, feature_dict):
                    best_label = None
                    best_similarity = -1.0
                    
                    for label, label_features in feature_dict.items():
                        if not label_features:
                            continue
                            
                        # Compute cosine similarity with all features for this label
                        label_tensor = torch.stack(label_features, dim=0)  # [N, D]
                        similarities = torch.nn.functional.cosine_similarity(
                            feat.unsqueeze(0), label_tensor, dim=1
                        )
                        max_sim = similarities.max().item()
                        
                        if max_sim > best_similarity:
                            best_similarity = max_sim
                            best_label = label
                    
                    return best_label, best_similarity
                
                label_a, sim_a = find_best_label_match(feat_a, feature_dict)
                label_b, sim_b = find_best_label_match(feat_b, feature_dict)
                
                # Only create entry if we found good matches for both features
                if label_a is not None and label_b is not None and label_a != label_b:
                    aid, bid = name2id[label_a], name2id[label_b]
                    
                    def _insert(xid, yid, ctrl_pts):
                        key = (xid, yid)
                        if key in lut.table:
                            old = lut.table[key]
                            if conflict_policy == "mean":
                                # Average the control points
                                ctrl_pts_list = ctrl_pts.tolist() if hasattr(ctrl_pts, 'tolist') else ctrl_pts
                                new_ctrl_pts = [(old_ctrl + new_ctrl) / 2.0 
                                               for old_ctrl, new_ctrl in zip(old.ctrl_pts, ctrl_pts_list)]
                                lut.table[key] = PairInfo(ctrl_pts=new_ctrl_pts)
                            elif conflict_policy == "max":
                                raise RuntimeError("Conflict policy 'max' not implemented for control points")
                            elif conflict_policy == "first":
                                raise RuntimeError("Conflict policy 'first' not implemented for control points")
                            else:
                                raise RuntimeError(f"Unknown conflict policy: {conflict_policy}")
                        else:
                            lut.table[key] = PairInfo(ctrl_pts=ctrl_pts.tolist())
                    
                    _insert(aid, bid, ctrl_pts)
                    if symmetric:
                        _insert(bid, aid, ctrl_pts)
        
        print(f"Built PairLUT from dataset with {len(lut.table)} entries")
        return lut

    def lookup(self, a_id: int, b_id: int, default: List[float] = None) -> PairInfo:
        info = self.table.get((a_id, b_id))
        if info is None and self.symmetric:
            info = self.table.get((b_id, a_id))
        if info is None:
            return PairInfo(ctrl_pts=default or [])
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
            }, f, indent=2)
        serial = {f"{a},{b}": {"ctrl_pts": v.ctrl_pts.tolist() if hasattr(v.ctrl_pts, 'tolist') else v.ctrl_pts}
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
        meta_path = p / "meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path, "r"))
            symmetric = bool(meta.get("symmetric", True))
            conflict_policy = meta.get("conflict_policy", "mean")
        with open(p / "pair_lut.json", "r") as f:
            serial = json.load(f)
        lut = cls(name2id, symmetric=symmetric, conflict_policy=conflict_policy)
        for k, v in serial.items():
            a, b = map(int, k.split(","))
            lut.table[(a, b)] = PairInfo(ctrl_pts=v.get("ctrl_pts", []))
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

    # TODO: Update config to use this and to NEVER use build_from_folders() or build_from_feature_dict() 
    # NOTE: UNTESTED, we may run out of mem... 
    def build_from_dataset_creator(self, label_to_id_filename: str, id_to_dino_feat_filenames: List[str], K: int = 32):
        """
        Build ObjectLUT from dataset creator output files.
        
        Args:
            label_to_id_filename: Path to CSV file with label,id columns (label names -> id)
            id_to_dino_feat_filenames: List of PyTorch file paths containing {"ids": tensor, "data": tensor}
            K: Number of centroids per object for k-means clustering
        """
        
        # Load label-to-id mapping from CSV
        label_to_ids = {}  # label -> list of ids
        with open(label_to_id_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                label = row['label']
                id_val = int(row['id'])
                if label not in label_to_ids:
                    label_to_ids[label] = []
                label_to_ids[label].append(id_val)
        
        # Load DINO features from PyTorch files
        all_ids = []
        all_features = []
        for filename in id_to_dino_feat_filenames:
            data = torch.load(filename, map_location='cpu')
            ids = data['ids']  # tensor of label indices
            features = data['data']  # tensor of dino features [N, D]
            all_ids.append(ids)
            all_features.append(features)
        
        # Concatenate all data
        all_ids = torch.cat(all_ids, dim=0)
        all_features = torch.cat(all_features, dim=0)
        
        # Group DINO features by label
        feature_dict = {}  # label -> list of dino_feat tensors
        for i, label_idx in enumerate(all_ids):
            # Get label name from the CSV mapping
            label_name = None
            for label, ids in label_to_ids.items():
                if label_idx.item() in ids:
                    label_name = label
                    break
            
            if label_name is None:
                raise ValueError(f"Label index {label_idx.item()} not found in CSV mapping")
                
            if label_name not in feature_dict:
                feature_dict[label_name] = []
            feature_dict[label_name].append(all_features[i])
        
        # Build centroids using the same logic as build_from_feature_dict
        all_centroids = []
        all_proto_ids = []
        C = None
        
        for name, vecs in feature_dict.items():
            if not vecs:
                continue
            
            # Ensure we have the name in our name2id mapping
            if name not in self.name2id:
                raise ValueError(f"Label '{name}' not found in name2id mapping")
                
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
            all_proto_ids.append(np.full((centroids.shape[0],), self.name2id[name], dtype=np.int32))
        
        if not all_centroids:
            raise RuntimeError("No centroids built from dataset creator files; check inputs.")
        
        self.prototypes = np.concatenate(all_centroids, axis=0).astype("float32")
        self.proto_obj_ids = np.concatenate(all_proto_ids, axis=0).astype("int32")
        d = self.prototypes.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.prototypes)
        
        print(f"Built ObjectLUT from dataset creator:")
        print(f"  - Total prototypes: {len(self.prototypes)}")
        print(f"  - Feature dimension: {d}")
        print(f"  - Objects: {len(feature_dict)}")
        
        # Show a few entries from feature_dict for debugging
        print(f"\nFeature dictionary sample (showing up to 5 entries):")
        print("=" * 60)
        for i, (label, features) in enumerate(list(feature_dict.items())[:5]):
            print(f"  [{i+1}] '{label}' -> {len(features)} feature vectors")
            if features:
                feat_shape = features[0].shape
                print(f"      Feature shape: {feat_shape}")
        if len(feature_dict) > 5:
            print(f"  ... and {len(feature_dict) - 5} more entries")
        print("=" * 60)

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
    def build_from_dataset(
        cls,
        dataset_files: List[str],
        label_to_id_filename: str,
        id_to_dino_feat_filenames: List[str],
        K: int = 32,
        open_set_threshold: float = 0.65,
        temperature: float = 20.0,
        symmetric: bool = True,
        conflict_policy: str = "mean",
    ) -> "LookupRiskModel":
        """
        Build LookupRiskModel from dataset files produced by dataset_creator.py and dataset_loader.py.
        
        Args:
            dataset_files: List of .pt files containing [dino_feat_a, dino_feat_b, control_pts]
            label_to_id_filename: CSV file with columns [label, id] produced by dino_to_label_write
            id_to_dino_feat_filenames: List of PyTorch file paths containing {"ids": tensor, "data": tensor}
            K: Number of centroids per object for k-means clustering
            open_set_threshold: Threshold for open set detection (default: 0.65)
            temperature: Temperature parameter for risk computation (default: 20.0)
            symmetric: Whether to make the lookup table symmetric (default: True)
            conflict_policy: How to handle conflicts - only "mean" supported for now (default: "mean")
            
        Returns:
            LookupRiskModel: Model built from dataset files
        """
        # Build PairLUT from dataset files
        pair_lut = PairLUT.from_dataset(
            dataset_files=dataset_files,
            label_to_id_filename=label_to_id_filename,
            id_to_dino_feat_filenames=id_to_dino_feat_filenames,
            symmetric=symmetric,
            conflict_policy=conflict_policy
        )
        
        # Build ObjectLUT from dataset creator files
        obj_lut = ObjectLUT(name2id=pair_lut.name2id)
        obj_lut.build_from_dataset_creator(
            label_to_id_filename=label_to_id_filename,
            id_to_dino_feat_filenames=id_to_dino_feat_filenames,
            K=K
        )
        
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
        reason_labels = specials  # No real reasons for now since we're using control points
        palette = {
            "__unknown__": [0, 0, 0],
            "__missing_pair__": [255, 255, 255],
            "__no_reason__": [128, 128, 128],
        }
        reason_palette_list = np.array([palette[name] for name in reason_labels], dtype=np.uint8)
        reason2idx = {name: i for i, name in enumerate(reason_labels)}
        reason_idx_vec = torch.full((num_obj,), reason2idx["__no_reason__"], dtype=torch.int32)
        tbl = self.pair_lut.table
        for (a, b), info in tbl.items():
            item_a = a
            item_b = b
            #import pdb; pdb.set_trace()
            if b == B_id:
                # Compute risk from control points (average of control points as a simple heuristic)
                if info.ctrl_pts:
                    #risk_val = float(np.mean(info.ctrl_pts))
                    num_ctrl_pts = len(info.ctrl_pts)
                    Bbern, t = bernstein_B(num_ctrl_pts, S=100, device="cpu", dtype=torch.float32)
                    d = 0.5 * t # 0.5 is MAX DISTANCE: TODO put in cfg           # [100]
                    # Reshape control points to [1, 1, K] for einsum compatibility
                    ctrl_pts_tensor = torch.tensor(info.ctrl_pts).to("cpu").view(1, 1, -1)  # [1, 1, K]
                    probs = torch.einsum("hwk,sk->hws", ctrl_pts_tensor, Bbern).clamp(0, 1)  # [1, 1, 100]
                    exp_un = torch.einsum("hws,s->hw", probs, d).contiguous()  # [1, 1]
                    risk_val = exp_un.item()  # Convert to scalar
                    risk_vec[a] = risk_val
                else:
                    risk_vec[a] = float(default_unknown)
                    #import pdb; pdb.set_trace()
                # For now, all pairs get the same reason index since we don't have text reasons
                reason_idx_vec[a] = int(reason2idx["__no_reason__"])
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

    model = LookupRiskModel.build_from_dataset(
        dataset_files=paths["dataset_files"],
        label_to_id_filename=paths["label_to_id_filename"],
        id_to_dino_feat_filenames=paths["id_to_dino_feat_filenames"],
        K=params.get("K", 32),
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
    assert model is not None, "invalid! model is none, pass in valid model!!!"

    # Debug: Print compact summary of pair lookup table
    print(f"pair_lut.table contains {len(model.pair_lut.table)} entries:")
    if len(model.pair_lut.table) > 0:
        # Show first 10 pairs compactly
        keys = list(model.pair_lut.table.keys())[:10]
        print(f"  First 10 pairs: {', '.join(f'({a},{b})' for a,b in keys)}")
        if len(model.pair_lut.table) > 10:
            print(f"  ... and {len(model.pair_lut.table) - 10} more")
        
        # Check specifically for stove pairs
        stove_pairs = [(a, b) for (a, b) in model.pair_lut.table.keys() if b == 5]
        print(f"  Pairs with stove (B=5): {len(stove_pairs)}")
        if stove_pairs:
            print(f"    {stove_pairs[:5]}")  # Show first 5 stove pairs
    else:
        print("  No pairs found!")

    # Debug: Show which objects have pairs (compact format)
    id2name_debug = model.pair_lut.id2name
    id2name_debug_k = list(id2name_debug.keys())

    for label, id_val in id2name_debug.items():
            print(f"  '{label}' -> {id_val}")
    
    if len(id2name_debug_k) > 0:
        print(f"Pair matrix ({len(id2name_debug_k)}x{len(id2name_debug_k)} objects):")
        # Show compact matrix - only first 10x10 if too large
        max_show = min(float('inf'), len(id2name_debug_k))
        col_header = "     " + " ".join(f"{k_i:>3}" for k_i in id2name_debug_k[:max_show])
        print(col_header)
        
        for j, k_j in enumerate(id2name_debug_k[:max_show]):
            row_values = []
            for i, k_i in enumerate(id2name_debug_k[:max_show]):
                k_ij = (k_i, k_j)
                val = 1 if k_ij in model.pair_lut.table else 0
                row_values.append(f"{val:>3}")
            row_label = f"{k_j:>3}:"
            print(row_label + " " + " ".join(row_values))
        
        if len(id2name_debug_k) > 10:
            print(f"  ... (showing first 10x10, total {len(id2name_debug_k)}x{len(id2name_debug_k)})")

    # Debug: Count entries per object ID in the pair lookup table
    pairs_seen = 0
    if len(id2name_debug_k) > 0:
        print("\nEntries per object ID in pair lookup table:")
        for obj_id in sorted(id2name_debug_k):
            # Count pairs where this object appears as first element
            count_as_first = sum(1 for (a, b) in model.pair_lut.table.keys() if a == obj_id)
            # Count pairs where this object appears as second element
            count_as_second = sum(1 for (a, b) in model.pair_lut.table.keys() if b == obj_id)
            # Total unique pairs involving this object
            total_pairs = len(set([(a, b) for (a, b) in model.pair_lut.table.keys() if a == obj_id or b == obj_id]))
            
            obj_name = id2name_debug.get(obj_id, f"unknown_{obj_id}")
            #print(f"  ID {obj_id} ({obj_name}): {count_as_first} as first, {count_as_second} as second, {total_pairs} total pairs")
            print(f"  ID {obj_id} ({obj_name}): {count_as_first} in row")
            pairs_seen += count_as_first

    # Calculate number of possible pairs: n choose 2
    n = len(id2name_debug_k)
    if n >= 2:
        n_choose_2 = n * (n + 1) // 2
        print(f"Pairs seen: {pairs_seen} Total possible pairs (n+1 choose 2): {n_choose_2} Percentage: {pairs_seen / n_choose_2}")

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
