# Standard library imports
import functools
import glob
import math
import os
import re
import threading
import time
from typing import List, Tuple, Dict
import random
from datetime import datetime

# Third-party imports
import h5py
import numpy as np
import cvxpy as cp
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pprint import pprint
from scipy.ndimage import gaussian_filter1d
import faiss

# Local imports
from open_world_risk.src.video_datastructures import ManipulationTimeline, VideoFrameData1
from open_world_risk.video_processing.process_video_utils import get_min_distances

def timeFunction(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        t0 = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.time() - t0
            print(f"[Timing] {dt:.4f}s {func.__qualname__} ")
            args_str = str(args[1:] if len(args) > 1 else [])[:100] + ("..." if len(str(args[1:] if len(args) > 1 else [])) > 100 else "")
            kwargs_str = str(kwargs)[:100] + ("..." if len(str(kwargs)) > 100 else "")
            #print(f"[Timing] \t args: {args_str}\n\t\t kwargs: {kwargs_str}")
    return inner

class HistogramDataset(Dataset):
    """   
    Expected tensor format: [N, 2048 + ctrl_pts] with layout:
    [featA(1024), featB(1024), control_points(ctrl_pts)]
    
    Input: concatenated DINO features [featA, featB] -> [2048]
    Target: control_points -> [ctrl_pts] (pre-computed Bezier control points)
    
    The dataset now contains pre-computed Bezier control points for efficient training.
    """
    
    def __init__(self, tensor_paths: list, ctrl_pts: int = 10, feat_size: int = 1024):
        super().__init__()
        
        # Load and concatenate all tensors
        all_tensors = []
        for path in tensor_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tensor file not found: {path}")
            
            tensor = torch.load(path, map_location="cpu")
            if not (isinstance(tensor, torch.Tensor) and tensor.ndim == 2):
                raise ValueError(f"Expected 2D tensor in {path}, got {tensor.shape}")
            
            expected_cols = (feat_size * 2) + ctrl_pts  # 2048 + ctrl_pts
            if tensor.shape[1] != expected_cols:
                raise ValueError(f"Expected {expected_cols} columns in {path}, got {tensor.shape[1]}")
            
            all_tensors.append(tensor)
            print(f"Loaded {tensor.shape[0]} samples from {path}")
        
        # Concatenate all tensors
        self.data = torch.cat(all_tensors, dim=0)
        print(f"Total dataset size: {self.data.shape[0]} samples")
        
        self.ctrl_pts_n = ctrl_pts
        self.feat_size = feat_size
        self.total_len = (feat_size * 2) + ctrl_pts

        # Define slice indices
        self.featA_start, self.featA_end = 0, feat_size
        self.featB_start, self.featB_end = feat_size, feat_size * 2
        self.ctrl_start, self.ctrl_end = feat_size * 2, self.total_len
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Extract features and control points
        featA = row[self.featA_start:self.featA_end].float()
        featB = row[self.featB_start:self.featB_end].float()
        control_points = row[self.ctrl_start:self.ctrl_end].float()
        
        # Always predict control points from features
        inputs = torch.cat([featA, featB], dim=0)  # [2048]
        target = control_points  # [ctrl_pts]
        return inputs, target
    
    def get_feature_dims(self):
        """Get input and target dimensions for this dataset"""
        return 2*self.feat_size, self.ctrl_pts_n  # input_dim, target_dim
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        Uses pre-computed Bezier control points for efficient training.
        
        Args:
            batch: List of (inputs, targets) tuples from __getitem__
            
        Returns:
            inputs: dict with keys:
                - 'dino1': [batch_size, 1024] - first DINO features
                - 'dino2': [batch_size, 1024] - second DINO features
            targets: dict with keys:
                - 'control_points': [batch_size, num_ctrl] - pre-computed Bezier control points
        """
        # Extract inputs and targets
        inputs, targets = zip(*batch)
        
        # Split concatenated inputs back into dino1 and dino2
        inputs_tensor = torch.stack(inputs, dim=0)  # [batch_size, 2048]
        dino1 = inputs_tensor[:, :self.feat_size]  # [batch_size, 1024]
        dino2 = inputs_tensor[:, self.feat_size:]  # [batch_size, 1024]
        
        # Stack targets (control points)
        control_points = torch.stack(targets, dim=0)  # [batch_size, ctrl_pts]
        
        # Return inputs and targets as dictionaries
        inputs_dict = {
            'dino1': dino1,  # [batch_size, 1024]
            'dino2': dino2   # [batch_size, 1024]
        }
        
        targets_dict = {'control_points': control_points}  # [batch_size, num_ctrl]
        
        return inputs_dict, targets_dict

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        """
        Create a DataLoader with the custom collate function automatically applied.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)

    @staticmethod
    def fit_bezier_monotone(y: np.ndarray, num_ctrl: int, clamp01: bool = True, solver_order=("OSQP","ECOS","SCS")) -> np.ndarray:
        """
        Solve min_c ||B c - y||_2^2 s.t. c is nondecreasing (and optionally 0<=c<=1).
        y: [S] probability samples in [0,1]
        returns c: [num_ctrl]
        """
        S = y.shape[0]
        t = np.linspace(0.0, 1.0, S, dtype=np.float64)
        B = HistogramDataset.bernstein_basis(num_ctrl, t)          # [S, K]

        c = cp.Variable(num_ctrl)
        obj = cp.Minimize(cp.sum_squares(B @ c - y))
        cons = [c[1:] - c[:-1] >= 0]              # monotone nondecreasing
        if clamp01:
            cons += [c >= 0, c <= 1]

        prob = cp.Problem(obj, cons)

        # Try solvers in order
        last_err = None
        for s in solver_order:
            try:
                prob.solve(solver=getattr(cp, s), verbose=False, warm_start=True)
                if c.value is not None:
                    return np.asarray(c.value, dtype=np.float64)
            except Exception as e:
                last_err = e
                continue

        # Fallback (no CVXPY solution): use sampled points as coarse controls and enforce monotone by cummax
        # Choose indices roughly evenly spaced across y
        idx = np.round(np.linspace(0, S-1, num_ctrl)).astype(int)
        c0 = y[idx].astype(np.float64)
        c0 = np.maximum.accumulate(c0)  # enforce nondecreasing
        if clamp01:
            c0 = np.clip(c0, 0.0, 1.0)
        return c0

    @staticmethod
    def bernstein_basis(num_ctrl: int, t: np.ndarray) -> np.ndarray:
        """
        Build Bernstein basis matrix B of shape [len(t), num_ctrl]
        B[i, k] = C(n, k) * t_i^k * (1 - t_i)^(n - k), with n = num_ctrl - 1.
        """
        assert num_ctrl >= 2, "num_ctrl must be at least 2"
        n = num_ctrl - 1
        T = t.reshape(-1, 1)                     # [S,1]
        one_minus_T = 1.0 - T                    # [S,1]
        # Precompute binomial coefficients
        coeffs = np.array([math.comb(n, k) for k in range(num_ctrl)], dtype=np.float64)  # [K]
        # Powers
        Tk = np.stack([np.power(T, k).squeeze(1) for k in range(num_ctrl)], axis=1)      # [S,K]
        Tmk = np.stack([np.power(one_minus_T, n - k).squeeze(1) for k in range(num_ctrl)], axis=1)  # [S,K]
        B = (Tk * Tmk) * coeffs[None, :]                                              # [S,K]
        return B.astype(np.float64)

def generateHistogram(video_frame_data: VideoFrameData1, manip_metadata: ManipulationTimeline, 
        frame_indices: list[int], max_distance: float = 0.5, min_entries=15, verbose=False) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Generate histograms of distances between manipulated objects and other in_dataset objects.
    
    Args:
        video_frame_data: VideoFrameData1 containing frame data with centroid_xyzs and in_dataset flags
        manip_metadata: ManipulationTimeline containing manipulation periods and timeline
        frame_indices: List of frame numbers corresponding to video_frame_data.frames indices
        max_distance: Maximum distance for histogram range (default: 0.5 meters)
        min_entries: Minimum number of distance entries required to create histogram
        verbose: Whether to print verbose output
    
    Returns:
        Dict mapping (manipulated_obj_id, other_obj_id) -> histogram array
    """
    
    # Step 1: Get all unique object IDs that are in_dataset across all frames
    in_dataset_objects = set()
    no_risk_objects = set()
    for frame_data in video_frame_data.frames:
        in_dataset_indices = torch.where(frame_data["in_dataset"])[0]
        no_risk_indices = torch.where(frame_data["no_risk"])[0]
        for idx in in_dataset_indices:
            in_dataset_objects.add(idx.item())
        for idx in no_risk_indices:
            no_risk_objects.add(idx.item())

    in_dataset_objects = in_dataset_objects - no_risk_objects

    in_dataset_objects = sorted(list(in_dataset_objects))
    no_risk_objects = sorted(list(no_risk_objects))
    if verbose:
        print(f"Found {len(in_dataset_objects)} objects in dataset: {in_dataset_objects}")
        print(f"Found {len(no_risk_objects)} objects in dataset: {no_risk_objects}")
    
    # Step 2: Store raw distances for each pair of object IDs
    # We'll create distance lists dynamically as we encounter manipulated objects
    raw_distances = {}
    
    # Step 3: Process each manipulation period
    for start_frame, end_frame in tqdm(manip_metadata.manipulation_periods, desc="Processing manipulation periods"):
        if verbose:
            print(f"Processing manipulation period: frames {start_frame} to {end_frame}")
        
        # Convert frame numbers to list positions
        # Use the same conversion as update_manip_metadata: pos = frame_number - frame_indices[0]
        start_pos = start_frame - frame_indices[0]
        end_pos = end_frame - frame_indices[0]
        
        # Validate positions are within bounds
        if start_pos < 0 or start_pos >= len(video_frame_data.frames):
            raise IndexError(f"start_frame {start_frame} -> pos {start_pos} out of bounds [0, {len(video_frame_data.frames)-1}]. frame_indices[0]={frame_indices[0]}")
        if end_pos < 0 or end_pos >= len(video_frame_data.frames):
            raise IndexError(f"end_frame {end_frame} -> pos {end_pos} out of bounds [0, {len(video_frame_data.frames)-1}]. frame_indices[0]={frame_indices[0]}")
        
        # Get manipulated object ID from the first frame of the period
        manipulated_obj_id = video_frame_data.frames[start_pos]["manip_idx"]
        
        if manipulated_obj_id == -1:
            raise ValueError(f"No manipulated object found for period {start_frame}-{end_frame}. This indicates the manipulation period was detected but no object is marked as manipulated.")
        if verbose:
            print(f"Manipulated object ID: {manipulated_obj_id}")
        
        # Step 4: Calculate distances for each frame in the manipulation period
        for frame_idx in range(start_frame, end_frame + 1):
            frame_pos = frame_idx - frame_indices[0]
            
            # Validate position is within bounds
            if frame_pos < 0 or frame_pos >= len(video_frame_data.frames):
                raise IndexError(f"frame_idx {frame_idx} -> pos {frame_pos} out of bounds [0, {len(video_frame_data.frames)-1}]. frame_indices[0]={frame_indices[0]}")
                
            frame_data = video_frame_data.frames[frame_pos]
            
            # Check if we have centroid data and the manipulated object is in this frame
            if ("centroid_xyzs" not in frame_data or 
                "in_dataset" not in frame_data or
                manipulated_obj_id >= len(frame_data["centroid_xyzs"]) or
                not frame_data["in_dataset"][manipulated_obj_id]):
                continue
            
            manipulated_centroid = frame_data["centroid_xyzs"][manipulated_obj_id]
            
            # Skip if manipulated centroid is invalid (all -1s)
            invalid_centroid = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32, device=manipulated_centroid.device)
            if torch.allclose(manipulated_centroid, invalid_centroid, atol=1e-6):
                continue
                        
            # Calculate distances to all other in_dataset objects
            for other_obj_id in in_dataset_objects:
                if (other_obj_id == manipulated_obj_id or other_obj_id >= len(frame_data["centroid_xyzs"]) or
                    not frame_data["in_dataset"][other_obj_id]):
                    continue
                
                other_centroid = frame_data["centroid_xyzs"][other_obj_id]
                invalid_centroid = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32, device=other_centroid.device)
                if torch.allclose(other_centroid, invalid_centroid, atol=1e-6):
                    continue
                
                # Create distance list for this pair if it doesn't exist
                # Guaranteed: manipulated_obj_id is the manipulated object, other_obj_id is the other object
                distance_key = (manipulated_obj_id, other_obj_id)
                if distance_key not in raw_distances:
                    raw_distances[distance_key] = []

                distance = torch.norm(manipulated_centroid - other_centroid).item()
                raw_distances[distance_key].append(distance)
            
            # NOTE: This will override previous entries, which is ok 
            for other_obj_id in no_risk_objects:
                if (other_obj_id == manipulated_obj_id or 
                    other_obj_id >= len(frame_data["centroid_xyzs"]) or
                    not frame_data["no_risk"][other_obj_id]):
                    continue
                
                other_centroid = frame_data["centroid_xyzs"][other_obj_id]
                
                # Skip if other centroid is invalid
                invalid_centroid = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32, device=other_centroid.device)
                if torch.allclose(other_centroid, invalid_centroid, atol=1e-6):
                    continue
                
                # Create distance list for this pair if it doesn't exist
                # Guaranteed: manipulated_obj_id is the manipulated object, other_obj_id is the other object
                distance_key = (manipulated_obj_id, other_obj_id)
                if distance_key not in raw_distances:
                    raw_distances[distance_key] = []
                
                # Set distance to -1.0 since we are just going to set this to be special
                distance = -1.0
                raw_distances[distance_key].append(distance)
    
    # Step 6: Convert raw distances to histograms (0-max_distance meters, 100 bins)
    histograms = {}
    for (manipulated_obj_id, other_obj_id), distances in raw_distances.items():
        if len(distances) > min_entries:  # Only create histogram if we have sufficient data
            if -1.0 in distances:
                # For no_risk objects, create an array of all 0 values
                distances_array = np.full(len(distances), 0)
                # Create histogram: 0-max_distance meters, 100 bins
                hist, bin_edges = np.histogram(distances_array, bins=100, range=(0, max_distance))
                histograms[(manipulated_obj_id, other_obj_id)] = hist.astype(np.int32)
                
                if verbose:
                    print(f"Obj {manipulated_obj_id} -> Obj {other_obj_id} (NO RISK): {len(distances)} measurements, "
                        f"min={distances_array.min():.3f}m, max={distances_array.max():.3f}m, "
                        f"mean={distances_array.mean():.3f}m")
            else:
                # Convert to numpy array for easier binning
                distances_array = np.array(distances)
                
                # Create histogram: 0-max_distance meters, 100 bins
                hist, bin_edges = np.histogram(distances_array, bins=100, range=(0, max_distance))
                histograms[(manipulated_obj_id, other_obj_id)] = hist.astype(np.int32)
                
                if verbose:
                    print(f"Obj {manipulated_obj_id} -> Obj {other_obj_id}: {len(distances)} measurements, "
                        f"min={distances_array.min():.3f}m, max={distances_array.max():.3f}m, "
                        f"mean={distances_array.mean():.3f}m")
    
    if verbose:
        print(f"Generated {len(histograms)} histograms from raw distances")
    return histograms

def get_object_labels_from_cosine_similarity(video_frame_data: VideoFrameData1, white_avg_tensor, white_labels, frame_idx=0) -> dict[int, str]:
    """
    Get object labels by finding the highest cosine similarity between object DINO embeddings 
    and white list embeddings for each object in the first frame.
    
    Args:
        video_frame_data: VideoFrameData1 containing frame data with dino_embeddings
        white_avg_tensor: Tensor of white list DINO embeddings [N_white, C]
        white_labels: List of labels corresponding to white_avg_tensor indices
    
    Returns:
        Dict mapping obj_id -> label (or "unknown" if no good match)
    """
    assert len(white_avg_tensor) == len(white_labels), f"Mismatch: white_avg_tensor has {len(white_avg_tensor)} items but white_labels has {len(white_labels)} items"
    
    if len(video_frame_data.frames) == 0:
        return {}
    
    # Get the first frame's DINO embeddings
    frame_data = video_frame_data.frames[frame_idx]    
    dino_embeddings = frame_data["dino_embeddings"]  # [N_objs, C]
    assert dino_embeddings.shape[0] > 0 and white_avg_tensor.shape[0] > 0, "Both dino_embeddings and white_avg_tensor must be non-empty"
    
    # Compute cosine similarities
    # dino_embeddings: [N_objs, C], white_avg_tensor: [N_white, C]
    cosine_similarities = F.cosine_similarity(
        dino_embeddings.unsqueeze(1),  # [N_objs, 1, C]
        white_avg_tensor.unsqueeze(0),  # [1, N_white, C]
        dim=2
    )  # [N_objs, N_white]
    
    # Get the best match for each object
    max_similarities, best_indices = torch.max(cosine_similarities, dim=1)  # [N_objs]
    
    # Create mapping from obj_id to label
    obj_labels = {}
    for obj_id in range(dino_embeddings.shape[0]):
        best_idx = best_indices[obj_id].item()
        similarity = max_similarities[obj_id].item()
        
        # Only use label if similarity is above threshold (e.g., 0.3)
        if similarity > 0.3 and best_idx < len(white_labels):
            obj_labels[obj_id] = white_labels[best_idx]
        else:
            obj_labels[obj_id] = f"obj_{obj_id}"
    
    return obj_labels

def combine_histograms_by_labels(
        histograms: Dict[Tuple[int, int], np.ndarray],
        video_frame_data: VideoFrameData1,
        white_avg_tensor,
        white_labels,
        obj_id_mask = None,
        verbose=False
    ) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[int, str], Dict[str, List[int]]]:
    """
    Convert (manipulated_obj_id, other_obj_id) keyed histograms into
    (manipulated_label, other_label) keyed histograms by labeling objects
    via cosine similarity to a whitelist, then summing matching label pairs.

    Returns:
        combined_histograms: {(manipulated_label, other_label) -> np.ndarray (100,)}
        obj_labels: {obj_id -> label}
        label_to_obj_ids: {label -> [list of obj_ids]}
    """
    # Get 5 random frame indices for sampling
    num_frames = len(video_frame_data.frames)
    num_samples = 5
    if num_frames <= num_samples:
        random_frame_indices = list(range(num_frames))
    else:
        random_frame_indices = random.sample(range(num_frames), num_samples)

    obj_labels_agg = dict()
    for i, frame_idx in enumerate(random_frame_indices):
        obj_labels_tmp = get_object_labels_from_cosine_similarity(video_frame_data, white_avg_tensor, white_labels)
        for key in obj_labels_tmp:
            if key in obj_labels_agg:
                obj_labels_agg[key].append(obj_labels_tmp[key])
            else:
                obj_labels_agg[key] = [obj_labels_tmp[key]]
        
    # Debug: Print all entries of obj_labels_agg
    if verbose:
        print("obj_labels_agg contents:")
        for obj_id, label_list in obj_labels_agg.items():
            print(f"  obj_id {obj_id}: {label_list}")
    # Find majority vote for each object ID
    obj_labels = {}
    for obj_id, label_list in obj_labels_agg.items():
        # Count occurrences of each label
        label_counts = {}
        for label in label_list:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find the label with the highest count (majority vote)
        majority_label = max(label_counts, key=label_counts.get)
        obj_labels[obj_id] = majority_label

    # Create reverse mapping from labels to object IDs
    label_to_obj_ids: Dict[str, List[int]] = {}
    for obj_id, label in obj_labels.items():
        if not obj_id_mask[obj_id]:
            continue
        if label not in label_to_obj_ids:
            label_to_obj_ids[label] = []
        label_to_obj_ids[label].append(obj_id)

    combined_histograms: Dict[Tuple[str, str], np.ndarray] = {}
    for (manipulated_obj_id, other_obj_id), hist_array in histograms.items():
        manipulated_label = obj_labels.get(manipulated_obj_id, f"obj_{manipulated_obj_id}")
        other_label = obj_labels.get(other_obj_id, f"obj_{other_obj_id}")
        label_key = (manipulated_label, other_label)
        if label_key in combined_histograms:
            combined_histograms[label_key] += hist_array
        else:
            combined_histograms[label_key] = hist_array.copy()

    return combined_histograms, obj_labels, label_to_obj_ids

def get_dino_map(video_frame_data: VideoFrameData1, obj_labels: Dict[int, str]) -> Dict[str, List[torch.Tensor]]:
    """
    Create a mapping from string labels to all DINO embeddings for that label across all frames.
    
    Args:
        video_frame_data: Contains dino_embeddings in each frame
        obj_labels: {obj_id -> label} mapping from combine_histograms_by_labels
        
    Returns:
        {label -> [list of dino embeddings across all frames]}
    """
    dino_map = {}
    
    for frame_data in video_frame_data.frames:
        if "dino_embeddings" not in frame_data:
            continue
            
        dino_embeddings = frame_data["dino_embeddings"]  # [N_objs, C]
        
        for obj_id in range(dino_embeddings.shape[0]):
            if obj_id in obj_labels:
                label = obj_labels[obj_id]
                if label not in dino_map:
                    dino_map[label] = []
                dino_map[label].append(dino_embeddings[obj_id])
    
    return dino_map

def get_dino_map_by_obj_id(video_frame_data: VideoFrameData1) -> Dict[int, List[torch.Tensor]]:
    """
    Create a mapping from object IDs to all DINO embeddings for that object across all frames.
    
    Args:
        video_frame_data: Contains dino_embeddings in each frame
        
    Returns:
        {obj_id -> [list of dino embeddings across all frames]}
    """
    dino_map = {}
    
    for i, frame_data in enumerate(video_frame_data.frames):
        assert "dino_embeddings" in frame_data, "dino_embeddings not found in frame_data"            
        dino_embeddings = frame_data["dino_embeddings"]  # [N_objs, C]
        
        for obj_id in range(dino_embeddings.shape[0]):
            if obj_id not in dino_map:
                dino_map[obj_id] = []
            dino_map[obj_id].append(dino_embeddings[obj_id])
    
    return dino_map

def extract_embedding_pairs_for_histograms(
        combined_histograms: Dict[Tuple[str, str], np.ndarray],
        obj_labels: Dict[int, str],
        video_frame_data: VideoFrameData1
    ) -> Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract DINO embedding pairs for each histogram in combined_histograms.
    
    For each (manipulated_label, other_label) pair, find the corresponding
    object IDs and return their DINO embeddings from the first frame.
    
    Args:
        combined_histograms: {(manipulated_label, other_label) -> histogram array}
        obj_labels: {obj_id -> label} mapping from combine_histograms_by_labels
        video_frame_data: Contains dino_embeddings in first frame
        
    Returns:
        {(manipulated_label, other_label) -> (manip_embedding, other_embedding)}
    """
    assert len(video_frame_data.frames) > 0 and "dino_embeddings" in video_frame_data.frames[0], "video_frame_data must have frames with dino_embeddings"    
    dino_embeddings = video_frame_data.frames[0]["dino_embeddings"]  # [N_objs, C]
    
    # Create reverse mapping: label -> obj_id
    label_to_obj_id = {label: obj_id for obj_id, label in obj_labels.items()}
    
    embedding_pairs = {}
    for (manipulated_label, other_label) in combined_histograms.keys():
        # Find object IDs for these labels
        manip_obj_id = label_to_obj_id.get(manipulated_label)
        other_obj_id = label_to_obj_id.get(other_label)
        
        if (manip_obj_id is not None and other_obj_id is not None and
            manip_obj_id < dino_embeddings.shape[0] and other_obj_id < dino_embeddings.shape[0]):
            
            manip_embedding = dino_embeddings[manip_obj_id]  # [C]
            other_embedding = dino_embeddings[other_obj_id]  # [C]
            embedding_pairs[(manipulated_label, other_label)] = (manip_embedding, other_embedding)
        else:
            print(f"Warning: Could not find embeddings for pair ({manipulated_label}, {other_label})")
    
    return embedding_pairs

def plot_cdf_histograms_with_object_ids(histograms: Dict[Tuple[int, int], np.ndarray], 
                                       output_dir: str = ".", prefix: str = "cdf_histograms", sigma=1.0,
                                       verbose=False):
    """
    Plot CDF histograms with object IDs for visualization during dataset creation.
    
    Args:
        histograms: {(manipulated_obj_id, other_obj_id) -> histogram array (100,)}
        output_dir: Directory to save the plots (default: current directory)
        prefix: Prefix for the saved plot filename (default: "cdf_histograms")
        sigma: Standard deviation for Gaussian smoothing (default: 1.0)
    """
    if not histograms:
        print("No histograms to plot!")
        return
    
    # Create figure with subplots
    n_histograms = len(histograms)
    n_cols = int(np.ceil(np.sqrt(n_histograms)))
    n_rows = int(np.ceil(n_histograms / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle different cases for axes indexing
    if n_histograms == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each histogram pair
    for i, ((manipulated_obj_id, other_obj_id), hist_array) in enumerate(histograms.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Create distance bins (0 to 0.5 meters range with 100 bins)
        max_distance = 0.5
        bin_edges = np.linspace(0, max_distance, 101)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = max_distance / 100
        
        # Apply Gaussian smoothing and create CDF
        smoothed_hist = gaussian_filter1d(hist_array.astype(float), sigma=sigma)
        cdf = cdf_from_histogram(hist_array, sigma=sigma)
        
        # Plot original histogram (transparent bars)
        ax.bar(bin_centers, hist_array, width=bin_width, alpha=0.3, color='steelblue', label='Original Histogram')
        
        # Plot smoothed histogram
        ax.plot(bin_centers, smoothed_hist, 'r-', linewidth=2, label='Smoothed Histogram')
        
        # Plot CDF on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(bin_centers, cdf, 'g-', linewidth=2, label='CDF')
        ax2.set_ylabel('Cumulative Probability', color='g')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Add horizontal lines for CDF reference
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
        ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90%')
        
        ax.set_title(f'Obj {manipulated_obj_id} → Obj {other_obj_id}\nTotal: {hist_array.sum()}')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_distance)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # Add text annotation
        ax.text(0.02, 0.98, f'Manipulated: Obj {manipulated_obj_id}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Hide empty subplots
    for i in range(n_histograms, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    plt.suptitle('CDF Histograms: Object Pairs with Distance Distributions', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{prefix}_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    if verbose:
        print(f"Saved CDF histogram plot to {plot_path}")
        print(f"Plotted {n_histograms} object pair histograms")

def create_dataset_single_dino_by_obj_id(histograms: Dict[Tuple[int, int], np.ndarray], 
                                       dino_map: Dict[int, List[torch.Tensor]], sigma=1.0, 
                                       plot_cdfs: bool = True, output_dir: str = ".", 
                                       dino_per_obj: int = 1, ctrl_pts: int = 10, verbose=False) -> torch.Tensor:
    """
    Create a dataset tensor from object ID histograms and DINO features.
    
    Args:
        histograms: {(manipulated_obj_id, other_obj_id) -> histogram array (100,)}
        dino_map: {obj_id -> [list of dino embeddings]}
        sigma: Standard deviation for Gaussian smoothing (default: 1.0)
        plot_cdfs: Whether to plot CDF histograms for visualization (default: True)
        output_dir: Directory to save CDF plots (default: current directory)
        dino_per_obj: Number of DINO features to sample per object (default: 1)
        ctrl_pts: Number of Bezier control points (default: 10)
        
    Returns:
        dataset: [total_combinations * dino_per_obj^2, 2048 + ctrl_pts] tensor with layout:
                [featA(1024), featB(1024), control_points(ctrl_pts)]
    """
    # Count total combinations first
    total_combinations = len(histograms)
    if total_combinations == 0:
        print("No combinations found!")
        return torch.empty((0, 2048 + ctrl_pts), dtype=torch.float32)

    # Calculate total dataset size: each histogram pair generates dino_per_obj^2 combinations
    total_dataset_size = total_combinations * (dino_per_obj ** 2)

    # Plot CDF histograms for visualization if requested
    if plot_cdfs:
        if verbose:
            print("Plotting CDF histograms with object IDs...")
        plot_cdf_histograms_with_object_ids(histograms, output_dir=output_dir, sigma=sigma)

    # Pre-allocate tensor
    dataset = torch.empty((total_dataset_size, 2048 + ctrl_pts), dtype=torch.float32)

    # Fill tensor efficiently
    idx = 0
    for (manipulated_obj_id, other_obj_id), histogram in histograms.items():
        manip_features = dino_map[manipulated_obj_id]
        other_features = dino_map[other_obj_id]

        # Randomly sample dino_per_obj features from each list
        #print(f"len manip_features {len(manip_features)}")
        #print(f"len other_features {len(other_features)}")
        sampled_manip_features = random.choices(manip_features, k=min(dino_per_obj, len(manip_features)))
        sampled_other_features = random.choices(other_features, k=min(dino_per_obj, len(other_features)))
        
        cdf = cdf_from_histogram(histogram, sigma=sigma)
        
        # Convert histogram to Bezier control points using the same method as collate_fn
        normalized_hist = cdf.copy()
        if normalized_hist.max() > 1.0 or normalized_hist.min() < 0.0:
            normalized_hist = (normalized_hist - normalized_hist.min()) / (normalized_hist.max() - normalized_hist.min() + 1e-8)
        
        coeffs = HistogramDataset.fit_bezier_monotone(y=normalized_hist, num_ctrl=ctrl_pts, clamp01=True)
        control_points = torch.from_numpy(coeffs).float()
        
        # Create all combinations of sampled features
        for manip_feat in sampled_manip_features:
            for other_feat in sampled_other_features:
                # Fill pre-allocated tensor directly
                dataset[idx, :1024] = manip_feat.cpu()
                dataset[idx, 1024:2048] = other_feat.cpu()
                dataset[idx, 2048:] = control_points
                idx += 1

    if verbose:
        print(f"Created dataset with {dataset.shape[0]} samples of shape {dataset.shape[1]}")
    return dataset

def generate_final_dataset(histograms: dict, video_frame_data: VideoFrameData1, output_dir: str = "histograms", 
                            dino_per_obj=15, ctrl_pts=10, verbose=False):
    """
    Generate dataset that we're going to train on 
    
    Args:
        histograms: Dict mapping (manipulated_obj_id, other_obj_id) -> histogram array
        video_frame_data: VideoFrameData1 containing frame data with dino_embeddings
        output_dir: Directory to save histogram files (default: "histograms")
        dino_per_obj: Number of DINO features to sample per object (default: 15)
        ctrl_pts: Number of Bezier control points (default: 10)
        verbose: Whether to print verbose output
    """

    assert histograms, "No histograms to plot!"

    dino_map = get_dino_map_by_obj_id(video_frame_data)
    dataset = create_dataset_single_dino_by_obj_id(histograms, dino_map, output_dir=output_dir, 
                                            dino_per_obj=dino_per_obj, ctrl_pts=ctrl_pts, verbose=verbose)

    if verbose:
        print(f"Dataset size: {dataset.shape[0]} samples")
        print(f"Dataset shape: {dataset.shape} (includes {ctrl_pts} control points)")
    
    # Save the dataset tensor
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    tensor_path = os.path.join(output_dir, f"histogram_tensor_{timestamp}.pt")
    torch.save(dataset, tensor_path)
    print(f"Saved dataset tensor to {tensor_path}")

    return tensor_path

def cdf_from_histogram(hist: np.ndarray, sigma=1.0) -> np.ndarray:
    """   
    Args:
        hist: Histogram array of shape (n_bins,)
        
    Returns:
        CDF array of shape (n_bins,) with values in [0, 1]
    """

    smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=sigma)
    hist_sum = np.sum(smoothed_hist)
    if hist_sum > 0:
        prob_density = smoothed_hist / hist_sum
    else:
        prob_density = smoothed_hist
    
    cdf = np.cumsum(prob_density)
    
    return cdf

def plot_combined_histograms(combined_histograms: dict, label_to_obj_ids: dict, 
                           max_distance: float, histograms_dir: str, 
                           title_suffix: str = "", filename_suffix: str = "", 
                           verbose=False):
    """
    Plot combined histograms in a single figure with subplots.
    
    Args:
        combined_histograms: Dict mapping (manipulated_label, other_label) -> histogram array
        label_to_obj_ids: Dict mapping label -> list of object IDs
        max_distance: Maximum distance for histogram range
        histograms_dir: Directory to save the plot
        title_suffix: Suffix to add to the plot title
        filename_suffix: Suffix to add to the filename
    """
    if not combined_histograms:
        print(f"No histograms to plot for {title_suffix}")
        return
    
    # Create figure with subplots
    n_histograms = len(combined_histograms)
    n_cols = int(np.ceil(np.sqrt(n_histograms)))
    n_rows = int(np.ceil(n_histograms / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle different cases for axes indexing
    if n_histograms == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each combined histogram
    for i, ((manipulated_label, other_label), hist_array) in enumerate(combined_histograms.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get object IDs for the labels
        manipulated_obj_ids = label_to_obj_ids.get(manipulated_label, [])
        other_obj_ids = label_to_obj_ids.get(other_label, [])
        
        # Print object IDs for this histogram
        if verbose:
            print(f"Histogram {i+1}: {manipulated_label} → {other_label}")
            print(f"  Manipulated object IDs: {manipulated_obj_ids}")
            print(f"  Reference object IDs: {other_obj_ids}")
            print(f"  Total interactions: {hist_array.sum()}")
            print()
        
        # Create distance bins (0-max_distance meters, 100 bins)
        bin_edges = np.linspace(0, max_distance, 101)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = max_distance / 100
        
        # Plot histogram
        ax.bar(bin_centers, hist_array, width=bin_width, alpha=0.7, color='steelblue')
        ax.set_title(f'Manipulated {manipulated_label} → {other_label}\nTotal: {hist_array.sum()}')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits
        ax.set_xlim(0, max_distance)
        
        # Add text annotation to clarify which is manipulated with object IDs
        manipulated_ids_str = f"IDs: {manipulated_obj_ids}" if manipulated_obj_ids else "No IDs"
        other_ids_str = f"IDs: {other_obj_ids}" if other_obj_ids else "No IDs"
        ax.text(0.02, 0.98, f'Manipulated: {manipulated_label}\n{manipulated_ids_str}\nOther: {other_label}\n{other_ids_str}', 
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Hide empty subplots
    for i in range(n_histograms, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    plt.suptitle(f'Distance Histograms: Manipulated Objects vs Other Objects{title_suffix}', fontsize=16)
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(histograms_dir, f"all_histograms{filename_suffix}.png")
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free memory
    
    if verbose:
        print(f"Saved combined histogram plot to {combined_plot_path}")
    return combined_plot_path

def plot_individual_histograms(combined_histograms: dict, label_to_obj_ids: dict, 
                             max_distance: float, histograms_dir: str, 
                             filename_suffix: str = "", verbose=False):
    """
    Plot individual histogram files with 4 subplots each (original, smoothed, CDF, CCDF).
    
    Args:
        combined_histograms: Dict mapping (manipulated_label, other_label) -> histogram array
        label_to_obj_ids: Dict mapping label -> list of object IDs
        max_distance: Maximum distance for histogram range
        histograms_dir: Directory to save the plots
        filename_suffix: Suffix to add to the filenames
    """
    if not combined_histograms:
        print(f"No individual histograms to plot for {filename_suffix}")
        return
    
    # Save individual histogram files and plots
    for (manipulated_label, other_label), hist_array in combined_histograms.items():
        # Get object IDs for the labels
        manipulated_obj_ids = label_to_obj_ids.get(manipulated_label, [])
        other_obj_ids = label_to_obj_ids.get(other_label, [])
        
        # Create filename with labels (sanitize for filesystem)
        manipulated_safe = "".join(c for c in manipulated_label if c.isalnum() or c in (' ', '-', '_')).rstrip()
        other_safe = "".join(c for c in other_label if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # Create distance bins (0-max_distance meters, 100 bins)
        bin_edges = np.linspace(0, max_distance, 101)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = max_distance / 100

        # Apply Gaussian smoothing to the histogram
        sigma = 1.0  # Standard deviation for Gaussian kernel
        smoothed_hist = gaussian_filter1d(hist_array.astype(float), sigma=sigma)
        
        # Calculate CDF from smoothed histogram using the new function
        cdf = cdf_from_histogram(hist_array, sigma=sigma)
        
        # Calculate CCDF (Complementary CDF) = 1 - CDF
        ccdf = 1.0 - cdf
        
        # Create individual plot with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Original Histogram
        ax1.bar(bin_centers, hist_array, width=bin_width, alpha=0.7, color='steelblue')
        ax1.set_title(f'Original Histogram: {manipulated_label} → {other_label}\nTotal Interactions: {hist_array.sum()}')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max_distance)
        
        # 2. Smoothed Histogram
        ax2.plot(bin_centers, smoothed_hist, 'r-', linewidth=2, label='Smoothed')
        ax2.bar(bin_centers, hist_array, width=bin_width, alpha=0.3, color='steelblue', label='Original')
        ax2.set_title(f'Smoothed Histogram (σ={sigma}): {manipulated_label} → {other_label}')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max_distance)
        ax2.legend()
        
        # 3. CDF (Cumulative Distribution Function)
        ax3.plot(bin_centers, cdf, 'g-', linewidth=2)
        ax3.set_title(f'CDF: {manipulated_label} → {other_label}')
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max_distance)
        ax3.set_ylim(0, 1)
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
        ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90%')
        ax3.legend()
        
        # 4. CCDF (Complementary CDF) - Risk Field
        ax4.plot(bin_centers, ccdf, 'purple', linewidth=2)
        ax4.set_title(f'CCDF Risk Field: {manipulated_label} → {other_label}')
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Risk Probability (1 - CDF)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, max_distance)
        ax4.set_ylim(0, 1)
        ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='10% Risk')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Risk')
        ax4.legend()
        
        # Add overall title
        fig.suptitle(f'Distance Analysis: {manipulated_label} → {other_label}', fontsize=16)
        
        # Add text annotation to the first subplot with object IDs
        manipulated_ids_str = f"IDs: {manipulated_obj_ids}" if manipulated_obj_ids else "No IDs"
        other_ids_str = f"IDs: {other_obj_ids}" if other_obj_ids else "No IDs"
        ax1.text(0.02, 0.98, f'Manipulated Object: {manipulated_label}\n{manipulated_ids_str}\nOther Object: {other_label}\n{other_ids_str}', 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = f"histogram_{manipulated_safe}_to_{other_safe}{filename_suffix}.png"
        plot_filepath = os.path.join(histograms_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    if verbose:
        print(f"Saved {len(combined_histograms)} individual histogram plots (.png) to {histograms_dir}/")

def plot_histograms(histograms: dict, video_frame_data: VideoFrameData1, 
                    white_avg_tensor, white_labels, grey_avg_tensor, grey_labels, 
                    max_distance: float = 0.5, output_dir: str = "histograms", verbose=False):
    """
    Plot histograms of distances between manipulated objects and other objects.
    
    Args:
        histograms: Dict mapping (manipulated_obj_id, other_obj_id) -> histogram array
        video_frame_data: VideoFrameData1 containing frame data with dino_embeddings
        white_avg_tensor: Tensor of white list DINO embeddings
        white_labels: List of labels corresponding to white_avg_tensor indices
        grey_avg_tensor: Tensor of white list DINO embeddings
        grey_labels: List of labels corresponding to white_avg_tensor indices
        max_distance: Maximum distance for histogram range (default: 0.5 meters)
        output_dir: Directory to save histogram files (default: "histograms")
    """

    assert len(white_avg_tensor) == len(white_labels), f"Mismatch: white_avg_tensor has {len(white_avg_tensor)} items but white_labels has {len(white_labels)} items"
    assert len(grey_avg_tensor) == len(grey_labels), f"Mismatch: grey_avg_tensor has {len(grey_avg_tensor)} items but grey_labels has {len(grey_labels)} items"
    os.makedirs(output_dir, exist_ok=True)

    # Create combined tensor of white and grey embeddings for histogram processing
    in_hist_tensor = torch.cat([white_avg_tensor, grey_avg_tensor], dim=0)
    in_hist_labels = white_labels + grey_labels

    # Filter histograms into two categories based on whether the other object is in white_list or grey_list
    histograms_in_dataset = {}
    histograms_no_risk = {}
    in_dataset_mask = torch.zeros(video_frame_data.frames[0]["sam_masks"].shape[0], dtype=torch.bool)
    no_risk_mask    = torch.zeros(video_frame_data.frames[0]["sam_masks"].shape[0], dtype=torch.bool)
    mask_indices = set([i for i in range(video_frame_data.frames[0]["sam_masks"].shape[0])])
    
    for (manipulated_obj_id, other_obj_id), hist_array in histograms.items():
        # Check if this histogram contains special -1.0 distances (indicating no_risk objects)
        # We can detect this by checking if all distances were 0 (which happens when we set -1.0 distances to 0)
        if np.all(hist_array[1:] == 0) and hist_array[0] > 0:
            # This is a no_risk histogram (all distances were binned to the first bin)
            if verbose:
                print(f" {manipulated_obj_id}, {other_obj_id}  -> Classified as NO_RISK")
            histograms_no_risk[(manipulated_obj_id, other_obj_id)] = hist_array
            no_risk_mask[other_obj_id] = True
            no_risk_mask[manipulated_obj_id] = True

        else:
            # This is a regular in_dataset histogram
            if verbose:
                print(f" {manipulated_obj_id}, {other_obj_id} -> Classified as IN_DATASET")
            histograms_in_dataset[(manipulated_obj_id, other_obj_id)] = hist_array
            in_dataset_mask[other_obj_id] = True 
            in_dataset_mask[manipulated_obj_id] = True

    if verbose:
        print(f"Filtered histograms: {len(histograms_in_dataset)} in_dataset, {len(histograms_no_risk)} no_risk")
    
    # Debug: Print some details about the filtered histograms
    if verbose:
        if histograms_in_dataset:
            print(f"In-dataset histogram keys (first 5): {list(histograms_in_dataset.keys())[:5]}")
        if histograms_no_risk:
            print(f"No-risk histogram keys (first 5): {list(histograms_no_risk.keys())[:5]}")

    combined_histograms_in_dataset, obj_labels_in_dataset, label_to_obj_ids_in_dataset = combine_histograms_by_labels(histograms_in_dataset, video_frame_data, in_hist_tensor, in_hist_labels, in_dataset_mask)
    combined_histograms_no_risk, obj_labels_no_risk, label_to_obj_ids_no_risk = combine_histograms_by_labels(histograms_no_risk, video_frame_data, in_hist_tensor, in_hist_labels, no_risk_mask)
    
    # Debug: Print results after combination
    if verbose:
        print(f"After combination: {len(combined_histograms_in_dataset)} in_dataset, {len(combined_histograms_no_risk)} no_risk")
        if combined_histograms_in_dataset:
            print(f"In-dataset combined keys: {list(combined_histograms_in_dataset.keys())}")
        if combined_histograms_no_risk:
            print(f"No-risk combined keys: {list(combined_histograms_no_risk.keys())}")
    
    # Process in_dataset histograms
    if verbose:
        print("\n=== Processing In-Dataset Histograms ===")
        print(f"About to plot {len(combined_histograms_in_dataset)} in-dataset histograms")
    plot_combined_histograms(combined_histograms_in_dataset, label_to_obj_ids_in_dataset, 
                           max_distance, output_dir, 
                           title_suffix=" (In-Dataset)", filename_suffix="_in_dataset")
    plot_individual_histograms(combined_histograms_in_dataset, label_to_obj_ids_in_dataset, 
                             max_distance, output_dir, filename_suffix="_in_dataset")
    
    # Process no_risk histograms
    if verbose:
        print("\n=== Processing No-Risk Histograms ===")
        print(f"About to plot {len(combined_histograms_no_risk)} no-risk histograms")
    plot_combined_histograms(combined_histograms_no_risk, label_to_obj_ids_no_risk, 
                           max_distance, output_dir, 
                           title_suffix=" (No-Risk)", filename_suffix="_no_risk")
    plot_individual_histograms(combined_histograms_no_risk, label_to_obj_ids_no_risk, 
                             max_distance, output_dir, filename_suffix="_no_risk")
    
    total_histograms = len(combined_histograms_in_dataset) + len(combined_histograms_no_risk)

    print(f"\nTotal histograms plotted: {total_histograms}")
    print(f"  - In-dataset: {len(combined_histograms_in_dataset)}")
    print(f"  - No-risk: {len(combined_histograms_no_risk)}")

def plot_hist(batch_targets_hist: torch.Tensor, batch_control_points: torch.Tensor = None, 
              output_dir: str = ".", prefix: str = "histogram", num_ctrl: int = 10):
    """
    Plot histogram data from batch targets and save to local directory.
    Optionally plot control points on top of the CDF.
    
    Args:
        batch_targets_hist: Tensor of shape (batch_size, hist_size) containing histogram data
        batch_control_points: Tensor of shape (batch_size, num_ctrl) containing Bezier control points (optional)
        output_dir: Directory to save the plot (default: current directory)
        prefix: Prefix for the saved plot filename (default: "histogram")
        num_ctrl: Number of control points (used if batch_control_points is None)
    """
    # Convert to numpy for plotting
    hist_data = batch_targets_hist.detach().cpu().numpy()
    
    # Create figure with subplots for each sample in the batch
    batch_size, hist_size = hist_data.shape
    n_cols = min(4, batch_size)  # Max 4 columns
    n_rows = int(np.ceil(batch_size / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle different cases for axes indexing
    if batch_size == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create bin centers (assuming 0 to 0.5 meters range with hist_size bins)
    max_distance = 0.5
    bin_edges = np.linspace(0, max_distance, hist_size + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = max_distance / hist_size
    
    # Plot each histogram in the batch
    for i in range(batch_size):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot histogram
        ax.bar(bin_centers, hist_data[i], width=bin_width, alpha=0.7, color='steelblue', label='Histogram')
        
        # Plot control points if provided
        if batch_control_points is not None:
            control_points = batch_control_points[i].detach().cpu().numpy()
            # Map control points to distance coordinates
            t_control = np.linspace(0, 1, len(control_points))
            x_control = t_control * max_distance
            
            # Create secondary y-axis for control points and Bezier curve
            ax2 = ax.twinx()
            ax2.scatter(x_control, control_points, color='orange', s=50, zorder=5, label='Control Points')
            
            # Plot Bezier curve from control points
            t_curve = np.linspace(0, 1, hist_size)
            B = HistogramDataset.bernstein_basis(len(control_points), t_curve)
            bezier_curve = B @ control_points
            x_curve = t_curve * max_distance
            ax2.plot(x_curve, bezier_curve, 'g--', linewidth=1.5, alpha=0.8, label='Bezier Curve')
            
            ax2.set_ylabel('Control Points / Bezier Curve', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim(0, 1)
        
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Count', color='steelblue')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_distance)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if batch_control_points is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        else:
            ax.legend(lines1, labels1, loc='upper right', fontsize=8)
    
    # Hide empty subplots
    for i in range(batch_size, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    plt.suptitle(f'Histogram Batch with Control Points (Size: {batch_size})', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{prefix}_batch_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Saved histogram plot to {plot_path}")

def test_histogram_dataset(tensor_paths: list[str], batch_size: int = 4, num_ctrl: int = 10, output_dir: str = ".", verbose: bool = False):
    """
    Test function for HistogramDataset to verify functionality and data integrity.
    
    Args:
        tensor_paths: List of paths to the .pt files containing the dataset tensors
        batch_size: Batch size for testing DataLoader
        num_ctrl: Number of Bezier control points for testing collate function
        output_dir: Directory to save test plots (default: current directory)
        verbose: Whether to print detailed success information (failures always printed)
    
    Returns:
        dict: Test results including dataset info, sample data, and validation results
    """
    if verbose:
        print("=" * 60)
        print("TESTING HistogramDataset")
        print("=" * 60)
    
    results = {}
    
    try:
        # Test 1: Dataset initialization
        if verbose:
            print("\n1. Testing dataset initialization...")
        dataset = HistogramDataset(tensor_paths, ctrl_pts=num_ctrl)
        results['dataset_loaded'] = True
        results['dataset_size'] = len(dataset)
        results['input_dim'], results['target_dim'] = dataset.get_feature_dims()
        
        if verbose:
            print(f"✓ Dataset loaded successfully")
            print(f"  - Dataset size: {len(dataset)} samples")
            print(f"  - Input dimension: {results['input_dim']}")
            print(f"  - Target dimension: {results['target_dim']}")
        
        # Test 2: Single sample access
        if verbose:
            print("\n2. Testing single sample access...")
        if len(dataset) > 0:
            sample_input, sample_target = dataset[0]
            results['sample_input_shape'] = sample_input.shape
            results['sample_target_shape'] = sample_target.shape
            
            if verbose:
                print(f"✓ Sample access successful")
                print(f"  - Input shape: {sample_input.shape}")
                print(f"  - Target shape: {sample_target.shape}")
                print(f"  - Input range: [{sample_input.min():.4f}, {sample_input.max():.4f}]")
                print(f"  - Target range: [{sample_target.min():.4f}, {sample_target.max():.4f}]")
            
            # Validate tensor format
            if sample_input.shape[0] != 2048:
                raise ValueError(f"Expected input size 2048, got {sample_input.shape[0]}")
            if sample_target.shape[0] != num_ctrl:
                raise ValueError(f"Expected target size {num_ctrl}, got {sample_target.shape[0]}")
            
            results['tensor_format_valid'] = True
            if verbose:
                print("✓ Tensor format validation passed")
        else:
            print("⚠ Dataset is empty")
            results['sample_input_shape'] = None
            results['sample_target_shape'] = None
            results['tensor_format_valid'] = False
        
        # Test 3: DataLoader with custom collate function
        if verbose:
            print("\n3. Testing DataLoader with custom collate function...")
        if len(dataset) > 0:
            # Create a small batch for testing
            test_batch_size = min(batch_size, len(dataset))
            dataloader = dataset.get_dataloader(
                batch_size=test_batch_size, 
                shuffle=False
            )
            
            # Get one batch
            batch_inputs, batch_targets = next(iter(dataloader))
            
            results['batch_loaded'] = True
            results['batch_size'] = test_batch_size
            results['dino1_shape'] = batch_inputs['dino1'].shape
            results['dino2_shape'] = batch_inputs['dino2'].shape
            results['control_points_shape'] = batch_targets['control_points'].shape
            
            if verbose:
                print(f"✓ DataLoader test successful")
                print(f"  - Batch size: {test_batch_size}")
                print(f"  - DINO1 shape: {batch_inputs['dino1'].shape}")
                print(f"  - DINO2 shape: {batch_inputs['dino2'].shape}")
                print(f"  - Control points shape: {batch_targets['control_points'].shape}")
            
            # Validate shapes
            expected_dino_shape = (test_batch_size, 1024)
            expected_ctrl_shape = (test_batch_size, num_ctrl)
            
            if batch_inputs['dino1'].shape != expected_dino_shape:
                raise ValueError(f"Expected dino1 shape {expected_dino_shape}, got {batch_inputs['dino1'].shape}")
            if batch_inputs['dino2'].shape != expected_dino_shape:
                raise ValueError(f"Expected dino2 shape {expected_dino_shape}, got {batch_inputs['dino2'].shape}")
            if batch_targets['control_points'].shape != expected_ctrl_shape:
                raise ValueError(f"Expected control_points shape {expected_ctrl_shape}, got {batch_targets['control_points'].shape}")
            
            results['batch_shapes_valid'] = True
            if verbose:
                print("✓ Batch shape validation passed")
            
            # Test data consistency
            if verbose:
                print("\n4. Testing data consistency...")
            # Check that dino1 + dino2 equals original input
            reconstructed_input = torch.cat([batch_inputs['dino1'], batch_inputs['dino2']], dim=1)
            original_input = torch.stack([dataset[i][0] for i in range(test_batch_size)], dim=0)
            
            if torch.allclose(reconstructed_input, original_input):
                results['data_consistency'] = True
                if verbose:
                    print("✓ Data consistency check passed")
            else:
                results['data_consistency'] = False
                print("⚠ Data consistency check failed")

            # Test Bezier control points validity
            if verbose:
                print("\n6. Testing Bezier control points...")
            ctrl_points = batch_targets['control_points']
            valid_ctrl_points = True
            
            for i in range(test_batch_size):
                ctrl = ctrl_points[i]
                
                # Check for NaN or Inf values first
                if torch.isnan(ctrl).any() or torch.isinf(ctrl).any():
                    print(f"⚠ Control points for sample {i} contain NaN or Inf values")
                    valid_ctrl_points = False
                    continue
                
                # Check if control points are in valid range [0, 1] with small tolerance for numerical errors
                tolerance = 1e-4  # Slightly more lenient tolerance
                min_val, max_val = ctrl.min().item(), ctrl.max().item()
                
                if min_val < -tolerance or max_val > 1 + tolerance:
                    print(f"⚠ Control points for sample {i} out of range [0,1]: [{min_val:.6f}, {max_val:.6f}]")
                    # Check if it's just a small numerical error that can be clamped
                    if min_val >= -1e-3 and max_val <= 1 + 1e-3:
                        print(f"  → Small numerical error detected, values can be clamped to [0,1]")
                        # Clamp the values to [0, 1] range
                        ctrl_clamped = torch.clamp(ctrl, 0.0, 1.0)
                        print(f"  → Clamped range: [{ctrl_clamped.min():.6f}, {ctrl_clamped.max():.6f}]")
                    else:
                        valid_ctrl_points = False
                else:
                    if verbose:
                        print(f"✓ Control points for sample {i} in valid range: [{min_val:.6f}, {max_val:.6f}]")
            
            results['bezier_valid'] = valid_ctrl_points
            if valid_ctrl_points:
                if verbose:
                    print("✓ Bezier control points validation passed")
            else:
                print("⚠ Bezier control points validation failed")
                
        else:
            print("⚠ Cannot test DataLoader - dataset is empty")
            results['batch_loaded'] = False
        
        # Test 7: Edge cases
        if verbose:
            print("\n7. Testing edge cases...")
        edge_cases = {}
        
        # Test with different indices
        if len(dataset) > 1:
            try:
                last_sample = dataset[len(dataset) - 1]
                edge_cases['last_sample'] = True
                if verbose:
                    print("✓ Last sample access successful")
            except Exception as e:
                edge_cases['last_sample'] = False
                print(f"⚠ Last sample access failed: {e}")
        
        # Test random sampling
        if len(dataset) > 0:
            try:
                random_idx = random.randint(0, len(dataset) - 1)
                random_sample = dataset[random_idx]
                edge_cases['random_sample'] = True
                if verbose:
                    print("✓ Random sample access successful")
            except Exception as e:
                edge_cases['random_sample'] = False
                print(f"⚠ Random sample access failed: {e}")
        
        results['edge_cases'] = edge_cases
        
        # Test 8: Plot control points visualization
        if verbose:
            print("\n8. Plotting control points visualization...")
        if len(dataset) > 0:
            try:
                plot_path = plot_control_points_from_dataset(
                    dataset, 
                    num_samples=min(5, len(dataset)), 
                    num_ctrl=num_ctrl, 
                    output_dir=output_dir, 
                    prefix="test_control_points"
                )
                results['control_points_plot'] = plot_path
                if verbose:
                    print("✓ Control points visualization created")
            except Exception as e:
                results['control_points_plot'] = None
                print(f"⚠ Control points plotting failed: {e}")
        else:
            results['control_points_plot'] = None
            if verbose:
                print("⚠ Cannot plot control points - dataset is empty")
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        all_tests_passed = all([
            results.get('dataset_loaded', False),
            results.get('tensor_format_valid', False),
            results.get('batch_loaded', False),
            results.get('batch_shapes_valid', False),
            results.get('data_consistency', False),
            results.get('bezier_valid', False)
        ])
        
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
            failed_tests = [k for k, v in results.items() if isinstance(v, bool) and not v]
            print(f"Failed tests: {failed_tests}")
        
        results['all_tests_passed'] = all_tests_passed
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        results['error'] = str(e)
        results['all_tests_passed'] = False
    
    return results

def plot_control_points_from_dataset(dataset, num_samples: int = 5, num_ctrl: int = 10, 
                                   output_dir: str = ".", prefix: str = "control_points"):
    """
    Plot control points directly from dataset samples with different shapes and colors for each control point.
    
    Args:
        dataset: HistogramDataset instance
        num_samples: Number of samples to plot (default: 5)
        num_ctrl: Number of control points (default: 10)
        output_dir: Directory to save the plot (default: current directory)
        prefix: Prefix for the saved plot filename (default: "control_points")
    """
    
    if len(dataset) == 0:
        print("⚠ Dataset is empty, cannot plot control points")
        return
    
    # Limit number of samples to dataset size
    num_samples = min(num_samples, len(dataset))
    
    # Define colors and shapes for each control point
    colors = plt.cm.tab10(np.linspace(0, 1, num_ctrl))
    shapes = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', 'X', 'P', '8', 'd']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot control points for each sample
    for sample_idx in range(num_samples):
        sample_input, sample_target = dataset[sample_idx]
        control_points = sample_target.detach().cpu().numpy()
        
        # Create x-coordinates (0 to 1 for control points)
        x_coords = np.linspace(0, 1, len(control_points))
        
        # Plot each control point with different shape and color
        for ctrl_idx, (x, y) in enumerate(zip(x_coords, control_points)):
            color = colors[ctrl_idx]
            shape = shapes[ctrl_idx % len(shapes)]
            
            ax.scatter(x, y, c=[color], marker=shape, s=100, 
                      label=f'Ctrl {ctrl_idx}' if sample_idx == 0 else "", 
                      alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Control Point Index (Normalized)', fontsize=12)
    ax.set_ylabel('Control Point Value', fontsize=12)
    ax.set_title(f'Control Points from Dataset Samples (n={num_samples})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add text box with sample info
    textstr = f'Samples: {num_samples}\nControl Points: {num_ctrl}\nDataset Size: {len(dataset)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{prefix}_dataset_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Control points plot saved: {plot_path}")
    return plot_path
