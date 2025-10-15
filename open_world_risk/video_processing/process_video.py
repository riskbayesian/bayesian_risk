# Standard library imports
import enum
import gc
import os
import pickle
import random
import shutil
from tabnanny import verbose
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import glob

# Third-party imports
import cv2
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from open_world_risk.src.api import call_hoist_former_manipulation_detection
from open_world_risk.src.compute_obj_part_feature import FeatureExtractor
from open_world_risk.src.point_cloud_3D import PointCloudGenerator
from open_world_risk.video_processing.process_video_utils import (
    data_loader,
    data_loader1,
    visualize_frame_data,
    visualize_frames_video,
    run_test_hoist_former_service_availability,
    create_straight_line_representation,
    point_to_line_distance,
    unit_direction_vector,
    load_process_video_config,
    save_frame_npys,
    sample_points_along_line,
    are_frames_in_same_manipulation_period,
    is_frame_in_any_manipulation_period,
    get_image_embedd_white_black_lst,
    get_manipulation_periods,
    is_flat_plane,
    save_sam_masks_as_images,
    get_percentage_overlap,
)
from open_world_risk.src.video_datastructures import (
    VideoFrameData1,
    VideoMetadata,
    VideoManipulationData,
    ManipulationTimeline,
    StraightLine,
    getEmptyFrameData,
)

from open_world_risk.src.obj_track_sam import (
    track_from_bbs,
    save_video_visualizations,
    create_video_from_vis,
)
from open_world_risk.src.dino_features import DinoFeatures

def run_manipulation_detection(rgb_frames, frame_indices, verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manip_scores = np.zeros((len(frame_indices), 1))
    manip_labels = np.zeros((len(frame_indices), 1))
    manip_masks = np.zeros((len(frame_indices), 1, rgb_frames[0].shape[0], rgb_frames[0].shape[1]))

    try:
        # Get the frames to process
        frames_to_process = rgb_frames[frame_indices]
        
        # first element of every returned value, is the frame number passed in
        manipulation_scores, manipulation_labels, manipulation_masks = call_hoist_former_manipulation_detection(
            rgb_frames=frames_to_process,
            save=verbose,
            batch_size_processing=3
        )
        print(f"Manipulation detection completed for {len(manipulation_scores)} frames")

        # Convert to list if it's a 2D numpy array
        if isinstance(manipulation_scores, np.ndarray) and manipulation_scores.dtype != object:
            manipulation_scores = [row for row in manipulation_scores]
            manipulation_labels = [row for row in manipulation_labels]
            manipulation_masks = [row for row in manipulation_masks]
        
        for i, frame_idx in enumerate(frame_indices):
            if i < len(manipulation_scores):
                scores = manipulation_scores[i]
                labels = manipulation_labels[i]
                masks = manipulation_masks[i]
                
                # Filter to keep only the highest confidence manipulation mask
                if len(scores) > 1:
                    # Find the index of the highest confidence score
                    highest_idx = np.argmax(scores)
                    
                    # Keep only the highest confidence mask
                    filtered_scores = scores[highest_idx]
                    filtered_labels = labels[highest_idx]
                    filtered_mask = masks[highest_idx]
                    assert filtered_mask.ndim == 2
                    manip_scores[i] = filtered_scores
                    manip_labels[i] = filtered_labels
                    manip_masks[i] = filtered_mask
                elif len(scores) == 1:
                    mask_squeezed = np.squeeze(masks, axis=0)
                    manip_scores[i] = scores[0]
                    manip_labels[i] = labels[0]
                    manip_masks[i] = mask_squeezed
                else:
                    pass
            else:
                assert False, f"Expected 1 manipulation mask per frame, got {len(scores)} for frame {frame_idx}"
        
        # Count how many frames had multiple masks filtered
        total_masks_before = sum(len(scores) for scores in manipulation_scores if len(scores) > 0)
        total_masks_after = sum(len(scores) for scores in manip_scores if len(scores) > 0)
        
        print(f"Mapped manipulation results to {len(frame_indices)} frame indices")
        print(f"Filtered manipulation masks: {total_masks_before} → {total_masks_after} (kept highest confidence per frame)")
        
    except Exception as e:
        assert False, f"Error during manipulation detection: {str(e)}"

    manip_scores = torch.from_numpy(manip_scores)
    manip_labels = torch.from_numpy(manip_labels)
    manip_masks = torch.from_numpy(manip_masks).bool().to_sparse()

    return manip_scores, manip_labels, manip_masks

def run_CLIP_SAM_segmentation(rgb_frames, frame_indices, feature_extractor, verbose: bool = False, batch_size=64, min_mask_size=1_000) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    sam_masks_lst: list[torch.tensor] = [None for i in range(len(frame_indices))]
    bbox_lst: list[torch.tensor] = [None for i in range(len(frame_indices))]

    num_batches = (len(frame_indices) + batch_size - 1) // batch_size    
    print(f"Processing {len(frame_indices)} frames in {num_batches} batches of {batch_size}")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(frame_indices))
        batch_frame_indices = frame_indices[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (frames {batch_start}-{batch_end-1})")
        for i, frame_idx in enumerate(tqdm(batch_frame_indices, desc=f"Batch {batch_idx + 1}", unit="frame", 
                                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
            
            rgb_image = rgb_frames[frame_idx]
            if rgb_image[0, 0, 0] > rgb_image[0, 0, 2]:  # Check if BGR
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            results = feature_extractor.process_single_image(image=rgb_image, verbose=verbose)

            # Preliminary filtering 
            masks = results.sam_masks.bool()
            input_boxes = results.input_boxes
            assert masks.shape[0] != 0, "No masks returned, empty"
            assert masks.shape[0] == input_boxes.shape[0], f"Warning: Mismatch masks ({masks.shape[0]}) & boxes ({input_boxes.shape[0]})"
            
            flat = masks.view(masks.shape[0], -1)
            valid = flat.any(dim=1)
            assert valid.any(), "No valid masks!"

            masks = masks[valid]
            input_boxes = input_boxes[valid]
            areas = masks.view(masks.shape[0], -1).sum(dim=1)

            # Filter out masks smaller than min_mask_size
            size_filter = areas >= min_mask_size
            assert size_filter.any(), f"No masks over {min_mask_size}"
            
            masks = masks[size_filter]
            input_boxes = input_boxes[size_filter]
            areas = areas[size_filter]

            sam_masks_lst[batch_start + i] = masks
            bbox_lst[batch_start + i] = input_boxes
                
            if verbose:
                print(f"Frame {frame_idx}: Found {sam_masks_lst[batch_start + i].shape[0] if sam_masks_lst[batch_start + i] is not None else 0} objects")
        
        for i, frame_idx in enumerate(tqdm(batch_frame_indices, desc=f"Batch {batch_idx + 1} CPU transfer...", unit="frame", 
                                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
            sam_masks_lst[batch_start + i] = sam_masks_lst[batch_start + i].cpu()
            bbox_lst[batch_start + i] = bbox_lst[batch_start + i].cpu()


        gc.collect()
        # Force GPU memory cleanup after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"Completed batch {batch_idx + 1}/{num_batches} - moved tensors to CPU")

    return sam_masks_lst, bbox_lst

def process_SAM_CLIP_Manip_Frame(batch_start, index, depth_frames_batch, video_frame_data, pc_generator, threshold, device):
    """
    Process a single frame to identify manipulated objects and compute 3D centroids.
    
    This function analyzes SAM masks against manipulation masks to find the object
    with the highest overlap ratio above the threshold. It computes 2D pixel centroids
    and 3D world centroids for all objects in the frame.
    
    Args:
        batch_start (int): Starting index of the current batch
        index (int): Index within the current batch (0-based)
        depth_frames_batch (torch.Tensor): Batch of depth frames [batch_size, H, W]
        video_frame_data (VideoFrameData1): Video data structure containing frame information
        pc_generator (PointCloudGenerator): Point cloud generator for 3D coordinate conversion
        threshold (float): Minimum overlap ratio threshold for valid manipulation detection
        device (torch.device): Device to perform computations on (CPU/GPU)
        
    Returns:
        VideoFrameData1: Updated video frame data with manipulation analysis results
        
    Side Effects:
        Updates the following fields in video_frame_data.frames[overall_idx]:
        - "manip_idx": Index of the manipulated object (-1 if none found)
        - "centroid_pixels": 2D pixel centroids for all objects [N, 2]
        - "centroid_xyzs": 3D world centroids for all objects [N, 3]
        - "manip_centroid_xyz": 3D centroid of the manipulated object [3] or empty tensor
        
    Notes:
        - Empty masks (all zeros) are skipped and assigned invalid centroids [-1.0, -1.0, -1.0]
        - Only objects with overlap ratio >= threshold are considered for manipulation
        - Among valid objects, the one with maximum overlap area is selected
        - GPU memory is cleaned up after processing
    """
    
    overall_idx = batch_start + index

    sam_masks = video_frame_data.frames[overall_idx]["sam_masks"].to(device).detach()
    manip_mask = video_frame_data.frames[overall_idx]["manip_mask"].to(device).to_dense().detach()
    manip_idx = -1
    n, h, w = sam_masks.shape

    overlap = torch.logical_and(sam_masks, manip_mask)
    overlap_sums = overlap.sum(dim=(1, 2))
    mask_sums = sam_masks.sum(dim=(1, 2))  # [n]
    
    # Filter out overlaps below threshold
    overlap_ratios = overlap_sums / mask_sums.clamp(min=1)
    valid_mask = overlap_ratios >= threshold
    
    mask_indices = torch.empty(0, dtype=torch.long)
    if valid_mask.any():
        # Find the mask with maximum overlap area among valid ones
        valid_overlap_sums = overlap_sums[valid_mask]
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        max_overlap_idx = valid_indices[torch.argmax(valid_overlap_sums)]
        mask_indices = torch.tensor([max_overlap_idx])       

    sam_mask_new = torch.zeros(sam_masks[0].shape).to(device).detach()

    manip_idx = -1
    if mask_indices.shape[0] > 0:
        manip_idx = int(mask_indices[0].item())

    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # [h, w], [h, w]
    ys = ys.to(device).detach()
    xs = xs.to(device).detach()
    mask_sums = sam_masks.sum(dim=(1, 2))  # [n]
    centroid_x = (sam_masks * xs).sum(dim=(1, 2)) / mask_sums.clamp(min=1)
    centroid_y = (sam_masks * ys).sum(dim=(1, 2)) / mask_sums.clamp(min=1)     
    centroid_pixels = torch.stack([centroid_x, centroid_y], dim=1)

    n, h, w = sam_masks.shape

    centroid_xyzs = torch.zeros(sam_masks.shape[0], 3).to(device).detach()
    for mask_idx in range(sam_masks.shape[0]):
        if not sam_masks[mask_idx].any():
            centroid_xyzs[mask_idx] = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32, device=device) # default invalid centroid
            continue

        mask_true_indices = torch.nonzero(sam_masks[mask_idx], as_tuple=True)
        coords = torch.stack([mask_true_indices[1], mask_true_indices[0]], dim=1)  # [num_pixels, 2]
        points_3d = pc_generator._create_point_cloud_collection_pts_torch(coords, depth_frames_batch[index])
        centroid_xyzs[mask_idx] = torch.squeeze(torch.mean(points_3d, dim=0), dim=0)

    video_frame_data.frames[overall_idx]["manip_idx"] = manip_idx
    video_frame_data.frames[overall_idx]["centroid_pixels"] = centroid_pixels
    video_frame_data.frames[overall_idx]["centroid_xyzs"] = centroid_xyzs
    video_frame_data.frames[overall_idx]["manip_centroid_xyz"] = centroid_xyzs[manip_idx] if manip_idx != -1 else torch.empty(0,3)

    # Clean up GPU memory
    del overlap, sam_mask_new, sam_masks
    del ys, xs, centroid_x, centroid_y, mask_true_indices, coords, points_3d
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    #print(f"CUDA MEMORY USAGE ALLOCATED {torch.cuda.memory_allocated()}  RESERVED {torch.cuda.memory_reserved()}")
    
    return video_frame_data

def process_SAM_CLIP_Manip(depth_frames, frame_indices, video_frame_data, pc_generator, batch_size=64, threshold=0.5):
    """
    Process video frames in batches to identify manipulated objects and compute 3D centroids.
    
    This function processes multiple frames in batches to analyze SAM masks against manipulation
    masks, identifying manipulated objects and computing their 2D/3D centroids. It handles
    GPU memory management by processing in batches and moving results back to CPU.
    
    Args:
        depth_frames (np.ndarray): Array of depth frames [N, H, W]
        frame_indices (list[int]): List of frame indices to process
        video_frame_data (VideoFrameData1): Video data structure containing frame information
        pc_generator (PointCloudGenerator): Point cloud generator for 3D coordinate conversion
        batch_size (int, optional): Number of frames to process per batch. Defaults to 64.
        threshold (float, optional): Minimum overlap ratio threshold for manipulation detection. Defaults to 0.5.
        
    Returns:
        VideoFrameData1: Updated video frame data with manipulation analysis results for all frames
        
    Side Effects:
        Updates the following fields in video_frame_data.frames for each processed frame:
        - "manip_idx": Index of the manipulated object (-1 if none found)
        - "centroid_pixels": 2D pixel centroids for all objects [N, 2]
        - "centroid_xyzs": 3D world centroids for all objects [N, 3]
        - "manip_centroid_xyz": 3D centroid of the manipulated object [3] or empty tensor
        - "manip_mask": Manipulation mask (moved to sparse format on CPU)
        - "sam_masks": SAM masks (moved to CPU)
        
    Notes:
        - Processes frames in batches to manage GPU memory efficiently
        - Automatically detects and uses CUDA if available, falls back to CPU
        - Performs GPU memory cleanup after each batch
        - Moves all computed tensors back to CPU after processing
        - Uses tqdm progress bars for batch and frame-level progress tracking
        - Empty masks are handled gracefully with invalid centroids [-1.0, -1.0, -1.0]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_batches = (len(frame_indices) + batch_size - 1) // batch_size
    print(f"Processing {len(frame_indices)} frames in {num_batches} batches of {batch_size}")
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(frame_indices))
        
        batch_frame_indices = frame_indices[batch_start:batch_end]
        depth_frames_batch = torch.tensor(depth_frames[batch_frame_indices]).to(device).detach()

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (frames {batch_start}-{batch_end-1})")
        for i, frame_idx in enumerate(tqdm(batch_frame_indices, desc=f"Batch {batch_idx + 1}", unit="frame", 
                                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
            
            video_frame_data = process_SAM_CLIP_Manip_Frame(batch_start, i, depth_frames_batch, video_frame_data,
                                                                            pc_generator, threshold, device)
            
        for i, frame_idx in enumerate(tqdm(batch_frame_indices, desc=f"Batch {batch_idx + 1} to CPU...", unit="frame", 
                                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
            overall_idx = batch_start + i
            video_frame_data.frames[overall_idx]["centroid_pixels"] = video_frame_data.frames[overall_idx]["centroid_pixels"].cpu()
            video_frame_data.frames[overall_idx]["centroid_xyzs"] = video_frame_data.frames[overall_idx]["centroid_xyzs"].cpu()
            video_frame_data.frames[overall_idx]["manip_centroid_xyz"] = video_frame_data.frames[overall_idx]["manip_centroid_xyz"].cpu()
            video_frame_data.frames[overall_idx]["manip_mask"] = video_frame_data.frames[overall_idx]["manip_mask"].to_sparse().cpu()
            video_frame_data.frames[overall_idx]["sam_masks"] = video_frame_data.frames[overall_idx]["sam_masks"].cpu()

        # Force GPU memory cleanup after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return video_frame_data

def update_manip_metadata(video_frame_data: VideoFrameData1, manip_metadata: ManipulationTimeline, 
                          frame_indices: list[int], pc_generator: PointCloudGenerator, depth_frames) -> tuple[VideoFrameData1, ManipulationTimeline]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manip_metadata = get_manipulation_periods(manip_metadata, video_frame_data, frame_indices)

    for idx, (start_idx, end_idx) in enumerate(tqdm(manip_metadata.manipulation_periods, desc="Updating manipulation metadata")):
        assert start_idx != end_idx, "Start and end index are the same; update_video_datasets()"
        start_idx_pos = start_idx - frame_indices[0]
        end_idx_pos = end_idx - frame_indices[0]

        # Update unit_dir_to_ends for manipulated objects & moved_dirs
        manip_xyzs = torch.stack([item["manip_centroid_xyz"] for i, item in enumerate(video_frame_data.frames[start_idx_pos:end_idx_pos+1])]).to(device).detach()
        manip_xyzs_diff = manip_xyzs[1:] - manip_xyzs[:-1]
        manip_xyzs_diff = torch.cat([manip_xyzs_diff, torch.zeros(1,3).to(device).detach()])

        start_pt = video_frame_data.frames[start_idx_pos]["manip_centroid_xyz"].to(device).detach()
        end_pt = video_frame_data.frames[end_idx_pos]["manip_centroid_xyz"].to(device).detach()

        unit_dir_to_ends = unit_direction_vector(manip_xyzs, end_pt.unsqueeze(0).expand_as(manip_xyzs))

        # Update entries 
        for j,k in enumerate(range(start_idx_pos, end_idx_pos+1)):
            video_frame_data.frames[k]["moved_dirs"] = manip_xyzs_diff[j]
            video_frame_data.frames[k]["unit_dir_to_end"] = unit_dir_to_ends[j]

        # Get straight line paths
        if torch.equal(start_pt, end_pt):
            print(f"Start and end point are the same: {start_pt}; Invalid path")
            continue

        manip_metadata.manipulation_points.append((start_pt, end_pt))
        straight_line = create_straight_line_representation(start_pt, end_pt)
        manip_metadata.straight_lines.append(straight_line)
    
    # update delta_to_straight_line for every object in frame within manipulation periods
    for idx, (start_idx, end_idx) in enumerate(manip_metadata.manipulation_periods):
        start_idx_pos = start_idx - frame_indices[0]
        end_idx_pos   = end_idx - frame_indices[0]
        depth_frames_torch = torch.from_numpy(depth_frames[start_idx:end_idx+1]).to(device) # mem constriant, don't put all on GPU at once

        n_sample_points = 200
        straight_line_i = manip_metadata.straight_lines[idx]
        line_points = sample_points_along_line(straight_line_i, n_sample_points).contiguous()

        d = 3
        # TODO: Don't keep reinit faiss.StandardGpuResources()
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        index.add(line_points)

        for i, frame_idx in tqdm(enumerate(range(start_idx, end_idx+1)), desc=f"Computing delta_to_straight_line for period {idx+1}/{len(manip_metadata.manipulation_periods)}", unit=" frame"):
            frame_idx_pos = frame_idx - frame_indices[0]

            sam_masks = video_frame_data.frames[frame_idx_pos]["sam_masks"].to(device)
            delta_to_straight_line = torch.zeros(sam_masks.shape[0], 1).detach()

            for obj_idx in range(sam_masks.shape[0]):
                # Skip if this is an empty sam_mask (all zeros)
                if not sam_masks[obj_idx].any():
                    continue
                points_3d = pc_generator._create_point_cloud_from_binary_mask_torch(sam_masks[obj_idx], depth_frames_torch[i])
                points_3d = points_3d.to(torch.float32).contiguous()
                D, I = index.search(points_3d, 1)
                min_D = torch.sqrt(torch.min(D))
                delta_to_straight_line[obj_idx] = min_D

            video_frame_data.frames[frame_idx_pos]["delta_to_straight_line"] = delta_to_straight_line
            del sam_masks

    for i in tqdm(range(len(manip_metadata.manipulation_points)), desc="Putting Manipulation Metadata to CPU..."):
        manip_metadata.straight_lines[i].start = manip_metadata.straight_lines[i].start.cpu()
        manip_metadata.straight_lines[i].end = manip_metadata.straight_lines[i].end.cpu()
        manip_metadata.straight_lines[i].direction = manip_metadata.straight_lines[i].direction.cpu()

    # Move all frames to CPU that have been processed
    for frame_idx in tqdm(frame_indices, desc="Putting extracted video_frame_data on CPU"):
        frame_idx_pos = frame_idx - frame_indices[0]
        video_frame_data.frames[frame_idx_pos]["delta_to_straight_line"] = video_frame_data.frames[frame_idx_pos]["delta_to_straight_line"].cpu()
        video_frame_data.frames[frame_idx_pos]["moved_dirs"] = video_frame_data.frames[frame_idx_pos]["moved_dirs"].cpu()
        video_frame_data.frames[frame_idx_pos]["unit_dir_to_end"] = video_frame_data.frames[frame_idx_pos]["unit_dir_to_end"].cpu()

    return video_frame_data, manip_metadata

def filter_dataset(video_frame_data: VideoFrameData1, frame_indices: list[int], dist_to_line_thres, 
                    manip_metadata: ManipulationTimeline, white_avg_tensor=torch.empty(0, 1024), 
                    black_avg_tensor=torch.empty(0, 1024), grey_avg_tensor=torch.empty(0, 1024), cosine_thres=0.4,
                    grey_list_flag=False):
    
    total_training_samples = 0
    total_possible_train_ex = 0
    total_grey_masked = 0

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Filtering dataset", unit="frame", total=len(frame_indices))):
        frame_data = video_frame_data.frames[i]
        dino_embeddings = frame_data["dino_embeddings"]  # [N_objs, C]
        
        if dino_embeddings.shape[0] == 0:  # No objects in this frame
            raise ValueError(f"No objects in frame {frame_idx}")
            
        n_objs = dino_embeddings.shape[0]
        
        # Compute cosine similarities using F.cosine_similarity
        # dino_embeddings: [N_objs, C], white_avg_tensor: [N_white, C]
        white_cosine_similarities = F.cosine_similarity(
            dino_embeddings.unsqueeze(1),  # [N_objs, 1, C]
            white_avg_tensor.unsqueeze(0),  # [1, N_white, C]
            dim=2
        )  # [N_objs, N_white]
        
        black_cosine_similarities = F.cosine_similarity(
            dino_embeddings.unsqueeze(1),  # [N_objs, 1, C]
            black_avg_tensor.unsqueeze(0),  # [1, N_black, C]
            dim=2
        )  # [N_objs, N_black]

        grey_cosine_similarities = F.cosine_similarity(
            dino_embeddings.unsqueeze(1),  # [N_objs, 1, C]
            grey_avg_tensor.unsqueeze(0),  # [1, N_grey, C]
            dim=2
        )  # [N_objs, N_grey]
        
        # Take maximum cosine similarity for each object across white/black/grey lists
        max_white_cosine = torch.max(white_cosine_similarities, dim=1)[0]  # [N_objs]
        max_black_cosine = torch.max(black_cosine_similarities, dim=1)[0]  # [N_objs]
        max_grey_cosine  = torch.max(grey_cosine_similarities, dim=1)[0]  # [N_objs]
        diff_cosine = max_white_cosine - max_black_cosine
        
        # Keep objects where white cosine similarity > black cosine similarity
        keep_mask = max_black_cosine <= cosine_thres
        keep_mask = ((max_white_cosine > cosine_thres) | (diff_cosine > 0.1)) & keep_mask

        # if we didn't include it last frame, don't include it this frame
        if i != 0:
            prev_frame_idx = frame_indices[i-1]
            if are_frames_in_same_manipulation_period(frame_idx, prev_frame_idx, manip_metadata):
                keep_mask = keep_mask & video_frame_data.frames[i-1]["in_dataset"]
            else:
                # Only apply distance filter if current frame is within a manipulation period
                if is_frame_in_any_manipulation_period(frame_idx, manip_metadata):
                    dists = video_frame_data.frames[i]["delta_to_straight_line"]
                    if dists.numel() == 0:
                        # No distances computed yet; skip distance filtering for this frame
                        pass
                    else:
                        if dists.dim() == 2 and dists.shape[1] == 1:
                            dists = dists.squeeze(1)
                        if dists.shape[0] == n_objs:
                            keep_mask = keep_mask & (dists >= 0) & (dists < dist_to_line_thres)
                        else:
                            # Mismatch; skip distance filtering to avoid size errors
                            pass
        else:
            # Frame 0: only apply distance filter if frame is within a manipulation period
            if is_frame_in_any_manipulation_period(frame_idx, manip_metadata):
                dists = video_frame_data.frames[i]["delta_to_straight_line"]
                if dists.numel() == 0:
                    pass
                else:
                    if dists.dim() == 2 and dists.shape[1] == 1:
                        dists = dists.squeeze(1)
                    if dists.shape[0] == n_objs:
                        keep_mask = keep_mask & (dists >= 0) & (dists < dist_to_line_thres)
                    else:
                        pass

        manip_idx = frame_data["manip_idx"]
        if manip_idx != -1:
            keep_mask[manip_idx] = True
        
        if grey_list_flag:
            grey_mask = torch.zeros(n_objs, dtype=torch.bool)
            grey_mask[keep_mask] = max_grey_cosine[keep_mask] > max_white_cosine[keep_mask]
            if i != 0:
                prev_frame_idx = frame_indices[i-1]
                # if the object was of no risk last time, it's also no risk this time
                grey_mask = grey_mask | video_frame_data.frames[i-1]["no_risk"]
            total_grey_masked += torch.sum(grey_mask).item()
            frame_data["no_risk"] = grey_mask
        
        total_training_samples += torch.sum(keep_mask).item()
        total_possible_train_ex += n_objs
        frame_data["in_dataset"] = keep_mask

    # Go through the array backwards and see if there's any no_risk that we missed 
    for i, frame_idx in enumerate(tqdm(frame_indices[::-1], desc="Filtering dataset reverse", unit="frame", total=len(frame_indices))):
        i = len(frame_indices) - 1 - i  # Convert reversed index back to original index
        if i > 0:
            continue
        video_frame_data.frames[i-1]["no_risk"] = video_frame_data.frames[i-1]["no_risk"] | video_frame_data.frames[i]["no_risk"]

    print("**************************************************************")
    print(f"🔎 Total number of training samples: {total_training_samples}")
    print(f"Total possible training examples: {total_possible_train_ex}")
    print(f"Percent retained: {(100 * total_training_samples / total_possible_train_ex):.04f}%")
    if grey_list_flag:
        print(f"Total grey masked objects: {total_grey_masked}")
        print(f"Percent grey masked: {(100 * total_grey_masked / total_training_samples):.04f}%" if total_training_samples > 0 else "Percent grey masked: 0.0000%")
    print("**************************************************************")
    
    return video_frame_data

def filter_dataset_depth(video_frame_data: VideoFrameData1, depth_frames: np.ndarray, frame_indices: list[int], 
                            min_seg_size: int = 10_000, verbose=False):
    """
    Filter out segmentations that form mostly large flat planes based on depth analysis.
    
    Args:
        video_frame_data: VideoFrameData1 containing frame data with sam_masks
        depth_frames: Array of depth frames [N, H, W]
        frame_indices: List of frame numbers corresponding to video_frame_data.frames indices
        manip_metadata: ManipulationTimeline containing manipulation periods
        min_seg_size: Minimum segmentation size in pixels to consider (default: 10,000)
    
    Returns:
        VideoFrameData1: Updated video_frame_data with filtered segmentations
    """
    if verbose:    
        print("🔍 Starting depth-based segmentation filtering...")
    
    # Statistics tracking
    total_segments_processed = 0
    total_segments_filtered = 0
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Filtering depth-based segmentations", unit="frame")):
        frame_data = video_frame_data.frames[i]
        sam_masks = frame_data["sam_masks"]  # Shape: [N, H, W]
        depth_frame = depth_frames[i]  # Shape: [H, W]        
        assert sam_masks.shape[0] != 0, "sam_masks empty; filter_dataset_depth()"

        # Init defaults if they're not initalized
        if 'no_risk' not in frame_data or len(frame_data['no_risk']) == 0:
            frame_data['no_risk'] = torch.zeros(sam_masks.shape[0], dtype=torch.bool)
            if verbose:
                print(f"frame_data['no_risk'] created len: {sam_masks.shape[0]}")

        if 'in_dataset' not in frame_data or len(frame_data['in_dataset']) == 0:
            frame_data['in_dataset'] = torch.ones(sam_masks.shape[0], dtype=torch.bool)
            if verbose:
                print(f"frame_data['in_dataset'] created len: {sam_masks.shape[0]}")
        
        # Process each segmentation mask
        for obj_idx in range(sam_masks.shape[0]):
            mask = sam_masks[obj_idx]  # Shape: [H, W]
            mask_np = mask.cpu().numpy().astype(bool)

            # Skip if mask is too small
            if np.sum(mask_np) < min_seg_size:
                continue
                
            total_segments_processed += 1
            masked_depth = depth_frame[mask_np]
            
            # TODO: Put this code into is_flat_plane()
            # Filter out invalid depth values (0, negative, or NaN)
            valid_depth_mask = (masked_depth > 0) & np.isfinite(masked_depth)
            if np.sum(valid_depth_mask) < min_seg_size * 0.5:  # Need at least 50% valid depth
                continue
                
            valid_depths = masked_depth[valid_depth_mask]
            
            valid_pixels = np.where(mask_np)
            valid_pixels = np.column_stack((valid_pixels[1], valid_pixels[0]))  # (x, y) format
            valid_pixels = valid_pixels[valid_depth_mask]
            
            # Convert to 3D points (simplified - assuming unit focal length for plane detection)
            # We only need relative positions for plane fitting, so exact intrinsics aren't critical
            points_3d = np.column_stack((
                valid_pixels[:, 0],  # x
                valid_pixels[:, 1],  # y  
                valid_depths         # z (depth)
            ))
            
            # Check if this segmentation forms a flat plane
            is_flat_plane_flag = is_flat_plane(points_3d, plane_threshold=0.02, min_points=100, verbose=verbose)
            
            if is_flat_plane_flag:
                if verbose:
                    print(f"  Filtering out object {obj_idx} in frame {frame_idx} - detected as flat plane")
                total_segments_filtered += 1
                frame_data["no_risk"][obj_idx] = True

        # leverage potential upstream filtering for items that should be in the dataset
        frame_data["no_risk"] = frame_data["no_risk"] & frame_data["in_dataset"]

        # Get object indices that are marked as no_risk
        no_risk_indices = torch.where(frame_data["no_risk"])[0]
        if len(no_risk_indices) == 0:
            continue
        reference_masks = [sam_masks[i] for i in no_risk_indices]

        # for each frame, label all object's that have near 100% overlap with table as out of 
        # dataset since these are table stains
        for obj_idx in range(sam_masks.shape[0]):
            if obj_idx in no_risk_indices:
                continue
            mask = sam_masks[obj_idx]
            for i, ref_mask in enumerate(reference_masks):
                percent = get_percentage_overlap(mask, ref_mask)
                if verbose:
                    print(f"Checking overlap between obj {obj_idx} and reference obj {no_risk_indices[i]} (flat plane): {percent:.3f}")
                if percent > 0.9:
                    frame_data["in_dataset"][obj_idx] = False
                    if verbose:
                        print(f"obj_idx {obj_idx} set to NOT in dataset")

    print(f"✅ Depth filtering complete:")
    print(f"   Total segments processed: {total_segments_processed}")
    print(f"   Total segments filtered: {total_segments_filtered}")
    print(f"   Filter rate: {(total_segments_filtered/total_segments_processed*100):.1f}%" if total_segments_processed > 0 else "   Filter rate: 0.0%")
    
    return video_frame_data

def track_objects(rgb_frames, video_frame_data: VideoFrameData1, frame_indices: list[int], verbose=False):
    """
    Track objects across video frames using SAM2 and convert to consistent tensor format.
    
    This function performs object tracking across specified frames using SAM2 video predictor,
    then converts the tracking results into a consistent tensor format where object IDs
    correspond directly to tensor indices. This enables downstream processing with consistent
    object identification across frames.
    
    Args:
        rgb_frames (np.ndarray): Array of RGB frames [N, H, W, 3]
        video_frame_data (VideoFrameData1): Video data structure containing frame information
        frame_indices (list[int]): List of frame indices to process for tracking
        verbose (bool, optional): Whether to save visualizations and debug outputs. Defaults to False.
        
    Returns:
        list[torch.Tensor]: List of mask tensors, one per frame, with shape [max_obj_id+1, H, W]
                           where index corresponds to object ID and contains binary mask
        
    Side Effects:
        If verbose=True:
        - Saves video visualizations to "obj_track_sam" directory
        - Creates object tracking video "object_tracking_video.mp4"
        - Saves individual mask images as "test_track_id_frame_{frame_idx}_obj_{obj_id}.png"
        
    Notes:
        - Uses SAM2 video predictor for robust object tracking across frames
        - Maintains consistent object IDs across all frames
        - Empty masks (object not present in frame) are represented as zeros
        - Tensor structure: [obj_id, H, W] where obj_id directly corresponds to object ID
        - Raises ValueError if expected frame is missing from tracking results
        - Object IDs are sorted for consistent ordering across frames
    """
    # Filter rgb_frames to only include the specified frame indices
    rgb_frames_filtered = rgb_frames[frame_indices]
    
    seed_bbox = video_frame_data.frames[0]["bboxes"]
    video_segments = track_from_bbs(rgb_frames_filtered, seed_bbox)
    
    if verbose:
        save_video_visualizations(video_segments, rgb_frames_filtered)
        create_video_from_vis(input_dir="obj_track_sam", output_filename="object_tracking_video.mp4", fps=30)

    # Convert video_segments to sam_masks_lst format with consistent object IDs
    sam_masks_lst: list[torch.Tensor] = [None for i in range(len(frame_indices))]
    
    # Get all unique object IDs across all frames to maintain consistency
    all_obj_ids = set()
    for frame_data in video_segments.values():
        all_obj_ids.update(frame_data.keys())
    all_obj_ids = sorted(list(all_obj_ids))  # Sort for consistent ordering
    
    print(f"Found {len(all_obj_ids)} unique objects across {len(video_segments)} frames")
    
    # Convert each frame's masks to tensor format
    for i, frame_idx in enumerate(frame_indices):
        if i in video_segments:
            frame_data = video_segments[i]
            
            # Create masks tensor with object ID as first dimension
            max_obj_id = max(all_obj_ids) if all_obj_ids else 0
            H, W = rgb_frames_filtered[i].shape[:2]
            
            # Initialize tensor with zeros: [max_obj_id+1, H, W]
            masks_tensor = torch.zeros((max_obj_id + 1, H, W), dtype=torch.uint8)
            
            for obj_id in all_obj_ids:
                if obj_id in frame_data:
                    mask = frame_data[obj_id]
                    # Remove extra dimension if present (same as save_video_visualizations)
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    masks_tensor[obj_id] = torch.from_numpy(mask.astype(np.uint8))
                # If obj_id not in frame_data, it remains zeros (empty mask)
                        
            sam_masks_lst[i] = masks_tensor
            
            # Save individual masks using object IDs
            if verbose:
                for obj_id in all_obj_ids:
                    output_path = f"test_track_id_frame_{frame_idx}_obj_{obj_id}.png"
                    mask_im = masks_tensor[obj_id].cpu().numpy() * 255
                    cv2.imwrite(output_path, mask_im)

        else:
            raise ValueError(f"Frame {frame_idx} not found in video_segments but was expected")
    
    return sam_masks_lst 

def get_dino_features_per_segmentation(rgb_frames, video_frame_data: VideoFrameData1, frame_indices: list[int], 
                                        dino_featurizer: DinoFeatures, target_h=600, n_samples=10):

    # Iterate through frames using frame_indices
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames for DINO features")):
        rgb_frame = rgb_frames[frame_idx]
        rgb_tensor = dino_featurizer.rgb_image_preprocessing(rgb_frame, target_h=target_h)
        dino_features = dino_featurizer.get_dino_features_rgb(rgb_tensor)

        _, h_dino, w_dino = dino_features.shape
        sam_masks = video_frame_data.frames[i]["sam_masks"]  # Shape: [N, H, W] where N is number of masks
        
        if sam_masks.numel() > 0:
            # Dino image too large (1024 channels) so need to downsize to match
            sam_masks_resized = F.interpolate(
                sam_masks.unsqueeze(1).float(),  # [N, 1, H_mask, W_mask]
                size=(h_dino, w_dino),
                mode='nearest'  # Use nearest neighbor to preserve binary mask values
            ).squeeze(1).bool()  # Back to [N, H_dino, W_dino] and convert to bool
            
            avg_dino_features_per_mask = []
            random_samples_per_mask = []
            
            for mask_idx in range(sam_masks_resized.shape[0]):
                current_mask = sam_masks_resized[mask_idx]  # [H_dino, W_dino]
                mask_indices = torch.nonzero(current_mask, as_tuple=True)  # (y_indices, x_indices)
                
                if len(mask_indices[0]) > 0:
                    masked_features_flat = dino_features[:, mask_indices[0], mask_indices[1]]  # [C, N]
                    avg_feature = torch.mean(masked_features_flat, dim=1)  # [C]
                    avg_dino_features_per_mask.append(avg_feature)
                    
                    # Sample n_samples random features from this mask
                    num_pixels = masked_features_flat.shape[1]
                    random_indices = torch.randperm(num_pixels, device=dino_features.device)[:n_samples]
                    random_samples = masked_features_flat[:, random_indices].T  # [C, n_samples] -> [n_samples, C]
                    random_samples_per_mask.append(random_samples)
                else:
                    # Empty mask - create zero tensors with correct dimensions
                    avg_dino_features_per_mask.append(torch.zeros(dino_features.shape[0], device=dino_features.device))  # [C]
                    random_samples_per_mask.append(torch.zeros(n_samples, dino_features.shape[0], device=dino_features.device))  # [n_samples, C]
            
            # Update the standard fields from getEmptyFrameData & put on CPU
            video_frame_data.frames[i]["dino_embeddings"] = torch.stack(avg_dino_features_per_mask, dim=0).cpu()  # [N, C]
            video_frame_data.frames[i]["dino_embedding_samples"] = torch.stack(random_samples_per_mask, dim=0).cpu()  # [N, n_samples, C]
            avg_dino_features_per_mask, random_samples_per_mask = None, None
            del avg_dino_features_per_mask, random_samples_per_mask

        else:
            # No SAM masks - this should not happen, throw an error
            raise ValueError(f"No SAM masks found for frame {frame_idx} at index {i}. This indicates a problem with object detection or tracking.")
    
    return video_frame_data

def process_npz_video(rgb_frames, depth_frames, camera_intrinsics_path: str, 
                      frame_range: tuple[int, int] = None, verbose: bool = False, batch_size=64, dist_to_line_thres=0.1,
                      white_avg_tensor=torch.empty(0, 1024), black_avg_tensor=torch.empty(0, 1024), 
                      grey_avg_tensor=torch.empty(0, 1024)) -> tuple[VideoFrameData1, ManipulationTimeline, list[int]]:
    """
    Process every image in a video and populate a VideoFrameData1 structure,
    including manipulation object detection from the hoist_former service.
    
    Args:
        rgb_frames: RGB frames (numpy array)
        depth_frames: Depth frames in meters (numpy array)
        camera_intrinsics_path: Path to camera intrinsics JSON file
                   (if None, temporary directories are created)
        frame_range: Optional tuple (start_idx, end_idx) for frame range to process
                   (if None, process all frames)
        verbose: Whether to print verbose output
        
    Returns:
        VideoFrameData1: Populated frame data structure with manipulation detection
    """

    num_frames, height, width, channels = rgb_frames.shape
    print(f"Loaded {num_frames} frames from input array")
    print(f"Frame dimensions: {width}x{height}")
    
    # Limit frames if specified
    if frame_range is not None:
        start_idx, end_idx = frame_range
        frame_indices = list(range(start_idx, min(end_idx, num_frames)))
        print(f"Processing frames {start_idx} to {min(end_idx, num_frames)-1} ({len(frame_indices)} frames)")
    else:
        frame_indices = list(range(num_frames))
        print(f"Processing all {num_frames} frames")
    
    print("Initializing vision pipeline models...")
    feature_extractor = FeatureExtractor()
    pc_generator = PointCloudGenerator(intrinsics=camera_intrinsics_path)
    dino_featurizer = DinoFeatures()
    assert run_test_hoist_former_service_availability() 
    
    # Create default FrameData objects with proper empty tensors
    video_frame_data = VideoFrameData1(frames=[getEmptyFrameData() for _ in range(len(frame_indices))])
    manip_metadata = ManipulationTimeline(
            manipulation_timeline=[0 for _ in range(len(frame_indices))],
            manipulation_periods=[],
            manipulation_points=[],
            straight_lines=[]
        )

    # Phase 1: Run manipulation detection
    print("=======================================================================================================")
    print("Phase 1: Running manipulation detection...")
    t0 = time.time()
    manip_scores, manip_labels, manip_masks = run_manipulation_detection(rgb_frames, frame_indices, verbose=verbose)
    run_manipulation_detection_time = time.time() - t0
    assert(manip_masks.shape[0] == len(frame_indices)), "The number of manipulation masks != number of frame indices"

    # Phase 2: Run vision pipeline (SAM + CLIP) with Object Tracking
    print("=======================================================================================================")
    print("Phase 2: Running vision pipeline (SAM + CLIP)...")
    t0 = time.time()
    sam_masks_lst, bbox_lst = run_CLIP_SAM_segmentation(rgb_frames, frame_indices, feature_extractor, verbose=verbose, batch_size=batch_size)
    gc.collect()
    
    # Update video_frame_data so we don't need to pass more to the function
    for i, frame_idx in enumerate(frame_indices):
        video_frame_data.frames[i]["bboxes"] = bbox_lst[i]
        video_frame_data.frames[i]["manip_mask"] = manip_masks[i]

    tracked_masks_lst = track_objects(rgb_frames, video_frame_data, frame_indices, verbose=verbose)
    
    # Update video_frame_data with tracked masks
    for i, frame_idx in enumerate(frame_indices):
        video_frame_data.frames[i]["sam_masks"] = tracked_masks_lst[i]

    video_frame_data = get_dino_features_per_segmentation(rgb_frames, video_frame_data, frame_indices, dino_featurizer, target_h=500, n_samples=10)

    run_CLIP_SAM_segmentation_time = time.time() - t0
    print(f"Object tracking completed. Updated sam_masks with tracked objects.")

    # Phase 3: Process 3D object data and integrate manipulation detection
    print("=======================================================================================================")
    print("Phase 3: Processing 3D object data and integrating manipulation detection...")
    t0 = time.time()
    video_frame_data = process_SAM_CLIP_Manip(depth_frames, frame_indices, video_frame_data, pc_generator, batch_size=batch_size)
    gc.collect()
    process_SAM_CLIP_Manip_time = time.time() - t0

    # Phase 4: Get straight line paths and update video datasets
    print("=======================================================================================================")
    print("Phase 4: Updating video datasets with manipulation paths...")
    t0 = time.time()
    video_frame_data, manip_metadata = update_manip_metadata(video_frame_data, manip_metadata, frame_indices, pc_generator, depth_frames)
    gc.collect()
    update_manip_metadata_time = time.time() - t0
    print("=======================================================================================================")

    # Phase 5: Filter dataset for training
    print("=======================================================================================================")
    print("Phase 5: Filter...")
    t0 = time.time()
    video_frame_data = filter_dataset(video_frame_data, frame_indices, dist_to_line_thres, manip_metadata, 
                                        white_avg_tensor, black_avg_tensor, grey_avg_tensor)

    # min_seg_size so large so we only capture wall & table
    # NOTE: Image Size: 720 * 1280 = 921,600 pixels approx 900,000 pixels
    video_frame_data = filter_dataset_depth(video_frame_data, depth_frames, frame_indices, min_seg_size=100_000, verbose=verbose)
    filter_dataset_time = time.time() - t0
    print("=======================================================================================================")       

    num_frames = len(frame_indices)
    print("================================= Pipeline Timing Summary =================================")
    print(f"Number of frames processed:       {num_frames}")
    print(f"Manipulation detection time:      {run_manipulation_detection_time:.2f} seconds")
    print(f"SAM + DINO segmentation time:     {run_CLIP_SAM_segmentation_time:.2f} seconds")
    print(f"3D object processing time:        {process_SAM_CLIP_Manip_time:.2f} seconds")
    print(f"Manipulation path update time:    {update_manip_metadata_time:.2f} seconds")
    print(f"Filter dataset time:              {filter_dataset_time:.2f} seconds")
    total_time = (
        run_manipulation_detection_time +
        run_CLIP_SAM_segmentation_time +
        process_SAM_CLIP_Manip_time +
        update_manip_metadata_time +
        filter_dataset_time
    )
    print("------------------------------------------------------------------------------------------")
    print(f"Total pipeline time:              {total_time:.2f} seconds")
    if total_time > 0:
        fps = num_frames / total_time
        print(f"Average processing speed:         {fps:.2f} frames/second")
    else:
        print("Average processing speed:         N/A (total time is zero)")
    print("==========================================================================================")

    return video_frame_data, manip_metadata, frame_indices

def runner(npz_paths: list[str], camera_intrinsics_path: str, dataset_output_dir: str, 
           frame_range_lst: List[tuple] = None, batch_size=64, save_npys=False, create_visualizations=True, dist_to_line_thres=0.5,
           object_white_list=[], object_black_list=[], object_grey_list=[], white_list_filenames=[], black_list_filenames=[], 
           grey_list_filenames=[]):

    output_dir_vis_lst = []
    output_dir_save_lst = []
    video_frame_data_lst = []
    manip_metadata_lst = []
    frame_indices_lst = []

    camera_intrinsics = PointCloudGenerator.load_intrinsics_from_file(camera_intrinsics_path)
    
    # Copy the process_video_config.yaml to the output directory for reference
    config_source_path = os.path.join(os.path.dirname(__file__), 'process_video_config.yaml')
    config_dest_path = os.path.join(dataset_output_dir, 'process_video_config.yaml')

    # Ensure the output directory exists
    os.makedirs(dataset_output_dir, exist_ok=True)
    shutil.copy2(config_source_path, config_dest_path)
    print(f"✅ Copied process_video_config.yaml to output directory: {config_dest_path}")

    print("Data to be processed: ")
    [print(data) for i, data in enumerate(npz_paths)]

    dino_featurizer = DinoFeatures()
    white_avg_tensor, black_avg_tensor, grey_avg_tensor, white_labels, black_labels, grey_labels = get_image_embedd_white_black_lst(dino_featurizer, object_white_list, object_black_list, object_grey_list, 
                                                                                                                                    white_list_filenames, black_list_filenames, grey_list_filenames)

    for i in range(len(npz_paths)):
        npz_path_color = npz_paths[i] + "/color.npy"
        npz_path_depth = npz_paths[i] + "/depth.npy"
        output_dir_vis  = os.path.join(dataset_output_dir, os.path.basename(npz_paths[i]) + "_viz")
        output_dir_save = os.path.join(dataset_output_dir, os.path.basename(npz_paths[i]) + "_dataset")
        output_dir_vis_lst.append(output_dir_vis)
        output_dir_save_lst.append(output_dir_save)

        frame_range = None
        if frame_range_lst is not None:
            frame_range = frame_range_lst[i]

        rgb_frames, depth_frames = data_loader1(npz_path_color, npz_path_depth)

        video_frame_data, manip_metadata, frame_indices = process_npz_video(rgb_frames=rgb_frames, depth_frames=depth_frames, 
                                                                            camera_intrinsics_path=camera_intrinsics_path, 
                                                                            frame_range=frame_range, verbose=False, batch_size=batch_size, 
                                                                            dist_to_line_thres=dist_to_line_thres,
                                                                            white_avg_tensor=white_avg_tensor, black_avg_tensor=black_avg_tensor, grey_avg_tensor=grey_avg_tensor)
        gc.collect()
        video_frame_data_lst.append(video_frame_data)
        manip_metadata_lst.append(manip_metadata)
        frame_indices_lst.append(frame_indices)

        print(f"Processed {len(video_frame_data.frames)} frames with manipulation detection")

        num_objs_orig = video_frame_data.frames[0]["sam_masks"].shape[0]

        # save dataset and test loading it with saving images to verify it saved correctly
        saved_vid_frame_data = video_frame_data.save_frames(output_dir_save, config_source_path, npz_paths[i], frame_indices)
        saved_manip_metadata = manip_metadata.save(output_dir=output_dir_save)

        frame_indices = None
        video_frame_data.delete_memory(); manip_metadata.delete_memory()
        video_frame_data, manip_metadata = None, None; del video_frame_data, manip_metadata
        
        video_frame_data_loaded = VideoFrameData1.load_frames(saved_vid_frame_data)
        manip_metadata_loaded = ManipulationTimeline.load(saved_manip_metadata)

        frame_indices_file = os.path.join(saved_vid_frame_data, "frame_indices.pkl")
        if os.path.exists(frame_indices_file):
            with open(frame_indices_file, 'rb') as f:
                frame_indices = pickle.load(f)

        num_objs_loaded = video_frame_data_loaded.frames[0]["sam_masks"].shape[0]
        assert num_objs_orig == num_objs_loaded, "The number of objects changed when saving"

        if create_visualizations:
            # only use for debugging mostly
            save_sam_masks_as_images(video_frame_data_loaded.frames[0], output_dir_vis, frame_indices[0], rgb_frames[0], verbose=verbose)
            visualize_frame_data(
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                video_frame_data=video_frame_data_loaded,
                manip_metadata=manip_metadata_loaded,
                output_dir=output_dir_vis,
                frame_indices=frame_indices,
                camera_intrinsics=camera_intrinsics["camera_intrinsics"],
                verbose=False,
                save_npys=save_npys,
                pca_subsample_frame_idx=0,
            )
            visualize_frames_video(output_dir_vis, frame_indices, fps=5)

        # Clean memory after processing video
        print("freeing memory: video_frame_data, manip_metadata, etc.")
        video_frame_data_loaded.delete_memory(); video_frame_data_loaded = None
        manip_metadata_loaded.delete_memory(); manip_metadata_loaded = None
        dino_featurizer = None
        del video_frame_data_loaded, dino_featurizer, manip_metadata_loaded
        gc.collect()
    
    # Flush any remaining data in buffer
    gc.collect()

    return dataset_output_dir

# Example usage
if __name__ == "__main__":

    cfg = load_process_video_config("open_world_risk/video_processing/process_video_config.yaml")
    npz_paths = cfg["npz_paths"]
    camera_intrinsics_path = cfg["camera_intrinsics_path"]
    output_dir = cfg["output_dir"]
    frame_range_lst = cfg.get("frame_range_lst", None)
    batch_size = cfg.get("batch_size", 64)
    save_npys = cfg["save_npys"]
    dist_to_line_thres = cfg.get("dist_to_line_thres", 0.5)
    object_white_list = cfg.get("white_list_images", [])
    object_black_list = cfg.get("black_list_images", [])
    object_grey_list  = cfg.get("grey_list_images", [])
    white_list_filenames = cfg["white_list_filenames"]
    black_list_filenames = cfg["black_list_filenames"]
    grey_list_filenames  = cfg["grey_list_filenames"]
    create_visualizations = cfg.get("create_visualizations", True)
    
    dataset_output_dir = runner(npz_paths=npz_paths, camera_intrinsics_path=camera_intrinsics_path, 
                                dataset_output_dir=output_dir, frame_range_lst=frame_range_lst, batch_size=batch_size, 
                                save_npys=save_npys, create_visualizations=create_visualizations, dist_to_line_thres=dist_to_line_thres, 
                                object_white_list=object_white_list, object_black_list=object_black_list, object_grey_list=object_grey_list,
                                white_list_filenames=white_list_filenames, black_list_filenames=black_list_filenames, grey_list_filenames=grey_list_filenames)
