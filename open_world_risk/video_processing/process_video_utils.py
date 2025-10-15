# Standard library imports
import os
import time
import hashlib
import pickle
import warnings
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess, shutil

# Third-party imports
import numpy as np
import cv2
import torch
import faiss
import yaml
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
import petname as PetnamePkg

# Local imports
from open_world_risk.src.api import test_hoist_former_service_availability
from open_world_risk.src.video_datastructures import (
    VideoFrameData1,
    ManipulationTimeline,
    StraightLine,
    VideoManipulationData,
)
from open_world_risk.src.dino_features import DinoFeatures

# Convert RuntimeWarning to exception to trigger breakpoints
warnings.filterwarnings('error', category=RuntimeWarning)

# Configure numpy to raise exceptions on invalid operations
np.seterr(divide='raise', invalid='raise')

# TODO: Put into a data loader file so we can have a file full of data loaders
# When we have a file full of data loaders, put a loader for the intrinisics file
def data_loader(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a .npz file.
    NOTE: depth is in mm, convert to meters
    For the output we want a 1:1 mapping of frames to rgb and depth frames

    Args:
        npz_path: Path to the .npz file

    Returns: 
        rgb_frames: RGB frames
        depth_frames: Depth frames in meters
    """
    data = np.load(npz_path)
    rgb_frames = data['rgb']
    depth_frames = data['depth']
    depth_frames = depth_frames / 1000.0

    return rgb_frames, depth_frames

def data_loader1(npz_path_color: str, npz_path_depth) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a .npz file.
    NOTE: This is from the data gathered in the arm room. 
    For the output we want a 1:1 mapping of frames to rgb and depth frames

    Args:
        npz_path: Path to the .npz file

    Returns: 
        rgb_frames: RGB frames
        depth_frames: Depth frames in meters
    """
    rgb_frames = np.load(npz_path_color)
    depth_frames = np.load(npz_path_depth)

    return rgb_frames, depth_frames

def run_test_hoist_former_service_availability():
    # Test if hoist_former service is available before running enhanced pipeline
    print("Checking hoist_former service availability...")
    
    # Retry mechanism with multiple attempts
    max_retries = 5
    retry_delay = 10  # seconds
    is_hoist_former_available = False
    
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries}...")
        is_hoist_former_available = test_hoist_former_service_availability()
        
        if is_hoist_former_available:
            print(f"✅ Hoist_former service is available on attempt {attempt + 1}")
            break
        else:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"❌ Service not available. Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Service failed to become available after {max_retries} attempts")
    
    print(f"hoist_former service available: {is_hoist_former_available}")
    return is_hoist_former_available

def create_depth_visualization(depth_frame, output_path: str, colormap: str = 'Greys'):
    """
    Create a depth visualization with a color bar showing depth values.
    
    Args:
        depth_frame: Depth frame in meters (numpy array)
        output_path: Path to save the visualization
        colormap: Matplotlib colormap name (default: 'viridis')
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [4, 1]})
    
    # Plot depth image
    im = ax1.imshow(depth_frame, cmap=colormap)
    ax1.set_title('Depth Visualization')
    ax1.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, cax=ax2, orientation='vertical')
    cbar.set_label('Depth (meters)', rotation=270, labelpad=20)
    
    # Set colorbar ticks to show actual depth range
    depth_min = np.nanmin(depth_frame)
    depth_max = np.nanmax(depth_frame)
    cbar.set_ticks([depth_min, depth_max])
    cbar.set_ticklabels([f'{depth_min:.2f}m', f'{depth_max:.2f}m'])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def fit_global_dino_pca(video_frame_data, frame_indices, subsample_frame_idx=None):
    """
    Fit a global PCA on DINO embeddings for consistent visualization.
    
    Args:
        video_frame_data: VideoFrameData1 containing all frame data
        frame_indices: List of frame indices to process
        subsample_frame_idx: If provided, fit PCA only on this frame's embeddings.
                           If None, fit on all frames' embeddings.
        
    Returns:
        sklearn.decomposition.PCA: Fitted PCA object
    """
    if subsample_frame_idx is not None:
        print(f"Fitting DINO PCA on subsample from frame {subsample_frame_idx}...")
        
        # Find the frame index in our list
        try:
            #frame_pos = frame_indices.index(subsample_frame_idx)
            frame_data = video_frame_data.frames[subsample_frame_idx]
            dino_embeddings = frame_data.get("dino_embeddings")
            
            if dino_embeddings is None or dino_embeddings.shape[0] == 0:
                print(f"Warning: No DINO embeddings found in frame {subsample_frame_idx}")
                return None
            
            embeddings_np = dino_embeddings.cpu().numpy()
            print(f"Fitting PCA on {embeddings_np.shape[0]} embeddings from frame {subsample_frame_idx}")
            
        except ValueError:
            print(f"Warning: Frame {subsample_frame_idx} not found in frame_indices")
            return None
    else:
        print("Fitting global DINO PCA using IncrementalPCA (memory efficient)...")
        return _fit_incremental_pca(video_frame_data, frame_indices)
    
    # Fit PCA using IncrementalPCA for better performance on large datasets
    print(f"Using IncrementalPCA for faster computation...")
    global_pca = IncrementalPCA(n_components=3, whiten=True, batch_size=min(1000, embeddings_np.shape[0]))
    global_pca.fit(embeddings_np)
    
    print("DINO PCA fitted successfully")
    return global_pca

def _fit_incremental_pca(video_frame_data, frame_indices, batch_size=1000):
    """
    Fit IncrementalPCA by processing embeddings in batches without loading everything into memory.
    
    Args:
        video_frame_data: VideoFrameData1 containing all frame data
        frame_indices: List of frame indices to process
        batch_size: Batch size for IncrementalPCA
        
    Returns:
        IncrementalPCA: Fitted PCA object
    """
    # Initialize IncrementalPCA
    global_pca = IncrementalPCA(n_components=3, whiten=True, batch_size=batch_size)
    
    # Process embeddings frame by frame using partial_fit
    print("Processing embeddings frame by frame...")
    total_embeddings = 0
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Fitting IncrementalPCA")):
        frame_data = video_frame_data.frames[i]
        dino_embeddings = frame_data.get("dino_embeddings")
        
        if dino_embeddings is not None and dino_embeddings.shape[0] > 0:
            embeddings_np = dino_embeddings.cpu().numpy()
            total_embeddings += embeddings_np.shape[0]
            
            # Use partial_fit to incrementally update PCA
            global_pca.partial_fit(embeddings_np)
    
    print(f"Processed {total_embeddings} total embeddings across {len(frame_indices)} frames")
    print("DINO IncrementalPCA fitted successfully")
    return global_pca

def get_pca_colors_from_embeddings(embeddings, pca_model):
    """
    Get RGB colors from embeddings using a pre-trained PCA model.
    
    Args:
        embeddings: torch.Tensor or np.ndarray of shape [N_objects, embedding_dim]
        pca_model: Pre-fitted sklearn PCA model
        
    Returns:
        np.ndarray: RGB colors of shape [N_objects, 3] with values in [0, 255]
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Transform using pre-trained PCA
    embeddings_3d = pca_model.transform(embeddings_np)  # Shape: [N_objects, 3]
    
    # Normalize to [0, 1] range and apply sigmoid for vibrant colors
    embeddings_3d_tensor = torch.from_numpy(embeddings_3d)
    embeddings_3d_normalized = torch.sigmoid(embeddings_3d_tensor * 2.0).numpy()
    
    # Convert to RGB colors [0, 255]
    rgb_colors = (embeddings_3d_normalized * 255).astype(np.uint8)
    
    return rgb_colors

def visualize_frame_data(
    rgb_frames, depth_frames, video_frame_data, manip_metadata, output_dir: str, 
    frame_indices: list, camera_intrinsics: np.ndarray, verbose: bool = False, save_npys = False,
    pca_subsample_frame_idx: int = 0
):
    """
    Visualize manipulation masks, SAM segmentations, DINO PCA, and depth images for each frame.
    
    Args:
        rgb_frames: RGB frames as numpy array
        depth_frames: Depth frames as numpy array
        video_frame_data: VideoFrameData1 containing processed object information including manipulation status
        manip_metadata: ManipulationTimeline containing manipulation information
        output_dir: Directory to save visualizations
        frame_indices: List of frame indices to process
        camera_intrinsics: 3x3 camera intrinsics matrix for 3D to 2D projection
        verbose: Whether to print progress
        save_npys: Whether we want to save some of the raw data
        pca_subsample_frame_idx: If provided, fit PCA only on this frame's embeddings for faster computation
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"save_npys: {save_npys}")
    
    # Fit PCA once for all frames (major performance optimization)
    # If pca_subsample_frame_idx is provided, fit PCA only on that frame's embeddings
    t_pca_global = time.time()
    global_pca = fit_global_dino_pca(video_frame_data, frame_indices, pca_subsample_frame_idx)
    print(f"Global PCA took: {time.time() - t_pca_global}")
    
    # Track manipulation paths drawn for summary
    manipulation_paths_drawn = 0

    for i, frame_idx in tqdm(enumerate(frame_indices), desc="Creating Visualizations", unit="frame", 
                         total=len(frame_indices), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        frame_data = video_frame_data.frames[i]
        rgb_frame = rgb_frames[frame_idx]
        depth_frame = depth_frames[frame_idx]
        manip_mask = frame_data["manip_mask"].to_dense().numpy()
        sam_masks = frame_data["sam_masks"]
        
        if frame_data["sam_masks"].shape[0] == 0:
            print(f"ERROR: Frame {frame_idx} has no objects! sam_masks shape: {sam_masks.shape}")
            print(f"Frame data type: {type(sam_masks)}")
            print(f"Frame data device: {sam_masks.device if hasattr(sam_masks, 'device') else 'N/A'}")
            print(f"Frame data content: {sam_masks}")
            assert False, f"No objects in frame {frame_idx}"

        # Initialize masks and colors for this frame
        mask_shape = sam_masks[0].shape
        depth_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_depth_visualization.png")
        create_depth_visualization(depth_frame, depth_out_path)            

        manip_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_manipulation_masks.png")
        # Check if manip_mask is empty and create a default mask if needed
        if manip_mask.size == 0 or np.all(manip_mask == 0):
            # Create an empty mask with the same shape as other masks
            manip_mask = np.zeros(mask_shape, dtype=np.uint8)
        
        if manip_mask.ndim == 3:
            manip_mask = manip_mask.squeeze()
        success = cv2.imwrite(manip_out_path, manip_mask.astype(np.uint8) * 255)
        if not success:
            print(f"manip_mask write success {success}")

        if save_npys:
            save_frame_npys(
                output_dir=output_dir,
                frame_idx=frame_idx,
                rgb_frame=rgb_frame,
                depth_frame=depth_frame,
                frame_data=frame_data,
            )
        
        color_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)
        object_colors = {}
        for obj_idx in range(sam_masks.shape[0]):
            mask = sam_masks[obj_idx].numpy()
            manipulated = True if obj_idx == frame_data["manip_idx"] else False

            # Assign a color: red for manipulated, otherwise unique color
            if obj_idx not in object_colors:
                # Use HSV to get visually distinct colors
                idx = len(object_colors)
                hue = (idx * 0.61803398875) % 1.0  # Golden ratio for color spacing
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
                color = tuple(int(255 * c) for c in rgb)
                # Avoid red for non-manipulated
                if color == (255, 0, 0):
                    color = (0, 255, 0)
                if manipulated:
                    color = (255, 0, 0)
                object_colors[obj_idx] = color

            # Overlay the mask with the color
            mask_bool = mask.astype(bool)
            for c in range(3):
                color_mask[..., c][mask_bool] = color[c]

        # Ensure manipulated object is drawn last (on top) in red
        if frame_data["manip_idx"] != -1:
            manip_mask_top = sam_masks[frame_data["manip_idx"]].numpy().astype(bool)
            if manip_mask_top.any():
                color = (255, 0, 0)  # red
                color_mask[..., 0][manip_mask_top] = color[0]
                color_mask[..., 1][manip_mask_top] = color[1]
                color_mask[..., 2][manip_mask_top] = color[2]

        # Overlay the color mask on the RGB frame
        overlay = rgb_frame.copy()
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)

        # Draw manipulation paths
        overlay = draw_manipulation_paths_on_frame(
            frame_idx=frame_idx,
            manip_metadata=manip_metadata,
            overlay_image=overlay,
            camera_intrinsics=camera_intrinsics,
            line_thickness=3
        )
        # Count if any manipulation paths were drawn for this frame
        for start_idx, end_idx in manip_metadata.manipulation_periods:
            if start_idx <= frame_idx <= end_idx:
                manipulation_paths_drawn += 1
                break

        # Save the visualization
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_objects.png")
        success = cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            print(f"overlay write success {success}")

        out_path_masks = os.path.join(output_dir, f"frame_{frame_idx:04d}_mask_visualization.png")
        success = cv2.imwrite(out_path_masks, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        if not success:
            print(f"out_path_masks write success {success}")

        # Create dataset inclusion visualization
        create_dataset_inclusion_visualization(frame_data, rgb_frame, frame_idx, output_dir)
        
        # Create DINO PCA visualization
        create_dino_pca_visualization_with_labels(frame_data, sam_masks, rgb_frame, frame_idx, output_dir, save_npys, global_pca)

    del manip_mask
    del sam_masks
    gc.collect()
        
    # Print summary of manipulation paths drawn
    if manipulation_paths_drawn > 0:
        print(f"✅ Drew manipulation paths on {manipulation_paths_drawn} frames")
    else:
        print("ℹ️  No manipulation paths drawn (no frames within manipulation periods or invalid projections)")
    
    # Print summary of dataset inclusion visualizations created
    dataset_inclusion_files = [f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith("_in_dataset.png")]
    print(f"✅ Created {len(dataset_inclusion_files)} dataset inclusion visualizations")

def create_dataset_inclusion_visualization(frame_data, rgb_frame, frame_idx: int, output_dir: str):
    """
    Create a visualization highlighting objects that are included in the training dataset.
    Objects in the dataset are highlighted in cyan.
    
    Args:
        frame_data: Frame data containing object information and dataset inclusion flags
        rgb_frame: RGB frame as numpy array
        frame_idx: Frame index for naming
        output_dir: Directory to save the visualization
        
    Returns:
        str: Path to the saved image
    """
    sam_masks = frame_data["sam_masks"]
    
    if sam_masks.shape[0] == 0:
        print(f"Warning: Frame {frame_idx} has no objects for dataset inclusion visualization")
        return None
    
    # Create color mask for dataset inclusion
    mask_shape = sam_masks[0].shape
    color_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)

    # Build flags once
    in_flags = []
    no_risk_flags = []
    for obj_idx in range(sam_masks.shape[0]):
        in_flag = False
        no_risk_flag = False
        if "in_dataset" in frame_data and obj_idx < len(frame_data["in_dataset"]):
            in_flag = frame_data["in_dataset"][obj_idx].item()
        if "no_risk" in frame_data and obj_idx < len(frame_data["no_risk"]):
            no_risk_flag = frame_data["no_risk"][obj_idx].item()
        in_flags.append(in_flag)
        no_risk_flags.append(no_risk_flag)

    # Pass 3: paint no_risk flags yellow 
    for obj_idx in range(sam_masks.shape[0]):
        if not no_risk_flags[obj_idx]:
            continue
        mask = sam_masks[obj_idx].numpy()
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        color_mask[..., 0][mask_bool] = 255
        color_mask[..., 1][mask_bool] = 255
        color_mask[..., 2][mask_bool] = 0

    # Pass 1: paint non-dataset objects first (transparent/black)
    for obj_idx in range(sam_masks.shape[0]):
        if in_flags[obj_idx]:
            continue
        mask = sam_masks[obj_idx].numpy()
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        # Black (transparent when blended)
        color_mask[..., 0][mask_bool] = 0
        color_mask[..., 1][mask_bool] = 0
        color_mask[..., 2][mask_bool] = 0

    # Pass 2: paint in-dataset objects last (cyan) so they sit on top
    for obj_idx in range(sam_masks.shape[0]):
        if not in_flags[obj_idx] or no_risk_flags[obj_idx]:
            continue
        mask = sam_masks[obj_idx].numpy()
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        color_mask[..., 0][mask_bool] = 0
        color_mask[..., 1][mask_bool] = 255
        color_mask[..., 2][mask_bool] = 255

    # Get indices where no_risk_flags is True
    no_risk_indices = [i for i, flag in enumerate(no_risk_flags) if flag]
    in_dataset_indices = [i for i, flag in enumerate(in_flags) if flag]
    
    # Overlay the color mask on the RGB frame
    overlay = rgb_frame.copy()
    alpha = 0.6  # Slightly more opaque for better visibility
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    
    # Add object ID labels on each mask
    for obj_idx in range(sam_masks.shape[0]):
        mask = sam_masks[obj_idx].numpy()
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        
        # Calculate centroid of the mask
        y_coords, x_coords = np.where(mask_bool)
        if len(y_coords) > 0 and len(x_coords) > 0:
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            
            # Add object ID text at centroid
            text = f"ID:{obj_idx}"
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background rectangle for better text visibility
            cv2.rectangle(overlay, 
                         (centroid_x - text_width//2 - 2, centroid_y - text_height - 2),
                         (centroid_x + text_width//2 + 2, centroid_y + baseline + 2),
                         (0, 0, 0), -1)  # Black background
            
            # Draw the object ID text
            cv2.putText(overlay, text, (centroid_x - text_width//2, centroid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Add text labels for clarity
    cv2.putText(overlay, f"Frame {frame_idx}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Cyan = In Dataset {in_dataset_indices}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, f"Yellow = No Risk {no_risk_indices}", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save the visualization
    out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_in_dataset.png")
    success = cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        print(f"Dataset inclusion visualization write failed for frame {frame_idx}")
        return None
    
    return out_path

# TODO: Break up this function 
def save_frame_npys(output_dir: str, frame_idx: int, rgb_frame: np.ndarray, depth_frame: np.ndarray,
                    frame_data, verbose=False) -> None:
    """
    Save per-frame debug artifacts when save_npys=True.
    Includes: rgb.npy, depth.npy, manipulation_mask.npy, mask_info.pkl, mask_ids_img.npy, mask_ids_viz.png
    """
    if verbose:
        print(f"[save_frame_npys] frame={frame_idx} -> start; output_dir='{output_dir}'")
        print(f"[save_frame_npys] rgb shape={rgb_frame.shape}, dtype={rgb_frame.dtype}; depth shape={depth_frame.shape}, dtype={depth_frame.dtype}")

    manip_mask = frame_data["manip_mask"].to_dense().numpy()
    if manip_mask.size == 0 or np.all(manip_mask == 0):
            manip_mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
    if manip_mask.ndim == 3:
        manip_mask = manip_mask.squeeze()
    if verbose:
        unique_vals = np.unique(manip_mask)
        print(f"[save_frame_npys] manip_mask shape={manip_mask.shape}, dtype={manip_mask.dtype}, unique={unique_vals.tolist()}")

    sam_masks = frame_data["sam_masks"]
    if verbose:
        print(f"[save_frame_npys] num SAM masks={sam_masks.shape[0] if hasattr(sam_masks, 'shape') else 'N/A'}")

    existed_before = os.path.exists(output_dir)
    if not existed_before:
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"[save_frame_npys] created directory: {output_dir}")
    else:
        if verbose:
            print(f"[save_frame_npys] using existing directory: {output_dir}")

    # Basic npys
    rgb_npy_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_rgb.npy")
    np.save(rgb_npy_out_path, rgb_frame)
    if verbose:
        print(f"[save_frame_npys] saved: {rgb_npy_out_path} ({rgb_frame.shape}, {rgb_frame.dtype})")

    depth_npy_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_depth.npy")
    np.save(depth_npy_out_path, depth_frame)
    if verbose:
        print(f"[save_frame_npys] saved: {depth_npy_out_path} ({depth_frame.shape}, {depth_frame.dtype})")

    manip_npy_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_manipulation_mask.npy")
    np.save(manip_npy_out_path, manip_mask)
    if verbose:
        print(f"[save_frame_npys] saved: {manip_npy_out_path} ({manip_mask.shape}, {manip_mask.dtype})")

    # Mask info and ids
    mask_shape = sam_masks[0].shape if sam_masks.shape[0] > 0 else depth_frame.shape
    mask_info_list = []
    mask_id_img = np.full((mask_shape[0], mask_shape[1]), -1, dtype=np.int8)

    for obj_idx in range(sam_masks.shape[0]):
        # Skip if this mask is empty
        if sam_masks[obj_idx].numel() == 0 or not sam_masks[obj_idx].any():
            continue
        manipulated = (obj_idx == frame_data["manip_idx"])
        sam_mask = sam_masks[obj_idx].numpy()
        mask_info_list.append([manipulated, sam_mask])

        mask_bool = sam_mask if sam_mask.dtype == bool else sam_mask.astype(bool)
        mask_id_img[mask_bool] = obj_idx

    with open(os.path.join(output_dir, f"frame_{frame_idx:04d}_mask_info.pkl"), "wb") as f:
        pickle.dump(mask_info_list, f)
    if verbose:
        print(f"[save_frame_npys] saved: {os.path.join(output_dir, f'frame_{frame_idx:04d}_mask_info.pkl')} (num_masks={len(mask_info_list)}, manip_idx={frame_data['manip_idx']})")
    np.save(os.path.join(output_dir, f"frame_{frame_idx:04d}_mask_ids_img.npy"), mask_id_img)
    if verbose:
        unique_ids = np.unique(mask_id_img)
        valid_ids = unique_ids[unique_ids != -1]
        print(f"[save_frame_npys] saved: {os.path.join(output_dir, f'frame_{frame_idx:04d}_mask_ids_img.npy')} (unique_valid_ids={valid_ids.tolist()})")

    # Create normalized visualization of mask IDs
    mask_id_img_viz = mask_id_img.copy()
    mask_id_img_viz[mask_id_img_viz == -1] = 0
    unique_mask_ids = np.unique(mask_id_img_viz[mask_id_img_viz != 0])

    if len(unique_mask_ids) > 0:
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_mask_ids, start=1)}
        for old_id, new_id in id_mapping.items():
            mask_id_img_viz[mask_id_img_viz == old_id] = new_id

        plt.figure(figsize=(10, 8))
        im = plt.imshow(mask_id_img_viz, cmap='tab20', interpolation='nearest')
        plt.title(f"Mask IDs for frame {frame_idx:04d} (Normalized)")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Normalized Mask ID")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"frame_{frame_idx:04d}_mask_ids_viz.png"), dpi=150, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"[save_frame_npys] saved: {os.path.join(output_dir, f'frame_{frame_idx:04d}_mask_ids_viz.png')} (normalized ID visualization)")
    else:
        print(f"Warning: No valid mask IDs found for frame {frame_idx}")

def save_dino_pca_npy(output_dir: str, frame_idx: int, pca_img: np.ndarray) -> None:
    """Save DINO PCA visualization as .npy for the given frame."""
    pca_npy_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_DINO_pca.npy")
    np.save(pca_npy_out_path, pca_img)

def visualize_frames_video(output_dir: str, frame_indices: list, fps: int = 10):
    """
    Create a video from saved visualization images with 4 quadrants per frame.
    
    Args:
        output_dir: Directory containing the saved visualization images
        frame_indices: List of frame indices to process
        fps: Frames per second for the output video
    """

    # Check if we have any frames to process
    if not frame_indices:
        print("No frame indices provided")
        return
    
    # Find the first frame to determine image dimensions
    first_frame_idx = frame_indices[0]
    
    # Load one image to get dimensions
    sample_path = os.path.join(output_dir, f"frame_{first_frame_idx:04d}_objects.png")
    if not os.path.exists(sample_path):
        print(f"Sample image not found: {sample_path}")
        return
    
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        print(f"Could not load sample image: {sample_path}")
        return
    
    h, w = sample_img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    video_path = os.path.join(output_dir, "combined_visualization.mp4")
    seq_dir = os.path.join(output_dir, "video_seq_tmp")
    os.makedirs(seq_dir, exist_ok=True)

    print(f"Video dimensions: {w}x{h}, FPS: {fps}")
    for frame_idx in tqdm(frame_indices, desc="Creating Video", unit="frame"):
        # Load the visualization images
        objects_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_objects.png")
        mask_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_mask_visualization.png")
        manip_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_manipulation_masks.png")
        dino_pca_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_dino_pca_visualization.png")
        dataset_inclusion_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_in_dataset.png")
        
        # Load images (use black image if not found)
        objects_img = cv2.imread(objects_path) if os.path.exists(objects_path) else np.zeros((h, w, 3), dtype=np.uint8)
        mask_img = cv2.imread(mask_path) if os.path.exists(mask_path) else np.zeros((h, w, 3), dtype=np.uint8)
        manip_img = cv2.imread(manip_path) if os.path.exists(manip_path) else np.zeros((h, w, 3), dtype=np.uint8)
        dino_pca_img = cv2.imread(dino_pca_path) if os.path.exists(dino_pca_path) else np.zeros((h, w, 3), dtype=np.uint8)
        dataset_inclusion_img = cv2.imread(dataset_inclusion_path) if os.path.exists(dataset_inclusion_path) else np.zeros((h, w, 3), dtype=np.uint8)
        
        # Resize images to quadrant size if needed
        quadrant_h, quadrant_w = h // 2, w // 2
        
        objects_img = cv2.resize(objects_img, (quadrant_w, quadrant_h))
        mask_img = cv2.resize(mask_img, (quadrant_w, quadrant_h))
        manip_img = cv2.resize(manip_img, (quadrant_w, quadrant_h))
        dino_pca_img = cv2.resize(dino_pca_img, (quadrant_w, quadrant_h))
        dataset_inclusion_img = cv2.resize(dataset_inclusion_img, (quadrant_w, quadrant_h))
        
        # Create the combined frame with 4 quadrants
        combined_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Upper left: Dataset inclusion visualization
        combined_frame[:quadrant_h, :quadrant_w] = dataset_inclusion_img
        
        # Upper right: Manipulation masks
        combined_frame[:quadrant_h, quadrant_w:] = manip_img
        
        # Lower left: DINO PCA visualization
        combined_frame[quadrant_h:, :quadrant_w] = dino_pca_img
        
        # Lower right: Objects visualization
        combined_frame[quadrant_h:, quadrant_w:] = objects_img
        
        # Add frame number text
        cv2.putText(combined_frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add quadrant labels
        cv2.putText(combined_frame, "Dataset Inclusion (Cyan=In Dataset)", (10, quadrant_h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_frame, "Manipulation Masks", (quadrant_w + 10, quadrant_h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_frame, "DINO PCA", (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_frame, "SAM MASK RGB Overlay (Red=Manip)", (quadrant_w + 10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame to video
        seq_idx = len([n for n in os.listdir(seq_dir) if n.endswith(".png")])
        cv2.imwrite(os.path.join(seq_dir, f"frame_{seq_idx:06d}.png"), combined_frame)
    
    # after the loop:
    inp = os.path.join(seq_dir, "frame_%06d.png")
    # try libx264 first, fall back to libopenh264 (you have openh264 per ffmpeg -encoders)
    cmds = [
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", inp,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_path],
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", inp,
         "-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_path],
    ]
    rc = 1
    for cmd in cmds:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            rc = 0
            break
    if rc != 0:
        raise RuntimeError("ffmpeg H.264 encoding failed")
    shutil.rmtree(seq_dir, ignore_errors=True)
    print(f"Video saved to: {video_path}")

def point_to_line_distance(point: tuple[float, float, float], line_rep: StraightLine) -> float:
    """
    Calculate the shortest distance from a point to a line.
    
    Args:
        point: 3D point (x, y, z)
        line_rep: StraightLine representation from create_straight_line_representation
        
    Returns:
        float: Distance from point to line, or float('inf') if point projects beyond line segment
    """
    point = np.array(point)
    
    # For line segments, we need to check if the closest point is within the segment
    start = line_rep.start
    direction = line_rep.direction
    
    # Vector from start to point
    start_to_point = point - start
    
    # Projection parameter t
    t = np.dot(start_to_point, direction)
    
    # Check if point projects beyond line segment bounds
    if t < 0 or t > line_rep.length:
        return float('inf')
    
    # Closest point on line segment
    closest_point = start + t * direction
    
    # Distance from point to closest point on line
    return np.linalg.norm(point - closest_point)

def check_objects_near_line(video_frame_data: VideoFrameData1, frame_idx: int, line_rep: StraightLine, radius: float = 0.1) -> list[int]:
    """
    Find all objects whose centroids are within radius r of the straight line.
    
    Args:
        video_frame_data: Video frame data containing object centroids
        frame_idx: Frame index to check
        line_rep: StraightLine representation from create_straight_line_representation
        radius: Radius around the line to check (default: 0.1 meters)
        
    Returns:
        list[int]: List of object IDs whose centroids are within radius r of the straight line
    """
    
    near_objects = []
    for obj_id, frameObjectData in video_frame_data.frames[frame_idx].items():
        
        distance = point_to_line_distance(frameObjectData.centroid_xyz, line_rep)
        if distance <= radius:
                near_objects.append(obj_id)           
            
    return near_objects

def create_straight_line_representation(start_pt: torch.Tensor, end_pt: torch.Tensor) -> StraightLine:
    """
    Create a straight line representation between two 3D points.
    
    Args:
        start_pt: Starting point (x, y, z)
        end_pt: Ending point (x, y, z)
        
    Returns:
        StraightLine: Pydantic model representing the straight line
        
    Raises:
        ValueError: If start and end points are identical
    """
    
    # Direction vector
    direction = end_pt - start_pt
    direction_norm = torch.linalg.vector_norm(direction)
    
    if direction_norm == 0:
        raise ValueError("Start and end points cannot be identical")
    
    # Normalized direction vector
    unit_direction = direction / direction_norm
    
    return StraightLine(
        start=start_pt,
        end=end_pt,
        direction=unit_direction,
        length=direction_norm
    )

def unit_direction_vector(start_pts: torch.Tensor, end_pts: torch.Tensor) -> torch.Tensor:
    """
    Calculate the unit direction vector between n 3D points. 
    Args:
        start_pts: [N,D] tensor of points along the path
        end_pts: [N,D] tensor of points along the path
    """
    assert start_pts.shape == end_pts.shape, f"input shapes not matching {start_pts.shape} != {end_pts.shape}"

    directions = end_pts - start_pts
    direction_norms = torch.linalg.vector_norm(directions, dim=1)
    direction_norms = direction_norms.unsqueeze(1)

    # Handle zero-length vectors (division by zero)
    unit_direction_vectors = torch.zeros_like(directions)
    non_zero_mask = direction_norms.squeeze(1) > 0
    if non_zero_mask.any():
        unit_direction_vectors[non_zero_mask] = directions[non_zero_mask] / direction_norms[non_zero_mask]
    
    return unit_direction_vectors

def draw_manipulation_paths_on_frame(frame_idx: int, manip_metadata: ManipulationTimeline, 
                                   overlay_image: np.ndarray, camera_intrinsics: np.ndarray,
                                   line_thickness: int = 3) -> np.ndarray:
    """
    Draw manipulation straight line paths on a frame if it's within a manipulation period.
    
    Args:
        frame_idx: Current frame index
        manip_metadata: ManipulationTimeline containing manipulation periods and straight lines
        overlay_image: RGB image to draw on (will be modified in place)
        camera_intrinsics: 3x3 camera intrinsics matrix
        line_thickness: Thickness of the line to draw
        
    Returns:
        np.ndarray: Modified overlay image with cyan lines drawn
    """
    # Check if this frame is within any manipulation period
    for i, (start_idx, end_idx) in enumerate(manip_metadata.manipulation_periods):
        if start_idx <= frame_idx <= end_idx:
            # This frame is within a manipulation period, draw the straight line
            if i >= len(manip_metadata.straight_lines):
                print(f"Warning: Frame {frame_idx} in manipulation period {i} but no corresponding straight line")
                continue
            straight_line = manip_metadata.straight_lines[i]
            
            # Project 3D line endpoints to 2D pixel coordinates
            start_3d = straight_line.start
            end_3d = straight_line.end
            
            # Extract camera intrinsics
            if camera_intrinsics.shape != (3, 3):
                print(f"Warning: Invalid camera intrinsics shape {camera_intrinsics.shape}, expected (3, 3)")
                return overlay_image
            fx = camera_intrinsics[0, 0]
            fy = camera_intrinsics[1, 1]
            cx = camera_intrinsics[0, 2]
            cy = camera_intrinsics[1, 2]
            
            # Project 3D points to 2D pixels
            def project_3d_to_2d(point_3d):
                x, y, z = point_3d
                if z <= 0:  # Point behind camera
                    return None
                u = int((x * fx / z) + cx)
                v = int((y * fy / z) + cy)
                return (u, v)
            
            start_2d = project_3d_to_2d(start_3d)
            end_2d = project_3d_to_2d(end_3d)
            
            # Only draw if both points are valid (in front of camera)
            if start_2d is not None and end_2d is not None:
                # Check if points are within image bounds
                h, w = overlay_image.shape[:2]
                if (0 <= start_2d[0] < w and 0 <= start_2d[1] < h and 
                    0 <= end_2d[0] < w and 0 <= end_2d[1] < h):
                    
                    # Draw cyan line (BGR format: cyan = (255, 255, 0))
                    cv2.line(overlay_image, start_2d, end_2d, (255, 255, 0), line_thickness)
                    
                    # Draw small circles at endpoints for better visibility
                    cv2.circle(overlay_image, start_2d, 5, (255, 255, 0), -1)  # Filled circle
                    cv2.circle(overlay_image, end_2d, 5, (255, 255, 0), -1)    # Filled circle
    
    return overlay_image

def get_min_distances(points_query, points_reference, tree=None):

    points_query_f32 = np.ascontiguousarray(points_query, dtype='float32')
    points_reference_f32 = np.ascontiguousarray(points_reference, dtype='float32')

    tree = faiss.IndexFlatL2(points_reference_f32.shape[1]).add(points_reference_f32) if not tree else tree
    faiss.omp_set_num_threads(6)  
    
    D, _ = tree.search(points_query, 1)
    min_distances_environment = np.sqrt(D).squeeze()

    index = faiss.IndexFlatL2(points_query_f32.shape[1])
    index.add(points_query_f32)
    
    # Query nearest neighbor for each point
    D, _ = index.search(points_reference_f32, 1)  # D is squared L2 distance
    min_distances_manipulated = np.sqrt(D).squeeze()

    return min_distances_environment, min_distances_manipulated

def get_images_config(img_list_dir: str) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all images from a directory and return as numpy arrays with filenames.
    
    Args:
        img_list_dir: Directory path containing images
        
    Returns:
        tuple: (images_list, filenames_list) where images_list contains RGB numpy arrays
               and filenames_list contains filenames without extensions
    """
    images_list = []
    filenames_list = []
    
    if os.path.exists(img_list_dir):
        for filename in os.listdir(img_list_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(img_list_dir, filename)
                # Load image as numpy array
                img = cv2.imread(image_path)
                if img is not None:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images_list.append(img_rgb)
                    # Strip file extension from filename
                    filename_no_ext = os.path.splitext(filename)[0]
                    filenames_list.append(filename_no_ext)
                else:
                    print(f"Warning: Could not load image {image_path}")
        print(f"Found {len(images_list)} images in {img_list_dir}")
    else:
        print(f"Warning: Directory {img_list_dir} does not exist")
    
    return images_list, filenames_list

def load_process_video_config(config_path: str) -> dict:
    """
    Load process_video configuration from a YAML file and expand placeholders.

    Supported expansions:
      - ~ and environment variables via os.path.expanduser/os.path.expandvars
      - {DATE_EPOCH} → "<MM_DD_YYYY>_<epoch_seconds>" for strings like dataset_output_dir

    Returns a dict with sensible defaults for optional fields.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        text = f.read()
        ext = os.path.splitext(config_path)[1].lower()
        if yaml is None:
            raise ImportError("PyYAML is not installed.")
        cfg = yaml.safe_load(text)

    def expand(value):
        if isinstance(value, str):
            v = os.path.expanduser(os.path.expandvars(value))
            if "{DATE_EPOCH}" in v:
                now = datetime.now()
                date = now.strftime("%m_%d_%Y")
                epoch = str(int(now.timestamp()))
                v = v.replace("{DATE_EPOCH}", f"{date}_{epoch}")
            if "{PETNAME}" in v:
                petname = PetnamePkg.generate()
                v = v.replace("{PETNAME}", petname)
            return v
        if isinstance(value, list):
            return [expand(x) for x in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    cfg = expand(cfg)

    # Minimal validation
    required_keys = ["npz_paths", "camera_intrinsics_path", "output_dir", "object_white_list", "object_black_list"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    white_list_dir = cfg["object_white_list"] #"data/dino_reference_ims/white_list"
    black_list_dir = cfg["object_black_list"] #"data/dino_reference_ims/black_list"
    grey_list_dir  = cfg["object_grey_list"]  #"data/dino_reference_ims/grey_list"
    
    white_list_images, white_list_filenames = get_images_config(white_list_dir)
    black_list_images, black_list_filenames = get_images_config(black_list_dir)
    grey_list_images, grey_list_filenames = get_images_config(grey_list_dir)
    
    cfg["white_list_images"] = white_list_images
    cfg["black_list_images"] = black_list_images
    cfg["grey_list_images"]  = grey_list_images
    cfg["white_list_filenames"] = white_list_filenames
    cfg["black_list_filenames"] = black_list_filenames
    cfg["grey_list_filenames"]  = grey_list_filenames

    return cfg

def create_dino_pca_visualization_with_labels(frame_data, sam_masks, rgb_frame, frame_idx, output_dir, save_npys=False, global_pca=None):
    """
    Minimal DINO PCA visualization: transform embeddings → color masks → simple labels → save.
    Requires a pre-fitted PCA (global_pca). Falls back to a blank image if unavailable.
    """

    h, w = rgb_frame.shape[:2]

    assert sam_masks is not None and sam_masks.shape[0] > 0, f"No SAM masks found for frame {frame_idx}"
    dino_embeddings = frame_data.get("dino_embeddings")
    assert dino_embeddings is not None and dino_embeddings.shape[0] > 0, f"No DINO embeddings found for frame {frame_idx}"
    assert global_pca is not None, f"Global PCA is required for frame {frame_idx}"

    # Transform embeddings to 3D and map to RGB via sigmoid in [0,255]
    embeddings_np = dino_embeddings.cpu().numpy()
    embeddings_3d = global_pca.transform(embeddings_np)  # [N,3]
    colors01 = 1.0 / (1.0 + np.exp(-2.0 * embeddings_3d))  # sigmoid(2x)
    colors255 = (colors01 * 255).astype(np.uint8)

    # Paint each object's mask with its PCA color
    pca_img = np.zeros((h, w, 3), dtype=np.uint8)
    masks_np = sam_masks.numpy()
    n = min(masks_np.shape[0], colors255.shape[0])
    for i in range(n):
        m = masks_np[i].astype(bool)
        if m.any():
            pca_img[m] = colors255[i]

    # Simple labels at mask centroids (no overlap handling)
    labels = frame_data.get("object_labels", [])
    for i in range(n):
        ys, xs = np.nonzero(masks_np[i])
        if ys.size == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        text = labels[i] if i < len(labels) else f"obj{i}"
        cv2.putText(pca_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(pca_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Save
    pca_out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_dino_pca_visualization.png")
    cv2.imwrite(pca_out_path, cv2.cvtColor(pca_img, cv2.COLOR_RGB2BGR))
    if save_npys:
        save_dino_pca_npy(output_dir, frame_idx, pca_img)
    return pca_out_path

def is_frame_in_any_manipulation_period(frame_idx: int, manip_metadata: ManipulationTimeline) -> bool:
    """
    Check if a frame is within any manipulation period.
    """
    if manip_metadata is None or not manip_metadata.manipulation_periods:
        return False
    for start_frame, end_frame in manip_metadata.manipulation_periods:
        if start_frame <= frame_idx <= end_frame:
            return True
    return False

def are_frames_in_same_manipulation_period(frame_idx_1: int, frame_idx_2: int, manip_metadata: ManipulationTimeline) -> bool:
    """
    Check if two frames are within the same manipulation period.
    
    Args:
        frame_idx_1: First frame index
        frame_idx_2: Second frame index  
        manip_metadata: ManipulationTimeline containing manipulation periods
        
    Returns:
        True if both frames are in the same manipulation period, False otherwise
    """
    if manip_metadata is None or not manip_metadata.manipulation_periods:
        return False
        
    for start_frame, end_frame in manip_metadata.manipulation_periods:
        if (start_frame <= frame_idx_1 <= end_frame and 
            start_frame <= frame_idx_2 <= end_frame):
            return True
    return False

def sample_points_along_line(straight_line: StraightLine, n_points: int = 100) -> torch.Tensor:
    """
    Sample n_points along a straight line between start and end points.
    
    Args:
        straight_line: StraightLine object with start, end, direction, and length
        n_points: Number of points to sample along the line
        
    Returns:
        torch.Tensor: [n_points, 3] tensor of sampled points along the line
    """
    device = straight_line.start.device
    t = torch.linspace(0, 1, n_points, device=device)
    points = straight_line.start.unsqueeze(0) + t.unsqueeze(1) * (straight_line.end - straight_line.start).unsqueeze(0)
    
    return points

def get_image_embedd_white_black_lst(dino_featurizer: DinoFeatures, white_list_ims: list[np.ndarray], black_list_ims: list[np.ndarray],
                                    grey_list_ims: list[np.ndarray], white_list_filenames: list[str], black_list_filenames: list[str],  
                                    grey_list_filenames: list[str], target_h=500):

    white_lst_avg_dino = []
    white_labels = []
    for i, white_im in enumerate(tqdm(white_list_ims, desc="Processing white list images")):
        rgb_tensor = dino_featurizer.rgb_image_preprocessing(white_im, target_h)
        dino_features = dino_featurizer.get_dino_features_rgb(rgb_tensor)
        # Compute average DINO feature across spatial dimensions [C, H, W] -> [C]
        avg_dino_feature = torch.mean(dino_features, dim=(1, 2))  # Average over H and W dimensions -> [C]
        white_lst_avg_dino.append(avg_dino_feature.cpu())
        white_labels.append(white_list_filenames[i])
        rgb_tensor, dino_features, avg_dino_feature = None, None, None
        del rgb_tensor, dino_features, avg_dino_feature

    # Stack all white list DINO features into a single tensor
    if white_lst_avg_dino:
        white_lst_avg_dino = torch.stack(white_lst_avg_dino)  # [N, C] where N is number of white list images
    else:
        white_lst_avg_dino = torch.empty(0, 1024)  # Empty tensor with correct feature dimension
        white_labels = []

    black_lst_avg_dino = []
    black_labels = []
    for i, black_im in enumerate(tqdm(black_list_ims, desc="Processing black list images")):
        rgb_tensor = dino_featurizer.rgb_image_preprocessing(black_im, target_h)
        dino_features = dino_featurizer.get_dino_features_rgb(rgb_tensor)
        # Compute average DINO feature across spatial dimensions [C, H, W] -> [C]
        avg_dino_feature = torch.mean(dino_features, dim=(1, 2))  # Average over H and W dimensions -> [C]
        black_lst_avg_dino.append(avg_dino_feature.cpu())
        black_labels.append(black_list_filenames[i])
        rgb_tensor, dino_features, avg_dino_feature = None, None, None
        del rgb_tensor, dino_features, avg_dino_feature
    
    # Stack all black list DINO features into a single tensor
    if black_lst_avg_dino:
        black_lst_avg_dino = torch.stack(black_lst_avg_dino)  # [N, C] where N is number of black list images
    else:
        black_lst_avg_dino = torch.empty(0, 1024)  # Empty tensor with correct feature dimension
        black_labels = []

    grey_lst_avg_dino = []
    grey_labels = []
    for i, grey_im in enumerate(tqdm(grey_list_ims, desc="Processing grey list images")):
        rgb_tensor = dino_featurizer.rgb_image_preprocessing(grey_im, target_h)
        dino_features = dino_featurizer.get_dino_features_rgb(rgb_tensor)
        # Compute average DINO feature across spatial dimensions [C, H, W] -> [C]
        avg_dino_feature = torch.mean(dino_features, dim=(1, 2))  # Average over H and W dimensions -> [C]
        grey_lst_avg_dino.append(avg_dino_feature.cpu())
        grey_labels.append(grey_list_filenames[i])
        rgb_tensor, dino_features, avg_dino_feature = None, None, None
        del rgb_tensor, dino_features, avg_dino_feature
    
    # Stack all grey list DINO features into a single tensor
    if grey_lst_avg_dino:
        grey_lst_avg_dino = torch.stack(grey_lst_avg_dino)  # [N, C] where N is number of grey list images
    else:
        grey_lst_avg_dino = torch.empty(0, 1024)  # Empty tensor with correct feature dimension
        grey_labels = []

    return white_lst_avg_dino, black_lst_avg_dino, grey_lst_avg_dino, white_labels, black_labels, grey_labels

def get_manipulation_periods(manip_metadata: ManipulationTimeline, video_frame_data: VideoFrameData1, frame_indices: list[int]):

    # Calculate manipulation timeline in one loop
    manip_metadata.manipulation_timeline = [
        1 if video_frame_data.frames[i]["manip_idx"] != -1 else 0
        for i in range(len(video_frame_data.frames))
    ]

    num_manip_idxs = sum(manip_metadata.manipulation_timeline)
    print(f"Number of manipulated frames {num_manip_idxs} vs total number {len(video_frame_data.frames)}")

    # Use sliding window to fill in gaps
    for i, frame_idx in enumerate(frame_indices):
        i_prev = i - 1
        i_next = i + 1
        if i_prev < 0:
            continue 
        elif i_next >= len(frame_indices):
            break

        # Check if we have manipulation on both sides of a gap
        if (manip_metadata.manipulation_timeline[i_prev] == 1 and 
            manip_metadata.manipulation_timeline[i_next] == 1 and 
            manip_metadata.manipulation_timeline[i] == 0 and  # Current frame is a gap
            video_frame_data.frames[i_prev]["manip_idx"] == video_frame_data.frames[i_next]["manip_idx"]):
            
            # Get the consistent manip_idx from surrounding frames
            consistent_manip_idx = video_frame_data.frames[i_prev]["manip_idx"]
            
            # Check if the gap frame has a valid centroid for this object
            if (consistent_manip_idx != -1 and 
                consistent_manip_idx < len(video_frame_data.frames[i]["centroid_xyzs"]) and
                not torch.equal(video_frame_data.frames[i]["centroid_xyzs"][consistent_manip_idx], 
                               torch.tensor([-1.0, -1.0, -1.0], device=video_frame_data.frames[i]["centroid_xyzs"].device))):
                
                # Fill the gap
                video_frame_data.frames[i]["manip_idx"] = consistent_manip_idx
                video_frame_data.frames[i]["manip_centroid_xyz"] = video_frame_data.frames[i]["centroid_xyzs"][consistent_manip_idx]
                manip_metadata.manipulation_timeline[i] = 1
    
    # Calculate manipulation periods from timeline
    start_idx, end_idx = None, None 
    for i, frame_idx in enumerate(frame_indices):
        if manip_metadata.manipulation_timeline[i] == 1:
            if start_idx is None:
                start_idx = frame_idx
        else:
            if start_idx is not None:
                end_idx = frame_idx -1
                if start_idx != end_idx: # One frame not enough data to make a straight line
                    manip_metadata.manipulation_periods.append((start_idx, end_idx))
                start_idx, end_idx = None, None

    if start_idx is not None and start_idx != frame_indices[-1]:
        manip_metadata.manipulation_periods.append((start_idx, frame_indices[-1]))
        start_idx = None

    return manip_metadata

def save_sam_masks_as_images(frame_data: dict, output_dir: str, frame_idx: int = 0, rgb_frame: np.ndarray = None, verbose=False) -> None:
    """
    Save each SAM mask from frame data as a separate image file.
    If rgb_frame is provided, also save RGBA images with masked regions visible and non-masked regions transparent.
    
    Args:
        frame_data: Frame data dictionary containing SAM masks (as defined in getEmptyFrameData)
        output_dir: Directory to save the mask images
        frame_idx: Frame index for naming (default: 0)
        verbose: Whether to print progress messages
        rgb_frame: Optional RGB frame to create RGBA masked images (shape: [H, W, 3])
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sam_masks = frame_data["sam_masks"]
    assert sam_masks.shape[0] > 0, f"No SAM masks to save for frame {frame_idx}"
    
    if verbose:
        print(f"Saving {sam_masks.shape[0]} SAM masks for frame {frame_idx}")
    
    for mask_idx in range(sam_masks.shape[0]):
        mask = sam_masks[mask_idx]
        mask_np = mask.numpy()
        
        # Ensure mask is 2D
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Save the binary mask image
        filename = f"sam_mask_{frame_idx:04d}_{mask_idx:03d}.png"
        filepath = os.path.join(output_dir, filename)
        
        success = cv2.imwrite(filepath, mask_uint8)
        assert success, f"Failed to save SAM mask {mask_idx} to {filepath}"
        if verbose:
            print(f"Saved SAM mask {mask_idx} to {filepath}")
        
        # If RGB frame is provided, also save RGBA image with transparency
        if rgb_frame is not None:
            # Ensure RGB frame is the right shape and type
            if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
                raise ValueError(f"RGB frame must have shape [H, W, 3], got {rgb_frame.shape}")
            
            # Convert mask to boolean for indexing
            mask_bool = mask_np.astype(bool)
            
            # Create RGBA image: initialize with transparent background
            rgba_image = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1], 4), dtype=np.uint8)
            
            # Set RGB values where mask is True
            rgba_image[mask_bool, :3] = rgb_frame[mask_bool]
            
            # Set alpha channel: 255 where mask is True, 0 where False (transparent)
            rgba_image[mask_bool, 3] = 255
            
            # Save RGBA image
            rgba_filename = f"sam_mask_rgba_{frame_idx:04d}_{mask_idx:03d}.png"
            rgba_filepath = os.path.join(output_dir, rgba_filename)
            
            # Convert RGBA to BGRA for OpenCV
            bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
            success = cv2.imwrite(rgba_filepath, bgra_image)
            assert success, f"Failed to save RGBA SAM mask {mask_idx} to {rgba_filepath}"
            if verbose:
                print(f"Saved RGBA SAM mask {mask_idx} to {rgba_filepath}")

def get_percentage_overlap(mask: torch.Tensor, reference_mask: torch.Tensor) -> float:
    """
    Calculate the percentage that the mask overlaps with the reference mask.
    Pure PyTorch implementation for efficiency and GPU compatibility.
    
    Args:
        mask: Binary mask (torch tensor or numpy array) - the mask to analyze
        reference_mask: Binary mask (torch tensor or numpy array) - the reference mask
        
    Returns:
        float: Percentage overlap (0.0 to 1.0) where 1.0 means complete overlap
    """
    assert isinstance(mask, torch.Tensor), f"mask must be a torch.Tensor, got {type(mask)}"
    assert isinstance(reference_mask, torch.Tensor), f"reference_mask must be a torch.Tensor, got {type(reference_mask)}"
    device = mask.device
    assert reference_mask.device == device, f"Tensors must be on the same device: mask on {device}, reference_mask on {reference_mask.device}"
    
    mask_bool = mask.bool()
    reference_mask_bool = reference_mask.bool()
    
    # Calculate intersection (AND operation)
    intersection = torch.logical_and(mask_bool, reference_mask_bool)
    
    # Calculate areas
    mask_area = mask_bool.sum().float()
    intersection_area = intersection.sum().float()
    
    # Handle edge case where mask has no area
    if mask_area == 0:
        return 0.0
    
    # Calculate percentage overlap
    overlap_percentage = (intersection_area / mask_area).item()
    
    return overlap_percentage

def is_flat_plane(points_3d: np.ndarray, plane_threshold: float = 0.02, min_points: int = 100, 
                    inlier_ratio_thres: float = 0.6, verbose=False) -> bool:
    """
    Determine if a set of 3D points forms a flat plane using RANSAC plane fitting.
    
    Args:
        points_3d: Array of 3D points [N, 3]
        plane_threshold: Maximum distance from plane to be considered inlier (meters)
        min_points: Minimum number of points needed for plane fitting
        
    Returns:
        bool: True if the points form a flat plane
    """
    if len(points_3d) < min_points:
        return False
    
    # Use RANSAC to fit a plane
    # Prepare data for plane fitting: z = ax + by + c
    X = points_3d[:, :2]  # x, y coordinates
    y = points_3d[:, 2]   # z coordinates (depth)
    
    # Scale the data for better numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit plane using RANSAC
    ransac = RANSACRegressor(
        residual_threshold=plane_threshold,
        min_samples=min_points // 10,  # At least 10% of points for initial fit
        max_trials=100,
        random_state=42
    )
    
    ransac.fit(X_scaled, y)
    
    # Calculate inlier ratio
    inlier_mask = ransac.inlier_mask_
    inlier_ratio = np.sum(inlier_mask) / len(points_3d)
    
    if verbose:
        print(f"  Plane threshold: {plane_threshold}, Min points: {min_points}, Inlier ratio threshold: {inlier_ratio_thres}")
        print(f"  Points: {len(points_3d)}, Inliers: {np.sum(inlier_mask)}, Inlier ratio: {inlier_ratio:.3f}")

    # Consider it a flat plane if >80% of points are inliers
    is_plane = inlier_ratio > inlier_ratio_thres
    
    # Additional check: calculate the standard deviation of residuals for inliers
    if is_plane:
        inlier_residuals = np.abs(y[inlier_mask] - ransac.predict(X_scaled[inlier_mask]))
        residual_std = np.std(inlier_residuals)
        
        # If residual standard deviation is very low, it's definitely a flat plane
        if residual_std < plane_threshold * 0.5:
            return True
            
        # Also check the range of depths - if very small, likely a flat surface
        depth_range = np.max(y[inlier_mask]) - np.min(y[inlier_mask])
        if depth_range < plane_threshold * 2:
            return True
    
    return is_plane
        
