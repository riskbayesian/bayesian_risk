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
import glob
import os


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

def load_image_depth(depth_path, target_size=None):
    """
    Load a single depth image.
    
    Args:
        depth_path: Path to depth image file
        target_size: Target size for image (width, height) or None to keep original
        
    Returns:
        depth_array: numpy array of depth frame in meters [H, W]
        filename: filename
    """
    
    assert os.path.exists(depth_path), f"Depth image not found: {depth_path}"
    
    # Load 16-bit depth image
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img, dtype=np.float16)
    
    # Convert from mm to meters (divide by 1000)
    depth_array = depth_array / 1000.0

    # Resize if requested (target_size is (width, height))
    if target_size is not None:
        target_w, target_h = tuple(target_size)
        # Use linear interpolation for depth values
        depth_array = cv2.resize(depth_array.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    filename = os.path.basename(depth_path)
    return depth_array, filename

def load_dir_depth(depth_dir, target_size=None, frame_range=None):
    """
    Load depth images from a directory.
    
    Args:
        depth_dir: Path to directory containing depth images
        target_size: Target size for images (not used for depth, kept for compatibility)
        frame_range: Tuple (start, end) or None to load all frames
        
    Returns:
        depth_frames: numpy array of depth frames in meters [N, H, W]
        filenames: list of filenames
    """
    
    # Find all PNG files in the directory
    pattern = os.path.join(depth_dir, "*.png")
    depth_files = sorted(glob.glob(pattern))
    
    assert depth_files, f"No PNG files found in depth directory: {depth_dir}"
    if frame_range is not None:
        start, end = frame_range
        depth_files = depth_files[start:end]
    
    depth_frames = []
    filenames = []
    
    for depth_file in tqdm(depth_files, desc="Loading depth frames"):
        # Load 16-bit depth image
        depth_img = Image.open(depth_file)
        depth_array = np.array(depth_img, dtype=np.float16)
        
        # Convert from mm to meters (divide by 1000)
        depth_array = depth_array / 1000.0

        # Resize if requested (target_size is (width, height))
        if target_size is not None:
            target_w, target_h = tuple(target_size)
            # Use linear interpolation for depth values
            depth_array = cv2.resize(depth_array.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        depth_frames.append(depth_array)
        filenames.append(os.path.basename(depth_file))
    
    depth_frames = np.array(depth_frames)
    return depth_frames, filenames

def load_dir_depth_mask(mask_dir, target_size=None, frame_range=None):
    """
    Load depth mask images from a directory.
    
    Args:
        mask_dir: Path to directory containing depth mask images
        target_size: Target size for images (not used for masks, kept for compatibility)
        frame_range: Tuple (start, end) or None to load all frames
        
    Returns:
        mask_frames: numpy array of boolean masks [N, H, W] where white=True, black=False
        filenames: list of filenames
    """
    
    # Find all PNG files in the directory
    pattern = os.path.join(mask_dir, "*.png")
    mask_files = sorted(glob.glob(pattern))
    
    assert mask_files, f"No PNG files found in depth directory: {mask_dir}"
    if frame_range is not None:
        start, end = frame_range
        mask_files = mask_files[start:end]
    
    mask_frames = []
    filenames = []
    
    for mask_file in tqdm(mask_files, desc="Loading mask frames"):
        # Load mask image
        mask_img = Image.open(mask_file)
        mask_array = np.array(mask_img)
        
        # Convert to boolean: white (255) = True, black (0) = False
        # Handle both grayscale and RGB images
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]  # Take first channel if RGB
        
        # Resize if requested (target_size is (width, height))
        if target_size is not None:
            target_w, target_h = tuple(target_size)
            mask_array = cv2.resize(mask_array, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask_array > 127  # Threshold at 127 (middle of 0-255)
        
        mask_frames.append(mask_bool)
        filenames.append(os.path.basename(mask_file))
    
    mask_frames = np.array(mask_frames)
    return mask_frames, filenames

def load_dir_img_data_09_22_2025_wrapper(directory_path, target_size=None, 
                                                folder_specified: str = "", frame_range=None, video_range=None):
    """
    Wrapper function to process all rgb_png directories in data_09_22_2025.
    Uses glob to find all directories matching the pattern */*/rgb_png/ and calls load_dir_img on each.
    
    Args:
        directory_path: Path to the data_09_22_2025 directory
        target_size: Target size for images
        folder_specified: If specified, only process the rgb_png directory that matches this folder name
        frame_range: Tuple (start, end) or None to load specific frames from each video
        video_range: Tuple (start, end) or None to process specific videos from the list
    """
        
    # Find all rgb_png directories using glob pattern
    pattern = os.path.join(directory_path, "*", "*", "rgb_png")
    rgb_dirs = glob.glob(pattern)
    rgb_dirs = sorted(rgb_dirs)

    # Filter by folder_specified if provided
    if folder_specified:
        filtered_dirs = []
        for rgb_dir in rgb_dirs:
            dir_name = os.path.basename(os.path.dirname(rgb_dir))
            if dir_name == folder_specified:
                filtered_dirs.append(rgb_dir)
        rgb_dirs = filtered_dirs
        
        if not rgb_dirs:
            print(f"No rgb_png directory found matching folder_specified: {folder_specified}")
            return {}

    # Sort videos by number of frames (small to large)
    dir_counts = []
    for d in rgb_dirs:
        num_frames = len(glob.glob(os.path.join(d, "*.png")))
        dir_counts.append((d, num_frames))
    dir_counts.sort(key=lambda x: x[1])
    rgb_dirs = [d for d, _ in dir_counts]
    if rgb_dirs:
        print("Sorted videos by frame count (ascending):")
        for d, c in dir_counts:
            print(f"  - {d}: {c} frames")

    # Apply video_range filtering if specified
    if video_range is not None:
        start, end = video_range
        rgb_dirs = rgb_dirs[start:end]
        print(f"Applied video_range ({start}, {end}), processing {len(rgb_dirs)} videos")

    print(f"Found {len(rgb_dirs)} rgb_png directories:")
    for rgb_dir in rgb_dirs:
        print(f"  - {rgb_dir}")
    
    results = {}
    rgb_frames = []
    depth_frames = []
    depth_mask_frames = []
    dir_names = []
    
    for i, rgb_dir in enumerate(rgb_dirs):
        print(f"\nProcessing directory: {rgb_dir} {i} / {len(rgb_dirs)}")

        # Load RGB frames
        frames_array, filenames = load_dir_img(rgb_dir, target_size, frame_range=frame_range)
        
        # Get parent directory path for depth and mask directories
        parent_dir = os.path.dirname(rgb_dir)  # e.g., data/data_09_22_2025/kitchen_to_laptop/kitchen_to_laptop
        depth_dir = os.path.join(parent_dir, "depth_png_mm")
        mask_dir = os.path.join(parent_dir, "depth_valid_mask")
        
        # Load depth frames and masks
        depth_frames_array = None
        mask_frames_array = None
        depth_filenames = None
        mask_filenames = None
        
        assert os.path.exists(depth_dir), f"Depth directory not found: {depth_dir}"
        depth_frames_array, depth_filenames = load_dir_depth(depth_dir, target_size, frame_range=frame_range)
        print(f"Loaded depth frames: {depth_frames_array.shape}")
        
        assert os.path.exists(mask_dir), f"Mask directory not found: {mask_dir}"
        mask_frames_array, mask_filenames = load_dir_depth_mask(mask_dir, target_size, frame_range=frame_range)
        print(f"Loaded mask frames: {mask_frames_array.shape}")
        
        # Store results with directory name as key
        dir_name = os.path.basename(os.path.dirname(rgb_dir))  # Get the parent directory name
        results[dir_name] = {
            'filenames': filenames,
            'directory': rgb_dir,
            'depth_frames': depth_frames_array,
            'depth_filenames': depth_filenames,
            'mask_frames': mask_frames_array,
            'mask_filenames': mask_filenames
        }
        dir_names.append(dir_name)
        rgb_frames.append(frames_array)
        depth_frames.append(depth_frames_array)
        depth_mask_frames.append(mask_frames_array)
        print(f"Successfully processed {dir_name}: RGB={frames_array.shape}, Depth={depth_frames_array.shape if depth_frames_array is not None else 'None'}, Mask={mask_frames_array.shape if mask_frames_array is not None else 'None'}")

    return results, rgb_frames, depth_frames, depth_mask_frames, dir_names

def load_dir_img(directory_path, target_size=None, frame_range=None):   
    png_pattern = os.path.join(directory_path, "*.png")
    png_files = glob.glob(png_pattern)
    assert png_files, f"No PNG files found in directory: {directory_path}"
    
    png_files = sorted(png_files)
    if frame_range is not None:
        start, end = frame_range
        png_files = png_files[start:end]
    
    print(f"Loading {len(png_files)} PNG files in {directory_path}")

    # Load first frame to determine target size if not specified
    first_img = Image.open(png_files[0]).convert("RGB")
    if target_size is None:
        target_size = first_img.size  # (width, height)
    else:
        target_size = tuple(target_size)
    
    print(f"Target frame size: {target_size[0]}x{target_size[1]}")
    
    frames = []
    filenames = []
    
    for png_file in tqdm(png_files, desc="Loading frames"):
        img = Image.open(png_file).convert("RGB")
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)  # [H, W, 3]d
        filename = os.path.splitext(os.path.basename(png_file))[0]
        frames.append(img_array)
        filenames.append(filename)
        #print(f"Loaded: {filename} - Shape: {img_array.shape}")
                
    assert frames, "No frames were successfully loaded"    
    frames_array = np.stack(frames, axis=0)  # [N_frames, H, W, 3]
    print(f"Successfully loaded {len(frames)} frames into array of shape: {frames_array.shape}")
    
    return frames_array, filenames

def load_img_png(image_path, target_size=None):
    img = Image.open(image_path).convert("RGB")

    if target_size is None:
        target_size = img.size  # (width, height)
    else:
        target_size = tuple(target_size)
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.uint8)  # [H, W, 3]d
    filename = os.path.splitext(os.path.basename(image_path))
    print(f"Loaded: {filename} - Shape: {img_array.shape}")

    frames = np.expand_dims(img_array, axis=0)  # [1, H, W, 3]
    filenames = [filename]

    return frames, filenames

def load_dir_npz(npz_path_color: str):
    rgb_frames = np.load(npz_path_color)
    return rgb_frames

def load_mov_file(mov_path, target_size=None, max_frames=None):
    """
    Load a .mov file and convert each frame to a numpy array.
    
    Args:
        mov_path: Path to the .mov file
        target_size: Optional tuple (width, height) to resize all frames to. 
                    If None, uses the original video dimensions.
        max_frames: Optional maximum number of frames to load. If None, loads all frames.
        
    Returns:
        np.ndarray: Array of shape [N_frames, H, W, 3] with dtype uint8
        dict: Video metadata including fps, total_frames, duration, etc.
    """
    
    if not os.path.exists(mov_path):
        raise FileNotFoundError(f"Video file not found: {mov_path}")
    
    # Open video file
    cap = cv2.VideoCapture(mov_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {mov_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Determine how many frames to load
    frames_to_load = min(total_frames, max_frames) if max_frames else total_frames
    print(f"Loading {frames_to_load} frames...")
    
    # Set target size
    if target_size is None:
        target_size = (width, height)
    else:
        target_size = tuple(target_size)
    
    print(f"Target frame size: {target_size[0]}x{target_size[1]}")
    
    # Pre-allocate array for all frames
    frames = []
    frame_count = 0
    
    # Progress bar
    pbar = tqdm(total=frames_to_load, desc="Loading video frames")
    
    while frame_count < frames_to_load:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_count}, stopping")
            break
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size if needed
        if frame_rgb.shape[:2][::-1] != target_size:  # (height, width) -> (width, height)
            frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        frames.append(frame_rgb)
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if not frames:
        print("Warning: No frames were successfully loaded")
        return np.array([]), {}
    
    # Stack all frames into a single numpy array
    frames_array = np.stack(frames, axis=0)  # [N_frames, H, W, 3]
    
    # Create metadata dictionary
    metadata = {
        'fps': fps,
        'total_frames': total_frames,
        'loaded_frames': len(frames),
        'original_width': width,
        'original_height': height,
        'target_width': target_size[0],
        'target_height': target_size[1],
        'duration': duration,
        'file_path': mov_path
    }
    
    print(f"Successfully loaded {len(frames)} frames into array of shape: {frames_array.shape}")
    
    return frames_array, metadata

def create_video_output(filename_prefix, output_dir, fps=10, video_name=None):
    """
    Create a video from all files with a given prefix in the output directory.
    
    Args:
        filename_prefix: Prefix to match files (e.g., "risk_field_matplotlib_frame_")
        output_dir: Directory containing the files
        fps: Frames per second for the output video
        video_name: Optional custom name for the output video file
        
    Returns:
        str: Path to the created video file, or None if no files found
    """
    
    # Find all files with the given prefix
    pattern = os.path.join(output_dir, f"{filename_prefix}*.png")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"No files found with prefix '{filename_prefix}' in {output_dir}")
        return None
    
    # Sort files to ensure proper frame order
    matching_files = sorted(matching_files)
    print(f"Found {len(matching_files)} files with prefix '{filename_prefix}'")
    
    # Load first image to get dimensions
    first_img = cv2.imread(matching_files[0])
    if first_img is None:
        print(f"Could not load first image: {matching_files[0]}")
        return None
    
    h, w = first_img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Create video filename
    if video_name is None:
        video_name = f"{filename_prefix.rstrip('_')}_video.mp4"
    video_path = os.path.join(output_dir, video_name)
    
    # Create temporary directory for video frames
    seq_dir = os.path.join(output_dir, "video_seq_tmp")
    os.makedirs(seq_dir, exist_ok=True)
    
    print(f"Creating video: {video_name}")
    print(f"Video dimensions: {w}x{h}, FPS: {fps}")
    
    # Process each frame
    for i, file_path in enumerate(tqdm(matching_files, desc="Creating Video", unit="frame")):
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not load {file_path}")
            continue
        
        # Resize if needed (shouldn't be necessary, but just in case)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        # Add frame number text
        cv2.putText(img, f"Frame {i:04d}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to video sequence directory
        cv2.imwrite(os.path.join(seq_dir, f"frame_{i:06d}.png"), img)
    
    # Create video using ffmpeg
    inp = os.path.join(seq_dir, "frame_%06d.png")
    
    # Try different ffmpeg encoders
    cmds = [
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", inp,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_path],
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", inp,
         "-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_path],
    ]
    
    rc = 1
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0:
                rc = 0
                break
        except FileNotFoundError:
            print("ffmpeg not found, trying next encoder...")
            continue
    
    if rc != 0:
        print("Error: ffmpeg H.264 encoding failed")
        shutil.rmtree(seq_dir, ignore_errors=True)
        return None
    
    # Clean up temporary directory
    shutil.rmtree(seq_dir, ignore_errors=True)
    print(f"Video saved to: {video_path}")
    
    return video_path

# ----------------- Bezier / Bernstein -----------------
# TODO: Function duplicated in src/dataset_loader.py as a staticmethod function, refactor to use that instead
def bernstein_B(num_ctrl: int, S: int = 100, device: str = "cuda", dtype=torch.float32):
    n = num_ctrl - 1
    t = torch.linspace(0, 1, S, device=device, dtype=dtype)
    comb = [math.comb(n, k) for k in range(num_ctrl)]
    B = []
    for k in range(num_ctrl):
        Bk = comb[k] * (t**k) * ((1.0 - t)**(n - k))
        B.append(Bk)
    return torch.stack(B, dim=1), t  # [S,K], [S]

def plot_coefficents(coeffs, device, K, out_dir, max_distance=0.5, frame_idx=None, label="", verbose=False):
    """
    Plot Bezier curves and coefficient statistics.
    
    Args:
        coeffs: Bezier coefficients [H, W, K]
        device: PyTorch device
        K: Number of coefficients
        out_dir: Output directory for plots (defaults to "debug")
        max_distance: Maximum distance in meters for risk field evaluation (defaults to 0.5)
        frame_idx: Optional frame index for filename and title
    """
    if verbose:
        print("\nCreating Bezier curve visualizations...")
    os.makedirs(out_dir, exist_ok=True)
    
    # Find unique coefficient vectors
    H, W = coeffs.shape[:2]
    coeffs_flat = coeffs.view(-1, K)  # [H*W, K]
    
    # Find unique vectors and their indices
    unique_coeffs, unique_indices = torch.unique(coeffs_flat, dim=0, return_inverse=True)
    unique_indices = unique_indices.view(H, W)  # [H, W]
    
    if verbose:
        print(f"Found {len(unique_coeffs)} unique coefficient vectors out of {H*W} total pixels")
    
    # Sample up to 9 unique vectors for plotting
    n_unique = min(len(unique_coeffs), 9)
    sample_indices = torch.randperm(len(unique_coeffs))[:n_unique]
    
    # Find pixel locations for each unique vector
    sample_pixels = []
    for idx in sample_indices:
        # Find first occurrence of this unique vector
        mask = (unique_indices == idx)
        if mask.any():
            h, w = torch.where(mask)[0][0].item(), torch.where(mask)[1][0].item()
            sample_pixels.append((h, w))
    
    if verbose:
        print(f"Plotting Bezier curves for {len(sample_pixels)} unique coefficient vectors")
    
    # Create Bezier evaluation points
    t_eval = torch.linspace(0, 1, 200, device=device)  # More points for smooth curves
    Bbern_eval, _ = bernstein_B(K, S=200, device=device, dtype=torch.float32)
    
    # Plot Bezier curves
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (h, w) in enumerate(sample_pixels):
        if i >= len(axes):
            break
            
        # Get coefficients for this pixel
        pixel_coeffs = coeffs[h, w, :].to(device)  # [K]
        
        # Evaluate Bezier curve: curve = sum(coeffs[k] * B[k](t))
        curve_values = torch.einsum("k,sk->s", pixel_coeffs, Bbern_eval)  # [200]
        
        # Convert to distance (0 to max_distance meters)
        distances = max_distance * t_eval.cpu().numpy()
        curve_vals = curve_values.detach().cpu().numpy()
        
        # Plot the curve
        axes[i].plot(distances, curve_vals, 'b-', linewidth=2, label='Bezier curve')
        axes[i].set_xlabel('Distance (m)')
        axes[i].set_ylabel('Risk Value')
        axes[i].set_title(f'Pixel ({h},{w})')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add coefficient values as text
        coeff_text = ', '.join([f'{c:.3f}' for c in pixel_coeffs.cpu().numpy()])
        axes[i].text(0.02, 0.98, f'Coeffs: [{coeff_text}]', 
                    transform=axes[i].transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(sample_pixels), len(axes)):
        axes[i].axis('off')
    
    frame_suffix = f"_frame_{frame_idx:04d}" if frame_idx is not None else ""
    plt.suptitle(f'Bezier Risk Curves - Sample Pixels (K={K} coefficients){label}{frame_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'bezier_curves_sample_pixels{label}{frame_suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Plot coefficient statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mean across spatial dimensions
    coeff_mean = coeffs.mean(dim=(0,1)).detach().cpu().numpy()
    axes[0].bar(range(K), coeff_mean)
    axes[0].set_title('Mean Coefficient Values')
    axes[0].set_xlabel('Coefficient Index')
    axes[0].set_ylabel('Mean Value')

    # Std across spatial dimensions
    coeff_std = coeffs.std(dim=(0,1)).detach().cpu().numpy()
    axes[1].bar(range(K), coeff_std)
    axes[1].set_title('Std Coefficient Values')
    axes[1].set_xlabel('Coefficient Index')
    axes[1].set_ylabel('Std Value')

    # Min/Max across spatial dimensions
    coeff_min = torch.amin(coeffs, dim=(0,1)).detach().cpu().numpy()  # Fixed: use torch.amin for multiple dims
    coeff_max = torch.amax(coeffs, dim=(0,1)).detach().cpu().numpy()  # Fixed: use torch.amax for multiple dims
    axes[2].fill_between(range(K), coeff_min, coeff_max, alpha=0.3, label='Min-Max Range')
    axes[2].plot(range(K), coeff_mean, 'o-', label='Mean')
    axes[2].set_title('Coefficient Range')
    axes[2].set_xlabel('Coefficient Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'coefficients_statistics{label}{frame_suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved Bezier curve plots to: {out_dir}")

def plot_risk_image(exp_arr: np.ndarray, cfg: dict, output_path: str, max_dist: int = 0.5, label="", verbose: bool = False):
    """
    Plot and save coefficients, probability maps (optional), heatmap, overlay, and matplotlib figure.
    
    Args:
        coeffs: Bezier coefficients [H, W, K]
        probs: Probability maps [H, W, S]
        exp_img: Expected distance image tensor [Ha, Wa]
        exp_arr: Expected distance array [Ha, Wa]
        norm: Normalized risk values [Ha, Wa]
        a_resized: Resized image A (PIL Image)
        cfg: Configuration dictionary
        Ha, Wa: Resized image A dimensions
        frame_idx: Optional frame index for filename and title
    """
    
    assert output_path.endswith('.png'), f"Output path must end with .png, got: {output_path}"

    # Normalize exp_arr for colormap using max_dist as the upper bound
    exp_norm = np.clip(exp_arr / max_dist, 0, 1)
    exp_uint8 = (exp_norm * 255).astype(np.uint8)
    
    # Apply colormap using OpenCV (convert matplotlib cmap to OpenCV)
    cmap_mapping = {
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL,
        'rainbow': cv2.COLORMAP_RAINBOW
    }
    
    cv_cmap = cmap_mapping.get(cfg["cmap"], cv2.COLORMAP_VIRIDIS)
    heat_bgr = cv2.applyColorMap(exp_uint8, cv_cmap)
    heat_rgb = heat_bgr[:, :, ::-1]  # Convert BGR to RGB
    
    # Create a larger canvas to accommodate text and colorbar
    canvas_height = heat_rgb.shape[0] + 200  # Extra space for text and colorbar
    canvas_width = heat_rgb.shape[1] + 150   # Extra space for colorbar
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background
    
    # Place heatmap on canvas
    canvas[50:50+heat_rgb.shape[0], 50:50+heat_rgb.shape[1]] = heat_rgb
    
    # Add title
    title_text = f'Risk Field - Expected Distance (Normalized)\n{label}'
    cv2.putText(canvas, title_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add stats text
    stats_text = f'Min: {exp_arr.min():.3f}m  Max: {exp_arr.max():.3f}m  Mean: {exp_arr.mean():.3f}m  Std: {exp_arr.std():.3f}m'
    cv2.putText(canvas, stats_text, (50, canvas_height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    # Create colorbar that matches the heatmap colormap
    colorbar_width = 20
    colorbar_height = heat_rgb.shape[0]
    colorbar_x = canvas_width - 100
    colorbar_y = 50
    
    # Build vertical gradient using same matplotlib colormap as heatmap
    try:
        from matplotlib import cm as mpl_cm
        cmap_name = cfg.get("cmap", "viridis")
        mpl_cmap = mpl_cm.get_cmap(cmap_name)
    except Exception:
        # Fallback to viridis if matplotlib is unavailable
        from matplotlib import cm as mpl_cm
        mpl_cmap = mpl_cm.get_cmap("viridis")

    grad = np.linspace(1.0, 0.0, colorbar_height, dtype=np.float32)[:, None]  # top=max, bottom=min
    grad_rgb = (mpl_cmap(grad)[:, :, :3] * 255).astype(np.uint8)               # [H,1,3]
    colorbar_img = np.repeat(grad_rgb, colorbar_width, axis=1)                 # [H,W,3]
    canvas[colorbar_y:colorbar_y + colorbar_height, colorbar_x:colorbar_x + colorbar_width] = colorbar_img
    
    # Add colorbar labels
    cv2.putText(canvas, f'{max_dist:.2f}', (colorbar_x + colorbar_width + 5, colorbar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, f'0.00', (colorbar_x + colorbar_width + 5, colorbar_y + colorbar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, 'Distance (m)', (colorbar_x - 10, colorbar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

def learn_pca_from_features(features):
    """
    Learn PCA parameters from DINO features.
    
    Args:
        features: DINO features [C, H, W]
        
    Returns:
        pca: Fitted PCA object
        feats_flat: Flattened features [H*W, C]
        H, W: Height and width of the features
    """
    # Convert features to numpy
    feats_np = features.detach().cpu().numpy()  # [C, H, W]
    
    # Reshape features for PCA: [C, H, W] -> [H*W, C]
    C, H, W = feats_np.shape
    feats_flat = feats_np.transpose(1, 2, 0).reshape(-1, C)  # [H*W, C]
    
    # Fit PCA on features
    pca = PCA(n_components=3, whiten=True)
    pca.fit(feats_flat)  # Fit but don't transform yet
    
    return pca, feats_flat, H, W

@timeFunction
def vis_dino_pca(features, image_name, out_dir="debug", pca=None):
    """
    Visualize DINO features using PCA and save as RGB image.
    
    Args:
        features: DINO features [C, H, W]
        image_name: Name for the output file (e.g., "featsA", "featsB")
        out_dir: Output directory for PCA visualizations (defaults to "debug")
        pca: Optional pre-fitted PCA object. If None, fits PCA on the input features.
    """
    print("=" * 50)
    print(f"DINO FEATURES PCA VISUALIZATION - {image_name}")
    print("=" * 50)

    # Save DINO features as PCA images
    os.makedirs(out_dir, exist_ok=True)

    # Convert features to numpy
    feats_np = features.detach().cpu().numpy()  # [C, H, W]

    print(f"DINO features shape: {feats_np.shape}")

    # Reshape features for PCA: [C, H, W] -> [H*W, C]
    C, H, W = feats_np.shape
    feats_flat = feats_np.transpose(1, 2, 0).reshape(-1, C)  # [H*W, C]

    print(f"Flattened features: {feats_flat.shape}")

    # Use provided PCA or fit new one
    if pca is not None:
        print("Using provided PCA parameters...")
        feats_pca = pca.transform(feats_flat)  # [H*W, 3]
    else:
        print("Fitting PCA on features...")
        pca = PCA(n_components=3, whiten=True)
        feats_pca = pca.fit_transform(feats_flat)  # [H*W, 3]

    # Convert to RGB using sigmoid (inspired by process_video_utils.py)
    print("Converting PCA to RGB...")
    feats_rgb = (1.0 / (1.0 + np.exp(-2.0 * feats_pca)) * 255).astype(np.uint8)  # [H*W, 3]

    # Reshape back to image format: [H*W, 3] -> [H, W, 3]
    feats_img = feats_rgb.reshape(H, W, 3)

    # Save PCA visualization
    print("Saving PCA visualization...")
    output_filename = f"{image_name}_pca.png"
    Image.fromarray(feats_img, mode='RGB').save(os.path.join(out_dir, output_filename))

    # Print PCA statistics
    print(f"\nPCA Statistics:")
    print(f"  Features PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  Features PCA total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    # Check if features are diverse
    unique_feats = len(np.unique(feats_flat.round(decimals=4)))
    print(f"  Unique values: {unique_feats}")

    if unique_feats < 100:
        print("  WARNING: Very few unique values in features! DINO may not be working properly.")

    print(f"\nSaved DINO PCA visualization to: {out_dir}/{output_filename}")

def plot_depth_risk_image(risk_at_depth: np.ndarray, cfg: dict, output_path: str, verbose: bool = False, save_npy=False):
    """
    Plot and save depth-based risk field as a 1-channel image using OpenCV.
    
    Args:
        risk_at_depth: Risk field values at depth locations [H, W] - numpy array
        cfg: Configuration dictionary (for colormap)
        output_path: Path to save the output image (must end with .png)
        max_dist: Maximum distance for normalization (default 0.5)
        label: Optional label for title
        verbose: Whether to print debug information
        
    Returns:
        None
    """
    
    assert output_path.endswith('.png'), f"Output path must end with .png, got: {output_path}"
    assert isinstance(risk_at_depth, np.ndarray), f"risk_at_depth must be numpy array, got {type(risk_at_depth)}"
    assert len(risk_at_depth.shape) == 2, f"risk_at_depth must be a 2D array (1-channel image), got shape: {risk_at_depth.shape}"

    # Ensure risk_at_depth is float32 and in [0, 1] range
    risk_float32 = risk_at_depth.astype(np.float32)
    
    # Save as PNG using PIL (convert float32 to uint8 for PNG compatibility)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert float32 [0, 1] to uint8 [0, 255] for PNG compatibility
    risk_uint8 = (risk_float32 * 255).astype(np.uint8)
    risk_pil = Image.fromarray(risk_uint8, mode='L')  # 'L' = grayscale mode
    risk_pil.save(output_path)
    
    if save_npy:
        numpy_path = output_path.replace('.png', '.npy')
        np.save(numpy_path, risk_at_depth)
    
    if verbose:
        print(f"Saved depth-based risk field visualization to: {output_path}")
        print(f"Risk field stats: min={risk_at_depth.min():.3f}, max={risk_at_depth.max():.3f}, mean={risk_at_depth.mean():.3f}")