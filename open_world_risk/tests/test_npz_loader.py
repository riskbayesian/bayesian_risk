import numpy as np
import json
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
from PIL import Image


def load_npz_data(color_npz, depth_npz, pose_npz, camera_info_json):
    """
    Load data from individual .npz files and camera info JSON.
    
    Args:
        color_npz: Path to RGB data .npz file
        depth_npz: Path to depth data .npz file
        pose_npz: Path to pose data .npz file
        camera_info_json: Path to camera info JSON file
        
    Returns:
        dict: Dictionary containing loaded data arrays
    """
    # Load pose data
    pose_data = np.load(pose_npz)
    poses = {
        'timestamps': pose_data['timestamps'],
        'positions': pose_data['positions'],
        'orientations': pose_data['orientations']
    }
    
    # Load depth data
    depth_data = np.load(depth_npz)
    depths = {
        'timestamps': depth_data['timestamps'],
        'data': depth_data['data']
    }
    
    # Load RGB data
    rgb_data = np.load(color_npz)
    rgbs = {
        'timestamps': rgb_data['timestamps'],
        'data': rgb_data['data']
    }

    # Analyze timestamp differences for each data type
    print("\n" + "="*60)
    print("ANALYZING RGB TIMESTAMP DIFFERENCES")
    print("="*60)
    analyze_timestamp_differences(rgbs['timestamps'], "RGB Timestamp Differences")
    
    print("\n" + "="*60)
    print("ANALYZING DEPTH TIMESTAMP DIFFERENCES")
    print("="*60)
    analyze_timestamp_differences(depths['timestamps'], "Depth Timestamp Differences")
    
    print("\n" + "="*60)
    print("ANALYZING POSE TIMESTAMP DIFFERENCES")
    print("="*60)
    analyze_timestamp_differences(poses['timestamps'], "Pose Timestamp Differences")
    
    # Synchronize pose data with RGB/depth data using timestamps
    print(f"Synchronizing pose data with RGB/depth data...")
    print(f"RGB/depth frames: {len(rgbs['timestamps'])}")
    print(f"Pose frames: {len(poses['timestamps'])}")
    
    # Find the closest pose for each RGB/depth timestamp
    synchronized_poses = {
        'timestamps': rgbs['timestamps'],  # Use RGB timestamps as reference
        'positions': [],
        'orientations': []
    }
    
    closest_idxs = set()

    for rgb_ts in rgbs['timestamps']:
        # Find the closest pose timestamp
        time_diffs = np.abs(poses['timestamps'] - rgb_ts)
        closest_idx = np.argmin(time_diffs)
        
        # We get that a lot of duplicates
        if closest_idx in closest_idxs:
            print(f"!!!!!!Duplicate closest index found: {closest_idx}!!!!!!")
            continue
        
        closest_idxs.add(closest_idx)

        #print(f"Closest pose timestamp: {poses['timestamps'][closest_idx]}")
        #print(f"Closest pose position: {poses['positions'][closest_idx]}")
        #print(f"Closest pose orientation: {poses['orientations'][closest_idx]}")
        #print('--------------------------------')

        # Check if the time difference is reasonable
        time_diff_ns = time_diffs[closest_idx]
        if time_diff_ns > 1e8:  # 0.1 seconds in nanoseconds
            # This warning should ideally not trigger based on the analysis
            print(f"⚠️ WARNING: Large time difference for timestamp {rgb_ts}: {time_diff_ns / 1e6:.3f} ms")
        
        synchronized_poses['positions'].append(poses['positions'][closest_idx])
        synchronized_poses['orientations'].append(poses['orientations'][closest_idx])
    
    synchronized_poses['positions'] = np.array(synchronized_poses['positions'])
    synchronized_poses['orientations'] = np.array(synchronized_poses['orientations'])
    
    print(f"Synchronization complete. Using {len(synchronized_poses['timestamps'])} synchronized poses.")
    
    # Load camera info data
    with open(camera_info_json, 'r') as f:
        camera_info = json.load(f)
    
    # Extract the first camera info entry (assuming camera parameters are constant)
    print("===================================================")
    print("🚨🚨🚨 ASSUMING CAMERA PARAMETERS ARE CONSTANT 🚨🚨🚨")
    print("===================================================")
    camera_params = camera_info['data'][0]
    analyze_camera_info(camera_info)
    
    # Create simplified camera intrinsics format
    intrinsics = {
        'fx': camera_params['K'][0][0],  # First element of K matrix
        'fy': camera_params['K'][1][1],  # Second element of K matrix
        'cx': camera_params['K'][0][2],  # Third element of first row
        'cy': camera_params['K'][1][2],  # Third element of second row
        'W': camera_params['width'],
        'H': camera_params['height']
    }
    
    return {
        'poses': synchronized_poses,  # Use synchronized poses
        'depths': depths,
        'rgbs': rgbs,
        'intrinsics': intrinsics
    }

# Ok it appears that all of the camera info entries are the same, so we can use the first one for processing
def analyze_camera_info(camera_info):
    """
    Analyze the camera info JSON file.
    
    Args:
        camera_info: Path to camera info JSON file
    """
    # Check if all camera info entries are the same
    if len(camera_info['data']) > 1:
        first_entry = camera_info['data'][0]
        all_same = True
        
        for i, entry in enumerate(camera_info['data'][1:], 1):
            if entry != first_entry:
                all_same = False
                print(f"⚠️ WARNING: Camera info entry {i} differs from the first entry!")
                print(f"   First entry: {first_entry}")
                print(f"   Entry {i}: {entry}")
                break
        
        if all_same:
            print("✅ All camera info entries are identical")
        else:
            print("🚨 WARNING: Camera parameters are NOT constant across all entries!")
            print("   Using only the first entry for processing.")
    else:
        print("✅ Only one camera info entry found")

def generate_depth_image_video(depth_data, output_path):
    """
    Generate depth images and save them to a 'depths' folder.
    
    Args:
        depth_data: Depth data
        output_path: Output path (not used, kept for compatibility)
    """
    # Extract depth images and timestamps
    depth_images = depth_data['data']
    timestamps = depth_data['timestamps']
    
    print(f"Generating depth images with {len(depth_images)} frames...")
    
    # Find global min and max for consistent colormap
    all_depths = np.concatenate([img.flatten() for img in depth_images])
    valid_depths = all_depths[np.logical_and(all_depths > 0, all_depths < np.inf)]
    
    if len(valid_depths) == 0:
        print("No valid depth values found!")
        return
    
    global_min = np.min(valid_depths)
    global_max = np.max(valid_depths)
    
    print(f"Depth range: {global_min:.3f} to {global_max:.3f}")
    
    # Create depths folder
    depths_folder = "depths"
    os.makedirs(depths_folder, exist_ok=True)
    print(f"Saving depth images to: {depths_folder}")
    
    # Set up normalization
    norm = Normalize(vmin=global_min, vmax=global_max)
    
    for i, depth_img in enumerate(depth_images):
        # Create a new figure for each frame to avoid recursion issues
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create the depth image with colormap
        im = ax.imshow(depth_img, cmap='viridis', norm=norm)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Depth', rotation=270, labelpad=15)
        
        # Add title with frame info
        ax.set_title(f'Frame {i+1}/{len(depth_images)} - Time: {timestamps[i]}', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save frame
        frame_path = os.path.join(depths_folder, f'frame_{i:06d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        
        # Close the figure to free memory
        plt.close(fig)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(depth_images)} frames")
    
    print(f"All depth images saved to: {depths_folder}")
    print(f"You can now use ffmpeg to create a video from these images.")

def generate_rgb_image_video(rgb_data, output_path):
    """
    Generate rgb images and save them to a 'rgbs' folder.
    
    Args:
        rgb_data: RGB data
        output_path: Output path (not used, kept for compatibility)
    """
    # Extract rgb images and timestamps
    rgb_images = rgb_data['data']
    timestamps = rgb_data['timestamps']
    
    print(f"Generating rgb images with {len(rgb_images)} frames...")
    
    # Create rgbs folder
    rgbs_folder = "rgbs"
    os.makedirs(rgbs_folder, exist_ok=True)
    print(f"Saving rgb images to: {rgbs_folder}")
    
    for i, rgb_img in enumerate(rgb_images):
        # Convert numpy array to PIL Image
        # RGB images are typically uint8 (0-255) or float (0-1)
        if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
            # If float, assume range 0-1 and convert to uint8
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.astype(np.uint8)
        else:
            rgb_img = rgb_img.astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(rgb_img)
        
        # Save frame
        frame_path = os.path.join(rgbs_folder, f'frame_{i:06d}.png')
        pil_image.save(frame_path)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(rgb_images)} frames")
    
    print(f"All rgb images saved to: {rgbs_folder}")
    print(f"You can now use ffmpeg to create a video from these images.")

def analyze_timestamp_differences(timestamps, title="Timestamp Differences"):
    """
    Analyze and plot histogram of differences between consecutive timestamps.
    
    Args:
        timestamps: Array of timestamps (in nanoseconds)
        title: Title for the histogram plot
    """
    if len(timestamps) < 2:
        print("Need at least 2 timestamps to calculate differences")
        return
    
    # Calculate differences between consecutive timestamps
    differences = np.diff(timestamps)
    
    # Convert to milliseconds for easier interpretation
    differences_ms = differences / 1e6
    
    print(f"Timestamp difference statistics:")
    print(f"  Number of differences: {len(differences)}")
    print(f"  Mean difference: {np.mean(differences_ms):.3f} ms")
    print(f"  Median difference: {np.median(differences_ms):.3f} ms")
    print(f"  Min difference: {np.min(differences_ms):.3f} ms")
    print(f"  Max difference: {np.max(differences_ms):.3f} ms")
    print(f"  Std deviation: {np.std(differences_ms):.3f} ms")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(differences_ms, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Time Difference (ms)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_diff = np.mean(differences_ms)
    median_diff = np.median(differences_ms)
    plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.3f} ms')
    plt.axvline(median_diff, color='green', linestyle='--', label=f'Median: {median_diff:.3f} ms')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some additional statistics
    print(f"\nPercentiles:")
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(differences_ms, p)
        print(f"  {p}th percentile: {value:.3f} ms")

if __name__ == "__main__":
    # Load the data
    data = load_npz_data(
        color_npz="./data/extracted_data_20250616_131333/rgb_data.npz",
        depth_npz="./data/extracted_data_20250616_131333/depth_data.npz",
        pose_npz="./data/extracted_data_20250616_131333/pose_data.npz",
        camera_info_json="./data/extracted_data_20250616_131333/camera_info.json"
    )
    
    #generate_depth_image_video(data['depths'], "depth_video.mp4")
    generate_rgb_image_video(data['rgbs'], "rgb_video.mp4")