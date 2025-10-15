#!/usr/bin/env python3

import os
import numpy as np
import cv2
import argparse
from pathlib import Path
from src.point_cloud_3D import PointCloudGenerator
import json
from tqdm import tqdm
import open3d as o3d

corrupted_img_idxs = [835, 1048, 1704]

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

    # Remove corrupted images (sort in descending order to avoid index shifting)
    corrupted_indices = sorted(corrupted_img_idxs, reverse=True)
    for idx in corrupted_indices:
        rgbs['timestamps'] = np.delete(rgbs['timestamps'], idx)
        rgbs['data'] = np.delete(rgbs['data'], idx, axis=0)
        depths['timestamps'] = np.delete(depths['timestamps'], idx)
        depths['data'] = np.delete(depths['data'], idx, axis=0)

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
    
    seen_pose_idxs = set()

    # Found that all of these errors occur before frame: 271, so just skip the first 271 frames
    rgbs['timestamps'] = rgbs['timestamps'][271:]
    rgbs['data'] = rgbs['data'][271:]
    depths['timestamps'] = depths['timestamps'][271:]
    depths['data'] = depths['data'][271:]

    for idx, rgb_ts in enumerate(rgbs['timestamps']):
        # Find the closest pose timestamp
        time_diffs = np.abs(poses['timestamps'] - rgb_ts)
        closest_idx = np.argmin(time_diffs) # NOTE: There are some repeated closest indices 
        
        # Check if the time difference is reasonable
        time_diff_ns = time_diffs[closest_idx]
        if time_diff_ns > 1e8:  # 0.1 seconds in nanoseconds
            # This warning should ideally not trigger based on the analysis
            print(f"⚠️ WARNING: Large time difference for timestamp {rgb_ts}: {time_diff_ns / 1e6:.3f} ms")
            continue # NOTE: We don't want to use this frame

        if closest_idx in seen_pose_idxs:
            print(f"!!!!!!Duplicate closest index found: {closest_idx}!!!!!!")
            print(f"RGB index:             {idx}")
            print(f"Pose index:            {closest_idx}")
            print(f"RGB timestamp:         {rgb_ts}")
            print(f"Pose timestamp chosen: {poses['timestamps'][closest_idx]}")
            continue
        
        synchronized_poses['positions'].append(poses['positions'][closest_idx])
        synchronized_poses['orientations'].append(poses['orientations'][closest_idx])
        seen_pose_idxs.add(closest_idx)
    
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

def get_frame_indices(rgbs, poses, max_frames=None, verbose=False):
    n_frames = len(rgbs['timestamps'])
    if verbose:
        print(f"Total frames available: {n_frames}")
    
    # Determine which frames to process
    if max_frames is not None and max_frames < n_frames:
        # Calculate step size for even spacing
        step = n_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
        print(f"Processing {max_frames} frames with even spacing from {n_frames} total frames")
        print(f"Step size: {step:.2f}")
    else:
        frame_indices = list(range(n_frames))
        print(f"Processing all {n_frames} frames")

    return frame_indices

def process_npz_data(color_npz, depth_npz, pose_npz, output_dir, camera_info_json, verbose=False, max_frames=None, save_pt_cloud=False, no_clip=False):
    """
    Process RGB and depth images from .npz files.
    
    Args:
        color_npz: Path to RGB data .npz file
        depth_npz: Path to depth data .npz file
        pose_npz: Path to pose data .npz file
        output_dir: Directory to save outputs
        camera_info_json: Path to camera info JSON file
        verbose: Whether to print verbose output
        max_frames: Maximum number of frames to process (if None, process all frames)
        save_pt_cloud: Whether to save a conglomerate point cloud
    """
    # Convert to Path objects
    output_dir = Path(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from .npz files and camera info
    if verbose:
        print("Loading data from .npz files and camera info...")
    data = load_npz_data(color_npz, depth_npz, pose_npz, camera_info_json)
    
    # Save camera intrinsics to a temporary file for PointCloudGenerator
    temp_intrinsics_path = output_dir / "temp_intrinsics.json"
    with open(temp_intrinsics_path, 'w') as f:
        json.dump(data['intrinsics'], f, indent=2)
    
    # Initialize models
    if verbose:
        print("Initializing models...")
    pc_generator = PointCloudGenerator(intrinsics=str(temp_intrinsics_path))
    if not no_clip:
        from src.compute_obj_part_feature import FeatureExtractor
        feature_extractor = FeatureExtractor()
    else:
        feature_extractor = None
    if verbose:
        print("Models initialized successfully")
    
    # Get number of frames
    frame_indices = get_frame_indices(data['rgbs'], data['poses'], max_frames)
    
    # Initialize lists for conglomerate point cloud if requested
    all_points = []
    all_colors = []
    all_embeddings = []
    
    # Store selected poses for analysis
    selected_pose_timestamps_for_analysis = []
    selected_pose_positions_for_analysis = []
    selected_pose_orientations_for_analysis = []
    
    # Initialize timing collection
    all_timings = []
    successful_frames = 0
    failed_frames = 0
    
    # Process selected frames
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames", unit="frame")):
        if verbose:
            # Get timestamps for debugging
            rgb_ts_ns = data['rgbs']['timestamps'][frame_idx]
            depth_ts_ns = data['depths']['timestamps'][frame_idx]
            pose_ts_ns = data['poses']['timestamps'][frame_idx]
            time_diff_ms = (rgb_ts_ns - pose_ts_ns) / 1e6

            print(f"\n--- Processing Frame {frame_idx} ---")
            print(f"  - RGB Timestamp:   {rgb_ts_ns}")
            print(f"  - Depth Timestamp: {depth_ts_ns}")
            print(f"  - Matched Pose TS: {pose_ts_ns} (Diff from RGB: {time_diff_ms:.3f} ms)")
        
        # Create output subdirectory for this frame
        frame_output_dir = output_dir / f"frame_{frame_idx:06d}"
        os.makedirs(frame_output_dir, exist_ok=True)
        
        try:
            # Get RGB and depth data for this frame
            rgb_image = data['rgbs']['data'][frame_idx]
            depth_image = data['depths']['data'][frame_idx]
            pose = {
                'position': data['poses']['positions'][frame_idx],
                'orientation': data['poses']['orientations'][frame_idx]
            }

            points, colors = pc_generator.get_point_cloud(rgb_image, depth_image, pose)
            
            if not no_clip:
                # Convert RGB image to the format expected by feature extractor
                rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                
                # 1. Process RGB image through FeatureExtractor
                results = feature_extractor.process_single_image(
                    image=rgb_image_rgb,
                    output_paths={'obj_feat_path': str(frame_output_dir / f'frame_{frame_idx:06d}_features.npy')},
                    save=True,
                    verbose=verbose
                )
                
                # Collect timing data
                frame_timings = results.timings.copy()
                frame_timings['frame_idx'] = frame_idx
                all_timings.append(frame_timings)
                
                # Get CLIP embeddings and masks directly from results
                sam_masks_clip = results.sam_masks_clip

                # Create per-point embeddings by mapping mask embeddings to individual points
                # This is the critical fix: we need to assign embeddings to each point based on which mask it belongs to
                per_point_embeddings = pc_generator.create_per_point_embeddings(
                    points=points,
                    sam_masks_clip=sam_masks_clip,
                    image_shape=(rgb_image.shape[0], rgb_image.shape[1]),
                    pose=pose
                )
            
            # Save RGB point cloud
            pc_generator.save_point_cloud(
                points=points,
                colors=colors,
                output_dir=str(frame_output_dir),
                cloud_type="rgb",
                base_filename=f"frame_{frame_idx:06d}"
            )
            
            # Store data for conglomerate point cloud if requested
            if save_pt_cloud:
                all_points.append(points)
                all_colors.append(colors)
                if not no_clip:
                    all_embeddings.append(per_point_embeddings)
            
            successful_frames += 1
            if verbose:
                print(f"Successfully processed frame {frame_idx}")
            
        except Exception as e:
            failed_frames += 1
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue
    
    # Create and save conglomerate point cloud if requested
    if save_pt_cloud and all_points:
        print("Creating conglomerate point cloud...")
        try:
            # Combine all points, colors, and embeddings
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            if not no_clip:
                combined_embeddings = np.vstack(all_embeddings)
            
            print(f"Combined point cloud: {len(combined_points)} points")
            print(f"Combined point cloud range:")
            print(f"  X: [{np.min(combined_points[:, 0]):.3f}, {np.max(combined_points[:, 0]):.3f}]")
            print(f"  Y: [{np.min(combined_points[:, 1]):.3f}, {np.max(combined_points[:, 1]):.3f}]")
            print(f"  Z: [{np.min(combined_points[:, 2]):.3f}, {np.max(combined_points[:, 2]):.3f}]")
            
            # Save conglomerate CLIP point cloud
            if not no_clip:
                pc_generator.save_point_cloud_with_embeddings(
                    points=combined_points,
                    colors=combined_colors,
                    embeddings=combined_embeddings,
                    output_dir=str(output_dir),
                    cloud_type="CLIP",
                    base_filename="conglomerate"
                )
            
            # Save conglomerate RGB point cloud
            pc_generator.save_point_cloud(
                points=combined_points,
                colors=combined_colors,
                output_dir=str(output_dir),
                cloud_type="rgb",
                base_filename="conglomerate"
            )
            
            print(f"Conglomerate point clouds saved to {output_dir}")
            
        except Exception as e:
            print(f"Error creating conglomerate point cloud: {str(e)}")

    # Save selected poses for analysis
    if selected_pose_positions_for_analysis:
        selected_poses_path = output_dir / "selected_poses_for_analysis.npz"
        np.savez(
            selected_poses_path,
            timestamps=np.array(selected_pose_timestamps_for_analysis),
            positions=np.array(selected_pose_positions_for_analysis),
            orientations=np.array(selected_pose_orientations_for_analysis)
        )
        print(f"\n[ANALYSIS] Saved selected poses to: {selected_poses_path}")
        print(f"           You can visualize this against the full trajectory with:")
        print(f"           python scripts/analyze_poses.py <path_to_full_pose_file.npz> --selected-poses-npz {selected_poses_path}")
    
    # Clean up temporary intrinsics file
    temp_intrinsics_path.unlink()
    
    # Print timing statistics
    if all_timings and not no_clip:
        print_timing_statistics(all_timings, successful_frames, failed_frames)
    else:
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Successful frames: {successful_frames}")
        print(f"Failed frames: {failed_frames}")
        if no_clip:
            print("CLIP feature extraction was disabled (--no-clip flag)")

def print_timing_statistics(all_timings, successful_frames, failed_frames):
    """Print detailed timing statistics from the processing pipeline."""
    print(f"\n{'='*60}")
    print(f"TIMING STATISTICS")
    print(f"{'='*60}")
    print(f"Total frames processed: {successful_frames + failed_frames}")
    print(f"Successful frames: {successful_frames}")
    print(f"Failed frames: {failed_frames}")
    print(f"Success rate: {successful_frames/(successful_frames + failed_frames)*100:.1f}%")
    
    if not all_timings:
        print("No timing data available")
        return
    
    # Define the specific timing components we want to track
    timing_components = [
        'yolo_bounding_box',
        'sam_mask_generation', 
        'pixelwise_clip',
        'mask_processing',
        'overhead',
        'total_time'
    ]
    
    print(f"\n{'='*60}")
    print(f"DETAILED TIMING BREAKDOWN (seconds)")
    print(f"{'='*60}")
    
    for key in timing_components:
        values = [timing.get(key, 0) for timing in all_timings if key in timing]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            median_val = np.median(values)
            
            print(f"{key:25s}:")
            print(f"  {'Mean':>10s}: {mean_val:8.3f}s")
            print(f"  {'Std':>10s}:  {std_val:8.3f}s")
            print(f"  {'Min':>10s}:  {min_val:8.3f}s")
            print(f"  {'Max':>10s}:  {max_val:8.3f}s")
            print(f"  {'Median':>10s}: {median_val:8.3f}s")
            print()
    
    # Calculate throughput based on total time
    total_times = [timing.get('total_time', 0) for timing in all_timings if 'total_time' in timing]
    if total_times:
        print(f"{'='*60}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        mean_total = np.mean(total_times)
        if mean_total > 0:
            fps = 1.0 / mean_total
            print(f"Average throughput: {fps:.2f} frames/second")
        
        # Calculate total processing time for all frames
        total_processing_time = sum(total_times)
        print(f"Total processing time for all frames: {total_processing_time:.3f}s")
    
    print(f"{'='*60}")

def main():
    """
    Run the vision pipeline on .npz data files.
    Usage:
        python run_vision_pipeline_npz.py --color-npz path/to/rgb_data.npz --depth-npz path/to/depth_data.npz --pose-npz path/to/pose_data.npz --camera-info path/to/camera_info.json --output-dir path/to/output [--max-frames N] [--save-pt-cloud]
    """
    parser = argparse.ArgumentParser(description='Run vision pipeline on .npz data files')
    parser.add_argument('--color-npz', type=str, required=True, help='Path to RGB data .npz file')
    parser.add_argument('--depth-npz', type=str, required=True, help='Path to depth data .npz file')
    parser.add_argument('--pose-npz', type=str, required=True, help='Path to pose data .npz file')
    parser.add_argument('--camera-info', type=str, required=True, help='Path to camera info JSON file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for pipeline results')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to process (with even spacing)')
    parser.add_argument('--save-pt-cloud', action='store_true', help='Save a conglomerate point cloud from all processed frames')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--no-clip', action='store_true', help='Do NOT run CLIP feature extraction') # NOTE: This is for Mac compatibility
    
    args = parser.parse_args()
    
    # Run the vision pipeline on .npz data
    process_npz_data(
        args.color_npz, 
        args.depth_npz, 
        args.pose_npz, 
        args.output_dir, 
        args.camera_info,
        verbose=args.verbose,
        max_frames=args.max_frames,
        save_pt_cloud=args.save_pt_cloud,
        no_clip=args.no_clip
    )

if __name__ == "__main__":
    main()
