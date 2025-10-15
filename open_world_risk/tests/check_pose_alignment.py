#!/usr/bin/env python3

import numpy as np
import argparse

def check_pose_alignment(color_npz, depth_npz, pose_npz):
    """
    Check alignment between RGB, depth, and pose data.
    """
    print(f"Loading data from:")
    print(f"  Color: {color_npz}")
    print(f"  Depth: {depth_npz}")
    print(f"  Pose:  {pose_npz}")
    
    # Load data
    color_data = np.load(color_npz)
    depth_data = np.load(depth_npz)
    pose_data = np.load(pose_npz)
    
    # Extract timestamps
    rgb_timestamps = color_data['timestamps']
    depth_timestamps = depth_data['timestamps']
    pose_timestamps = pose_data['timestamps']
    
    print(f"\n=== DATA SUMMARY ===")
    print(f"RGB frames: {len(rgb_timestamps)}")
    print(f"Depth frames: {len(depth_timestamps)}")
    print(f"Pose frames: {len(pose_timestamps)}")
    
    print(f"\n=== TIMESTAMP RANGES ===")
    print(f"RGB:   {rgb_timestamps[0]:.3f} to {rgb_timestamps[-1]:.3f}")
    print(f"Depth: {depth_timestamps[0]:.3f} to {depth_timestamps[-1]:.3f}")
    print(f"Pose:  {pose_timestamps[0]:.3f} to {pose_timestamps[-1]:.3f}")
    
    # Check if lengths match
    if len(rgb_timestamps) == len(depth_timestamps) == len(pose_timestamps):
        print(f"\n✓ All streams have same length: {len(rgb_timestamps)}")
        
        # Check timestamp alignment
        rgb_depth_diff = np.max(np.abs(rgb_timestamps - depth_timestamps))
        rgb_pose_diff = np.max(np.abs(rgb_timestamps - pose_timestamps))
        depth_pose_diff = np.max(np.abs(depth_timestamps - pose_timestamps))
        
        print(f"\n=== TIMESTAMP DIFFERENCES ===")
        print(f"RGB vs Depth max diff: {rgb_depth_diff:.6f}")
        print(f"RGB vs Pose max diff:  {rgb_pose_diff:.6f}")
        print(f"Depth vs Pose max diff: {depth_pose_diff:.6f}")
        
        if rgb_depth_diff < 1e-6 and rgb_pose_diff < 1e-6:
            print("✓ All timestamps are perfectly aligned")
        else:
            print("⚠️ Timestamps are not perfectly aligned!")
            
        # Show specific frame indices that the pipeline would select
        n_frames = len(rgb_timestamps)
        max_frames = 3
        step = n_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
        
        print(f"\n=== PIPELINE FRAME SELECTION (max_frames={max_frames}) ===")
        print(f"Step size: {step:.2f}")
        print(f"Selected indices: {frame_indices}")
        
        for i, frame_idx in enumerate(frame_indices):
            rgb_ts = rgb_timestamps[frame_idx]
            depth_ts = depth_timestamps[frame_idx]
            pose_ts = pose_timestamps[frame_idx]
            pose_pos = pose_data['positions'][frame_idx]
            pose_quat = pose_data['orientations'][frame_idx]
            
            print(f"\nFrame {frame_idx}:")
            print(f"  Timestamps: RGB={rgb_ts:.3f}, Depth={depth_ts:.3f}, Pose={pose_ts:.3f}")
            print(f"  Position: {pose_pos}")
            print(f"  Orientation: {pose_quat}")
            
            # Check if this matches what we see in analyze_poses.py
            if i == 0:
                print(f"  Expected from analyze_poses.py: pos(  -2.260,    0.172,    0.932)")
                print(f"  Actual:                      pos({pose_pos[0]:8.3f}, {pose_pos[1]:8.3f}, {pose_pos[2]:8.3f})")
        
        # Check if the pose data matches what analyze_poses.py shows
        print(f"\n=== COMPARISON WITH ANALYZE_POSES.PY ===")
        print("First few poses from analyze_poses.py:")
        print("[1743721526285068800.000] pos(  -2.260,    0.172,    0.932)")
        print("[1743721527124988160.000] pos(  -2.019,    0.173,    0.984)")
        print("[1743721527955063296.000] pos(  -1.719,    0.124,    1.000)")
        
        print("\nFirst few poses from this data:")
        for i in range(min(3, len(pose_timestamps))):
            ts = pose_timestamps[i]
            pos = pose_data['positions'][i]
            print(f"[{ts:.0f}] pos({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f})")
        
        # Check if timestamps are in the same range
        print(f"\nTimestamp ranges:")
        print(f"analyze_poses.py: 1743721526285068800.000 to 1743721562954714624.000")
        print(f"this data:        {pose_timestamps[0]:.0f} to {pose_timestamps[-1]:.0f}")
        
    else:
        print("⚠️ Different number of frames in data streams!")
        print(f"RGB: {len(rgb_timestamps)}, Depth: {len(depth_timestamps)}, Pose: {len(pose_timestamps)}")

def main():
    parser = argparse.ArgumentParser(description='Check alignment between RGB, depth, and pose data')
    parser.add_argument('--color-npz', type=str, required=True, help='Path to RGB data .npz file')
    parser.add_argument('--depth-npz', type=str, required=True, help='Path to depth data .npz file')
    parser.add_argument('--pose-npz', type=str, required=True, help='Path to pose data .npz file')
    
    args = parser.parse_args()
    
    check_pose_alignment(args.color_npz, args.depth_npz, args.pose_npz)

if __name__ == "__main__":
    main() 