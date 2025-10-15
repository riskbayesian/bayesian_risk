#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
from matplotlib.lines import Line2D

def analyze_poses(pose_npz_path, output_dir=None, save_plots=True, selected_poses_path=None):
    """
    Analyze pose data from a .npz file and create visualizations.
    
    Args:
        pose_npz_path: Path to the pose .npz file
        output_dir: Directory to save plots (if None, plots are shown but not saved)
        save_plots: Whether to save plots to files
    """
    print(f"Loading pose data from: {pose_npz_path}")
    
    # Load pose data
    pose_data = np.load(pose_npz_path)

    # Load selected poses if provided
    selected_poses = None
    if selected_poses_path:
        try:
            selected_poses = np.load(selected_poses_path)
            print(f"Loaded {len(selected_poses['timestamps'])} selected poses for comparison from: {selected_poses_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find selected poses file at '{selected_poses_path}'")
    
    # Extract data
    timestamps = pose_data['timestamps']
    positions = pose_data['positions']
    orientations = pose_data['orientations']
    
    print(f"\n=== POSE DATA SUMMARY ===")
    print(f"Number of poses: {len(timestamps)}")
    print(f"Time range: {timestamps[0]:.3f} to {timestamps[-1]:.3f} seconds")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    
    print(f"\n=== POSITION STATISTICS ===")
    print(f"Position shape: {positions.shape}")
    print(f"X range: [{np.min(positions[:, 0]):.6f}, {np.max(positions[:, 0]):.6f}]")
    print(f"Y range: [{np.min(positions[:, 1]):.6f}, {np.max(positions[:, 1]):.6f}]")
    print(f"Z range: [{np.min(positions[:, 2]):.6f}, {np.max(positions[:, 2]):.6f}]")
    
    # Calculate position changes
    position_changes = np.diff(positions, axis=0)
    total_distance = np.sum(np.linalg.norm(position_changes, axis=1))
    max_step = np.max(np.linalg.norm(position_changes, axis=1))
    mean_step = np.mean(np.linalg.norm(position_changes, axis=1))
    
    print(f"Total distance traveled: {total_distance:.6f} units")
    print(f"Maximum step size: {max_step:.6f} units")
    print(f"Mean step size: {mean_step:.6f} units")
    
    print(f"\n=== ORIENTATION STATISTICS ===")
    print(f"Orientation shape: {orientations.shape}")
    
    # Check quaternion normalization
    quat_norms = np.linalg.norm(orientations, axis=1)
    print(f"Quaternion norms - min: {np.min(quat_norms):.6f}, max: {np.max(quat_norms):.6f}")
    if np.any(np.abs(quat_norms - 1.0) > 1e-6):
        print("⚠️ WARNING: Some quaternions are not normalized!")
    else:
        print("✓ All quaternions are properly normalized")
    
    # Calculate orientation changes
    orientation_changes = np.diff(orientations, axis=0)
    max_orientation_change = np.max(np.linalg.norm(orientation_changes, axis=1))
    mean_orientation_change = np.mean(np.linalg.norm(orientation_changes, axis=1))
    
    print(f"Maximum orientation change: {max_orientation_change:.6f}")
    print(f"Mean orientation change: {mean_orientation_change:.6f}")
    
    print(f"\n=== DETAILED POSE VALUES ===")
    print("Format: [timestamp] position(x,y,z) orientation(w,x,y,z)")
    print("-" * 80)
    
    # Print first 10 and last 10 poses, plus every 100th pose in between
    indices_to_print = list(range(10))  # First 10
    if len(timestamps) > 20:
        indices_to_print.extend(range(100, len(timestamps), 100))  # Every 100th
    indices_to_print.extend(range(max(10, len(timestamps)-10), len(timestamps)))  # Last 10
    
    for i in sorted(set(indices_to_print)):
        if i < len(timestamps):
            print(f"[{timestamps[i]:8.3f}] pos({positions[i, 0]:8.3f}, {positions[i, 1]:8.3f}, {positions[i, 2]:8.3f}) "
                  f"quat({orientations[i, 0]:8.6f}, {orientations[i, 1]:8.6f}, {orientations[i, 2]:8.6f}, {orientations[i, 3]:8.6f})")
    
    # Create visualizations
    create_pose_visualizations(timestamps, positions, orientations, output_dir, save_plots, selected_poses)

def create_pose_visualizations(timestamps, positions, orientations, output_dir=None, save_plots=True, selected_poses=None):
    """
    Create various visualizations of the pose data.
    """
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. 3D trajectory plot
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1, alpha=0.7, label='Camera trajectory')
    
    # Plot start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='blue', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='o', label='End')
    
    # Add camera orientation indicators
    if selected_poses is not None:
        # If we have selected poses, only show indicators for them
        sel_positions = selected_poses['positions']
        sel_orientations = selected_poses['orientations']
        
        # Plot selected points as large dots
        ax.scatter(sel_positions[:, 0], sel_positions[:, 1], sel_positions[:, 2], 
                   c='cyan', s=100, marker='o', label='Selected Poses', 
                   edgecolors='black', depthshade=False, zorder=10)

        print(f"\n[DEBUG] Drawing XYZ basis vectors for {len(sel_positions)} selected poses...")
        for i in range(len(sel_positions)):
            q = sel_orientations[i]
            
            # Skip if quaternion is invalid to avoid errors
            if not np.isclose(np.linalg.norm(q), 1.0): continue

            # Get rotation object from quaternion
            r = Rotation.from_quat(q)
            
            origin = sel_positions[i]
            if i < 5: # Print first 5 for verification
                # Get Roll, Pitch, Yaw in degrees for debugging
                # Using 'zyx' sequence for yaw, pitch, roll.
                yaw, pitch, roll = r.as_euler('zyx', degrees=True)
                print(f"  - Drawing axes for pose {i} at [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}]")
                print(f"    - Quaternion (x,y,z,w): [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
                print(f"    - RPY (deg): [Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}]")

            # Get rotation matrix from the object
            R = r.as_matrix()
            
            # The columns of the rotation matrix are the basis vectors of the camera's
            # local coordinate system expressed in the world frame.
            x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]
            
            arrow_length = 0.5 # Increased from 0.15 for better visibility
            
            # X-axis (blue)
            ax.quiver(origin[0], origin[1], origin[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      color='blue', length=arrow_length, arrow_length_ratio=0.3, linewidth=1.5)
            # Y-axis (red)
            ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      color='red', length=arrow_length, arrow_length_ratio=0.3, linewidth=1.5)
            # Z-axis (black)
            ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      color='black', length=arrow_length, arrow_length_ratio=0.3, linewidth=1.5)
    else:
        # Default behavior: show sparse indicators for the whole trajectory
        step = max(1, len(positions)) // 20
        for i in range(0, len(positions), step):
            q = orientations[i]
            # Skip invalid quaternions
            if not np.isclose(np.linalg.norm(q), 1.0): continue
            R = Rotation.from_quat(q).as_matrix()
            forward_vec = R[:, 2] # Z-axis
            ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                      forward_vec[0], forward_vec[1], forward_vec[2],
                      color='orange', length=0.1, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory in World Space')
    
    # Add a custom legend for the basis vectors if they were plotted
    handles, labels = ax.get_legend_handles_labels()
    if selected_poses is not None:
        legend_elements = [ Line2D([0], [0], color='blue', lw=2, label='X-axis'),
                            Line2D([0], [0], color='red', lw=2, label='Y-axis'),
                            Line2D([0], [0], color='black', lw=2, label='Z-axis') ]
        handles.extend(legend_elements)

    ax.legend(handles=handles)
    
    # Make axes equal
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_plots and output_dir:
        output_path = Path(output_dir) / "camera_trajectory_3d.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D trajectory plot to: {output_path}")
    
    plt.show()
    
    # 2. Position vs time plots
    fig, axes = plt.subplots(3, 1, figsize=(fig_size[0], 10))
    
    axes[0].plot(timestamps, positions[:, 0], 'r-', linewidth=1)
    axes[0].set_ylabel('X Position')
    axes[0].set_title('Camera Position vs Time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(timestamps, positions[:, 1], 'g-', linewidth=1)
    axes[1].set_ylabel('Y Position')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(timestamps, positions[:, 2], 'b-', linewidth=1)
    axes[2].set_ylabel('Z Position')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].grid(True, alpha=0.3)
    
    if save_plots and output_dir:
        output_path = Path(output_dir) / "position_vs_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved position vs time plot to: {output_path}")
    
    plt.show()
    
    # 3. Orientation vs time plots
    fig, axes = plt.subplots(4, 1, figsize=(fig_size[0], 12))
    
    for i in range(4):
        axes[i].plot(timestamps, orientations[:, i], linewidth=1)
        axes[i].set_ylabel(f'Quaternion {i}')
        axes[i].grid(True, alpha=0.3)
    
    axes[0].set_title('Camera Orientation (Quaternion) vs Time')
    axes[-1].set_xlabel('Time (seconds)')
    
    if save_plots and output_dir:
        output_path = Path(output_dir) / "orientation_vs_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved orientation vs time plot to: {output_path}")
    
    plt.show()
    
    # 4. 2D trajectory projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY projection
    axes[0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.7)
    axes[0].scatter(positions[0, 0], positions[0, 1], c='blue', s=50, marker='o', label='Start')
    axes[0].scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, marker='o', label='End')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Projection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # XZ projection
    axes[1].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1, alpha=0.7)
    axes[1].scatter(positions[0, 0], positions[0, 2], c='blue', s=50, marker='o', label='Start')
    axes[1].scatter(positions[-1, 0], positions[-1, 2], c='red', s=50, marker='o', label='End')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    # YZ projection
    axes[2].plot(positions[:, 1], positions[:, 2], 'b-', linewidth=1, alpha=0.7)
    axes[2].scatter(positions[0, 1], positions[0, 2], c='blue', s=50, marker='o', label='Start')
    axes[2].scatter(positions[-1, 1], positions[-1, 2], c='red', s=50, marker='o', label='End')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Projection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        output_path = Path(output_dir) / "trajectory_projections.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory projections to: {output_path}")
    
    plt.show()

def main():
    """
    Main function to analyze pose data from command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze pose data from .npz file')
    parser.add_argument('pose_npz', type=str, help='Path to pose data .npz file')
    parser.add_argument('--selected-poses-npz', type=str, help='Path to selected poses .npz file from the vision pipeline for comparison')
    parser.add_argument('--output-dir', type=str, help='Directory to save plots')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots to files')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Analyze poses
    analyze_poses(args.pose_npz, output_dir, save_plots=not args.no_save, selected_poses_path=args.selected_poses_npz)

if __name__ == "__main__":
    main() 