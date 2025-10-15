#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.stats import iqr

def analyze_timestamps(rgb_npz_path, depth_npz_path, pose_npz_path, output_dir=None):
    """
    Analyzes and visualizes timestamp data from RGB, Depth, and Pose .npz files.
    """
    print("Loading timestamps...")
    try:
        rgb_ts = np.load(rgb_npz_path)['timestamps']
        depth_ts = np.load(depth_npz_path)['timestamps']
        pose_ts = np.load(pose_npz_path)['timestamps']
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # --- Data Conversion & Summary ---
    print("\n--- Timestamp Summary ---")
    
    streams_out = {}
    
    for name, ts_arr_orig in [('RGB', rgb_ts), ('Depth', depth_ts), ('Pose', pose_ts)]:
        # Convert to seconds if they are large numbers (likely nanoseconds)
        if np.mean(ts_arr_orig) > 1e12:  # Heuristic for nanoseconds
            ts_arr = ts_arr_orig.astype(np.float64) / 1e9
            unit = "s"
            is_ns = True
        else:
            ts_arr = ts_arr_orig
            unit = " (original units)"
            is_ns = False

        streams_out[name] = ts_arr
        duration = ts_arr[-1] - ts_arr[0]
        frequency = (len(ts_arr) - 1) / duration if duration > 0 else 0
        
        print(f"{name}:")
        if is_ns:
            print(f"  - Converted from nanoseconds to seconds.")
        print(f"  - Entries: {len(ts_arr)}")
        print(f"  - Start Time: {ts_arr[0]:.3f}{unit}")
        print(f"  - End Time:   {ts_arr[-1]:.3f}{unit}")
        print(f"  - Duration:   {duration:.3f}{unit}")
        print(f"  - Avg Freq:   {frequency:.2f} Hz")

    # --- Inter-Frame Interval Analysis ---
    print("\n--- Inter-Frame Interval Analysis (Time between frames) ---")
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle('Distribution of Inter-Frame Intervals (Deltas)')
    for i, (name, ts) in enumerate(streams_out.items()):
        deltas = np.diff(ts)
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        
        print(f"{name} Deltas:")
        print(f"  - Mean: {mean_delta*1000:.3f} ms  (StdDev: {std_delta*1000:.3f} ms)")
        print(f"  - Min:  {np.min(deltas)*1000:.3f} ms | Max: {np.max(deltas)*1000:.3f} ms")

        # Use Freedman-Diaconis rule for robust binning
        bin_width = 2 * iqr(deltas) / (len(deltas) ** (1/3))
        if bin_width > 0:
            bins = int((np.max(deltas) - np.min(deltas)) / bin_width)
        else:
            bins = 30
        
        axes1[i].hist(deltas * 1000, bins=min(bins, 200), alpha=0.75)
        axes1[i].set_title(f'{name} Stream')
        axes1[i].set_xlabel('Time Difference (ms)')
        axes1[i].set_ylabel('Frequency')
        axes1[i].grid(True, alpha=0.5)

    if output_dir:
        plt.savefig(output_dir / "inter_frame_intervals.png", dpi=300)
    
    # --- Synchronization Analysis ---
    print("\n--- Synchronization Analysis (Closest timestamp between streams) ---")

    # 1. Find closest POSE for each RGB
    time_diffs_rgb_pose = []
    rgb_ts_sec = streams_out['RGB']
    pose_ts_sec = streams_out['Pose']
    depth_ts_sec = streams_out['Depth']

    for ts_rgb in rgb_ts_sec:
        diffs = np.abs(pose_ts_sec - ts_rgb)
        time_diffs_rgb_pose.append(np.min(diffs))
    
    time_diffs_rgb_pose = np.array(time_diffs_rgb_pose)

    print(f"RGB vs Pose Sync:")
    print(f"  - Mean Diff: {np.mean(time_diffs_rgb_pose)*1000:.3f} ms")
    print(f"  - Max Diff:  {np.max(time_diffs_rgb_pose)*1000:.3f} ms")

    # 2. Find closest DEPTH for each RGB
    time_diffs_rgb_depth = []
    for ts_rgb in rgb_ts_sec:
        diffs = np.abs(depth_ts_sec - ts_rgb)
        time_diffs_rgb_depth.append(np.min(diffs))

    time_diffs_rgb_depth = np.array(time_diffs_rgb_depth)
    
    print(f"RGB vs Depth Sync:")
    print(f"  - Mean Diff: {np.mean(time_diffs_rgb_depth)*1000:.3f} ms")
    print(f"  - Max Diff:  {np.max(time_diffs_rgb_depth)*1000:.3f} ms")

    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
    fig2.suptitle('Distribution of Synchronization Time Differences')

    axes2[0].hist(time_diffs_rgb_pose * 1000, bins=50, alpha=0.75, color='orange')
    axes2[0].set_title('RGB vs Pose')
    axes2[0].set_xlabel('Time Difference (ms)')
    axes2[0].set_ylabel('Frequency')
    axes2[0].grid(True, alpha=0.5)

    axes2[1].hist(time_diffs_rgb_depth * 1000, bins=50, alpha=0.75, color='green')
    axes2[1].set_title('RGB vs Depth')
    axes2[1].set_xlabel('Time Difference (ms)')
    axes2[1].grid(True, alpha=0.5)
    
    if output_dir:
        plt.savefig(output_dir / "synchronization_analysis.png", dpi=300)
        print(f"\nPlots saved to '{output_dir}'")
        
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze timestamps from RGB, Depth, and Pose NPZ files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--rgb-npz', type=str, required=True, help='Path to RGB data .npz file')
    parser.add_argument('--depth-npz', type=str, required=True, help='Path to depth data .npz file')
    parser.add_argument('--pose-npz', type=str, required=True, help='Path to pose data .npz file')
    parser.add_argument('--output-dir', type=str, help='Directory to save analysis plots')
    
    args = parser.parse_args()
    
    output_path = None
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
    analyze_timestamps(args.rgb_npz, args.depth_npz, args.pose_npz, output_path)

if __name__ == "__main__":
    main() 