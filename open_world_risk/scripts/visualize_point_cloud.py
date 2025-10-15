import open3d as o3d
import argparse
import numpy as np
from pathlib import Path
import glob
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from src.point_cloud_3D import PointCloudGenerator
import gc
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_display_available():
    """
    Check if a display is available for visualization.
    """
    return 'DISPLAY' in os.environ and os.environ['DISPLAY'] != ''

def visualize_ply(ply_path):
    """
    Visualize a point cloud from a .ply file.
    
    Args:
        ply_path: Path to the .ply file
    """
    try:
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_points():
            raise ValueError("Point cloud is empty")
        return pcd
        
    except Exception as e:
        print(f"Error reading point cloud {ply_path}: {str(e)}")
        return None

def visualize_multiple_plys(ply_dir, cloud_type, save_ply=False, max_dirs=100):
    """
    Visualize multiple point clouds from a directory.
    
    Args:
        ply_dir: Directory containing .ply files
        cloud_type: Type of point cloud to visualize ('rgb' or 'CLIP')
        save_ply: Whether to save the combined point cloud as a .ply file
        max_dirs: Maximum number of directories to process (with even spacing)
    """
    try:
        # Find all .ply files in the directory and subdirectories
        ply_pattern = f"**/*point_cloud_{cloud_type}_*.ply"
        ply_files = sorted(glob.glob(str(Path(ply_dir) / ply_pattern), recursive=True))
        
        if not ply_files:
            raise ValueError(f"No {cloud_type} point cloud files found in {ply_dir}")
            
        print(f"Found {len(ply_files)} {cloud_type} point cloud files")
        
        # Group files by their parent directory
        files_by_dir = {}
        for ply_file in ply_files:
            parent_dir = Path(ply_file).parent
            if parent_dir not in files_by_dir:
                files_by_dir[parent_dir] = []
            files_by_dir[parent_dir].append(ply_file)
        
        # Sort directories to ensure consistent ordering
        sorted_dirs = sorted(files_by_dir.keys())
        total_dirs = len(sorted_dirs)
        print(f"Found {total_dirs} directories with {cloud_type} point cloud files")
        
        # Sample directories with even spacing if we have more than max_dirs
        if total_dirs > max_dirs:
            # Calculate step size for even spacing
            step = total_dirs / max_dirs
            selected_indices = [int(i * step) for i in range(max_dirs)]
            selected_dirs = [sorted_dirs[i] for i in selected_indices]
            print(f"Sampling {max_dirs} directories with even spacing from {total_dirs} total directories")
        else:
            selected_dirs = sorted_dirs
            print(f"Processing all {total_dirs} directories")
        
        # Read point clouds from selected directories
        point_clouds = []
        for i, directory in enumerate(selected_dirs):
            dir_files = files_by_dir[directory]
            for ply_file in dir_files:
                pcd = visualize_ply(ply_file)
                if pcd is not None:
                    point_clouds.append(pcd)
                    print(f"Loaded point cloud {len(point_clouds)}: {ply_file}")
        
        if not point_clouds:
            raise ValueError("No valid point clouds could be loaded")
            
        print(f"Successfully loaded {len(point_clouds)} point clouds from {len(selected_dirs)} directories")
        
        # Combine all point clouds if saving is requested
        if save_ply:
            print("Combining point clouds for saving...")
            combined_points = []
            combined_colors = []
            
            for pcd in point_clouds:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5
                combined_points.append(points)
                combined_colors.append(colors)
            
            # Stack all points and colors
            all_points = np.vstack(combined_points)
            all_colors = np.vstack(combined_colors)
            
            # Create combined point cloud
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(all_points)
            combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
            
            # Save to data directory
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            output_filename = f"combined_{cloud_type}_point_cloud.ply"
            output_path = data_dir / output_filename
            
            success = o3d.io.write_point_cloud(str(output_path), combined_pcd)
            if success:
                print(f"Combined point cloud saved to: {output_path}")
                print(f"Total points: {len(all_points)}")
            else:
                print(f"Failed to save combined point cloud to: {output_path}")
        
        # Check if display is available
        if not check_display_available():
            print("No display available. Saving point cloud statistics instead.")
            for i, pcd in enumerate(point_clouds):
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                print(f"Point cloud {i+1}: {len(points)} points")
                if colors is not None:
                    print(f"  Color range: {colors.min():.3f} - {colors.max():.3f}")
                print(f"  Point range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
                      f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
                      f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            return
        
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        if not vis.create_window():
            print("Failed to create visualization window. Display may not be available.")
            return
        
        # Add all point clouds to visualizer
        for pcd in point_clouds:
            vis.add_geometry(pcd)
        
        # Set visualization options
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = 2.0
            opt.background_color = np.array([0, 0, 0])
            opt.light_on = True
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error visualizing point clouds: {str(e)}")

def run_pca_on_point_clouds(ply_dir, n_components=3, batch_size=10):
    """
    Run PCA on CLIP embeddings from multiple point clouds.
    
    Args:
        ply_dir: Directory containing point cloud files and their embeddings
        n_components: Number of PCA components to compute
        batch_size: Number of directories to process in each batch to manage memory
    
    Returns:
        pca: Fitted PCA object
        transformed_embeddings: Dictionary mapping point cloud names to their PCA-transformed embeddings
    """
    # Find all CLIP point cloud files
    ply_pattern = f"**/*point_cloud_CLIP_*.ply"
    ply_files = sorted(glob.glob(str(Path(ply_dir) / ply_pattern), recursive=True))
    
    if not ply_files:
        raise ValueError(f"No CLIP point cloud files found in {ply_dir}")
    
    print(f"Found {len(ply_files)} CLIP point cloud files")
    
    # Group files by their parent directory
    files_by_dir = {}
    for ply_file in ply_files:
        parent_dir = Path(ply_file).parent
        if parent_dir not in files_by_dir:
            files_by_dir[parent_dir] = []
        files_by_dir[parent_dir].append(ply_file)
    
    # Sort directories to ensure consistent ordering
    sorted_dirs = sorted(files_by_dir.keys())
    print(f"Found {len(sorted_dirs)} directories with CLIP point cloud files")
    
    # Load all embeddings, grouped by directory
    all_embeddings = []
    cloud_names = []
    successful_dirs = []
    pc_generator = PointCloudGenerator()
    
    # Process directories in batches to manage memory
    for batch_start in range(0, len(sorted_dirs), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_dirs))
        batch_dirs = sorted_dirs[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(sorted_dirs) + batch_size - 1)//batch_size}")
        
        for i, directory in enumerate(batch_dirs):
            global_idx = batch_start + i
            dir_files = files_by_dir[directory]
            dir_success = True
            dir_embeddings = []
            dir_names = []
            
            print(f"Processing directory {global_idx + 1}/{len(sorted_dirs)}: {directory}")
            
            for ply_file in dir_files:
                try:
                    points, colors, embeddings, metadata = pc_generator.load_point_cloud_with_embeddings(ply_file)
                    
                    # Check if embeddings are per-point (should have same number of rows as points)
                    if embeddings.shape[0] != len(points):
                        print(f"  Warning: Embedding count ({embeddings.shape[0]}) doesn't match point count ({len(points)}) in {ply_file}")
                        print("  This suggests embeddings are mask-level, not point-level. Skipping...")
                        dir_success = False
                        continue
                        
                    dir_embeddings.append(embeddings)
                    dir_names.append(Path(ply_file).stem)
                    #print(f"  Loaded per-point embeddings from: {ply_file} - {embeddings.shape}")
                except Exception as e:
                    print(f"  Error loading embeddings from {ply_file}: {str(e)}")
                    dir_success = False
                    continue
            
            if dir_success and dir_embeddings:
                successful_dirs.append(directory)
                all_embeddings.extend(dir_embeddings)
                cloud_names.extend(dir_names)
            else:
                print(f"  Warning: Directory {directory} had loading errors")
        
        # Clear some memory after each batch
        import gc
        gc.collect()
    
    # If the last directory had errors, remove its embeddings
    if len(successful_dirs) < len(sorted_dirs):
        last_dir = sorted_dirs[-1]
        if last_dir not in successful_dirs:
            print(f"Ignoring last directory due to errors: {last_dir}")
            # Embeddings from failed directories were never added, so no removal needed
    
    if not all_embeddings:
        raise ValueError("No valid per-point embeddings could be loaded")
    
    print(f"Successfully loaded embeddings from {len(successful_dirs)} directories")
    print(f"Total embedding arrays: {len(all_embeddings)}")
    
    # Stack all embeddings with memory management
    print("Stacking embeddings...")
    try:
        # Store embedding shapes before clearing the list
        embedding_shapes = [emb.shape[0] for emb in all_embeddings]
        
        X = np.vstack(all_embeddings)
        print(f"Total embeddings for PCA: {X.shape}")
        
        # Clear the list to free memory
        del all_embeddings
        gc.collect()
        
        # Remove points with zero embeddings (points not covered by any mask)
        print("Filtering non-zero embeddings...")
        non_zero_mask = np.any(X != 0, axis=1)
        X_filtered = X[non_zero_mask]
        print(f"Non-zero embeddings for PCA: {X_filtered.shape}")
        
        if len(X_filtered) == 0:
            raise ValueError("No non-zero embeddings found for PCA")
        
        # Fit PCA
        print("Fitting PCA...")
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X_filtered)
        
        # Create full transformed array with zeros for points without embeddings
        print("Creating full transformed array...")
        full_transformed = np.zeros((X.shape[0], n_components))
        full_transformed[non_zero_mask] = transformed
        
        # Clear X to free memory
        del X, X_filtered
        gc.collect()
        
        # Split transformed embeddings back into individual point clouds
        print("Splitting embeddings back to individual point clouds...")
        transformed_embeddings = {}
        start_idx = 0
        for i, cloud_name in enumerate(cloud_names):
            n_points = embedding_shapes[i]
            transformed_embeddings[cloud_name] = full_transformed[start_idx:start_idx + n_points]
            start_idx += n_points
        
        return pca, transformed_embeddings
        
    except MemoryError:
        print("Memory error occurred. Try reducing batch_size or processing fewer directories.")
        raise
    except Exception as e:
        print(f"Error during PCA processing: {str(e)}")
        raise

def run_iterative_pca_on_point_clouds(ply_dir, n_components=3, batch_size=1000, max_dirs=100):
    """
    Run iterative PCA on CLIP embeddings from multiple point clouds.
    This is memory-efficient for large datasets.
    
    Args:
        ply_dir: Directory containing point cloud files and their embeddings
        n_components: Number of PCA components to compute
        batch_size: Number of embeddings to process in each batch
        max_dirs: Maximum number of directories to process (with even spacing)
    
    Returns:
        pca: Fitted PCA object
        transformed_embeddings: Dictionary mapping point cloud names to their PCA-transformed embeddings
    """
    
    # Find all CLIP point cloud files
    ply_pattern = f"**/*point_cloud_CLIP_*.ply"
    ply_files = sorted(glob.glob(str(Path(ply_dir) / ply_pattern), recursive=True))
    
    if not ply_files:
        raise ValueError(f"No CLIP point cloud files found in {ply_dir}")
    
    print(f"Found {len(ply_files)} CLIP point cloud files")
    
    # Group files by their parent directory
    files_by_dir = {}
    for ply_file in ply_files:
        parent_dir = Path(ply_file).parent
        if parent_dir not in files_by_dir:
            files_by_dir[parent_dir] = []
        files_by_dir[parent_dir].append(ply_file)
    
    # Sort directories to ensure consistent ordering
    sorted_dirs = sorted(files_by_dir.keys())
    total_dirs = len(sorted_dirs)
    print(f"Found {total_dirs} directories with CLIP point cloud files")
    
    # Sample directories with even spacing if we have more than max_dirs
    if total_dirs > max_dirs:
        # Calculate step size for even spacing
        step = total_dirs / max_dirs
        selected_indices = [int(i * step) for i in range(max_dirs)]
        selected_dirs = [sorted_dirs[i] for i in selected_indices]
        print(f"Sampling {max_dirs} directories with even spacing from {total_dirs} total directories")
    else:
        selected_dirs = sorted_dirs
        print(f"Processing all {total_dirs} directories")
    
    # Initialize incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    pc_generator = PointCloudGenerator()
    
    # First pass: determine embedding dimension and count total non-zero embeddings
    embedding_length = None
    total_nonzero_embeddings = 0
    cloud_info = []  # Store (cloud_name, start_idx, n_points, n_nonzero) for each cloud
    
    print("First pass: counting embeddings and determining dimensions...")
    for directory in selected_dirs:
        dir_files = files_by_dir[directory]
        for ply_file in dir_files:
            try:
                points, colors, embeddings, metadata = pc_generator.load_point_cloud_with_embeddings(ply_file)
                
                if embedding_length is None:
                    embedding_length = embeddings.shape[1]
                elif embedding_length != embeddings.shape[1]:
                    raise ValueError(f"Embedding length mismatch: {embedding_length} != {embeddings.shape[1]}")
                
                # Count non-zero embeddings
                non_zero_mask = np.any(embeddings != 0, axis=1)
                n_nonzero = np.sum(non_zero_mask)
                total_nonzero_embeddings += n_nonzero
                
                cloud_name = Path(ply_file).stem
                cloud_info.append((cloud_name, embeddings.shape[0], n_nonzero))
                
            except Exception as e:
                print(f"Error loading embeddings from {ply_file}: {str(e)}")
                continue
    
    if total_nonzero_embeddings == 0:
        raise ValueError("No non-zero embeddings found")
    
    print(f"Total non-zero embeddings: {total_nonzero_embeddings}")
    print(f"Embedding dimension: {embedding_length}")
    
    # Second pass: fit incremental PCA
    print("Second pass: fitting incremental PCA...")
    current_nonzero_idx = 0
    
    for i, (cloud_name, n_points, n_nonzero) in enumerate(cloud_info):
        try:
            # Find the corresponding ply file
            ply_file = None
            for directory in selected_dirs:
                dir_files = files_by_dir[directory]
                for pf in dir_files:
                    if Path(pf).stem == cloud_name:
                        ply_file = pf
                        break
                if ply_file:
                    break
            
            if ply_file is None:
                print(f"Warning: Could not find file for cloud {cloud_name}")
                continue
                
            points, colors, embeddings, metadata = pc_generator.load_point_cloud_with_embeddings(ply_file)
            
            # Get non-zero embeddings
            non_zero_mask = np.any(embeddings != 0, axis=1)
            embeddings_nonzero = embeddings[non_zero_mask]
            
            if len(embeddings_nonzero) > 0:
                # Process in batches for memory efficiency
                for j in range(0, len(embeddings_nonzero), batch_size):
                    batch = embeddings_nonzero[j:j + batch_size]
                    ipca.partial_fit(batch)
                
                print(f"Fitted PCA on cloud {i+1}/{len(cloud_info)}: {cloud_name} ({len(embeddings_nonzero)} non-zero embeddings)")
            
            current_nonzero_idx += n_nonzero
            
        except Exception as e:
            print(f"Error processing embeddings from {ply_file}: {str(e)}")
            continue
        
        # Garbage collection every 10 clouds
        if (i + 1) % 10 == 0:
            gc.collect()
    
    print("PCA fitting completed!")
    
    # Third pass: transform embeddings
    print("Third pass: transforming embeddings...")
    transformed_embeddings = {}
    current_nonzero_idx = 0
    
    for i, (cloud_name, n_points, n_nonzero) in enumerate(cloud_info):
        try:
            # Find the corresponding ply file
            ply_file = None
            for directory in selected_dirs:
                dir_files = files_by_dir[directory]
                for pf in dir_files:
                    if Path(pf).stem == cloud_name:
                        ply_file = pf
                        break
                if ply_file:
                    break
            
            if ply_file is None:
                print(f"Warning: Could not find file for cloud {cloud_name}")
                continue
                
            points, colors, embeddings, metadata = pc_generator.load_point_cloud_with_embeddings(ply_file)
            
            # Get non-zero embeddings
            non_zero_mask = np.any(embeddings != 0, axis=1)
            embeddings_nonzero = embeddings[non_zero_mask]
            
            if len(embeddings_nonzero) > 0:
                # Transform non-zero embeddings
                transformed_nonzero = ipca.transform(embeddings_nonzero)
                
                # Create full transformed array for this cloud
                cloud_transformed = np.zeros((n_points, n_components))
                cloud_transformed[non_zero_mask] = transformed_nonzero
                
                transformed_embeddings[cloud_name] = cloud_transformed
                
                print(f"Transformed cloud {i+1}/{len(cloud_info)}: {cloud_name}")
            else:
                # All embeddings are zero
                transformed_embeddings[cloud_name] = np.zeros((n_points, n_components))
            
        except Exception as e:
            print(f"Error transforming embeddings from {ply_file}: {str(e)}")
            continue
        
        # Garbage collection every 10 clouds
        if (i + 1) % 10 == 0:
            gc.collect()
    
    print("Transformation completed!")
    
    return ipca, transformed_embeddings

def visualize_pca_results(transformed_embeddings, ply_dir, save_ply=False, max_dirs=100):
    """
    Visualize PCA results by creating a conglomerate point cloud.
    
    Args:
        transformed_embeddings: Dictionary of transformed embeddings
        ply_dir: Directory containing original point clouds
        save_ply: Whether to save the combined point cloud as a .ply file
        max_dirs: Maximum number of directories to process (with even spacing)
    """
    # Find all point cloud files
    ply_pattern = f"**/*point_cloud_CLIP_*.ply"
    ply_files = sorted(glob.glob(str(Path(ply_dir) / ply_pattern), recursive=True))
    
    # Group files by their parent directory
    files_by_dir = {}
    for ply_file in ply_files:
        parent_dir = Path(ply_file).parent
        if parent_dir not in files_by_dir:
            files_by_dir[parent_dir] = []
        files_by_dir[parent_dir].append(ply_file)
    
    # Sort directories to ensure consistent ordering
    sorted_dirs = sorted(files_by_dir.keys())
    total_dirs = len(sorted_dirs)
    print(f"Found {total_dirs} directories with CLIP point cloud files")
    
    # Sample directories with even spacing if we have more than max_dirs
    if total_dirs > max_dirs:
        # Calculate step size for even spacing
        step = total_dirs / max_dirs
        selected_indices = [int(i * step) for i in range(max_dirs)]
        selected_dirs = [sorted_dirs[i] for i in selected_indices]
        print(f"Sampling {max_dirs} directories with even spacing from {total_dirs} total directories")
    else:
        selected_dirs = sorted_dirs
        print(f"Processing all {total_dirs} directories")
    
    # Load all point clouds from selected directories
    all_points = []
    all_colors = []
    pc_generator = PointCloudGenerator()
    
    for directory in tqdm(selected_dirs, desc="Loading point clouds for visualization"):
        dir_files = files_by_dir[directory]
        for ply_file in dir_files:
            try:
                points, colors, _, _ = pc_generator.load_point_cloud_with_embeddings(ply_file)
                all_points.append(points)
                all_colors.append(colors)
            except Exception as e:
                print(f"Error loading point cloud {ply_file}: {str(e)}")
                continue
    
    # Combine all points
    combined_points = np.vstack(all_points)
    
    # Create visualization colors from PCA results
    # Each point should have its own PCA-transformed embedding
    combined_embeddings = np.vstack(list(transformed_embeddings.values()))
    
    # Normalize to [0,1] range for visualization
    # Handle the case where some embeddings might be zero (points not covered by masks)
    non_zero_mask = np.any(combined_embeddings != 0, axis=1)
    
    if np.any(non_zero_mask):
        # Normalize only non-zero embeddings
        non_zero_embeddings = combined_embeddings[non_zero_mask]
        min_vals = non_zero_embeddings.min(axis=0)
        max_vals = non_zero_embeddings.max(axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        combined_colors = np.zeros_like(combined_embeddings)
        combined_colors[non_zero_mask] = (non_zero_embeddings - min_vals) / range_vals
        
        # Set points without embeddings to a neutral color (gray)
        combined_colors[~non_zero_mask] = 0.5
    else:
        # All embeddings are zero, use neutral color
        combined_colors = np.full_like(combined_embeddings, 0.5)
    
    # Save combined point cloud if requested
    if save_ply:
        print("Saving combined PCA point cloud...")
        combined_pcd = pc_generator.to_open3d(combined_points, combined_colors)
        
        # Save to data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        output_filename = "combined_pca_point_cloud.ply"
        output_path = data_dir / output_filename
        
        success = o3d.io.write_point_cloud(str(output_path), combined_pcd)
        if success:
            print(f"Combined PCA point cloud saved to: {output_path}")
            print(f"Total points: {len(combined_points)}")
        else:
            print(f"Failed to save combined PCA point cloud to: {output_path}")
    
    # Check if display is available
    if not check_display_available():
        print("No display available. Saving PCA visualization as image instead.")
        
        # Create 2D scatter plot of first two PCA components
        plt.figure(figsize=(12, 8))
        
        # Plot points with non-zero embeddings
        if np.any(non_zero_mask):
            plt.scatter(combined_embeddings[non_zero_mask, 0], 
                       combined_embeddings[non_zero_mask, 1], 
                       c=combined_colors[non_zero_mask], 
                       alpha=0.6, s=1)
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Visualization of Point Cloud Embeddings')
        plt.colorbar()
        
        # Save the plot
        output_path = Path(ply_dir) / "pca_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to: {output_path}")
        plt.close()
        return
    
    # Create and visualize point cloud
    pcd = pc_generator.to_open3d(combined_points, combined_colors)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    if not vis.create_window():
        print("Failed to create visualization window. Display may not be available.")
        return
    
    # Add point cloud to visualizer
    vis.add_geometry(pcd)
    
    # Set visualization options
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = 2.0
        opt.background_color = np.array([0, 0, 0])
        opt.light_on = True
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud(s) from .ply file(s)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ply-path', type=str, help='Path to a single .ply file')
    group.add_argument('--ply-dir', type=str, help='Directory containing .ply files')
    parser.add_argument('--cloud-type', type=str, choices=['rgb', 'CLIP', 'CLIP-raw'], 
                      help='Type of point cloud to visualize (required when using --ply-dir)')
    parser.add_argument('--n-components', type=int, default=3,
                      help='Number of PCA components to compute (only used with CLIP-raw)')
    parser.add_argument('--batch-size', type=int, default=10,
                      help='Number of directories to process in each batch (only used with CLIP-raw)')
    parser.add_argument('--save-ply', action='store_true',
                      help='Save the combined point cloud as a .ply file in the data directory')
    parser.add_argument('--max-dirs', type=int, default=100,
                      help='Maximum number of directories to process with even spacing (default: 100)')
    
    args = parser.parse_args()
    
    if args.ply_path:
        # Single file visualization
        pcd = visualize_ply(args.ply_path)
        if pcd is not None:
            # Save single point cloud if requested
            if args.save_ply:
                print("Saving single point cloud...")
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                
                output_filename = "single_point_cloud.ply"
                output_path = data_dir / output_filename
                
                success = o3d.io.write_point_cloud(str(output_path), pcd)
                if success:
                    print(f"Point cloud saved to: {output_path}")
                    points = np.asarray(pcd.points)
                    print(f"Total points: {len(points)}")
                else:
                    print(f"Failed to save point cloud to: {output_path}")
            
            # Check if display is available
            if not check_display_available():
                print("No display available. Showing point cloud statistics instead.")
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                print(f"Point cloud: {len(points)} points")
                if colors is not None:
                    print(f"Color range: {colors.min():.3f} - {colors.max():.3f}")
                print(f"Point range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
                      f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
                      f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
                return
            
            # Create visualization window
            vis = o3d.visualization.Visualizer()
            if not vis.create_window():
                print("Failed to create visualization window. Display may not be available.")
                return
            
            # Add point cloud to visualizer
            vis.add_geometry(pcd)
            
            # Set visualization options
            opt = vis.get_render_option()
            if opt is not None:
                opt.point_size = 2.0
                opt.background_color = np.array([0, 0, 0])
                opt.light_on = True
            
            # Run visualizer
            vis.run()
            vis.destroy_window()
    else:
        # Directory-based visualization
        if not args.cloud_type:
            parser.error("--cloud-type is required when using --ply-dir")
            
        if args.cloud_type == 'CLIP-raw':
            # Run iterative PCA on all point clouds
            ipca, transformed_embeddings = run_iterative_pca_on_point_clouds(args.ply_dir, args.n_components, args.batch_size, args.max_dirs)

            # Visualize results
            visualize_pca_results(transformed_embeddings, args.ply_dir, args.save_ply, args.max_dirs)
        else:
            # Regular visualization
            visualize_multiple_plys(args.ply_dir, args.cloud_type, args.save_ply, args.max_dirs)

if __name__ == "__main__":
    main()
