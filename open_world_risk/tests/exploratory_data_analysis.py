import torch
import os
import glob
from pprint import pprint

def analyze_pt_file(file_path):
    """
    Analyze and print contents of a .pt file
    
    Args:
        file_path: Path to the .pt file
    Returns:
        int: Number of 4x4 matrices in the file
    """
    print(f"\nAnalyzing file: {file_path}")
    print("-" * 50)
    
    try:
        # Try loading with weights_only=False first
        try:
            data = torch.load(file_path, weights_only=False)
        except Exception:
            # If that fails, try with weights_only=True
            data = torch.load(file_path, weights_only=True)
        
        print("File loaded successfully!")
        print("\nData type:", type(data))
        
        if isinstance(data, torch.Tensor):
            print("\nTensor shape:", data.shape)
            print("Tensor dtype:", data.dtype)
            num_matrices = data.shape[0]  # Number of 4x4 matrices is the first dimension
            
            # Only print detailed contents for the first file we encounter
            if not hasattr(analyze_pt_file, 'example_shown'):
                print("\nExample matrix contents (first 5 matrices):")
                pprint(data[:5])
                analyze_pt_file.example_shown = True
            
            return num_matrices
            
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return 0

def main():
    # Base directory containing the .pt files
    base_dir = "/home/ubuntu/knucklesVirgina/remote_sync/rgbd_06_14_2025/2025-04-03_163252/training_poses"
    
    # Find all pose files
    pose_files = sorted(glob.glob(os.path.join(base_dir, "poses_*.pt")))
    
    if not pose_files:
        print(f"No pose files found in {base_dir}")
        return
    
    print(f"Found {len(pose_files)} pose files")
    print("\nAnalyzing each file...")
    
    # Analyze each file and collect statistics
    total_matrices = 0
    file_stats = []
    
    for file_path in pose_files:
        file_name = os.path.basename(file_path)
        num_matrices = analyze_pt_file(file_path)
        total_matrices += num_matrices
        file_stats.append((file_name, num_matrices))
    
    # Print summary
    print("\nSummary of matrices per file:")
    print("-" * 50)
    for file_name, count in file_stats:
        print(f"{file_name}: {count} matrices")
    print("-" * 50)
    print(f"Total number of 4x4 matrices across all files: {total_matrices}")

if __name__ == "__main__":
    main()
