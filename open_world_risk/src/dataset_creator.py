# Standard library imports
import gc
import os
import pickle
import shutil
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import glob
import yaml
import csv

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm
import petname as PetnamePkg
from pprint import pprint

# Local imports
from open_world_risk.src.dataset_loader import (
    generateHistogram,
    plot_histograms,
    generate_final_dataset,
    test_histogram_dataset,
)
from open_world_risk.video_processing.process_video_utils import (
    get_image_embedd_white_black_lst,
    get_images_config,
    save_sam_masks_as_images,
)
from open_world_risk.src.video_datastructures import (
    VideoFrameData1,
    ManipulationTimeline,
)
from open_world_risk.src.dino_features import DinoFeatures

def match_dino_to_label(dino_feats, label_dino):
    """
    Match DINO features to labels based on cosine similarity.
    
    Args:
        dino_feats: PyTorch tensor of DINO features to be matched
        label_dino: PyTorch tensor of reference DINO features for labels
        label_str: List of label strings corresponding to label_dino features
        
    Returns:
        list: List of tuples [label_idx, dino_feat] where label_idx is the index
              of the most similar label_dino feature
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(dino_feats, torch.Tensor):
        dino_feats = torch.tensor(dino_feats)
    if not isinstance(label_dino, torch.Tensor):
        label_dino = torch.tensor(label_dino)
    
    # Ensure both tensors have the same dtype (convert to float32)
    dino_feats = dino_feats.float()
    label_dino = label_dino.float()
    
    # Normalize features for cosine similarity
    dino_feats_norm = torch.nn.functional.normalize(dino_feats, p=2, dim=1)
    label_dino_norm = torch.nn.functional.normalize(label_dino, p=2, dim=1)
    
    # Compute cosine similarity matrix
    # Shape: (num_dino_feats, num_labels)
    cosine_sim = torch.mm(dino_feats_norm, label_dino_norm.t())
    
    # Find the label with highest similarity for each DINO feature
    # Shape: (num_dino_feats,)
    max_sim_indices = torch.argmax(cosine_sim, dim=1)
    
    # Create list of [label_idx, dino_feat] tuples
    label_dino_matches = []
    for i, label_idx in enumerate(max_sim_indices):
        label_dino_matches.append([label_idx.item(), dino_feats[i]])
    
    return label_dino_matches

def dino_to_label_write(label_dino_matches: List[List], label_str_lst: List[str], output_dir: str = ".", csv_filename: str = "label_to_id.csv", 
                        pt_filename: str = "id_to_dino_feat.pt", write_csv: bool = True):
    """
    Write label-to-DINO mapping to CSV and PyTorch files with linking IDs.
    
    Args:
        label_dino_matches: List of [label_idx, dino_feat] tuples from match_dino_to_label
        label_str: List of label strings corresponding to label indices
        output_dir: Directory to save the output files (default: current directory)
        csv_filename: Name of the CSV file (default: "label_to_id.csv")
        pt_filename: Name of the PyTorch file (default: "id_to_dino_feat.pt")
        write_csv: Whether to write the CSV file (default: True)
    
    Returns:
        tuple: (csv_path, pt_path) - paths to the created files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, csv_filename) if write_csv else None
    pt_path = os.path.join(output_dir, pt_filename)
    
    # Prepare data structures
    csv_data = []  # List of (label, id) tuples - 1:1 mapping

    if write_csv:
        for idx, label in enumerate(label_str_lst):
            csv_data.append((label, idx))

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['label', 'id'])  # Header
            writer.writerows(csv_data)

    ids = torch.tensor([x[0] for x in label_dino_matches], dtype=torch.long)
    data = torch.stack([x[1] for x in label_dino_matches])  # shape [N, D]
    assert ids.shape[0] == data.shape[0], "ids and data are different lengths!!!"

    torch.save({"ids": ids, "data": data}, pt_path)

    print(f"Written label-to-ID mapping:")
    if write_csv:
        print(f"  - CSV file: {csv_path} ({len(csv_data)} entries)")
    print(f"  - PyTorch file: {pt_path} ids: ({ids.shape[0]} data: {data.shape[0]} DINO features)")
    
    return csv_path, pt_path

def match_dino_to_label_wrapper(video_frame_data, all_avg_tensor, all_labels, batch_size=1_000, 
                                output_dir=".", csv_filename="label_to_id.csv", pt_filename="id_to_dino_feat.pt",
                                filename_suffix="", write_csv_flag=True):
    """
    Process DINO features from video frames in batches and write label mappings.
    
    Args:
        video_frame_data: VideoFrameData1 object containing frame data
        all_avg_tensor: PyTorch tensor of reference DINO features for labels
        all_labels: List of label strings corresponding to all_avg_tensor features
        batch_size: Number of DINO features to process in each batch (default: 1000)
        output_dir: Directory to save the output files (default: current directory)
        csv_filename: Name of the CSV file (default: "label_to_id.csv")
        pt_filename: Name of the PyTorch file (default: "id_to_dino_feat.pt")
        write_csv: Whether to write the CSV file (default: True)
    
    Returns:
        tuple: (csv_path, pt_path) - paths to the created files
    """

    # Get total number of frames for progress tracking
    total_frames = len(video_frame_data.frames)
    print(f"Processing {total_frames} frames in batches of {batch_size}")
    
    # Process frames in batches
    for batch_idx, batch_start in enumerate(tqdm(range(0, total_frames, batch_size), desc="Processing batches")):
        batch_end = min(batch_start + batch_size, total_frames)
        batch_frames = video_frame_data.frames[batch_start:batch_end]
        
        batch_dino_feats = []
        for frame_idx, frame in enumerate(batch_frames):
            dino_feats = frame["dino_embeddings"] # TODO: add the random sampling feats
            batch_dino_feats.append(dino_feats)

        batch_dino_feats = torch.cat(batch_dino_feats, dim=0)
        label_dino_matches = match_dino_to_label(batch_dino_feats, all_avg_tensor)

        write_csv = True if batch_start == 0 and write_csv_flag else False
        pt_filename_batch = pt_filename.replace('.pt', '') + filename_suffix + f"_{batch_idx}.pt"
        csv_path, pt_path = dino_to_label_write(label_dino_matches, all_labels, output_dir=output_dir, csv_filename=csv_filename, 
                                pt_filename=pt_filename_batch, write_csv=write_csv)

def discover_dataset_folders(processed_video_dir, verbose=False) -> List[str]:
    """
    Discover all folders ending with '_dataset' in the specified directory(ies).
    
    Args:
        processed_video_dir: Single directory path or list of directory paths to search
        
    Returns:
        List[str]: List of discovered dataset folder paths
    """
    discovered_paths = []
    if processed_video_dir is None: 
        return discovered_paths
    
    if isinstance(processed_video_dir, list):
        # Handle multiple directories
        for dir_path in processed_video_dir:
            if os.path.exists(dir_path):
                # Find all subdirectories ending with _dataset
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path) and item.endswith("_dataset"):
                        discovered_paths.append(item_path)
            else:
                raise FileNotFoundError(f"processed_video_dir does not exist: {dir_path}")
    else:
        if os.path.exists(processed_video_dir):
            for item in os.listdir(processed_video_dir):
                item_path = os.path.join(processed_video_dir, item)
                if os.path.isdir(item_path) and item.endswith("_dataset"):
                    discovered_paths.append(item_path)
        else:
            raise FileNotFoundError(f"processed_video_dir does not exist: {processed_video_dir}")
    
    if discovered_paths:
        if verbose:
            print(f"Auto-discovered {len(discovered_paths)} dataset folders:")
            for path in discovered_paths:
                print(f"  - {path}")
    else:
        raise ValueError("No _dataset folders found in processed_video_dir")
    
    return discovered_paths

def load_cfg_file(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        text = f.read()
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

    # Validate required keys
    required_keys = ["processed_video_dir", "processed_video_paths", "camera_intrinsics_path", 
                    "output_dir", "object_white_list", "object_black_list", "object_grey_list", 
                    "dino_per_obj", "max_distance"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    processed_video_dir = cfg["processed_video_dir"]
    discovered_paths = discover_dataset_folders(processed_video_dir)
    
    # Add discovered paths to processed_video_paths
    if cfg["processed_video_paths"] is None:
        cfg["processed_video_paths"] = []
    cfg["processed_video_paths"].extend(discovered_paths)


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

def load_processed_video_dataset(dataset_path: str) -> tuple[VideoFrameData1, ManipulationTimeline, list[int]]:
    if not dataset_path.endswith("_dataset"):
        raise ValueError(f"Dataset path should end with '_dataset': {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Load video frame data from HDF5
    video_frame_data = VideoFrameData1.load_frames(dataset_path)
    
    # Load frame indices
    frame_indices_file = os.path.join(dataset_path, "frame_indices.pkl")
    if os.path.exists(frame_indices_file):
        with open(frame_indices_file, 'rb') as f:
            frame_indices = pickle.load(f)
    else:
        raise FileNotFoundError(f"frame_indices.pkl not found at {frame_indices_file}")
    
    # Load saved manipulation metadata from manip_timeline_data folder
    manip_timeline_dir = os.path.join(dataset_path, "manip_timeline_data")
    if os.path.exists(manip_timeline_dir):
        manip_metadata = ManipulationTimeline.load(manip_timeline_dir)
    else:
        raise FileNotFoundError(f"manip_timeline_data not found at {manip_timeline_dir}")

    print(f"Loaded dataset from {dataset_path}")
    print(f"  - Frames: {len(video_frame_data.frames)}")
    print(f"  - Manipulation periods: {len(manip_metadata.manipulation_periods)}")
    print(f"  - Frame indices: [{frame_indices[0]}, {frame_indices[-1]}]")
    
    return video_frame_data, manip_metadata, frame_indices

def custom_combine_labels(white_avg_tensor, grey_avg_tensor, white_labels, grey_labels):
    # NOTE: This is a custom written function to combine labels for objects for our table experiments

    white_labels_to_combine = [["pot", "cup", "water_bottle_metal", "solo_cup", "single-blank-cup-free-png", "water_bottle"], 
                                ["stove", "electric_stove"], ["penguin-chick-plush", "stuffed_penguin"]]
    # no entries in dataset
    white_labels_to_drop = ["paper_plate", "dry_easer"]
    white_labels_dict = dict()
    
    grey_avg_mean = grey_avg_tensor.mean(dim=0, keepdim=True)  # [1, d]

    for i in range(len(white_labels)):
        white_label = white_labels[i]
        white_tensor = white_avg_tensor[i]
        
        # Skip labels that should be dropped
        if white_label in white_labels_to_drop:
            continue
            
        # Check if this label should be combined with others
        combined = False
        for combine_group in white_labels_to_combine:
            if white_label in combine_group:
                # Check if any other label from this group is already in the dict
                for existing_key in white_labels_dict.keys():
                    if existing_key in combine_group:
                        # Add to existing group
                        white_labels_dict[existing_key].append(white_tensor)
                        combined = True
                        break
                
                if not combined:
                    # First label from this group, create new entry
                    white_labels_dict[white_label] = [white_tensor]
                    combined = True
                break
        
        if not combined:
            # Label not in any combine group, add as is
            white_labels_dict[white_label] = [white_tensor]

    out_labels = []
    out_avg_tensor = []
    for key in white_labels_dict:
        # Average the tensors for this key
        if len(white_labels_dict[key]) == 1:
            avg_tensor = white_labels_dict[key][0]
        else:
            avg_tensor = torch.stack(white_labels_dict[key]).mean(dim=0)
        
        out_labels.append(key)
        out_avg_tensor.append(avg_tensor)

    out_labels.append(grey_labels[0])
    out_avg_tensor += grey_avg_mean
    out_avg_tensor = torch.stack(out_avg_tensor)

    print(f"out_labels: {len(out_labels)} entries: {out_labels}")
    print(f"out_avg_tensor: {out_avg_tensor.shape}")

    return out_avg_tensor, out_labels

def runner(processed_video_paths: List[str], camera_intrinsics_path: str, output_dir: str,
          white_list_images: List, black_list_images: List, grey_list_images: List,
          white_list_filenames: List[str], black_list_filenames: List[str], grey_list_filenames: List[str],
          label_to_id_filename: str, id_to_dino_feat_filename: str,
          max_distance: float = 0.5, dino_per_obj: int = 15, min_entries: int = 10, verbose: bool = False) -> str:
    """
    Main runner function that processes multiple datasets and generates histograms.
    
    Args:
        processed_video_paths: List of paths to processed video datasets
        camera_intrinsics_path: Path to camera intrinsics JSON file
        output_dir: Output directory for generated datasets
        white_list_images: List of white list image arrays
        black_list_images: List of black list image arrays
        grey_list_images: List of grey list image arrays
        white_list_filenames: List of white list filenames
        black_list_filenames: List of black list filenames
        grey_list_filenames: List of grey list filenames
        max_distance: Maximum distance for histogram range
        dino_per_obj: Number of DINO samples per object
        verbose: Whether to print verbose output
        
    Returns:
        str: Path to the output directory
    """
    print("=" * 80)
    print("DATASET CREATOR - Processing Multiple Video Datasets")
    print("=" * 80)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    dino_featurizer = DinoFeatures()

    # Process each dataset
    all_histograms = {}
    all_video_frame_data = []
    all_manip_metadata = []
    all_frame_indices = []
    dataset_paths = []
    total_frames_processed = 0
    processed_video_paths = sorted(processed_video_paths)
    
    # mainly for plotting what's in/not in our dataset
    white_avg_tensor, black_avg_tensor, grey_avg_tensor, white_labels, black_labels, grey_labels = get_image_embedd_white_black_lst(dino_featurizer, white_list_images, black_list_images, grey_list_images, 
                                                                                                                                    white_list_filenames, black_list_filenames, grey_list_filenames)
    
    all_labels = white_labels + grey_labels
    all_avg_tensor = torch.cat([white_avg_tensor, grey_avg_tensor])

    custom_avg_tensor, custom_labels = custom_combine_labels(white_avg_tensor, grey_avg_tensor, white_labels, grey_labels)

    for i, dataset_path in enumerate(processed_video_paths):
        print(f"\n{'='*60}")
        print(f"Processing dataset {i+1}/{len(processed_video_paths)}: {dataset_path}")

        video_frame_data, manip_metadata, frame_indices = load_processed_video_dataset(dataset_path)

        write_csv_flag = True if i == 0 else False
        filename_suffix = f"_video_{i}"
        match_dino_to_label_wrapper(video_frame_data, custom_avg_tensor, custom_labels, batch_size=1_000, 
                                output_dir=output_dir, csv_filename=label_to_id_filename, 
                                pt_filename=id_to_dino_feat_filename, filename_suffix=filename_suffix, 
                                write_csv_flag=write_csv_flag)

        histograms = generateHistogram(video_frame_data, manip_metadata, frame_indices, 
                                        max_distance=max_distance, min_entries=min_entries, verbose=verbose)
        assert len(histograms) != 0, "generateHistogram failed, empty histograms returned"

        all_histograms.update(histograms)
        all_video_frame_data.append(video_frame_data)
        all_manip_metadata.append(manip_metadata)
        all_frame_indices.extend(frame_indices)
        total_frames_processed += len(frame_indices)
        
        dataset_path = generate_final_dataset(
            histograms, video_frame_data,
            output_dir=output_dir, dino_per_obj=dino_per_obj
        )
        dataset_paths.append(dataset_path)

        dataset_name = os.path.basename(dataset_path).replace("_dataset", "")
        dataset_name = os.path.splitext(dataset_name)[0]  # Remove .pt extension
        dataset_name = dataset_name.replace("histogram_tensor_", "").replace("_", "-")  # Clean up naming
        histogram_output_dir = os.path.join(output_dir, f"histograms_{dataset_name}")
        
        save_sam_masks_as_images(video_frame_data.frames[0], histogram_output_dir, frame_indices[0], verbose=verbose)
        plot_histograms(
            histograms, video_frame_data,
            white_avg_tensor, white_labels, grey_avg_tensor, grey_labels,
            max_distance=max_distance, output_dir=histogram_output_dir, verbose=verbose
        )

        video_frame_data.delete_memory()
        manip_metadata.delete_memory()
        video_frame_data, manip_metadata = None, None
        del video_frame_data, manip_metadata
        
    test_histogram_dataset(dataset_paths, batch_size=64, num_ctrl=10, output_dir=output_dir, verbose=verbose)
    
    print(f"\n{'='*60}")
    print("DATASET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Final dataset: {dataset_path}")
    print(f"Total histograms: {len(all_histograms)}")
    print(f"Total frames processed: {total_frames_processed}")
    
    return output_dir

def main():
    """
    Main function that loads configuration and runs dataset creation.
    """
    # Load configuration
    config_path = "open_world_risk/src/dataset_config.yaml"
    cfg = load_cfg_file(config_path)
    
    # Extract parameters
    processed_video_paths = cfg["processed_video_paths"]
    camera_intrinsics_path = cfg["camera_intrinsics_path"]
    output_dir = cfg["output_dir"]

    # Ensure output directory exists and copy the config file into it for record-keeping
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(output_dir, "dataset_config.yaml"))

    object_white_list = cfg["object_white_list"]
    object_black_list = cfg["object_black_list"]
    object_grey_list = cfg["object_grey_list"]
    white_list_images = cfg["white_list_images"]
    black_list_images = cfg["black_list_images"]
    grey_list_images = cfg["grey_list_images"]
    white_list_filenames = cfg["white_list_filenames"]
    black_list_filenames = cfg["black_list_filenames"]
    grey_list_filenames  = cfg["grey_list_filenames"]
    max_distance = cfg["max_distance"]
    dino_per_obj = cfg["dino_per_obj"]
    min_entries = cfg["min_entries"]
    verbose = cfg.get("verbose", False)
    label_to_id_filename = cfg["label_to_id_filename"]
    id_to_dino_feat_filename = cfg["id_to_dino_feat_filename"]
    
    print("Dataset Creator Configuration:")
    print(f"  - Processed video paths:")
    for path in processed_video_paths:
        print(f"    {path}")
    print(f"  - Camera intrinsics: {camera_intrinsics_path}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - White list images: {len(white_list_images)} images")
    print(f"  - Black list images: {len(black_list_images)} images")
    print(f"  - Grey list images: {len(grey_list_images)} images")
    print(f"  - White list filenames ({len(white_list_filenames)}): {', '.join(white_list_filenames[:3])}{'...' if len(white_list_filenames) > 3 else ''}")
    print(f"  - Black list filenames ({len(black_list_filenames)}): {', '.join(black_list_filenames[:3])}{'...' if len(black_list_filenames) > 3 else ''}")
    print(f"  - Grey list filenames ({len(grey_list_filenames)}): {', '.join(grey_list_filenames[:3])}{'...' if len(grey_list_filenames) > 3 else ''}")
    print(f"  - Max distance: {max_distance}")
    print(f"  - Min entries: {min_entries}")
    print(f"  - DINO per object: {dino_per_obj}")
    print(f"  - Verbose: {verbose}")

    # Run dataset creation
    result_dir = runner(
        processed_video_paths=processed_video_paths,
        camera_intrinsics_path=camera_intrinsics_path,
        output_dir=output_dir,
        white_list_images=white_list_images,
        black_list_images=black_list_images,
        grey_list_images=grey_list_images,
        white_list_filenames=white_list_filenames,
        black_list_filenames=black_list_filenames,
        grey_list_filenames=grey_list_filenames,
        max_distance=max_distance, 
        dino_per_obj=dino_per_obj,
        min_entries=min_entries,
        label_to_id_filename = label_to_id_filename,
        id_to_dino_feat_filename = id_to_dino_feat_filename,
        verbose=verbose
    )
    
    print(f"\nDataset creation completed successfully!")
    print(f"Results saved to: {result_dir}")

if __name__ == "__main__":
    main()