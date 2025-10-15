import os
import cv2
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional

from inference.predictor import VisualizationDemo
from inference.visualizer import TrackVisualizer
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode

def process_numpy_imgs(numpy_images: np.ndarray, save: bool = False, batch_size_processing: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Process a batch of numpy images and return model predictions.
    
    Args:
        numpy_images (np.ndarray): Batch of images with shape (B, H, W, C) in 8-bit RGB format
        save (bool): Whether to save visualization outputs for debugging
        batch_size_processing (int): Number of frames to process at once (default: 3)
    
    Returns:
        Tuple containing:
        - scores (List[np.ndarray]): Confidence scores for each image
        - labels (List[np.ndarray]): Class labels for each image  
        - masks (List[np.ndarray]): Segmentation masks for each image
    """
    
    # Initialize the model
    weights_path = './pretrained_models/trained_model.pth'
    cfg_file = './configs/hoist/hoistformer.yaml'
    demo_inference = VisualizationDemo(cfg_file, weights_path)
    
    # Convert batch to list of individual images
    batch_size = numpy_images.shape[0]
    frames_list = []
    
    for i in range(batch_size):
        # Convert RGB to BGR (OpenCV format)
        img_bgr = numpy_images[i][:, :, ::-1]  # RGB -> BGR
        frames_list.append(img_bgr)
    
    # Get model predictions - process in smaller batches to avoid memory issues
    all_scores = []
    all_labels = []
    all_masks = []
    
    # Use the provided batch_size_processing parameter
    print(f"Processing {len(frames_list)} frames in batches of {batch_size_processing}...")
    
    for i in range(0, len(frames_list), batch_size_processing):
        batch_frames = frames_list[i:i + batch_size_processing]
        batch_end = min(i + batch_size_processing, len(frames_list))
        print(f'Processing batch {i//batch_size_processing + 1}/{(len(frames_list) + batch_size_processing - 1)//batch_size_processing} (frames {i}-{batch_end-1})')
        
        try:
            batch_predictions = demo_inference.predictor(batch_frames)
            
            # Extract batch results
            batch_scores = batch_predictions["pred_scores"]
            batch_labels = batch_predictions["pred_labels"]
            batch_masks = batch_predictions["pred_masks"]
            
            # Filter by confidence threshold
            keep_mask = np.array(batch_scores) > 0.5
            
            if len(batch_scores) > 0 and np.any(keep_mask):
                # Reorganize masks by frame for this batch
                batch_frame_masks = list(zip(*batch_masks))
                
                # Process each frame in this batch
                for frame_idx_in_batch in range(len(batch_frames)):
                    global_frame_idx = i + frame_idx_in_batch
                    
                    if frame_idx_in_batch < len(batch_frame_masks):
                        # Get masks for this frame
                        frame_mask_tensors = torch.stack(batch_frame_masks[frame_idx_in_batch], dim=0)
                        frame_masks_filtered = frame_mask_tensors[keep_mask]
                        
                        # Store results for this frame
                        all_scores.append(np.array(batch_scores)[keep_mask])
                        all_labels.append(np.array(batch_labels)[keep_mask])
                        all_masks.append(frame_masks_filtered.cpu().numpy())
                    else:
                        # No masks for this frame
                        all_scores.append(np.array([]))
                        all_labels.append(np.array([]))
                        all_masks.append(np.array([]))
            else:
                # No detections in this batch
                for _ in range(len(batch_frames)):
                    all_scores.append(np.array([]))
                    all_labels.append(np.array([]))
                    all_masks.append(np.array([]))
                    
        except Exception as e:
            print(f"Error processing batch {i//batch_size_processing + 1}: {e}")
            # If batch fails, try with smaller batch size
            if batch_size_processing > 1:
                print(f"Retrying with smaller batch size: {batch_size_processing//2}")
                return process_numpy_imgs(numpy_images, save, batch_size_processing//2)
            else:
                raise e
    
    print(f"Processed all {len(all_scores)} frames successfully")
    
    # Prepare return values
    scores_per_image = all_scores
    labels_per_image = all_labels
    masks_per_image = all_masks
    
    # Optional visualization saving
    if save:
        save_debug_visualizations_batch(frames_list, scores_per_image, labels_per_image, masks_per_image, demo_inference.metadata)
    
    return scores_per_image, labels_per_image, masks_per_image


def save_debug_visualizations(frames: List[np.ndarray], predictions: Dict, metadata) -> None:
    """
    Save visualization outputs for debugging purposes.
    
    Args:
        frames (List[np.ndarray]): List of BGR images
        predictions (Dict): Model predictions
        metadata: Dataset metadata for visualization
    """
    
    # Create output directory
    debug_dir = './debug_outputs/'
    os.makedirs(debug_dir, exist_ok=True)
    
    # Extract prediction components
    image_size = predictions["image_size"]
    pred_scores = predictions["pred_scores"]
    pred_labels = predictions["pred_labels"]
    pred_masks = predictions["pred_masks"]
    
    # Filter by confidence
    keep_instance_ids_np = np.array(pred_scores) > 0.5
    keep_scores = []
    keep_labels = []
    keep_instance_ids = []
    
    for id, is_keep in enumerate(keep_instance_ids_np):
        if is_keep:
            keep_instance_ids.append(id + 1)
            keep_scores.append(pred_scores[id])
            keep_labels.append(pred_labels[id])
    
    keep_instance_ids_remap = list(range(1, len(keep_instance_ids) + 1))
    
    # Reorganize masks by frame
    frame_masks = list(zip(*pred_masks))
    
    # Generate visualizations for each frame
    for frame_idx, frame in enumerate(frames):
        visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
        ins = Instances(image_size)
        
        if len(pred_scores) > 0:
            ins.scores = keep_scores
            ins.pred_classes = keep_labels
            ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)[keep_instance_ids_np]
        
        vis_output = visualizer.draw_instance_predictions(predictions=ins, instance_ids=keep_instance_ids_remap)
        
        # Save visualization
        vis_img = vis_output.get_image()
        save_path = os.path.join(debug_dir, f'debug_frame_{frame_idx:03d}.jpg')
        cv2.imwrite(save_path, vis_img)
        
        print(f"Saved debug visualization: {save_path}")


def save_debug_visualizations_batch(frames: List[np.ndarray], scores_per_image: List[np.ndarray], 
                                   labels_per_image: List[np.ndarray], masks_per_image: List[np.ndarray], 
                                   metadata) -> None:
    """
    Save visualization outputs for debugging purposes with batch processing results.
    
    Args:
        frames (List[np.ndarray]): List of BGR images
        scores_per_image (List[np.ndarray]): Scores for each image
        labels_per_image (List[np.ndarray]): Labels for each image
        masks_per_image (List[np.ndarray]): Masks for each image
        metadata: Dataset metadata for visualization
    """
    
    # Create output directory
    debug_dir = './debug_outputs/'
    os.makedirs(debug_dir, exist_ok=True)
    
    # Generate visualizations for each frame
    for frame_idx, frame in enumerate(frames):
        if frame_idx < len(scores_per_image):
            visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
            
            if len(scores_per_image[frame_idx]) > 0:
                # Create Instances object with detections
                ins = Instances((frame.shape[0], frame.shape[1]))
                ins.scores = scores_per_image[frame_idx]
                ins.pred_classes = labels_per_image[frame_idx]
                ins.pred_masks = torch.from_numpy(masks_per_image[frame_idx])
                instance_ids = list(range(1, len(scores_per_image[frame_idx]) + 1))
                
                vis_output = visualizer.draw_instance_predictions(predictions=ins, instance_ids=instance_ids)
            else:
                # Skip visualization for frames with no detections
                print(f"Skipping visualization for frame {frame_idx} (no detections)")
                continue
            
            # Save visualization
            vis_img = vis_output.get_image()
            save_path = os.path.join(debug_dir, f'debug_frame_{frame_idx:03d}.jpg')
            cv2.imwrite(save_path, vis_img)
            
            print(f"Saved debug visualization: {save_path}")


# Example usage function (not called automatically)
def example_usage():
    """
    Example of how to use the process_numpy_imgs function.
    """
    # Create dummy batch of images (B, H, W, C) in RGB format
    batch_size = 3
    height, width = 480, 640
    dummy_images = np.random.randint(0, 255, (batch_size, height, width, 3), dtype=np.uint8)
    
    # Process the images
    scores, labels, masks = process_numpy_imgs(dummy_images, save=True)
    
    # Print results
    for i in range(batch_size):
        print(f"Image {i}:")
        print(f"  Scores: {scores[i]}")
        print(f"  Labels: {labels[i]}")
        print(f"  Masks shape: {masks[i].shape if len(masks[i]) > 0 else 'No masks'}")
        print()


def data_loader(npz_path: str):
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

if __name__ == "__main__":
    npz_path = "data/rgbd_camera2_1752279418.npz"

    rgb_frames, depth_frames = data_loader(npz_path)

    scores, labels, masks = process_numpy_imgs(rgb_frames, save=True)
    # Print results
    for i in range(len(rgb_frames)):
        print(f"Image {i}:")
        print(f"  Scores: {scores[i]}")
        print(f"  Labels: {labels[i]}")
        print(f"  Masks shape: {masks[i].shape if len(masks[i]) > 0 else 'No masks'}")
        print()