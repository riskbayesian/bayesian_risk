import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

def save_video_visualizations(video_segments, rgb_frames, output_dir="obj_track_sam"):
    """Save video frames with overlaid segmentation masks and object IDs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Delete all existing images in the folder first
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(output_dir, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print(f"Saving frames to {output_dir}/ directory...")
    
    # Define stable colors for consistent object coloring across frames
    colors = [
        [255, 0, 0],    # Bright red
        [0, 255, 0],    # Bright green  
        [0, 0, 255],    # Bright blue
        [255, 255, 0],  # Bright yellow
        [255, 0, 255],  # Bright magenta
        [0, 255, 255],  # Bright cyan
        [255, 128, 0],  # Bright orange
        [128, 0, 255],  # Bright purple
        [255, 192, 203], # Pink
        [128, 128, 0],  # Olive
        [0, 128, 128],  # Teal
        [128, 0, 0],    # Maroon
    ]
    
    # Track which color each object ID gets (for consistency across frames)
    obj_id_to_color_idx = {}
    
    for frame_idx, frame_data in tqdm(video_segments.items(), desc="Saving visualization frames", unit="frame"):
        obj_ids = list(frame_data.keys())
        masks_np = np.array([frame_data[obj_id] for obj_id in obj_ids])
        
        # Get the original RGB frame
        rgb_frame = rgb_frames[frame_idx]
        
        # Create a copy for visualization
        vis_frame = rgb_frame.copy()
        
        # Store label information for drawing after masks
        label_info = []
        
        # First pass: Draw all masks
        for i, obj_id in enumerate(obj_ids):
            mask = masks_np[i]  # Shape: (1, height, width) - need to squeeze
            
            # Remove the extra dimension if present
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # Now shape: (height, width)
            
            if mask.sum() == 0:  # Skip empty masks
                continue
                
            # Assign stable color to this object ID
            if obj_id not in obj_id_to_color_idx:
                obj_id_to_color_idx[obj_id] = len(obj_id_to_color_idx) % len(colors)
            
            color_idx = obj_id_to_color_idx[obj_id]
            color = colors[color_idx]
            
            # Create a colored mask
            colored_mask = np.zeros_like(rgb_frame)
            colored_mask[mask] = color
            
            # Overlay the colored mask
            vis_frame[mask] = colored_mask[mask]
            
            # Find mask center for label placement
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                label_info.append((obj_id, center_x, center_y))
        
        # Second pass: Draw all labels on top of masks
        for obj_id, center_x, center_y in label_info:
            label = f"Obj{obj_id}"
            # Use larger font and thicker border for better visibility
            font_scale = 1.0
            thickness = 2
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Draw black background rectangle with padding
            padding = 5
            cv2.rectangle(vis_frame, 
                         (center_x - padding, center_y - text_size[1] - padding), 
                         (center_x + text_size[0] + padding, center_y + padding), 
                         (0, 0, 0), -1)
            
            # Draw white text
            cv2.putText(vis_frame, label, (center_x, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Save the frame
        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
    
    print(f"All frames saved to {output_dir}/ directory")
    print(f"Processed {len(video_segments)} frames with {len(obj_id_to_color_idx)} unique objects")

def create_video_from_vis(input_dir="obj_track_sam", output_filename="object_tracking_video.mp4", fps=30):
    """Create a video from the visualization frames saved by save_video_visualizations."""
    
    # Get all frame files and sort them
    frame_pattern = os.path.join(input_dir, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    
    if not frame_files:
        print(f"No frame files found in {input_dir}/")
        return
    
    print(f"Found {len(frame_files)} frames in {input_dir}/")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_files[0]}")
        return
    
    height, width, channels = first_frame.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Create output path in the same directory as input frames
    output_path = os.path.join(input_dir, output_filename)
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Could not create video writer for {output_path}")
        return
    
    print(f"Creating video: {output_path} at {fps} FPS...")
    
    # Write each frame to the video
    for i, frame_file in enumerate(tqdm(frame_files, desc="Writing frames to video", unit="frame")):
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_file}")

    # Release everything
    out.release()
    
    print(f"Video saved successfully: {output_path}")
    print(f"Total frames: {len(frame_files)}")
    print(f"Duration: {len(frame_files)/fps:.2f} seconds")

def track_from_bbs(rgb_frames, boxes_np):
    sam2_checkpoint = "open_world_risk/src/pretrained_models/sam2.1_hiera_large.pt"
    # Use the config name relative to the sam2 package's config root so Hydra can resolve it
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    frame_count = 0
    chunk_size = 100

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=rgb_frames[frame_count:chunk_size])

    # Add each box as a separate object id (1..K)
    for obj_idx, box_xyxy in enumerate(boxes_np):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_idx + 1,
            box=box_xyxy,
            clear_old_points=True,
            normalize_coords=True,
        )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    while frame_count < len(rgb_frames):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[frame_count] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            frame_count += 1
            if frame_count % chunk_size == 0:
                break

        if frame_count == len(rgb_frames):
            break

        print(f"resetting predictor frame_count {frame_count}")
        inference_state = predictor.init_state(video_path=rgb_frames[frame_count:chunk_size+frame_count])
        predictor.reset_state(inference_state)

        last_frame_masks = video_segments[frame_count - 1]  # Get masks from previous frame
        
        # Add each object from the last frame to the new chunk
        for obj_idx, (obj_id, mask) in enumerate(last_frame_masks.items()):
            # Remove extra dimension if present
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            
            if mask.sum() == 0:  # Skip empty masks
                continue
                
            # Find bounding box from mask
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                box = [x_min, y_min, x_max, y_max]
                
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,  # First frame of new chunk
                    obj_id=obj_id,
                    box=box,
                    clear_old_points=True,
                    normalize_coords=True,
                )

    return video_segments


