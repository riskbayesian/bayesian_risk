import os
import os.path as osp
import cv2
import numpy as np
import argparse
from inference.predictor import VisualizationDemo

def process_frames_in_batches(frames_bgr_np, demo_inference, batch_size=5):
    """Process frames in smaller batches to reduce memory usage"""
    all_predictions = []
    all_vis_output = []
    
    for i in range(0, len(frames_bgr_np), batch_size):
        batch_frames = frames_bgr_np[i:i + batch_size]
        print(f'Processing batch {i//batch_size + 1}/{(len(frames_bgr_np) + batch_size - 1)//batch_size}')
        
        try:
            predictions, vis_output = demo_inference.run_on_video(batch_frames)
            all_predictions.append(predictions)
            all_vis_output.extend(vis_output)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # If batch fails, try with smaller batch size
            if batch_size > 1:
                print(f"Retrying with smaller batch size: {batch_size//2}")
                return process_frames_in_batches(frames_bgr_np, demo_inference, batch_size//2)
            else:
                raise e
    
    return all_predictions, all_vis_output

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for Demo')
    parser.add_argument('--video_frames_path', required=True, metavar='path to images', help='path to images')
    parser.add_argument('--batch_size', type=int, default=5, help='number of frames to process at once (default: 5)')
    args = parser.parse_args()

    weights_path = './pretrained_models/trained_model.pth'
    cfg_file = './configs/hoist/hoistformer.yaml'
    demo_inference = VisualizationDemo(cfg_file, weights_path)

    root = args.video_frames_path
    output_path = './output_results/'
    videos = os.listdir(root)
    for vid_idx, vid in enumerate(videos):
        print(f'Processing video: {vid_idx}, total: {len(videos)}')
        frames_bgr_np = []
        frames = sorted(os.listdir(osp.join(root, vid)))
        for frm in frames:
            frm_np = cv2.imread(osp.join(root, vid, frm))
            frames_bgr_np.append(frm_np)
        
        # Process frames in batches
        predictions, vis_output = process_frames_in_batches(frames_bgr_np, demo_inference, args.batch_size)
        
        os.makedirs(osp.join(output_path, vid), exist_ok=True)
        for frm_idx, vis_img in enumerate(vis_output):
            frm = vis_img.get_image()
            save_path = osp.join(output_path, vid, f'output_frm_{frm_idx}.jpg')
            cv2.imwrite(save_path, frm)