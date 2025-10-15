# Standard library imports
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Set

# Third-party imports
import numpy as np
import torch
from pydantic import BaseModel, Field, validator
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle
import h5py

# Pydantic models for well-defined data structures

class VideoMetadata(BaseModel):
    """Metadata for a single video"""
    
    class Config:
        arbitrary_types_allowed = True
    
    frame_count: int = Field(..., description="Total number of frames")
    resolution: Tuple[int, int] = Field(..., description="Video resolution (width, height)")
    k_matrix: Optional[np.ndarray] = Field(None, description="Camera intrinsic matrix (3x3 K matrix)")
    camera_intrinsics_path: Optional[str] = Field(None, description="Path to camera intrinsics JSON file")
    
    @validator('k_matrix')
    def validate_k_matrix(cls, v):
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError('k_matrix must be a numpy array')
            if v.shape != (3, 3):
                raise ValueError('k_matrix must be a 3x3 array')
        return v

class StraightLine(BaseModel):
    """Straight line representation between two 3D points"""
    
    class Config:
        arbitrary_types_allowed = True
    
    start: torch.Tensor = Field(..., description="Starting point (x, y, z)")
    end: torch.Tensor = Field(..., description="Ending point (x, y, z)")
    direction: torch.Tensor = Field(..., description="Normalized unit direction vector")
    length: float = Field(..., description="Length of the line segment")

    def delete_memory(self):
        """Delete all tensors and clear memory for this StraightLine object"""
        # Delete all torch tensors
        tensor_fields = ['start', 'end', 'direction']
        
        for field in tensor_fields:
            if hasattr(self, field) and getattr(self, field) is not None:
                tensor = getattr(self, field)
                if isinstance(tensor, torch.Tensor):
                    del tensor
                    setattr(self, field, None)
        
        # Clear GPU cache if tensors were on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class ManipulationTimeline(BaseModel):
    """Timeline data for object manipulation per video"""
    class Config:
        arbitrary_types_allowed = True
    manipulation_timeline: List[int] = Field(..., description="1's and 0's per frame indicating manipulation")
    manipulation_periods: List[Tuple[int, int]] = Field(..., description="List of (start_frame, end_frame) tuples for manipulation periods")
    manipulation_points: List[Tuple[torch.Tensor, torch.Tensor]] = Field(..., description="List of xyz points each (start, end) tuples from manipulation periods")
    straight_lines: List[StraightLine] = Field(..., description="Straight line representations for manipulation paths")

    def delete_memory(self):
        """Delete all memory for this ManipulationTimeline object"""
        # Delete manipulation_points tensors
        if hasattr(self, 'manipulation_points') and self.manipulation_points is not None:
            for start_pt, end_pt in self.manipulation_points:
                if isinstance(start_pt, torch.Tensor):
                    del start_pt
                if isinstance(end_pt, torch.Tensor):
                    del end_pt
            self.manipulation_points.clear()
            self.manipulation_points = None
        
        # Delete straight_lines
        if hasattr(self, 'straight_lines') and self.straight_lines is not None:
            for straight_line in self.straight_lines:
                if straight_line is not None:
                    straight_line.delete_memory()
                    del straight_line
            self.straight_lines.clear()
            self.straight_lines = None
        
        # Clear simple lists
        if hasattr(self, 'manipulation_timeline'):
            self.manipulation_timeline.clear()
            self.manipulation_timeline = None
        
        if hasattr(self, 'manipulation_periods'):
            self.manipulation_periods.clear()
            self.manipulation_periods = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save(self, output_dir: str) -> str:
        """
        Save ManipulationTimeline to a directory.
        
        Args:
            output_dir: Directory to save the manipulation timeline data
            
        Returns:
            str: Path to the saved directory
        """
        out = Path(output_dir)
        manip_dir = out / "manip_timeline_data"
        manip_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manipulation timeline (simple list)
        timeline_file = manip_dir / "manipulation_timeline.pkl"
        with open(timeline_file, 'wb') as f:
            pickle.dump(self.manipulation_timeline, f)
        
        # Save manipulation periods (list of tuples)
        periods_file = manip_dir / "manipulation_periods.pkl"
        with open(periods_file, 'wb') as f:
            pickle.dump(self.manipulation_periods, f)
        
        # Save manipulation points (list of tensor tuples)
        points_file = manip_dir / "manipulation_points.pkl"
        with open(points_file, 'wb') as f:
            pickle.dump(self.manipulation_points, f)
        
        # Save straight lines (list of StraightLine objects)
        straight_lines_file = manip_dir / "straight_lines.pkl"
        with open(straight_lines_file, 'wb') as f:
            pickle.dump(self.straight_lines, f)
        
        # Save metadata
        metadata = {
            "num_periods": len(self.manipulation_periods),
            "num_points": len(self.manipulation_points),
            "num_straight_lines": len(self.straight_lines),
            "timeline_length": len(self.manipulation_timeline)
        }
        metadata_file = manip_dir / "manipulation_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved ManipulationTimeline to {manip_dir}")
        print(f"  - Timeline length: {len(self.manipulation_timeline)}")
        print(f"  - Manipulation periods: {len(self.manipulation_periods)}")
        print(f"  - Manipulation points: {len(self.manipulation_points)}")
        print(f"  - Straight lines: {len(self.straight_lines)}")
        
        return str(manip_dir)

    @classmethod
    def load(cls, dir_path: str) -> "ManipulationTimeline":
        """
        Load ManipulationTimeline from a directory.
        
        Args:
            dir_path: Directory containing the manipulation timeline data
            
        Returns:
            ManipulationTimeline: Loaded manipulation timeline instance
        """
        base = Path(dir_path)
        
        if not base.exists():
            raise FileNotFoundError(f"ManipulationTimeline directory not found: {dir_path}")
        
        # Load manipulation timeline
        timeline_file = base / "manipulation_timeline.pkl"
        if timeline_file.exists():
            with open(timeline_file, 'rb') as f:
                manipulation_timeline = pickle.load(f)
        else:
            manipulation_timeline = []
        
        # Load manipulation periods
        periods_file = base / "manipulation_periods.pkl"
        if periods_file.exists():
            with open(periods_file, 'rb') as f:
                manipulation_periods = pickle.load(f)
        else:
            manipulation_periods = []
        
        # Load manipulation points
        points_file = base / "manipulation_points.pkl"
        if points_file.exists():
            with open(points_file, 'rb') as f:
                manipulation_points = pickle.load(f)
        else:
            manipulation_points = []
        
        # Load straight lines
        straight_lines_file = base / "straight_lines.pkl"
        if straight_lines_file.exists():
            with open(straight_lines_file, 'rb') as f:
                straight_lines = pickle.load(f)
        else:
            straight_lines = []
        
        # Load metadata for validation
        metadata_file = base / "manipulation_metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            print(f"Loaded ManipulationTimeline from {base}")
            print(f"  - Timeline length: {metadata.get('timeline_length', len(manipulation_timeline))}")
            print(f"  - Manipulation periods: {metadata.get('num_periods', len(manipulation_periods))}")
            print(f"  - Manipulation points: {metadata.get('num_points', len(manipulation_points))}")
            print(f"  - Straight lines: {metadata.get('num_straight_lines', len(straight_lines))}")
        else:
            print(f"Loaded ManipulationTimeline from {base} (no metadata file)")
        
        return cls(
            manipulation_timeline=manipulation_timeline,
            manipulation_periods=manipulation_periods,
            manipulation_points=manipulation_points,
            straight_lines=straight_lines
        )

class VideoManipulationData(BaseModel):
    """Manipulation data for all objects in a video per video"""
    video_manipulation_data: ManipulationTimeline = Field(..., description="Manipulation data for all objects in a video")

class VideoFrameData1(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    frames: List[Dict] = Field(..., description="A list of TensorDict's, each having tensors representing all data")

    def delete_memory(self):
        """Delete all memory for this VideoFrameData1 object and its nested FrameData objects"""
        if hasattr(self, 'frames') and self.frames is not None:
            # Delete each FrameData object
            for i, frame_data in enumerate(self.frames):
                if frame_data is not None:
                    del frame_data
            # Clear the frames list
            self.frames.clear()
            self.frames = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save_frames(self, out_dir: str, config_path: str, video_name: str, frame_indices: list[int], 
                   to_cpu: bool = True, compression: str = "gzip", compression_opts: int = 9) -> str:
        """
        Save frames to HDF5 format with built-in compression.
        
        Args:
            out_dir: Output directory
            config_path: Path to config file
            video_name: Name of the video
            frame_indices: List of frame indices
            to_cpu: Whether to move tensors to CPU
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (1-9 for gzip)
        """
        
        # Type assertions for input validation
        assert isinstance(out_dir, str), f"out_dir must be str, got {type(out_dir)}"
        assert isinstance(config_path, str), f"config_path must be str, got {type(config_path)}"
        assert isinstance(video_name, str), f"video_name must be str, got {type(video_name)}"
        assert isinstance(frame_indices, list), f"frame_indices must be list, got {type(frame_indices)}"
        assert all(isinstance(idx, int) for idx in frame_indices), "All frame_indices must be integers"
        assert isinstance(to_cpu, bool), f"to_cpu must be bool, got {type(to_cpu)}"
        assert isinstance(compression, str), f"compression must be str, got {type(compression)}"
        assert isinstance(compression_opts, int), f"compression_opts must be int, got {type(compression_opts)}"

        out = Path(out_dir)
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        hdf5_file = frames_dir / "frames.h5"
        
        with h5py.File(hdf5_file, 'w') as f:
            # Create a group for metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['video_name'] = video_name
            metadata_group.attrs['total_frames'] = len(self.frames)
            metadata_group.attrs['compression'] = compression
            metadata_group.attrs['compression_opts'] = compression_opts
            
            # Create a group for frames
            frames_group = f.create_group('frames')
            
            # Process each frame
            for i, frame in enumerate(tqdm(self.frames, desc="Saving frames to HDF5")):
                frame_group = frames_group.create_group(f'frame_{i:06d}')
                
                for k, v in frame.items():
                    if isinstance(v, torch.Tensor):
                        v_np = v.detach()
                        v_np = v_np.cpu() if to_cpu else v_np
                        v_np = v_np.to_dense() if v_np.is_sparse else v_np
                        v_np = v_np.numpy()
                        
                        # Store tensor data with compression
                        frame_group.create_dataset(
                            k, 
                            data=v_np, 
                            compression=compression,
                            compression_opts=compression_opts,
                            chunks=True,
                            shuffle=True
                        )
                        
                        # Store tensor metadata
                        frame_group.attrs[f'{k}_dtype'] = str(v.dtype)
                        frame_group.attrs[f'{k}_shape'] = list(v.shape)
                        frame_group.attrs[f'{k}_device'] = str(v.device)
                    else:
                        # Store non-tensor data as attributes
                        frame_group.attrs[k] = v

        # Save config file copy
        config_src = Path(config_path)
        config_dst = out / config_src.name
        shutil.copy2(config_src, config_dst)
        
        # Save video name to text file
        video_name_file = out / "video_name.txt"
        with open(video_name_file, 'w') as f:
            f.write(video_name)

        # Save frame indices to pickle file
        frame_indices_file = out / "frame_indices.pkl"
        with open(frame_indices_file, 'wb') as f:
            pickle.dump(frame_indices, f)

        print(f"Saved {len(self.frames)} frames to HDF5 file: {hdf5_file}")
        print(f"Saved config file: {config_dst}")
        print(f"Saved video name file: {video_name_file}")
        print(f"Saved frame indices file: {frame_indices_file}")
        return str(out)


    @classmethod
    def load_frames(cls, dir_path: str, device: str = "cpu") -> "VideoFrameData1":
        """
        Load frames from HDF5 format.
        """
        base = Path(dir_path)
        frames_dir = base / "frames"
        hdf5_file = frames_dir / "frames.h5"
        
        if not hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
        
        frames = []
        
        with h5py.File(hdf5_file, 'r') as f:
            frames_group = f['frames']
            frame_keys = sorted(frames_group.keys())
            
            print(f"Loading {len(frame_keys)} frames from HDF5...")
            
            for frame_key in tqdm(frame_keys, desc="Loading frames from HDF5"):
                frame_group = frames_group[frame_key]
                frame_data = {}
                
                # Load tensor data
                for dataset_name in frame_group.keys():
                    data = frame_group[dataset_name][:]
                    dtype_str = frame_group.attrs[f'{dataset_name}_dtype']
                    shape = frame_group.attrs[f'{dataset_name}_shape']
                    
                    if dtype_str.startswith('torch.'):
                        dtype_str = dtype_str[6:]  # Remove 'torch.' prefix

                    # Map string to torch dtype
                    dtype_map = {
                        'float32': torch.float32,
                        'float16': torch.float16, 
                        'int8': torch.int8,
                        'uint8': torch.uint8,
                        'bool': torch.bool
                    }
                    tensor = torch.tensor(data, dtype=dtype_map.get(dtype_str, torch.float32))
                    
                    # Move to device
                    if device != "cpu":
                        tensor = tensor.to(device)
                    
                    frame_data[dataset_name] = tensor
                
                # Load non-tensor attributes
                for attr_name, attr_value in frame_group.attrs.items():
                    if not attr_name.endswith(('_dtype', '_shape', '_device')):
                        frame_data[attr_name] = attr_value
                
                frames.append(frame_data)
        
        print(f"Total frames loaded: {len(frames)}")
        return cls(frames=frames)

    @classmethod
    def get_hdf5_info(cls, dir_path: str) -> dict:
        """
        Get information about HDF5 file if it exists.
        
        Returns:
            dict: HDF5 metadata including compression, total_frames, etc.
        """
        base = Path(dir_path)
        frames_dir = base / "frames"
        hdf5_file = frames_dir / "frames.h5"
        
        if not hdf5_file.exists():
            return {"format": "not_hdf5", "file_exists": False}
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                metadata_group = f['metadata']
                
                info = {
                    "format": "hdf5",
                    "file_exists": True,
                    "file_size_mb": hdf5_file.stat().st_size / (1024 * 1024),
                    "video_name": metadata_group.attrs.get('video_name', 'unknown'),
                    "total_frames": metadata_group.attrs.get('total_frames', 0),
                    "compression": metadata_group.attrs.get('compression', 'unknown'),
                    "compression_opts": metadata_group.attrs.get('compression_opts', 0)
                }
                
                return info
        except Exception as e:
            return {"format": "hdf5_error", "error": str(e)}

class FeatureExtractionResults(BaseModel):
    """Results from the feature extraction pipeline"""
    
    class Config:
        arbitrary_types_allowed = True
    
    image: np.ndarray = Field(..., description="Original input image")
    sam_masks: torch.Tensor = Field(..., description="SAM segmentation masks")
    input_boxes: torch.Tensor = Field(..., description="Input bounding boxes from YOLO")
    clip_embeddings: torch.Tensor = Field(..., description="CLIP embeddings tensor")
    timings: Dict[str, float] = Field(..., description="Primarily debugging info")

    def delete_memory(self, clear_cuda_cache: bool = True) -> None:
            """
            Release references to large fields to help GC/CUDA free memory.
            Safe for both Pydantic v1 and v2.
            """
            # Get declared model field names (v1/v2 compatible)
            if hasattr(self, "__fields__"):          # Pydantic v1
                field_names = list(self.__fields__.keys())
            elif hasattr(self, "model_fields"):      # Pydantic v2
                field_names = list(self.model_fields.keys())
            else:
                field_names = []

            # Drop references to known fields
            for name in field_names:
                if not hasattr(self, name):
                    continue
                val = getattr(self, name)
                # Break autograd links if any (harmless if not required)
                if isinstance(val, torch.Tensor):
                    try:
                        if val.is_leaf and val.requires_grad:
                            val.detach_()
                    except Exception:
                        pass
                # Remove attribute (frees reference)
                try:
                    delattr(self, name)
                except Exception:
                    # Fallback in case Pydantic guards attribute deletion
                    self.__dict__.pop(name, None)

            # Remove any extra public attrs that might hold memory
            for k in list(self.__dict__.keys()):
                if not k.startswith("_") and k not in field_names:
                    self.__dict__.pop(k, None)

            # Encourage Python/CUDA to reclaim memory
            gc.collect()
            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

############################################################
# TensorDict helpers
############################################################

def getEmptyFrameData(height: int = 720, width: int = 1280, dino_dim: int = 1024, samples: int = 10, device=None):
    """
    Create an empty dictionary for a single frame with the expected keys.
    """
    return {
        "manip_idx": -1,
        "manip_mask": torch.empty(0, height, width, device=device, dtype=torch.bool).to_sparse(),
        "manip_centroid_xyz": torch.empty(0, 3, device=device),
        "unit_dir_to_end": torch.empty(0, 3, device=device),
        "moved_dirs": torch.empty(0, 3, device=device),
        "centroid_pixels": torch.empty(0, 2, device=device, dtype=torch.uint8),
        "centroid_xyzs": torch.empty(0, 3, device=device),
        "delta_to_straight_line": torch.empty(0, 1, device=device),
        "sam_masks": torch.empty(0, height, width, device=device, dtype=torch.uint8),
        "dino_embeddings": torch.empty(0, dino_dim, device=device),
        "dino_embedding_samples": torch.empty(0, samples, dino_dim, device=device),
        "bboxes": torch.empty(0, 4, device=device),
        "in_dataset": torch.empty(0, dtype=torch.bool, device=device),
        "no_risk": torch.empty(0, dtype=torch.bool, device=device),
        "object_labels": []
    }