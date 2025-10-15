# Standard library imports
import argparse
import json
import os
import time
from datetime import datetime

# Third-party imports
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from sklearn.decomposition import IncrementalPCA, PCA
import torch
from torch.serialization import add_safe_globals
from tqdm import tqdm

# Add numpy array reconstruction to safe globals
add_safe_globals(['numpy.core.multiarray._reconstruct'])

class PointCloudGenerator:
    def __init__(self, intrinsics=None, fx=None, fy=None, cx=None, cy=None):
        """
        Initialize PointCloudGenerator with camera intrinsics.
        
        Args:
            intrinsics: Path to camera intrinsics JSON file
            fx, fy: Camera focal lengths
            cx, cy: Camera principal points
        """
        if intrinsics:
            intrinsics_data = PointCloudGenerator.load_intrinsics_from_file(intrinsics)
            self.camera_intrinsics = intrinsics_data['camera_intrinsics']
            self.image_width = intrinsics_data.get('image_width')
            self.image_height = intrinsics_data.get('image_height')
            self.distortion_coeffs = intrinsics_data.get('distortion_coeffs')
            self.rotation_matrix = intrinsics_data.get('rotation_matrix')
            self.projection_matrix = intrinsics_data.get('projection_matrix')
        elif all([fx, fy, cx, cy]):
            self.camera_intrinsics = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
        else:
            #raise ValueError("Either intrinsics file or all of fx, fy, cx, cy must be provided")
            print("===================================================")
            print("⚠️⚠️⚠️ WARNING: No camera intrinsics provided. ⚠️⚠️⚠️")
            print("===================================================")
            # put this here so we can use the point cloud file loader without camera intrinsics
            self.camera_intrinsics = None
        
        # Extract individual camera parameters
        if self.camera_intrinsics is not None:
            self.fx = self.camera_intrinsics[0, 0]
            self.fy = self.camera_intrinsics[1, 1]
            self.cx = self.camera_intrinsics[0, 2]
            self.cy = self.camera_intrinsics[1, 2]
            
            # Create CPU tensors
            self.fx_cpu = torch.tensor(self.fx, dtype=torch.float32)
            self.fy_cpu = torch.tensor(self.fy, dtype=torch.float32)
            self.cx_cpu = torch.tensor(self.cx, dtype=torch.float32)
            self.cy_cpu = torch.tensor(self.cy, dtype=torch.float32)
            
            # Create GPU tensors (if CUDA is available)
            if torch.cuda.is_available():
                self.fx_gpu = torch.tensor(self.fx, dtype=torch.float32, device='cuda')
                self.fy_gpu = torch.tensor(self.fy, dtype=torch.float32, device='cuda')
                self.cx_gpu = torch.tensor(self.cx, dtype=torch.float32, device='cuda')
                self.cy_gpu = torch.tensor(self.cy, dtype=torch.float32, device='cuda')
            else:
                self.fx_gpu = None
                self.fy_gpu = None
                self.cx_gpu = None
                self.cy_gpu = None
        else:
            # Set to None if no camera intrinsics provided
            self.fx = self.fy = self.cx = self.cy = None
            self.fx_cpu = self.fy_cpu = self.cx_cpu = self.cy_cpu = None
            self.fx_gpu = self.fy_gpu = self.cx_gpu = self.cy_gpu = None
    
    # NOTE: This is not used anymore, but is kept here for reference
    # Not sure why this is FAILING or if there was some preprocessing that was done to
    # to the camera intrinsics json files
    '''
    def load_intrinsics_from_file(self, intrinsics_path):
        """Load camera intrinsics from JSON file."""
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = json.load(f)
            
            # Extract camera parameters from JSON
            fx = intrinsics_data['fx']
            fy = intrinsics_data['fy']
            cx = intrinsics_data['cx']
            cy = intrinsics_data['cy']
            
            # Create intrinsics matrix
            self.camera_intrinsics = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
            
            print(f"Loaded camera intrinsics from {intrinsics_path}")
            print(f"Image dimensions: {intrinsics_data['W']}x{intrinsics_data['H']}")
            print(f"Focal lengths: fx={fx}, fy={fy}")
            print(f"Principal point: cx={cx}, cy={cy}")
            
        except Exception as e:
            raise ValueError(f"Could not read camera intrinsics from {intrinsics_path}: {str(e)}")
    '''
    
    @staticmethod
    def load_intrinsics_from_file(intrinsics_path):
        """Load camera intrinsics from JSON file.
        
        Args:
            intrinsics_path: Path to the camera intrinsics JSON file
            
        The JSON file should contain:
        - "K": 3x3 camera intrinsics matrix
        - "width": Image width
        - "height": Image height
        - "D": Distortion coefficients (optional)
        - "R": Rotation matrix (optional)
        - "P": Projection matrix (optional)
        
        Returns:
            dict: Dictionary containing camera intrinsics data with keys:
                - 'camera_intrinsics': 3x3 camera intrinsics matrix
                - 'image_width': Image width (optional)
                - 'image_height': Image height (optional)
                - 'distortion_coeffs': Distortion coefficients (optional)
                - 'rotation_matrix': Rotation matrix (optional)
                - 'projection_matrix': Projection matrix (optional)
        """
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = json.load(f)
            
            # Extract camera intrinsics matrix K
            if 'K' not in intrinsics_data:
                raise ValueError("Camera intrinsics matrix 'K' not found in JSON file")
            
            K_matrix = np.array(intrinsics_data['K'])
            
            # Validate K matrix dimensions
            if K_matrix.shape != (3, 3):
                raise ValueError(f"Camera intrinsics matrix K must be 3x3, got shape {K_matrix.shape}")
            
            # Extract other useful parameters
            image_width = intrinsics_data.get('width', None)
            image_height = intrinsics_data.get('height', None)
            distortion_coeffs = np.array(intrinsics_data.get('D', [0.0, 0.0, 0.0, 0.0, 0.0]))
            rotation_matrix = np.array(intrinsics_data.get('R', np.eye(3)))
            projection_matrix = np.array(intrinsics_data.get('P', None))
            
            # Extract individual camera parameters for convenience
            fx = K_matrix[0, 0]
            fy = K_matrix[1, 1]
            cx = K_matrix[0, 2]
            cy = K_matrix[1, 2]
            
            #print(f"Loaded camera intrinsics from {intrinsics_path}")
            if image_width and image_height:
                print(f"Image dimensions: {image_width}x{image_height}")
            #print(f"Focal lengths: fx={fx:.3f}, fy={fy:.3f}")
            #print(f"Principal point: cx={cx:.3f}, cy={cy:.3f}")
            
            # Print distortion coefficients if they exist
            #if np.any(distortion_coeffs != 0):
            #    print(f"Distortion coefficients: {distortion_coeffs}")
            
            # Return dictionary with all intrinsics data
            return {
                'camera_intrinsics': K_matrix,
                'image_width': image_width,
                'image_height': image_height,
                'distortion_coeffs': distortion_coeffs,
                'rotation_matrix': rotation_matrix,
                'projection_matrix': projection_matrix
            }
            
        except FileNotFoundError:
            raise ValueError(f"Could not find camera intrinsics file: {intrinsics_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {intrinsics_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading camera intrinsics from {intrinsics_path}: {str(e)}")

    def load_color_image(self, image_path):
        """
        Load and preprocess a color image.
        
        Args:
            image_path: Path to the color image file
            
        Returns:
            numpy.ndarray: RGB image array
        """
        color_image = cv2.imread(image_path)
        if color_image is None:
            raise ValueError(f"Could not read color image from {image_path}")
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    def load_depth_image(self, depth_path):
        """
        Load and preprocess a depth image from a .pt file.
        
        Args:
            depth_path: Path to the depth image .pt file
            
        Returns:
            numpy.ndarray: Depth image array
        """
        try:
            try:
                depth_tensor = torch.load(depth_path, weights_only=False)
            except Exception:
                depth_tensor = torch.load(depth_path, weights_only=True)
                
            if isinstance(depth_tensor, torch.Tensor):
                depth_image = depth_tensor.cpu().numpy()
            else:
                depth_image = np.array(depth_tensor)
                
            if len(depth_image.shape) > 2:
                depth_image = depth_image.squeeze()
                
            print(f"Successfully loaded depth image with shape: {depth_image.shape}")
            return depth_image
                
        except Exception as e:
            raise ValueError(f"Could not read depth image from {depth_path}: {str(e)}")
    
    def process_images(self, color_path, depth_path, pose=None):
        """
        Load and process color and depth images to create a point cloud.
        
        Args:
            color_path: Path to the color image file
            depth_path: Path to the depth image .pt file
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            tuple: (points, colors) where:
                - points: numpy.ndarray of shape (N, 3) containing 3D points
                - colors: numpy.ndarray of shape (N, 3) containing RGB colors
        """
        # Load images
        color_image = self.load_color_image(color_path)
        depth_image = self.load_depth_image(depth_path)
        
        # Generate point cloud
        return self.get_point_cloud(color_image, depth_image, pose)
    
    def _create_point_cloud_from_rgbd(self, color_image, depth_image, pose=None):
        """
        Create a point cloud from RGB-D images and camera intrinsics.
        Optionally transform points into world frame using camera pose.
        
        Args:
            color_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) or (H, W, 1)
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            tuple: (points, colors) where:
                - points: numpy.ndarray of shape (N, 3) containing 3D points
                - colors: numpy.ndarray of shape (N, 3) containing RGB colors
        """
        # Ensure depth_image is 2D
        if len(depth_image.shape) == 3:
            depth_image = depth_image.squeeze()
        
        # Get image dimensions
        H, W = depth_image.shape
        
        # Create meshgrid of pixel coordinates
        u_coords = np.arange(W)
        v_coords = np.arange(H)
        U_grid, V_grid = np.meshgrid(u_coords, v_coords)
        
        # Extract camera intrinsics
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]
        
        # Convert depth to 3D points
        x = (U_grid - cx) * depth_image / fx
        y = (V_grid - cy) * depth_image / fy
        z = depth_image
        
        # Stack coordinates
        points = np.stack((x, y, z), axis=-1)
        
        # Reshape points and colors to Nx3 arrays
        points_reshaped = points.reshape(-1, 3)
        colors_reshaped = color_image.reshape(-1, 3)
        
        # Normalize colors to [0, 1] range
        colors_reshaped = colors_reshaped.astype(np.float32) / 255.0
        
        # Remove invalid points (where depth is 0 or inf)
        valid_mask = np.logical_and(
            np.logical_and(points_reshaped[:, 2] > 0.2, points_reshaped[:, 2] < 5), #np.inf),
            np.all(np.isfinite(points_reshaped), axis=1)
        )
        
        # Get valid points and colors
        points_valid = points_reshaped[valid_mask]
        colors_valid = colors_reshaped[valid_mask]
        
        # Transform points to world frame if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Use scipy for proper quaternion to rotation matrix conversion

            # apply rotation 90 degrees around the x axis
            camera_2_robot_frame = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            #gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = rotation.as_matrix() @ camera_2_robot_frame #@ gl_to_cv
            q = rotation.as_quat()

            #assert np.allclose(q_npz, q), f"Quaternions do not match: {q_npz} != {q}"

            # Transform points: R * points + t
            #import pdb; pdb.set_trace()

            points_valid = points_valid @ R.T + pose['position']
        
        #print(f"Created point cloud with {len(points_valid)} points")
        #print(f"Color range: min={np.min(colors_valid)}, max={np.max(colors_valid)}")
        
        return points_valid, colors_valid
    
    def _create_sampled_point_cloud_from_im_and_depth(self, pixel_coords, image, depth_image, pose=None):
        """
        Create 3D points from multiple pixel coordinates using matrix operations.
        
        Args:
            pixel_coords: Array of (u, v) pixel coordinates shape (N, 2)
            image: (H,W,D) - image with labels/features
            depth_image: Depth image (H, W) or (H, W, 1)
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            points: numpy.ndarray 3D point coordinates (N, 3) for valid points
            labels: numpy.ndarray (N, D) corresponding labels from image
        """
        assert len(pixel_coords) > 0, "No pixel coordinates provided"
        
        # Ensure depth_image is 2D
        if len(depth_image.shape) == 3:
            depth_image = depth_image.squeeze()
        
        H, W = depth_image.shape
        
        # Convert to numpy array if needed
        pixel_coords = np.array(pixel_coords)
        u_coords = pixel_coords[:, 0]
        v_coords = pixel_coords[:, 1]
        
        # Check if pixel coordinates are valid
        valid_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        
        if not np.any(valid_mask):
            assert False, "No valid pixel coordinates"
        
        # Get depth values for valid pixels
        valid_u = u_coords[valid_mask].astype(int)
        valid_v = v_coords[valid_mask].astype(int)
        depths = depth_image[valid_v, valid_u]
        
        # Check if depths are valid
        depth_valid_mask = (depths >= 0.0) & np.isfinite(depths)
        
        if not np.any(depth_valid_mask):
            import pdb; pdb.set_trace()
            assert False, "No valid depths CHECK YOUR DEPTH IMAGE UNITS"
        
        # Get final valid coordinates and depths
        final_u = valid_u[depth_valid_mask]
        final_v = valid_v[depth_valid_mask]
        final_depths = depths[depth_valid_mask]
        
        # Extract camera intrinsics
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]
        
        # Convert pixels to 3D points using matrix operations
        x = (final_u - cx) * final_depths / fx
        y = (final_v - cy) * final_depths / fy
        z = final_depths
        
        # Stack coordinates
        points_3d = np.stack([x, y, z], axis=1)
        
        # Get corresponding labels from image at the same pixel coordinates
        # Use the same valid coordinates that passed depth validation
        labels = image[final_v, final_u]  # Shape: (N, D) where D is the image channel dimension
        
        # Transform points to world frame if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis
            camera_2_robot_frame = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = rotation.as_matrix() @ camera_2_robot_frame
            
            # Transform points: R * points + t
            points_3d = points_3d @ R.T + pose['position']
        
        if len(points_3d) == 0:
            assert False, "No valid 3D points"

        return points_3d, labels

    def _create_sampled_point_cloud_from_im_and_depth_torch(self, pixel_coords, image, depth_image, pose=None):
        """
        Create 3D points from multiple pixel coordinates using PyTorch operations.
        
        Args:
            pixel_coords: Tensor of (u, v) pixel coordinates shape (N, 2)
            image: (H,W,D) - image with labels/features (numpy array)
            depth_image: Depth image (H, W) or (H, W, 1) (numpy array)
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            points: torch.Tensor 3D point coordinates (N, 3) for valid points
            labels: torch.Tensor (N, D) corresponding labels from image
        """
        assert len(pixel_coords) > 0, "No pixel coordinates provided"
        
        # Ensure depth_image is 2D
        if len(depth_image.shape) == 3:
            depth_image = depth_image.squeeze()
        
        H, W = depth_image.shape
        
        # Extract coordinates
        u_coords = pixel_coords[:, 0]
        v_coords = pixel_coords[:, 1]
        
        # Check if pixel coordinates are valid
        valid_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        
        if not torch.any(valid_mask):
            assert False, "No valid pixel coordinates"
        
        # Get depth values for valid pixels
        valid_u = u_coords[valid_mask].long()
        valid_v = v_coords[valid_mask].long()
        depths = depth_image[valid_v, valid_u]
        
        # Check if depths are valid
        depth_valid_mask = (depths >= 0.0) & torch.isfinite(depths)
        
        if not torch.any(depth_valid_mask):
            import pdb; pdb.set_trace()
            assert False, "No valid depths CHECK YOUR DEPTH IMAGE UNITS"
        
        # Get final valid coordinates and depths
        final_u = valid_u[depth_valid_mask]
        final_v = valid_v[depth_valid_mask]
        final_depths = depths[depth_valid_mask]
        
        # Use pre-defined camera intrinsics tensors based on device
        if pixel_coords.device.type == 'cuda':
            fx_tensor = self.fx_gpu
            fy_tensor = self.fy_gpu
            cx_tensor = self.cx_gpu
            cy_tensor = self.cy_gpu
        else:
            fx_tensor = self.fx_cpu
            fy_tensor = self.fy_cpu
            cx_tensor = self.cx_cpu
            cy_tensor = self.cy_cpu
        
        # Convert pixels to 3D points using tensor operations
        x = (final_u.float() - cx_tensor) * final_depths / fx_tensor
        y = (final_v.float() - cy_tensor) * final_depths / fy_tensor
        z = final_depths
        
        # Stack coordinates
        points_3d = torch.stack([x, y, z], dim=1)
        
        # Get corresponding labels from image at the same pixel coordinates
        # Use the same valid coordinates that passed depth validation
        labels = image[final_v, final_u]  # Shape: (N, D) where D is the image channel dimension
        
        # Convert to float32 if the image is uint8 (RGB images)
        if labels.dtype == torch.uint8:
            labels = labels.float() / 255.0  # Normalize to [0, 1] range
        
        # Transform points to world frame if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis
            camera_2_robot_frame = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], 
                                               dtype=torch.float32, device=pixel_coords.device)
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=pixel_coords.device) @ camera_2_robot_frame
            
            # Transform points: R * points + t
            pose_position = torch.tensor(pose['position'], dtype=torch.float32, device=pixel_coords.device)
            points_3d = points_3d @ R.T + pose_position
        
        if len(points_3d) == 0:
            assert False, "No valid 3D points"

        return points_3d, labels
    
    def get_point_cloud(self, rgb_image, depth_image, pose=None):
        """
        Create a point cloud from RGB-D images.
        
        Args:
            rgb_image: RGB image (H, W, 3) as numpy array
            depth_image: Depth image (H, W) or (H, W, 1) as numpy array
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            tuple: (points, colors) where:
                - points: numpy.ndarray of shape (N, 3) containing 3D points
                - colors: numpy.ndarray of shape (N, 3) containing RGB colors
        """
        # Ensure images have the same dimensions
        if rgb_image.shape[:2] != depth_image.shape[:2]:
            raise ValueError(f"Color and depth images must have the same dimensions. Color: {rgb_image.shape[:2]}, Depth: {depth_image.shape}")
        
        # Ensure that the pose if not None is a dict with the keys 'position' and 'orientation' of shape (3,) and (4,) respectively
        # This is to ensure that the pose is valid and can be used to transform the points
        if pose is not None:
            if not isinstance(pose, dict):
                raise ValueError("Pose must be a dictionary with keys 'position' and 'orientation'")
            if not isinstance(pose['position'], np.ndarray) or pose['position'].shape != (3,):
                raise ValueError("Pose position must be a numpy array of shape (3,)")
            if not isinstance(pose['orientation'], np.ndarray) or pose['orientation'].shape != (4,):
                raise ValueError("Pose orientation must be a numpy array of shape (4,)")
            
        return self._create_point_cloud_from_rgbd(rgb_image, depth_image, pose)
    
    def to_open3d(self, points, colors):
        """
        Convert numpy arrays to Open3D point cloud.
        
        Args:
            points: numpy.ndarray of shape (N, 3) containing 3D points
            colors: numpy.ndarray of shape (N, 3) containing RGB colors
            
        Returns:
            open3d.geometry.PointCloud: Open3D point cloud object
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    
    def save_point_cloud(self, points, colors, output_dir=None, cloud_type="rgb", base_filename=None, verbose=False):
        """
        Save a point cloud to a file.
        
        Args:
            points: numpy.ndarray of shape (N, 3) containing 3D points
            colors: numpy.ndarray of shape (N, 3) containing RGB colors
            output_dir: Optional output directory path
            cloud_type: Type of point cloud (e.g. "rgb", "CLIP", or "CLIP-raw")
            base_filename: Optional base filename to prepend to the output filename
        """
        # Convert to Open3D format
        pcd = self.to_open3d(points, colors)
        
        if output_dir:
            output_file = self._get_output_filename(output_dir, cloud_type, base_filename)
            o3d.io.write_point_cloud(output_file, pcd)
            if verbose:
                print(f"Point cloud saved to {output_file}")
            return output_file
        return None

    def create_per_point_embeddings(self, points, sam_masks_clip, image_shape, pose=None):
        """
        Create per-point embeddings by mapping mask embeddings to individual points.
        
        Args:
            points: numpy.ndarray of shape (N, 3) containing 3D points
            sam_masks_clip: List of (mask, embedding) tuples from SAM+CLIP processing
            image_shape: Tuple of (H, W) for the original image
            pose: Optional camera pose used to transform points
            
        Returns:
            numpy.ndarray: Per-point embeddings of shape (N, embedding_dim)
        """
        if not sam_masks_clip:
            # If no masks, return zero embeddings
            embedding_dim = 1024  # CLIP embedding dimension
            return np.zeros((len(points), embedding_dim))
        
        H, W = image_shape
        embedding_dim = sam_masks_clip[0][1].shape[0]  # Get embedding dimension from first mask
        
        # Initialize per-point embeddings
        per_point_embeddings = np.zeros((len(points), embedding_dim))
        
        # If pose was used, we need to transform points back to camera coordinates
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Use the same transformations as in _create_point_cloud_from_rgbd
            camera_2_robot_frame = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = rotation.as_matrix() @ camera_2_robot_frame
            
            # Transform points back to camera frame: (points - t) @ R.T
            points_camera = (points - pose['position']) @ R.T
        else:
            points_camera = points
        
        # Convert 3D camera coordinates back to 2D pixel coordinates
        if self.camera_intrinsics is not None:
            fx = self.camera_intrinsics[0, 0]
            fy = self.camera_intrinsics[1, 1]
            cx = self.camera_intrinsics[0, 2]
            cy = self.camera_intrinsics[1, 2]
            
            # Project 3D points to 2D pixels
            u_coords = (points_camera[:, 0] * fx / points_camera[:, 2] + cx).astype(int)
            v_coords = (points_camera[:, 1] * fy / points_camera[:, 2] + cy).astype(int)
        else:
            # Fallback: assume points are already in pixel coordinates
            # This is a rough approximation and may not be accurate
            u_coords = points_camera[:, 0].astype(int)
            v_coords = points_camera[:, 1].astype(int)
        
        # Clip coordinates to image bounds
        u_coords = np.clip(u_coords, 0, W - 1)
        v_coords = np.clip(v_coords, 0, H - 1)
        
        # Assign embeddings to points based on which mask they belong to
        for mask_idx, (mask, embedding) in enumerate(sam_masks_clip):
            # Get mask coordinates where mask is True
            mask_coords = np.where(mask)
            mask_v_coords, mask_u_coords = mask_coords
            
            # Find points that fall within this mask
            for i, (u, v) in enumerate(zip(u_coords, v_coords)):
                if u < W and v < H and mask[v, u]:
                    per_point_embeddings[i] = embedding
        
        #print(f"Created per-point embeddings: {per_point_embeddings.shape}")
        #print(f"Number of masks: {len(sam_masks_clip)}")
        #print(f"Points with non-zero embeddings: {np.sum(np.any(per_point_embeddings != 0, axis=1))}")
        
        return per_point_embeddings

    def save_point_cloud_with_embeddings(self, points, colors, embeddings, output_dir, cloud_type="CLIP", base_filename=None, verbose=False):
        """
        Save point cloud with full embeddings stored separately.
        
        Args:
            points: numpy.ndarray of shape (N, 3) containing 3D points
            colors: numpy.ndarray of shape (N, 3) containing RGB colors for visualization
            embeddings: numpy.ndarray of shape (N, D) containing full CLIP embeddings
            output_dir: Output directory path
            cloud_type: Type of point cloud
            base_filename: Optional base filename
        """
        # Save the point cloud for visualization
        pcd_file = self.save_point_cloud(points, colors, output_dir, cloud_type, base_filename)
        
        # Process in batches manually
        batch_size = 1000
        pca = IncrementalPCA(n_components=3)
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Computing PCA"):
            batch = embeddings[i:i+batch_size]
            pca.partial_fit(batch)
        pca_components = pca.transform(embeddings)
        
        # Normalize PCA components to [0, 1] range for RGB coloring
        pca_min = np.min(pca_components, axis=0)
        pca_max = np.max(pca_components, axis=0)
        pca_colors = (pca_components - pca_min) / (pca_max - pca_min)
        
        # Save the PCA-colored point cloud (using embeddings PCA for colors)
        pca_pcd_file = self.save_point_cloud(points, pca_colors, output_dir, f"{cloud_type}_PCA", base_filename, verbose)
        
        # Save the full embeddings separately
        if pcd_file:
            embeddings_file = pcd_file.replace('.ply', '_embeddings.npy')
            np.save(embeddings_file, embeddings)
            if verbose:
                print(f"Embeddings saved to {embeddings_file}")
            
            # Save metadata
            metadata = {
                'num_points': len(points),
                'embedding_dim': embeddings.shape[1],
                'cloud_type': cloud_type,
                'timestamp': datetime.now().isoformat(),
                'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'pca_components': pca.components_.tolist()
            }
            metadata_file = pcd_file.replace('.ply', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            if verbose:
                print(f"Metadata saved to {metadata_file}")
            
            return pcd_file, embeddings_file, pca_pcd_file
        return None, None, None

    def load_point_cloud_with_embeddings(self, pcd_file):
        """
        Load a point cloud and its associated embeddings.
        
        Args:
            pcd_file: Path to the .ply file
            
        Returns:
            tuple: (points, colors, embeddings, metadata) where:
                - points: numpy.ndarray of shape (N, 3)
                - colors: numpy.ndarray of shape (N, 3)
                - embeddings: numpy.ndarray of shape (N, D)
                - metadata: dict containing point cloud metadata
        """
        # Load point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Try to load embeddings from .npy file first
        embeddings_file = pcd_file.replace('.ply', '_embeddings.npy')
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
        else:
            # Try to load from .npz file
            base_name = os.path.splitext(os.path.basename(pcd_file))[0]
            frame_idx = base_name.split('_')[1]  # Extract frame number
            npz_file = os.path.join(os.path.dirname(pcd_file), f'frame_{frame_idx}_features_sam_masks_clip.npz')
            if os.path.exists(npz_file):
                embeddings_data = np.load(npz_file)
                embeddings = embeddings_data['embeddings']
            else:
                raise FileNotFoundError(f"No embeddings file found for {pcd_file}")
        
        # Load metadata if it exists
        metadata_file = pcd_file.replace('.ply', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'num_points': len(points),
                'embedding_dim': embeddings.shape[1],
                'cloud_type': 'CLIP',
                'timestamp': datetime.now().isoformat()
            }
            
        return points, colors, embeddings, metadata

    def visualize_point_cloud(self, points, colors, output_dir=None, cloud_type="rgb", base_filename=None):
        """
        Visualize the point cloud using Open3D.
        If running in a headless environment, saves the point cloud to a file instead.
        
        Args:
            points: numpy.ndarray of shape (N, 3) containing 3D points
            colors: numpy.ndarray of shape (N, 3) containing RGB colors
            output_dir: Optional output directory path
            cloud_type: Type of point cloud (e.g. "rgb" or "CLIP")
            base_filename: Optional base filename to prepend to the output filename
        """
        # Convert to Open3D format
        pcd = self.to_open3d(points, colors)
        
        try:
            # Create visualization window
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            
            # Add point cloud to visualizer
            vis.add_geometry(pcd)
            
            # Set default camera view
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.array([0, 0, 0])
            opt.light_on = True
            
            # Run visualizer
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print(f"Could not create visualization window: {e}")
        
        # Save point cloud if output directory is provided
        return self.save_point_cloud(points, colors, output_dir, cloud_type, base_filename)
    
    def _create_point_cloud_collection_pts(self, pixel_coords, depth_image, pose=None):
        """
        Create 3D points from multiple pixel coordinates using matrix operations.
        
        Args:
            pixel_coords: Array of (u, v) pixel coordinates shape (N, 2) (y, x)
            depth_image: Depth image (H, W) or (H, W, 1)
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            numpy.ndarray: 3D point coordinates (N, 3) or None for invalid points
        """
        assert len(pixel_coords) > 0, "No pixel coordinates provided"
        
        # Ensure depth_image is 2D
        if len(depth_image.shape) == 3:
            depth_image = depth_image.squeeze()
        
        H, W = depth_image.shape
        
        # Convert to numpy array if needed
        pixel_coords = np.array(pixel_coords)
        u_coords = pixel_coords[:, 0]
        v_coords = pixel_coords[:, 1]
        
        # Check if pixel coordinates are valid
        valid_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        
        if not np.any(valid_mask):
            assert False, "No valid pixel coordinates"
        
        # Get depth values for valid pixels
        valid_u = u_coords[valid_mask].astype(int)
        valid_v = v_coords[valid_mask].astype(int)
        depths = depth_image[valid_v, valid_u]
        
        # Check if depths are valid
        #depth_valid_mask = (depths > 0.2) & (depths < 5.0) & np.isfinite(depths)
        #depth_valid_mask = (depths > 0.0) & (depths < 15.0) & np.isfinite(depths)
        depth_valid_mask = (depths >= 0.0) & np.isfinite(depths) # TODO: remove this
        
        if not np.any(depth_valid_mask):
            import pdb; pdb.set_trace()
            assert False, "No valid depths CHECK YOUR DEPTH IMAGE UNITS"
        
        # Get final valid coordinates and depths
        final_u = valid_u[depth_valid_mask]
        final_v = valid_v[depth_valid_mask]
        final_depths = depths[depth_valid_mask]
        
        # Extract camera intrinsics
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]
        
        # Convert pixels to 3D points using matrix operations
        x = (final_u - cx) * final_depths / fx
        y = (final_v - cy) * final_depths / fy
        z = final_depths
        
        # Stack coordinates
        points_3d = np.stack([x, y, z], axis=1)
        
        # Transform points to world frame if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis
            camera_2_robot_frame = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = rotation.as_matrix() @ camera_2_robot_frame
            
            # Transform points: R * points + t
            points_3d = points_3d @ R.T + pose['position']
        
        if len(points_3d) == 0:
            assert False, "No valid 3D points"

        return points_3d

    def _create_point_cloud_collection_pts_torch(self, pixel_coords, depth_image, pose=None):
        """
        Create 3D points from multiple pixel coordinates using PyTorch operations.
        
        Args:
            pixel_coords: Tensor of (u, v) pixel coordinates shape (N, 2) (y, x)
            depth_image: Depth image (H, W) or (H, W, 1) (pytorch tensor)
            pose: Optional dict containing camera pose with keys:
                - 'position': pytorch tensor of shape (3,) containing camera position
                - 'orientation': pytorch tensor of shape (4,) containing camera orientation quaternion
            
        Returns:
            torch.Tensor: 3D point coordinates (N, 3) or None for invalid points
        """
        assert len(pixel_coords) > 0, "No pixel coordinates provided"
        
        # Ensure depth_image is 2D
        if len(depth_image.shape) == 3:
            depth_image = depth_image.squeeze()
        
        H, W = depth_image.shape
        
        # Extract coordinates
        u_coords = pixel_coords[:, 0]
        v_coords = pixel_coords[:, 1]
        
        # Check if pixel coordinates are valid
        valid_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        
        if not torch.any(valid_mask):
            assert False, "No valid pixel coordinates"
        
        # Get depth values for valid pixels
        valid_u = u_coords[valid_mask].long()
        valid_v = v_coords[valid_mask].long()
        depths = depth_image[valid_v, valid_u]
        
        # Check if depths are valid
        depth_valid_mask = (depths >= 0.0) & torch.isfinite(depths)
        
        if not torch.any(depth_valid_mask):
            assert False, "No valid depths CHECK YOUR DEPTH IMAGE UNITS"
        
        # Get final valid coordinates and depths
        final_u = valid_u[depth_valid_mask]
        final_v = valid_v[depth_valid_mask]
        final_depths = depths[depth_valid_mask]
        
        # Use pre-defined camera intrinsics tensors based on device
        if pixel_coords.device.type == 'cuda':
            fx_tensor = self.fx_gpu
            fy_tensor = self.fy_gpu
            cx_tensor = self.cx_gpu
            cy_tensor = self.cy_gpu
        else:
            fx_tensor = self.fx_cpu
            fy_tensor = self.fy_cpu
            cx_tensor = self.cx_cpu
            cy_tensor = self.cy_cpu
        
        # Convert pixels to 3D points using tensor operations
        x = (final_u.float() - cx_tensor) * final_depths / fx_tensor
        y = (final_v.float() - cy_tensor) * final_depths / fy_tensor
        z = final_depths
        
        # Stack coordinates
        points_3d = torch.stack([x, y, z], dim=1)
        
        # Transform points to world frame if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis
            camera_2_robot_frame = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], 
                                               dtype=torch.float32, device=pixel_coords.device)
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=pixel_coords.device) @ camera_2_robot_frame
            
            # Transform points: R * points + t
            pose_position = torch.tensor(pose['position'], dtype=torch.float32, device=pixel_coords.device)
            points_3d = points_3d @ R.T + pose_position
        
        if len(points_3d) == 0:
            assert False, "No valid 3D points"

        return points_3d

    def project_to_image_torch(self, points_3d, colors, image_shape, pose=None):
        """
        Project 3D point cloud points back to a 2D image using camera intrinsics.
        
        Args:
            points_3d: torch.Tensor of shape (N, 3) containing 3D points in world coordinates
            colors: torch.Tensor of shape (N, 3) containing RGB colors for each point
            image_shape: Tuple of (H, W) for the target image dimensions
            pose: Optional dict containing camera pose with keys:
                - 'position': torch.Tensor of shape (3,) containing camera position
                - 'orientation': torch.Tensor of shape (4,) containing camera orientation quaternion
            
        Returns:
            tuple: (projected_image, depth_image) where:
                - projected_image: torch.Tensor of shape (H, W, 3) containing the projected RGB image
                - depth_image: torch.Tensor of shape (H, W) containing the depth values
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not available. Cannot project points to image.")
        
        H, W = image_shape
        device = points_3d.device
        
        # Transform points from world coordinates to camera coordinates if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis (inverse of the transformation used in point cloud generation)
            camera_2_robot_frame = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], 
                                               dtype=torch.float32, device=device)
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz.cpu().numpy())
            R = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=device) @ camera_2_robot_frame
            
            # Transform points: (points - t) @ R.T (inverse of the world transformation)
            pose_position = pose['position']
            points_camera = (points_3d - pose_position) @ R.T
        else:
            points_camera = points_3d
        
        # Use pre-defined camera intrinsics tensors based on device
        if device.type == 'cuda':
            fx_tensor = self.fx_gpu
            fy_tensor = self.fy_gpu
            cx_tensor = self.cx_gpu
            cy_tensor = self.cy_gpu
        else:
            fx_tensor = self.fx_cpu
            fy_tensor = self.fy_cpu
            cx_tensor = self.cx_cpu
            cy_tensor = self.cy_cpu
        
        # Project 3D points to 2D pixel coordinates
        # Using the inverse of the depth-to-point conversion:
        # u = x * fx / z + cx
        # v = y * fy / z + cy
        x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
        
        # Avoid division by zero
        valid_mask = z > 0
        
        if not torch.any(valid_mask):
            # Return empty images if no valid points
            projected_image = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
            depth_image = torch.zeros((H, W), dtype=torch.float32, device=device)
            return projected_image, depth_image
        
        # Only process valid points
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        colors_valid = colors[valid_mask]
        
        # Project to pixel coordinates
        u_coords = (x_valid * fx_tensor / z_valid + cx_tensor).long()
        v_coords = (y_valid * fy_tensor / z_valid + cy_tensor).long()
        
        # Filter points that fall within image bounds
        in_bounds_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        
        if not torch.any(in_bounds_mask):
            # Return empty images if no points in bounds
            projected_image = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
            depth_image = torch.zeros((H, W), dtype=torch.float32, device=device)
            return projected_image, depth_image
        
        # Get final valid coordinates and data
        final_u = u_coords[in_bounds_mask]
        final_v = v_coords[in_bounds_mask]
        final_z = z_valid[in_bounds_mask]
        final_colors = colors_valid[in_bounds_mask]
        
        # Initialize output images
        projected_image = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        depth_image = torch.zeros((H, W), dtype=torch.float32, device=device)
        
        # For overlapping points, we'll use the closest point (smallest depth)
        # Create a combined index for efficient processing
        pixel_indices = final_v * W + final_u
        
        # Sort by depth to handle overlapping points (closest points first)
        sorted_indices = torch.argsort(final_z)
        sorted_pixel_indices = pixel_indices[sorted_indices]
        sorted_colors = final_colors[sorted_indices]
        sorted_depths = final_z[sorted_indices]
        
        # Handle overlapping points by keeping only the first occurrence (closest point)
        # Use a dictionary approach since torch.unique with return_index has issues
        seen_pixels = {}
        final_unique_indices = []
        
        for i, pixel_idx in enumerate(sorted_pixel_indices):
            if pixel_idx.item() not in seen_pixels:
                seen_pixels[pixel_idx.item()] = i
                final_unique_indices.append(i)
        
        if final_unique_indices:
            final_unique_indices = torch.tensor(final_unique_indices, device=device)
            unique_colors = sorted_colors[final_unique_indices]
            unique_depths = sorted_depths[final_unique_indices]
            unique_pixel_indices = sorted_pixel_indices[final_unique_indices]
            
            # Convert back to 2D coordinates
            unique_v = unique_pixel_indices // W
            unique_u = unique_pixel_indices % W
            
            # Assign colors and depths
            projected_image[unique_v, unique_u] = unique_colors
            depth_image[unique_v, unique_u] = unique_depths
        
        return projected_image, depth_image

    def project_to_image_1D_torch(self, points_3d, values, image_shape, pose=None):
        """
        Project 3D point cloud risk values back to a 2D image using camera intrinsics.
        
        Args:
            points_3d: torch.Tensor of shape (N, 3) containing 3D points in world coordinates
            values: torch.Tensor of shape (N,) containing values for each point
            image_shape: Tuple of (H, W) for the target image dimensions
            pose: Optional dict containing camera pose with keys:
                - 'position': torch.Tensor of shape (3,) containing camera position
                - 'orientation': torch.Tensor of shape (4,) containing camera orientation quaternion
            
        Returns:
            torch.Tensor: Image of shape (H, W) containing the projected values
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not available. Cannot project points to image.")
        
        H, W = image_shape
        device = points_3d.device
        
        # Transform points from world coordinates to camera coordinates if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis (inverse of the transformation used in point cloud generation)
            camera_2_robot_frame = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], 
                                               dtype=torch.float32, device=device)
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz.cpu().numpy())
            R = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=device) @ camera_2_robot_frame
            
            # Transform points: (points - t) @ R.T (inverse of the world transformation)
            pose_position = pose['position']
            points_camera = (points_3d - pose_position) @ R.T
        else:
            points_camera = points_3d
        
        # Use pre-defined camera intrinsics tensors based on device
        if device.type == 'cuda':
            fx_tensor = self.fx_gpu
            fy_tensor = self.fy_gpu
            cx_tensor = self.cx_gpu
            cy_tensor = self.cy_gpu
        else:
            fx_tensor = self.fx_cpu
            fy_tensor = self.fy_cpu
            cx_tensor = self.cx_cpu
            cy_tensor = self.cy_cpu
        
        # Project 3D points to 2D pixel coordinates
        x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
        
        # Avoid division by zero
        valid_mask = z > 0
        
        if not torch.any(valid_mask):
            # Return empty risk image if no valid points
            image = torch.zeros((H, W), dtype=torch.float32, device=device)
            return image
        
        # Only process valid points
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        values_valid = values[valid_mask]
        
        # Project to pixel coordinates
        u_coords = (x_valid * fx_tensor / z_valid + cx_tensor).long()
        v_coords = (y_valid * fy_tensor / z_valid + cy_tensor).long()
        
        # Filter points that fall within image bounds
        in_bounds_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H)
        
        if not torch.any(in_bounds_mask):
            # Return empty risk image if no points in bounds
            image = torch.zeros((H, W), dtype=torch.float32, device=device)
            return image
        
        # Get final valid coordinates and data
        final_u = u_coords[in_bounds_mask]
        final_v = v_coords[in_bounds_mask]
        final_z = z_valid[in_bounds_mask]
        final_values = values_valid[in_bounds_mask]
        
        # Initialize output risk image
        image = torch.zeros((H, W), dtype=torch.float32, device=device)
        
        # For overlapping points, we'll use the maximum risk value (most conservative)
        # Create a combined index for efficient processing
        pixel_indices = final_v * W + final_u
        
        # Sort by depth to handle overlapping points (closest points first)
        sorted_indices = torch.argsort(final_z)
        sorted_pixel_indices = pixel_indices[sorted_indices]
        sorted_values = final_values[sorted_indices]
        
        # Handle overlapping points by keeping the maximum risk value
        # Use a dictionary approach since torch.unique with return_index has issues
        seen_pixels = {}
        final_unique_indices = []
        
        for i, pixel_idx in enumerate(sorted_pixel_indices):
            pixel_idx_item = pixel_idx.item()
            if pixel_idx_item not in seen_pixels:
                seen_pixels[pixel_idx_item] = i
                final_unique_indices.append(i)
            else:
                # Update with maximum risk value
                existing_idx = seen_pixels[pixel_idx_item]
                if sorted_values[i] > sorted_values[existing_idx]:
                    seen_pixels[pixel_idx_item] = i
        
        if final_unique_indices:
            final_unique_indices = torch.tensor(list(seen_pixels.values()), device=device)
            unique_values = sorted_values[final_unique_indices]
            unique_pixel_indices = sorted_pixel_indices[final_unique_indices]
            
            # Convert back to 2D coordinates
            unique_v = unique_pixel_indices // W
            unique_u = unique_pixel_indices % W
            
            # Assign risk values
            image[unique_v, unique_u] = unique_values
        
        return image

    def get_K_matrix_torch(self, device=None):
        """       
        Args:
            device: Optional device to place the tensor on. If None, uses CUDA if available. 
        Returns:
            torch.Tensor: K matrix of shape (3, 3) on the specified device
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not available. Cannot return K matrix.")

        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        K_matrix = torch.tensor(self.camera_intrinsics, dtype=torch.float32, device=device)
        return K_matrix

    def _get_output_filename(self, output_dir=None, cloud_type="rgb", base_filename=None):
        """Generate output filename with timestamp.
        
        Args:
            output_dir: Optional output directory path
            cloud_type: Type of point cloud (e.g. "rgb" or "CLIP")
            base_filename: Optional base filename to prepend to the output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_filename:
            filename = f"{base_filename}_point_cloud_{cloud_type}_{timestamp}.ply"
        else:
            filename = f"point_cloud_{cloud_type}_{timestamp}.ply"
        
        # If no output directory specified, use local data folder
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _create_point_cloud_from_binary_mask_torch(self, binary_mask, depth_image, pose=None):
        """
        Create 3D points from a binary mask and depth image using PyTorch operations.
        
        Args:
            binary_mask: torch.Tensor of shape (H, W) containing binary mask (True/False or 1/0)
            depth_image: torch.Tensor of shape (H, W) containing depth values
            pose: Optional dict containing camera pose with keys:
                - 'position': numpy.ndarray of shape (3,) containing camera position
                - 'orientation': numpy.ndarray of shape (4,) containing camera orientation quaternion
            
        Returns:
            torch.Tensor: 3D point coordinates (N, 3) for all mask pixels
        """
        # Ensure inputs are 2D
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask.squeeze()
        if len(depth_image.shape) == 3:
            depth_image = depth_image.squeeze()
        
        H, W = depth_image.shape
        
        # Get pixel coordinates where mask is True
        mask_coords = torch.where(binary_mask)
        v_coords, u_coords = mask_coords
        
        if len(u_coords) == 0:
            # Return empty tensor if no mask pixels
            return torch.empty((0, 3), dtype=torch.float32, device=depth_image.device)
        
        # Get depth values for mask pixels
        depths = depth_image[v_coords, u_coords]
        
        # Check if depths are valid
        depth_valid_mask = (depths >= 0.0) & torch.isfinite(depths)
        
        if not torch.any(depth_valid_mask):
            # Return empty tensor if no valid depths
            return torch.empty((0, 3), dtype=torch.float32, device=depth_image.device)
        
        # Get final valid coordinates and depths
        final_u = u_coords[depth_valid_mask]
        final_v = v_coords[depth_valid_mask]
        final_depths = depths[depth_valid_mask]
        
        # Use pre-defined camera intrinsics tensors based on device
        if depth_image.device.type == 'cuda':
            fx_tensor = self.fx_gpu
            fy_tensor = self.fy_gpu
            cx_tensor = self.cx_gpu
            cy_tensor = self.cy_gpu
        else:
            fx_tensor = self.fx_cpu
            fy_tensor = self.fy_cpu
            cx_tensor = self.cx_cpu
            cy_tensor = self.cy_cpu
        
        # Convert pixels to 3D points using tensor operations
        x = (final_u.float() - cx_tensor) * final_depths / fx_tensor
        y = (final_v.float() - cy_tensor) * final_depths / fy_tensor
        z = final_depths
        
        # Stack coordinates
        points_3d = torch.stack([x, y, z], dim=1)
        
        # Transform points to world frame if pose is provided
        if pose is not None:
            # Convert quaternion to rotation matrix
            q_npz = pose['orientation']  # Data is in [x, y, z, w] format
            
            # Apply rotation 90 degrees around the x axis
            camera_2_robot_frame = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], 
                                               dtype=torch.float32, device=depth_image.device)
            
            # Create rotation object from quaternion [x, y, z, w]
            rotation = Rotation.from_quat(q_npz)
            R = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=depth_image.device) @ camera_2_robot_frame
            
            # Transform points: R * points + t
            pose_position = torch.tensor(pose['position'], dtype=torch.float32, device=depth_image.device)
            points_3d = points_3d @ R.T + pose_position
        
        return points_3d

    def _create_point_cloud_from_binary_masks_torch(self, binary_masks, depth_images, n_points=8_000, pose=None):
        """
        CPU-optimized: sample indices first, then back-project only sampled pixels.
        Returns (M, n_points, 3). Rows with no valid pixels are zeros.
        """
        with torch.no_grad():
            assert binary_masks.ndim == 3 and depth_images.ndim == 3
            M, H, W = binary_masks.shape
            assert depth_images.shape == (M, H, W)
            dtype  = depth_images.dtype

            # Intrinsics on CPU as 0-dim tensors (avoid device/dtype churn in loop)
            device = binary_masks.device
            if device.type == 'cuda':
                fx = self.fx_gpu
                fy = self.fy_gpu
                cx = self.cx_gpu
                cy = self.cy_gpu
            else:
                fx = self.fx_cpu
                fy = self.fy_cpu
                cx = self.cx_cpu
                cy = self.cy_cpu

            
            # Joint validity
            valid = binary_masks.bool() & (depth_images >= 0.0) & torch.isfinite(depth_images)  # (M,H,W)
            
            # Early-out
            if not valid.any():
                print("No valid points")
                return torch.zeros((M, n_points, 3), dtype=dtype, device=device)

            out = torch.zeros((M, n_points, 3), dtype=dtype, device=device)
            for i in range(M):
                vmask = valid[i].view(-1)  # (H*W)
                idx = torch.nonzero(vmask, as_tuple=False).squeeze(1)  # (K,)
                K = idx.numel()
                
                if K == 0:
                    # already zeroed
                    continue

                # Sample indices first
                if K >= n_points:
                    # sample without replacement (fast + less bias)
                    sel = torch.randperm(K, device=device)[:n_points]
                    sel = idx[sel]
                else:
                    # with replacement to fill
                    sel = idx[torch.randint(0, K, (n_points,), device=device)]

                # Convert to (v,u)
                v = torch.div(sel, W, rounding_mode='floor')
                u = sel - v * W

                # Fetch depths and back-project
                di = depth_images[i].view(-1)
                z = di[sel]                                   # (n_points,)
                uf = u.to(dtype); vf = v.to(dtype)
                x = (uf - cx) * z / fx
                y = (vf - cy) * z / fy
                pts = torch.stack((x, y, z), dim=1)          # (n_points, 3)
                out[i] = pts

            return out

def main():
    """
    Create and visualize a point cloud from RGB-D images.
    Usage:
        python point_cloud_3D.py --color color_image.png --depth depth_image.pt --intrinsics camera_intrinsics.json
        OR
        python point_cloud_3D.py --color color_image.png --depth depth_image.pt --fx 525.0 --fy 525.0 --cx 320.0 --cy 240.0
    """
    parser = argparse.ArgumentParser(description='Create point cloud from RGB-D images')
    parser.add_argument('--color', type=str, required=True, help='Path to color image (RGB)')
    parser.add_argument('--depth', type=str, required=True, help='Path to depth image (.pt file)')
    parser.add_argument('--intrinsics', type=str, help='Path to camera intrinsics (JSON file)')
    parser.add_argument('--fx', type=float, help='Camera focal length x')
    parser.add_argument('--fy', type=float, help='Camera focal length y')
    parser.add_argument('--cx', type=float, help='Camera principal point x')
    parser.add_argument('--cy', type=float, help='Camera principal point y')
    parser.add_argument('--output-dir', type=str, help='Output directory for point cloud files')
    
    args = parser.parse_args()
    
    # Initialize point cloud generator
    point_cloud_generator = PointCloudGenerator(
        intrinsics=args.intrinsics,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy
    )
    
    # Process images and generate point cloud
    points, colors = point_cloud_generator.process_images(args.color, args.depth)
    
    # Visualize point cloud
    point_cloud_generator.visualize_point_cloud(points, colors, args.output_dir)

if __name__ == "__main__":
    main()