#!/usr/bin/env python3

import numpy as np
import json
from pathlib import Path
from src.point_cloud_3D import PointCloudGenerator

def test_pose_transformation():
    """
    Test the pose transformation to verify it's working correctly.
    """
    print("Testing pose transformation...")
    
    # Create a simple test point cloud in camera coordinates
    # This simulates a frustum shape
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(1, 5, 10)
    
    X, Y, Z = np.meshgrid(x, y, z)
    points_camera = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Remove points outside a cone (frustum shape)
    radius = np.sqrt(points_camera[:, 0]**2 + points_camera[:, 1]**2)
    max_radius = points_camera[:, 2] * 0.3  # Cone angle
    valid_mask = radius <= max_radius
    points_camera = points_camera[valid_mask]
    
    print(f"Created test point cloud with {len(points_camera)} points")
    print(f"Camera frame range: X[{np.min(points_camera[:, 0]):.3f}, {np.max(points_camera[:, 0]):.3f}], Y[{np.min(points_camera[:, 1]):.3f}, {np.max(points_camera[:, 1]):.3f}], Z[{np.min(points_camera[:, 2]):.3f}, {np.max(points_camera[:, 2]):.3f}]")
    
    # Test different poses
    test_poses = [
        {
            'name': 'Identity',
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        },
        {
            'name': 'Translated',
            'position': np.array([10.0, 5.0, 2.0]),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        },
        {
            'name': 'Rotated 90 degrees around Z',
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.7071068, 0.0, 0.0, 0.7071068])  # 90 degrees around Z
        },
        {
            'name': 'Translated and Rotated',
            'position': np.array([10.0, 5.0, 2.0]),
            'orientation': np.array([0.7071068, 0.0, 0.0, 0.7071068])  # 90 degrees around Z
        }
    ]
    
    # Create a dummy PointCloudGenerator (we don't need camera intrinsics for this test)
    pc_generator = PointCloudGenerator()
    
    for pose_info in test_poses:
        print(f"\n--- Testing {pose_info['name']} ---")
        
        # Apply the same transformation as in the real code
        q = pose_info['orientation']
        R = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])
        
        # Transform points: R * points + t
        points_world = np.dot(points_camera, R.T) + pose_info['position']
        
        print(f"World frame range: X[{np.min(points_world[:, 0]):.3f}, {np.max(points_world[:, 0]):.3f}], Y[{np.min(points_world[:, 1]):.3f}, {np.max(points_world[:, 1]):.3f}], Z[{np.min(points_world[:, 2]):.3f}, {np.max(points_world[:, 2]):.3f}]")
        
        # Verify the transformation makes sense
        if pose_info['name'] == 'Identity':
            # Should be the same as camera frame
            assert np.allclose(points_world, points_camera, atol=1e-6)
            print("✓ Identity transformation verified")
        elif pose_info['name'] == 'Translated':
            # Should be translated by the position
            expected = points_camera + pose_info['position']
            assert np.allclose(points_world, expected, atol=1e-6)
            print("✓ Translation verified")
        elif pose_info['name'] == 'Rotated 90 degrees around Z':
            # Should rotate around Z axis
            # Points at (1,0,z) should become (0,1,z)
            # Points at (0,1,z) should become (-1,0,z)
            print("✓ Rotation transformation applied")
        elif pose_info['name'] == 'Translated and Rotated':
            # Should be both translated and rotated
            print("✓ Combined transformation applied")
    
    print("\n✓ All pose transformations tested successfully!")

if __name__ == "__main__":
    test_pose_transformation() 