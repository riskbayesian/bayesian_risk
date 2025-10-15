#!/usr/bin/env python3

from rosbags.rosbag2 import Reader
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import geometry_msgs__msg__PoseStamped as PoseStamped
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import sensor_msgs__msg__CameraInfo as CameraInfo
from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImage
import sys
from collections import defaultdict
import numpy as np
from cv_bridge import CvBridge
import cv2
import json
from datetime import datetime
import os

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Define the topics and their message types
TARGET_TOPICS = {
    '/vrpn_mocap/fmu/pose': PoseStamped,
    '/zed/zed_node/depth/depth_registered': Image,
    '/zed/zed_node/rgb/camera_info': CameraInfo,
    '/zed/zed_node/rgb/image_rect_color/compressed': CompressedImage
}

# Register message types
# register_types({
#     'geometry_msgs/msg/PoseStamped': """
#     builtin_interfaces/Time stamp
#     string frame_id
#     geometry_msgs/Pose pose
#     """,
#     'geometry_msgs/msg/Pose': """
#     geometry_msgs/Point position
#     geometry_msgs/Quaternion orientation
#     """,
#     'geometry_msgs/msg/Point': """
#     float64 x
#     float64 y
#     float64 z
#     """,
#     'geometry_msgs/msg/Quaternion': """
#     float64 x
#     float64 y
#     float64 z
#     float64 w
#     """,
#     'sensor_msgs/msg/Image': """
#     std_msgs/Header header
#     uint32 height
#     uint32 width
#     string encoding
#     uint8 is_bigendian
#     uint32 step
#     uint8[] data
#     """,
#     'sensor_msgs/msg/CameraInfo': """
#     std_msgs/Header header
#     uint32 height
#     uint32 width
#     string distortion_model
#     float64[] d
#     float64[9] k
#     float64[9] r
#     float64[12] p
#     """,
#     'sensor_msgs/msg/CompressedImage': """
#     std_msgs/Header header
#     string format
#     uint8[] data
#     """,
#     'std_msgs/msg/Header': """
#     builtin_interfaces/Time stamp
#     string frame_id
#     """,
#     'builtin_interfaces/msg/Time': """
#     int32 sec
#     uint32 nanosec
#     """
# })

class BagExtractor:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.bridge = CvBridge()
        self.data = {
            'pose': {'timestamps': [], 'data': []},
            'depth': {'timestamps': [], 'data': []},
            'camera_info': {'timestamps': [], 'data': []},
            'rgb': {'timestamps': [], 'data': []}
        }
        
    def extract_data(self):
        """Extract data from the bag file for specified topics."""
        #try:
        # Create a type store to use if the bag has no message definitions.
        typestore = get_typestore(Stores.ROS2_FOXY)

        # Create reader instance and open for reading.
        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # Process based on topic type
                if connection.topic == '/vrpn_mocap/fmu/pose':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    pose_data = self._extract_pose(msg)
                    if pose_data is not None:
                        self.data['pose']['timestamps'].append(timestamp)
                        self.data['pose']['data'].append(pose_data)
                    
                elif connection.topic == '/zed/zed_node/depth/depth_registered':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    depth_data = self._extract_depth(msg)
                    if depth_data is not None:
                        self.data['depth']['timestamps'].append(timestamp)
                        self.data['depth']['data'].append(depth_data)
                    
                elif connection.topic == '/zed/zed_node/rgb/camera_info':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    camera_info = self._extract_camera_info(msg)
                    if camera_info is not None:
                        self.data['camera_info']['timestamps'].append(timestamp)
                        self.data['camera_info']['data'].append(camera_info)
                    
                elif connection.topic == '/zed/zed_node/rgb/image_rect_color/compressed':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    rgb_data = self._extract_compressed_image(msg)
                    if rgb_data is not None:
                        self.data['rgb']['timestamps'].append(timestamp)
                        self.data['rgb']['data'].append(rgb_data)
            
            # Convert lists to numpy arrays
            self._convert_to_numpy()
            
            # Save the data
            self._save_data()
                
    def _extract_pose(self, msg):
        """Extract pose data from PoseStamped message."""
        # Print message structure for debugging
        return {
            'position': np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]),
            'orientation': np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
        }
    
    def _extract_depth(self, msg):
        """Extract depth image data."""
        # Print debug information
        # print(f"Depth message info:")
        # print(f"  Height: {msg.height}")
        # print(f"  Width: {msg.width}")
        # print(f"  Encoding: {msg.encoding}")
        # print(f"  Step: {msg.step}")
        # print(f"  Data size: {len(msg.data)} bytes")
        
        # Convert to numpy array based on encoding
        if msg.encoding == '32FC1':
            # 32-bit float (4 bytes per pixel)
            depth_array = np.frombuffer(msg.data, dtype=np.float32)
        else:
            # Default to uint16 (2 bytes per pixel)
            depth_array = np.frombuffer(msg.data, dtype=np.uint16)
            
        # print(f"  Array size: {depth_array.size}")
        # print(f"  Expected size: {msg.height * msg.width}")
        
        # Reshape the array
        depth_array = depth_array.reshape((msg.height, msg.width))
        
        # For float depth data, we might want to handle invalid values
        if msg.encoding == '32FC1':
            # Replace NaN and inf values with 0
            depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return depth_array
    
    def _extract_camera_info(self, msg):
        """Extract camera info data."""
        return {
            'width': msg.width,
            'height': msg.height,
            'distortion_model': msg.distortion_model,
            'D': np.array(msg.d),
            'K': np.array(msg.k).reshape(3, 3),
            'R': np.array(msg.r).reshape(3, 3),
            'P': np.array(msg.p).reshape(3, 4)
        }
    
    def _extract_compressed_image(self, msg):
        """Extract compressed image data."""
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    
    def _convert_to_numpy(self):
        """Convert all data to numpy arrays."""
        for key in self.data:
            self.data[key]['timestamps'] = np.array(self.data[key]['timestamps'])
            if key == 'pose':
                positions = np.array([d['position'] for d in self.data[key]['data']])
                orientations = np.array([d['orientation'] for d in self.data[key]['data']])
                self.data[key]['data'] = {
                    'positions': positions,
                    'orientations': orientations
                }
            elif key == 'camera_info':
                # Keep camera info as a list of dictionaries
                pass
            else:
                self.data[key]['data'] = np.array(self.data[key]['data'])
    
    def _save_data(self):
        """Save the extracted data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"extracted_data_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numpy arrays
        np.savez(f"{output_dir}/pose_data.npz",
                 timestamps=self.data['pose']['timestamps'],
                 positions=self.data['pose']['data']['positions'],
                 orientations=self.data['pose']['data']['orientations'])
        
        np.savez(f"{output_dir}/depth_data.npz",
                 timestamps=self.data['depth']['timestamps'],
                 data=self.data['depth']['data'])
        
        np.savez(f"{output_dir}/rgb_data.npz",
                 timestamps=self.data['rgb']['timestamps'],
                 data=self.data['rgb']['data'])
        
        # Save camera info as JSON
        with open(f"{output_dir}/camera_info.json", 'w') as f:
            json.dump({
                'timestamps': self.data['camera_info']['timestamps'].tolist(),
                'data': [{
                    'width': info['width'],
                    'height': info['height'],
                    'distortion_model': info['distortion_model'],
                    'D': info['D'].tolist(),
                    'K': info['K'].tolist(),
                    'R': info['R'].tolist(),
                    'P': info['P'].tolist()
                } for info in self.data['camera_info']['data']]
            }, f, indent=2)
        
        print(f"Data saved to directory: {output_dir}")
        print(f"Number of messages extracted:")
        print(f"  Pose messages: {len(self.data['pose']['timestamps'])}")
        print(f"  Depth images: {len(self.data['depth']['timestamps'])}")
        print(f"  Camera info: {len(self.data['camera_info']['timestamps'])}")
        print(f"  RGB images: {len(self.data['rgb']['timestamps'])}")

# Main execution
bag_path = Path("rosbag2_2025_04_03-16_04_30")
extractor = BagExtractor(bag_path)
extractor.extract_data()

#%%