#!/usr/bin/env python3
"""
ZMQ Robot Interface for real robot communication
"""

import zmq
import numpy as np
import cv2


class ZMQRobotInterface:
    """ZMQ interface for bidirectional robot communication"""
    
    def __init__(self, request_port=5555, response_port=5556, timeout=30000):
        """Initialize ZMQ sockets for bidirectional communication"""
        self.context = zmq.Context()
        # Socket for sending commands to robot
        self.request_socket = self.context.socket(zmq.REQ)
        self.request_socket.connect(f"tcp://localhost:{request_port}")
        self.request_socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.request_socket.setsockopt(zmq.SNDTIMEO, timeout)
        print(f"ZMQ: Connected to robot on port {request_port}")
    
    def get_observation(self):
        """
        Request current observation from robot
        Returns:
            dict: {
                'rgb': np.ndarray (H, W, 3) uint8,
                'depth': np.ndarray (H, W) uint16,
                'K': np.ndarray (3, 3) camera intrinsics,
                'cam_extrinsic': np.ndarray (4, 4) camera extrinsics,
                'gripper_pose': dict with 'position' and 'orientation',
                'gripper_open': float [0, 1]
            }
        """
        request = {'type': 'get_observation'}
        self.request_socket.send_json(request)
        
        response = self.request_socket.recv_json()
        
        if response['status'] != 'success':
            raise RuntimeError(f"Failed to get observation: {response.get('message', 'Unknown error')}")
        
        data = response['data']
        
        # Decode images from base64 or raw bytes
        rgb = self._decode_image(data['rgb'], dtype=np.uint8)
        depth = self._decode_image(data['depth'], dtype=np.uint16)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'K': np.array(data['K']).reshape(3, 3),
            'cam_extrinsic': np.array(data['cam_extrinsic']).reshape(4, 4),
            'gripper_pose': data['gripper_pose'],
            'gripper_open': data.get('gripper_open', 1.0)
        }
    
    def execute_trajectory(self, trajectory, mode='grasp', blocking=True):
        """
        Send trajectory to robot for execution
        
        Args:
            trajectory: List of 4x4 pose matrices (numpy or list) or dict with 'poses' and 'gripper_open'
            mode: 'grasp' or 'manip'
            blocking: Wait for execution completion
            
        Returns:
            dict: {'status': 'success'/'failed', 'message': str}
        """
        # Convert trajectory to serializable format
        if isinstance(trajectory, list):
            # Handle list of numpy arrays or nested lists
            poses_list = [pose.tolist() if hasattr(pose, 'tolist') else pose for pose in trajectory]
            request = {
                'type': 'execute_trajectory',
                'mode': mode,
                'poses': poses_list,
                'blocking': blocking
            }
        elif isinstance(trajectory, np.ndarray):
            # Handle single numpy array or array of arrays
            if trajectory.ndim == 2:
                # Single 4x4 matrix
                poses_list = [trajectory.tolist()]
            else:
                poses_list = [pose.tolist() for pose in trajectory]
            request = {
                'type': 'execute_trajectory',
                'mode': mode,
                'poses': poses_list,
                'blocking': blocking
            }
        else:
            raise ValueError(f"Invalid trajectory type: {type(trajectory)}")
        
        self.request_socket.send_json(request)
        response = self.request_socket.recv_json()
        
        return response
    
    def close_gripper(self, blocking=True):
        """Close gripper"""
        request = {'type': 'close_gripper', 'blocking': blocking}
        self.request_socket.send_json(request)
        return self.request_socket.recv_json()
    
    def open_gripper(self, blocking=True):
        """Open gripper"""
        request = {'type': 'open_gripper', 'blocking': blocking}
        self.request_socket.send_json(request)
        return self.request_socket.recv_json()
    
    def reset_robot(self, blocking=True):
        """Reset robot to home position"""
        request = {'type': 'reset_robot', 'blocking': blocking}
        self.request_socket.send_json(request)
        return self.request_socket.recv_json()
    
    def _decode_image(self, data, dtype=np.uint8):
        """Decode image from base64 PNG"""
        import base64
        if isinstance(data, str):
            # Base64 encoded PNG
            img_bytes = base64.b64decode(data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            # Decode PNG (cv2.imdecode returns BGR for color images)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            # Convert BGR to RGB for color images
            if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(data, dict):
            # Contains shape info
            img_bytes = base64.b64decode(data['data'])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            raise ValueError(f"Unknown image data format: {type(data)}")
    
    def close(self):
        """Close ZMQ sockets"""
        self.request_socket.close()
        self.context.term()
