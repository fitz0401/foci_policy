#!/usr/bin/env python3
"""
FOCI Policy RealWorld Script
"""

import os
os.environ["WP_CUDA_ALLOCATOR"] = "malloc"
import sys
import argparse
import json
import time
import numpy as np
import cv2
import torch
import yaml
import zmq
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from foci_policy.model.foci_actor import FOCIActor
from foci_real_world.pose_estimator.mask_generator import MaskGenerator
from foci_real_world.pose_estimator.fp_real_world_utils import (
    preprocess_depth,
    estimate_poses_online,
    estimate_poses_online_optimized,
)
from foundation_pose.wrapper import FoundationPoseWrapper
from foci_real_world.utils.preprocess_realworld_data import to_o3d_pcd
from foci_real_world.scripts.zmq_robot_interface import ZMQRobotInterface

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

TABLE_HEIGHT = 0.055  # meters

class FOCIPolicyRobotController:
    """Real robot controller using FOCI policy"""
    
    def __init__(self, task_name, config_dir='../config', checkpoint_dir='../checkpoints',
                 assets_dir='../assets', device='cuda:0', voxel_size=0.004, num_points=2048):
        """
        Initialize robot controller
        
        Args:
            task_name: Task name (must be in task_config.yaml)
            config_dir: Configuration directory
            checkpoint_dir: Checkpoint directory
            assets_dir: Assets directory (meshes)
            device: torch device
            voxel_size: Voxel size for downsampling
            num_points: Number of points per object
        """
        self.task_name = task_name
        self.config_dir = Path(config_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.assets_dir = Path(assets_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.voxel_size = voxel_size
        self.num_points = num_points
        
        print(f"Initializing FOCI Policy Controller for task: {task_name}")
        print(f"Device: {self.device}")
        
        # Load task configuration
        self._load_task_config()
        
        # Store checkpoint paths for delayed loading
        self.grasp_ckpt = self.checkpoint_dir / 'foci' / 'real' / 'grasp' / 'best_model.pth'
        self.manip_ckpt = self.checkpoint_dir / 'foci' / 'real' / 'manip' / 'best_model.pth'
        self.model_config_dir = self.checkpoint_dir / '..' / 'config'
        
        if not self.grasp_ckpt.exists():
            raise FileNotFoundError(f"Grasp checkpoint not found: {self.grasp_ckpt}")
        if not self.manip_ckpt.exists():
            raise FileNotFoundError(f"Manip checkpoint not found: {self.manip_ckpt}")
        
        # Set actor references to None (will be loaded on demand)
        self.grasp_actor = None
        self.manip_actor = None
        
        # Initialize language encoder and encode task instructions
        print("Encoding language instructions...")
        from foci_policy.model.foci_dataloader import AddLanguageEmbedding
        self.language_encoder = AddLanguageEmbedding(device=self.device)
        # Encode grasp instruction
        data_grasp = {'language': self.lan_grasp}
        data_grasp = self.language_encoder(data_grasp)
        self.lan_grasp_embedding = data_grasp['language_embedding']
        # Encode manip instruction
        data_manip = {'language': self.lan_manip}
        data_manip = self.language_encoder(data_manip)
        self.lan_manip_embedding = data_manip['language_embedding']
        print(f"✓ Encoded grasp instruction: '{self.lan_grasp}'")
        print(f"✓ Encoded manip instruction: '{self.lan_manip}'")
        print("✓ Initialization complete\n")
    
    def _load_task_config(self):
        """Load task configuration from task_config.yaml"""
        task_config_path = self.config_dir / 'task_config.yaml'
        if not task_config_path.exists():
            task_config_path = Path(__file__).parent / 'configs' / 'task_config.yaml'
        
        with open(task_config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        if self.task_name not in all_configs:
            raise ValueError(f"Task {self.task_name} not found in task_config.yaml")
        
        self.task_config = all_configs[self.task_name]
        
        # Extract object information
        self.base_object = self.task_config['objects'][0]  # pa
        self.moving_object = self.task_config['objects'][1]  # pb
        self.base_mesh = self.assets_dir / f"{self.task_config['mesh_names'][0]}.obj"
        self.moving_mesh = self.assets_dir / f"{self.task_config['mesh_names'][1]}.obj"
        self.lan_grasp = self.task_config['lan_grasp']
        self.lan_manip = self.task_config['lan_manip']
        
        # Verify meshes exist
        if not self.base_mesh.exists():
            raise FileNotFoundError(f"Base mesh not found: {self.base_mesh}")
        if not self.moving_mesh.exists():
            raise FileNotFoundError(f"Moving mesh not found: {self.moving_mesh}")
        
        print(f"Task config loaded:")
        print(f"  Base object: {self.base_object} (mesh: {self.base_mesh.name})")
        print(f"  Moving object: {self.moving_object} (mesh: {self.moving_mesh.name})")
        print(f"  Grasp instruction: '{self.lan_grasp}'")
        print(f"  Manip instruction: '{self.lan_manip}'")
    
    def _load_actor(self, mode='grasp'):
        """Load FOCI actor on demand"""
        if mode == 'grasp' and self.grasp_actor is None:
            print(f"Loading grasp actor...")
            torch.cuda.empty_cache()
            self.grasp_actor = FOCIActor(
                checkpoint_path=str(self.grasp_ckpt),
                config_dir=self.model_config_dir,
                device=self.device
            )
            print(f"✓ Grasp actor loaded")
        elif mode == 'manip' and self.manip_actor is None:
            print(f"Loading manip actor...")
            torch.cuda.empty_cache()
            self.manip_actor = FOCIActor(
                checkpoint_path=str(self.manip_ckpt),
                config_dir=self.model_config_dir,
                device=self.device
            )
            print(f"✓ Manip actor loaded")
    
    def _cleanup_actors(self):
        """Clean up all FOCI actors to free GPU memory"""
        if self.grasp_actor is not None:
            del self.grasp_actor
            self.grasp_actor = None
        if self.manip_actor is not None:
            del self.manip_actor
            self.manip_actor = None
        torch.cuda.empty_cache()
        print("✓ FOCI actors cleaned up")
    
    def postprocess_trajectory(self, trajectory, mode='grasp'):
        """ Post-process trajectory for real robot execution. """
        if len(trajectory) == 0:
            return trajectory
        
        if mode == 'grasp':
            goal_pose = trajectory[-1].copy()
            pre_grasp_pose = goal_pose.copy()
            gripper_z_axis = goal_pose[:3, 2]  # Z-axis of gripper frame (forward direction)
            pre_grasp_pose[:3, 3] -= gripper_z_axis * 0.06  # Move back 6cm
            trajectory = [pre_grasp_pose, goal_pose]
        elif mode == 'manip':
            if self.task_name == 'sweep_dust':
                for pose in trajectory:
                    pose[2, 3] += 0.01  # Lift 1cm for dust sweeping
            elif self.task_name == 'insert_tube':
                for pose in trajectory:
                    pose[:3, 3] -= pose[:3, 2] * 0.01  # Move back 1cm along gripper z-axis
                    pose[2, 3] += 0.01  # Lift 1cm before insertion
        # Reference pose is the last waypoint
        ref_pose = trajectory[-1]
        ref_rot = Rotation.from_matrix(ref_pose[:3, :3])
        # Process each waypoint
        for i, pose in enumerate(trajectory):
            # 1. Enforce minimum table height
            if pose[2, 3] < TABLE_HEIGHT:
                pose[2, 3] = TABLE_HEIGHT
            # 2. Handle gripper symmetry (skip the reference pose itself)
            if i < len(trajectory) - 1:
                curr_rot = Rotation.from_matrix(pose[:3, :3])
                # Compute rotation difference
                rot_diff = ref_rot * curr_rot.inv()
                angle = rot_diff.magnitude()  # rotation angle in radians
                # If rotation difference > 90 degrees, use equivalent rotation (flip 180°)
                if angle > np.pi / 2:
                    # Apply 180-degree rotation around gripper Z-axis (in gripper frame)
                    flip_rot = Rotation.from_euler('z', np.pi)
                    # Transform to world frame and apply
                    new_rot = curr_rot * flip_rot
                    pose[:3, :3] = new_rot.as_matrix()
        return trajectory
    
    def estimate_object_poses(self, obs, verbose=False, save_debug=True):
        """ Estimate object poses using FoundationPose """        
        debug_dir = Path(__file__).parent / 'debug' if save_debug else None
        result = estimate_poses_online_optimized(
            rgb=obs['rgb'],
            depth=obs['depth'],
            K=obs['K'],
            cam_extrinsic=obs['cam_extrinsic'],
            base_object_name=self.base_object,
            moving_object_name=self.moving_object,
            base_mesh_path=str(self.base_mesh),
            moving_mesh_path=str(self.moving_mesh),
            verbose=verbose,
            save_debug=debug_dir
        )
        return result
 
    def extract_point_clouds(self, obs, pose_result):
        """ Extract masked point clouds for both objects """
        rgb = obs['rgb']
        depth = obs['depth']
        K = obs['K']
        cam_extrinsic = obs['cam_extrinsic']
        
        # Preprocess depth
        if depth.dtype == np.uint16:
            depth = preprocess_depth(depth)
        
        base_mask = pose_result['base_mask']
        moving_mask = pose_result['moving_mask']
        
        # Extract masked point clouds
        def extract_masked_pcd(mask):
            depth_h, depth_w = depth.shape
            mask_h, mask_w = mask.shape
            if mask_h != depth_h or mask_w != depth_w:
                mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
            
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            v, u = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
            valid_depth = (depth > 0.01) & (depth < 3.0)
            valid_mask = mask.astype(bool) & valid_depth
            
            v_valid = v[valid_mask]
            u_valid = u[valid_mask]
            z_valid = depth[valid_mask]
            
            x = (u_valid - cx) * z_valid / fx
            y = (v_valid - cy) * z_valid / fy
            z = z_valid
            points_cam = np.stack([x, y, z], axis=1)
            
            # Transform to world coordinates
            points_world = (cam_extrinsic[:3, :3] @ points_cam.T).T + cam_extrinsic[:3, 3]
            
            colors = rgb[valid_mask].astype(np.float32) / 255.0
            return points_world, colors
        
        pa_points_raw, pa_colors_raw = extract_masked_pcd(base_mask)
        pb_points_raw, pb_colors_raw = extract_masked_pcd(moving_mask)
        
        # Downsample and sample to num_points
        def process_pcd(points, colors):
            if len(points) == 0:
                return np.zeros((self.num_points, 3)), np.zeros((self.num_points, 3))
            
            pcd = to_o3d_pcd(points, colors)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            if len(pcd.points) > self.num_points:
                indices = np.random.choice(len(pcd.points), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(pcd.points), self.num_points, replace=True)
            
            return np.asarray(pcd.points)[indices], np.asarray(pcd.colors)[indices]
        
        pa_points, pa_colors = process_pcd(pa_points_raw, pa_colors_raw)
        pb_points, pb_colors = process_pcd(pb_points_raw, pb_colors_raw)
        
        return {
            'pa_points': pa_points,
            'pa_colors': pa_colors,
            'pb_points': pb_points,
            'pb_colors': pb_colors
        }

    def predict_trajectory(self, obs, pose_result, pcd_result, mode='grasp'):
        """ Predict trajectory using FOCI """
        self._load_actor(mode=mode)
        
        gripper_dict = obs['gripper_pose']
        gripper_pos = np.array([
            gripper_dict['position']['x'],
            gripper_dict['position']['y'],
            gripper_dict['position']['z']
        ])
        gripper_quat = np.array([
            gripper_dict['orientation']['x'],
            gripper_dict['orientation']['y'],
            gripper_dict['orientation']['z'],
            gripper_dict['orientation']['w']
        ])
        gripper_pose = np.eye(4)
        gripper_pose[:3, :3] = Rotation.from_quat(gripper_quat).as_matrix()
        gripper_pose[:3, 3] = gripper_pos

        pa_pose = pose_result['base_pose']
        pb_pose = pose_result['moving_pose']
        pa_pcd = to_o3d_pcd(pcd_result['pa_points'], pcd_result['pa_colors'])
        pb_pcd = to_o3d_pcd(pcd_result['pb_points'], pcd_result['pb_colors'])

        if mode == 'grasp':
            actor = self.grasp_actor
            language_embedding = self.lan_grasp_embedding.unsqueeze(0).to(self.device)
        elif mode == 'manip':
            actor = self.manip_actor
            language_embedding = self.lan_manip_embedding.unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        traj_world = actor.act(
            pa_pcd=pa_pcd,
            pb_pcd=pb_pcd,
            pa_pose=pa_pose,
            pb_pose=pb_pose,
            gripper_pose=gripper_pose,
            language_embedding=language_embedding
        )

        # Post-process trajectory
        traj_world = self.postprocess_trajectory(traj_world, mode=mode)
        
        # Clean up actor immediately after use
        if mode == 'grasp':
            del self.grasp_actor
            self.grasp_actor = None
        elif mode == 'manip':
            del self.manip_actor
            self.manip_actor = None
        torch.cuda.empty_cache()
        print(f"✓ {mode.capitalize()} actor cleaned up")

        return traj_world
    
    @staticmethod
    def draw_pose_axis(ax, pose, length=0.08, label=None):
        origin = pose[:3, 3]
        R = pose[:3, :3]
        colors = ['r', 'g', 'b']
        axis_labels = ['x', 'y', 'z']
        for i in range(3):
            axis = R[:, i] * length
            ax.plot([origin[0], origin[0]+axis[0]],
                    [origin[1], origin[1]+axis[1]],
                    [origin[2], origin[2]+axis[2]],
                    color=colors[i], linewidth=2)
            if label is not None:
                ax.text(origin[0]+axis[0], origin[1]+axis[1], origin[2]+axis[2], f'{label}-{axis_labels[i]}', color=colors[i], fontsize=8)

    def visualize_prediction(self, obs, pose_result, pcd_result, trajectory, mode='grasp', save_image=True):
        """Visualize predicted trajectory"""
        geometries = []
        
        # Extract full scene point cloud from observation
        rgb = obs['rgb']
        depth = obs['depth']
        K = obs['K']
        cam_extrinsic = obs['cam_extrinsic']
        # Preprocess depth if needed
        if depth.dtype == np.uint16:
            depth_processed = preprocess_depth(depth)
        else:
            depth_processed = depth
        # Generate full scene point cloud
        depth_h, depth_w = depth_processed.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        v, u = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
        valid_depth = (depth_processed > 0.01) & (depth_processed < 3.0)
        v_valid = v[valid_depth]
        u_valid = u[valid_depth]
        z_valid = depth_processed[valid_depth]
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid
        points_cam = np.stack([x, y, z], axis=1)
        scene_points = (cam_extrinsic[:3, :3] @ points_cam.T).T + cam_extrinsic[:3, 3]
        scene_colors = rgb[valid_depth].astype(np.float32) / 255.0
        if len(scene_points) > 10000:
            indices = np.random.choice(len(scene_points), 10000, replace=False)
            scene_points = scene_points[indices]
            scene_colors = scene_colors[indices]
        
        # World frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(world_frame)
        
        # Full scene point cloud (dimmed)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors * 0.5)  # Dim the scene
        geometries.append(scene_pcd)
        
        # Point clouds
        pa_pcd = o3d.geometry.PointCloud()
        pa_pcd.points = o3d.utility.Vector3dVector(pcd_result['pa_points'])
        pa_pcd.colors = o3d.utility.Vector3dVector(pcd_result['pa_colors'])
        geometries.append(pa_pcd)
        
        pb_pcd = o3d.geometry.PointCloud()
        pb_pcd.points = o3d.utility.Vector3dVector(pcd_result['pb_points'])
        pb_pcd.colors = o3d.utility.Vector3dVector(pcd_result['pb_colors'])
        geometries.append(pb_pcd)
        
        # Object poses
        pa_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        pa_frame.transform(pose_result['base_pose'])
        geometries.append(pa_frame)
        
        pb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        pb_frame.transform(pose_result['moving_pose'])
        geometries.append(pb_frame)
        
        # Trajectory
        for i, pose in enumerate(trajectory):
            traj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            traj_frame.transform(pose)
            geometries.append(traj_frame)
        
        # Trajectory line
        if len(trajectory) > 1:
            traj_positions = [pose[:3, 3] for pose in trajectory]
            lines = [[i, i+1] for i in range(len(traj_positions)-1)]
            colors = [[1.0, 0.6, 0.0] for _ in lines]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(traj_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
        
        if save_image:
            script_dir = Path(__file__).parent
            debug_dir = script_dir / 'debug' / self.task_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization data
            vis_data = {
                'scene_points': scene_points,
                'scene_colors': scene_colors,
                'pa_points': pcd_result['pa_points'],
                'pa_colors': pcd_result['pa_colors'],
                'pb_points': pcd_result['pb_points'],
                'pb_colors': pcd_result['pb_colors'],
                'base_pose': pose_result['base_pose'],
                'moving_pose': pose_result['moving_pose'],
                'trajectory': np.array([pose for pose in trajectory]),
                'mode': mode,
                'rgb': rgb,
                'depth': depth,
                'K': K,
                'cam_extrinsic': cam_extrinsic
            }
            
            data_path = debug_dir / f'vis_data_{mode}.npz'
            np.savez(data_path, **vis_data)
            print(f"✓ Visualization data saved to {data_path}")
            
            save_path = debug_dir / f'foci_prediction_{mode}.png'
            
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                # Use matplotlib for headless rendering
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot scene point cloud (dim background)
                if len(scene_points) > 5000:
                    scene_indices = np.random.choice(len(scene_points), 5000, replace=False)
                    scene_pts_plot = scene_points[scene_indices]
                    scene_cols_plot = scene_colors[scene_indices] * 0.5
                else:
                    scene_pts_plot = scene_points
                    scene_cols_plot = scene_colors * 0.5
                ax.scatter(scene_pts_plot[:, 0], scene_pts_plot[:, 1], scene_pts_plot[:, 2],
                          c=scene_cols_plot, s=0.5, alpha=0.3, label='Scene')
                
                # Plot point clouds
                pa_pts = pcd_result['pa_points']
                pa_cols = pcd_result['pa_colors']
                ax.scatter(pa_pts[:, 0], pa_pts[:, 1], pa_pts[:, 2], 
                          c=pa_cols, s=2, alpha=0.8, label='Base object')
                
                pb_pts = pcd_result['pb_points']
                pb_cols = pcd_result['pb_colors']
                ax.scatter(pb_pts[:, 0], pb_pts[:, 1], pb_pts[:, 2], 
                          c=pb_cols, s=2, alpha=0.8, label='Moving object')
                
                # Plot trajectory
                if len(trajectory) > 1:
                    traj_positions = np.array([pose[:3, 3] for pose in trajectory])
                    ax.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2],
                           'o-', color='orange', linewidth=2, markersize=4, label='Trajectory')

                # Base object pose
                self.draw_pose_axis(ax, pose_result['base_pose'], length=0.08, label='base')
                # Moving object pose
                self.draw_pose_axis(ax, pose_result['moving_pose'], length=0.08, label='moving')

                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'FOCI Prediction - {mode.upper()} mode')
                ax.legend()
                xyz_limits = np.array([
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.4]
                ])
                xyz_center = np.mean(xyz_limits, axis=1)
                xyz_range = np.max(xyz_limits[:,1] - xyz_limits[:,0]) / 2
                for ctr, set_lim in zip(xyz_center, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
                    set_lim(ctr - xyz_range, ctr + xyz_range)

                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ Visualization saved to {save_path}")
            except Exception as e:
                print(f"⚠ Visualization save failed (SSH/headless environment): {e}")
                print(f"  Skipping visualization...")
        else:
            try:
                o3d.visualization.draw_geometries(
                    geometries,
                    window_name=f"FOCI Prediction - {mode.upper()} mode",
                    width=1280,
                    height=720
                )
            except Exception as e:
                print(f"⚠ Visualization failed (no display): {e}")
    
    def execute_task(self, robot_interface, visualize=False):
        """ Execute full pick and place task """
        print("="*70)
        print(f"Executing task: {self.task_name}")
        print("="*70)
        
        try:
            # ========== GRASP PHASE ==========
            print("\n[1/2] GRASP PHASE")
            print("-"*70)
            
            # Get observation
            print("Getting observation...")
            obs = robot_interface.get_observation()
            print("✓ Observation received")
            
            # Estimate poses
            print("Estimating object poses...")
            pose_result = self.estimate_object_poses(obs, verbose=True)
            
            # Extract point clouds
            print("Extracting point clouds...")
            pcd_result = self.extract_point_clouds(obs, pose_result)
            print(f"✓ PA points: {pcd_result['pa_points'].shape}")
            print(f"✓ PB points: {pcd_result['pb_points'].shape}")
            
            # Predict grasp trajectory
            print("Predicting grasp trajectory...")
            grasp_traj = self.predict_trajectory(obs, pose_result, pcd_result, mode='grasp')
            print(f"✓ Predicted {len(grasp_traj)} waypoints")
            
            # Visualize if requested
            if visualize:
                print(f"  Grasp traj: {grasp_traj}")
                self.visualize_prediction(obs, pose_result, pcd_result, grasp_traj, mode='grasp', save_image=True)
            
            # Execute grasp trajectory
            print("Executing grasp trajectory...")
            result = robot_interface.execute_trajectory(grasp_traj, mode='grasp', blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Grasp execution failed: {result.get('message', 'Unknown error')}")
            print("✓ Grasp trajectory executed")
            time.sleep(1.0)

            # Close gripper
            print("Closing gripper...")
            result = robot_interface.close_gripper(blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Gripper close failed: {result.get('message', 'Unknown error')}")
            print("✓ Gripper closed")
            time.sleep(2.0)  # Wait for secure grasp
            
            # Lift up after grasping
            lift_height = 0.0
            if self.task_name not in ["open_drawer"]:
                print("Lifting object...")
                lift_height = 0.20
                final_grasp_pose = grasp_traj[-1].copy()
                lift_pose = final_grasp_pose.copy()
                lift_pose[2, 3] += lift_height  # Move up in world Z axis
                lift_traj = [lift_pose]
                result = robot_interface.execute_trajectory(lift_traj, mode='lift', blocking=True)
                if result['status'] != 'success':
                    raise RuntimeError(f"Lift failed: {result.get('message', 'Unknown error')}")
                print("✓ Object lifted")
                time.sleep(1.0)
            
            # ========== MANIPULATION PHASE ==========
            print("\n[2/2] MANIPULATION PHASE")
            print("-"*70)
            
            # Get new observation
            print("Getting observation...")
            obs = robot_interface.get_observation()
            print("✓ Observation received")
            
            # Estimate poses
            print("Estimating object poses...")
            # pose_result_new = self.estimate_object_poses(obs, verbose=True)
            # Use previous base pose and lifted moving pose
            pa_pose_current = pose_result['base_pose']
            pb_pose_current = pose_result['moving_pose'].copy()
            pb_pose_current[:3, 3] += np.array([0, 0, lift_height])
            pose_result_manip = {
                'base_pose': pa_pose_current,
                'moving_pose': pb_pose_current,
                'base_mask': pose_result['base_mask'],
                'moving_mask': pose_result['moving_mask']
            }
            print(f"✓ Base pose (unchanged): {pa_pose_current[:3, 3]}")
            print(f"✓ Moving pose (lifted): {pb_pose_current[:3, 3]}")
            
            # Extract point clouds
            print("Extracting point clouds...")
            pcd_result = self.extract_point_clouds(obs, pose_result_manip)
            print(f"✓ PA points: {pcd_result['pa_points'].shape}")
            print(f"✓ PB points: {pcd_result['pb_points'].shape}")
            
            # Predict manipulation trajectory
            print("Predicting manipulation trajectory...")
            manip_traj = self.predict_trajectory(obs, pose_result_manip, pcd_result, mode='manip')
            print(f"✓ Predicted {len(manip_traj)} waypoints")
            
            # Visualize if requested
            if visualize:
                print(f"  Manip traj: {manip_traj}")
                self.visualize_prediction(obs, pose_result_manip, pcd_result, manip_traj, mode='manip', save_image=True)
            
            # Execute manipulation trajectory
            print("Executing manipulation trajectory...")
            result = robot_interface.execute_trajectory(manip_traj, mode='manip', blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Manip execution failed: {result.get('message', 'Unknown error')}")
            print("✓ Manipulation trajectory executed")
            time.sleep(2.0)
            
            # Open gripper
            if self.task_name in ["open_drawer", "insert_tube", "scale_grape"]:
                print("Opening gripper...")
                result = robot_interface.open_gripper(blocking=True)
                if result['status'] != 'success':
                    raise RuntimeError(f"Gripper open failed: {result.get('message', 'Unknown error')}")
                print("✓ Gripper opened")
                time.sleep(2.0)
            
            # Reset robot to home position
            print("\nResetting robot to home position...")
            result = robot_interface.reset_robot(blocking=True)
            if result['status'] != 'success':
                print(f"⚠ Reset failed: {result.get('message', 'Unknown error')}")
            else:
            
                print("✓ Robot reset to home position")
            
            # Open gripper
            print("Opening gripper...")
            result = robot_interface.open_gripper(blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Gripper open failed: {result.get('message', 'Unknown error')}")
            print("✓ Gripper opened")
            time.sleep(2.0)

            print("\n" + "="*70)
            print("✓ TASK COMPLETED SUCCESSFULLY")
            print("="*70)
            
            return {'status': 'success', 'message': 'Task completed'}
            
        except Exception as e:
            print("\n" + "="*70)
            print(f"✗ TASK FAILED: {e}")
            print("="*70)
            import traceback
            traceback.print_exc()
            
            # Recovery: open gripper and reset robot
            print("\n=== Starting Error Recovery ===")
            try:
                # Give MoveIt time to cleanup
                print("Waiting for MoveIt cleanup...")
                time.sleep(2.0)
                
                # Open gripper first
                print("Opening gripper...")
                gripper_result = robot_interface.open_gripper(blocking=True)
                if gripper_result['status'] == 'success':
                    print("✓ Gripper opened")
                time.sleep(1.0)
                
                # Reset robot with retries
                print("Resetting robot to home position...")
                max_retries = 2
                reset_success = False
                
                for attempt in range(max_retries):
                    if attempt > 0:
                        print(f"Retry attempt {attempt + 1}/{max_retries}...")
                        time.sleep(2.0)
                    
                    result = robot_interface.reset_robot(blocking=True)
                    if result['status'] == 'success':
                        reset_success = True
                        print("✓ Robot reset to home position")
                        break
                    else:
                        print(f"⚠ Reset attempt {attempt + 1} failed: {result.get('message', 'Unknown error')}")
                
                if not reset_success:
                    print("⚠ All reset attempts failed - manual intervention may be needed")
                else:
                    print("✓ Recovery complete - robot ready for next task")
                    
            except Exception as reset_error:
                print(f"✗ Recovery error: {reset_error}")
            
            return {'status': 'failed', 'message': str(e)}
        
        finally:
            print("\n=== Cleaning up GPU memory ===")
            self._cleanup_actors()
            if hasattr(self, 'language_encoder'):
                del self.language_encoder
                self.language_encoder = None
            torch.cuda.empty_cache()
            print("✓ GPU memory cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description='Execute FOCI policy on real robot via ZMQ'
    )
    parser.add_argument('--task', type=str, required=True,
                       help='Task name from task_config.yaml')
    parser.add_argument('--config_dir', type=str, 
                       default='../configs',
                       help='Configuration directory')
    # TODO: unify checkpoint dirs
    parser.add_argument('--checkpoint_dir', type=str,
                       default='../../foci_policy/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--assets_dir', type=str,
                       default='../assets',
                       help='Assets directory (meshes)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='torch device')
    parser.add_argument('--zmq_port', type=int, default=5555,
                       help='ZMQ port for robot communication')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions before execution')
    parser.add_argument('--voxel_size', type=float, default=0.004,
                       help='Voxel size for downsampling')
    parser.add_argument('--num_points', type=int, default=2048,
                       help='Number of points per object')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    config_dir = script_dir / args.config_dir
    checkpoint_dir = script_dir / args.checkpoint_dir
    assets_dir = script_dir / args.assets_dir
    
    # Initialize controller
    controller = FOCIPolicyRobotController(
        task_name=args.task,
        config_dir=config_dir,
        checkpoint_dir=checkpoint_dir,
        assets_dir=assets_dir,
        device=args.device,
        voxel_size=args.voxel_size,
        num_points=args.num_points
    )
    
    # Initialize robot interface
    print("\nConnecting to robot via ZMQ...")
    robot = ZMQRobotInterface(request_port=args.zmq_port)
    print("✓ Connected to robot\n")
    
    try:
        # Reset robot to home position before starting
        print("Resetting robot to home position...")
        reset_result = robot.reset_robot(blocking=True)
        if reset_result['status'] != 'success':
            print(f"⚠ Reset failed: {reset_result.get('message', 'Unknown error')}")
            print("Continuing anyway...")
        else:
            print("✓ Robot reset to home position")
        
        # Open gripper
        print("Opening gripper...")
        gripper_result = robot.open_gripper(blocking=True)
        if gripper_result['status'] == 'success':
            print("✓ Gripper opened")
        
        # Wait for stabilization
        print("Waiting for robot to stabilize...")
        time.sleep(2.0)
        print("✓ Ready to start\n")
        
        # Execute task
        result = controller.execute_task(robot, visualize=args.visualize)
        
        # Print result
        if result['status'] == 'success':
            print("\n✓ Execution successful!")
            return 0
        else:
            print(f"\n✗ Execution failed: {result['message']}")
            return 1
    
    finally:
        # Cleanup
        print("\n=== Final cleanup ===")
        if 'controller' in locals():
            controller._cleanup_actors()
        torch.cuda.empty_cache()
        print("✓ All GPU resources released")
        robot.close()
        print("\nZMQ connection closed")


if __name__ == '__main__':
    exit(main())
