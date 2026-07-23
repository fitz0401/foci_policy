#!/usr/bin/env python3
"""
Baseline Replay Method: ICP Registration + Trajectory Replay
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from foci_real_world.pose_estimator.mask_generator import MaskGenerator
from foci_real_world.scripts.zmq_robot_interface import ZMQRobotInterface
from foci_real_world.utils.preprocess_replay_data import get_mask_from_generator

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

TABLE_HEIGHT = 0.055  # meters


def extract_masked_point_cloud(rgb, depth, mask, K, cam_extrinsic):
    """Extract masked point cloud from RGB-D"""
    # Preprocess depth
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0  # Convert to meters
    
    # Resize mask if needed
    depth_h, depth_w = depth.shape
    mask_h, mask_w = mask.shape
    if mask_h != depth_h or mask_w != depth_w:
        mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
    
    # Get intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel grid
    v, u = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
    
    # Valid pixels
    valid_depth = (depth > 0.01) & (depth < 3.0)
    valid_mask = mask.astype(bool) & valid_depth
    
    v_valid = v[valid_mask]
    u_valid = u[valid_mask]
    z_valid = depth[valid_mask]
    
    # Back-project to 3D
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid
    points_cam = np.stack([x, y, z], axis=1)
    
    # Transform to world coordinates
    points_world = (cam_extrinsic[:3, :3] @ points_cam.T).T + cam_extrinsic[:3, 3]
    
    # Get colors
    colors = rgb[valid_mask].astype(np.float32) / 255.0
    return points_world, colors


def icp_registration(source_points, target_points, initial_transform=np.eye(4), 
                     voxel_size=0.005):
    """
    Perform two-stage ICP registration from source to target.
    """
    # Create point clouds if needed
    if not isinstance(source_points, o3d.geometry.PointCloud):
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
    else:
        source_pcd = source_points
    
    if not isinstance(target_points, o3d.geometry.PointCloud):
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
    else:
        target_pcd = target_points
    
    # Downsample for efficiency
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    
    threshold = voxel_size * 4  # Distance threshold
    
    # First: point-to-point ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    # Then: point-to-plane ICP refinement
    reg_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, reg_p2p.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    print(f"    Refined ICP Fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.4f}")
    
    return reg_result.transformation


class ReplayController:
    """Replay controller using ICP registration"""
    
    def __init__(self, task_name, demo_dir, dataset_root='../dataset'):
        """
        Initialize replay controller
        
        Args:
            task_name: Task name
            demo_dir: Demo directory name
            dataset_root: Dataset root directory
        """
        self.task_name = task_name
        self.dataset_root = Path(dataset_root)
        self.demo_path = self.dataset_root / demo_dir
        
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo directory not found: {self.demo_path}")
        
        print(f"Initializing Replay Controller")
        print(f"  Task: {task_name}")
        print(f"  Demo: {demo_dir}")
        
        # Load replay data
        replay_data_path = self.demo_path / 'replay_data.pkl'
        if not replay_data_path.exists():
            raise FileNotFoundError(
                f"Replay data not found: {replay_data_path}\n"
                f"Please run preprocess_replay_data.py first"
            )
        
        with open(replay_data_path, 'rb') as f:
            self.replay_data = pickle.load(f)
        
        self.base_object = self.replay_data['base_object']
        self.moving_object = self.replay_data['moving_object']
        
        print(f"  Base object: {self.base_object}")
        print(f"  Moving object: {self.moving_object}")
        
        # Initialize mask generator
        print("Initializing MaskGenerator...")
        self.mask_generator = MaskGenerator()
        
        print("✓ Initialization complete\n")
    
    def segment_objects(self, obs):
        """Segment objects from current observation"""
        rgb = obs['rgb']
        depth = obs['depth']
        K = obs['K']
        cam_extrinsic = obs['cam_extrinsic']
        
        print("Segmenting objects...")
        
        # Generate masks
        base_mask = get_mask_from_generator(self.mask_generator, rgb, self.base_object)
        moving_mask = get_mask_from_generator(self.mask_generator, rgb, self.moving_object)
        
        if base_mask is None:
            raise RuntimeError(f"Failed to generate mask for {self.base_object}")
        if moving_mask is None:
            raise RuntimeError(f"Failed to generate mask for {self.moving_object}")
        
        # Extract point clouds
        pa_points, pa_colors = extract_masked_point_cloud(
            rgb, depth, base_mask, K, cam_extrinsic
        )
        pb_points, pb_colors = extract_masked_point_cloud(
            rgb, depth, moving_mask, K, cam_extrinsic
        )
        
        print(f"  Base object points: {len(pa_points)}")
        print(f"  Moving object points: {len(pb_points)}")
        
        return {
            'pa_points': pa_points,
            'pa_colors': pa_colors,
            'pb_points': pb_points,
            'pb_colors': pb_colors,
        }
    
    def compute_transformation(self, current_pcd):
        """
        Compute transformation from demo initial frame to current frame using two-stage ICP.
        Returns T_base, T_moving such that T @ demo_points ≈ current_points
        """
        print("Computing ICP transformation...")
        
        # Get demo initial point clouds
        pa_init = self.replay_data['pa_points_init']
        pb_init = self.replay_data['pb_points_init']
        
        # Current point clouds
        pa_curr = current_pcd['pa_points']
        pb_curr = current_pcd['pb_points']
        
        # Perform ICP on base object
        print("  ICP on base object...")
        T_base = icp_registration(
            pa_init, pa_curr, 
            initial_transform=np.eye(4),
            voxel_size=0.005
        )
        
        # Perform ICP on moving object
        print("  ICP on moving object...")
        T_moving = icp_registration(
            pb_init, pb_curr,
            initial_transform=np.eye(4),
            voxel_size=0.005
        )
        return T_base, T_moving
    
    def generate_trajectory(self, T_base, T_moving, mode='grasp'):
        """
        Generate trajectory by applying appropriate transformation to demo gripper poses.
        
        Args:
            T_base: (4, 4) base object transformation matrix
            T_moving: (4, 4) moving object transformation matrix  
            mode: 'grasp' or 'manip'
        """
        if mode == 'grasp':
            start_pose = self.replay_data['grasp_start_pose']
            end_pose = self.replay_data['grasp_end_pose']
            transformation = T_moving
            print(f"  Using moving object transformation for {mode}")
        elif mode == 'manip':
            start_pose = self.replay_data['manip_start_pose']
            end_pose = self.replay_data['manip_end_pose']
            transformation = T_base  
            print(f"  Using base object transformation for {mode}")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Apply transformation to poses
        new_start = transformation @ start_pose
        new_end = transformation @ end_pose
        
        # Enforce minimum table height
        new_start[2, 3] = max(new_start[2, 3], TABLE_HEIGHT)
        new_end[2, 3] = max(new_end[2, 3], TABLE_HEIGHT)
        
        trajectory = [new_start, new_end]
        
        print(f"  Generated {len(trajectory)} waypoints for {mode}")
        return trajectory
    
    @staticmethod
    def draw_pose_axis(ax, pose, length=0.08, label=None):
        """Draw coordinate frame for a pose"""
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
            if label is not None and i == 0:
                ax.text(origin[0], origin[1], origin[2], label, fontsize=8)
    
    def visualize_prediction(self, current_pcd, transformation, trajectory, mode='grasp', save_image=True):
        """Visualize ICP registration result and predicted trajectory"""
        if not save_image:
            return
        
        script_dir = Path(__file__).parent
        debug_dir = script_dir / 'debug' / f'replay_{self.task_name}'
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = debug_dir / f'replay_prediction_{mode}.png'
        
        try:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get demo initial point clouds
            pa_init = self.replay_data['pa_points_init']
            pb_init = self.replay_data['pb_points_init']
            
            # Current point clouds with colors
            pa_curr = current_pcd['pa_points']
            pb_curr = current_pcd['pb_points']
            pa_colors = current_pcd['pa_colors']
            pb_colors = current_pcd['pb_colors']
            
            # Transform demo point clouds using ICP result
            pa_init_transformed = (transformation[:3, :3] @ pa_init.T).T + transformation[:3, 3]
            pb_init_transformed = (transformation[:3, :3] @ pb_init.T).T + transformation[:3, 3]
            
            # Downsample for visualization
            def downsample_for_vis(points, colors=None, max_points=2000):
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    if colors is not None:
                        return points[indices], colors[indices]
                    return points[indices], None
                return points, colors
            
            pa_curr_vis, pa_colors_vis = downsample_for_vis(pa_curr, pa_colors)
            pb_curr_vis, pb_colors_vis = downsample_for_vis(pb_curr, pb_colors)
            pa_init_vis, _ = downsample_for_vis(pa_init_transformed)
            pb_init_vis, _ = downsample_for_vis(pb_init_transformed)
            
            # Plot current scene with RGB colors
            ax.scatter(pa_curr_vis[:, 0], pa_curr_vis[:, 1], pa_curr_vis[:, 2],
                      c=pa_colors_vis, s=3, alpha=0.6, label='Base object (current)')
            ax.scatter(pb_curr_vis[:, 0], pb_curr_vis[:, 1], pb_curr_vis[:, 2],
                      c=pb_colors_vis, s=3, alpha=0.6, label='Moving object (current)')
            
            # Plot demo point clouds after ICP transformation (single colors)
            ax.scatter(pa_init_vis[:, 0], pa_init_vis[:, 1], pa_init_vis[:, 2],
                      c='cyan', s=2, alpha=0.4, marker='^', label='Base object (demo aligned)')
            ax.scatter(pb_init_vis[:, 0], pb_init_vis[:, 1], pb_init_vis[:, 2],
                      c='yellow', s=2, alpha=0.4, marker='^', label='Moving object (demo aligned)')
            
            # Plot trajectory
            if len(trajectory) > 1:
                traj_positions = np.array([pose[:3, 3] for pose in trajectory])
                ax.plot(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2],
                       'r-', linewidth=3, alpha=0.8, label='Predicted trajectory')
                ax.scatter(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2],
                          c='red', s=50, alpha=0.9, marker='o')
            
            # Draw trajectory poses
            for i, pose in enumerate(trajectory):
                self.draw_pose_axis(ax, pose, length=0.04)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Replay Method - ICP Registration + Trajectory ({mode.upper()} mode)')
            ax.legend(loc='upper left', fontsize=8)
            
            # Set equal aspect ratio for all axes
            all_points = np.vstack([pa_curr_vis, pb_curr_vis, pa_init_vis, pb_init_vis])
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
            
            # Calculate the max range to make axes equal
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            z_center = (z_max + z_min) / 2
            
            # Set equal ranges for all axes
            margin = 0.1
            plot_range = max_range / 2 + margin
            ax.set_xlim([x_center - plot_range, x_center + plot_range])
            ax.set_ylim([y_center - plot_range, y_center + plot_range])
            ax.set_zlim([max(0, z_center - plot_range), z_center + plot_range])
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Visualization saved to {save_path}")
            
            # Save data for offline visualization
            vis_data = {
                'pa_curr': pa_curr,
                'pb_curr': pb_curr,
                'pa_init': pa_init,
                'pb_init': pb_init,
                'pa_init_transformed': pa_init_transformed,
                'pb_init_transformed': pb_init_transformed,
                'transformation': transformation,
                'trajectory': np.array([pose for pose in trajectory]),
                'mode': mode
            }
            data_path = debug_dir / f'replay_vis_data_{mode}.npz'
            np.savez(data_path, **vis_data)
            print(f"✓ Visualization data saved to {data_path}")
            
        except Exception as e:
            print(f"⚠ Visualization save failed: {e}")
            import traceback
            traceback.print_exc()
    
    def execute_task(self, robot_interface):
        """Execute full task using replay method"""
        print("="*70)
        print(f"Executing task: {self.task_name} (Replay Method)")
        print("="*70)
        
        try:
            # Get current observation
            print("\nGetting observation...")
            obs = robot_interface.get_observation()
            print("✓ Observation received")
            
            # Segment objects
            current_pcd = self.segment_objects(obs)
            
            T_base, T_moving = self.compute_transformation(current_pcd)
            
            # ========== GRASP PHASE ==========
            print("\n[1/2] GRASP PHASE")
            print("-"*70)
            
            # Generate grasp trajectory
            print("Generating grasp trajectory...")
            grasp_traj = self.generate_trajectory(T_base, T_moving, mode='grasp')
            
            # Visualize grasp prediction (use moving object transformation for visualization)
            print("Saving grasp visualization...")
            self.visualize_prediction(current_pcd, T_moving, grasp_traj, mode='grasp', save_image=True)
            
            # Execute grasp trajectory
            print("Executing grasp trajectory...")
            result = robot_interface.execute_trajectory(grasp_traj, mode='grasp', blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Grasp execution failed: {result.get('message', '')}")
            print("✓ Grasp trajectory executed")
            time.sleep(1.0)
            
            # Close gripper
            print("Closing gripper...")
            result = robot_interface.close_gripper(blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Gripper close failed: {result.get('message', '')}")
            print("✓ Gripper closed")
            time.sleep(2.0)
            
            # Lift object
            print("Lifting object...")
            lift_height = 0.0
            if self.task_name not in ["open_drawer"]:
                lift_height = 0.2
                lift_pose = grasp_traj[-1].copy()
                lift_pose[2, 3] += lift_height
                result = robot_interface.execute_trajectory([lift_pose], mode='grasp', blocking=True)
                if result['status'] != 'success':
                    raise RuntimeError(f"Lift failed: {result.get('message', '')}")
                print("✓ Object lifted")
                time.sleep(1.0)
            
            # ========== MANIPULATION PHASE ==========
            print("\n[2/2] MANIPULATION PHASE")
            print("-"*70)
            
            # Generate manipulation trajectory
            print("Generating manipulation trajectory...")
            manip_traj = self.generate_trajectory(T_base, T_moving, mode='manip')
            
            # Visualize manipulation prediction (use base object transformation for visualization)
            print("Saving manipulation visualization...")
            self.visualize_prediction(current_pcd, T_base, manip_traj, mode='manip', save_image=True)
            
            # Execute manipulation trajectory
            print("Executing manipulation trajectory...")
            result = robot_interface.execute_trajectory(manip_traj, mode='manip', blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Manip execution failed: {result.get('message', '')}")
            print("✓ Manipulation trajectory executed")
            time.sleep(2.0)
            
            # Open gripper
            print("Opening gripper...")
            result = robot_interface.open_gripper(blocking=True)
            if result['status'] != 'success':
                raise RuntimeError(f"Gripper open failed: {result.get('message', '')}")
            print("✓ Gripper opened")
            time.sleep(2.0)
            
            # Reset robot
            print("\nResetting robot to home position...")
            result = robot_interface.reset_robot(blocking=True)
            if result['status'] != 'success':
                print(f"⚠ Reset failed: {result.get('message', '')}")
            else:
                print("✓ Robot reset to home position")
            
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
            
            # Recovery
            print("\n=== Starting Error Recovery ===")
            try:
                time.sleep(2.0)
                print("Opening gripper...")
                robot_interface.open_gripper(blocking=True)
                time.sleep(1.0)
                
                print("Resetting robot...")
                result = robot_interface.reset_robot(blocking=True)
                if result['status'] == 'success':
                    print("✓ Recovery complete")
            except Exception as reset_error:
                print(f"✗ Recovery error: {reset_error}")
            
            return {'status': 'failed', 'message': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Execute replay baseline on real robot via ZMQ'
    )
    parser.add_argument('--task', type=str, required=True,
                       help='Task name')
    parser.add_argument('--demo', type=str, required=True,
                       help='Demo directory name')
    parser.add_argument('--dataset_root', type=str, default='../dataset',
                       help='Dataset root directory')
    parser.add_argument('--zmq_port', type=int, default=5555,
                       help='ZMQ port for robot communication')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    dataset_root = script_dir / args.dataset_root
    
    # Initialize controller
    controller = ReplayController(
        task_name=args.task,
        demo_dir=args.demo,
        dataset_root=dataset_root
    )
    
    # Initialize robot interface
    print("Connecting to robot via ZMQ...")
    robot = ZMQRobotInterface(request_port=args.zmq_port)
    print("✓ Connected to robot\n")
    
    try:
        # Reset robot
        print("Resetting robot to home position...")
        reset_result = robot.reset_robot(blocking=True)
        if reset_result['status'] != 'success':
            print(f"⚠ Reset failed, continuing anyway...")
        else:
            print("✓ Robot reset to home position")
        
        # Open gripper
        print("Opening gripper...")
        robot.open_gripper(blocking=True)
        print("✓ Gripper opened")
        
        # Wait for stabilization
        print("Waiting for robot to stabilize...")
        time.sleep(2.0)
        print("✓ Ready to start\n")
        
        # Execute task
        result = controller.execute_task(robot)
        
        # Print result
        if result['status'] == 'success':
            print("\n✓ Execution successful!")
            return 0
        else:
            print(f"\n✗ Execution failed: {result['message']}")
            return 1
    
    finally:
        robot.close()
        print("\nZMQ connection closed")


if __name__ == '__main__':
    exit(main())
