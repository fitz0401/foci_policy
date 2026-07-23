#!/usr/bin/env python3
"""
Preprocess real-world demo data for FOCI training.
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import open3d as o3d
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from foci_real_world.pose_estimator.fp_real_world_utils import (
    extract_poses_from_demo, load_demo_data, postprocess_object_trajectory
)
from foci_real_world.pose_estimator.mask_generator import MaskGenerator


def plot_coordinate_frame(ax, pose_mat, scale=0.08, linewidth=2):
    """Plot coordinate frame axes in matplotlib."""
    origin = pose_mat[:3, 3]
    x_axis = pose_mat[:3, 0] * scale
    y_axis = pose_mat[:3, 1] * scale
    z_axis = pose_mat[:3, 2] * scale
    
    # Plot X (red), Y (green), Z (blue)
    ax.plot([origin[0], origin[0] + x_axis[0]], 
            [origin[1], origin[1] + x_axis[1]], 
            [origin[2], origin[2] + x_axis[2]], 'r-', linewidth=linewidth)
    ax.plot([origin[0], origin[0] + y_axis[0]], 
            [origin[1], origin[1] + y_axis[1]], 
            [origin[2], origin[2] + y_axis[2]], 'g-', linewidth=linewidth)
    ax.plot([origin[0], origin[0] + z_axis[0]], 
            [origin[1], origin[1] + z_axis[1]], 
            [origin[2], origin[2] + z_axis[2]], 'b-', linewidth=linewidth)


# Lower logging verbosity
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def to_array(tensor):
    """ Convert tensor to array """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None, grey=False, red=False, orange=False, normals=None):
    """ Convert tensor/array to open3d PointCloud, xyz: [N, 3] """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        normals = to_array(normals)
        # pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
        pcd.normals = o3d.utility.Vector3dVector(to_array(normals))
    if colors is not None:
        # pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
        pcd.colors = o3d.utility.Vector3dVector(to_array(colors))
    if grey:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pts)+0.1)
    if red:
        c = np.zeros_like(pts)
        c[:,0] = 1.0
        pcd.colors = o3d.utility.Vector3dVector(c)
    if orange:
        c = np.zeros_like(pts)
        c[:,:] = np.asarray([255, 165, 0])/255
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd


def load_task_info(demo_dir):
    """Load task information from JSON file"""
    task_info_path = os.path.join(demo_dir, 'task_info.json')
    if not os.path.exists(task_info_path):
        raise FileNotFoundError(f"task_info.json not found in {demo_dir}")
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)
    return task_info


def load_trajectory(demo_dir):
    """Load gripper trajectory from JSON file"""
    traj_path = os.path.join(demo_dir, 'trajectory.json')
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"trajectory.json not found in {demo_dir}")
    with open(traj_path, 'r') as f:
        trajectory = json.load(f)
    return trajectory


def load_camera_intrinsics(demo_dir):
    """Load camera intrinsics from JSON file"""
    intrinsics_path = os.path.join(demo_dir, 'camera_intrinsics.json')
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"camera_intrinsics.json not found in {demo_dir}")
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def load_camera_extrinsics(demo_dir):
    """Load camera extrinsics from JSON file"""
    extrinsics_path = os.path.join(demo_dir, 'camera_extrinsics.json')
    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"camera_extrinsics.json not found in {demo_dir}")
    with open(extrinsics_path, 'r') as f:
        extrinsics = json.load(f)
    tx = extrinsics['translation']['x']
    ty = extrinsics['translation']['y']
    tz = extrinsics['translation']['z']
    qx = extrinsics['rotation']['x']
    qy = extrinsics['rotation']['y']
    qz = extrinsics['rotation']['z']
    qw = extrinsics['rotation']['w']
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    translation = np.array([tx, ty, tz]).reshape((3, 1))
    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = rotation
    extrinsic_mat[:3, 3:] = translation
    return extrinsic_mat


def extract_point_cloud_from_rgbd(rgb_path, depth_path, intrinsics, depth_scale=1000.0):
    """ Extract point cloud from RGB-D images. """
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    # Create Open3D images
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth)
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    # Create camera intrinsics
    fx = intrinsics['K'][0]
    fy = intrinsics['K'][4]
    cx = intrinsics['K'][2]
    cy = intrinsics['K'][5]
    width = intrinsics['width']
    height = intrinsics['height']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic
    )
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def extract_masked_point_cloud(rgb_path, depth_path, mask, intrinsics, depth_scale=1000.0):
    """ Extract masked point cloud from RGB-D images. """
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale  # Convert to meters
    
    # Resize mask to match depth dimensions if needed
    depth_h, depth_w = depth.shape
    mask_h, mask_w = mask.shape
    if mask_h != depth_h or mask_w != depth_w:
        mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
    
    # Get camera intrinsics
    fx = intrinsics['K'][0]
    fy = intrinsics['K'][4]
    cx = intrinsics['K'][2]
    cy = intrinsics['K'][5]
    
    # Create pixel coordinate grid
    v, u = np.meshgrid(np.arange(depth_h), np.arange(depth_w), indexing='ij')
    
    # Apply mask and depth validity
    valid_depth = (depth > 0.01) & (depth < 3.0)  # Valid depth range
    valid_mask = mask.astype(bool) & valid_depth
    
    # Get valid pixels
    v_valid = v[valid_mask]
    u_valid = u[valid_mask]
    z_valid = depth[valid_mask]
    
    # Back-project to 3D
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid
    points = np.stack([x, y, z], axis=1)
    
    # Get colors for valid pixels
    colors = rgb[valid_mask].astype(np.float32) / 255.0
    return points, colors


def parse_gripper_pose(gripper_dict):
    """ Parse gripper pose from trajectory dict. """
    pos_dict = gripper_dict['gripper_pose']['position']
    position = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']])
    ori_dict = gripper_dict['gripper_pose']['orientation']
    orientation = np.array([ori_dict['x'], ori_dict['y'], ori_dict['z'], ori_dict['w']])  # quaternion [x, y, z, w]
    # Normalize quaternion
    orientation = orientation / np.linalg.norm(orientation)
    # Convert to 4x4 transformation matrix
    rotation_matrix = Rotation.from_quat(orientation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    return transform_matrix


def save_data_from_demo(demo_dir, save_dir, mask_generator=None, 
                        voxel_size=0.004, num_points=2048, 
                        disp=False, verbose=False):
    """ Process a single real-world demo and save preprocessed data. """
    demo_dir = Path(demo_dir)
    
    # Load task info
    task_info = load_task_info(demo_dir)
    task_name = task_info['task_name']
    moving_object = task_info['moving_object']
    base_object = task_info['base_object']
    grasp_interval = task_info['grasp_interval']
    manip_interval = task_info['manip_interval']

    print(f"Processing demo: {demo_dir.name}")
    print(f"  Task: {task_name}")
    print(f"  Grasp interval: {grasp_interval}")
    print(f"  Manip interval: {manip_interval}")
    
    # Load trajectory
    trajectory = load_trajectory(demo_dir)
    total_frames = len(trajectory)
    
    # Load camera intrinsics
    intrinsics = load_camera_intrinsics(demo_dir)
    extrinsics_mat = load_camera_extrinsics(demo_dir)
    
    # Initialize mask generator if not provided
    if mask_generator is None:
        print("Initializing MaskGenerator...")
        mask_generator = MaskGenerator()
    
    # Extract object poses using Foundation Pose
    print("Extracting object poses with Foundation Pose...")
    
    # Find mesh files
    assets_dir = demo_dir.parent.parent / 'assets'
    moving_mesh = assets_dir / f"{moving_object}.obj"
    base_mesh = assets_dir / f"{base_object}.obj"
    
    if not moving_mesh.exists():
        raise FileNotFoundError(f"Moving object mesh not found: {moving_mesh}")
    if not base_mesh.exists():
        raise FileNotFoundError(f"Base object mesh not found: {base_mesh}")
    
    # Extract poses for moving object
    print(f"  Extracting poses for moving object ({moving_object})...")
    moving_poses_data = extract_poses_from_demo(
        demo_dir=str(demo_dir),
        object_name=moving_object,
        mesh_path=str(moving_mesh),
        mask_generator=mask_generator,
        verbose=verbose,
        visualize_mask=False
    )
    if not moving_poses_data['success']:
        raise RuntimeError(f"Failed to extract poses for moving object")
    
    # Extract poses for base object
    print(f"  Extracting poses for base object ({base_object})...")
    base_poses_data = extract_poses_from_demo(
        demo_dir=str(demo_dir),
        object_name=base_object,
        mesh_path=str(base_mesh),
        mask_generator=mask_generator,
        verbose=verbose,
        visualize_mask=False
    )
    if not base_poses_data['success']:
        raise RuntimeError(f"Failed to extract poses for base object")
    print(f"  Successfully extracted poses for {len(moving_poses_data['poses'])} frames")
    
    # Apply postprocessing to improve pose quality
    print(f"  Applying trajectory postprocessing...")
    postprocessed = postprocess_object_trajectory(
        moving_poses_dict=moving_poses_data['poses'],
        base_poses_dict=base_poses_data['poses'],
        trajectory=trajectory,
        grasp_interval=grasp_interval,
        manip_interval=manip_interval,
        fix_base=True,
        verbose=verbose
    )
    # Update with postprocessed poses
    moving_poses_data['poses'] = postprocessed['moving_poses']
    base_poses_data['poses'] = postprocessed['base_poses']
    print(f"  ✓ Postprocessing complete")
    
    # Process each frame
    frames_data = []
    prev_pa_points = None
    prev_pa_colors = None
    prev_pb_points = None
    prev_pb_colors = None
    
    for t in tqdm(range(total_frames), desc="Processing frames"):
        frame_key = f"frame_{t:04d}"
        
        # Paths to RGB and depth
        rgb_path = demo_dir / 'color' / f'{t:04d}.png'
        depth_path = demo_dir / 'depth' / f'{t:04d}.png'
        
        if not rgb_path.exists() or not depth_path.exists():
            print(f"Warning: Missing RGB or depth for frame {t}, skipping")
            continue
        
        # Get gripper pose
        gripper_pose_mat = parse_gripper_pose(trajectory[t])
        
        # Get object poses
        pa_pose_mat = base_poses_data['poses'].get(frame_key)
        pb_pose_mat = moving_poses_data['poses'].get(frame_key)
        
        # Get masks using XMem++
        pa_mask = base_poses_data['masks'].get(frame_key)
        pb_mask = moving_poses_data['masks'].get(frame_key)
        
        # Extract masked point clouds
        if pa_mask is not None:
            pa_points_raw, pa_colors_raw = extract_masked_point_cloud(
                str(rgb_path), str(depth_path), pa_mask, intrinsics
            )
        else:
            raise RuntimeError(f"No mask for base object at frame {t}")
        
        if pb_mask is not None:
            pb_points_raw, pb_colors_raw = extract_masked_point_cloud(
                str(rgb_path), str(depth_path), pb_mask, intrinsics
            )
        else:
            raise RuntimeError(f"No mask for moving object at frame {t}")
        
        # Transform point clouds to world coordinates
        if len(pa_points_raw) > 0:
            pa_points_raw = (extrinsics_mat[:3, :3] @ pa_points_raw.T).T + extrinsics_mat[:3, 3]
        if len(pb_points_raw) > 0:
            pb_points_raw = (extrinsics_mat[:3, :3] @ pb_points_raw.T).T + extrinsics_mat[:3, 3]
        
        # Voxel downsampling
        if len(pa_points_raw) > 0:
            pa_pcd = to_o3d_pcd(pa_points_raw, pa_colors_raw)
            pa_pcd = pa_pcd.voxel_down_sample(voxel_size=voxel_size)
            
            if len(pa_pcd.points) > num_points:
                indices = np.random.choice(len(pa_pcd.points), num_points, replace=False)
            else:
                indices = np.random.choice(len(pa_pcd.points), num_points, replace=True)
            frame_pa_points = np.asarray(pa_pcd.points)[indices]
            frame_pa_colors = np.asarray(pa_pcd.colors)[indices]
            prev_pa_points = frame_pa_points
            prev_pa_colors = frame_pa_colors
        else:
            if prev_pa_points is not None:
                frame_pa_points = prev_pa_points
                frame_pa_colors = prev_pa_colors
            else:
                raise RuntimeError("No points for base object and no previous points to fallback")
        
        if len(pb_points_raw) > 0:
            pb_pcd = to_o3d_pcd(pb_points_raw, pb_colors_raw)
            pb_pcd = pb_pcd.voxel_down_sample(voxel_size=voxel_size)
            
            if len(pb_pcd.points) > num_points:
                indices = np.random.choice(len(pb_pcd.points), num_points, replace=False)
            else:
                indices = np.random.choice(len(pb_pcd.points), num_points, replace=True)
            frame_pb_points = np.asarray(pb_pcd.points)[indices]
            frame_pb_colors = np.asarray(pb_pcd.colors)[indices]
            prev_pb_points = frame_pb_points
            prev_pb_colors = frame_pb_colors
        else:
            if prev_pb_points is not None:
                frame_pb_points = prev_pb_points
                frame_pb_colors = prev_pb_colors
            else:
                raise RuntimeError("No points for moving object and no previous points to fallback")
        
        frames_data.append({
            'timestep': t,
            'gripper_pose_mat': gripper_pose_mat,
            'pa_pose_mat': pa_pose_mat,
            'pb_pose_mat': pb_pose_mat,
            'pa_points': frame_pa_points,
            'pa_colors': frame_pa_colors,
            'pb_points': frame_pb_points,
            'pb_colors': frame_pb_colors,
        })
    
    # Determine keyframes from intervals
    max_frame_idx = len(frames_data) - 1
    pick_frame = min(grasp_interval[1], max_frame_idx)
    place_frame = min(manip_interval[1], max_frame_idx)
    
    keyframes = {
        'pick': pick_frame,
        'place': place_frame
    }
    
    data_dic = {
        'task_name': task_name,
        'total_frames': total_frames,
        'frames_data': frames_data,
        'key_times': keyframes,
        'lan_pick': task_info['lang_grasp'],
        'lan_place': task_info['lang_manip'],
        'grasp_interval': grasp_interval[1] - grasp_interval[0],
        'manip_interval': manip_interval[1] - manip_interval[0]
    }
    
    # Save to pickle file
    save_folder = Path(save_dir) / task_name
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Count existing files to determine filename
    num_pkl_files = len(list(save_folder.glob('*.pkl')))
    fname = f'{num_pkl_files}.pkl'
    
    save_path = save_folder / fname
    with open(save_path, 'wb') as f:
        pickle.dump(data_dic, f)
    
    print(f"  Saved to: {save_path}")
    
    # Visualization if requested
    if disp:
        print("Visualizing intervals...")
        
        # Visualize grasp interval
        print(f"Visualizing grasp interval: [{grasp_interval[0]}, {grasp_interval[1]}]")
        geometries_grasp = []
        
        # World frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        geometries_grasp.append(world_frame)
        
        # Grasp interval: gripper poses (start and end)
        grasp_start_idx = grasp_interval[0]
        grasp_end_idx = grasp_interval[1]
        
        if grasp_start_idx < len(frames_data) and grasp_end_idx < len(frames_data):
            # Gripper start pose
            gripper_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            gripper_start.transform(frames_data[grasp_start_idx]['gripper_pose_mat'])
            gripper_start.paint_uniform_color([1, 0, 0])  # Red for start
            geometries_grasp.append(gripper_start)
            
            # Gripper end pose
            gripper_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            gripper_end.transform(frames_data[grasp_end_idx]['gripper_pose_mat'])
            gripper_end.paint_uniform_color([0, 1, 0])  # Green for end
            geometries_grasp.append(gripper_end)
            
            # Arrow connecting start to end
            start_pos = frames_data[grasp_start_idx]['gripper_pose_mat'][:3, 3]
            end_pos = frames_data[grasp_end_idx]['gripper_pose_mat'][:3, 3]
            arrow_vec = end_pos - start_pos
            arrow_length = np.linalg.norm(arrow_vec)
            if arrow_length > 1e-6:
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.005,
                    cone_radius=0.01,
                    cylinder_height=arrow_length * 0.8,
                    cone_height=arrow_length * 0.2
                )
                arrow_dir = arrow_vec / arrow_length
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, arrow_dir)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                if rotation_axis_norm > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    rotation_angle = np.arccos(np.clip(np.dot(z_axis, arrow_dir), -1.0, 1.0))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                    arrow.rotate(R, center=[0, 0, 0])
                arrow.translate(start_pos)
                arrow.paint_uniform_color([1, 0.5, 0])  # Orange
                geometries_grasp.append(arrow)
            
            # Moving object poses at start and end of grasp
            pb_pose_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            pb_pose_start.transform(frames_data[grasp_start_idx]['pb_pose_mat'])
            geometries_grasp.append(pb_pose_start)
            pb_pose_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            pb_pose_end.transform(frames_data[grasp_end_idx]['pb_pose_mat'])
            geometries_grasp.append(pb_pose_end)

            # Moving object point cloud at end of grasp
            pb_points = frames_data[grasp_end_idx]['pb_points']
            pb_colors = frames_data[grasp_end_idx]['pb_colors']
            if len(pb_points) > 0 and not np.all(pb_points == 0):
                pb_pcd = to_o3d_pcd(pb_points, pb_colors)
                geometries_grasp.append(pb_pcd)
        
        # Save matplotlib visualization
        debug_dir = demo_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if grasp_start_idx < len(frames_data) and grasp_end_idx < len(frames_data):
                # Plot moving object point cloud
                pb_points = frames_data[grasp_end_idx]['pb_points']
                pb_colors = frames_data[grasp_end_idx]['pb_colors']
                if len(pb_points) > 0:
                    step = max(1, len(pb_points) // 2000)
                    ax.scatter(pb_points[::step, 0], pb_points[::step, 1], pb_points[::step, 2],
                             c=pb_colors[::step], s=1, alpha=0.6)
                
                # Plot gripper frames (start/end) and trajectory
                plot_coordinate_frame(ax, frames_data[grasp_start_idx]['gripper_pose_mat'], scale=0.05)
                plot_coordinate_frame(ax, frames_data[grasp_end_idx]['gripper_pose_mat'], scale=0.05)
                
                gripper_positions = np.array([frames_data[t]['gripper_pose_mat'][:3, 3] 
                                             for t in range(grasp_start_idx, min(grasp_end_idx + 1, len(frames_data)))])
                ax.plot(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
                       'gray', linewidth=2, alpha=0.5)
                
                # Plot object frames (start/end) and trajectory
                plot_coordinate_frame(ax, frames_data[grasp_start_idx]['pb_pose_mat'], scale=0.04)
                plot_coordinate_frame(ax, frames_data[grasp_end_idx]['pb_pose_mat'], scale=0.04)
                
                pb_positions = np.array([frames_data[t]['pb_pose_mat'][:3, 3] 
                                        for t in range(grasp_start_idx, min(grasp_end_idx + 1, len(frames_data)))])
                ax.plot(pb_positions[:, 0], pb_positions[:, 1], pb_positions[:, 2],
                       'orange', linewidth=2, linestyle='--', alpha=0.7)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{task_name} - Grasp [{grasp_interval[0]}-{grasp_interval[1]}]')
            xyz_limits = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
            xyz_center = np.mean(xyz_limits, axis=1)
            xyz_range = np.max(xyz_limits[:,1] - xyz_limits[:,0]) / 2
            for ctr, set_lim in zip(xyz_center, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
                set_lim(ctr - xyz_range, ctr + xyz_range)
            
            plt.savefig(debug_dir / 'grasp_interval.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Grasp interval saved to {debug_dir / 'grasp_interval.png'}")
        except Exception as e:
            print(f"  ⚠ Failed to save grasp visualization: {e}")
        
        # Show Open3D visualization
        try:
            print("Showing grasp interval (Open3D). Close window to continue.")
            o3d.visualization.draw_geometries(
                geometries_grasp,
                window_name=f'{task_name}_grasp_interval',
                width=1200,
                height=800
            )
        except Exception as e:
            print(f"  ⚠ Open3D visualization failed (headless environment): {e}")
        
        # Visualize manip interval
        print(f"Visualizing manip interval: [{manip_interval[0]}, {manip_interval[1]}]")
        geometries_manip = []
        
        # World frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        geometries_manip.append(world_frame)
        
        # Manip interval: gripper poses (start and end)
        manip_start_idx = manip_interval[0]
        manip_end_idx = manip_interval[1]
        
        if manip_start_idx < len(frames_data) and manip_end_idx < len(frames_data):
            # Gripper start pose
            gripper_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            gripper_start.transform(frames_data[manip_start_idx]['gripper_pose_mat'])
            gripper_start.paint_uniform_color([1, 0, 0])  # Red for start
            geometries_manip.append(gripper_start)
            
            # Gripper end pose
            gripper_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            gripper_end.transform(frames_data[manip_end_idx]['gripper_pose_mat'])
            gripper_end.paint_uniform_color([0, 1, 0])  # Green for end
            geometries_manip.append(gripper_end)
            
            # Arrow connecting start to end
            start_pos = frames_data[manip_start_idx]['gripper_pose_mat'][:3, 3]
            end_pos = frames_data[manip_end_idx]['gripper_pose_mat'][:3, 3]
            arrow_vec = end_pos - start_pos
            arrow_length = np.linalg.norm(arrow_vec)
            if arrow_length > 1e-6:
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.005,
                    cone_radius=0.01,
                    cylinder_height=arrow_length * 0.8,
                    cone_height=arrow_length * 0.2
                )
                arrow_dir = arrow_vec / arrow_length
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, arrow_dir)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                if rotation_axis_norm > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    rotation_angle = np.arccos(np.clip(np.dot(z_axis, arrow_dir), -1.0, 1.0))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                    arrow.rotate(R, center=[0, 0, 0])
                arrow.translate(start_pos)
                arrow.paint_uniform_color([1, 0.5, 0])  # Orange
                geometries_manip.append(arrow)

            # Base object poses at start of manip
            pa_pose_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            pa_pose_start.transform(frames_data[manip_start_idx]['pa_pose_mat'])
            geometries_manip.append(pa_pose_start)

            # Moving object poses at start and end of manip
            pb_pose_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            pb_pose_start.transform(frames_data[manip_start_idx]['pb_pose_mat'])
            geometries_manip.append(pb_pose_start)
            pb_pose_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            pb_pose_end.transform(frames_data[manip_end_idx]['pb_pose_mat'])
            geometries_manip.append(pb_pose_end)
            
            # Moving object point clouds at start and end of manip
            pb_points_start = frames_data[manip_start_idx]['pb_points']
            pb_colors_start = frames_data[manip_start_idx]['pb_colors']
            if len(pb_points_start) > 0 and not np.all(pb_points_start == 0):
                pb_pcd_start = to_o3d_pcd(pb_points_start, pb_colors_start)
                # pb_pcd_start.paint_uniform_color([1, 0, 0])  # Red for start
                geometries_manip.append(pb_pcd_start)
            
            pb_points_end = frames_data[manip_end_idx]['pb_points']
            pb_colors_end = frames_data[manip_end_idx]['pb_colors']
            if len(pb_points_end) > 0 and not np.all(pb_points_end == 0):
                pb_pcd_end = to_o3d_pcd(pb_points_end, pb_colors_end)
                # pb_pcd_end.paint_uniform_color([0, 1, 0])  # Green for end
                geometries_manip.append(pb_pcd_end)
            
            # Base object at end of manip
            pa_points = frames_data[manip_end_idx]['pa_points']
            pa_colors = frames_data[manip_end_idx]['pa_colors']
            if len(pa_points) > 0 and not np.all(pa_points == 0):
                pa_pcd = to_o3d_pcd(pa_points, pa_colors)
                geometries_manip.append(pa_pcd)
        
        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if manip_start_idx < len(frames_data) and manip_end_idx < len(frames_data):
                # Plot base object point cloud
                pa_points = frames_data[manip_end_idx]['pa_points']
                pa_colors = frames_data[manip_end_idx]['pa_colors']
                if len(pa_points) > 0:
                    step = max(1, len(pa_points) // 2000)
                    ax.scatter(pa_points[::step, 0], pa_points[::step, 1], pa_points[::step, 2],
                             c=pa_colors[::step], s=1, alpha=0.6)
                
                # Plot moving object point clouds (start and end)
                for idx, alpha in [(manip_start_idx, 0.3), (manip_end_idx, 0.6)]:
                    pb_pts = frames_data[idx]['pb_points']
                    pb_cols = frames_data[idx]['pb_colors']
                    if len(pb_pts) > 0:
                        step = max(1, len(pb_pts) // 2000)
                        ax.scatter(pb_pts[::step, 0], pb_pts[::step, 1], pb_pts[::step, 2],
                                 c=pb_cols[::step], s=1, alpha=alpha)
                
                # Plot gripper frames (start/end) and trajectory
                plot_coordinate_frame(ax, frames_data[manip_start_idx]['gripper_pose_mat'], scale=0.05)
                plot_coordinate_frame(ax, frames_data[manip_end_idx]['gripper_pose_mat'], scale=0.05)
                
                gripper_positions = np.array([frames_data[t]['gripper_pose_mat'][:3, 3] 
                                             for t in range(manip_start_idx, min(manip_end_idx + 1, len(frames_data)))])
                ax.plot(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
                       'gray', linewidth=2, alpha=0.5)
                
                # Plot object frames (start/end) and trajectory
                plot_coordinate_frame(ax, frames_data[manip_start_idx]['pa_pose_mat'], scale=0.06)
                plot_coordinate_frame(ax, frames_data[manip_start_idx]['pb_pose_mat'], scale=0.02)
                plot_coordinate_frame(ax, frames_data[manip_end_idx]['pb_pose_mat'], scale=0.02)
                
                pb_positions = np.array([frames_data[t]['pb_pose_mat'][:3, 3] 
                                        for t in range(manip_start_idx, min(manip_end_idx + 1, len(frames_data)))])
                ax.plot(pb_positions[:, 0], pb_positions[:, 1], pb_positions[:, 2],
                       'orange', linewidth=2, linestyle='--', alpha=0.7)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{task_name} - Manip [{manip_interval[0]}-{manip_interval[1]}]')
            xyz_limits = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
            xyz_center = np.mean(xyz_limits, axis=1)
            xyz_range = np.max(xyz_limits[:,1] - xyz_limits[:,0]) / 2
            for ctr, set_lim in zip(xyz_center, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
                set_lim(ctr - xyz_range, ctr + xyz_range)
            
            plt.savefig(debug_dir / 'manip_interval.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Manip interval saved to {debug_dir / 'manip_interval.png'}")
        except Exception as e:
            print(f"  ⚠ Failed to save manip visualization: {e}")
        
        # Show Open3D visualization
        try:
            print("Showing manip interval (Open3D). Close window to continue.")
            o3d.visualization.draw_geometries(
                geometries_manip,
                window_name=f'{task_name}_manip_interval',
                width=1200,
                height=800
            )
        except Exception as e:
            print(f"  ⚠ Open3D visualization failed (headless environment): {e}")
    return data_dic


def build_dataset(dataset_root, save_dir, num_demos=None,
                 voxel_size=0.004, num_points=2048,
                 disp=False, verbose=False):
    """ Build dataset from all demos in dataset_root. """
    dataset_root = Path(dataset_root)
    # Find all demo directories
    demo_dirs = sorted([d for d in dataset_root.iterdir() 
                       if d.is_dir() and (d / 'task_info.json').exists()])
    if len(demo_dirs) == 0:
        print(f"No valid demo directories found in {dataset_root}")
        return
    
    # Limit number of demos if specified
    if num_demos is not None and num_demos > 0:
        demo_dirs = demo_dirs[:num_demos]
    print(f"Found {len(demo_dirs)} demos to process")
    print("="*70)
    
    # Initialize mask generator once for all demos
    print("Initializing MaskGenerator...")
    mask_generator = MaskGenerator()
    
    # Process each demo
    for demo_dir in demo_dirs:
        try:
            save_data_from_demo(
                demo_dir=demo_dir,
                save_dir=save_dir,
                mask_generator=mask_generator,
                voxel_size=voxel_size,
                num_points=num_points,
                disp=disp,
                verbose=verbose
            )
        except Exception as e:
            print(f"Error processing {demo_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    print("\n" + "="*70)
    print("Dataset preprocessing complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess real-world demo data for FOCI training'
    )
    parser.add_argument('--dataset_root', type=str,
                       default='../dataset',
                       help='Root directory containing demo folders')
    parser.add_argument('--save_dir', type=str,
                       default='../../foci_dataset_realworld',
                       help='Directory to save preprocessed data')
    parser.add_argument('--num_demos', type=int, default=None,
                       help='Number of demos to process (default: all)')
    parser.add_argument('--voxel_size', type=float, default=0.004,
                       help='Voxel size for downsampling')
    parser.add_argument('--num_points', type=int, default=2048,
                       help='Number of points to sample')
    parser.add_argument('--disp', action='store_true',
                       help='Display visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # dataset path
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / args.dataset_root
    save_dir = script_dir / args.save_dir
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return
    
    print(f"Dataset root: {dataset_root}")
    print(f"Save directory: {save_dir}")
    
    build_dataset(
        dataset_root=dataset_root,
        save_dir=save_dir,
        num_demos=args.num_demos,
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        disp=args.disp,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
