#!/usr/bin/env python3
"""
Extract replay data from real-world demos.
Extracts first frame point clouds and grasp/manip interval gripper poses.
"""

import os
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from foci_real_world.pose_estimator.mask_generator import MaskGenerator


def get_mask_from_generator(mask_generator, rgb, object_name):
    """Get object mask using GroundedSAM mask generator"""
    image_pil = Image.fromarray(rgb)
    objects_list = [object_name]
    # Get bounding boxes
    boxes, logits, phrases = mask_generator.get_scene_object_bboxes(
        image_pil, objects_list, visualize=False, logdir=None
    )
    # Get segmentation masks
    segmasks = mask_generator.get_segmentation_masks(
        image_pil, boxes, logits, phrases, visualize=False, save_path=None
    )
    if len(segmasks) > 0:
        return segmasks[0].astype(np.uint8)
    else:
        return None


def load_task_info(demo_dir):
    """Load task information from JSON file"""
    task_info_path = os.path.join(demo_dir, 'task_info.json')
    with open(task_info_path, 'r') as f:
        return json.load(f)


def load_trajectory(demo_dir):
    """Load gripper trajectory from JSON file"""
    traj_path = os.path.join(demo_dir, 'trajectory.json')
    with open(traj_path, 'r') as f:
        return json.load(f)


def load_camera_intrinsics(demo_dir):
    """Load camera intrinsics from JSON file"""
    intrinsics_path = os.path.join(demo_dir, 'camera_intrinsics.json')
    with open(intrinsics_path, 'r') as f:
        return json.load(f)


def load_camera_extrinsics(demo_dir):
    """Load camera extrinsics from JSON file"""
    extrinsics_path = os.path.join(demo_dir, 'camera_extrinsics.json')
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
    translation = np.array([tx, ty, tz])
    
    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = rotation
    extrinsic_mat[:3, 3] = translation
    return extrinsic_mat


def parse_gripper_pose(gripper_dict):
    """Parse gripper pose from trajectory dict"""
    pos_dict = gripper_dict['gripper_pose']['position']
    position = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']])
    
    ori_dict = gripper_dict['gripper_pose']['orientation']
    orientation = np.array([ori_dict['x'], ori_dict['y'], ori_dict['z'], ori_dict['w']])
    orientation = orientation / np.linalg.norm(orientation)
    
    rotation_matrix = Rotation.from_quat(orientation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    return transform_matrix


def extract_masked_point_cloud(rgb_path, depth_path, mask, intrinsics, depth_scale=1000.0):
    """Extract masked point cloud from RGB-D images"""
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
    
    # Resize mask to match depth
    depth_h, depth_w = depth.shape
    mask_h, mask_w = mask.shape
    if mask_h != depth_h or mask_w != depth_w:
        mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
    
    # Get intrinsics
    fx = intrinsics['K'][0]
    fy = intrinsics['K'][4]
    cx = intrinsics['K'][2]
    cy = intrinsics['K'][5]
    
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
    points = np.stack([x, y, z], axis=1)
    
    # Get colors
    colors = rgb[valid_mask].astype(np.float32) / 255.0
    return points, colors


def process_demo(demo_dir, mask_generator=None, verbose=False):
    """
    Process a demo and extract replay data.
    
    Returns:
        dict with keys:
            - pa_points_init: (N, 3) first frame base object points
            - pa_colors_init: (N, 3) first frame base object colors
            - pb_points_init: (N, 3) first frame moving object points
            - pb_colors_init: (N, 3) first frame moving object colors
            - grasp_start_pose: (4, 4) gripper pose at grasp start
            - grasp_end_pose: (4, 4) gripper pose at grasp end
            - manip_start_pose: (4, 4) gripper pose at manip start
            - manip_end_pose: (4, 4) gripper pose at manip end
    """
    demo_dir = Path(demo_dir)
    
    # Load task info
    task_info = load_task_info(demo_dir)
    moving_object = task_info['moving_object']
    base_object = task_info['base_object']
    grasp_interval = task_info['grasp_interval']
    manip_interval = task_info['manip_interval']
    
    if verbose:
        print(f"Processing demo: {demo_dir.name}")
        print(f"  Base object: {base_object}")
        print(f"  Moving object: {moving_object}")
        print(f"  Grasp interval: {grasp_interval}")
        print(f"  Manip interval: {manip_interval}")
    
    # Load trajectory
    trajectory = load_trajectory(demo_dir)
    
    # Load camera parameters
    intrinsics = load_camera_intrinsics(demo_dir)
    extrinsics = load_camera_extrinsics(demo_dir)
    
    # Initialize mask generator
    if mask_generator is None:
        mask_generator = MaskGenerator()
    
    # Process first frame
    first_frame_idx = 0
    rgb_path = demo_dir / 'color' / f'{first_frame_idx:04d}.png'
    depth_path = demo_dir / 'depth' / f'{first_frame_idx:04d}.png'
    
    if verbose:
        print(f"  Extracting first frame point clouds...")
    
    # Load first frame RGB
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Generate masks for both objects
    base_mask = get_mask_from_generator(mask_generator, rgb, base_object)
    moving_mask = get_mask_from_generator(mask_generator, rgb, moving_object)
    if base_mask is None:
        raise RuntimeError(f"Failed to generate mask for base object: {base_object}")
    if moving_mask is None:
        raise RuntimeError(f"Failed to generate mask for moving object: {moving_object}")
    
    # Extract point clouds
    pa_points, pa_colors = extract_masked_point_cloud(
        str(rgb_path), str(depth_path), base_mask, intrinsics
    )
    pb_points, pb_colors = extract_masked_point_cloud(
        str(rgb_path), str(depth_path), moving_mask, intrinsics
    )
    
    # Transform to world coordinates
    pa_points = (extrinsics[:3, :3] @ pa_points.T).T + extrinsics[:3, 3]
    pb_points = (extrinsics[:3, :3] @ pb_points.T).T + extrinsics[:3, 3]
    
    if verbose:
        print(f"    Base object points: {len(pa_points)}")
        print(f"    Moving object points: {len(pb_points)}")
    
    # Extract gripper poses at interval boundaries
    grasp_start_idx = grasp_interval[0]
    grasp_end_idx = grasp_interval[1]
    manip_start_idx = manip_interval[0]
    manip_end_idx = manip_interval[1]
    
    grasp_start_pose = parse_gripper_pose(trajectory[grasp_start_idx])
    grasp_end_pose = parse_gripper_pose(trajectory[grasp_end_idx])
    manip_start_pose = parse_gripper_pose(trajectory[manip_start_idx])
    manip_end_pose = parse_gripper_pose(trajectory[manip_end_idx])
    
    if verbose:
        print(f"  Extracted gripper poses:")
        print(f"    Grasp start (frame {grasp_start_idx})")
        print(f"    Grasp end (frame {grasp_end_idx})")
        print(f"    Manip start (frame {manip_start_idx})")
        print(f"    Manip end (frame {manip_end_idx})")
    
    replay_data = {
        'pa_points_init': pa_points,
        'pa_colors_init': pa_colors,
        'pb_points_init': pb_points,
        'pb_colors_init': pb_colors,
        'grasp_start_pose': grasp_start_pose,
        'grasp_end_pose': grasp_end_pose,
        'manip_start_pose': manip_start_pose,
        'manip_end_pose': manip_end_pose,
        'base_object': base_object,
        'moving_object': moving_object,
    }
    
    # Save to demo directory
    save_path = demo_dir / 'replay_data.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(replay_data, f)
    
    if verbose:
        print(f"  Saved to: {save_path}")
    
    return replay_data


def process_all_demos(dataset_root, num_demos=None, verbose=False):
    """Process all demos in dataset root"""
    dataset_root = Path(dataset_root)
    
    # Find demo directories
    demo_dirs = sorted([d for d in dataset_root.iterdir() 
                       if d.is_dir() and (d / 'task_info.json').exists()])
    
    if len(demo_dirs) == 0:
        print(f"No valid demos found in {dataset_root}")
        return
    
    if num_demos is not None:
        demo_dirs = demo_dirs[:num_demos]
    
    print(f"Found {len(demo_dirs)} demos to process")
    print("="*70)
    
    # Initialize mask generator once
    print("Initializing MaskGenerator...")
    mask_generator = MaskGenerator()
    
    # Process each demo
    for demo_dir in tqdm(demo_dirs, desc="Processing demos"):
        try:
            process_demo(demo_dir, mask_generator=mask_generator, verbose=verbose)
        except Exception as e:
            print(f"Error processing {demo_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Replay data extraction complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract replay data from demos')
    parser.add_argument('--dataset_root', type=str, default='../dataset',
                       help='Root directory containing demo folders')
    parser.add_argument('--num_demos', type=int, default=None,
                       help='Number of demos to process')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / args.dataset_root
    
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return
    
    print(f"Dataset root: {dataset_root}")
    
    process_all_demos(
        dataset_root=dataset_root,
        num_demos=args.num_demos,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
