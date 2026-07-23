"""
Visualize object poses extracted by Foundation Pose.

Usage:
    python visualize_fp_poses.py --task_name close_jar --episode_idx 0 \\
        --mesh_dir ../assets/RLBench_mesh
"""

import numpy as np
import os
import argparse
import pickle
from pathlib import Path
from rlbench.utils import get_stored_demo
import open3d as o3d
from foci_policy.pose_estimator.foundation_pose_utils import extract_poses_for_demo
from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
from foci_policy.config.config_utils import get_mesh_names
from utils_rlbench.process_demo import get_pick_place_keyframes

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RAW_DATA_PATH = REPO_ROOT / 'data' / 'rlbench_data'
DEFAULT_MESH_DIR = REPO_ROOT / 'foci_policy' / 'assets' / 'RLBench_mesh'


def visualize_pose_trajectory(demo, poses_data, task_name, mesh_dir, camera_name='front', vis_frames=None, max_frame=None):
    """ Visualize object pose trajectories with mesh. """
    if not poses_data['success']:
        print("Error: Pose extraction failed, cannot visualize")
        return

    mesh_names = get_mesh_names(task_name)
    if not mesh_names:
        raise ValueError(f"No mesh names configured for task '{task_name}'")

    geometries = []
    
    # World frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(world_frame)
    
    # Add point clouds at selected frames
    if vis_frames is None:
        vis_frames = [0, len(demo)//2, len(demo)-1]
    
    for frame_idx in vis_frames:
        if frame_idx >= len(demo):
            continue
        if max_frame is not None and frame_idx > max_frame:
            continue
        frame = demo[frame_idx]
        pcd_data = getattr(frame, f'{camera_name}_point_cloud')  # (3, H, W)
        rgb_data = getattr(frame, f'{camera_name}_rgb')  # (3, H, W) or (H, W, 3)
        # Reshape to (N, 3)
        if pcd_data.shape[0] == 3:  # (3, H, W)
            points_world = pcd_data.reshape(3, -1).T  # (N, 3)
            if rgb_data.shape[0] == 3:  # (3, H, W)
                colors = rgb_data.reshape(3, -1).T / 255.0
            else:  # (H, W, 3)
                colors = rgb_data.reshape(-1, 3) / 255.0
        else:  # (H, W, 3)
            points_world = pcd_data.reshape(-1, 3)
            colors = rgb_data.reshape(-1, 3) / 255.0
        # Filter valid points
        valid_mask = (points_world[:, 2] > 0.01) & (points_world[:, 2] < 2.5)
        points_world = points_world[valid_mask]
        colors = colors[valid_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        geometries.append(pcd)
    
    # Visualize object trajectories with meshes
    colors_obj = [[1, 0, 0], [0, 0, 1]]  # Red for moving, Blue for base

    for i, mesh_name in enumerate(mesh_names[:2]):
        mesh_path = os.path.join(mesh_dir, task_name, f"{mesh_name}.obj")
        if not os.path.exists(mesh_path):
            print(f"Warning: Mesh not found: {mesh_path}")
            continue
        mesh_base = o3d.io.read_triangle_mesh(mesh_path)
        mesh_base.compute_vertex_normals()
        
        # Get poses
        if i == 0:  # moving object
            poses_dict = poses_data['moving_poses']
        else:  # base object
            poses_dict = poses_data['base_poses']
        
        # Show meshes at intervals
        frame_indices = sorted(poses_dict.keys())
        if max_frame is not None:
            frame_indices = [t for t in frame_indices if t <= max_frame]
        if len(frame_indices) == 0:
            continue
        for idx, t in enumerate(frame_indices[::5]):  # Every 5th frame
            pose = poses_dict[t]
            mesh = o3d.geometry.TriangleMesh(mesh_base)
            mesh.transform(pose)
            # Color fade based on time
            alpha = 0.3 + 0.7 * (idx / max(len(frame_indices[::5]) - 1, 1))
            color = np.array(colors_obj[i]) * alpha
            mesh.paint_uniform_color(color)
            geometries.append(mesh)
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            coord_frame.transform(pose)
            geometries.append(coord_frame)
    
    # Show visualization
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f'Foundation Pose - {task_name}',
        width=1280,
        height=960
    )


def save_poses(poses_data, task_name, episode_idx, output_dir='./foundationpose_outputs'):
    """Save extracted poses to file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{task_name}_ep{episode_idx}_poses.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(poses_data, f)
    print(f"Saved poses to: {output_file}")
    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Foundation Pose results')
    parser.add_argument('--task_name', type=str, default='close_jar')
    parser.add_argument('--episode_idx', type=int, default=0)
    parser.add_argument('--mesh_dir', type=str, default=str(DEFAULT_MESH_DIR))
    parser.add_argument('--data_path', type=str, default=None, help='Path to RLBench demo data (overrides default)')
    parser.add_argument('--camera_name', type=str, default='front',
                        choices=['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                        help='Camera view used for mask generation and pose tracking')
    parser.add_argument('--no_vis', action='store_true', help='Skip visualization')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--postprocess', action='store_true', 
                        help='Apply postprocessing to object trajectory')

    args = parser.parse_args()

    # Load demo
    if args.data_path is not None:
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = SCRIPT_DIR / data_path
    else:
        data_path = DEFAULT_RAW_DATA_PATH / args.task_name / 'variation0' / 'episodes'
    if not data_path.exists():
        raise FileNotFoundError(
            f'Raw RLBench demo path not found: {data_path}. '
            f'Expected data under {DEFAULT_RAW_DATA_PATH}.'
        )
    demo = get_stored_demo(data_path=str(data_path), index=args.episode_idx, init_matrix=False)
    keyframes = get_pick_place_keyframes(demo, task_name=args.task_name)
    place_frame = keyframes['place']
    extractor = DemoObjectExtractor.from_config(args.task_name)

    first_frame_mask = {
        'moving': extractor.get_grasp_mask(demo[0], camera_name=args.camera_name),
        'base': extractor.get_target_mask(demo[0], camera_name=args.camera_name),
    }

    print(f"Loaded demo: {args.task_name}, episode {args.episode_idx}, {len(demo)} frames")
    print(f"Visualizing poses in frame range [0, {place_frame}] (up to place keyframe)")

    # Extract poses
    mesh_dir = str(Path(args.mesh_dir).expanduser())
    if not os.path.isabs(mesh_dir):
        mesh_dir = str(SCRIPT_DIR / mesh_dir)
    poses_data = extract_poses_for_demo(
        demo=demo,
        task_name=args.task_name,
        mesh_dir=mesh_dir,
        camera_name=args.camera_name,
        verbose=args.verbose,
        postprocess=args.postprocess,
        first_frame_mask=first_frame_mask
    )

    if poses_data['success']:
        print(f"  Moving object: {len(poses_data['moving_poses'])} frames")
        print(f"  Base object: {len(poses_data['base_poses'])} frames")
        print(f"  Postprocessing applied: {poses_data.get('postprocessed', False)}")

        # Print statistics
        if len(poses_data['moving_poses']) > 0:
            moving_trans = np.array([p[:3, 3] for p in poses_data['moving_poses'].values()])
            print(f"  Moving object displacement: {np.linalg.norm(moving_trans[-1] - moving_trans[0]):.3f}m")

        if len(poses_data['base_poses']) > 0:
            base_trans = np.array([p[:3, 3] for p in poses_data['base_poses'].values()])
            base_std = base_trans.std(axis=0)
            print(f"  Base object stability (std): [{base_std[0]:.4f}, {base_std[1]:.4f}, {base_std[2]:.4f}]m")

        # Save
        if args.save:
            save_poses(poses_data, args.task_name, args.episode_idx)

        # Visualize
        if not args.no_vis:
            visualize_pose_trajectory(
                demo,
                poses_data,
                args.task_name,
                mesh_dir,
                camera_name=args.camera_name,
                max_frame=place_frame,
            )
    else:
        print("\n✗ Pose extraction failed")
        print("Check:")
        print("  0. FoundationPose checkpoint in `data/model_weight/foundation_pose`")
        print("  1. Mesh files are present")
        print("  2. Task configuration are correct (see `foci_policy/assets/readme.md`)")
        print("  3. Object are difficult to detect (small, symmetrical, or partially visible)")
