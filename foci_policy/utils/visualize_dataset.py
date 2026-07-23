#!/usr/bin/env python3
"""
Visualize preprocessed dataset for FOCI training.
"""

import numpy as np
import os
import sys
import argparse
import pickle
import open3d as o3d
import yaml
from pathlib import Path
from foci_policy.utils.transform_utils import to_o3d_pcd


def create_trajectory_line(positions, color, radius=0.003):
    """Create line set for trajectory visualization."""
    if len(positions) < 2:
        return None
    points = positions
    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def add_waypoint_markers(geometries, frames_data, indices, pose_key, color, frame_size=0.02, sphere_radius=0.004):
    """Add a coordinate frame + sphere marker for every waypoint index."""
    for idx in indices:
        if 0 <= idx < len(frames_data) and frames_data[idx].get(pose_key) is not None:
            pose_mat = frames_data[idx][pose_key]
            waypoint_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            waypoint_frame.transform(pose_mat)
            geometries.append(waypoint_frame)

            waypoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            waypoint_sphere.translate(pose_mat[:3, 3])
            waypoint_sphere.paint_uniform_color(color)
            geometries.append(waypoint_sphere)


def visualize_demo(pkl_path, demo_idx=0):
    """Visualize a single demo from pkl file."""
    print(f"Loading demo {demo_idx} from {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    task_name = data['task_name']
    frames_data = data['frames_data']
    keyframes = data['key_times']
    total_frames = data['total_frames']
    grasp_interval = data.get('grasp_interval', 10)
    manip_interval = data.get('manip_interval', 10)
    
    pick_idx = keyframes['pick']
    place_idx = keyframes['place']
    
    # Calculate interval ranges
    grasp_start = max(0, pick_idx - grasp_interval)
    grasp_end = min(len(frames_data), pick_idx + 1)
    manip_start = max(0, place_idx - manip_interval)
    manip_end = min(len(frames_data), place_idx + 1)

    grasp_waypoints = list(range(grasp_start, grasp_end))
    manip_waypoints = list(range(manip_start, manip_end))
    
    print(f"  Task: {task_name}")
    print(f"  Total frames: {total_frames}")
    print(f"  Pick frame: {pick_idx}")
    print(f"  Place frame: {place_idx}")
    print(f"  Grasp interval: [{grasp_start}, {grasp_end})")
    print(f"  Manip interval: [{manip_start}, {manip_end})")
    print(f"  Grasp waypoints (n={len(grasp_waypoints)}): {grasp_waypoints}")
    print(f"  Manip waypoints (n={len(manip_waypoints)}): {manip_waypoints}")
    print(f"  Language (grasp): {data.get('lan_pick', 'N/A')}")
    print(f"  Language (manip): {data.get('lan_place', 'N/A')}")
    
    geometries = []
    
    # World coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    geometries.append(world_frame)
    
    # Extract trajectories
    gripper_traj = np.array([f['gripper_pose_mat'][:3, 3] for f in frames_data])
    moving_traj = []
    base_traj = []
    
    for f in frames_data:
        if f['pb_pose_mat'] is not None:
            moving_traj.append(f['pb_pose_mat'][:3, 3])
        if f['pa_pose_mat'] is not None:
            base_traj.append(f['pa_pose_mat'][:3, 3])
    
    moving_traj = np.array(moving_traj) if len(moving_traj) > 0 else None
    base_traj = np.array(base_traj) if len(base_traj) > 0 else None
    
    # 1. Add gripper trajectory (red)
    gripper_line = create_trajectory_line(gripper_traj, [1, 0, 0])
    if gripper_line is not None:
        geometries.append(gripper_line)
    
    # 2. Add all grasp phase gripper waypoints (not just start/end)
    add_waypoint_markers(
        geometries=geometries,
        frames_data=frames_data,
        indices=grasp_waypoints,
        pose_key='gripper_pose_mat',
        color=[1, 0.35, 0.35],
        frame_size=0.028,
        sphere_radius=0.004,
    )
    
    # 3. Add moving object trajectory (green)
    if moving_traj is not None and len(moving_traj) > 1:
        moving_line = create_trajectory_line(moving_traj, [0, 1, 0])
        if moving_line is not None:
            geometries.append(moving_line)
    
    # 4. Add base object trajectory (blue) - should be mostly static
    if base_traj is not None and len(base_traj) > 1:
        base_line = create_trajectory_line(base_traj, [0, 0, 1])
        if base_line is not None:
            geometries.append(base_line)
    
    # 5. Add all manip phase moving-object waypoints (not just a few key points)
    add_waypoint_markers(
        geometries=geometries,
        frames_data=frames_data,
        indices=manip_waypoints,
        pose_key='pb_pose_mat',
        color=[0.15, 0.15, 0.15],
        frame_size=0.03,
        sphere_radius=0.0045,
    )
    
    # 6. Add base object frame at first frame
    if 0 < len(frames_data) and frames_data[0]['pa_pose_mat'] is not None:
        pa_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        pa_frame.transform(frames_data[0]['pa_pose_mat'])
        geometries.append(pa_frame)
    
    # 7. Add point clouds with original colors at key frames
    # Moving object at pick frame (with original colors)
    if pick_idx < len(frames_data):
        pb_points_pick = frames_data[pick_idx]['pb_points']
        pb_colors_pick = frames_data[pick_idx]['pb_colors']
        print(f"  Point counts: pb@pick={len(pb_points_pick)}")
        if len(pb_points_pick) > 0 and not np.all(pb_points_pick == 0):
            pb_pcd_pick = to_o3d_pcd(pb_points_pick, pb_colors_pick)
            geometries.append(pb_pcd_pick)
    
    # Moving object at place frame (with original colors)
    if place_idx < len(frames_data):
        pb_points_place = frames_data[place_idx]['pb_points']
        pb_colors_place = frames_data[place_idx]['pb_colors']
        print(f"  Point counts: pb@place={len(pb_points_place)}")
        if len(pb_points_place) > 0 and not np.all(pb_points_place == 0):
            pb_pcd_place = to_o3d_pcd(pb_points_place, pb_colors_place)
            geometries.append(pb_pcd_place)
    
    # Base object at first frame (with original colors)
    if 0 < len(frames_data):
        pa_points_start = frames_data[0]['pa_points']
        pa_colors_start = frames_data[0]['pa_colors']
        print(f"  Point counts: pa@start={len(pa_points_start)}")
        if len(pa_points_start) > 0 and not np.all(pa_points_start == 0):
            pa_pcd_start = to_o3d_pcd(pa_points_start, pa_colors_start)
            geometries.append(pa_pcd_start)
    
    # Visualize
    print(f"\nVisualizing demo {demo_idx}:")
    print(f"  Legend:")
    print(f"    - Red line: Gripper trajectory")
    print(f"    - Light red waypoint markers: ALL grasp waypoints (gripper), n={len(grasp_waypoints)}")
    print(f"    - Green line: Moving object trajectory")
    print(f"    - Blue line: Base object trajectory")
    print(f"    - Dark markers: ALL manip waypoints (moving object), n={len(manip_waypoints)}")
    print(f"    - RGB frame: Base object pose (first frame)")
    print(f"    - Point clouds with original colors:")
    print(f"      * Moving object at pick and place frames")
    print(f"      * Base object at first frame")
    print(f"\nClose window to continue...\n")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f'{task_name} - Demo {demo_idx}',
        width=1920,
        height=1080,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize preprocessed dataset'
    )
    parser.add_argument('--task_name', type=str, default=None,
                       help='Task name (e.g., close_jar). If omitted, use dataloader config tasks.')
    parser.add_argument('--dataset', type=str, default='simu',
                       choices=['simu', 'realworld'],
                       help='Dataset type: simu (default) or realworld')
    parser.add_argument('--num_demos', type=int, default=None,
                       help='Number of demos to visualize (default: all)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting demo index (default: 0)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent

    def visualize_task(task_name):
        if args.dataset == 'simu':
            dataset_dir = script_dir / '../../foci_dataset' / task_name
        else:  # realworld
            dataset_dir = script_dir / '../../foci_dataset_realworld' / task_name

        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            return

        pkl_files = sorted(list(dataset_dir.glob('*.pkl')))
        if len(pkl_files) == 0:
            print(f"Warning: No pkl files found in {dataset_dir}")
            return

        pkl_files = pkl_files[args.start_idx:]
        if args.num_demos is not None and args.num_demos > 0:
            pkl_files = pkl_files[:args.num_demos]

        print("="*70)
        print(f"Dataset directory: {dataset_dir}")
        print(f"Found {len(pkl_files)} demos to visualize")
        print(f"Dataset type: {args.dataset}")
        print("="*70)
        print()

        for i, pkl_path in enumerate(pkl_files):
            demo_idx = args.start_idx + i
            try:
                visualize_demo(pkl_path, demo_idx)
            except KeyboardInterrupt:
                print("\nVisualization interrupted by user.")
                return
            except Exception as e:
                print(f"Error visualizing demo {demo_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if args.task_name is None:
        dataloader_cfg_path = script_dir / '../config/dataloader.yaml'
        with open(dataloader_cfg_path, 'r', encoding='utf-8') as f:
            dataloader_cfg = yaml.safe_load(f)
        task_list = dataloader_cfg.get('task_list', []) if isinstance(dataloader_cfg, dict) else []
        if not isinstance(task_list, list) or len(task_list) == 0:
            print(f"Error: task_list missing in {dataloader_cfg_path}")
            return
        for task_name in task_list:
            visualize_task(task_name)
    else:
        visualize_task(args.task_name)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
