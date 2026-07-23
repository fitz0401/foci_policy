#!/usr/bin/env python3
"""
Offline Visualization Tool for FOCI Policy Predictions
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path


def load_vis_data(data_path):
    """Load visualization data from npz file"""
    data = np.load(data_path, allow_pickle=True)
    return {
        'scene_points': data['scene_points'],
        'scene_colors': data['scene_colors'],
        'pa_points': data['pa_points'],
        'pa_colors': data['pa_colors'],
        'pb_points': data['pb_points'],
        'pb_colors': data['pb_colors'],
        'base_pose': data['base_pose'],
        'moving_pose': data['moving_pose'],
        'trajectory': data['trajectory'],
        'mode': str(data['mode']),
        'rgb': data['rgb'],
        'depth': data['depth'],
        'K': data['K'],
        'cam_extrinsic': data['cam_extrinsic']
    }


def visualize_data(data, show_scene=True, show_objects=True, show_trajectory=True):
    """
    Visualize loaded data using Open3D
    
    Args:
        data: Dictionary containing visualization data
        show_scene: Whether to show full scene point cloud
        show_objects: Whether to show object point clouds
        show_trajectory: Whether to show predicted trajectory
    """
    geometries = []
    
    # World frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    geometries.append(world_frame)
    
    # Full scene point cloud (dimmed)
    if show_scene:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(data['scene_points'])
        scene_pcd.colors = o3d.utility.Vector3dVector(data['scene_colors'])
        geometries.append(scene_pcd)
        print(f"Scene points: {len(data['scene_points'])}")
    
    # Object point clouds
    if show_objects:
        pa_pcd = o3d.geometry.PointCloud()
        pa_pcd.points = o3d.utility.Vector3dVector(data['pa_points'])
        pa_pcd.colors = o3d.utility.Vector3dVector(data['pa_colors'])
        geometries.append(pa_pcd)
        print(f"Base object points: {len(data['pa_points'])}")
        
        pb_pcd = o3d.geometry.PointCloud()
        pb_pcd.points = o3d.utility.Vector3dVector(data['pb_points'])
        pb_pcd.colors = o3d.utility.Vector3dVector(data['pb_colors'])
        geometries.append(pb_pcd)
        print(f"Moving object points: {len(data['pb_points'])}")
        
        # Object pose frames
        # Base object frame (blue)
        pa_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        pa_frame.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        pa_frame.transform(data['base_pose'])
        geometries.append(pa_frame)
        
        # Moving object frame (green)
        pb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        pb_frame.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        pb_frame.transform(data['moving_pose'])
        geometries.append(pb_frame)
    
    # Trajectory
    if show_trajectory:
        trajectory = data['trajectory']
        print(f"Trajectory waypoints: {len(trajectory)}")
        
        # Trajectory frames (red)
        for i, pose in enumerate(trajectory):
            traj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            traj_frame.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            traj_frame.transform(pose)
            geometries.append(traj_frame)
        
        # Trajectory line (red)
        if len(trajectory) > 1:
            traj_positions = [pose[:3, 3] for pose in trajectory]
            lines = [[i, i+1] for i in range(len(traj_positions)-1)]
            colors = [[1.0, 0.0, 0.0] for _ in lines]  # Red
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(traj_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
    
    # Visualize
    print(f"\nVisualizing {data['mode']} mode prediction...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Shift + Mouse: Pan")
    print("  - Q or ESC: Close window")
    
    # fixed camera parameters
    pinhole_params = o3d.camera.PinholeCameraParameters()
    # JSON extrinsic is stored in column-major order, convert to row-major 4x4 matrix
    pinhole_params.extrinsic = np.array([
        [-0.68727912573669869,  0.72638953602803225,  0.0023759788676099569,  0.073178639663463135],
        [ 0.44299234700276252,  0.42172843948099537, -0.79114025547301581, -0.37330050625058769],
        [-0.57567802096621967, -0.54268164266161112, -0.61163015858810632,  1.4557593633209303],
        [ 0.0,                  0.0,                  0.0,                   1.0]
    ])
    pinhole_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1610,
        height=986,
        fx=853.9010481314566,
        fy=853.9010481314566,
        cx=804.5,
        cy=492.5
    )

    def custom_draw_geometries(geometries, window_name, width, height, left, top):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=width, height=height, left=left, top=top)
        for g in geometries:
            vis.add_geometry(g)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(pinhole_params, allow_arbitrary=True)
        vis.run()
        vis.destroy_window()

    custom_draw_geometries(
        geometries,
        window_name=f"FOCI Prediction - {data['mode'].upper()} mode",
        width=1280,
        height=720,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(
        description='Offline visualization of FOCI policy predictions'
    )
    parser.add_argument('data_path', type=str,
                       help='Path to vis_data_*.npz file or directory containing them')
    parser.add_argument('--no-scene', action='store_true',
                       help='Hide full scene point cloud')
    parser.add_argument('--no-objects', action='store_true',
                       help='Hide object point clouds')
    parser.add_argument('--no-trajectory', action='store_true',
                       help='Hide predicted trajectory')
    parser.add_argument('--mode', type=str, choices=['grasp', 'manip', 'both'],
                       default='both',
                       help='Which mode to visualize (default: both)')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    # If directory is provided, find npz files
    if data_path.is_dir():
        if args.mode == 'both':
            npz_files = sorted(data_path.glob('vis_data_*.npz'))
        else:
            npz_files = [data_path / f'vis_data_{args.mode}.npz']
    else:
        npz_files = [data_path]
    
    if not npz_files:
        print(f"Error: No visualization data files found in {data_path}")
        return 1
    
    for npz_file in npz_files:
        if not npz_file.exists():
            print(f"Warning: {npz_file} not found, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Loading visualization data from: {npz_file}")
        print('='*70)
        
        try:
            data = load_vis_data(npz_file)
            visualize_data(
                data,
                show_scene=not args.no_scene,
                show_objects=not args.no_objects,
                show_trajectory=not args.no_trajectory
            )
        except Exception as e:
            print(f"Error visualizing {npz_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return 0


if __name__ == '__main__':
    exit(main())
