"""
Compute interaction intervals using change point detection.

Phases:
- Grasp phase: Gripper approaching and grasping object (gripper-object interaction)
- Manipulation phase: Moving object to target location (object-base interaction)
"""
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import ruptures as rpt
import open3d as o3d

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, '../../')))
sys.path.append(os.path.abspath(os.path.join(file_dir, '../../../')))

from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs
from scipy.spatial.transform import Rotation as R
from foci_policy.config.config_utils import get_interaction_interval_pen_factor
from utils_rlbench.process_demo import get_pick_place_keyframes, flat_pcd_image

# Camera configuration for point cloud extraction
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
DEFAULT_PEN_FACTOR = get_interaction_interval_pen_factor()

O3D_CAMERA_CONFIG_PATH = os.path.abspath(
    os.path.join(file_dir, 'ScreenCamera_2026-03-23-17-53-10.json')
)
O3D_WINDOW_SIZE = (1280, 850)

GRIPPER_PKL_PATH = os.path.abspath(os.path.join(file_dir, '../assets/gripper.pkl'))
GRIPPER_RETREAT_M = 0.05
_GRIPPER_TEMPLATE = None


def get_gripper_pose_from_obs(obs):
    """Extract gripper pose from observation."""
    if not hasattr(obs, 'gripper_pose') or obs.gripper_pose is None:
        return None
    gripper_pose = obs.gripper_pose
    if len(gripper_pose.shape) == 1 and len(gripper_pose) == 7:
        position = gripper_pose[:3]
        quaternion = gripper_pose[3:7]
        quaternion = quaternion / np.linalg.norm(quaternion)
        return position, quaternion
    elif len(gripper_pose.shape) == 2 and gripper_pose.shape == (4, 4):
        position = gripper_pose[:3, 3]
        r = R.from_matrix(gripper_pose[:3, :3])
        quaternion = r.as_quat()
        return position, quaternion
    return None


def get_gt_flat_masks_for_frame(frame, task_name, cameras, ext=None):
    """Return (pa_flat_mask, pb_flat_mask) directly from the demo frame masks.

    This mirrors the GT path used in the test script: it reads the per-camera
    object masks from the demo itself rather than deriving them from flat mask
    post-processing.

    ext: optional pre-created DemoObjectExtractor (pass to avoid per-frame creation).
    """
    from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor

    if ext is None:
        ext = DemoObjectExtractor.from_config(task_name)
    pa_parts, pb_parts = [], []
    for cam in cameras:
        rgb = getattr(frame, f'{cam}_rgb', None)
        h, w = (rgb.shape[:2] if rgb is not None else (128, 128))
        try:
            pb_mask = ext.get_grasp_mask(frame, camera_name=cam)
        except Exception:
            pb_mask = np.zeros((h, w), dtype=np.uint8)
        if ext.target_obj_name:
            try:
                pa_mask = ext.get_target_mask(frame, camera_name=cam)
            except Exception:
                pa_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            pa_mask = np.zeros((h, w), dtype=np.uint8)
        pb_parts.append(pb_mask.reshape(-1).astype(bool))
        pa_parts.append(pa_mask.reshape(-1).astype(bool))
    return np.concatenate(pa_parts), np.concatenate(pb_parts)


def compute_bbox_center(points):
    """Compute center of bounding box from 3D points."""
    if len(points) == 0:
        return None
    points = np.asarray(points)
    if points.shape[0] == 0:
        return None
    bbox_min = np.amin(points, axis=0)
    bbox_max = np.amax(points, axis=0)
    center = (bbox_min + bbox_max) / 2.0
    return center


def compute_velocity(positions, dt=1.0):
    """Compute linear velocity and speed from position trajectory."""
    if len(positions) <= 1:
        return np.array([]), np.array([])
    velocities = np.diff(positions, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)
    return velocities, speeds


def compute_angular_velocity(quaternions, dt=1.0):
    """Compute angular velocity and speed from quaternion trajectory."""
    if len(quaternions) <= 1:
        return np.array([]), np.array([])
    angular_velocities = []
    for i in range(len(quaternions) - 1):
        q1 = R.from_quat(quaternions[i])
        q2 = R.from_quat(quaternions[i + 1])
        q_rel = q2 * q1.inv()
        rotvec = q_rel.as_rotvec()
        angular_vel = rotvec / dt
        angular_velocities.append(angular_vel)
    angular_velocities = np.array(angular_velocities)
    angular_speeds = np.linalg.norm(angular_velocities, axis=1)
    return angular_velocities, angular_speeds


def extract_pointclouds_at_frame(demo, task_name, frame_idx, mask_provider=None, cameras=None):
    """Extract object point clouds at specific frame."""
    obs = demo[frame_idx]
    cameras = cameras if cameras is not None and len(cameras) > 0 else CAMERAS
    frame_obs_dict = extract_obs(obs, cameras, t=frame_idx, prev_action=None)
    frame_pcd_flat, frame_flat_features, frame_flat_mask, _ = flat_pcd_image(
        frame_obs_dict,
        cameras,
        include_mask=True,
    )
    if mask_provider is not None:
        pa_binary_mask, pb_binary_mask = mask_provider(frame_idx, frame_obs_dict, frame_flat_mask)
    else:
        pa_binary_mask, pb_binary_mask = get_gt_flat_masks_for_frame(obs, task_name, cameras)
    moving_points = frame_pcd_flat[0][pb_binary_mask]
    moving_colors = frame_flat_features[0][pb_binary_mask]
    base_points = frame_pcd_flat[0][pa_binary_mask]
    base_colors = frame_flat_features[0][pa_binary_mask]
    return moving_points, moving_colors, base_points, base_colors


def extract_signals_from_demo(demo, task_name, expansion_factor=1.0, mask_provider=None, cameras=None):
    """Extract multi-modal signals: velocities + distances."""
    timesteps, gripper_positions, gripper_orientations = [], [], []
    gripper_to_moving_distances, moving_to_base_distances = [], []

    print("Extracting signals from demo...")
    cameras = cameras if cameras is not None and len(cameras) > 0 else CAMERAS
    # Create extractor once for the no-mask-provider path (avoids per-frame overhead)
    _gt_ext = None
    if mask_provider is None:
        from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
        try:
            _gt_ext = DemoObjectExtractor.from_config(task_name)
        except Exception:
            pass
    for t, obs in enumerate(demo):
        timesteps.append(t)
        gripper_pose = get_gripper_pose_from_obs(obs)
        
        if gripper_pose:
            gripper_positions.append(gripper_pose[0])
            gripper_orientations.append(gripper_pose[1])
        else:
            gripper_positions.append(gripper_positions[-1] if gripper_positions else np.zeros(3))
            gripper_orientations.append(gripper_orientations[-1] if gripper_orientations else np.array([0, 0, 0, 1]))
        
        try:
            frame_obs_dict = extract_obs(obs, cameras, t=t, prev_action=None)
            frame_pcd_flat, _, frame_flat_mask, _ = flat_pcd_image(
                frame_obs_dict,
                cameras,
                include_mask=True,
            )
            if mask_provider is not None:
                pa_binary_mask, pb_binary_mask = mask_provider(t, frame_obs_dict, frame_flat_mask)
            else:
                pa_binary_mask, pb_binary_mask = get_gt_flat_masks_for_frame(obs, task_name, cameras, ext=_gt_ext)
            
            moving_center = compute_bbox_center(frame_pcd_flat[0][pb_binary_mask])
            base_center = compute_bbox_center(frame_pcd_flat[0][pa_binary_mask])
            
            # Compute distances
            if gripper_pose and moving_center is not None:
                gripper_to_moving_distances.append(np.linalg.norm(gripper_pose[0] - moving_center))
            else:
                gripper_to_moving_distances.append(gripper_to_moving_distances[-1] if gripper_to_moving_distances else 1.0)
            
            if moving_center is not None and base_center is not None:
                moving_to_base_distances.append(np.linalg.norm(moving_center - base_center))
            else:
                moving_to_base_distances.append(moving_to_base_distances[-1] if moving_to_base_distances else 1.0)
        except:
            gripper_to_moving_distances.append(gripper_to_moving_distances[-1] if gripper_to_moving_distances else 1.0)
            moving_to_base_distances.append(moving_to_base_distances[-1] if moving_to_base_distances else 1.0)
    
    # Convert to numpy arrays
    gripper_positions = np.array(gripper_positions)
    gripper_orientations = np.array(gripper_orientations)
    
    # Compute velocities
    _, gripper_linear_speeds = compute_velocity(gripper_positions, dt=1.0)
    _, gripper_angular_speeds = compute_angular_velocity(gripper_orientations, dt=1.0)
    
    # Pad velocity arrays to match timestep length (they are 1 shorter due to diff)
    if len(gripper_linear_speeds) > 0:
        gripper_linear_speeds = np.concatenate([gripper_linear_speeds, [gripper_linear_speeds[-1]]])
        gripper_angular_speeds = np.concatenate([gripper_angular_speeds, [gripper_angular_speeds[-1]]])
    else:
        gripper_linear_speeds = np.zeros(len(timesteps))
        gripper_angular_speeds = np.zeros(len(timesteps))
    
    return {
        'timesteps': np.array(timesteps),
        'gripper_positions': gripper_positions,
        'gripper_orientations': gripper_orientations,
        'gripper_linear_speeds': gripper_linear_speeds,
        'gripper_angular_speeds': gripper_angular_speeds,
        'gripper_to_moving_distance': np.array(gripper_to_moving_distances),
        'moving_to_base_distance': np.array(moving_to_base_distances)
    }


def normalize_and_combine_signals(linear, angular, distance, return_magnitude=False):
    """ Normalize and combine signals into 3D array or return combined magnitude. """
    linear_norm = (linear - np.min(linear)) / (np.max(linear) - np.min(linear) + 1e-8)
    angular_norm = (angular - np.min(angular)) / (np.max(angular) - np.min(angular) + 1e-8)
    distance_norm = (distance - np.min(distance)) / (np.max(distance) - np.min(distance) + 1e-8)
    stacked = np.column_stack([linear_norm, angular_norm, distance_norm])
    if return_magnitude:
        return np.linalg.norm(stacked, axis=1)
    return stacked


def detect_change_points_adaptive(signal, min_size=3, pen_factor=DEFAULT_PEN_FACTOR):
    """ Adaptive change point detection: automatically determines the number of change points. """
    if len(signal) < min_size * 2:
        return []
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples = len(signal)
    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        # Adaptive penalty based on sequence length
        if n_samples < 50:
            adjusted_pen = pen_factor * 0.5  # More aggressive for short sequences
        else:
            adjusted_pen = pen_factor
        penalty_value = adjusted_pen * np.log(n_samples) * signal.shape[1]
        change_points = algo.predict(pen=penalty_value)
        if change_points and change_points[-1] == len(signal):
            change_points = change_points[:-1]
        # Retry with lower penalty if no change points found
        if len(change_points) == 0 and n_samples >= 20:
            penalty_value = penalty_value * 0.3
            change_points = algo.predict(pen=penalty_value)
            if change_points and change_points[-1] == len(signal):
                change_points = change_points[:-1]
        return change_points
    except Exception as e:
        print(f"Pelt failed: {e}")
        return []


def find_interaction_start_with_changepoints(keyframe_idx, change_points):
    """ Find interaction start: the last change point before keyframe. """
    if keyframe_idx <= 0:
        return 0
    # Filter change points before keyframe
    valid_change_points = [cp for cp in change_points if cp < keyframe_idx]
    if not valid_change_points:
        return 0
    # Return the closest change point before keyframe
    return max(valid_change_points)


def compute_intervals_for_demo(demo, task_name: str, expansion_factor: float = 1.0,
                               use_change_detection: bool = True,
                               pen_factor: Optional[float] = None, keyframes: Dict = None,
                               mask_provider=None, cameras=None,
                               use_gt_masks: bool = False) -> Dict:
    """ Compute interaction intervals for a single demo. """
    if pen_factor is None:
        pen_factor = get_interaction_interval_pen_factor(task_name)

    # Get keyframes
    if keyframes is None:
        keyframes = get_pick_place_keyframes(demo, task_name=task_name)
    pick_frame = keyframes['pick']
    place_frame = keyframes['place']
    cameras = cameras if cameras is not None and len(cameras) > 0 else CAMERAS
    if use_gt_masks and mask_provider is None:
        mask_provider = lambda t, _obs_dict, _flat_mask: get_gt_flat_masks_for_frame(demo[t], task_name, cameras)
    
    # Extract signals
    signals = extract_signals_from_demo(
        demo,
        task_name,
        expansion_factor,
        mask_provider=mask_provider,
        cameras=cameras,
    )
    if use_change_detection:
        # GRASP PHASE
        grasp_end = min(pick_frame + 1, len(signals['timesteps']))
        grasp_combined = normalize_and_combine_signals(
            signals['gripper_linear_speeds'][:grasp_end],
            signals['gripper_angular_speeds'][:grasp_end],
            signals['gripper_to_moving_distance'][:grasp_end]
        )
        grasp_change_points = detect_change_points_adaptive(grasp_combined, min_size=3, pen_factor=pen_factor)
        grasp_start = find_interaction_start_with_changepoints(pick_frame, grasp_change_points)
        grasp_start = max(0, grasp_start)
        grasp_interval = pick_frame - grasp_start
        
        # MANIPULATION PHASE
        manip_end = min(place_frame + 1, len(signals['timesteps']))
        manip_start_search = max(pick_frame, 0)
        manip_combined = normalize_and_combine_signals(
            signals['gripper_linear_speeds'][manip_start_search:manip_end],
            signals['gripper_angular_speeds'][manip_start_search:manip_end],
            signals['moving_to_base_distance'][manip_start_search:manip_end]
        )
        manip_change_points = detect_change_points_adaptive(manip_combined, min_size=3, pen_factor=pen_factor)
        manip_change_points = [cp + manip_start_search for cp in manip_change_points]
        manip_start = find_interaction_start_with_changepoints(place_frame, manip_change_points)
        manip_start = max(pick_frame, manip_start)
        manip_interval = place_frame - manip_start
        
    else:
        # Simple method without change detection
        grasp_start = max(0, pick_frame - 10)
        grasp_interval = pick_frame - grasp_start
        manip_start = max(pick_frame, place_frame - 10)
        manip_interval = place_frame - manip_start
    return {
        'grasp_interval': grasp_interval,
        'manip_interval': manip_interval,
        'grasp_start': grasp_start,
        'manip_start': manip_start,
        'pick_frame': pick_frame,
        'place_frame': place_frame
    }


# ===============Visualization Functions=================
def _prepare_o3d_colors(raw_colors, num_points):
    """Use raw point colors directly; normalize to [0, 1] for Open3D."""
    colors = np.asarray(raw_colors)
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != num_points:
        return np.full((num_points, 3), 0.75, dtype=np.float32)
    colors = colors.astype(np.float32)
    if np.max(colors) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _load_gripper_template():
    """Load gripper point cloud template from assets/gripper.pkl once."""
    global _GRIPPER_TEMPLATE
    if _GRIPPER_TEMPLATE is not None:
        return _GRIPPER_TEMPLATE

    if not os.path.exists(GRIPPER_PKL_PATH):
        print(f"Warning: gripper asset not found: {GRIPPER_PKL_PATH}")
        _GRIPPER_TEMPLATE = None
        return None

    try:
        import pickle
        with open(GRIPPER_PKL_PATH, 'rb') as f:
            gripper_data = pickle.load(f)

        template_points = gripper_data.get('points_vox', gripper_data.get('points'))
        template_colors = gripper_data.get('colors_vox', gripper_data.get('colors'))
        if template_points is None or len(template_points) == 0:
            _GRIPPER_TEMPLATE = None
            return None

        template_points = np.asarray(template_points, dtype=np.float32)
        template_points = template_points - np.mean(template_points, axis=0, keepdims=True)
        template_colors = _prepare_o3d_colors(template_colors, len(template_points))
        _GRIPPER_TEMPLATE = {'points': template_points, 'colors': template_colors}
        return _GRIPPER_TEMPLATE
    except Exception as e:
        print(f"Warning: failed to load gripper asset: {e}")
        _GRIPPER_TEMPLATE = None
        return None


def _load_o3d_camera_parameters():
    """Load Open3D camera parameters from JSON config if available."""
    if not os.path.exists(O3D_CAMERA_CONFIG_PATH):
        print(f"Warning: camera config not found: {O3D_CAMERA_CONFIG_PATH}")
        return None
    try:
        return o3d.io.read_pinhole_camera_parameters(O3D_CAMERA_CONFIG_PATH)
    except Exception as e:
        print(f"Warning: failed to read camera config: {e}")
        return None


def _make_gripper_pcd(gripper_pose, scene_center):
    """Create transformed gripper point cloud from template + pose."""
    if gripper_pose is None:
        return None
    template = _load_gripper_template()
    if template is None:
        return None

    gripper_points = template['points']
    gripper_colors = template['colors']
    rotation_matrix = R.from_quat(gripper_pose[1]).as_matrix()
    # Move the gripper backward by 5 cm along its local +z axis.
    gripper_translation = gripper_pose[0] - rotation_matrix[:, 2] * GRIPPER_RETREAT_M
    transformed_points = gripper_points @ rotation_matrix.T + gripper_translation
    transformed_points = transformed_points - scene_center

    gripper_pcd = o3d.geometry.PointCloud()
    gripper_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    gripper_pcd.colors = o3d.utility.Vector3dVector(gripper_colors)
    return gripper_pcd


def _make_object_pcd(points, colors, scene_center):
    """Build object point cloud using raw per-point RGB colors."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points - scene_center)
    pcd.colors = o3d.utility.Vector3dVector(_prepare_o3d_colors(colors, len(points)))
    return pcd


def _draw_single_state_window(moving_points, moving_colors,
                              base_points, base_colors,
                              gripper_pose, window_name, left=50, top=50):
    """Render one state with camera loaded from JSON config."""
    point_sets = []
    for pts in [moving_points, base_points]:
        arr = np.asarray(pts)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0:
            point_sets.append(arr)
    if len(point_sets) == 0:
        print(f"Warning: no valid points for {window_name}")
        return

    scene_center = np.mean(np.vstack(point_sets), axis=0)
    geometries = [
        _make_object_pcd(moving_points, moving_colors, scene_center),
        _make_object_pcd(base_points, base_colors, scene_center),
        _make_gripper_pcd(gripper_pose, scene_center),
    ]
    geometries = [g for g in geometries if g is not None]
    if len(geometries) == 0:
        print(f"Warning: no renderable geometry for {window_name}")
        return

    width, height = O3D_WINDOW_SIZE
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, left=left, top=top)
    for geo in geometries:
        vis.add_geometry(geo)

    camera_params = _load_o3d_camera_parameters()
    if camera_params is not None:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    vis.run()
    vis.destroy_window()


def plot_3d_pointcloud_visualization(demo, task_name, result, mask_provider=None):
    """
    For each interval, render start and end in separate windows using the same camera config.
    """
    episode_idx = result['episode_idx']
    grasp_start = result['grasp_start']
    manip_start = result['manip_start']
    pick_frame = result['pick_frame']
    place_frame = result['place_frame']

    # Extract interval boundary point clouds.
    grasp_start_moving, grasp_start_moving_c, grasp_start_base, grasp_start_base_c = extract_pointclouds_at_frame(demo, task_name, grasp_start, mask_provider=mask_provider)
    grasp_end_moving, grasp_end_moving_c, grasp_end_base, grasp_end_base_c = extract_pointclouds_at_frame(demo, task_name, pick_frame, mask_provider=mask_provider)
    manip_start_moving, manip_start_moving_c, manip_start_base, manip_start_base_c = extract_pointclouds_at_frame(demo, task_name, manip_start, mask_provider=mask_provider)
    manip_end_moving, manip_end_moving_c, manip_end_base, manip_end_base_c = extract_pointclouds_at_frame(demo, task_name, place_frame, mask_provider=mask_provider)

    # Get gripper poses for interval boundaries.
    grasp_start_gripper_pose = get_gripper_pose_from_obs(demo[grasp_start])
    grasp_end_gripper_pose = get_gripper_pose_from_obs(demo[pick_frame])
    manip_start_gripper_pose = get_gripper_pose_from_obs(demo[manip_start])
    manip_end_gripper_pose = get_gripper_pose_from_obs(demo[place_frame])

    print(f"\nDisplaying 3D point cloud visualization...")
    print(f"Grasp interval: [{grasp_start} -> {pick_frame}]")
    _draw_single_state_window(
        grasp_start_moving, grasp_start_moving_c,
        grasp_start_base, grasp_start_base_c,
        grasp_start_gripper_pose,
        window_name=f'{task_name} Demo {episode_idx} - Grasp Start (t={grasp_start})',
        left=50,
        top=50
    )
    _draw_single_state_window(
        grasp_end_moving, grasp_end_moving_c,
        grasp_end_base, grasp_end_base_c,
        grasp_end_gripper_pose,
        window_name=f'{task_name} Demo {episode_idx} - Grasp End (t={pick_frame})',
        left=1400,
        top=50
    )

    print(f"Manip interval: [{manip_start} -> {place_frame}]")
    _draw_single_state_window(
        manip_start_moving, manip_start_moving_c,
        manip_start_base, manip_start_base_c,
        manip_start_gripper_pose,
        window_name=f'{task_name} Demo {episode_idx} - Manip Start (t={manip_start})',
        left=50,
        top=50
    )
    _draw_single_state_window(
        manip_end_moving, manip_end_moving_c,
        manip_end_base, manip_end_base_c,
        manip_end_gripper_pose,
        window_name=f'{task_name} Demo {episode_idx} - Manip End (t={place_frame})',
        left=1400,
        top=50
    )


def plot_interaction_analysis(result: Dict, task_name: str, output_dir: str = './analysis_results'):
    """
    Plot interaction analysis with signals, change points, and detected intervals.
    
    Creates two plots:
    1. Grasp phase: obs -> pick (gripper-object interaction)
    2. Manipulation phase: pick -> place (object-base interaction)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    signals = result['signals']
    timesteps = signals['timesteps']
    episode_idx = result['episode_idx']
    
    pick_frame = result['pick_frame']
    place_frame = result['place_frame']
    grasp_start = result['grasp_start']
    manip_start = result['manip_start']

    # Thicker line defaults for clearer visualization.
    lw_signal = 2.2
    lw_combined = 3.0
    lw_marker = 2.6
    lw_cp = 1.8

    def save_paper_full_demo_plot(out_path: str):
        """Save one paper figure over the full demo timeline in a unified coordinate frame."""
        full_timesteps = timesteps
        linear_full = signals['gripper_linear_speeds']
        angular_full = signals['gripper_angular_speeds']

        # Use grasp distance before pick and manip distance from pick onwards.
        distance_full = np.array(signals['gripper_to_moving_distance'], copy=True)
        if pick_frame < len(distance_full):
            distance_full[pick_frame:] = signals['moving_to_base_distance'][pick_frame:]

        combined_full = normalize_and_combine_signals(
            linear_full,
            angular_full,
            distance_full,
            return_magnitude=True
        )

        fig, ax_linear = plt.subplots(1, 1, figsize=(36, 4))
        ax_angular = ax_linear.twinx()
        ax_distance = ax_linear.twinx()
        ax_combined = ax_linear.twinx()

        # Use calmer signal colors and emphasize the combined curve.
        ax_linear.plot(full_timesteps, linear_full, linestyle='--', color='b', linewidth=lw_signal)
        ax_angular.plot(full_timesteps, angular_full, linestyle='--', color='g', linewidth=lw_signal)
        ax_distance.plot(full_timesteps, distance_full, linestyle='--', color='c', linewidth=lw_signal)
        ax_combined.plot(full_timesteps, combined_full, linestyle='-', color='purple', linewidth=lw_combined + 1.0)

        # Keep interval interior light yellow, make interval boundaries bolder, and switch change points to black.
        interval_boundaries = [grasp_start, pick_frame, manip_start, place_frame]
        for t in interval_boundaries:
            ax_linear.axvline(t, color='black', linestyle='--', linewidth=lw_marker + 1.4)
        all_change_points = sorted(set(result.get('change_points_grasp', []) + result.get('change_points_manip', [])))
        for cp in all_change_points:
            if full_timesteps[0] <= cp <= full_timesteps[-1]:
                ax_linear.axvline(cp, color='black', linestyle='--', linewidth=lw_cp, alpha=0.8)

        ax_linear.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.16)
        ax_linear.axvspan(manip_start, place_frame, color='yellow', alpha=0.16)

        ax_linear.grid(True, alpha=0.3)
        ax_linear.set_xlim(full_timesteps[0], full_timesteps[-1])

        # Keep paper style clean: no titles/legends/axis labels/ticks.
        for ax in [ax_linear, ax_angular, ax_distance, ax_combined]:
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, labelright=False, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Combined paper plot saved to {out_path}")
        plt.close()
    
    # === PLOT 1: Grasp Phase ===
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    grasp_end = pick_frame
    grasp_timesteps = timesteps[:grasp_end+1]
    
    # Linear speed
    ax = axes[0]
    ax.plot(grasp_timesteps, signals['gripper_linear_speeds'][:grasp_end+1], 'b-', linewidth=lw_signal, label='Linear Speed')
    ax.axvline(grasp_start, color='red', linestyle='--', linewidth=lw_marker, label='Grasp Start')
    ax.axvline(pick_frame, color='red', linestyle='--', linewidth=lw_marker, label='Pick Frame')
    ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3, label='Interaction Interval')
    if 'change_points_grasp' in result and result['change_points_grasp']:
        for i, cp in enumerate(result['change_points_grasp']):
            if cp <= grasp_end:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7, 
                          label='Change Points' if i == 0 else '')
                ax.text(cp, ax.get_ylim()[1]*0.95, f'{cp}', fontsize=8, ha='center', color='green')
    ax.set_ylabel('Linear Speed (m/s)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Grasp Phase Analysis - {task_name} (Demo {episode_idx})', fontsize=13, fontweight='bold')
    
    # Angular speed
    ax = axes[1]
    ax.plot(grasp_timesteps, signals['gripper_angular_speeds'][:grasp_end+1], 'g-', linewidth=lw_signal, label='Angular Speed')
    ax.axvline(grasp_start, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvline(pick_frame, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3)
    if 'change_points_grasp' in result and result['change_points_grasp']:
        for cp in result['change_points_grasp']:
            if cp <= grasp_end:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7)
    ax.set_ylabel('Angular Speed (rad/s)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Distance: Gripper to Moving Object
    ax = axes[2]
    ax.plot(grasp_timesteps, signals['gripper_to_moving_distance'][:grasp_end+1], 'c-', linewidth=lw_signal, label='Gripper-Object Distance')
    ax.axvline(grasp_start, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvline(pick_frame, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3)
    if 'change_points_grasp' in result and result['change_points_grasp']:
        for cp in result['change_points_grasp']:
            if cp <= grasp_end:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7)
    ax.set_ylabel('Distance (m)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Combined signal magnitude
    ax = axes[3]
    grasp_combined = normalize_and_combine_signals(
        signals['gripper_linear_speeds'][:grasp_end+1],
        signals['gripper_angular_speeds'][:grasp_end+1],
        signals['gripper_to_moving_distance'][:grasp_end+1],
        return_magnitude=True
    )
    
    ax.plot(grasp_timesteps, grasp_combined, 'purple', linewidth=lw_combined, label='Combined Signal')
    ax.axvline(grasp_start, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvline(pick_frame, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3)
    if 'change_points_grasp' in result and result['change_points_grasp']:
        for cp in result['change_points_grasp']:
            if cp <= grasp_end:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7)
    ax.set_ylabel('Signal Magnitude', fontsize=11)
    ax.set_xlabel('Frame', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    grasp_plot_path = os.path.join(output_dir, f'{task_name}_demo{episode_idx}_grasp_phase.png')
    plt.savefig(grasp_plot_path, dpi=150, bbox_inches='tight')
    print(f"Grasp phase plot saved to {grasp_plot_path}")
    plt.close()

    # === PLOT 2: Manipulation Phase ===
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    manip_timesteps = timesteps[pick_frame:place_frame+1]
    manip_offset = pick_frame
    
    # Linear speed
    ax = axes[0]
    ax.plot(manip_timesteps, signals['gripper_linear_speeds'][pick_frame:place_frame+1], 'b-', linewidth=lw_signal, label='Linear Speed')
    ax.axvline(manip_start, color='red', linestyle='--', linewidth=lw_marker, label='Manip Start')
    ax.axvline(place_frame, color='red', linestyle='--', linewidth=lw_marker, label='Place Frame')
    ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3, label='Interaction Interval')
    if 'change_points_manip' in result and result['change_points_manip']:
        for i, cp in enumerate(result['change_points_manip']):
            if pick_frame <= cp <= place_frame:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7,
                          label='Change Points' if i == 0 else '')
                ax.text(cp, ax.get_ylim()[1]*0.95, f'{cp}', fontsize=8, ha='center', color='green')
    ax.set_ylabel('Linear Speed (m/s)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Manipulation Phase Analysis - {task_name} (Demo {episode_idx})', fontsize=13, fontweight='bold')
    
    # Angular speed
    ax = axes[1]
    ax.plot(manip_timesteps, signals['gripper_angular_speeds'][pick_frame:place_frame+1], 'g-', linewidth=lw_signal, label='Angular Speed')
    ax.axvline(manip_start, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvline(place_frame, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3)
    if 'change_points_manip' in result and result['change_points_manip']:
        for cp in result['change_points_manip']:
            if pick_frame <= cp <= place_frame:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7)
    ax.set_ylabel('Angular Speed (rad/s)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Distance: Moving Object to Base
    ax = axes[2]
    ax.plot(manip_timesteps, signals['moving_to_base_distance'][pick_frame:place_frame+1], 'c-', linewidth=lw_signal, label='Object-Base Distance')
    ax.axvline(manip_start, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvline(place_frame, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3)
    if 'change_points_manip' in result and result['change_points_manip']:
        for cp in result['change_points_manip']:
            if pick_frame <= cp <= place_frame:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7)
    ax.set_ylabel('Distance (m)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Combined signal magnitude
    ax = axes[3]
    manip_combined = normalize_and_combine_signals(
        signals['gripper_linear_speeds'][pick_frame:place_frame+1],
        signals['gripper_angular_speeds'][pick_frame:place_frame+1],
        signals['moving_to_base_distance'][pick_frame:place_frame+1],
        return_magnitude=True
    )
    
    ax.plot(manip_timesteps, manip_combined, 'purple', linewidth=lw_combined, label='Combined Signal')
    ax.axvline(manip_start, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvline(place_frame, color='red', linestyle='--', linewidth=lw_marker)
    ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3)
    if 'change_points_manip' in result and result['change_points_manip']:
        for cp in result['change_points_manip']:
            if pick_frame <= cp <= place_frame:
                ax.axvline(cp, color='green', linestyle=':', linewidth=lw_cp, alpha=0.7)
    ax.set_ylabel('Signal Magnitude', fontsize=11)
    ax.set_xlabel('Frame', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    manip_plot_path = os.path.join(output_dir, f'{task_name}_demo{episode_idx}_manip_phase.png')
    plt.savefig(manip_plot_path, dpi=150, bbox_inches='tight')
    print(f"Manipulation phase plot saved to {manip_plot_path}")
    plt.close()

    paper_plot_path = os.path.join(output_dir, f'{task_name}_demo{episode_idx}_paper.png')
    save_paper_full_demo_plot(paper_plot_path)


def compute_multi_demo_interaction_intervals(root_dir: str, task_name: str, var_num: int = 0,
                                  num_demos: int = 10, expansion_factor: float = 1.0,
                                  use_change_detection: bool = True,
                                  pen_factor: Optional[float] = None,
                                  plot_results: bool = False, viz3d: bool = False,
                                  use_gt_masks: bool = False,
                                  output_dir: str = './analysis_results') -> Dict:
    """ Compute average interaction intervals across multiple demos. """
    if pen_factor is None:
        pen_factor = get_interaction_interval_pen_factor(task_name)

    print(f"\n{'='*60}")
    print(f"Computing Interaction Intervals with Change Point Detection")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"Variation: {var_num}")
    print(f"Number of demos: {num_demos}")
    print(f"Expansion factor: {expansion_factor}")
    print(f"Use change detection: {use_change_detection}")
    print(f"Penalty factor: {pen_factor}")
    print(f"Use GT masks: {use_gt_masks}")
    print(f"{'='*60}\n")
    
    grasp_intervals = []
    manip_intervals = []
    valid_demos = []
    data_path = os.path.join(root_dir, f'{task_name}/variation{var_num}/episodes')
    
    for episode_idx in range(num_demos):
        print(f"Processing demo {episode_idx + 1}/{num_demos}...", end=' ')
        try:
            demo= get_stored_demo(data_path=data_path, index=episode_idx, init_matrix=False)
            result = compute_intervals_for_demo(
                demo,
                task_name,
                expansion_factor,
                use_change_detection,
                pen_factor,
                cameras=CAMERAS,
                use_gt_masks=use_gt_masks,
            ) 
            if result is None:
                print("✗ Failed")
                continue
            
            # Add extra info for visualization
            result['total_frames'] = len(demo)
            result['episode_idx'] = episode_idx
            grasp_intervals.append(result['grasp_interval'])
            manip_intervals.append(result['manip_interval'])
            valid_demos.append(episode_idx)
            print(f"✓ Grasp: {result['grasp_interval']} frames, Manip: {result['manip_interval']} frames")
            
            # Generate visualizations if requested
            if plot_results or viz3d:
                visual_mask_provider = (
                    (lambda t, _obs_dict, _flat_mask: get_gt_flat_masks_for_frame(demo[t], task_name, CAMERAS))
                    if use_gt_masks else None
                )
                signals = extract_signals_from_demo(
                    demo,
                    task_name,
                    expansion_factor,
                    mask_provider=visual_mask_provider,
                    cameras=CAMERAS,
                )
                result['signals'] = signals
                
                # Compute change points for visualization
                if use_change_detection:
                    pick_frame = result['pick_frame']
                    place_frame = result['place_frame']
                    
                    grasp_end = min(pick_frame + 1, len(signals['timesteps']))
                    grasp_combined = normalize_and_combine_signals(
                        signals['gripper_linear_speeds'][:grasp_end],
                        signals['gripper_angular_speeds'][:grasp_end],
                        signals['gripper_to_moving_distance'][:grasp_end]
                    )
                    result['change_points_grasp'] = detect_change_points_adaptive(grasp_combined, min_size=3, pen_factor=pen_factor)
                    
                    manip_end = min(place_frame + 1, len(signals['timesteps']))
                    manip_start_search = max(pick_frame, 0)
                    manip_combined = normalize_and_combine_signals(
                        signals['gripper_linear_speeds'][manip_start_search:manip_end],
                        signals['gripper_angular_speeds'][manip_start_search:manip_end],
                        signals['moving_to_base_distance'][manip_start_search:manip_end]
                    )
                    manip_change_points = detect_change_points_adaptive(manip_combined, min_size=3, pen_factor=pen_factor)
                    result['change_points_manip'] = [cp + manip_start_search for cp in manip_change_points]
                else:
                    result['change_points_grasp'] = []
                    result['change_points_manip'] = []
                
                if plot_results:
                    plot_interaction_analysis(result, task_name, output_dir)
                
                if viz3d:
                    plot_3d_pointcloud_visualization(
                        demo,
                        task_name,
                        result,
                        mask_provider=visual_mask_provider,
                    )
                    
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    if len(grasp_intervals) == 0:
        print("\nNo valid demos found!")
        return None
    
    # Compute statistics
    grasp_intervals = np.array(grasp_intervals)
    manip_intervals = np.array(manip_intervals)
    
    results = {
        'grasp_intervals': grasp_intervals.tolist(),
        'manip_intervals': manip_intervals.tolist(),
        'avg_grasp_interval': float(np.mean(grasp_intervals)),
        'avg_manip_interval': float(np.mean(manip_intervals)),
        'std_grasp_interval': float(np.std(grasp_intervals)),
        'std_manip_interval': float(np.std(manip_intervals)),
        'median_grasp_interval': float(np.median(grasp_intervals)),
        'median_manip_interval': float(np.median(manip_intervals)),
        'min_grasp_interval': int(np.min(grasp_intervals)),
        'max_grasp_interval': int(np.max(grasp_intervals)),
        'min_manip_interval': int(np.min(manip_intervals)),
        'max_manip_interval': int(np.max(manip_intervals)),
        'num_valid_demos': len(valid_demos),
        'valid_demo_indices': valid_demos,
        'task_name': task_name,
        'expansion_factor': expansion_factor
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"Valid demos: {results['num_valid_demos']}/{num_demos}")
    print(f"\nGrasp Interaction Interval (frames):")
    print(f"  Mean:   {results['avg_grasp_interval']:.2f} ± {results['std_grasp_interval']:.2f}")
    print(f"  Median: {results['median_grasp_interval']:.1f}")
    print(f"  Range:  [{results['min_grasp_interval']}, {results['max_grasp_interval']}]")
    print(f"\nManipulation Interaction Interval (frames):")
    print(f"  Mean:   {results['avg_manip_interval']:.2f} ± {results['std_manip_interval']:.2f}")
    print(f"  Median: {results['median_manip_interval']:.1f}")
    print(f"  Range:  [{results['min_manip_interval']}, {results['max_manip_interval']}]")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute interaction intervals using change point detection'
    )
    parser.add_argument('--task', type=str, required=True, default='close_jar',
                       help='Task name')
    parser.add_argument('--num_demos', type=int, default=1,
                       help='Number of demos to analyze (default: 10)')
    parser.add_argument('--var', type=int, default=0,
                       help='Task variation number (default: 0)')
    parser.add_argument('--root_dir', type=str, default='../../data/rlbench_data',
                       help='Root directory for RLBench data')
    parser.add_argument('--expansion_factor', type=float, default=1.0,
                       help='Bounding box expansion factor (default: 1.0)')
    parser.add_argument('--no_change_detection', action='store_true',
                       help='Disable change point detection (use simple bbox method)')
    parser.add_argument('--pen_factor', type=float, default=None,
                       help='Override the configured penalty factor for adaptive detection')
    parser.add_argument('--plot', action='store_true',
                       help='Generate 2D plots for each demo')
    parser.add_argument('--viz3d', action='store_true',
                       help='Show interactive 3D point cloud visualization for each demo')
    parser.add_argument('--use_gt_masks', action='store_true',
                       help='Load per-frame GT masks directly from demos instead of derived masks')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory for plots and results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    results = compute_multi_demo_interaction_intervals(
        root_dir=args.root_dir,
        task_name=args.task,
        var_num=args.var,
        num_demos=args.num_demos,
        expansion_factor=args.expansion_factor,
        use_change_detection=not args.no_change_detection,
        pen_factor=args.pen_factor,
        plot_results=args.plot,
        viz3d=args.viz3d,
        use_gt_masks=args.use_gt_masks,
        output_dir=args.output_dir
    )
    
    # Save to JSON if output path specified
    if results and args.output:
        import json
        # Remove non-serializable data
        results_to_save = {k: v for k, v in results.items() if k != 'signals'}
        with open(args.output, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
