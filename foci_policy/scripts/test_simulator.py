"""
Test FOCI Actor on RLBench simulator
"""
import os

# Reduce CUDA memory fragmentation during long FoundationPose evaluations.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import time
import argparse
import warnings
import open3d as o3d
import math
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.utils import get_stored_demo
from termcolor import colored
from foci_policy.utils.transform_utils import Transform, Rotation, to_o3d_pcd
from foci_policy.model.foci_actor import FOCIActor
from foci_policy.pose_estimator.object_pose_seq_extractor import (
    get_object_poses_live,
    extract_object_pcds_from_obs,
)
from foci_policy.pose_estimator.foundation_pose_utils import extract_poses_from_observation
from foci_policy.config.config_utils import (
    get_dataloader_config,
    get_task_class,
    get_task_language,
)
from utils_rlbench.process_demo import get_pick_place_keyframes
from foci_policy.model.foci_dataloader import AddLanguageEmbedding
from foci_policy.utils.compute_interaction_intervals import compute_intervals_for_demo
from torch_geometric.data import Data


warnings.filterwarnings("ignore")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(_SCRIPT_DIR, '../../data/rlbench_data'))
CONFIG_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '../config'))
GRASP_CKPT = os.path.normpath(os.path.join(_SCRIPT_DIR, '../checkpoints/foci/simu/grasp/best_model.pth'))
MANIP_CKPT = os.path.normpath(os.path.join(_SCRIPT_DIR, '../checkpoints/foci/simu/manip/best_model.pth'))
DEFAULT_DEVICE = str(get_dataloader_config().get('device', 'cuda:0'))


def restore_coppeliasim_qt_environment():
    """Undo Qt plugin overrides applied by GUI-enabled OpenCV wheels."""
    coppeliasim_root = os.environ.get('COPPELIASIM_ROOT')
    if not coppeliasim_root:
        return
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = coppeliasim_root
    if '/cv2/qt/' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)


parser = argparse.ArgumentParser(description='Test FOCI Actor on RLBench')
parser.add_argument('--task', type=str, default='close_jar', help='Task name')
parser.add_argument('--var', type=int, default=0, help='Task variation')
parser.add_argument('--n_tests', type=int, default=5, help='Number of test episodes')
parser.add_argument('--start_test', type=int, default=5, help='Starting episode index')
parser.add_argument('--disp', action='store_true', default=False, help='Display visualization')
parser.add_argument('--device', type=str, default=None,
                    help='CUDA device override, e.g. cuda:1 or 1 (default: dataloader.yaml)')
parser.add_argument('--voxel_size', type=float, default=0.004, help='Voxel size for downsampling')
parser.add_argument('--num_points', type=int, default=2048, help='Number of points')
parser.add_argument('--traj_length', type=int, default=3, help='Trajectory length for expert mode (default: 3)')
parser.add_argument('--pose_method', type=str, default='fp',
                    choices=['gt', 'fp', 'foundation_pose'],
                    help='Pose estimation method: gt (ground truth), fp/foundation_pose (Foundation Pose)')
parser.add_argument('--mesh_dir', type=str, default=None,
                    help='Mesh directory for Foundation Pose (default: ../assets/RLBench_mesh)')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Visualize failures and save FoundationPose debug artifacts')
parser.add_argument('--record', action='store_true', default=False,
                    help='Record simulator video for each test episode')
parser.add_argument('--save_dir', type=str, default='./debug/foci_videos/',
                    help='Directory to save recorded videos')
parser.add_argument('--record_fps', type=float, default=20.0,
                    help='FPS used when encoding recorded videos (20 matches RLBench dt=0.05)')
parser.add_argument('--record_width', type=int, default=1280,
                    help='Output video width')
parser.add_argument('--record_height', type=int, default=720,
                    help='Output video height')
parser.add_argument('--record_camera_speed', type=float, default=0.0,
                    help='Cinematic camera orbit speed in radians per sim step')
parser.add_argument('--add_noise', type=str, default='none', 
                    choices=['none', 'mild', 'medium', 'hard'],
                    help='Add noise to Foundation Pose estimates: none, mild (±0.5cm, ±2°), medium (±1cm, ±5°), hard (±2cm, ±10°)')
parser.add_argument('--actor', type=str, default='foci', choices=['foci', 'expert'],
                    help='Actor type: foci (FOCI Actor), or expert (Expert trajectories from demo)')

args = parser.parse_args()
args.device = args.device or DEFAULT_DEVICE
if args.device.isdigit():
    args.device = f'cuda:{args.device}'
if args.device != 'cuda' and not args.device.startswith('cuda:'):
    parser.error(f"Invalid CUDA device '{args.device}'; use cuda:N or N")
if args.pose_method in ['fp', 'foundation_pose']:
    if args.mesh_dir is None:
        args.mesh_dir = os.path.normpath(os.path.join(_SCRIPT_DIR, '../assets/RLBench_mesh'))
        print(f"Using default mesh directory: {args.mesh_dir}")


def get_gripper_pose(obs):
    """Extract gripper pose from observation"""
    trans = obs.gripper_pose[:3]
    quat = obs.gripper_pose[3:]
    quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
    if quat[-1] < 0:
        quat = -quat
    return Transform(rotation=Rotation.from_quat(quat), translation=trans)


def get_object_poses(obs, task_name='close_jar', task=None, stage=1):
    """Extract object poses from live simulator via DemoObjectExtractor."""
    return get_object_poses_live(task_name, task=task, stage=stage)


def extract_point_clouds_from_obs(obs, task_name='close_jar', voxel_size=0.004, num_points=2048,
                                   moving_mask_index=None, variation=0, task=None):
    """Extract object point clouds from current live observation.

    Uses DemoObjectExtractor + Shape.get_handle() for the default path.
    moving_mask_index: if not None, signals that stage=2 objects (grasp_2/target_2)
        should be extracted using DemoObjectExtractor — used for stacking tasks.
    """
    if moving_mask_index is not None:
        # Second-stage extraction for multi-object stacking tasks.
        # Uses DemoObjectExtractor stage=2 to get grasp_2/target_2 masks via live handles.
        _CAMS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor as _DExt
        _ext2 = _DExt.from_config(task_name, stage=2)
        all_points, all_colors = [], []
        pa_bin_parts, pb_bin_parts = [], []
        visibility = []
        for cam in _CAMS:
            pcd      = getattr(obs, f'{cam}_point_cloud').reshape(-1, 3)
            rgb      = getattr(obs, f'{cam}_rgb').reshape(-1, 3)
            cam_mask = getattr(obs, f'{cam}_mask')
            valid    = ~np.all(pcd == 0, axis=1)
            all_points.append(pcd[valid])
            all_colors.append(rgb[valid] / 255.0)
            _n  = int(np.prod(cam_mask.shape[:2]))
            _lm = _ext2.get_masks_live(cam_mask, task=task)
            pa_mask = _lm.get('target', np.zeros(_n, dtype=np.uint8)).reshape(-1).astype(bool)
            pb_mask = _lm.get('grasp',  np.zeros(_n, dtype=np.uint8)).reshape(-1).astype(bool)
            pa_valid = pa_mask[valid]
            pb_valid = pb_mask[valid]
            pa_bin_parts.append(pa_valid)
            pb_bin_parts.append(pb_valid)
            visibility.append(
                f"{cam}: target={int(pa_mask.sum())}/{int(pa_valid.sum())}, "
                f"grasp={int(pb_mask.sum())}/{int(pb_valid.sum())}"
            )
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        pa_bin = np.concatenate(pa_bin_parts)
        pb_bin = np.concatenate(pb_bin_parts)
        pa_pts_raw, pa_cols_raw = all_points[pa_bin], all_colors[pa_bin]
        pb_pts_raw, pb_cols_raw = all_points[pb_bin], all_colors[pb_bin]
        if len(pa_pts_raw) == 0 or len(pb_pts_raw) == 0:
            raise ValueError(
                "No valid stage-2 object points "
                f"(target={len(pa_pts_raw)}, grasp={len(pb_pts_raw)}). "
                "Per-camera mask/valid-depth pixels: " + "; ".join(visibility)
            )
        pa_pcd = to_o3d_pcd(pa_pts_raw, pa_cols_raw).voxel_down_sample(voxel_size)
        pb_pcd = to_o3d_pcd(pb_pts_raw, pb_cols_raw).voxel_down_sample(voxel_size)
        def _sample(pcd):
            n = len(pcd.points)
            idx = np.random.choice(n, num_points, replace=(n < num_points))
            return np.asarray(pcd.points)[idx], np.asarray(pcd.colors)[idx]
        pa_points, pa_colors = _sample(pa_pcd)
        pb_points = np.asarray(pb_pcd.points)
        pb_colors = np.asarray(pb_pcd.colors)
        n = len(pb_pcd.points)
        idx = np.random.choice(n, num_points, replace=(n < num_points))
        pb_points = pb_points[idx]
        pb_colors = pb_colors[idx]
        return pa_points, pa_colors, pb_points, pb_colors

    return extract_object_pcds_from_obs(
        obs, task_name, variation=variation,
        voxel_size=voxel_size, num_points=num_points, task=task,
    )


def add_noise_to_pose(pose_matrix, noise_level='none'):
    """ Add noise to pose matrix (translation and rotation) """
    if noise_level == 'none':
        return pose_matrix
    
    # Define noise parameters (max absolute values)
    noise_params = {
        'mild': {'trans': 0.005, 'rot': 2.0},      # ±0.5cm, ±2°
        'medium': {'trans': 0.01, 'rot': 5.0},     # ±1cm, ±5°
        'hard': {'trans': 0.02, 'rot': 10.0}       # ±2cm, ±10°
    }
    
    if noise_level not in noise_params:
        return pose_matrix
    
    params = noise_params[noise_level]
    
    # Add translation noise (uniform sampling from -max to +max)
    trans_noise = np.random.uniform(-params['trans'], params['trans'], size=3)
    
    # Add rotation noise (uniform sampling from -max to +max degrees)
    rot_noise_deg = np.random.uniform(-params['rot'], params['rot'], size=3)
    rot_noise_rad = np.deg2rad(rot_noise_deg)
    
    # Create rotation matrix from Euler angles (XYZ convention)
    from scipy.spatial.transform import Rotation as R
    rot_noise_matrix = R.from_euler('xyz', rot_noise_rad).as_matrix()
    
    # Apply noise to pose
    noisy_pose = pose_matrix.copy()
    noisy_pose[:3, 3] += trans_noise  # Add translation noise
    noisy_pose[:3, :3] = noisy_pose[:3, :3] @ rot_noise_matrix  # Add rotation noise
    
    return noisy_pose


def get_object_poses_fp(obs, task_name='close_jar', mesh_dir=None, estimators=None,
                        verbose=False, add_noise='none', moving_object_index=0,
                        task=None, save_debug=False):
    """Get object poses using Foundation Pose.
    
    Args:
        moving_object_index: Which moving object to extract (0=first, 1=second for stacking tasks).
    """
    if mesh_dir is None:
        mesh_dir = os.path.normpath(os.path.join(_SCRIPT_DIR, '../assets/RLBench_mesh'))
    try:
        # Extract poses from current observation
        result = extract_poses_from_observation(
            obs=obs,
            task_name=task_name,
            mesh_dir=mesh_dir,
            camera_name='front',
            estimators=estimators,
            verbose=verbose,
            moving_object_index=moving_object_index,
            task=task,
            save_debug=save_debug,
        )
        # Check if poses were successfully extracted
        if not result['success']:
            raise RuntimeError("Foundation Pose failed: could not extract poses")
        base_pose = result['base_pose']
        moving_pose = result['moving_pose']
        
        # Add noise if requested
        if add_noise != 'none':
            base_pose = add_noise_to_pose(base_pose, add_noise)
            moving_pose = add_noise_to_pose(moving_pose, add_noise)
            if verbose:
                print(f"Added {add_noise} noise to poses")
        
        poses = {
            'object_0': Transform.from_matrix(base_pose),    # base
            'object_1': Transform.from_matrix(moving_pose)   # moving
        }
        if verbose:
            print(f"Foundation Pose extracted successfully:")
            print(f"  Base object position: {base_pose[:3, 3]}")
            print(f"  Moving object position: {moving_pose[:3, 3]}")
        return poses, result['estimators']
    except Exception as e:
        raise RuntimeError(f"Foundation Pose failed: {e}")


def delta_along_axis(trans, ori, axis='z', delta=-0.05):
    """Move along gripper's local axis"""
    if axis == 'x':
        column_num = 0
    elif axis == 'y':
        column_num = 1
    elif axis == 'z':
        column_num = 2
    else:
        raise ValueError('axis is wrong')
    axis_vec = ori.as_matrix()[:, column_num]
    delta_vec = delta * axis_vec
    trans = trans + delta_vec
    return trans, ori


def trans_along_axis(trans, ori, axis='z', delta=-0.05):
    """Move along world axis"""
    if axis == 'x':
        column_num = 0
    elif axis == 'y':
        column_num = 1
    elif axis == 'z':
        column_num = 2
    else:
        raise ValueError('axis is wrong')
    axis_vec = np.eye(3)[:, column_num]
    delta_vec = delta * axis_vec
    trans = trans + delta_vec
    return trans, ori


def visualize_debug_scene(pa_pcd, pb_pcd, pa_pose, pb_pose, gripper_pose, 
                         predicted_traj, mode='grasp'):
    """ Visualize and save debug scene when path planning fails. """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create visualization
    geometries = []
    
    # Add point clouds
    pa_pcd_vis = o3d.geometry.PointCloud(pa_pcd)
    pb_pcd_vis = o3d.geometry.PointCloud(pb_pcd)
    geometries.extend([pa_pcd_vis, pb_pcd_vis])
    
    # Add coordinate frame at origin
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(origin_frame)
    
    # Add base object (pa) frame - Red
    pa_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
    pa_frame.paint_uniform_color([1, 0, 0])  # Red
    pa_frame.transform(pa_pose)
    geometries.append(pa_frame)
    
    # Add moving object (pb) frame - Blue
    pb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
    pb_frame.paint_uniform_color([0, 0, 1])  # Blue
    pb_frame.transform(pb_pose)
    geometries.append(pb_frame)
    
    # Add gripper frame - Green
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
    gripper_frame.paint_uniform_color([0, 1, 0])  # Green
    gripper_frame.transform(gripper_pose)
    geometries.append(gripper_frame)
    
    # Add predicted trajectory
    traj_positions = [traj[:3, 3] for traj in predicted_traj]
    if len(traj_positions) > 1:
        lines = [[i, i+1] for i in range(len(traj_positions)-1)]
        colors = [[1.0, 0.6, 0.0] for _ in lines]  # Orange
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(traj_positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)
    # Add spheres at waypoints (orange)
    for i, pos in enumerate(traj_positions):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        sphere.paint_uniform_color([1, 0.5, 0])  # yellow
        sphere.translate(pos)
        geometries.append(sphere)
        # Add frame at each waypoint (smaller)
        wp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
        wp_frame.transform(predicted_traj[i])
        geometries.append(wp_frame)
    # Add arrows to indicate predicted trajectory direction (orange)
    for i in range(len(traj_positions) - 1):
        start = traj_positions[i]
        end = traj_positions[i + 1]
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cone_radius=0.004, cylinder_height=0.01, cone_height=0.01)
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 1e-6:
            direction = direction / length
            z_axis = np.array([0, 0, 1])
            v = np.cross(z_axis, direction)
            c = np.dot(z_axis, direction)
            if np.linalg.norm(v) < 1e-6:
                R_mat = np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R_mat = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
            arrow.rotate(R_mat, center=np.zeros(3))
            arrow.paint_uniform_color([1.0, 0.6, 0.0])  # Orange
            arrow.translate(start + 0.7 * (end - start))
            geometries.append(arrow)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Debug Scene - {mode.upper()} mode - {timestamp}",
        width=1280,
        height=720
    )


def extract_expert_trajectory(
    demo,
    task_name,
    keyframes,
    mode='grasp',
    traj_length=3,
    pen_factor=None,
):
    """ Extract expert trajectory from demo with specified length. """
    intervals_result = compute_intervals_for_demo(
        demo=demo,
        task_name=task_name,
        expansion_factor=1.0,
        use_change_detection=True,
        pen_factor=pen_factor,
        keyframes=keyframes
    )
    if mode == 'grasp':
        # Goal is pick_frame
        goal_frame = keyframes['pick']
        interval = intervals_result['grasp_interval']
    elif mode == 'manip':
        # Goal is place_frame
        goal_frame = keyframes['place']
        interval = intervals_result['manip_interval']
    else:
        raise ValueError(f"Unknown mode: {mode}")
    start_frame = max(0, goal_frame - interval + 1)
    frame_indices = np.linspace(start_frame, goal_frame, traj_length, dtype=int)
    # Extract gripper poses from these frames
    trajectory_matrices = []
    for frame_idx in frame_indices:
        obs = demo[frame_idx]
        trans = obs.gripper_pose[:3]
        quat = obs.gripper_pose[3:]
        quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
        pose = Transform(rotation=Rotation.from_quat(quat), translation=trans)
        trajectory_matrices.append(pose.as_matrix())
    return trajectory_matrices


def translate_trajectory_world(trajectory, translation):
    translated_traj = []
    translation = np.array(translation)
    for pose in trajectory:
        new_pose = pose.copy()
        new_pose[:3, 3] += translation
        translated_traj.append(new_pose)
    return translated_traj

def translate_trajectory_local(trajectory, translation):
    translated_traj = []
    translation = np.array(translation)
    for pose in trajectory:
        new_pose = pose.copy()
        # Local translation: t_new = R @ t_local + t
        local_translation = pose[:3, :3] @ translation
        new_pose[:3, 3] += local_translation
        translated_traj.append(new_pose)
    return translated_traj

def rotate_trajectory_local(trajectory, axis='y', angle_deg=0):
    rotated_traj = []
    angle_rad = math.radians(angle_deg)
    # Build rotation matrix based on axis
    if axis == 'x':
        R_local = np.array([
            [1, 0, 0],
            [0, math.cos(angle_rad), -math.sin(angle_rad)],
            [0, math.sin(angle_rad), math.cos(angle_rad)]
        ])
    elif axis == 'y':
        R_local = np.array([
            [math.cos(angle_rad), 0, math.sin(angle_rad)],
            [0, 1, 0],
            [-math.sin(angle_rad), 0, math.cos(angle_rad)]
        ])
    elif axis == 'z':
        R_local = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
    for pose in trajectory:
        new_pose = pose.copy()
        # Local rotation: R_new = R @ R_local
        new_pose[:3, :3] = pose[:3, :3] @ R_local
        rotated_traj.append(new_pose)
    return rotated_traj

TABLE_HEIGHT = 0.755
def adjust_demo_traj(task_name, trajectory, mode='manip', gripper_pose_current=None):
    adjusted_traj = trajectory
    if mode == 'grasp':
        for pose in adjusted_traj:
            if pose[2, 3] < TABLE_HEIGHT:
                pose[2, 3] = TABLE_HEIGHT
        if task_name == 'turn_tap':
            adjusted_traj = translate_trajectory_local(adjusted_traj, [0, 0, 0.02])
    elif mode == 'manip':
        if task_name == 'close_jar':
            adjusted_traj = translate_trajectory_world(trajectory, [0, 0, 0.05])
        elif task_name in ('insert_onto_square_peg', 'place_shape_in_shape_sorter'):
            adjusted_traj = translate_trajectory_world(trajectory, [0, 0, 0.01])
        elif task_name == 'sweep_to_dustpan_of_size':
            adjusted_traj = translate_trajectory_local(trajectory, [0.06, 0, 0])
            adjusted_traj = translate_trajectory_world(adjusted_traj, [0, 0, -0.03])
            for pose in adjusted_traj:
                if pose[2, 3] < 1.075:
                    pose[2, 3] = 1.075
        elif task_name == 'push_buttons':
            if gripper_pose_current is None:
                raise ValueError('gripper_pose_current must be provided for push_buttons')
            adjusted_traj = []
            base = np.array(gripper_pose_current).copy()
            for dz in (-0.04, -0.08, -0.12):
                p = base.copy()
                p[:3, 3] = p[:3, 3] + np.array([0.0, 0.0, dz])
                adjusted_traj.append(p)
        elif task_name in ("open_drawer"):
            adjusted_traj = trajectory[-1:]
        elif task_name in ('stack_blocks', 'stack_cups'):
            adjusted_traj = translate_trajectory_world(trajectory, [0, 0, 0.03])
        elif task_name in ('reach_and_drag'):
            adjusted_traj = translate_trajectory_world(trajectory, [0, 0, 0.03])
            adjusted_traj = translate_trajectory_local(adjusted_traj, [0.0, -0.03, 0])
        for pose in adjusted_traj:
            if pose[2, 3] < TABLE_HEIGHT:
                pose[2, 3] = TABLE_HEIGHT
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return adjusted_traj


class EpisodeVideoRecorder:
    def __init__(self, scene, fps=20.0, camera_resolution=(1280, 720), camera_speed=0.0):
        self._scene = scene
        self._fps = fps
        self._camera_resolution = list(camera_resolution)
        self._camera_speed = camera_speed
        self._frames = []
        self._cam = None
        self._cam_origin = None
        self._prev_cam_pose = None

    def _build_camera(self):
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self._cam = VisionSensor.create(self._camera_resolution)
        self._cam.set_pose(cam_placeholder.get_pose())
        self._cam.set_parent(cam_placeholder)
        self._cam_origin = Dummy('cam_cinematic_base')

    def start(self):
        if self._cam is None:
            self._build_camera()
        self._prev_cam_pose = self._cam.get_pose()
        self._scene.register_step_callback(self._capture_each_sim_step)

    def stop(self):
        self._scene.register_step_callback(None)
        if self._cam is not None and self._prev_cam_pose is not None:
            self._cam.set_pose(self._prev_cam_pose)

    def reset_episode(self):
        self._frames = []

    def _capture_each_sim_step(self):
        if self._cam_origin is not None and self._camera_speed != 0.0:
            self._cam_origin.rotate([0, 0, self._camera_speed])
        self._frames.append((self._cam.capture_rgb() * 255.).astype(np.uint8))

    def capture_now(self):
        if self._cam is None:
            return
        self._frames.append((self._cam.capture_rgb() * 255.).astype(np.uint8))

    def save(self, path):
        if not self._frames:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import cv2
        video_size = tuple(self._cam.get_resolution())
        video = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
            video_size)
        for frame in self._frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()


def main(args):
    # CoppeliaSim requires an X server even in headless mode.
    # If no DISPLAY is set, relaunch under xvfb-run automatically.
    if not args.disp and not os.environ.get('DISPLAY'):
        import subprocess, sys
        print("No DISPLAY found — relaunching under xvfb-run...")
        result = subprocess.run(
            ['xvfb-run', '-a', sys.executable] + sys.argv,
            env={**os.environ, 'DISPLAY': ':99'},
        )
        sys.exit(result.returncode)

    # FoundationPose uses the unqualified 'cuda' device internally, so setting
    # the current device here keeps it on the same GPU as the FOCI actors.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for simulator evaluation')
    torch.cuda.set_device(args.device)
    print(f"Using CUDA device: {args.device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    # Load Actor for grasp mode
    if args.actor == 'expert':
        print("Running in EXPERT mode - using demo trajectories")
        actor_grasp = None
        actor_manip = None
        language_encoder = None
        use_language = False
        traj_length = args.traj_length
        print(f"Trajectory length: {traj_length}")
        actor_name = 'EXPERT'
    else:
        if args.actor == 'foci':
            print(f"Loading FOCI Actor for grasp mode...")
            actor_grasp = FOCIActor(
                checkpoint_path=GRASP_CKPT,
                config_dir=CONFIG_DIR,
                device=args.device,
                voxel_size=args.voxel_size,
                num_points=args.num_points,
                deterministic=True,
            )
            print(f"Grasp Actor loaded in '{actor_grasp.mode}' mode")
            
            # Load FOCI Actor for manip mode
            print(f"Loading FOCI Actor for manip mode...")
            actor_manip = FOCIActor(
                checkpoint_path=MANIP_CKPT,
                config_dir=CONFIG_DIR,
                device=args.device,
                voxel_size=args.voxel_size,
                num_points=args.num_points,
                deterministic=True,
            )
            print(f"Manip Actor loaded in '{actor_manip.mode}' mode")
        else:
            raise ValueError(f"Unknown actor type: {args.actor}")
        traj_length = actor_grasp.prediction_length
        use_language = actor_grasp.use_language
        
        if use_language:
            language_encoder = AddLanguageEmbedding(device=args.device)
            print("Language embedding enabled")
        else:
            language_encoder = None
            print("Language embedding disabled")

    # Setup RLBench environment
    restore_coppeliasim_qt_environment()
    headless = not args.disp
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode, DATA_FOLDER, obs_config, headless=headless)
    env.launch()
    
    # Get task
    task_class = get_task_class(args.task)
    task = env.get_task(task_class)
    task.set_variation(args.var)
    
    if args.actor != 'expert':
        actor_name = 'FOCI'
    print(f'Testing {actor_name} Actor on {task.get_name()} for {args.n_tests} tests, starting from episode {args.start_test}')
    results = []

    episode_recorder = None
    if args.record:
        episode_recorder = EpisodeVideoRecorder(
            scene=task._scene,
            fps=args.record_fps,
            camera_resolution=(args.record_width, args.record_height),
            camera_speed=args.record_camera_speed,
        )
        episode_recorder.start()
    
    for j in range(args.n_tests):
        episode_status = 'failure'
        # Load demo
        EPISODES_FOLDER = f'{task.get_name()}/variation{args.var}/episodes'
        data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
        demo = get_stored_demo(data_path=data_path, index=args.start_test + j, init_matrix=False)
        
        print(f'\n{"="*60}')
        print(f'Task: {args.task}, Test: {j+1}/{args.n_tests}')
        print(f'{"="*60}')
        
        # Reset environment
        descriptions, obs = task.reset_to_demo(demo)
        if episode_recorder is not None:
            episode_recorder.reset_episode()
            episode_recorder.capture_now()
        
        # Extract keyframes for expert 
        if args.actor == 'expert':
            keyframes = get_pick_place_keyframes(demo, task_name=args.task)
            print(f"Keyframes: pick={keyframes['pick']}, place={keyframes['place']}")
            
        # Extract point clouds
        pa_points, pa_colors, pb_points, pb_colors = extract_point_clouds_from_obs(obs, task_name=args.task, task=task)
        pa_pcd = to_o3d_pcd(pa_points, pa_colors)
        pb_pcd = to_o3d_pcd(pb_points, pb_colors)
        
        # Get initial poses from observation
        gripper_pose_init = get_gripper_pose(obs)
        fp_estimators = None
        if args.pose_method in ['fp', 'foundation_pose']:
            print("Using Foundation Pose to estimate object poses...")
            # object_poses_init, fp_estimators = get_object_poses_fp(demo[0], task_name=args.task, mesh_dir=args.mesh_dir, verbose=True)
            object_poses_init, fp_estimators = get_object_poses_fp(
                obs, task_name=args.task, mesh_dir=args.mesh_dir, verbose=True,
                add_noise=args.add_noise, task=task, save_debug=args.debug,
            )
            if args.add_noise != 'none':
                print(f"Added {args.add_noise} noise to Foundation Pose estimates")
            print("Foundation Pose complete")
        else:
            object_poses_init = get_object_poses(obs, task_name=args.task, task=task)
        
        pa_pose = object_poses_init['object_0'].as_matrix()
        pb_pose = object_poses_init['object_1'].as_matrix()
        gripper_pose = gripper_pose_init.as_matrix()
        
        initial_trans = gripper_pose_init.translation
        initial_ori = gripper_pose_init.rotation
        initial_action = np.concatenate([initial_trans, initial_ori.as_quat(), [1.0]], axis=-1)
        
        reward = 0
        terminate = False
        
        # ========== PRE-PHASE: ESTIMATE SECOND OBJECT POSE FOR STACKING TASKS ==========
        # For stack_blocks and stack_cups, pre-estimate the second moving object pose
        object_poses_2 = None
        pa_pose_2 = None
        pb_pose_2 = None
        from foci_policy.config.config_utils import get_task_obj_config as _get_obj_cfg
        _has_stage2 = 'grasp_2' in _get_obj_cfg(args.task)
        if _has_stage2:
            print("\n" + "="*60)
            print("PRE-PHASE: Estimating second object pose for stacking task")
            print("="*60)

            if args.pose_method in ['fp', 'foundation_pose']:
                print("Using Foundation Pose to estimate second moving object pose...")
                del fp_estimators
                fp_estimators = None
                object_poses_2, fp_estimators = get_object_poses_fp(
                    obs, task_name=args.task, mesh_dir=args.mesh_dir,
                    verbose=False, add_noise=args.add_noise,
                    moving_object_index=1, task=task, save_debug=args.debug,
                )
                print("Foundation Pose complete for second object")
            else:
                # GT mode: use grasp_2/target_2 from task_obj_config.yaml
                object_poses_2 = get_object_poses(obs, task_name=args.task, task=task, stage=2)

            if object_poses_2 is not None:
                pa_pose_2 = object_poses_2['object_0'].as_matrix()
                pb_pose_2 = object_poses_2['object_1'].as_matrix()
                print(f"Second object poses estimated successfully")
        
        # ========== PHASE 0 (OPTIONAL): OPEN DRAWER ==========
        # For put_item_in_drawer: open the drawer first. The multi-task actor handles both
        # open_drawer and put_item_in_drawer; point clouds/poses/language use 'open_drawer' config.
        if args.task == 'put_item_in_drawer' and args.actor != 'expert':
            print("\n" + "="*60)
            print("PHASE 0: OPEN DRAWER")
            print("="*60)

            pa_points_od, pa_colors_od, pb_points_od, pb_colors_od = extract_point_clouds_from_obs(
                obs, task_name='open_drawer', task=task
            )
            pa_pcd_od = to_o3d_pcd(pa_points_od, pa_colors_od)
            pb_pcd_od = to_o3d_pcd(pb_points_od, pb_colors_od)
            object_poses_od = get_object_poses(obs, task_name='open_drawer', task=task)
            pa_pose_od = object_poses_od['object_0'].as_matrix()
            pb_pose_od = object_poses_od['object_1'].as_matrix()
            gripper_pose_od = gripper_pose_init.as_matrix()

            # --- Open drawer grasp: reach and grip the drawer handle ---
            language_emb_od = None
            if use_language and language_encoder is not None:
                dummy_data_od = Data()
                dummy_data_od['language'] = get_task_language('open_drawer', mode='grasp')
                language_emb_od = language_encoder(dummy_data_od)['language_embedding'].unsqueeze(0).to(actor_grasp.device)

            time_start = time.time()
            od_grasp_traj = actor_grasp.act(pa_pcd_od, pb_pcd_od, pa_pose_od, pb_pose_od, gripper_pose_od, language_emb_od)
            print(f'Open drawer grasp inference time: {time.time()-time_start:.3f}s')
            od_grasp_traj = adjust_demo_traj('open_drawer', od_grasp_traj, mode='grasp')
            print(f'Open drawer grasp trajectory has {len(od_grasp_traj)} waypoints')

            print("Executing open drawer grasp...")
            od_pick_trans, od_pick_ori = None, None
            try:
                for wp_idx, waypoint_matrix in enumerate(od_grasp_traj):
                    wp_pose = Transform.from_matrix(waypoint_matrix)
                    wp_trans, wp_ori = wp_pose.translation, wp_pose.rotation
                    print(f"  Waypoint {wp_idx+1}/{len(od_grasp_traj)}: {wp_trans}")
                    task.step(np.concatenate([wp_trans, wp_ori.as_quat(), [1.0]]), test_mode=True)
                    time.sleep(0.2)
                last_wp = Transform.from_matrix(od_grasp_traj[-1])
                od_pick_trans, od_pick_ori = last_wp.translation, last_wp.rotation
                task.step(np.concatenate([od_pick_trans, od_pick_ori.as_quat(), [1.0]]), test_mode=True)
                time.sleep(0.2)
                task.step(np.concatenate([od_pick_trans, od_pick_ori.as_quat(), [0.0]]))
                time.sleep(0.2)
            except Exception as e:
                print(f"Warning: open drawer grasp failed: {e}")

            # --- Open drawer manip: pull drawer open (no vertical lift) ---
            obs = task._scene.get_observation()
            gripper_pose_od_after = get_gripper_pose(obs).as_matrix()
            pa_points_od2, pa_colors_od2, pb_points_od2, pb_colors_od2 = extract_point_clouds_from_obs(
                obs, task_name='open_drawer', task=task
            )
            pa_pcd_od2 = to_o3d_pcd(pa_points_od2, pa_colors_od2)
            pb_pcd_od2 = to_o3d_pcd(pb_points_od2, pb_colors_od2)

            language_emb_od_m = None
            if use_language and language_encoder is not None:
                dummy_data_od_m = Data()
                dummy_data_od_m['language'] = get_task_language('open_drawer', mode='manip')
                language_emb_od_m = language_encoder(dummy_data_od_m)['language_embedding'].unsqueeze(0).to(actor_manip.device)

            time_start = time.time()
            od_manip_traj = actor_manip.act(
                pa_pcd_od2, pb_pcd_od2, pa_pose_od, pb_pose_od, gripper_pose_od_after, language_emb_od_m
            )
            print(f'Open drawer manip inference time: {time.time()-time_start:.3f}s')
            od_manip_traj = adjust_demo_traj('open_drawer', od_manip_traj, mode='manip', gripper_pose_current=gripper_pose_od_after)
            print(f'Open drawer manip trajectory has {len(od_manip_traj)} waypoints')

            od_place_trans, od_place_ori = None, None
            print("Executing open drawer manip (pulling open)...")
            try:
                for wp_idx, waypoint_matrix in enumerate(od_manip_traj):
                    wp_pose = Transform.from_matrix(waypoint_matrix)
                    wp_trans, wp_ori = wp_pose.translation, wp_pose.rotation
                    print(f"  Waypoint {wp_idx+1}/{len(od_manip_traj)}: {wp_trans}")
                    task.step(np.concatenate([wp_trans, wp_ori.as_quat(), [0.0]]), test_mode=True)
                    time.sleep(0.2)
                last_wp_m = Transform.from_matrix(od_manip_traj[-1])
                od_place_trans, od_place_ori = last_wp_m.translation, last_wp_m.rotation
                # Release drawer handle
                task.step(np.concatenate([od_place_trans, od_place_ori.as_quat(), [1.0]]))
                time.sleep(0.2)
            except Exception as e:
                print(f"Warning: open drawer manip failed: {e}")

            # Retract from drawer before picking item
            if od_place_trans is not None:
                print("Retracting from drawer...")
                post_od_trans, post_od_ori = trans_along_axis(od_place_trans, od_place_ori, axis='z', delta=0.2)
                task.step(np.concatenate([post_od_trans, post_od_ori.as_quat(), [1.0]]), test_mode=True)
                time.sleep(0.2)

            # Refresh observation and gripper pose for PHASE 1
            obs = task._scene.get_observation()
            gripper_pose = get_gripper_pose(obs).as_matrix()
            print("Drawer opened — proceeding to pick item.")

        # ========== PHASE 1: GRASP MODE ==========
        print("\n" + "="*60)
        print("PHASE 1: GRASP MODE - Predict and execute grasp trajectory")
        print("="*60)
        lift_height = 0.0
        
        if args.actor == 'expert':
            # Extract expert trajectory from demo
            print("Extracting expert grasp trajectory from demo...")
            grasp_trajectory = extract_expert_trajectory(demo, args.task, keyframes, mode='grasp', traj_length=traj_length)
        else:
            # Prepare language embedding if needed
            language_emb = None
            if use_language and language_encoder is not None:
                language_text = get_task_language(args.task, mode='grasp')
                # Create Data object with language field
                dummy_data = Data()
                dummy_data['language'] = language_text
                result = language_encoder(dummy_data)
                language_emb = result['language_embedding'].unsqueeze(0).to(actor_grasp.device)
            
            # Use model to predict trajectory
            time_start = time.time()
            grasp_trajectory = actor_grasp.act(pa_pcd, pb_pcd, pa_pose, pb_pose, gripper_pose, language_emb)
            infer_time = time.time() - time_start
            print(f'Grasp inference time: {infer_time:.3f}s')
        grasp_trajectory = adjust_demo_traj(args.task, grasp_trajectory, mode='grasp')
        print(f'Grasp trajectory has {len(grasp_trajectory)} waypoints')
        
        # Execute grasp trajectory waypoints (gripper open)
        print("\nExecuting grasp trajectory...")
        try:
            for wp_idx, waypoint_matrix in enumerate(grasp_trajectory):
                waypoint_pose = Transform.from_matrix(waypoint_matrix)
                wp_trans, wp_ori = waypoint_pose.translation, waypoint_pose.rotation
                print(f"  Waypoint {wp_idx+1}/{len(grasp_trajectory)}: {wp_trans}")
                # Move to waypoint with gripper open
                wp_action = np.concatenate([wp_trans, wp_ori.as_quat(), [1.0]], axis=-1)
                res = task.step(wp_action, test_mode=True)
                time.sleep(0.2)
            # Confirm final pick position with gripper open, then close
            print("\nClosing gripper to grasp object...")
            last_waypoint = Transform.from_matrix(grasp_trajectory[-1])
            pick_trans, pick_ori = last_waypoint.translation, last_waypoint.rotation
            pick_open = np.concatenate([pick_trans, pick_ori.as_quat(), [1.0]], axis=-1)
            res = task.step(pick_open, test_mode=True)
            time.sleep(0.2)
            pick_close = np.concatenate([pick_trans, pick_ori.as_quat(), [0.0]], axis=-1)
            _ = task.step(pick_close)
            time.sleep(0.2)
        except Exception as e:
            print(f"\n✗ Error during grasp execution: {e}")
            results.append(0)
            print(f'Test {j+1} result: FAILURE (grasp execution error)')
            if args.debug:
                visualize_debug_scene(
                    pa_pcd=pa_pcd,
                    pb_pcd=pb_pcd,
                    pa_pose=pa_pose,
                    pb_pose=pb_pose,
                    gripper_pose=gripper_pose,
                    predicted_traj=grasp_trajectory,
                    mode='grasp'            
                    )
            if episode_recorder is not None:
                video_name = f'{task.get_name()}_var{args.var}_ep{args.start_test + j}_{episode_status}.mp4'
                episode_recorder.save(os.path.join(args.save_dir, video_name))
            continue

        # Check if object is grasped
        if args.task not in ('turn_tap', 'push_buttons', 'slide_block_to_color_target', 'open_drawer', 'close_box'):
            lift_height = 0.15
            grasped_object_list = task._robot.gripper.get_grasped_objects()
            if len(grasped_object_list) > 0:
                print(colored(f"\n✓ Successfully grasped: {grasped_object_list[0].get_name()}", "green"))
            else:
                print(colored(f"\n✗ Grasp failed", "red"))
                obs, _, _ = task.step(initial_action)
                results.append(0)
                print(f'Test {j+1} result: FAILURE (grasp failed)')
                if args.debug:
                    visualize_debug_scene(
                        pa_pcd=pa_pcd,
                        pb_pcd=pb_pcd,
                        pa_pose=pa_pose,
                        pb_pose=pb_pose,
                        gripper_pose=gripper_pose,
                        predicted_traj=grasp_trajectory,
                        mode='grasp'            
                    )
                if episode_recorder is not None:
                    video_name = f'{task.get_name()}_var{args.var}_ep{args.start_test + j}_{episode_status}.mp4'
                    episode_recorder.save(os.path.join(args.save_dir, video_name))
                continue
            
            # Lift up after grasp
            print("\nLifting object...")
            post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=lift_height)
            post_pick_action = np.concatenate([post_pick_trans, post_pick_ori.as_quat(), [0.0]], axis=-1)
            res = task.step(post_pick_action, test_mode=True)
            time.sleep(0.2)
        
        # ========== PHASE 2: MANIP MODE ==========
        print("\n" + "="*60)
        print("PHASE 2: MANIP MODE - Predict and execute manipulation trajectory")
        print("="*60)
        
        # Get current gripper pose after grasp
        obs = task._scene.get_observation()
        gripper_pose_after_grasp = get_gripper_pose(obs)
        gripper_pose_current = gripper_pose_after_grasp.as_matrix()

        # Get current pcd
        pa_points, pa_colors, pb_points, pb_colors = extract_point_clouds_from_obs(obs, task_name=args.task, task=task)
        pa_pcd = to_o3d_pcd(pa_points, pa_colors)
        pb_pcd = to_o3d_pcd(pb_points, pb_colors)
        
        # Assume base is static, and moving object is lifted by gripper
        pa_pose_current = pa_pose
        pb_pose[:3, 3] += np.array([0, 0, lift_height])
        pb_pose_current = pb_pose
        
        if args.actor == 'expert':
            # Extract expert trajectory from demo
            print("Extracting expert manip trajectory from demo...")
            manip_trajectory = extract_expert_trajectory(demo, args.task, keyframes, mode='manip', traj_length=traj_length)
        else:
            # Prepare language embedding if needed
            language_emb = None
            if use_language and language_encoder is not None:
                language_text = get_task_language(args.task, mode='manip')
                dummy_data = Data()
                dummy_data['language'] = language_text
                result = language_encoder(dummy_data)
                language_emb = result['language_embedding'].unsqueeze(0).to(actor_manip.device)
            
            # Use model to predict trajectory
            time_start = time.time()
            manip_trajectory = actor_manip.act(pa_pcd, pb_pcd, pa_pose_current, pb_pose_current, gripper_pose_current, language_emb)
            infer_time = time.time() - time_start
            print(f'Manip inference time: {infer_time:.3f}s')
        manip_trajectory = adjust_demo_traj(args.task, manip_trajectory, mode='manip', gripper_pose_current=gripper_pose_current)
        print(f'Manip trajectory has {len(manip_trajectory)} waypoints')
        
        # Execute manip trajectory waypoints (gripper closed)
        print("\nExecuting manip trajectory...")
        # Disable collision for the block during manip so the arm doesn't route around it
        _slide_block_shape = None
        if args.task == 'slide_block_to_color_target':
            try:
                from pyrep.objects.shape import Shape
                _slide_block_shape = Shape('block')
                _slide_block_shape.set_collidable(False)
                print("Collision disabled for block during manip")
            except Exception as _e:
                print(f"Warning: could not disable block collision: {_e}")
        try:
            for wp_idx, waypoint_matrix in enumerate(manip_trajectory):
                waypoint_pose = Transform.from_matrix(waypoint_matrix)
                wp_trans, wp_ori = waypoint_pose.translation, waypoint_pose.rotation
                print(f"  Waypoint {wp_idx+1}/{len(manip_trajectory)}: {wp_trans}")
                wp_action = np.concatenate([wp_trans, wp_ori.as_quat(), [0.0]], axis=-1)
                res = task.step(wp_action, test_mode=True)
                time.sleep(0.2)

            # Confirm final place position with gripper closed
            last_waypoint = Transform.from_matrix(manip_trajectory[-1])
            place_trans, place_ori = last_waypoint.translation, last_waypoint.rotation
            place_close = np.concatenate([place_trans, place_ori.as_quat(), [0.0]], axis=-1)
            res = task.step(place_close, test_mode=True)
            time.sleep(0.2)

            # Open gripper to release
            final_gripper_state = [1.0]
            print("Opening gripper to release object...")
            place_open = np.concatenate([place_trans, place_ori.as_quat(), final_gripper_state], axis=-1)
            _ = task.step(place_open)
            time.sleep(0.2)
        except Exception as e:
            print(f"\n✗ Error during manip execution: {e}")
            results.append(0)
            print(f'Test {j+1} result: FAILURE (manip execution error)')
            if args.debug:
                visualize_debug_scene(
                    pa_pcd=pa_pcd,
                    pb_pcd=pb_pcd,
                    pa_pose=pa_pose_current,
                    pb_pose=pb_pose_current,
                    gripper_pose=gripper_pose_current,
                    predicted_traj=manip_trajectory,
                    mode='manip'
                )
            if episode_recorder is not None:
                video_name = f'{task.get_name()}_var{args.var}_ep{args.start_test + j}_{episode_status}.mp4'
                episode_recorder.save(os.path.join(args.save_dir, video_name))
            continue
        finally:
            if _slide_block_shape is not None:
                _slide_block_shape.set_collidable(True)
                print("Block collision re-enabled")

        # Check task completion
        obs, reward, terminate = task.step(place_open)
        time.sleep(0.1)

        # Retract after place
        print("\nRetracting...")
        post_place_trans, post_place_ori = trans_along_axis(place_trans, place_ori, axis='z', delta=0.2)
        post_place_action = np.concatenate([post_place_trans, post_place_ori.as_quat(), final_gripper_state], axis=-1)
        res = task.step(post_place_action, test_mode=True)
        time.sleep(0.2)
        
        # ========== PHASE 3 (OPTIONAL): SECOND GRASP-MANIP FOR STACKING TASKS ==========
        if args.task in ('stack_blocks', 'stack_cups') and not terminate:
            print("\n" + "="*60)
            print("PHASE 3: SECOND GRASP-MANIP CYCLE (Stacking Task)")
            print("="*60)
            
            # Second object poses were pre-estimated in PRE-PHASE
            if pa_pose_2 is not None and pb_pose_2 is not None:
                # Check if this task has a second stage in task_obj_config
                from foci_policy.config.config_utils import get_task_obj_config as _get_cfg2
                _has_stage2_phase3 = 'grasp_2' in _get_cfg2(args.task)

                if _has_stage2_phase3:
                    # ===== SECOND GRASP PHASE =====
                    print("\n" + "-"*60)
                    print("Second Grasp Phase")
                    print("-"*60)

                    # Extract point clouds for second object using DemoObjectExtractor stage=2
                    obs = task._scene.get_observation()
                    pa_points, pa_colors, pb_points, pb_colors = extract_point_clouds_from_obs(
                        obs, task_name=args.task, moving_mask_index=True, task=task
                    )
                    pa_pcd = to_o3d_pcd(pa_points, pa_colors)
                    pb_pcd = to_o3d_pcd(pb_points, pb_colors)
                    
                    gripper_pose_init_2 = get_gripper_pose(obs)
                    gripper_pose_2 = gripper_pose_init_2.as_matrix()
                
                if args.actor == 'expert':
                    grasp_trajectory_2 = extract_expert_trajectory(demo, args.task, keyframes, mode='grasp', traj_length=traj_length)
                else:
                    language_emb = None
                    if use_language and language_encoder is not None:
                        language_text = get_task_language(args.task, mode='grasp')
                        dummy_data = Data()
                        dummy_data['language'] = language_text
                        result = language_encoder(dummy_data)
                        language_emb = result['language_embedding'].unsqueeze(0).to(actor_grasp.device)
                    
                    time_start = time.time()
                    grasp_trajectory_2 = actor_grasp.act(pa_pcd, pb_pcd, pa_pose_2, pb_pose_2, gripper_pose_2, language_emb)
                    infer_time = time.time() - time_start
                    print(f'Second grasp inference time: {infer_time:.3f}s')
                
                grasp_trajectory_2 = adjust_demo_traj(args.task, grasp_trajectory_2, mode='grasp')
                print(f'Second grasp trajectory has {len(grasp_trajectory_2)} waypoints')
                
                # Execute second grasp trajectory
                print("\nExecuting second grasp trajectory...")
                try:
                    for wp_idx, waypoint_matrix in enumerate(grasp_trajectory_2):
                        waypoint_pose = Transform.from_matrix(waypoint_matrix)
                        wp_trans, wp_ori = waypoint_pose.translation, waypoint_pose.rotation
                        print(f"  Waypoint {wp_idx+1}/{len(grasp_trajectory_2)}: {wp_trans}")
                        wp_action = np.concatenate([wp_trans, wp_ori.as_quat(), [1.0]], axis=-1)
                        res = task.step(wp_action, test_mode=True)
                        time.sleep(0.2)
                    
                    # Close gripper for second grasp
                    print("\nClosing gripper to grasp second object...")
                    last_waypoint = Transform.from_matrix(grasp_trajectory_2[-1])
                    pick_trans_2, pick_ori_2 = last_waypoint.translation, last_waypoint.rotation
                    pick_open_2 = np.concatenate([pick_trans_2, pick_ori_2.as_quat(), [1.0]], axis=-1)
                    res = task.step(pick_open_2, test_mode=True)
                    time.sleep(0.2)
                    pick_close_2 = np.concatenate([pick_trans_2, pick_ori_2.as_quat(), [0.0]], axis=-1)
                    _ = task.step(pick_close_2)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"\n✗ Error during second grasp execution: {e}")
                    if args.debug:
                        visualize_debug_scene(
                            pa_pcd=pa_pcd,
                            pb_pcd=pb_pcd,
                            pa_pose=pa_pose_2,
                            pb_pose=pb_pose_2,
                            gripper_pose=gripper_pose_2,
                            predicted_traj=grasp_trajectory_2,
                            mode='grasp_2'            
                        )
                
                # Lift after second grasp
                print("\nLifting second object...")
                lift_height_2 = 0.2
                post_pick_trans_2, post_pick_ori_2 = trans_along_axis(pick_trans_2, pick_ori_2, axis='z', delta=lift_height_2)
                post_pick_action_2 = np.concatenate([post_pick_trans_2, post_pick_ori_2.as_quat(), [0.0]], axis=-1)
                res = task.step(post_pick_action_2, test_mode=True)
                time.sleep(0.2)
                
                # ===== SECOND MANIP PHASE =====
                print("\n" + "-"*60)
                print("Second Manip Phase")
                print("-"*60)
                
                obs = task._scene.get_observation()
                pa_points, pa_colors, pb_points, pb_colors = extract_point_clouds_from_obs(
                    obs, task_name=args.task, moving_mask_index=True, task=task
                )
                pa_pcd = to_o3d_pcd(pa_points, pa_colors)
                pb_pcd = to_o3d_pcd(pb_points, pb_colors)
                
                gripper_pose_after_grasp_2 = get_gripper_pose(obs)
                gripper_pose_current_2 = gripper_pose_after_grasp_2.as_matrix()
                
                # Use pre-estimated poses from PRE-PHASE
                pa_pose_current_2 = pa_pose_2
                pb_pose_2_copy = pb_pose_2.copy()
                pb_pose_2_copy[:3, 3] += np.array([0, 0, lift_height_2])
                pb_pose_current_2 = pb_pose_2_copy
                
                
                if args.actor == 'expert':
                    manip_trajectory_2 = extract_expert_trajectory(demo, args.task, keyframes, mode='manip', traj_length=traj_length)
                else:
                    language_emb = None
                    if use_language and language_encoder is not None:
                        language_text = get_task_language(args.task, mode='manip')
                        dummy_data = Data()
                        dummy_data['language'] = language_text
                        result = language_encoder(dummy_data)
                        language_emb = result['language_embedding'].unsqueeze(0).to(actor_manip.device)
                    
                    time_start = time.time()
                    manip_trajectory_2 = actor_manip.act(pa_pcd, pb_pcd, pa_pose_current_2, pb_pose_current_2, gripper_pose_current_2, language_emb)
                    infer_time = time.time() - time_start
                    print(f'Second manip inference time: {infer_time:.3f}s')
                
                manip_trajectory_2 = adjust_demo_traj(args.task, manip_trajectory_2, mode='manip', gripper_pose_current=gripper_pose_current_2)
                print(f'Second manip trajectory has {len(manip_trajectory_2)} waypoints')
                
                # Execute second manip trajectory
                print("\nExecuting second manip trajectory...")
                try:
                    for wp_idx, waypoint_matrix in enumerate(manip_trajectory_2):
                        waypoint_pose = Transform.from_matrix(waypoint_matrix)
                        wp_trans, wp_ori = waypoint_pose.translation, waypoint_pose.rotation
                        print(f"  Waypoint {wp_idx+1}/{len(manip_trajectory_2)}: {wp_trans}")
                        wp_action = np.concatenate([wp_trans, wp_ori.as_quat(), [0.0]], axis=-1)
                        res = task.step(wp_action, test_mode=True)
                        time.sleep(0.2)
                    
                    last_waypoint = Transform.from_matrix(manip_trajectory_2[-1])
                    place_trans_2, place_ori_2 = last_waypoint.translation, last_waypoint.rotation
                    place_close_2 = np.concatenate([place_trans_2, place_ori_2.as_quat(), [0.0]], axis=-1)
                    res = task.step(place_close_2, test_mode=True)
                    time.sleep(0.2)
                    
                    print("Opening gripper to release second object...")
                    place_open_2 = np.concatenate([place_trans_2, place_ori_2.as_quat(), [1.0]], axis=-1)
                    obs, reward, terminate = task.step(place_open_2)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"\n✗ Error during second manip execution: {e}")
                    if args.debug:
                        visualize_debug_scene(
                            pa_pcd=pa_pcd,
                            pb_pcd=pb_pcd,
                            pa_pose=pa_pose_current_2,
                            pb_pose=pb_pose_current_2,
                            gripper_pose=gripper_pose_current_2,
                            predicted_traj=manip_trajectory_2,
                            mode='manip_2'
                        )
                
                # Retract after second place
                print("\nRetracting after second place...")
                post_place_trans_2, post_place_ori_2 = trans_along_axis(place_trans_2, place_ori_2, axis='z', delta=0.1)
                post_place_action_2 = np.concatenate([post_place_trans_2, post_place_ori_2.as_quat(), [1.0]], axis=-1)
                res = task.step(post_place_action_2, test_mode=True)
                time.sleep(0.2)
            else:
                print(f"Warning: moving_mask_index_2 not found in config for {args.task}")
        
        # Return to initial position
        obs, reward, terminate = task.step(initial_action)
        
        print(f'\n{"="*60}')
        print(f'Reward: {reward}, Terminate: {terminate}')
        
        if terminate:
            print(colored("✓ Task completed successfully!", "green"))
            episode_status = 'success'
        else:
            print(colored("✗ Task failed", "red"))
            if args.debug:
                visualize_debug_scene(
                    pa_pcd=pa_pcd,
                    pb_pcd=pb_pcd,
                    pa_pose=pa_pose_current,
                    pb_pose=pb_pose_current,
                    gripper_pose=gripper_pose_current,
                    predicted_traj=manip_trajectory,
                    mode='manip'            
                )
        results.append(reward)
        print(f'Test {j+1} result: {"SUCCESS" if reward > 0 else "FAILURE"}')
        if episode_recorder is not None:
            video_name = f'{task.get_name()}_var{args.var}_ep{args.start_test + j}_{episode_status}.mp4'
            episode_recorder.save(os.path.join(args.save_dir, video_name))
    
    # Print summary
    num_scenes = len(results)
    success_rate = np.asarray(results).mean()
    print('')
    print('='*60)
    print(f'Test Summary')
    print('='*60)
    print(f'Task: {args.task}')
    print(f'Total tests: {num_scenes}')
    print(f'Success rate: {success_rate*100:.2f}%')
    print('='*60)

    if episode_recorder is not None:
        episode_recorder.stop()

    env.shutdown()


if __name__ == "__main__":
    main(args)
