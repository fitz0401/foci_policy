import numpy as np
import os
import open3d as o3d
import pickle
from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs
from foci_policy.utils.transform_utils import Rotation, Transform, to_o3d_pcd
from foci_policy.config.config_utils import (
    get_dataloader_config,
    get_task_language,
)
from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
from foci_policy.pose_estimator.object_pose_seq_extractor import (
    extract_poses_for_demo_gt,
)
from foci_policy.pose_estimator.foundation_pose_utils import (
    extract_poses_for_demo,
)
from compute_interaction_intervals import compute_intervals_for_demo
from utils_rlbench.process_demo import get_pick_place_keyframes, flat_pcd_image

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_RAW_DATA_PATH = os.path.join(dir_path, '../../data/rlbench_data')
DEFAULT_SAVE_DATA_PATH = os.path.join(dir_path, '../../foci_dataset')

_WARNED_MASK_ISSUES = set()


def _warn_once(key, message):
    if key in _WARNED_MASK_ISSUES:
        return
    _WARNED_MASK_ISSUES.add(key)
    print(message)


def _fmt_keys(keys, limit=8):
    keys = list(keys)
    if len(keys) <= limit:
        return keys
    extra = len(keys) - limit
    return keys[:limit] + [f'...(+{extra} more)']


def _to_numpy_array(x):
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    return np.asarray(x)


def get_camera_masks_for_frame(frame, task_name, cameras, variation=0):
    """Return per-camera 2D masks for the moving and base objects at one frame. """
    ext = DemoObjectExtractor.from_config(task_name)
    ext.variation = variation
    masks = {}
    for cam in cameras:
        rgb = getattr(frame, f'{cam}_rgb', None)
        h, w = (rgb.shape[:2] if rgb is not None else (128, 128))
        try:
            moving_mask = ext.get_grasp_mask(frame, camera_name=cam)
        except Exception:
            moving_mask = np.zeros((h, w), dtype=np.uint8)
        if ext.target_obj_name:
            try:
                base_mask = ext.get_target_mask(frame, camera_name=cam)
            except Exception:
                base_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            base_mask = np.zeros((h, w), dtype=np.uint8)
        masks[cam] = {
            'moving': moving_mask.astype(np.uint8),
            'base':   base_mask.astype(np.uint8),
        }
    return masks


def _align_binary_mask(mask, target_len):
    mask = _to_numpy_array(mask).reshape(-1)
    if mask.dtype != np.bool_:
        mask = mask > 0

    if mask.size == target_len:
        return mask

    if mask.size % target_len == 0:
        # Some masks are flattened from multi-channel images (e.g., RGB),
        # so collapse channel-expanded entries back to one bool per point.
        channel_factor = mask.size // target_len
        return mask.reshape(target_len, channel_factor).any(axis=1)

    if target_len % mask.size == 0:
        return np.repeat(mask, target_len // mask.size)

    raise ValueError(
        f"Cannot align mask of length {mask.size} to target length {target_len}."
    )


def get_gt_flat_masks_for_frame(frame, task_name, variation, cameras):
    """Return (pa_flat_mask, pb_flat_mask) using DemoObjectExtractor.from_config().

    Uses the explicit grasp/target object names from task_obj_config.yaml.
    Reads per-episode handle IDs from frame.misc['gt_objects'] (enriched demos)
    or falls back to the auto-discover cache.  Masks are H×W per camera,
    concatenated across cameras in the same order as flat_pcd_image.

    pa = target/base, pb = grasp/moving.
    """
    ext = DemoObjectExtractor.from_config(task_name)
    ext.variation = variation
    misc = getattr(frame, 'misc', None) or {}
    gt_objs = misc.get('gt_objects') or {}
    ext.resolve_auto_names(frame=frame)
    if ext._auto_target and ext.target_obj_name is None:
        _warn_once(
            (task_name, 'target', 'auto_missing'),
            f"Warning: auto target name not resolved for task '{task_name}'. "
            f"misc has chosen_pillar_name={misc.get('chosen_pillar_name')}")
    _grasp_names = ext.grasp_obj_name if isinstance(ext.grasp_obj_name, list) else ([ext.grasp_obj_name] if ext.grasp_obj_name else [])
    _target_names = ext.target_obj_name if isinstance(ext.target_obj_name, list) else ([ext.target_obj_name] if ext.target_obj_name else [])
    if _grasp_names and gt_objs and not any(n in gt_objs for n in _grasp_names):
        _warn_once(
            (task_name, 'grasp', str(ext.grasp_obj_name), 'missing_handle'),
            f"Warning: gt_objects missing grasp '{ext.grasp_obj_name}' for task '{task_name}'. "
            f"gt_objects keys={_fmt_keys(gt_objs.keys())}")
    if _target_names and gt_objs and not any(n in gt_objs for n in _target_names):
        _warn_once(
            (task_name, 'target', str(ext.target_obj_name), 'missing_handle'),
            f"Warning: gt_objects missing target '{ext.target_obj_name}' for task '{task_name}'. "
            f"gt_objects keys={_fmt_keys(gt_objs.keys())}")
    pa_parts, pb_parts = [], []
    for cam in cameras:
        rgb = getattr(frame, f'{cam}_rgb', None)
        h, w = (rgb.shape[:2] if rgb is not None else (128, 128))
        try:
            pb_mask = ext.get_grasp_mask(frame, camera_name=cam)
        except Exception as e:
            _warn_once(
                (task_name, 'grasp', str(ext.grasp_obj_name), 'mask_error'),
                f"Warning: failed to get grasp mask for task '{task_name}', "
                f"name='{ext.grasp_obj_name}', camera='{cam}': {e}")
            pb_mask = np.zeros((h, w), dtype=np.uint8)
        if ext.target_obj_name:
            try:
                pa_mask = ext.get_target_mask(frame, camera_name=cam)
            except Exception as e:
                _warn_once(
                    (task_name, 'target', str(ext.target_obj_name), 'mask_error'),
                    f"Warning: failed to get target mask for task '{task_name}', "
                    f"name='{ext.target_obj_name}', camera='{cam}': {e}")
                pa_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            pa_mask = np.zeros((h, w), dtype=np.uint8)
        pb_parts.append(pb_mask.reshape(-1).astype(bool))
        pa_parts.append(pa_mask.reshape(-1).astype(bool))
    return np.concatenate(pa_parts), np.concatenate(pb_parts)


def get_gripper_pose(frame):
    translation = frame.gripper_pose[:3]
    quat = frame.gripper_pose[3:]
    quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
    if quat[-1] < 0:
        quat = -quat
    trans = Transform(rotation=Rotation.from_quat(quat), translation=translation)
    return trans


def save_data_from_demo(root_dir, save_data_path, task_name='open_drawer',
                        var_num=0, episode_idx=0, disp=False,
                        voxel_size=0.004, num_points=2048,
                        pose_method='gt', mesh_dir=None, cameras=None,
                        postprocess=True, pen_factor=None):
    TASK = task_name
    VAR_NUM = var_num
    ROOT_DIR = root_dir
    ## 0. Find first matching task_name* directory
    import glob
    task_dirs = sorted(glob.glob(os.path.join(ROOT_DIR, f'{TASK}*')))
    if not task_dirs:
        raise FileNotFoundError(f"No directory matching {TASK}* found in {ROOT_DIR}")
    task_dir = task_dirs[0]
    EPISODES_FOLDER = f'variation{VAR_NUM}/episodes'
    if cameras is not None and len(cameras) > 0:
        CAMERAS = cameras
    else:
        CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
    data_path = os.path.join(task_dir, EPISODES_FOLDER)
    demo = get_stored_demo(data_path=data_path, index=episode_idx, init_matrix=False)

    ## 1. Object Masks
    save_folder = os.path.join(save_data_path, task_name)
    os.makedirs(save_folder, exist_ok=True)

    use_gt_masks = (pose_method == 'gt')
    first_frame_masks = None
    if not use_gt_masks:
        first_camera = CAMERAS[0]
        first_frame_masks = get_camera_masks_for_frame(demo[0], task_name, [first_camera], variation=var_num)[first_camera]

    ## 2. Key Frames
    keyframes = get_pick_place_keyframes(demo, task_name=task_name)
    print('TASK: {}, VAR: {}, EPISODE ID: {}, NUM_STEP {}, PICK_T {}, PLACE_T {}'\
          .format(task_name, var_num, episode_idx, len(demo), keyframes['pick'], keyframes['place']))

    # 3. Object Poses
    poses_data = None
    if pose_method == 'gt':
        print(f"Extracting GT poses (postprocess={postprocess}) ...")
        poses_data = extract_poses_for_demo_gt(
            demo=demo,
            task_name=task_name,
            keyframes=keyframes,
            variation=var_num,
            postprocess=postprocess,
            verbose=False,
        )
    elif pose_method == 'fp' or pose_method == 'foundation_pose':
        # Get object poses via Foundation Pose
        if mesh_dir is None:
            mesh_dir = os.path.join(dir_path, '../assets/RLBench_mesh')
        print(f"Running Foundation Pose extraction...")
        poses_data = extract_poses_for_demo(
            demo=demo,
            task_name=task_name,
            mesh_dir=mesh_dir,
            camera_name=CAMERAS[0],
            verbose=False,
            postprocess=True,
            first_frame_mask=first_frame_masks,
        )
        if poses_data['success']:
            print(f"Foundation Pose complete. Got {len(poses_data['moving_poses'])} poses")
        else:
            raise RuntimeError("Foundation Pose extraction failed.")

    ## 4. Frame Data (pcd, poses, masks)
    frames_data = []
    prev_pa_points = None
    prev_pa_colors = None
    prev_pb_points = None
    prev_pb_colors = None
    for t, frame in enumerate(demo):
        # Extract point clouds for current frame
        frame_obs_dict = extract_obs(frame, CAMERAS, t=t, prev_action=None)
        frame_pcd_flat, frame_flat_features, frame_flat_mask, frame_pcds = flat_pcd_image(
            frame_obs_dict, CAMERAS, include_mask=True,
        )
        pa_binary_mask, pb_binary_mask = get_gt_flat_masks_for_frame(
            frame, task_name, var_num, CAMERAS,
        )
        frame_points = _to_numpy_array(frame_pcd_flat[0])
        frame_colors = _to_numpy_array(frame_flat_features[0])
        pa_binary_mask = _align_binary_mask(pa_binary_mask, frame_points.shape[0])
        pb_binary_mask = _align_binary_mask(pb_binary_mask, frame_points.shape[0])
        frame_pa_points = frame_points[pa_binary_mask]
        frame_pa_colors = frame_colors[pa_binary_mask]
        frame_pb_points = frame_points[pb_binary_mask]
        frame_pb_colors = frame_colors[pb_binary_mask]

        # Get poses
        gripper_pose = get_gripper_pose(frame)
        # Using GT/ICP/Foundation Pose sequence poses
        pa_pose_mat = poses_data['base_poses'][t] if t in poses_data['base_poses'] else None
        pb_pose_mat = poses_data['moving_poses'][t] if t in poses_data['moving_poses'] else None
        
        # Voxel downsampling for memory efficiency (4mm grid)
        pa_pcd = to_o3d_pcd(frame_pa_points, frame_pa_colors)
        pb_pcd = to_o3d_pcd(frame_pb_points, frame_pb_colors)
        pa_pcd = pa_pcd.voxel_down_sample(voxel_size=voxel_size)
        pb_pcd = pb_pcd.voxel_down_sample(voxel_size=voxel_size)

        # Convert back to numpy arrays and ensure num_points points
        # If empty, fill with previous frame's points or zeros
        if len(pa_pcd.points) > 0:
            if len(pa_pcd.points) > num_points:
                indices = np.random.choice(len(pa_pcd.points), num_points, replace=False)
            else:
                indices = np.random.choice(len(pa_pcd.points), num_points, replace=True)
            frame_pa_points = np.asarray(pa_pcd.points)[indices]
            frame_pa_colors = np.asarray(pa_pcd.colors)[indices]
            prev_pa_points = frame_pa_points
            prev_pa_colors = frame_pa_colors
        else:
            if prev_pa_points is not None and prev_pa_colors is not None:
                frame_pa_points = prev_pa_points
                frame_pa_colors = prev_pa_colors
            else:
                frame_pa_points = np.zeros((num_points, 3), dtype=np.float32)
                frame_pa_colors = np.zeros((num_points, 3), dtype=np.float32)
        if len(pb_pcd.points) > 0:
            if len(pb_pcd.points) > num_points:
                indices = np.random.choice(len(pb_pcd.points), num_points, replace=False)
            else:
                indices = np.random.choice(len(pb_pcd.points), num_points, replace=True)
            frame_pb_points = np.asarray(pb_pcd.points)[indices]
            frame_pb_colors = np.asarray(pb_pcd.colors)[indices]
            prev_pb_points = frame_pb_points
            prev_pb_colors = frame_pb_colors
        else:
            if prev_pb_points is not None and prev_pb_colors is not None:
                frame_pb_points = prev_pb_points
                frame_pb_colors = prev_pb_colors
            else:
                frame_pb_points = np.zeros((num_points, 3), dtype=np.float32)
                frame_pb_colors = np.zeros((num_points, 3), dtype=np.float32)
        
        frames_data.append({
            'timestep': t,
            'gripper_pose_mat': gripper_pose.as_matrix(),
            'pa_pose_mat': pa_pose_mat,
            'pb_pose_mat': pb_pose_mat,
            'pa_points': frame_pa_points,
            'pa_colors': frame_pa_colors,
            'pb_points': frame_pb_points,
            'pb_colors': frame_pb_colors,
        })

    ## 5. Interaction Intervals
    intervals_result = None
    try:
        print(f"Computing interaction intervals...")
        intervals_result = compute_intervals_for_demo(
            demo=demo,
            task_name=task_name,
            expansion_factor=1.0,
            use_change_detection=True,
            pen_factor=pen_factor,
            keyframes=keyframes,
            cameras=CAMERAS,
            mask_provider=(
                lambda t, _obs_dict, _flat_mask: get_gt_flat_masks_for_frame(demo[t], task_name, var_num, CAMERAS)
            ),
        )
        if intervals_result:
            print(f"  Grasp interval: {intervals_result['grasp_interval']} frames")
            print(f"  Manip interval: {intervals_result['manip_interval']} frames")
    except Exception as e:
        raise RuntimeError(f"Error computing interaction intervals: {e}")

    data_dic = {
        'task_name': task_name,
        'total_frames': len(demo),
        'frames_data': frames_data,
        'key_times': keyframes,
        'lan_pick': get_task_language(task_name, mode='grasp'),
        'lan_place': get_task_language(task_name, mode='manip'),
        'grasp_interval': intervals_result['grasp_interval'],
        'manip_interval': intervals_result['manip_interval'],
        'fp_moving_score': poses_data.get('moving_reg_score') if poses_data is not None else None,
        'fp_base_score': poses_data.get('base_reg_score') if poses_data is not None else None,
    }
    
    num_pkl_files = len([f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f)) and f.endswith('.pkl')])
    # assert num_pkl_files <= episode_idx # delete the fold and preprocess again
    fname = '{}.pkl'.format(num_pkl_files)
    with open(os.path.join(save_folder, fname), 'wb') as f:
        pickle.dump(data_dic, f)
    
    if disp:  
        try:
            print(f"Showing key frames in 3D...")          
            for frame_name, frame_idx in [('pick', keyframes['pick']), ('place', keyframes['place'])]:
                if frame_idx < len(frames_data):
                    geometries = []
                    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    # geometries.append(coord_frame)
                    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    gripper_frame.transform(frames_data[frame_idx]['gripper_pose_mat'])
                    geometries.append(gripper_frame)
                    
                    frame_pa_points = frames_data[frame_idx]['pa_points']
                    frame_pa_colors = frames_data[frame_idx]['pa_colors']
                    frame_pb_points = frames_data[frame_idx]['pb_points']
                    frame_pb_colors = frames_data[frame_idx]['pb_colors']
                    
                    # object_0 (base object)
                    pa_pose_mat = frames_data[frame_idx]['pa_pose_mat']
                    if pa_pose_mat is not None and len(frame_pa_points) > 0:
                        pa_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                        pa_frame.transform(pa_pose_mat)
                        geometries.append(pa_frame)
                        pa_pcd = to_o3d_pcd(frame_pa_points, frame_pa_colors)
                        pa_pcd.paint_uniform_color([0, 0, 1])  # blue
                        geometries.append(pa_pcd)
                        
                    # object_1 (moving object)
                    pb_pose_mat = frames_data[frame_idx]['pb_pose_mat']
                    if pb_pose_mat is not None and len(frame_pb_points) > 0:
                        pb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                        pb_frame.transform(pb_pose_mat)
                        geometries.append(pb_frame)
                        
                        pb_pcd = to_o3d_pcd(frame_pb_points, frame_pb_colors)
                        pb_pcd.paint_uniform_color([0, 1, 0])  # green
                        geometries.append(pb_pcd)
                    
                    print(f"Showing {frame_name} frame (index: {frame_idx}). Close window to continue.")
                    o3d.visualization.draw_geometries(
                        geometries,
                        window_name=f'{task_name}_{frame_name} (Frame {frame_idx})',
                        width=800,
                        height=600
                    )           
        except Exception as e:
            print(f"Warning: Could not show 3D scenes: {e}")
            
def build_dataset(num_demos=10, root_dir=DEFAULT_RAW_DATA_PATH, save_data_path=DEFAULT_SAVE_DATA_PATH, task_name='open_drawer',
                  var_num=0, disp=False, voxel_size=0.004, num_points=2048,
                  pose_method='gt', mesh_dir=None, cameras=None, postprocess=True,
                  pen_factor=None):
    for i in range(num_demos):
        save_data_from_demo(root_dir=root_dir, save_data_path=save_data_path, task_name=task_name, var_num=var_num,
                            episode_idx=i, disp=disp, voxel_size=voxel_size,
                            num_points=num_points, pose_method=pose_method, mesh_dir=mesh_dir, cameras=cameras,
                            postprocess=postprocess, pen_factor=pen_factor)
        

import argparse

parser = argparse.ArgumentParser(description='Preprocess Raw RLBench demos')
parser.add_argument('--task_name', type=str, default='all')
parser.add_argument('--num_demos', type=int, default=1)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--voxel_size', type=float, default=0.004)
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--pose_method', type=str, default='gt', 
                    choices=['gt', 'fp', 'foundation_pose'],
                    help='Pose extraction method: gt (ground truth), fp (Foundation Pose)')
parser.add_argument('--mesh_dir', type=str, default=None,
                    help='Mesh directory (required for Foundation Pose)')
parser.add_argument('--raw_data_path', type=str, default=DEFAULT_RAW_DATA_PATH,
                    help='Path to raw RLBench data (default: %(default)s)')
parser.add_argument('--save_data_path', type=str, default=DEFAULT_SAVE_DATA_PATH,
                    help='Path to save processed data (default: %(default)s)')
parser.add_argument('--cameras', type=str, default=None,
                    help='Optional comma-separated camera override, e.g. front or front,left_shoulder,right_shoulder,wrist')
parser.add_argument('--no_postprocess', action='store_true', default=False,
                    help='Disable trajectory post-processing for GT poses')

args = parser.parse_args()

if args.pose_method in ['fp', 'foundation_pose'] and args.mesh_dir is None:
    print("Warning: --mesh_dir required for Foundation Pose. Using default: ../assets/RLBench_mesh")

camera_list = None
if args.cameras is not None:
    camera_list = [c.strip() for c in args.cameras.split(',') if c.strip()]
    if len(camera_list) == 0:
        camera_list = None

postprocess = not args.no_postprocess
dataloader_cfg = get_dataloader_config()

if args.task_name == 'all':
    task_list = dataloader_cfg.get('task_list', [])
    if not isinstance(task_list, list):
        raise RuntimeError('task_list in dataloader.yaml must be a list')
    for task in task_list:
        build_dataset(num_demos=args.num_demos, root_dir=args.raw_data_path, save_data_path=args.save_data_path, task_name=task, disp=args.disp,
                      voxel_size=args.voxel_size, num_points=args.num_points,
                      pose_method=args.pose_method, mesh_dir=args.mesh_dir, cameras=camera_list,
                      postprocess=postprocess)
else:
    build_dataset(num_demos=args.num_demos, root_dir=args.raw_data_path, save_data_path=args.save_data_path, task_name=args.task_name, disp=args.disp,
                  voxel_size=args.voxel_size, num_points=args.num_points,
                  pose_method=args.pose_method, mesh_dir=args.mesh_dir, cameras=camera_list,
                  postprocess=postprocess)
