"""
Foundation Pose utilities for RLBench.
"""

import numpy as np
import cv2
import os
from foundation_pose.wrapper import FoundationPoseWrapper
from scipy.spatial.transform import Rotation
from foci_policy.config.config_utils import get_task_obj_config, get_mesh_names
from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
from utils_rlbench.process_demo import get_pick_place_keyframes
BASE_STABILITY_THRESHOLD = 0.02  # meters, std of base object position
MOVING_DISP_REL_DIFF_THRESHOLD = 0.1  # relative difference threshold between moving object and gripper


def _to_uint8_rgb(rgb):
    rgb = np.asarray(rgb)
    # RLBench may store RGB as CHW; convert to HWC for PIL/OpenCV consumers.
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _to_binary_mask_2d(mask):
    """Normalize mask to a 2D uint8 binary image for FoundationPose/OpenCV."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            # If multiple mask proposals exist, merge them conservatively.
            mask = (mask > 0).any(axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after normalization, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def preprocess_depth(depth, max_depth=2.0):
    """Preprocess depth map with bilateral filtering."""
    depth = np.clip(depth, 0.01, max_depth)
    depth_mm = (depth * 1000).astype(np.float32)
    depth_filtered = cv2.bilateralFilter(depth_mm, d=5, sigmaColor=10, sigmaSpace=10)
    return (depth_filtered / 1000.0).astype(np.float32)


def validate_camera_intrinsics(K):
    """Handle negative focal lengths (OpenGL convention)."""
    K = K.copy()
    K[0, 0] = abs(K[0, 0])
    K[1, 1] = abs(K[1, 1])
    return K


def get_camera_data(frame, camera_name='front'):
    """Extract RGB, depth (in meters), mask, extrinsic, intrinsic from RLBench frame."""
    rgb = getattr(frame, f'{camera_name}_rgb')
    depth_raw = getattr(frame, f'{camera_name}_depth')
    mask = getattr(frame, f'{camera_name}_mask')
    if rgb is None or depth_raw is None:
        raise ValueError(
            f"Camera '{camera_name}' image data not stored in this demo "
            f"(rgb={'ok' if rgb is not None else 'None'}, depth={'ok' if depth_raw is not None else 'None'}). "
            f"Demo was recorded without this camera enabled."
        )
    extrinsic = frame.misc.get(f'{camera_name}_camera_extrinsics')  # camera to world
    intrinsic = frame.misc.get(f'{camera_name}_camera_intrinsics')
    intrinsic = validate_camera_intrinsics(intrinsic)
    near = frame.misc.get(f'{camera_name}_camera_near', 0.01)
    far = frame.misc.get(f'{camera_name}_camera_far', 4.5)
    # Note: We need depth to be in meters for Foundation Pose
    # For online execution, depth is already in meters
    if depth_raw.max() <= 1.0 and depth_raw.min() >= 0:
        depth = near + depth_raw * (far - near)
    else:
        depth = depth_raw
    depth = preprocess_depth(depth)
    return rgb, depth, mask, extrinsic, intrinsic


def force_z_axis_up(pose):
    """
    Aggressively force Z-axis to point straight up (0, 0, 1).
    Preserves translation and tries to keep X-axis direction when possible.
    """
    translation = pose[:3, 3].copy()
    R = pose[:3, :3]
    
    # Try to preserve X-axis direction
    x_axis = R[:, 0].copy()
    z_axis_target = np.array([0.0, 0.0, 1.0])
    
    # Compute Y-axis as Z × X (right-hand rule)
    y_axis = np.cross(z_axis_target, x_axis)
    y_norm = np.linalg.norm(y_axis)
    
    if y_norm < 1e-6:
        # X-axis is already along Z; pick arbitrary orthogonal X-axis
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.cross(z_axis_target, x_axis)
        y_norm = np.linalg.norm(y_axis)
    
    y_axis = y_axis / y_norm
    # Recompute X to ensure orthogonality: X = Y × Z
    x_axis = np.cross(y_axis, z_axis_target)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Build new rotation matrix
    R_new = np.column_stack([x_axis, y_axis, z_axis_target])
    
    # Build result pose
    pose_new = np.eye(4)
    pose_new[:3, :3] = R_new
    pose_new[:3, 3] = translation
    return pose_new


def ensure_z_axis_up(pose):
    """ Ensure object's z-axis points upward (positive z direction). """
    # Extract z-axis from rotation matrix (third column)
    z_axis = pose[:3, 2]
    world_z = np.array([0, 0, 1])
    # Check if z-axis is pointing down (dot product with world z-axis is negative)
    if np.dot(z_axis, world_z) < 0:
        # Rotate 180 degrees around x-axis to flip z-axis
        R_flip = np.eye(4)
        R_flip[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()
        pose = pose @ R_flip
    return pose


def extract_poses_from_observation(obs, task_name, mesh_dir, camera_name='front',
                                   estimators=None, verbose=False,
                                   first_frame_mask=None, moving_object_index=0,
                                   task=None, save_debug=False):
    """Extract object poses from current observation.
    Args:
        moving_object_index: Which moving object to extract (0=first, 1=second for stacking tasks).
                            When 1, reads moving_mask_index_2 from config instead of moving_mask_index.
        task: optional live PyRep task object, used to resolve __AUTO__ names in online mode.
    """
    stage = 2 if moving_object_index == 1 else 1
    _ext = DemoObjectExtractor.from_config(task_name, stage=stage)
    # Resolve __AUTO__ names from demo frame misc or live task object
    _ext.resolve_auto_names(frame=obs, task=task)
    _grasp_name = _ext.grasp_obj_name
    _target_name = _ext.target_obj_name
    if isinstance(_grasp_name, list):
        _grasp_name = _grasp_name[0]
    if isinstance(_target_name, list):
        _target_name = _target_name[0]
    objects = [n for n in [_grasp_name, _target_name] if n is not None]
    # Use role-based keys to avoid collision when grasp/target names are identical
    roles = ['grasp', 'target'][:len(objects)]
    _mesh_names = get_mesh_names(task_name) or objects
    mesh_names = _mesh_names[:len(objects)]
    if estimators is None:
        task_mesh_dir = os.path.join(mesh_dir, task_name)
        estimators = {}
        for role, obj_name, mesh_name in zip(roles, objects, mesh_names):
            debug_dir = f'./fp_debug/{task_name}_{obj_name}' if save_debug else None
            wrapper = FoundationPoseWrapper(
                mesh_dir=task_mesh_dir,
                debug_dir=debug_dir
            )
            wrapper.update_grasp_obj_name(mesh_name)
            debug_level = 2 if save_debug else -1
            estimators[role] = wrapper.create_estimator(debug_level=debug_level)
        if verbose:
            print(f"Initialized Foundation Pose for {task_name}")
    rgb, depth, mask, cam_extrinsic, K = get_camera_data(obs, camera_name)
    # Flip rotation to convert from OpenGL to RLBench coordinate system
    R_flip = np.eye(4)
    R_flip[:3, :3] = Rotation.from_euler('zyx', [np.pi, 0, 0]).as_matrix()
    base_pose = None
    moving_pose = None
    moving_reg_score = None
    base_reg_score = None
    for obj_idx, (role, obj_name) in enumerate(zip(roles, objects)):
        estimator = estimators[role]
        try:
            # First successful frame for each object: register using mask
            if estimator.pose_last is None:
                if first_frame_mask is not None:
                    obj_mask = first_frame_mask["moving" if obj_idx == 0 else "base"]
                    if obj_mask.ndim == 3:
                        obj_mask = obj_mask[:, :, 0]  # take R channel if RGB mask slipped through
                else:
                    # Use DemoObjectExtractor live-mode masks from the current observation
                    _live_masks = _ext.get_masks_live(mask)
                    _role = 'grasp' if obj_idx == 0 else 'target'
                    obj_mask = _live_masks.get(_role)
                    if obj_mask is None:
                        raise RuntimeError(
                            f"DemoObjectExtractor could not resolve live mask for '{obj_name}' "
                            f"(role={_role}) — ensure a running CoppeliaSim/PyRep environment."
                        )
                    if obj_mask.ndim == 3:
                        obj_mask = obj_mask[..., 0]
                if obj_mask.sum() < 100 and verbose:
                    print(f"  Warning: {obj_name} mask too small ({obj_mask.sum()} pixels)")
                
                pose_cam = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=obj_mask, iteration=10)
                pose_world = cam_extrinsic @ R_flip @ pose_cam
                # Ensure z-axis points upward for consistent orientation (only for the first frame)
                # pose_world = ensure_z_axis_up(pose_world)
                pose_world = force_z_axis_up(pose_world)
                # score is only available when register fully ran (pose_last is set on success)
                if estimator.pose_last is not None and hasattr(estimator, 'scores'):
                    try:
                        reg_score = float(estimator.scores[0])
                    except Exception:
                        reg_score = None
                else:
                    reg_score = None
                if verbose:
                    score_info = ("%.4f" % reg_score) if reg_score is not None else "N/A (registration may have failed — mask too small?)"
                    print(f"  {obj_name} register score: {score_info}")
            else:
                pose_cam = estimator.track_one(K=K, rgb=rgb, depth=depth, iteration=2)
                pose_world = cam_extrinsic @ R_flip @ pose_cam
                reg_score = None  # scorer not called during tracking

            if obj_idx == 0:
                moving_pose = pose_world
                moving_reg_score = reg_score
            else:
                base_pose = pose_world
                base_reg_score = reg_score
            if verbose:
                print(f"  {obj_name}: {pose_world[:3, 3]}")
        except Exception as e:
            if verbose:
                print(f"  {obj_name} failed: {e}")
            continue
    success = (base_pose is not None) and (moving_pose is not None)
    return {
        'base_pose': base_pose,
        'moving_pose': moving_pose,
        'success': success,
        'estimators': estimators,
        'moving_reg_score': moving_reg_score,
        'base_reg_score': base_reg_score,
    }


def get_gripper_pose(frame):
    translation = frame.gripper_pose[:3]
    quat = frame.gripper_pose[3:]
    quat = quat / np.linalg.norm(quat)
    if quat[-1] < 0:
        quat = -quat
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
    pose_mat[:3, 3] = translation
    return pose_mat


def validate_pose_estimation(poses_data, demo, keyframes, verbose=False):
    """Validate quality of pose estimation. """
    moving_poses = poses_data.get('moving_poses', {})
    base_poses = poses_data.get('base_poses', {})
    pick_frame = keyframes.get('pick', len(demo) // 3)
    place_frame = keyframes.get('place', 2 * len(demo) // 3)
    result = {
        'moving_valid': False,
        'base_valid': False
    }
    
    # Validate moving object: compare start/end displacement with gripper
    if len(moving_poses) > 0:
        frames_in_range = [t for t in range(pick_frame, min(place_frame + 1, len(demo))) 
                           if t in moving_poses]
        if len(frames_in_range) >= 2:
            t0, t1 = frames_in_range[0], frames_in_range[-1]
            gripper_start = get_gripper_pose(demo[t0])[:3, 3]
            gripper_end = get_gripper_pose(demo[t1])[:3, 3]
            object_start = moving_poses[t0][:3, 3]
            object_end = moving_poses[t1][:3, 3]
            gripper_object_start_dist = np.linalg.norm(object_start - gripper_start)
            gripper_object_end_dist = np.linalg.norm(object_end - gripper_end)
            result['moving_valid'] = gripper_object_start_dist < MOVING_DISP_REL_DIFF_THRESHOLD and \
                gripper_object_end_dist < MOVING_DISP_REL_DIFF_THRESHOLD
            if verbose:
                print(f"\nMoving object validation:")
                print(f"  Frames analyzed: {len(frames_in_range)} (pick to place)")
                print(f"  Gripper start-object start dist: {gripper_object_start_dist:.4f} m, \
                      Gripper end-object end dist: {gripper_object_end_dist:.4f} m")
                print(f"  Valid: {result['moving_valid']} (threshold: {MOVING_DISP_REL_DIFF_THRESHOLD})")
        else:
            result['moving_valid'] = False
    
    # Validate base object: check position stability
    if len(base_poses) > 0:
        base_positions = np.array([base_poses[t][:3, 3] for t in sorted(base_poses.keys())])
        position_std = base_positions.std(axis=0)
        max_std = position_std.max()
        result['base_std'] = max_std
        result['base_valid'] = max_std < BASE_STABILITY_THRESHOLD
        if verbose:
            print(f"\nBase object validation:")
            print(f"  Frames analyzed: {len(base_poses)}")
            print(f"  Position std: [{position_std[0]:.4f}, {position_std[1]:.4f}, {position_std[2]:.4f}]m")
            print(f"  Max std: {max_std:.4f}m")
            print(f"  Valid: {result['base_valid']} (threshold: {BASE_STABILITY_THRESHOLD})")
    return result


def extract_poses_for_demo(demo, task_name, mesh_dir, start_frame=0, end_frame=None,
                           camera_name='front', verbose=False, postprocess=True,
                           first_frame_mask=None):
    """Extract poses for an entire RLBench demo sequence using Foundation Pose.
    If first_frame_mask is provided, it will be used for the first frame.
    """
    if end_frame is None:
        end_frame = len(demo) - 1
    # Fail fast if camera data is unavailable — avoids an infinite re-init loop
    # where estimators are created and discarded every frame due to a silent exception.
    try:
        get_camera_data(demo[start_frame], camera_name)
    except (ValueError, AttributeError) as e:
        raise RuntimeError(
            f"Camera '{camera_name}' data is not available in this demo: {e}"
        ) from e
    if verbose:
        print(f"Extracting {task_name}: frames {start_frame}-{end_frame}")
    estimators = None
    base_poses = {}
    moving_poses = {}
    moving_reg_score = None
    base_reg_score = None
    for t in range(start_frame, end_frame + 1):
        try:
            result = extract_poses_from_observation(
                obs=demo[t],
                task_name=task_name,
                mesh_dir=mesh_dir,
                camera_name=camera_name,
                estimators=estimators,
                verbose=verbose,
                first_frame_mask=first_frame_mask,
            )
            estimators = result['estimators']
            if result['base_pose'] is not None:
                base_poses[t] = result['base_pose']
            if result['moving_pose'] is not None:
                moving_poses[t] = result['moving_pose']
            # Capture registration scores from the first frame only
            if t == start_frame:
                moving_reg_score = result.get('moving_reg_score')
                base_reg_score = result.get('base_reg_score')
        except Exception as e:
            if verbose:
                print(f"  [Pose] Frame {t} failed: {e}")
            continue         
        if verbose and (t - start_frame) % 50 == 0:
            print(f"  Frame {t}/{end_frame}")  
    if verbose:
        print(f"  Complete: {len(moving_poses)} moving, {len(base_poses)} base poses")
    if verbose:
        print(f"  [FP Score] moving obj register score: "
              f"{('%.4f' % moving_reg_score) if moving_reg_score is not None else 'N/A'} "
              f"| base obj register score: "
              f"{('%.4f' % base_reg_score) if base_reg_score is not None else 'N/A'} "
              f"(higher = better hypothesis match)")
    
    poses_data = {
        'base_poses': base_poses,
        'moving_poses': moving_poses,
        'success': len(moving_poses) > 0 and len(base_poses) > 0,
        'num_frames': end_frame - start_frame + 1,
        'moving_reg_score': moving_reg_score,
        'base_reg_score': base_reg_score,
    }

    if not postprocess:
        poses_data['postprocessed'] = False
        return poses_data

    keyframes = get_pick_place_keyframes(demo, task_name=task_name)
    print(f"  [FP Postprocess] Applying postprocessing...")
    poses_data = postprocess_object_trajectory(
        poses_data=poses_data,
        demo=demo,
        keyframes=keyframes,
        fix_base=True,
        verbose=verbose
    )
    poses_data['postprocessed'] = True
    return poses_data


def postprocess_object_trajectory(poses_data, demo, keyframes, fix_base=False, verbose=False):
    """ Postprocess object trajectory using gripper motion. """
    moving_poses = poses_data.get('moving_poses', {})
    
    if len(moving_poses) == 0:
        if verbose:
            print("Warning: No moving object poses to postprocess")
        return poses_data
    pick_frame = keyframes.get('pick', len(demo) // 3)
    place_frame = keyframes.get('place', 2 * len(demo) // 3)
    if verbose:
        print(f"\nPostprocessing moving object trajectory:")
        print(f"  Pick frame: {pick_frame}, Place frame: {place_frame}")
        print(f"  Original: {len(moving_poses)} poses")
    
    # Step 1: Find first successful pose
    sorted_frames = sorted(moving_poses.keys())
    if len(sorted_frames) == 0:
        if verbose:
            print("  Error: No valid poses found")
        return poses_data
    first_frame = sorted_frames[0]
    first_pose = moving_poses[first_frame]
    if verbose:
        print(f"  First valid pose at frame {first_frame}")
    
    # Step 2: Fill 0 to pick with first pose
    refined_moving_poses = {}
    for t in range(0, min(pick_frame + 1, len(demo))):
        refined_moving_poses[t] = first_pose.copy()
    if verbose:
        print(f"  Filled frames 0-{pick_frame} with first pose")
    
    # Step 3: Compute gripper-to-object transform at pick frame
    if pick_frame < len(demo):
        gripper_pose_at_pick = get_gripper_pose(demo[pick_frame])
        # Use pick frame pose if available, otherwise use first pose
        object_pose_at_pick = refined_moving_poses.get(pick_frame, first_pose)
        gripper_to_object = np.linalg.inv(gripper_pose_at_pick) @ object_pose_at_pick
        if verbose:
            print(f"  Computed gripper-to-object transform at pick")
        
        # Step 4: From pick + 1 to place, compute from gripper motion
        for t in range(pick_frame + 1, min(place_frame + 1, len(demo))):
            gripper_pose = get_gripper_pose(demo[t])
            refined_moving_poses[t] = gripper_pose @ gripper_to_object
        if verbose:
            print(f"  Computed trajectory from gripper motion (pick to place)")
        
        # Step 5: After place, keep stationary
        if place_frame < len(demo):
            place_pose = refined_moving_poses[place_frame]
            for t in range(place_frame + 1, len(demo)):
                refined_moving_poses[t] = place_pose.copy()
            if verbose:
                print(f"  Kept object stationary after place")
    
    # Update poses_data
    updated_poses_data = poses_data.copy()
    updated_poses_data['moving_poses'] = refined_moving_poses
    
    # Fix base object if needed
    if fix_base:
        base_poses = poses_data.get('base_poses', {})
        if len(base_poses) > 0:
            sorted_base_frames = sorted(base_poses.keys())
            first_base_frame = sorted_base_frames[0]
            first_base_pose = base_poses[first_base_frame]
            refined_base_poses = {}
            for t in range(0, len(demo)):
                refined_base_poses[t] = first_base_pose.copy()
            updated_poses_data['base_poses'] = refined_base_poses
            if verbose:
                print(f"\nBase object postprocessing:")
                print(f"  Using first valid pose (frame {first_base_frame}) for all frames")
                print(f"  Total frames: {len(refined_base_poses)}")
    
    updated_poses_data['success'] = len(refined_moving_poses) > 0 and len(updated_poses_data.get('base_poses', {})) > 0
    if verbose:
        print(f"  Final: {len(refined_moving_poses)} poses")
        # Compare with original
        if len(moving_poses) > 0 and len(refined_moving_poses) > 0:
            orig_trans = np.array([p[:3, 3] for p in moving_poses.values()])
            new_trans = np.array([refined_moving_poses[t][:3, 3] for t in sorted(refined_moving_poses.keys()) if t in moving_poses])
            if len(new_trans) > 0:
                diff = np.linalg.norm(orig_trans[:len(new_trans)] - new_trans, axis=1).mean()
                print(f"  Average pose difference: {diff:.4f}m")
    return updated_poses_data
