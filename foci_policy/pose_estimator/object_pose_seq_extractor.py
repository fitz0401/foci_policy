import numpy as np
from foci_policy.utils.transform_utils import Rotation, Transform, to_o3d_pcd


def get_object_poses_gt(obs, task_name: str) -> dict:
    """Extract grasp/target object GT poses from a stored demo frame.

    Uses task_obj_config.yaml to identify the grasp/target objects.
    Returns dict with keys: 'grasp'/'moving'/'object_1' (grasped object) and
    'target'/'base'/'object_0' (placement target).  Values are Transform objects.
    """
    from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
    ext = DemoObjectExtractor.from_config(task_name)
    poses: dict = {}
    if ext.grasp_obj_name:
        try:
            mat = ext.get_grasp_pose_matrix(obs)
            t = Transform(rotation=Rotation.from_matrix(mat[:3, :3]),
                          translation=mat[:3, 3])
            poses['grasp'] = poses['moving'] = poses['object_1'] = t
        except Exception:
            pass
    if ext.target_obj_name:
        try:
            mat = ext.get_target_pose_matrix(obs)
            t = Transform(rotation=Rotation.from_matrix(mat[:3, :3]),
                          translation=mat[:3, 3])
            poses['target'] = poses['base'] = poses['object_0'] = t
        except Exception:
            pass
    return poses


def extract_poses_for_demo_gt(demo, task_name: str, keyframes=None, variation: int = 0,
                               postprocess: bool = True, verbose: bool = False) -> dict:
    """Extract GT pose sequences for grasp/target objects across all demo frames.

    Uses task_obj_config.yaml to identify objects.
    Returns: {'base_poses': {t: 4×4}, 'moving_poses': {t: 4×4}, 'success': bool, ...}
    """
    from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
    from foci_policy.pose_estimator.foundation_pose_utils import postprocess_object_trajectory

    ext = DemoObjectExtractor.from_config(task_name)
    ext.variation = variation
    seq = ext.get_pose_sequence(demo)   # {'grasp': {t: 4×4}, 'target': {t: 4×4}}

    base_poses   = seq.get('target', {})   # target = base = pa
    moving_poses = seq.get('grasp',  {})   # grasp  = moving = pb

    poses_data = {
        'base_poses':       base_poses,
        'moving_poses':     moving_poses,
        'success':          len(base_poses) > 0 and len(moving_poses) > 0,
        'num_frames':       len(demo),
        'moving_reg_score': None,
        'base_reg_score':   None,
        'postprocessed':    False,
    }

    if postprocess and keyframes is not None:
        poses_data = postprocess_object_trajectory(
            poses_data=poses_data,
            demo=demo,
            keyframes=keyframes,
            fix_base=False,
            verbose=verbose,
        )
        poses_data['postprocessed'] = True

    return poses_data


def get_object_poses_live(task_name: str, task=None, stage: int = 1) -> dict:
    """Extract grasp/target object poses from a running CoppeliaSim environment.

    Uses task_obj_config.yaml to identify objects.
    Returns dict with keys: 'object_0' (base/target), 'object_1' (moving/grasp),
    and aliases 'grasp', 'target', 'moving', 'base'.  Values are Transform objects.

    stage: 1 (default) reads grasp/target; 2 reads grasp_2/target_2
           (multi-stage tasks like stack_blocks / stack_cups).
    """
    from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor
    ext = DemoObjectExtractor.from_config(task_name, stage=stage)
    live_poses = ext.get_poses_live(task=task)   # {'grasp': 4×4, 'target': 4×4}
    result: dict = {}
    if 'grasp' in live_poses:
        mat = live_poses['grasp']
        t = Transform(rotation=Rotation.from_matrix(mat[:3, :3]),
                      translation=mat[:3, 3])
        result['grasp'] = result['moving'] = result['object_1'] = t
    if 'target' in live_poses:
        mat = live_poses['target']
        t = Transform(rotation=Rotation.from_matrix(mat[:3, :3]),
                      translation=mat[:3, 3])
        result['target'] = result['base'] = result['object_0'] = t
    return result


def extract_object_pcds_from_obs(obs, task_name: str, variation: int = 0,
                                  voxel_size: float = 0.004, num_points: int = 2048,
                                  cameras=None, task=None) -> tuple:
    """Extract base/moving object point clouds from a live sim observation.

    Uses DemoObjectExtractor + Shape.get_handle() — no config_utils dependency.

    Returns: (pa_points, pa_colors, pb_points, pb_colors)
        pa = base = target object
        pb = moving = grasp object
    Requires a running CoppeliaSim/PyRep environment.
    """
    from pyrep.objects.shape import Shape
    from pyrep.const import ObjectType
    from foci_policy.rlbench_env.object_extractor import DemoObjectExtractor

    if cameras is None:
        cameras = ['front', 'left_shoulder', 'right_shoulder', 'wrist']

    ext = DemoObjectExtractor.from_config(task_name)
    if task is not None:
        ext.resolve_auto_names(task=task)
    if ext.grasp_obj_name is None:
        raise ValueError(f"No grasp object resolved for task '{task_name}'.")

    def _subtree_handles_one(name):
        try:
            obj = Shape(name)
        except Exception:
            from pyrep.backend import sim as _psim
            from pyrep.objects.object import Object as _PObj
            obj = _PObj(_psim.simGetObjectHandle(name))
        hs = []
        try:
            for child in obj.get_objects_in_tree(
                    object_type=ObjectType.SHAPE, exclude_base=False):
                hs.append(int(child.get_handle()))
        except Exception:
            pass
        return set(hs) if hs else {int(obj.get_handle())}

    def _subtree_handles(name_or_list):
        names = name_or_list if isinstance(name_or_list, list) else [name_or_list]
        result = set()
        for n in names:
            result |= _subtree_handles_one(n)
        return result

    grasp_handles  = _subtree_handles(ext.grasp_obj_name)
    target_handles = (_subtree_handles(ext.target_obj_name)
                      if ext.target_obj_name else set())

    all_pts, all_cols, grasp_flags, target_flags = [], [], [], []
    for cam in cameras:
        pcd_hw3 = getattr(obs, f'{cam}_point_cloud')    # (H, W, 3)
        rgb_hw3 = getattr(obs, f'{cam}_rgb')            # (H, W, 3)
        mask_hw = getattr(obs, f'{cam}_mask')           # (H, W) integer handle IDs
        pts  = pcd_hw3.reshape(-1, 3)
        cols = rgb_hw3.reshape(-1, 3)
        ids  = mask_hw.reshape(-1).astype(np.int32)
        valid = ~np.all(pts == 0, axis=1)
        all_pts.append(pts[valid])
        all_cols.append(cols[valid] / 255.0)
        grasp_flags.append(np.isin(ids[valid], list(grasp_handles)))
        if target_handles:
            target_flags.append(np.isin(ids[valid], list(target_handles)))

    all_pts  = np.concatenate(all_pts,  axis=0)
    all_cols = np.concatenate(all_cols, axis=0)
    pb_mask  = np.concatenate(grasp_flags,  axis=0)
    pb_pts_raw  = all_pts[pb_mask]
    pb_cols_raw = all_cols[pb_mask]

    if target_handles and target_flags:
        pa_mask = np.concatenate(target_flags, axis=0)
        pa_pts_raw  = all_pts[pa_mask]
        pa_cols_raw = all_cols[pa_mask]
    else:
        pa_pts_raw  = np.zeros((0, 3), dtype=np.float32)
        pa_cols_raw = np.zeros((0, 3), dtype=np.float32)

    def _sample(pts, cols):
        if len(pts) == 0:
            return (np.zeros((num_points, 3), dtype=np.float32),
                    np.zeros((num_points, 3), dtype=np.float32))
        pcd_o3d = to_o3d_pcd(pts, cols)
        pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
        n = len(pcd_o3d.points)
        idx = np.random.choice(n, num_points, replace=(n < num_points))
        return np.asarray(pcd_o3d.points)[idx], np.asarray(pcd_o3d.colors)[idx]

    pa_pts, pa_cols = _sample(pa_pts_raw,  pa_cols_raw)
    pb_pts, pb_cols = _sample(pb_pts_raw,  pb_cols_raw)
    return pa_pts, pa_cols, pb_pts, pb_cols
