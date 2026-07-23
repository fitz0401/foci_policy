"""
Extract per-object masks and GT poses using task_obj_config.yaml.

Two operating modes:

  LIVE (sim is running):
    ext = DemoObjectExtractor.from_config('close_jar')
    pose_dict  = ext.get_poses_live()            # uses Shape.get_pose()
    mask_dict  = ext.get_masks_live(front_mask)  # uses Shape.get_handle()

  DEMO (stored demo, enriched with gt_objects in frame.misc):
    ext = DemoObjectExtractor.from_config('close_jar')
    pose = ext.get_grasp_pose_matrix(frame)
    mask = ext.get_grasp_mask(frame)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, List, Dict
from scipy.spatial.transform import Rotation

from .utils.rlbench_objects import task_object_dict


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def rgb_mask_to_handle_ids(mask_rgb: np.ndarray) -> np.ndarray:
    """RLBench RGB seg mask (H×W×3 uint8) → integer handle IDs (H×W int32).
    Encoding: handle = R + G*256 + B*65536
    """
    m = mask_rgb.astype(np.int32)
    return m[:, :, 0] + m[:, :, 1] * 256 + m[:, :, 2] * 65536


def pose7d_to_matrix(pose7d: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4×4 homogeneous matrix."""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = pose7d[:3]
    mat[:3, :3] = Rotation.from_quat(pose7d[3:]).as_matrix()
    return mat


def _sim_object(name: str):
    """Return a PyRep Object for any CoppeliaSim type (Shape, Dummy, Joint…).

    Some objects (e.g. compound groups like dollar_stack) are not SHAPE type,
    so Shape(name) would throw.  This helper falls back to the generic Object
    wrapper so callers work regardless of the underlying type.
    Requires a running PyRep/CoppeliaSim context.
    """
    from pyrep.objects.shape import Shape
    try:
        return Shape(name)
    except Exception:
        from pyrep.backend import sim as _psim
        from pyrep.objects.object import Object
        return Object(_psim.simGetObjectHandle(name))


def _sim_shape_handles(obj) -> list:
    """Return a list of all SHAPE handles in obj's CoppeliaSim subtree.

    Uses exclude_base=False so the object itself is included when it is a
    SHAPE, and child SHAPE visual sub-objects are always collected.
    Segmentation masks encode these child handles, not the parent's handle.
    """
    from pyrep.const import ObjectType
    hs = []
    try:
        for child in obj.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False):
            hs.append(int(child.get_handle()))
    except Exception:
        pass
    return hs or [int(obj.get_handle())]


def _resolve_name(name_spec, variation: int) -> Optional[str]:
    """Resolve a name that may be a plain string, a {variation: name} dict, or '__NONE__'."""
    if isinstance(name_spec, str):
        return None if name_spec == '__NONE__' else name_spec
    if isinstance(name_spec, dict):
        if variation in name_spec:
            return name_spec[variation]
        return list(name_spec.values())[0]
    return str(name_spec)


def _binary_mask(handle_map: np.ndarray,
                 handle_id: Union[int, List[int]]) -> np.ndarray:
    if isinstance(handle_id, (list, tuple)):
        out = np.zeros(handle_map.shape, dtype=np.uint8)
        for h in handle_id:
            out |= (handle_map == h).astype(np.uint8)
        return out
    return (handle_map == handle_id).astype(np.uint8)


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class DemoObjectExtractor:
    """Unified mask + GT-pose extractor that works with or without a running sim.

    Always created via from_config() — reads stable grasp/target names from
    task_obj_config.yaml rather than resolving variation-dependent names.

    DEMO mode relies on enriched demos that store gt_objects in frame.misc.
    """

    def __init__(self, task_name: str, variation: int = 0):
        if task_name not in task_object_dict:
            raise KeyError(
                f"'{task_name}' not in task_object_dict. "
                f"Available: {list(task_object_dict.keys())}"
            )
        self.task_name = task_name
        self.variation = variation

        task_cfg = task_object_dict[task_name]
        self.grasp_obj_name: Optional[str] = _resolve_name(
            task_cfg['grasp_object_name'], variation)
        self.target_obj_name: Optional[str] = _resolve_name(
            task_cfg.get('target_object_name', '__NONE__'), variation)
        self._auto_grasp = False
        self._auto_target = False

    @classmethod
    def from_config(cls, task_name: str, stage: int = 1) -> 'DemoObjectExtractor':
        """Create an extractor using grasp/target names from task_obj_config.yaml.

        Unlike the normal constructor (which resolves names from task_object_dict
        using a variation index), this factory uses the explicit, variation-
        independent names written in task_obj_config.yaml.  Use this in
        preprocessing scripts and online tests where you want a stable, single
        grasp+target pair regardless of variation.

        stage: 1 (default) reads 'grasp'/'target'; 2 reads 'grasp_2'/'target_2'
               (for multi-stage tasks like stack_blocks / stack_cups).

        grasp_obj_name / target_obj_name may be a str or List[str]:
          - Masks  → all listed objects are merged (union).
          - Poses  → only the FIRST object in the list is used.
        """
        from foci_policy.config.config_utils import get_task_obj_config
        cfg = get_task_obj_config(task_name)
        instance = object.__new__(cls)
        instance.task_name   = task_name
        instance.variation   = 0
        if stage == 2:
            raw_grasp  = cfg.get('grasp_2')  or None
            raw_target = cfg.get('target_2') or None
        else:
            raw_grasp  = cfg.get('grasp')   or None
            raw_target = cfg.get('target')  or None
        # __AUTO__ only applies to plain strings (never to lists)
        instance._auto_grasp  = (raw_grasp  == '__AUTO__')
        instance._auto_target = (raw_target == '__AUTO__')
        instance.grasp_obj_name  = None if instance._auto_grasp  else raw_grasp
        instance.target_obj_name = None if instance._auto_target else raw_target
        return instance

    # ------------------------------------------------------------------
    # LIVE MODE  (sim must be running)
    # ------------------------------------------------------------------

    def get_masks_live(self, mask_rgb: np.ndarray, task=None) -> Dict[str, np.ndarray]:
        """Return {'grasp': H×W uint8, 'target': H×W uint8} using live Shape handles.
        Requires a running CoppeliaSim/PyRep environment.

        Accepts both (H,W) integer mask (live sim obs.front_mask) and
        (H,W,3) RGB mask (stored demo frame.front_mask).

        Collects handles for the named shape AND all shape children in its subtree,
        because compound objects (e.g. 'crackers') have visual sub-shapes whose
        handles differ from the parent handle encoded in the scene graph.
        """
        result: Dict[str, np.ndarray] = {}
        if mask_rgb.ndim == 3:
            handle_map = rgb_mask_to_handle_ids(mask_rgb)
        else:
            handle_map = mask_rgb.astype(np.int32)
        self.resolve_auto_names(task=task)
        for role, name in [('grasp', self.grasp_obj_name),
                           ('target', self.target_obj_name)]:
            if name is None:
                continue
            try:
                if isinstance(name, list):
                    handles: list = []
                    for n in name:
                        handles.extend(_sim_shape_handles(_sim_object(n)))
                else:
                    handles = _sim_shape_handles(_sim_object(name))
                result[role] = _binary_mask(handle_map, handles)
            except Exception as e:
                raise RuntimeError(f"Cannot get handle for '{name}': {e}") from e
        return result

    def get_poses_live(self, task=None) -> Dict[str, np.ndarray]:
        """Return {'grasp': 4×4, 'target': 4×4} using live Shape.get_pose().
        Requires a running CoppeliaSim/PyRep environment.
        """
        result: Dict[str, np.ndarray] = {}
        self.resolve_auto_names(task=task)
        for role, name in [('grasp', self.grasp_obj_name),
                           ('target', self.target_obj_name)]:
            if name is None:
                continue
            try:
                pose7 = np.array(_sim_object(name).get_pose())
                result[role] = pose7d_to_matrix(pose7)
            except Exception as e:
                raise RuntimeError(f"Cannot get pose for '{name}': {e}") from e
        return result

    def get_pose_live(self, obj_name: str) -> np.ndarray:
        """4×4 pose for a single named object. Requires running sim."""
        return pose7d_to_matrix(np.array(_sim_object(obj_name).get_pose()))

    # ------------------------------------------------------------------
    # DEMO MODE  (stored demo with gt_objects in frame.misc)
    # ------------------------------------------------------------------

    def get_mask(self, frame, obj_name: str, camera_name: str = 'front') -> np.ndarray:
        """Binary H×W mask from a stored demo frame.

        Requires enriched demos with gt_objects stored in frame.misc.
        """
        misc = getattr(frame, 'misc', None) or {}
        handle_id = None

        # gt_objects dict (new format — all objects, subtree handles)
        gt_objs = misc.get('gt_objects') or {}
        if obj_name in gt_objs:
            handle_id = gt_objs[obj_name].get('handles')

        # Legacy per-role misc keys (grasp_obj_handle / target_obj_handle)
        if handle_id is None:
            for role_key in ('grasp', 'target'):
                if misc.get(f'{role_key}_obj_name') == obj_name:
                    handle_id = misc.get(f'{role_key}_obj_handle')
                    break

        if handle_id is None:
            raise RuntimeError(
                f"No handle ID for '{obj_name}' in frame.misc for task '{self.task_name}'. "
                f"Regenerate demos with GT enrichment (dataset_generator_per_var.py)."
            )
        mask_rgb = getattr(frame, f'{camera_name}_mask', None)
        if mask_rgb is None:
            raise ValueError(
                f"'{camera_name}_mask' is None in this demo (camera not recorded). "
                f"Use one of: front, left_shoulder, right_shoulder, wrist."
            )
        handle_map = rgb_mask_to_handle_ids(mask_rgb) if mask_rgb.ndim == 3 else mask_rgb.astype(np.int32)
        return _binary_mask(handle_map, handle_id)

    def get_grasp_mask(self, frame, camera_name: str = 'front') -> np.ndarray:
        self.resolve_auto_names(frame=frame)
        obj_name = self.grasp_obj_name
        if obj_name is None:
            raise ValueError(f"No grasp object defined for '{self.task_name}'.")
        return self._get_mask_for_name(frame, obj_name, camera_name)

    def get_target_mask(self, frame, camera_name: str = 'front') -> np.ndarray:
        self.resolve_auto_names(frame=frame)
        obj_name = self.target_obj_name
        if obj_name is None:
            raise ValueError(f"No target object defined for '{self.task_name}'.")
        return self._get_mask_for_name(frame, obj_name, camera_name)

    def _get_mask_for_name(self, frame, obj_name, camera_name: str) -> np.ndarray:
        """Return mask for obj_name, merging masks when obj_name is a list."""
        if isinstance(obj_name, list):
            masks = [self.get_mask(frame, n, camera_name) for n in obj_name]
            out = masks[0].copy()
            for m in masks[1:]:
                out = np.maximum(out, m)
            return out
        return self.get_mask(frame, obj_name, camera_name)

    def get_pose_7d(self, frame, obj_name: str) -> np.ndarray:
        """[x,y,z,qx,qy,qz,qw] from a stored demo frame.

        Requires enriched demos with gt_objects stored in frame.misc.
        """
        misc = getattr(frame, 'misc', None) or {}

        # gt_objects dict — prefer lds_idx for per-frame accuracy,
        # fall back to stored static pose for objects not tracked in lds.
        gt_objs = misc.get('gt_objects') or {}
        if obj_name in gt_objs:
            entry = gt_objs[obj_name]
            byte_off = entry.get('lds_byte_offset')
            grp_idx  = entry.get('lds_idx')
            lds = frame.task_low_dim_state
            if byte_off is not None and lds is not None and len(lds) >= byte_off + 7:
                return lds[byte_off: byte_off + 7].astype(np.float64)
            if grp_idx is not None and lds is not None and len(lds) >= (grp_idx + 1) * 7:
                return lds[grp_idx * 7: grp_idx * 7 + 7].astype(np.float64)
            stored = entry.get('pose')
            if stored is not None:
                return np.array(stored, dtype=np.float64)

        # Legacy misc keys (MyScene / old enriched demos)
        misc_key = self._misc_key_for(obj_name)
        if misc_key and misc.get(misc_key) is not None:
            return np.array(misc[misc_key], dtype=np.float64)

        raise RuntimeError(
            f"No pose for '{obj_name}' in frame.misc for task '{self.task_name}'. "
            f"Regenerate demos with GT enrichment (dataset_generator_per_var.py)."
        )

    def get_pose_matrix(self, frame, obj_name: str) -> np.ndarray:
        """4×4 pose matrix from stored demo."""
        return pose7d_to_matrix(self.get_pose_7d(frame, obj_name))

    def get_grasp_pose_matrix(self, frame) -> np.ndarray:
        self.resolve_auto_names(frame=frame)
        obj_name = self.grasp_obj_name
        if obj_name is None:
            raise ValueError(f"No grasp object defined for '{self.task_name}'.")
        name = obj_name[0] if isinstance(obj_name, list) else obj_name
        return self.get_pose_matrix(frame, name)

    def get_target_pose_matrix(self, frame) -> np.ndarray:
        self.resolve_auto_names(frame=frame)
        obj_name = self.target_obj_name
        if obj_name is None:
            raise ValueError(f"No target object defined for '{self.task_name}'.")
        name = obj_name[0] if isinstance(obj_name, list) else obj_name
        return self.get_pose_matrix(frame, name)

    def get_pose_sequence(self, demo) -> Dict[str, Dict[int, np.ndarray]]:
        """Extract 4×4 pose matrices for every frame in a stored demo.

        Returns:
            {'grasp': {t: 4×4 ndarray}, 'target': {t: 4×4 ndarray}}
        Frames where extraction fails are silently skipped.
        """
        result: Dict[str, Dict[int, np.ndarray]] = {'grasp': {}, 'target': {}}
        for t, frame in enumerate(demo):
            self.resolve_auto_names(frame=frame)
            for role, name in [('grasp', self.grasp_obj_name),
                               ('target', self.target_obj_name)]:
                if name is None:
                    continue
                # Use first element of list for pose (pose is a single object)
                actual_name = name[0] if isinstance(name, list) else name
                try:
                    result[role][t] = self.get_pose_matrix(frame, actual_name)
                except Exception:
                    pass
        return result

    def extract(self, frame, camera_name: str = 'front') -> Dict[str, dict]:
        """Convenience: extract both grasp and target masks + poses at once.

        Returns:
            {'grasp': {'name', 'mask', 'pose_7d', 'pose_matrix'},
             'target': {'name', 'mask', 'pose_7d', 'pose_matrix'}}
        """
        result = {}
        self.resolve_auto_names(frame=frame)
        for role, name in [('grasp', self.grasp_obj_name),
                           ('target', self.target_obj_name)]:
            if name is None:
                continue
            entry: dict = {'name': name}
            try:
                entry['mask'] = self._get_mask_for_name(frame, name, camera_name)
            except Exception as e:
                entry['mask'] = None
                entry['mask_error'] = str(e)
            try:
                pose_name = name[0] if isinstance(name, list) else name
                entry['pose_7d'] = self.get_pose_7d(frame, pose_name)
                entry['pose_matrix'] = pose7d_to_matrix(entry['pose_7d'])
            except Exception as e:
                entry['pose_7d'] = None
                entry['pose_matrix'] = None
                entry['pose_error'] = str(e)
            result[role] = entry
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> dict:
        return {
            'task_name':       self.task_name,
            'variation':       self.variation,
            'grasp_obj_name':  self.grasp_obj_name,
            'target_obj_name': self.target_obj_name,
        }

    def __repr__(self) -> str:
        return (f"DemoObjectExtractor('{self.task_name}', variation={self.variation}, "
                f"grasp='{self.grasp_obj_name}', target='{self.target_obj_name}')")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _misc_key_for(self, obj_name: str) -> Optional[str]:
        """Return the frame.misc key that stores this object's pose (MyScene convention)."""
        grasp = self.grasp_obj_name
        if isinstance(grasp, list):
            if obj_name in grasp:
                return 'grasp_obj_pose'
        elif obj_name == grasp:
            return 'grasp_obj_pose'
        target = self.target_obj_name
        if isinstance(target, list):
            if obj_name in target:
                return 'target_obj_pose'
        elif obj_name == target:
            return 'target_obj_pose'
        return None

    def resolve_auto_names(self, frame=None, task=None) -> Dict[str, Optional[str]]:
        """Resolve __AUTO__ grasp/target names using demo misc or live task.
        Returns current {'grasp': name|None, 'target': name|None}.
        """
        if self._auto_grasp and self.grasp_obj_name is None:
            self._resolve_auto_name(role='grasp', frame=frame, task=task)
        if self._auto_target and self.target_obj_name is None:
            self._resolve_auto_name(role='target', frame=frame, task=task)
        return {'grasp': self.grasp_obj_name, 'target': self.target_obj_name}

    def _resolve_auto_name(self, role: str, frame=None, task=None) -> Optional[str]:
        """Resolve auto-configured object names from demo misc or live task."""
        if role == 'grasp' and not self._auto_grasp:
            return self.grasp_obj_name
        if role == 'target' and not self._auto_target:
            return self.target_obj_name

        name = None
        if frame is not None:
            misc = getattr(frame, 'misc', None) or {}
            name = misc.get(f'{role}_obj_name')
            if name is None:
                name = misc.get(f'stage0_{role}_obj_name')
            if name is None and role == 'target':
                name = misc.get('chosen_pillar_name')

        if name is None and task is not None and role == 'target':
            task_obj = task._task if hasattr(task, '_task') else task
            if hasattr(task_obj, '_chosen_pillar_name'):
                name = task_obj._chosen_pillar_name

        if name:
            if role == 'grasp':
                self.grasp_obj_name = name
            else:
                self.target_obj_name = name
        return name
