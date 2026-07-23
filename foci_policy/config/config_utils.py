from __future__ import annotations
import os
import yaml
import rlbench.tasks as tasks
from typing import Dict, List, Optional, Set, Union


# ---------------------------------------------------------------------------
# task_obj_config.yaml loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'task_obj_config.yaml')
_config_cache: dict | None = None
_DATALOADER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'dataloader.yaml')
_dataloader_config_cache: dict | None = None


def _load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f) or {}
    return _config_cache


def get_dataloader_config() -> dict:
    """Return the shared dataloader configuration."""
    global _dataloader_config_cache
    if _dataloader_config_cache is None:
        with open(_DATALOADER_CONFIG_PATH, 'r', encoding='utf-8') as f:
            _dataloader_config_cache = yaml.safe_load(f) or {}
    return _dataloader_config_cache


def get_interaction_interval_pen_factor(
    task_name: Optional[str] = None,
    default: float = 0.5,
) -> float:
    """Return the task-specific penalty factor, or the dataloader default."""
    pen_factor = get_dataloader_config().get('pen_factor', default)
    if task_name is not None:
        task_config = _load_config().get(task_name, {})
        pen_factor = task_config.get('pen_factor', pen_factor)
    return float(pen_factor)


# ---------------------------------------------------------------------------
# task_obj_config API
# ---------------------------------------------------------------------------

def get_task_obj_config(task_name: str) -> Dict[str, Union[str, List[str], float, None]]:
    """Return task object config dict for the task.

    Keys always present: 'grasp', 'target'.
    Optional keys: 'grasp_2', 'target_2', 'lan_grasp', 'lan_manip',
                   'mesh_names', 'camera', 'pen_factor'.

    grasp / target values are str or list[str].
    Raises KeyError if task_name is not in task_obj_config.yaml.
    """
    cfg = _load_config()
    if task_name not in cfg:
        raise KeyError(
            f"'{task_name}' not in task_obj_config.yaml. "
            f"Available: {list(cfg.keys())}"
        )
    entry = cfg[task_name]
    result: Dict[str, Union[str, List[str], float, None]] = {
        'grasp':  entry.get('grasp')  or None,
        'target': entry.get('target') or None,
    }
    for key in (
        'grasp_2',
        'target_2',
        'lan_grasp',
        'lan_manip',
        'mesh_names',
        'camera',
        'pen_factor',
    ):
        if key in entry:
            result[key] = entry[key] if key == 'pen_factor' else entry[key] or None
    return result


def get_task_language(task_name: str, mode: str = 'grasp') -> str:
    """Return the language instruction for grasp or manip phase."""
    cfg = _load_config()
    if task_name not in cfg:
        return f'{mode} the object'
    return cfg[task_name].get(f'lan_{mode}') or f'{mode} the object'


def get_task_camera(task_name: str, default: str = 'front') -> str:
    """Return the configured camera for a task, or the default fallback."""
    cfg = _load_config()
    if task_name not in cfg:
        return default
    camera = cfg[task_name].get('camera', default)
    if isinstance(camera, (list, tuple)):
        camera = camera[0] if camera else default
    return camera or default


def get_mesh_names(task_name: str) -> Optional[List[str]]:
    """Return [moving_mesh_name, base_mesh_name] for Foundation Pose, or None."""
    cfg = _load_config()
    if task_name not in cfg:
        return None
    names = cfg[task_name].get('mesh_names')
    if not names:
        return None
    return names if isinstance(names, list) else [str(names), str(names)]


def get_all_task_obj_names(task_name: str) -> Set[str]:
    """Return all unique object names configured for a task (all variations).

    Reads task_object_dict from rlbench_objects.py and flattens every
    grasp_object_name / target_object_name entry into a set, skipping
    '__NONE__' and empty strings.
    """
    from foci_policy.rlbench_env.utils.rlbench_objects import task_object_dict
    if task_name not in task_object_dict:
        return set()
    task_cfg = task_object_dict[task_name]
    names: Set[str] = set()
    for field in ('grasp_object_name', 'target_object_name'):
        spec = task_cfg.get(field, '__NONE__')
        if isinstance(spec, str):
            if spec and spec != '__NONE__':
                names.add(spec)
        elif isinstance(spec, dict):
            for v in spec.values():
                if v and v != '__NONE__':
                    names.add(str(v))
    return names


# ---------------------------------------------------------------------------
# RLBench task class registry
# ---------------------------------------------------------------------------

TASK_CLASSES = {
    'close_jar': tasks.CloseJar,
    'meat_off_grill': tasks.MeatOffGrill,
    'place_wine_at_rack_location': tasks.PlaceWineAtRackLocation,
    'put_money_in_safe': tasks.PutMoneyInSafe,
    'turn_tap': tasks.TurnTap,
    'reach_and_drag': tasks.ReachAndDrag,
    'light_bulb_in': tasks.LightBulbIn,
    'push_buttons': tasks.PushButtons,
    'slide_block_to_color_target': tasks.SlideBlockToColorTarget,
    'place_cups': tasks.PlaceCups,
    'sweep_to_dustpan_of_size': tasks.SweepToDustpanOfSize,
    'open_drawer': tasks.OpenDrawer,
    'place_shape_in_shape_sorter': tasks.PlaceShapeInShapeSorter,
    'insert_onto_square_peg': tasks.InsertOntoSquarePeg,
    'put_groceries_in_cupboard': tasks.PutGroceriesInCupboard,
    'put_item_in_drawer': tasks.PutItemInDrawer,
    'stack_blocks': tasks.StackBlocks,
    'stack_cups': tasks.StackCups,
    'close_box': tasks.CloseBox,
}


def get_task_class(task_name: str):
    return TASK_CLASSES[task_name]
