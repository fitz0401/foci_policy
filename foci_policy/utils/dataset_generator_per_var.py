import os
import shutil
import sys

# CoppeliaSim/RLBench needs a real X server context in headless mode.
# Auto-relaunch under Xvfb when DISPLAY is missing.
if not os.environ.get('DISPLAY'):
    if os.environ.get('FOCI_XVFB_STARTED') != '1':
        xvfb_run = shutil.which('xvfb-run')
        if xvfb_run:
            os.environ['FOCI_XVFB_STARTED'] = '1'
            os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
            os.execvp(
                xvfb_run,
                [
                    xvfb_run,
                    '-a',
                    '-s',
                    '-screen 0 1280x720x24 +extension GLX +render -noreset',
                    sys.executable,
                    *sys.argv,
                ],
            )
        raise RuntimeError(
            'No DISPLAY detected and xvfb-run not found. '
            'Install Xvfb (`sudo apt install xvfb`) and re-run with xvfb-run, or provide a DISPLAY.'
        )

from multiprocessing import Process, Manager
from pyrep.const import RenderMode
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import pickle
import warnings
from pathlib import Path
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np
from absl import app
from absl import flags
import yaml

FLAGS = flags.FLAGS
REPO_ROOT = Path(__file__).resolve().parents[2]
DATALOADER_CONFIG = REPO_ROOT / 'foci_policy' / 'config' / 'dataloader.yaml'
RAW_DATA_ROOT = REPO_ROOT / 'data'
SAVE_ROOT = RAW_DATA_ROOT / 'rlbench_data'

flags.DEFINE_string('save_path',
                    str(SAVE_ROOT),
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', 1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_integer('variation', 0,
                     'the variation to collect per task. from 0 to some number.')
flags.DEFINE_boolean('disp', False,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_boolean('enhance_gt', True,
                     'Whether to enrich demos with GT handles/poses.')


def _enrich_demo_with_gt(demo, task_name: str, variation: int,
                         task_env=None) -> bool:
    """Stamp GT handle IDs and GT poses into every frame's misc while the sim is live.
    Stores two things in each frame's obs.misc:

    1. misc['gt_objects'] : dict  — ALL objects configured for this task that
       are actually present in the scene, keyed by CoppeliaSim object name:
           {name: {'handles': [h1, h2, ...],  # subtree handles for mask lookup
                   'pose':    [x,y,z,qx,qy,qz,qw],  # live pose at last frame
                   'lds_byte_offset': int or None}}  # byte offset into task_low_dim_state

    2. Backward-compatible per-role keys (for existing code):
           grasp_obj_name / grasp_obj_handle / grasp_obj_pose
           target_obj_name / target_obj_handle / target_obj_pose
       These use the variation-specific names from task_obj_config.yaml.

    Returns True if at least one object was enriched.
    """
    try:
        from pyrep.objects.shape import Shape
        from pyrep.const import ObjectType
        from foci_policy.config.config_utils import (
            get_all_task_obj_names, get_task_obj_config
        )
    except ImportError as e:
        warnings.warn(f"GT enrichment skipped (import error): {e}")
        return False

    # --- Collect ALL unique object names configured for this task ---
    all_names = get_all_task_obj_names(task_name)
    if not all_names:
        return False

    last_lds = (demo[-1].task_low_dim_state
                if demo[-1].task_low_dim_state is not None else None)
    n_groups = len(last_lds) // 7 if last_lds is not None else 0

    def _subtree_handles(obj):
        hs = []
        try:
            for child in obj.get_objects_in_tree(
                    object_type=ObjectType.SHAPE, exclude_base=False):
                hs.append(int(child.get_handle()))
        except Exception:
            pass
        hs = hs or [int(obj.get_handle())]
        return hs if len(hs) > 1 else hs[0]

    # Build handle → LDS byte-offset map from task._initial_objs_in_scene.
    # Must accumulate actual byte offsets: joints add 8 entries (7+1),
    # force sensors add 13 (7+6), plain shapes/dummies add 7.
    h2g: dict = {}
    if task_env is not None:
        try:
            from pyrep.const import ObjectType as _ObjType
            _offset = 0
            for (_o, _otype) in task_env._task._initial_objs_in_scene:
                h2g[int(_o.get_handle())] = _offset
                _offset += 7
                if _otype == _ObjType.JOINT:
                    _offset += 1
                elif _otype == _ObjType.FORCE_SENSOR:
                    _offset += 6
            print(f"  GT enrich: h2g built ({len(h2g)} handles, LDS total={_offset})")
        except Exception as _e:
            print(f"  GT enrich: WARNING h2g build failed: {_e}")

    def _pos_match(live_xyz, used: set):
        """Position-based LDS lookup — fallback when h2g is unavailable.
        Returns byte offset (g*7), consistent with h2g semantics.
        """
        best_g, best_d = None, float('inf')
        for g in range(n_groups):
            if g * 7 in used:
                continue
            d = float(np.linalg.norm(live_xyz - last_lds[g * 7: g * 7 + 3]))
            if d < best_d:
                best_d, best_g = d, g
        return (best_g * 7, best_d) if best_d < 0.05 else (None, best_d)

    def _find_lds(shape_obj):
        """Find LDS group for shape_obj: handle map → child handles → pos match."""
        h = int(shape_obj.get_handle())
        # 1. Direct handle
        if h in h2g:
            return h2g[h], 'handle'
        # 2. SHAPE children (handles task-base objects excluded from h2g)
        if h2g:
            try:
                for c in shape_obj.get_objects_in_tree(
                        ObjectType.SHAPE, exclude_base=False):
                    ch = int(c.get_handle())
                    if ch != h and ch in h2g:
                        return h2g[ch], 'child-handle'
            except Exception:
                pass
        # 3. Position matching (original fallback — works for static objects)
        idx, d = _pos_match(np.array(shape_obj.get_pose()[:3]), used_lds)
        return idx, f'pos({d:.3f}m)'

    # frame-0 LDS for the canonical pose snapshot (= initial object position)
    lds_0 = demo[0].task_low_dim_state
    lds_0 = np.asarray(lds_0) if lds_0 is not None else None

    # Build gt_objects for all names that exist in the scene
    gt_objects: dict = {}
    used_lds: set = set()
    for name in sorted(all_names):
        try:
            shape = Shape(name)
        except Exception:
            # Not a SHAPE type (e.g. Dummy grouping object) — try generic Object
            try:
                from pyrep.backend import sim as _psim
                from pyrep.objects.object import Object as _PObj
                shape = _PObj(_psim.simGetObjectHandle(name))
            except Exception:
                continue  # truly not present in this scene

        handles = _subtree_handles(shape)
        lds_idx, how = _find_lds(shape)

        if lds_idx is not None:
            used_lds.add(lds_idx)
            # Snapshot = frame-0 LDS pose (= initial position, NOT end-of-demo)
            # lds_idx is a byte offset, so read lds_0[lds_idx:lds_idx+7]
            if (lds_0 is not None and lds_0.ndim == 1
                    and len(lds_0) >= lds_idx + 7):
                live_pose7 = lds_0[lds_idx: lds_idx + 7].tolist()
            else:
                live_pose7 = list(shape.get_pose())
        else:
            live_pose7 = list(shape.get_pose())  # best we can do

        gt_objects[name] = {
            'handles':         handles,
            'pose':            live_pose7,
            'lds_byte_offset': lds_idx,   # byte offset (not group index)
        }
        print(f"  GT enrich: {name} handles={handles}  lds_byte_offset={lds_idx}  [{how}]")

    if not gt_objects:
        return False

    # Backward-compat role keys from task_obj_config
    compat_roles: list = []
    try:
        cfg = get_task_obj_config(task_name)
        grasp_name = cfg.get('grasp')
        target_name = cfg.get('target')
        if task_name == 'insert_onto_square_peg':
            if task_env is not None and hasattr(task_env, '_task') and hasattr(task_env._task, '_chosen_pillar_name'):
                chosen = task_env._task._chosen_pillar_name
                if chosen:
                    target_name = chosen
        for role, cname in [('grasp', grasp_name), ('target', target_name)]:
            # cname may be a list (e.g. reach_and_drag/turn_tap use
            # [obj1, obj2] in task_obj_config.yaml) — use the first entry,
            # matching the "first object" pose convention used elsewhere
            # (e.g. get_grasp_pose_matrix / get_target_pose_matrix).
            cname0 = cname[0] if isinstance(cname, list) else cname
            if cname0 and cname0 in gt_objects:
                compat_roles.append((role, cname0, gt_objects[cname0]))
    except KeyError:
        pass

    # Stamp each frame
    n_stamped = 0
    for obs in demo:
        try:
            if obs.misc is None:
                obs.misc = {}
            lds = obs.task_low_dim_state
            if lds is not None:
                lds = np.asarray(lds)   # ensure numpy array for slicing / .tolist()

            # Per-frame gt_objects: handles static; pose updated from lds if available
            per_frame_gt: dict = {}
            for name, info in gt_objects.items():
                idx = info['lds_byte_offset']
                if (idx is not None and lds is not None
                        and lds.ndim == 1 and len(lds) >= idx + 7):
                    pose7 = lds[idx: idx + 7].tolist()
                else:
                    pose7 = info['pose']   # static snapshot
                per_frame_gt[name] = {
                    'handles':         info['handles'],
                    'pose':            pose7,
                    'lds_byte_offset': idx,
                }
            obs.misc['gt_objects'] = per_frame_gt

            # Backward-compat keys
            for role, name, info in compat_roles:
                obs.misc[f'{role}_obj_name']   = name
                obs.misc[f'{role}_obj_handle'] = info['handles']
                idx = info['lds_byte_offset']
                if (idx is not None and lds is not None
                        and lds.ndim == 1 and len(lds) >= idx + 7):
                    obs.misc[f'{role}_obj_pose'] = lds[idx: idx + 7].tolist()
            if task_name == 'insert_onto_square_peg' and task_env is not None:
                if hasattr(task_env, '_task') and hasattr(task_env._task, '_chosen_pillar_name'):
                    obs.misc['chosen_pillar_name'] = task_env._task._chosen_pillar_name
            n_stamped += 1
        except Exception as e:
            import traceback
            print(f'  [warn] frame stamp failed: {e}')
            traceback.print_exc()

    print(f'  GT enrich: stamped {n_stamped}/{len(demo)} frames')
    return n_stamped > 0


def check_and_make(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_tasks_from_dataloader(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    tasks = config.get('task_list', [])
    if tasks is None:
        return []
    return [t for t in tasks if t]


def save_demo(demo, example_path):
    from concurrent.futures import ThreadPoolExecutor

    # Overhead camera is not used downstream — skip it entirely.
    cam_dirs = {
        'left_shoulder_rgb':   LEFT_SHOULDER_RGB_FOLDER,
        'left_shoulder_depth': LEFT_SHOULDER_DEPTH_FOLDER,
        'left_shoulder_mask':  LEFT_SHOULDER_MASK_FOLDER,
        'right_shoulder_rgb':   RIGHT_SHOULDER_RGB_FOLDER,
        'right_shoulder_depth': RIGHT_SHOULDER_DEPTH_FOLDER,
        'right_shoulder_mask':  RIGHT_SHOULDER_MASK_FOLDER,
        'wrist_rgb':   WRIST_RGB_FOLDER,
        'wrist_depth': WRIST_DEPTH_FOLDER,
        'wrist_mask':  WRIST_MASK_FOLDER,
        'front_rgb':   FRONT_RGB_FOLDER,
        'front_depth': FRONT_DEPTH_FOLDER,
        'front_mask':  FRONT_MASK_FOLDER,
    }
    paths = {k: os.path.join(example_path, v) for k, v in cam_dirs.items()}
    for p in paths.values():
        check_and_make(p)

    def _save_frame(args):
        i, obs = args
        Image.fromarray(obs.left_shoulder_rgb).save(
            os.path.join(paths['left_shoulder_rgb'], IMAGE_FORMAT % i))
        utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(paths['left_shoulder_depth'], IMAGE_FORMAT % i))
        Image.fromarray((obs.left_shoulder_mask * 255).astype(np.uint8)).save(
            os.path.join(paths['left_shoulder_mask'], IMAGE_FORMAT % i))

        Image.fromarray(obs.right_shoulder_rgb).save(
            os.path.join(paths['right_shoulder_rgb'], IMAGE_FORMAT % i))
        utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(paths['right_shoulder_depth'], IMAGE_FORMAT % i))
        Image.fromarray((obs.right_shoulder_mask * 255).astype(np.uint8)).save(
            os.path.join(paths['right_shoulder_mask'], IMAGE_FORMAT % i))

        Image.fromarray(obs.wrist_rgb).save(
            os.path.join(paths['wrist_rgb'], IMAGE_FORMAT % i))
        utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(paths['wrist_depth'], IMAGE_FORMAT % i))
        Image.fromarray((obs.wrist_mask * 255).astype(np.uint8)).save(
            os.path.join(paths['wrist_mask'], IMAGE_FORMAT % i))

        Image.fromarray(obs.front_rgb).save(
            os.path.join(paths['front_rgb'], IMAGE_FORMAT % i))
        utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE).save(
            os.path.join(paths['front_depth'], IMAGE_FORMAT % i))
        Image.fromarray((obs.front_mask * 255).astype(np.uint8)).save(
            os.path.join(paths['front_mask'], IMAGE_FORMAT % i))

        # Null out after saving so the pickle is lightweight.
        obs.left_shoulder_rgb = obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = obs.right_shoulder_mask = None
        obs.overhead_rgb = obs.overhead_depth = None
        obs.overhead_point_cloud = obs.overhead_mask = None
        obs.wrist_rgb = obs.wrist_depth = None
        obs.wrist_point_cloud = obs.wrist_mask = None
        obs.front_rgb = obs.front_depth = None
        obs.front_point_cloud = obs.front_mask = None

    # Parallel per-frame saves (I/O bound; PIL releases the GIL during encode+write)
    n_workers = min(len(demo), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for exc in pool.map(_save_frame, enumerate(demo)):
            pass  # map() re-raises any exception from a worker

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    # Overhead camera is unused in all downstream pipelines — disable to save
    # ~20 % render time per sim step and skip its image saves.
    obs_config.overhead_camera.set_all(False)

    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode= MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config= obs_config,
        headless= not FLAGS.disp)
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)

            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process0', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)

        if FLAGS.variation >= task_env.variation_count():
            print('the provided variation number {} exceeds available variations {}'.format(FLAGS.variation, task_env.variation_count()))
            return
        task_env.set_variation(FLAGS.variation)

        descriptions,obs = task_env.reset()
        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % FLAGS.variation)

        check_and_make(variation_path)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), FLAGS.variation, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                # Stamp GT handle IDs + poses into misc while sim is still live.
                if FLAGS.enhance_gt:
                    try:
                        _enrich_demo_with_gt(demo, task_env.get_name(), FLAGS.variation,
                                             task_env=task_env)
                    except Exception as e:
                        import traceback
                        print(f'  [warn] GT enrichment failed (demo still saved): {e}')
                        traceback.print_exc()
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):
    if FLAGS.disp and not os.environ.get('DISPLAY'):
        raise RuntimeError(
            'No DISPLAY detected. Use --nodisp for headless mode, or run under Xvfb.'
        )

    # Handle common absl bool-flag misuse: "--enhance_gt False"
    if FLAGS.enhance_gt and any(str(a).lower() == 'false' for a in argv[1:]):
        warnings.warn(
            "Detected positional 'False'. Use --enhance_gt=false or --noenhance_gt. "
            "Auto-disabling GT enrichment for this run."
        )
        FLAGS.enhance_gt = False
    available_task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                            if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in available_task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks
    else:
        if not DATALOADER_CONFIG.exists():
            raise FileNotFoundError('Dataloader config not found: %s' % DATALOADER_CONFIG)
        task_files = load_tasks_from_dataloader(DATALOADER_CONFIG)
        if not task_files:
            raise ValueError('No tasks found in dataloader config: %s' % DATALOADER_CONFIG)
        for t in task_files:
            if t not in available_task_files:
                raise ValueError('Task %s from dataloader config not recognised!.' % t)

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]
    [t.start() for t in processes]
    [t.join() for t in processes]

    failed = [
        (idx, p.exitcode)
        for idx, p in enumerate(processes)
        if p.exitcode not in (0, None)
    ]
    if failed:
        details = ', '.join([f'worker {idx}: exitcode={code}' for idx, code in failed])
        raise RuntimeError(f'Data collection workers crashed: {details}')

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)
