import os
import torch
import pickle
import random
import numpy as np
import open3d as o3d
from pathlib import Path
import foci_policy.model.clip_revised as clip_revised
from foci_policy.model.clip_revised.clip import build_model, tokenize
from foci_policy.utils.analyze_fp_quality import select_best_demo
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List

_MESH_DIR = Path(__file__).resolve().parent.parent / "assets" / "RLBench_mesh"


def compute_relative_pose(pose_a, pose_b):
    """
    Compute relative pose in object B's frame: rel_pose = inv(pose_b) @ pose_a
    """
    return np.linalg.inv(pose_b) @ pose_a

def pad_trajectory(trajectory, target_length):
    """
    Pad trajectory to target length
    """
    if len(trajectory) >= target_length:
        return trajectory[:target_length]
    # Pad with the goal state
    goal_frame = trajectory[-1]
    padding = [goal_frame] * (target_length - len(trajectory))
    return trajectory + padding

def sample_trajectory(frames_data, start_time, goal_time, prediction_length=5):
    """
    Sample trajectory from current_time to goal_time with prediction length
    """
    if goal_time < start_time:
        return []
    # Generate all available indices from current to goal
    all_indices = list(range(start_time, goal_time + 1))
    time_span = len(all_indices)
    if time_span <= prediction_length:
        # If interval is shorter than prediction_length, use all frames and pad
        indices = all_indices
    else:
        lin_idx = np.linspace(0, time_span - 1, prediction_length, dtype=int)
        indices = [all_indices[i] for i in lin_idx]
    # Extract trajectory frames
    trajectory = [frames_data[i] if i < len(frames_data) else frames_data[-1] for i in indices]
    # Pad with goal state at the beginning if needed
    trajectory = pad_trajectory(trajectory, prediction_length)
    return trajectory

class FOCI_Dataset(InMemoryDataset):
    def __init__(self, root: str, mode: str = 'grasp', train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 prediction_length: int = 5,
                 num_demos: int = 10,
                 task_list: Optional[List[str]] = None,
                 sampling_ratio: float = 0.6,
                 max_demos_per_task: int = 50,
                 force_reload: bool = True,
                 optimal: bool = False):
        """
        FOCI Dataset for grasp and manipulation tasks
        
        Args:
            root: Root directory containing dataset
            mode: 'grasp' for gripper ↔ moving object, 'manip' for moving object ↔ base object
            train: Whether this is training set
            prediction_length: Length of trajectory sequences
            num_demos: Number of demos to load per task (candidate pool)
            task_list: List of task names to load. If None, loads all available tasks.
            sampling_ratio: Sampling ratio for generating multiple samples per demo
            max_demos_per_task: Maximum number of demos to consider per task
            force_reload: If True, delete existing processed files and reprocess from scratch
            optimal: If True, select the best-aligned demo (by ICP) from the pool instead of all n demos
        """
        self.mode = mode
        self.optimal = optimal
        self.prediction_length = prediction_length
        self.sampling_ratio = sampling_ratio
        self.max_demos_per_task = max_demos_per_task
        self.pcd_path = root
        self.task_list = []
        if task_list is None:
            root_path = Path(root)
            if root_path.exists():
                self.task_list = [d.name for d in root_path.iterdir() 
                                 if d.is_dir() and not d.name.startswith('.') 
                                 and d.name not in ['processed', 'raw', '__pycache__']]
        else:
            self.task_list = task_list
        print(f"Using task list: {self.task_list}")
        if not self.task_list:
            raise ValueError(f"No tasks found in root directory: {root}")
        self.demos = num_demos
        if force_reload:
            processed_dir = os.path.join(root, 'processed')
            for fname in [f'training_{mode}.pt', f'test_{mode}.pt']:
                fpath = os.path.join(processed_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"[force_reload] Deleted {fpath}")
        
        super(FOCI_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self) -> str:
        return 'data'

    @property
    def processed_file_names(self) -> List[str]:
        return [f'training_{self.mode}.pt', f'test_{self.mode}.pt']

    def download(self):
        return
    
    def process(self):
        data_list = []
        self.total_attempts = 0  # Track total sample generation attempts
        self.skipped_samples = 0  # Track samples skipped due to missing poses
        for task_name in self.task_list:
            num_demo = 0
            task_dir = os.path.join(self.pcd_path, task_name)
            if not os.path.exists(task_dir):
                continue

            available_pkls = sorted(f for f in os.listdir(task_dir) if f.endswith('.pkl'))
            if self.optimal:
                pool = available_pkls[:self.demos]
                if pool:
                    best = select_best_demo(Path(task_dir), task_name, _MESH_DIR,
                                           self.mode, pool)
                    available_pkls = [best]
            for fname in available_pkls:
                # collect only specified number of demos
                if num_demo >= self.demos:
                    break
                num_demo += 1
                
                datapoint = pickle.load(open(os.path.join(task_dir, fname), 'rb'))
                
                # Extract frame data
                frames_data = datapoint['frames_data']
                key_times = datapoint['key_times']
                
                # Generate trajectory samples based on mode
                if self.mode == 'grasp':
                    # Gripper ↔ Moving object interaction
                    self._process_grasp_mode(data_list, datapoint, frames_data, key_times, task_name)
                elif self.mode == 'manip':
                    # Moving object ↔ Base object interaction
                    self._process_manip_mode(data_list, datapoint, frames_data, key_times, task_name)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Log summary of processing
        total_num = len(data_list)
        if self.skipped_samples > 0:
            print(f"\n⚠️  Dataset Processing Summary ({self.mode} mode):")
            print(f"   Attempted: {self.total_attempts} samples")
            print(f"   Skipped: {self.skipped_samples} samples (missing poses)")
            print(f"   Valid: {total_num} samples ({100*total_num/self.total_attempts:.1f}%)\n")
        
        train_size = int(total_num * 0.8)
        random.seed(42)
        random.shuffle(data_list)
        train_data = data_list[:train_size]
        test_data = data_list[train_size:]     
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(test_data), self.processed_paths[1])
        # torch.save(self.collate(data_list[:int(total_num)]), self.processed_paths[0])
        # torch.save(self.collate(data_list[:int(total_num)]), self.processed_paths[1])
    
    def _process_grasp_mode(self, data_list, datapoint, frames_data, key_times, task_name):
        """
        Process grasp mode: gripper ↔ moving object
        Generate data pairs: current_state -> trajectory
        """
        # Key time points for grasp mode
        pick = key_times['pick']
        # Use computed grasp_interval from preprocessing
        grasp_interval = datapoint.get('grasp_interval', 10)  # Fallback to 10 if not available
        interval_start = max(0, pick - grasp_interval)
        # Generate candidate current times based on computed interaction length
        num_samples = max(1, int(self.sampling_ratio * (pick + 1 - 0)))
        num_samples = min(num_samples, self.max_demos_per_task)
        candidate_times = np.linspace(0, pick, num=num_samples, dtype=int).tolist()
        # Sample target trajectory
        target_traj = sample_trajectory(
            frames_data, interval_start, pick, self.prediction_length
        )
        # Generate multiple data pairs by sampling different current times
        if target_traj and len(target_traj) > 1:
            for current_t in candidate_times:
                current_frame = frames_data[current_t]
                self.total_attempts += 1
                # Convert to gripper@obj coordinate system
                traj_data = self._convert_to_gripper_obj_coords(
                    target_traj, current_frame, current_t, task_name, datapoint
                )
                if traj_data:
                    data_list.append(traj_data)
                else:
                    self.skipped_samples += 1
    
    def _process_manip_mode(self, data_list, datapoint, frames_data, key_times, task_name):
        """
        Process manipulation mode: moving object ↔ base object
        Generate data pairs: current_state -> trajectory
        """
        # Key time points for manipulation mode
        pick = key_times['pick']
        place = key_times['place']
        # Use computed manip_interval from preprocessing
        manip_interval = datapoint.get('manip_interval', 10)  # Fallback to 10 if not available
        interval_start = max(0, place - manip_interval)
        # Generate candidate current times based on computed interaction length
        num_samples = max(1, int(self.sampling_ratio * (place + 1 - pick)))
        num_samples = min(num_samples, self.max_demos_per_task)
        candidate_times = np.linspace(pick, place, num=num_samples, dtype=int).tolist()
        # Sample trajectory
        target_traj = sample_trajectory(
            frames_data, interval_start, place, self.prediction_length
        )
        # Generate multiple data pairs by sampling different current times
        if target_traj and len(target_traj) > 1:
            for current_t in candidate_times:
                current_frame = frames_data[current_t]
                self.total_attempts += 1
                traj_data = self._convert_to_obj_base_coords(
                    target_traj, current_frame, current_t, task_name, datapoint
                )
                if traj_data:
                    data_list.append(traj_data)
                else:
                    self.skipped_samples += 1
    
    def _convert_to_gripper_obj_coords(self, trajectory, current_frame, current_time, task_name, datapoint):
        """
        Convert trajectory to gripper@obj coordinate system
        Grasp mode: express everything in pb (moving object) coordinate system
        - pb point cloud is canonicalized (centered and aligned with initial pose)
        - gripper trajectory is expressed relative to pb
        """
        if trajectory is None or len(trajectory) < 2:
            return None
            
        current_gripper_pose = current_frame['gripper_pose_mat']
        current_obj_pose = current_frame['pb_pose_mat']
        if current_gripper_pose is None or current_obj_pose is None:
            return None
        
        # Extract point clouds from current frame
        pa_points = torch.from_numpy(current_frame['pa_points']).float()
        pa_colors = torch.from_numpy(current_frame['pa_colors']).float()
        pb_points = torch.from_numpy(current_frame['pb_points']).float()
        pb_colors = torch.from_numpy(current_frame['pb_colors']).float()
        
        # Canonicalize pb point cloud (moving object) and pa (context)
        R = current_obj_pose[:3, :3]
        t = current_obj_pose[:3, 3]
        pb_points_canonical = ((pb_points - t) @ R).float()
        pa_points_canonical = ((pa_points - t) @ R).float()

        # Compute relative poses for entire trajectory (gripper relative to pb)
        pred_rel_poses = []
        for frame in trajectory:
            gripper_pose = frame['gripper_pose_mat']
            obj_pose = frame['pb_pose_mat']
            if gripper_pose is None or obj_pose is None:
                return None  # Skip this sample if any frame has missing pose
            # gripper@obj coordinate system
            pred_rel_pose = compute_relative_pose(gripper_pose, obj_pose)
            pred_rel_poses.append(pred_rel_pose)
        
        # Current state: gripper relative pose to pb
        current_gripper_rel_pose = compute_relative_pose(current_gripper_pose, current_obj_pose)
                
        # Create data structure
        pred_rel_poses_tensor = torch.stack([torch.from_numpy(pose).float() for pose in pred_rel_poses])
        
        data = Data(
            # Point clouds (pa for context, pb canonicalized)
            pa_points=pa_points_canonical,  # Canonicalized pa points
            pa_colors=pa_colors,
            pb_points=pb_points_canonical,  # Canonicalized pb points
            pb_colors=pb_colors,
            
            # Current state pose (gripper relative to pb)
            current_pose=torch.from_numpy(current_gripper_rel_pose).float(),
            
            # Trajectory (target, gripper relative to pb)
            trajectory_poses=pred_rel_poses_tensor,
            trajectory_length=torch.tensor(len(pred_rel_poses)),
            
            # Metadata
            task_name=task_name,
            mode=self.mode,
            current_time=torch.tensor(current_time),
            
            # Language instructions
            language=datapoint.get('lan_pick', ''),
        )
        return data

    def _convert_to_obj_base_coords(self, trajectory, current_frame, current_time, task_name, datapoint):
        """
        Convert trajectory to obj@base coordinate system
        Manip mode: express everything in pa (base object) coordinate system
        - pa point cloud is canonicalized (centered and aligned with initial pose)
        - pb point cloud maintains relative relationship to pa
        - object (pb) trajectory is expressed relative to base (pa)
        """
        if trajectory is None or len(trajectory) < 2:
            return None
            
        # Use explicitly passed current frame data
        current_obj_pose = current_frame['pb_pose_mat']
        current_base_pose = current_frame['pa_pose_mat']
        if current_obj_pose is None or current_base_pose is None:
            return None
        
        # Extract point clouds from current frame
        pa_points = torch.from_numpy(current_frame['pa_points']).float()
        pa_colors = torch.from_numpy(current_frame['pa_colors']).float()
        pb_points = torch.from_numpy(current_frame['pb_points']).float()
        pb_colors = torch.from_numpy(current_frame['pb_colors']).float()
        
        # Canonicalize pa point cloud (base object) and pb (moving object)
        R = current_base_pose[:3, :3]
        t = current_base_pose[:3, 3]
        pb_points_canonical = ((pb_points - t) @ R).float()
        pa_points_canonical = ((pa_points - t) @ R).float()

        # Compute relative poses for entire trajectory (pb relative to pa)
        pred_rel_poses = []
        for frame in trajectory:
            obj_pose = frame['pb_pose_mat']
            base_pose = frame['pa_pose_mat']
            if obj_pose is None or base_pose is None:
                return None
            # obj@base coordinate system
            pred_rel_pose = compute_relative_pose(obj_pose, base_pose)
            pred_rel_poses.append(pred_rel_pose)
        
        # Current state: object relative pose to base
        current_obj_rel_pose = compute_relative_pose(current_obj_pose, current_base_pose)
        
        # Create data structure
        pred_rel_poses_tensor = torch.stack([torch.from_numpy(pose).float() for pose in pred_rel_poses])

        data = Data(
            # Point clouds
            pa_points=pa_points_canonical,  # Canonicalized pa (base) points
            pa_colors=pa_colors,
            pb_points=pb_points_canonical,  # Canonicalized pb (moving) points
            pb_colors=pb_colors,
            
            # Current state pose (pb relative to pa)
            current_pose=torch.from_numpy(current_obj_rel_pose).float(),
            
            # Trajectory (target) - pb trajectory relative to pa
            trajectory_poses=pred_rel_poses_tensor,
            trajectory_length=torch.tensor(len(pred_rel_poses)),
            
            # Metadata
            task_name=task_name,
            mode=self.mode,
            current_time=torch.tensor(current_time),
            
            # Language instructions
            language=datapoint.get('lan_place', ''),
        )
        return data


class AddLanguageEmbedding:
    """Add language embeddings to the data"""
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        clip_model_name = "ViT-B/32"
        model, _ = clip_revised.load(clip_model_name, device=self.device)
        self.clip_vit32 = build_model(model.state_dict()).to(self.device)
        self.clip_vit32.eval()
        del model

    def __call__(self, data):
        language = data['language']
        with torch.no_grad():
            tokens = tokenize([language]).to(self.device)
            text_feat, _ = self.clip_vit32.encode_text_with_embeddings(tokens)
        # Squeeze to remove batch dimension: (1, 512) -> (512,)
        data['language_embedding'] = text_feat.squeeze(0).cpu().float()
        return data

# TODO: augmentation
class TrajectoryAugmentation:
    """Augmentation for trajectory data"""
    def __init__(self, add_noise=True, noise_std=0.01, random_rotation=True):
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_rotation = random_rotation
    
    def __call__(self, data):
        if self.random_rotation:
            # Apply random rotation to point clouds and trajectory
            rotation = Rotation.random()
            rot_matrix = torch.from_numpy(rotation.as_matrix()).float()
            
            # Rotate point clouds
            data['pa_points'] = data['pa_points'] @ rot_matrix.T
            data['pb_points'] = data['pb_points'] @ rot_matrix.T
            
            # Rotate trajectory poses
            trajectory_poses = data['trajectory_poses']
            rot_4x4 = torch.eye(4)
            rot_4x4[:3, :3] = rot_matrix
            
            # Apply rotation to each pose in trajectory
            for i in range(len(trajectory_poses)):
                trajectory_poses[i] = rot_4x4 @ trajectory_poses[i] @ rot_4x4.T
        
        if self.add_noise:
            # Add noise to point clouds
            pa_noise = torch.randn_like(data['pa_points']) * self.noise_std
            pb_noise = torch.randn_like(data['pb_points']) * self.noise_std
            data['pa_points'] = data['pa_points'] + pa_noise
            data['pb_points'] = data['pb_points'] + pb_noise
            
            # Add noise to trajectory poses (translation part)
            traj_noise = torch.randn(len(data['trajectory_poses']), 3) * self.noise_std
            data['trajectory_poses'][:, :3, 3] += traj_noise
        
        return data


def check_dataloader(dataset, num_samples=3):
    """Check dataloader: print dataset size and visualize random samples"""
    print(f"\n{'='*60}")
    print(f"Dataset Check - Mode: {dataset.mode}")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    
    # Sample random indices
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in sample_indices:
        data = dataset[idx]
        
        print(f"\n{'-'*60}")
        print(f"Sample {idx}:")
        print(f"  Task: {data.task_name}")
        print(f"  Mode: {data.mode}")
        print(f"  Current time: {data.current_time.item()}")
        print(f"  PA points shape: {data.pa_points.shape}")
        print(f"  PB points shape: {data.pb_points.shape}")
        print(f"  Trajectory length: {data.trajectory_length.item()}")
        print(f"  Current pose:\n{data.current_pose}")
        print(f"  Language: {data.language}")
        
        # Visualize
        print(f"\nVisualizing sample {idx}... Close window to continue.")
        
        geometries = []
        
        # World coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        geometries.append(world_frame)
        
        # PA point cloud
        pa_pcd = o3d.geometry.PointCloud()
        pa_pcd.points = o3d.utility.Vector3dVector(data.pa_points.numpy())
        pa_pcd.colors = o3d.utility.Vector3dVector(data.pa_colors.numpy())
        pa_pcd.paint_uniform_color([0, 0, 1])  # Blue
        geometries.append(pa_pcd)
        
        # PB point cloud
        pb_pcd = o3d.geometry.PointCloud()
        pb_pcd.points = o3d.utility.Vector3dVector(data.pb_points.numpy())
        pb_pcd.colors = o3d.utility.Vector3dVector(data.pb_colors.numpy())
        pb_pcd.paint_uniform_color([0, 1, 0])  # Green
        geometries.append(pb_pcd)
        
        # Current pose (relative pose)
        current_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        current_frame.transform(data.current_pose.numpy())
        geometries.append(current_frame)
        
        # Trajectory poses, orange line, and orange arrows
        traj_positions = [data.trajectory_poses[i].numpy()[:3, 3] for i in range(data.trajectory_length.item())]
        for i in range(data.trajectory_length.item()):
            traj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            traj_frame.transform(data.trajectory_poses[i].numpy())
            geometries.append(traj_frame)
        # Add orange line connecting waypoints
        if len(traj_positions) > 1:
            lines = [[i, i+1] for i in range(len(traj_positions)-1)]
            colors = [[1.0, 0.6, 0.0] for _ in lines]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(traj_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
        # Add arrows to indicate trajectory direction (orange)
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
                arrow.paint_uniform_color([1.0, 0.6, 0.0])  # orange
                arrow.translate(start + 0.7 * (end - start))
                geometries.append(arrow)
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Sample {idx} - {data.task_name} ({data.mode})",
            width=800,
            height=600
        )
    
    print(f"\n{'='*60}")
    print("Check complete!")
    print(f"{'='*60}\n")
