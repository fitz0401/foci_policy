"""
FOCI Actor for trajectory prediction using FOCI model (two-stage GMM + waypoint decoder).
"""
import torch
import numpy as np
from pathlib import Path
from foci_policy.scripts.train_foci import load_config, load_point_encoder_config, load_dataloader_config, create_model
from foci_policy.utils.transform_utils import matrix_to_pose_9d, pose_9d_to_matrix, create_identity_pose_9d


class FOCIActor:
    def __init__(
        self, 
        checkpoint_path,
        config_dir='../config',
        device='cuda:0',
        voxel_size=0.004,
        num_points=2048,
        use_color=True,
        deterministic=True,
    ):
        """
        Args:
            checkpoint_path: Path to trained FOCI model checkpoint
            config_dir: Directory containing configuration files
            device: CUDA device ID (e.g., 'cuda:0')
            voxel_size: Voxel size for downsampling (default: 0.004, same as preprocessing)
            num_points: Fixed number of points after sampling/padding (default: 2048)
            use_color: Whether to use point cloud colors (future feature)
            deterministic: Whether to use deterministic goal sampling (True: argmax, False: stochastic)
        """
        torch.cuda.set_device(device)
        self.device = torch.device(device)
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.use_color = use_color
        self.deterministic = deterministic
        
        # Load checkpoint first to get the saved config
        print(f"Loading FOCI checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config_dir = Path(config_dir)
        dataloader_path = config_dir / 'dataloader.yaml'
        dataloader_config = load_config(dataloader_path)
        
        # Get model config from checkpoint
        model_config = checkpoint['config']['model']
        print(f"Using config from checkpoint: mode={model_config['mode']}")
        
        # Load point encoder config (architecture-specific, not mode-specific)
        pcd_encoder_path = config_dir / 'pcd_encoder' / 'masked_pointnet_encoder.yaml'
        point_encoder_cfg = load_point_encoder_config(pcd_encoder_path)
        
        # Create FOCI model with correct mode from checkpoint
        self.model = create_model(model_config, point_encoder_cfg, dataloader_config)
        self.model = self.model.to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        self.model.eval()
        self.mode = model_config['mode']
        self.prediction_length = dataloader_config.get('prediction_length', 5)
        self.use_language = model_config.get('use_language', False)
        
        print(f"FOCI Actor initialized in '{self.mode}' mode")
        print(f"Prediction length: {self.prediction_length}")
        print(f"Deterministic sampling: {self.deterministic}")
    
    def preprocess_pcd(self, pcd, reference_pose):
        """
        Preprocess point cloud following preprocess_raw_rlbench_demo.py:
        1. Voxel downsampling
        2. Fixed number of points (random sampling if more, padding if less)
        3. Canonicalization using reference pose
        Args:
            pcd: Open3D point cloud
            reference_pose: (4, 4) numpy array, reference pose for canonicalization
        Returns:
            points: (N, 3) tensor, canonicalized points with fixed size
            colors: (N, 3) tensor, point colors with fixed size
        """
        # Step 1: Voxel downsampling
        pcd_sample = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pts = np.asarray(pcd_sample.points)
        cols = np.asarray(pcd_sample.colors) if pcd_sample.has_colors() else np.ones_like(pts) * 0.5
        
        # Step 2: Fixed number of points — matches training (random choice with replacement)
        if len(pts) >= self.num_points:
            indices = np.random.choice(len(pts), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(pts), self.num_points, replace=True)
        points = pts[indices]
        colors = cols[indices]
        
        # To torch tensor
        points = torch.from_numpy(points).float()
        colors = torch.from_numpy(colors).float()
        
        # Step 3: Canonicalize - transform points to reference coordinate system
        R = reference_pose[:3, :3]
        t = reference_pose[:3, 3]
        points_canonical = (points - torch.from_numpy(t).float()) @ torch.from_numpy(R).float()
        
        return points_canonical, colors
    
    def act(
        self, 
        pa_pcd, 
        pb_pcd,
        pa_pose,
        pb_pose,
        gripper_pose,
        language_embedding=None,
    ):
        """
        Predict trajectory using FOCI model (two-stage: goal + actions).
        Args:
            pa_pcd: Open3D point cloud of base object (pa)
            pb_pcd: Open3D point cloud of moving object (pb)
            pa_pose: (4, 4) numpy array, pa object pose in world frame
            pb_pose: (4, 4) numpy array, pb object pose in world frame
            gripper_pose: (4, 4) numpy array, gripper pose in world frame
                - grasp mode: current gripper pose for computing relative pose to pb
                - manip mode: gripper pose when picking pb (for transforming pb trajectory to gripper trajectory)
            language_embedding: (1, 512) tensor, language embedding for task description
        Returns:
            trajectory_matrices: (T, 4, 4) numpy array
                - grasp mode: absolute gripper trajectory in world frame
                - manip mode: absolute gripper trajectory in world frame
        """
        # Determine reference pose based on mode
        if self.mode == 'grasp':
            # Grasp mode: canonicalize using pb pose (moving object)
            reference_pose = pb_pose
        else:  # manip mode
            # Manip mode: canonicalize using pa pose (base object)
            reference_pose = pa_pose
        
        # Preprocess point clouds with canonicalization
        pa_points, pa_colors = self.preprocess_pcd(pa_pcd, reference_pose)
        pb_points, pb_colors = self.preprocess_pcd(pb_pcd, reference_pose)
        
        # Move to device and add batch dimension
        pa_points = pa_points.to(self.device).unsqueeze(0)  # (1, N, 3)
        pb_points = pb_points.to(self.device).unsqueeze(0)  # (1, N, 3)
        
        # Compute relative pose for model input
        if self.mode == 'grasp':
            # Grasp mode: gripper relative to pb
            current_pose_matrix = np.linalg.inv(pb_pose) @ gripper_pose
        elif self.mode == 'manip':
            # Manip mode: pb relative to pa
            current_pose_matrix = np.linalg.inv(pa_pose) @ pb_pose
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        current_pose_matrix = torch.from_numpy(current_pose_matrix).float().to(self.device).unsqueeze(0)  # (1, 4, 4)
        current_pose_9d = matrix_to_pose_9d(current_pose_matrix)  # (1, 9)
        
        # Prepare input based on mode
        identity_pose = create_identity_pose_9d((1,), device=self.device)  # (1, 9)
        
        if self.mode == 'grasp':
            # Grasp mode: predict gripper trajectory relative to pb
            pa_current_pose = current_pose_9d  # Not used in grasp mode
            pb_current_pose = identity_pose
        else:  # manip mode
            # Manip mode: predict pb trajectory relative to pa
            pa_current_pose = identity_pose
            pb_current_pose = current_pose_9d
        
        # Run inference with FOCI model
        with torch.no_grad():
            pred_trajectory_9d = self.model(
                pa_points=pa_points,
                pb_points=pb_points,
                pa_pose=pa_current_pose,
                pb_pose=pb_current_pose,
                language_embedding=language_embedding,
                return_gmm_params=False,
                deterministic_goal=self.deterministic,
            )  # (1, T, 9)
        
        # Convert to transformation matrices
        pred_trajectory_relative = pose_9d_to_matrix(pred_trajectory_9d.squeeze(0))  # (T, 4, 4)
        trajectory_matrices = []
        
        if self.mode == 'grasp':
            # Grasp mode: transform gripper trajectory from pb frame to world frame
            pb_pose_tensor = torch.from_numpy(pb_pose).float().to(self.device)
            for i in range(self.prediction_length):
                abs_pose = pb_pose_tensor @ pred_trajectory_relative[i]
                trajectory_matrices.append(abs_pose)
            
        else:  # manip mode
            # Manip mode: transform pb trajectory to gripper trajectory in world frame
            gripper_pick_matrix = torch.from_numpy(gripper_pose).float().to(self.device)
            pa_pose_tensor = torch.from_numpy(pa_pose).float().to(self.device)
            pb_pose_tensor = torch.from_numpy(pb_pose).float().to(self.device)
            for i in range(self.prediction_length):
                # Step 1: pb absolute pose in world frame
                pb_abs_pose = pa_pose_tensor @ pred_trajectory_relative[i]
                # Step 2: pb transformation relative to initial pb_pose
                pb_delta = pb_abs_pose @ torch.inverse(pb_pose_tensor)
                # Step 3: Apply transformation to gripper
                gripper_abs_pose = pb_delta @ gripper_pick_matrix
                trajectory_matrices.append(gripper_abs_pose)
        
        trajectory_matrices = torch.stack(trajectory_matrices, dim=0)  # (T, 4, 4)
        trajectory_matrices = trajectory_matrices.cpu().numpy()
        return trajectory_matrices
    
    @torch.no_grad()
    def predict_trajectory(
        self,
        pa_pcd,
        pb_pcd,
        pa_pose,
        pb_pose,
        gripper_pose,
        language_embedding=None,
    ):
        """Alias for act() method for consistency with other actors"""
        return self.act(pa_pcd, pb_pcd, pa_pose, pb_pose, gripper_pose, language_embedding)
