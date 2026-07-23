#!/usr/bin/env python3
"""
Test Foundation Pose on real-world demo data.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
from pathlib import Path
from foci_real_world.pose_estimator.fp_real_world_utils import (
    extract_poses_from_demo,
    visualize_pose_trajectory
)
from foci_real_world.pose_estimator.mask_generator import MaskGenerator

# Lower logging verbosity
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description='Test Foundation Pose on real-world demo'
    )
    parser.add_argument('--demo_dir', type=str, 
                       default='../dataset/demo_001',
                       help='Path to demo directory')
    parser.add_argument('--object_name', type=str, default='banana',
                       help='Object name for segmentation')
    parser.add_argument('--mesh_path', type=str, default=None,
                       help='Path to object mesh file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize mask on first frame')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir / args.demo_dir
    if args.mesh_path is None:
        mesh_path = script_dir / '../assets' / f'{args.object_name}.obj'
    else:
        mesh_path = script_dir / args.mesh_path
    
    if not demo_dir.exists():
        print(f"Demo directory not found: {demo_dir}")
        return
    
    if not mesh_path.exists():
        print(f"Mesh file not found: {mesh_path}")
        return
    
    print(f"Demo: {demo_dir}")
    print(f"Object: {args.object_name}")
    print(f"Mesh: {mesh_path}")
    print("="*70)
    
    print("\nInitializing GroundedSAM...")
    mask_generator = MaskGenerator()
    
    # Extract poses from demo
    print("\nExtracting poses from demo...")
    poses_data = extract_poses_from_demo(
        demo_dir=str(demo_dir),
        object_name=args.object_name,
        mesh_path=str(mesh_path),
        mask_generator=mask_generator,
        verbose=args.verbose,
        visualize_mask=args.visualize
    )
    
    if poses_data['success']:
        print(f"\n✓ Successfully extracted {len(poses_data['poses'])} poses")
        
        # Print statistics
        import numpy as np
        poses = poses_data['poses']
        positions = np.array([p[:3, 3] for p in poses.values()])
        
        print(f"\nTrajectory statistics:")
        print(f"  Start position: [{positions[0][0]:.3f}, {positions[0][1]:.3f}, {positions[0][2]:.3f}]")
        print(f"  End position:   [{positions[-1][0]:.3f}, {positions[-1][1]:.3f}, {positions[-1][2]:.3f}]")
        print(f"  Displacement:   {np.linalg.norm(positions[-1] - positions[0]):.3f}m")
        print(f"  Position std:   [{positions.std(axis=0)[0]:.4f}, {positions.std(axis=0)[1]:.4f}, {positions.std(axis=0)[2]:.4f}]")
        
        # Visualize trajectory
        print("\nVisualizing trajectory (close window to exit)...")
        visualize_pose_trajectory(
            poses_data=poses_data,
            demo_dir=str(demo_dir),
            mesh_path=str(mesh_path)
        )
    else:
        print("\n✗ Pose extraction failed")


if __name__ == '__main__':
    main()
