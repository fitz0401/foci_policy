#!/usr/bin/env python3
"""
Merge multiple mesh parts from RLBench TTM file into a single mesh.
Preserves the relative transformations between parts.
"""
import os
import argparse
from pathlib import Path
import numpy as np
import rlbench

# Headless mode
os.environ['COPPELIASIM_HEADLESS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pyrep import PyRep
from pyrep.objects.shape import Shape


def merge_mesh_parts(ttm_file, part_names, output_path, verbose=False):
    """Merge multiple parts into a single mesh, preserving transformations."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Merging parts: {', '.join(part_names)}")
        print(f"From: {ttm_file}")
        print(f"{'='*70}\n")
    
    pr = PyRep()
    pr.launch(headless=True)
    pr.start()
    
    try:
        # Load TTM
        pr.import_model(str(ttm_file))
        
        merged_vertices = []
        merged_faces = []
        vertex_offset = 0
        
        # Process each part
        for part_name in part_names:
            try:
                shape = Shape(part_name)
                
                # Get mesh data (in local coordinates)
                vertices, indices, normals = shape.get_mesh_data()
                
                if vertices is None or len(vertices) == 0:
                    if verbose:
                        print(f"⚠ {part_name}: empty mesh, skipping")
                    continue
                
                vertices = np.array(vertices).reshape(-1, 3)
                
                # Get transformation matrix (local to world)
                transform = shape.get_matrix()
                
                # Apply transformation to vertices
                vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                vertices_world = (transform @ vertices_homogeneous.T).T[:, :3]
                
                # Add to merged data
                merged_vertices.append(vertices_world)
                
                if indices is not None and len(indices) > 0:
                    indices = np.array(indices).reshape(-1, 3)
                    # Adjust indices by current vertex offset
                    merged_faces.append(indices + vertex_offset)
                    vertex_offset += len(vertices_world)
                
                if verbose:
                    print(f"✓ {part_name}: {len(vertices)} vertices, {len(indices)} faces")
            
            except Exception as e:
                if verbose:
                    print(f"⚠ {part_name}: failed - {e}")
                continue
        
        if not merged_vertices:
            print("❌ No valid meshes to merge")
            return False
        
        # Combine all data
        all_vertices = np.vstack(merged_vertices)
        all_faces = np.vstack(merged_faces) if merged_faces else None
        
        # Write OBJ file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"# Merged mesh from RLBench TTM\n")
            f.write(f"# Parts: {', '.join(part_names)}\n\n")
            
            # Write vertices
            for v in all_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces
            if all_faces is not None:
                for face in all_faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"\n✓ Merged mesh saved: {output_path}")
        print(f"  Total vertices: {len(all_vertices)}")
        print(f"  Total faces: {len(all_faces) if all_faces is not None else 0}")
        print(f"  Parts merged: {len(merged_vertices)}")
        
        return True
        
    finally:
        pr.stop()
        pr.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple mesh parts from RLBench TTM file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge parts from a configured task
  python merge_mesh_parts.py --task close_jar \\
      --parts jar0 jar_lid0 \\
      --output meshes/close_jar/jar.obj --verbose
        """
    )
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., close_jar)')
    parser.add_argument('--parts', type=str, nargs='+', required=True, help='Part names to merge')
    parser.add_argument('--output', type=str, required=True, help='Output OBJ file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--rlbench-root', type=str, help='RLBench root directory')
    
    args = parser.parse_args()
    
    # Locate TTM file
    if args.rlbench_root:
        rlbench_root = Path(args.rlbench_root)
        task_ttms_dir = rlbench_root / 'rlbench' / 'task_ttms'
    else:
        task_ttms_dir = Path(rlbench.__file__).resolve().parent / 'task_ttms'
    
    ttm_file = task_ttms_dir / f'{args.task}.ttm'
    
    if not ttm_file.exists():
        print(f"❌ TTM file not found: {ttm_file}")
        return
    
    # Merge meshes
    success = merge_mesh_parts(
        ttm_file=ttm_file,
        part_names=args.parts,
        output_path=args.output,
        verbose=args.verbose
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
