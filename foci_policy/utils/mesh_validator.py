"""
Mesh validation and preparation tools for Foundation Pose.

Foundation Pose requires meshes to be centered at origin.
This tool helps validate and fix mesh files.
"""

import numpy as np
import open3d as o3d
import os
import argparse
import shutil
import json


def check_mesh_centered(mesh_path, tolerance=0.01):
    """
    Check if a mesh is centered at origin.
    
    Args:
        mesh_path: Path to mesh file
        tolerance: Maximum allowed distance from origin (meters)
    
    Returns:
        dict with 'centered', 'center', 'offset' keys
    """
    if not os.path.exists(mesh_path):
        return {'error': 'File not found'}
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    
    center = vertices.mean(axis=0)
    center_dist = np.linalg.norm(center)
    
    return {
        'centered': center_dist <= tolerance,
        'center': center,
        'offset': center_dist,
        'num_vertices': len(vertices),
        'bbox_min': vertices.min(axis=0),
        'bbox_max': vertices.max(axis=0),
    }


def recenter_mesh(mesh_path, output_path=None, method='centroid', backup=True):
    """
    Re-center a mesh to origin.
    
    Args:
        mesh_path: Input mesh file
        output_path: Output path (if None, overwrites input after backup)
        method: 'centroid', 'bbox_center', or 'bottom_center'
        backup: Create backup before overwriting
    
    Returns:
        Translation vector applied
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    
    # Compute translation
    if method == 'centroid':
        translation = -vertices.mean(axis=0)
    elif method == 'bbox_center':
        bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        translation = -bbox_center
    elif method == 'bottom_center':
        bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        translation = np.array([-bbox_center[0], -bbox_center[1], -vertices.min(axis=0)[2]])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply translation
    mesh.translate(translation)
    
    # Save
    if output_path is None:
        output_path = mesh_path
        if backup:
            backup_path = mesh_path + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy2(mesh_path, backup_path)
    
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    return translation


def validate_task_meshes(task_name, mesh_base_dir='../assets/RLBench_mesh', 
                        tolerance=0.01, auto_fix=False):
    """
    Validate all meshes for a task.
    
    Args:
        task_name: Task name
        mesh_base_dir: Base directory for meshes
        tolerance: Maximum allowed offset from origin
        auto_fix: Automatically recenter meshes if needed
    
    Returns:
        dict with validation results
    """
    mesh_dir = os.path.join(mesh_base_dir, task_name)
    
    if not os.path.exists(mesh_dir):
        return {'error': f'Directory not found: {mesh_dir}'}
    
    # Find all obj files
    obj_files = [f for f in os.listdir(mesh_dir) 
                 if f.endswith('.obj') and not f.endswith('.backup.obj')]
    
    results = {}
    all_centered = True
    
    for obj_file in obj_files:
        mesh_path = os.path.join(mesh_dir, obj_file)
        check_result = check_mesh_centered(mesh_path, tolerance)
        
        results[obj_file] = check_result
        
        if not check_result.get('centered', False):
            all_centered = False
            
            if auto_fix:
                print(f"Auto-fixing {obj_file}...")
                translation = recenter_mesh(mesh_path, method='centroid', backup=True)
                check_result['fixed'] = True
                check_result['translation'] = translation.tolist()
    
    return {
        'task_name': task_name,
        'mesh_dir': mesh_dir,
        'all_centered': all_centered,
        'meshes': results
    }


def print_validation_report(results):
    """Print a human-readable validation report."""
    print(f"\n{'='*60}")
    print(f"Mesh Validation Report: {results['task_name']}")
    print(f"{'='*60}")
    print(f"Location: {results['mesh_dir']}")
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    for mesh_name, mesh_info in results['meshes'].items():
        print(f"\n{mesh_name}:")
        
        if 'error' in mesh_info:
            print(f"  ERROR: {mesh_info['error']}")
            continue
        
        print(f"  Vertices: {mesh_info['num_vertices']}")
        print(f"  Center: [{mesh_info['center'][0]:.4f}, {mesh_info['center'][1]:.4f}, {mesh_info['center'][2]:.4f}]")
        print(f"  Offset from origin: {mesh_info['offset']:.4f}m")
        
        if mesh_info['centered']:
            print(f"  ✓ Mesh is properly centered")
        else:
            print(f"  ⚠️  Mesh is NOT centered - will cause pose errors!")
            if mesh_info.get('fixed'):
                print(f"  ✓ Auto-fixed (backup created)")
    
    print(f"\n{'='*60}")
    if results['all_centered']:
        print("✓ All meshes are properly centered - Ready for Foundation Pose")
    else:
        print("⚠️  Some meshes need recentering - Run with --fix to correct")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and fix meshes for Foundation Pose')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name')
    parser.add_argument('--mesh_dir', type=str, default='../assets/RLBench_mesh',
                        help='Base directory containing mesh files')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Maximum allowed offset from origin (meters)')
    parser.add_argument('--fix', action='store_true',
                        help='Automatically recenter meshes if needed (creates backups)')
    parser.add_argument('--single_file', type=str, default=None,
                        help='Check/fix a single mesh file instead of entire task')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Check single file
        result = check_mesh_centered(args.single_file, args.tolerance)
        print(f"\nMesh: {args.single_file}")
        print(f"Centered: {result['centered']}")
        print(f"Center: {result['center']}")
        print(f"Offset: {result['offset']:.4f}m")
        
        if not result['centered'] and args.fix:
            print("\nRecentering mesh...")
            translation = recenter_mesh(args.single_file, method='centroid', backup=True)
            print(f"Translation applied: {translation}")
            print("Done! (Backup created)")
    else:
        # Check entire task
        results = validate_task_meshes(
            task_name=args.task_name,
            mesh_base_dir=args.mesh_dir,
            tolerance=args.tolerance,
            auto_fix=args.fix
        )
        
        print_validation_report(results)