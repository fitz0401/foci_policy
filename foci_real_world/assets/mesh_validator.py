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
    """ Check if a mesh is centered at origin. """
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
    """ Re-center a mesh to origin. """
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



def validate_meshes_in_dir(obj_names, tolerance=0.01, auto_fix=False):
    """ Validate (and optionally fix) OBJ meshes in the current directory. """
    mesh_dir = os.path.dirname(os.path.abspath(__file__))
    if obj_names == ['all']:
        obj_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj') and not f.endswith('.backup.obj')]
    else:
        obj_files = [f for f in obj_names if f.endswith('.obj') and os.path.exists(os.path.join(mesh_dir, f))]
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
        'mesh_dir': mesh_dir,
        'all_centered': all_centered,
        'meshes': results
    }


def print_validation_report(results):
    """Print a human-readable validation report."""
    print(f"\n{'='*60}")
    print(f"Mesh Validation Report")
    print(f"{'='*60}")
    print(f"Location: {results['mesh_dir']}")
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
    parser = argparse.ArgumentParser(description='Validate and fix OBJ meshes in current directory')
    parser.add_argument('--obj_name', type=str, nargs='+', default=['all'],
                        help='OBJ file name(s) to check/fix, or "all" for all .obj files in this directory')
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Maximum allowed offset from origin (meters)')
    parser.add_argument('--fix', action='store_true',
                        help='Automatically recenter meshes if needed (creates backups)')
    args = parser.parse_args()

    results = validate_meshes_in_dir(
        obj_names=args.obj_name,
        tolerance=args.tolerance,
        auto_fix=args.fix
    )
    print_validation_report(results)