#!/usr/bin/env python3
"""
Extract and export mesh data from RLBench .ttm files using PyRep.
This script loads the TTM file, extracts object information, and can export meshes to OBJ format.
"""
import os
import argparse
from pathlib import Path
import numpy as np
import rlbench

# Headless mode - MUST be before any Qt/PyRep imports
os.environ['COPPELIASIM_HEADLESS'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pyrep import PyRep
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape


def export_mesh_to_obj(shape, output_path):
    """ Export a Shape object's mesh to OBJ file. """
    try:
        # Get mesh data (vertices and indices)
        vertices, indices, normals = shape.get_mesh_data()
        
        if vertices is None or len(vertices) == 0:
            return False
        
        # Reshape vertices to Nx3
        vertices = np.array(vertices).reshape(-1, 3)
        
        # Write OBJ file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"# Exported from RLBench TTM file\n")
            f.write(f"# Object: {shape.get_name()}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (indices are 0-based, OBJ uses 1-based)
            if indices is not None and len(indices) > 0:
                indices = np.array(indices).reshape(-1, 3)
                for face in indices:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        return True
    
    except Exception as e:
        print(f"  Error exporting mesh: {e}")
        return False


def get_mesh_info(obj):
    """Extract mesh information from an object."""
    info = {
        'name': obj.get_name(),
        'type': obj.get_type(),
        'handle': obj.get_handle(),
    }
    
    # Try to get mesh-related information
    if isinstance(obj, Shape):
        try:
            # Check if it's a pure shape or mesh
            info['is_mesh'] = obj.is_mesh()
            info['is_compound'] = obj.is_compound_shape()
        except:
            pass
        
        try:
            # Get mesh path if available
            mesh_path = obj.get_mesh_path()
            if mesh_path:
                info['mesh_path'] = mesh_path
        except:
            pass
    
    return info


def analyze_ttm_file(ttm_file, target_objects=None, verbose=False, export_dir=None, task_name=None):
    """ Analyze a TTM file and extract object/mesh information. """
    ttm_path = Path(ttm_file)
    
    if not ttm_path.exists():
        print(f"❌ TTM file not found: {ttm_file}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {ttm_path.name}")
    print(f"{'='*70}\n")
    
    # Launch PyRep in headless mode
    pr = PyRep()
    pr.launch(headless=True)
    pr.start()
    
    try:
        # Load the TTM file as a model
        model_handle = pr.import_model(str(ttm_path))
        print(f"✓ Scene loaded successfully\n")
        
        # Get all objects
        from pyrep.const import ObjectType
        all_objects = pr.get_objects_in_tree(object_type=ObjectType.SHAPE)
        print(f"Total shape objects in scene: {len(all_objects)}\n")
        
        # Analyze objects
        results = {
            'ttm_file': str(ttm_path),
            'total_objects': len(all_objects),
            'target_objects': {},
            'all_shapes': [],
            'exported_count': 0,
        }
        
        # Determine which objects to export
        objects_to_export = []
        
        if export_dir:
            if target_objects:
                # Export specified objects
                for target_name in target_objects:
                    try:
                        obj = Shape(target_name)
                        objects_to_export.append(obj)
                    except:
                        pass
            else:
                # Export all non-floor, non-boundary objects
                for obj in all_objects:
                    if isinstance(obj, Shape):
                        name = obj.get_name().lower()
                        # Skip floor and boundary objects
                        if 'floor' not in name and 'boundary' not in name and 'wall' not in name:
                            objects_to_export.append(obj)
        
        # Export meshes
        if export_dir and objects_to_export:
            export_path = Path(export_dir)
            if task_name:
                task_export_dir = export_path / task_name
            else:
                task_export_dir = export_path / ttm_path.stem
            print(f"\n💾 Exporting meshes to: {task_export_dir}")
            print(f"{'='*70}\n")
            task_export_dir.mkdir(parents=True, exist_ok=True)
            for obj in objects_to_export:
                obj_name = obj.get_name()
                obj_file = task_export_dir / f"{obj_name}.obj"
                print(f"Exporting: {obj_name}...")
                success = export_mesh_to_obj(obj, obj_file)
                if success:
                    print(f"  ✓ Saved to: {obj_file.relative_to(export_path)}")
                    results['exported_count'] += 1
                else:
                    print(f"  ⚠ Failed to export (may be empty mesh)")
                print()
        
        # If target objects specified, search for them
        if target_objects:
            print(f"🔍 Searching for target objects: {', '.join(target_objects)}\n")
            
            for target_name in target_objects:
                try:
                    obj = Shape(target_name)
                    info = get_mesh_info(obj)
                    results['target_objects'][target_name] = info
                    
                    print(f"✓ Found: {target_name}")
                    print(f"  Type: {info.get('type', 'Unknown')}")
                    print(f"  Handle: {info.get('handle', 'Unknown')}")
                    if 'mesh_path' in info:
                        print(f"  Mesh path: {info['mesh_path']}")
                    else:
                        print(f"  Mesh path: Not available (may be embedded in TTM)")
                    
                    if 'is_mesh' in info:
                        print(f"  Is mesh: {info['is_mesh']}")
                    if 'is_compound' in info:
                        print(f"  Is compound: {info['is_compound']}")
                    
                    print()
                    
                except Exception as e:
                    print(f"⚠ Could not find object '{target_name}': {e}\n")
        
        # Show all shapes if verbose
        if verbose:
            print(f"\n{'='*70}")
            print("All Shape objects in scene:")
            print(f"{'='*70}\n")
            
            for obj in all_objects:
                try:
                    if isinstance(obj, Shape):
                        info = get_mesh_info(obj)
                        results['all_shapes'].append(info)
                        
                        name = info['name']
                        print(f"• {name}")
                        print(f"    Type: {info.get('type', 'Unknown')}")
                        
                        if 'mesh_path' in info:
                            print(f"    Mesh path: {info['mesh_path']}")
                        
                        if 'is_mesh' in info and info['is_mesh']:
                            print(f"    [MESH]")
                        if 'is_compound' in info and info['is_compound']:
                            print(f"    [COMPOUND]")
                        
                        print()
                except:
                    pass
        
        return results
        
    finally:
        pr.stop()
        pr.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Extract and export mesh data from RLBench TTM files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all objects in a task
  python extract_mesh_from_ttm.py --task close_jar --verbose
  
  # Export all task objects (auto-filters floor/boundary)
  python extract_mesh_from_ttm.py --task close_jar --export
  
  # Export specific objects only
  python extract_mesh_from_ttm.py --task close_jar --objects jar_lid0 jar0 --export
  
  # Export to custom directory
  python extract_mesh_from_ttm.py --task close_box --export --output /path/to/meshes
  
  # Export multiple tasks
  python extract_mesh_from_ttm.py --task close_jar close_box --export
        """
    )
    parser.add_argument('--task', type=str, nargs='+', help='Task name(s) (e.g., close_jar)')
    parser.add_argument('--ttm', type=str, help='Direct path to TTM file')
    parser.add_argument('--objects', type=str, nargs='+', help='Specific object names to export')
    parser.add_argument('--export', action='store_true', help='Export meshes to OBJ files')
    parser.add_argument('--output', type=str, help='Output directory (default: ./meshes)')
    parser.add_argument('--verbose', action='store_true', help='Show all objects in scene')
    parser.add_argument('--rlbench-root', type=str, help='RLBench root directory')
    
    args = parser.parse_args()
    
    # Determine export directory
    export_dir = None
    if args.export:
        if args.output:
            export_dir = Path(args.output)
        else:
            # Default: meshes/ in the same directory as script
            export_dir = Path(__file__).resolve().parent / 'meshes'
    
    # Process tasks
    tasks_to_process = []
    
    if args.ttm:
        # Direct TTM file
        tasks_to_process.append({
            'ttm_file': Path(args.ttm),
            'task_name': Path(args.ttm).stem
        })
    elif args.task:
        # Task names
        if args.rlbench_root:
            rlbench_root = Path(args.rlbench_root)
            task_ttms_dir = rlbench_root / 'rlbench' / 'task_ttms'
        else:
            task_ttms_dir = Path(rlbench.__file__).resolve().parent / 'task_ttms'
        
        for task_name in args.task:
            ttm_file = task_ttms_dir / f'{task_name}.ttm'
            
            if not ttm_file.exists():
                print(f"❌ TTM file not found for task '{task_name}': {ttm_file}")
                continue
            
            tasks_to_process.append({
                'ttm_file': ttm_file,
                'task_name': task_name
            })
    else:
        parser.print_help()
        return
    
    # Process each task
    for i, task_info in enumerate(tasks_to_process):
        if len(tasks_to_process) > 1:
            print(f"\n{'#'*70}")
            print(f"# Task {i+1}/{len(tasks_to_process)}: {task_info['task_name']}")
            print(f"{'#'*70}")
        
        results = analyze_ttm_file(
            task_info['ttm_file'],
            target_objects=args.objects,
            verbose=args.verbose,
            export_dir=export_dir,
            task_name=task_info['task_name']
        )
        
        if results:
            print(f"\n{'='*70}")
            print("Summary:")
            print(f"{'='*70}")
            print(f"Task: {task_info['task_name']}")
            print(f"TTM file: {results['ttm_file']}")
            print(f"Total objects: {results['total_objects']}")
            
            if export_dir:
                print(f"Exported meshes: {results['exported_count']}")
            
            if results['target_objects']:
                print(f"Target objects found: {len(results['target_objects'])}")
                for name, info in results['target_objects'].items():
                    has_mesh_path = 'mesh_path' in info
                    status = "✓" if has_mesh_path else "⚠"
                    print(f"  {status} {name}: {'has mesh path' if has_mesh_path else 'embedded in TTM'}")


if __name__ == '__main__':
    main()
