#!/usr/bin/env python3
"""
Evaluate pose-estimation quality for all demos of a task.

For each demo, at frame 0:
  1. Load the observed point cloud of moving obj (pb) and base obj (pa) from the pkl.
  2. Load the corresponding meshes, sample them to point clouds, and transform
     them by the estimated poses stored in the pkl.
  3. Run ICP between the observed cloud and the pose-transformed mesh cloud to
     measure alignment quality (fitness ↑, RMSE ↓).
  4. Report and rank demos by ICP fitness, separately for moving and base objects.

Optional: visualise the raw vs mesh-aligned clouds for each demo.

Usage:
    python analyze_fp_quality.py --task_name close_jar
    python analyze_fp_quality.py --task_name close_jar --vis
    python analyze_fp_quality.py --task_name close_jar --mesh_dir ../assets/RLBench_mesh
    python analyze_fp_quality.py  # all tasks in dataset_dir
"""

import sys
import argparse
import pickle
import numpy as np
import open3d as o3d
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "../../foci_dataset"
DEFAULT_MESH_DIR = SCRIPT_DIR / "../assets/RLBench_mesh"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

import sys as _sys
_sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from foci_policy.config.config_utils import get_mesh_names as _get_mesh_names_cfg


def _get_mesh_names(task_name: str):
    """Return (moving_mesh_name, base_mesh_name) or raise KeyError."""
    names = _get_mesh_names_cfg(task_name)
    if not names or len(names) < 2:
        raise KeyError(task_name)
    return names[0], names[1]


# ---------------------------------------------------------------------------
# Mesh → point cloud
# ---------------------------------------------------------------------------

def mesh_to_pcd(mesh_path: Path, pose: np.ndarray, n_points: int = 4096) -> o3d.geometry.PointCloud:
    """Load mesh, sample n_points, transform by pose, return PointCloud."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices():
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.transform(pose)
    return pcd


# ---------------------------------------------------------------------------
# ICP alignment
# ---------------------------------------------------------------------------

def icp_align(source: o3d.geometry.PointCloud,
              target: o3d.geometry.PointCloud,
              max_dist: float = 0.05) -> dict:
    """
    Align source → target with point-to-point ICP.
    Returns dict with keys: fitness, inlier_rmse, transformation.
    fitness ∈ [0,1]: fraction of correspondences within max_dist (↑ better).
    inlier_rmse: RMSE of inlier correspondences (↓ better).
    """
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return {
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "transformation": result.transformation,
    }


# ---------------------------------------------------------------------------
# Per-demo evaluation
# ---------------------------------------------------------------------------

def evaluate_demo(pkl_path: Path, moving_mesh: Path, base_mesh: Path,
                  n_mesh_points: int = 4096, icp_max_dist: float = 0.05) -> dict:
    """
    Load frame 0 from pkl, run ICP for moving and base objects.
    Returns dict with keys:
        moving_fitness, moving_rmse, base_fitness, base_rmse
    or raises on error.
    """
    with open(pkl_path, "rb") as f:
        dp = pickle.load(f)

    frame0 = dp["frames_data"][0]
    pa_pose = frame0["pa_pose_mat"]  # base
    pb_pose = frame0["pb_pose_mat"]  # moving

    # Observed point clouds (from pkl)
    obs_moving = o3d.geometry.PointCloud()
    obs_moving.points = o3d.utility.Vector3dVector(frame0["pb_points"])
    obs_base = o3d.geometry.PointCloud()
    obs_base.points = o3d.utility.Vector3dVector(frame0["pa_points"])

    results = {}

    # Moving object (pb)
    if pb_pose is not None and moving_mesh.exists():
        mesh_moving = mesh_to_pcd(moving_mesh, pb_pose, n_mesh_points)
        r = icp_align(mesh_moving, obs_moving, icp_max_dist)
        results["moving_fitness"] = r["fitness"]
        results["moving_rmse"] = r["inlier_rmse"]
        results["moving_icp_transform"] = r["transformation"]
        results["mesh_moving_pcd"] = mesh_moving
    else:
        results["moving_fitness"] = None
        results["moving_rmse"] = None

    # Base object (pa)
    if pa_pose is not None and base_mesh.exists():
        mesh_base = mesh_to_pcd(base_mesh, pa_pose, n_mesh_points)
        r = icp_align(mesh_base, obs_base, icp_max_dist)
        results["base_fitness"] = r["fitness"]
        results["base_rmse"] = r["inlier_rmse"]
        results["base_icp_transform"] = r["transformation"]
        results["mesh_base_pcd"] = mesh_base
    else:
        results["base_fitness"] = None
        results["base_rmse"] = None

    results["obs_moving_pcd"] = obs_moving
    results["obs_base_pcd"] = obs_base
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _colorize(pcd: o3d.geometry.PointCloud, rgb) -> o3d.geometry.PointCloud:
    c = o3d.geometry.PointCloud(pcd)
    c.paint_uniform_color(rgb)
    return c


def visualize_demo(demo_name: str, eval_result: dict):
    """
    Show observed (grey) vs mesh-aligned (colored) clouds for one demo.
    Blue = moving obj mesh, Red = base obj mesh.
    Green = observed moving, Orange = observed base.
    """
    geoms = []

    obs_moving = eval_result.get("obs_moving_pcd")
    obs_base = eval_result.get("obs_base_pcd")
    mesh_moving = eval_result.get("mesh_moving_pcd")
    mesh_base = eval_result.get("mesh_base_pcd")

    if obs_moving is not None:
        geoms.append(_colorize(obs_moving, [0.2, 0.8, 0.2]))   # green
    if obs_base is not None:
        geoms.append(_colorize(obs_base, [1.0, 0.5, 0.0]))      # orange
    if mesh_moving is not None:
        t = eval_result.get("moving_icp_transform", np.eye(4))
        aligned = o3d.geometry.PointCloud(mesh_moving)
        aligned.transform(t)
        geoms.append(_colorize(aligned, [0.2, 0.2, 1.0]))       # blue
    if mesh_base is not None:
        t = eval_result.get("base_icp_transform", np.eye(4))
        aligned = o3d.geometry.PointCloud(mesh_base)
        aligned.transform(t)
        geoms.append(_colorize(aligned, [1.0, 0.2, 0.2]))       # red

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    mf = eval_result.get("moving_fitness")
    bf = eval_result.get("base_fitness")
    title = (f"{demo_name}  |  moving fitness={mf:.3f}  base fitness={bf:.3f}"
             if mf is not None and bf is not None else demo_name)
    print(f"  [vis] {title}  (close window to continue)")
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1280, height=720)


# ---------------------------------------------------------------------------
# Task-level report
# ---------------------------------------------------------------------------

def report_task(task_name: str, task_dir: Path, mesh_dir: Path,
                visualise: bool,
                n_mesh_points: int, icp_max_dist: float):
    try:
        moving_mesh_name, base_mesh_name = _get_mesh_names(task_name)
    except KeyError:
        print(f"  [SKIP] '{task_name}' not in task_obj_config.yaml")
        return

    task_mesh_dir = mesh_dir / task_name
    moving_mesh = task_mesh_dir / f"{moving_mesh_name}.obj"
    base_mesh = task_mesh_dir / f"{base_mesh_name}.obj"

    pkls = sorted(task_dir.glob("*.pkl"))
    if not pkls:
        print(f"  [SKIP] no pkl files in {task_dir}")
        return

    print(f"\n{'━' * 64}")
    print(f"  Task : {task_name}   ({len(pkls)} demos)")
    print(f"  Meshes: moving={moving_mesh_name}.obj  base={base_mesh_name}.obj")
    print(f"{'━' * 64}")

    rows = []
    for pkl_path in pkls:
        try:
            res = evaluate_demo(pkl_path, moving_mesh, base_mesh,
                                n_mesh_points=n_mesh_points,
                                icp_max_dist=icp_max_dist)
            rows.append((pkl_path.name, res))
            if visualise:
                visualize_demo(pkl_path.name, res)
        except Exception as exc:
            print(f"  [ERR] {pkl_path.name}: {exc}")

    if not rows:
        return

    # Sort by moving fitness desc, then base fitness desc
    def _sort_key(r):
        mf = r[1].get("moving_fitness") or 0.0
        bf = r[1].get("base_fitness") or 0.0
        return (-mf, -bf)

    rows_sorted = sorted(rows, key=_sort_key)

    # Header
    print(f"\n  {'Demo':<20}  {'moving_fit':>10}  {'moving_rmse':>11}  "
          f"{'base_fit':>8}  {'base_rmse':>9}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*9}")
    for name, res in rows_sorted:
        mf = res.get("moving_fitness")
        mr = res.get("moving_rmse")
        bf = res.get("base_fitness")
        br = res.get("base_rmse")
        mf_s = f"{mf:.4f}" if mf is not None else "N/A"
        mr_s = f"{mr:.4f}" if mr is not None else "N/A"
        bf_s = f"{bf:.4f}" if bf is not None else "N/A"
        br_s = f"{br:.4f}" if br is not None else "N/A"
        print(f"  {name:<20}  {mf_s:>10}  {mr_s:>11}  {bf_s:>8}  {br_s:>9}")

    # Best demo recommendation
    best_moving = max(rows, key=lambda r: r[1].get("moving_fitness") or 0.0)
    best_base = max(rows, key=lambda r: r[1].get("base_fitness") or 0.0)
    print(f"\n  Best for grasp (moving obj): {best_moving[0]}"
          f"  (moving_fitness={best_moving[1].get('moving_fitness'):.4f})")
    print(f"  Best for manip  (base obj) : {best_base[0]}"
          f"  (base_fitness={best_base[1].get('base_fitness'):.4f})")


# ---------------------------------------------------------------------------
# Public API: select best demo from a candidate pool
# ---------------------------------------------------------------------------

_best_demo_cache: dict = {}  # (task_dir, pool_tuple, n_points, max_dist) → (fname, score)

def select_best_demo(task_dir: Path, task_name: str, mesh_dir: Path, mode: str,
                     pool_fnames: list, n_mesh_points: int = 4096,
                     icp_max_dist: float = 0.05) -> str:
    """
    Evaluate ICP alignment quality for demos in pool_fnames and return the best one.
    Selects by average of moving_fitness and base_fitness (same result for both modes).
    Results are cached so grasp and manip always pick the same demo.
    Falls back to pool_fnames[0] on any error.
    """
    cache_key = (str(task_dir), tuple(pool_fnames), n_mesh_points, icp_max_dist)
    if cache_key in _best_demo_cache:
        best_fname, best_score = _best_demo_cache[cache_key]
        print(f"[optimal] {task_name}/{mode}: '{best_fname}' (avg_fitness={best_score:.3f}) [cached]")
        return best_fname

    try:
        moving_mesh_name, base_mesh_name = _get_mesh_names(task_name)
    except KeyError:
        return pool_fnames[0]

    task_mesh_dir = mesh_dir / task_name
    moving_mesh = task_mesh_dir / f"{moving_mesh_name}.obj"
    base_mesh   = task_mesh_dir / f"{base_mesh_name}.obj"

    best_fname, best_score = pool_fnames[0], float('-inf')
    for fname in pool_fnames:
        try:
            res = evaluate_demo(task_dir / fname, moving_mesh, base_mesh,
                                n_mesh_points, icp_max_dist)
            mf = res.get('moving_fitness') or 0.0
            bf = res.get('base_fitness') or 0.0
            score = (mf + bf) / 2.0
            if score > best_score:
                best_score, best_fname = score, fname
        except Exception:
            continue

    _best_demo_cache[cache_key] = (best_fname, best_score)
    print(f"[optimal] {task_name}/{mode}: '{best_fname}' (avg_fitness={best_score:.3f})")
    return best_fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank FOCI demos by ICP pose-alignment quality at frame 0."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR),
                        help="Path to foci_dataset directory.")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Analyse a single task (default: all tasks).")
    parser.add_argument("--mesh_dir", type=str, default=str(DEFAULT_MESH_DIR),
                        help="Directory containing per-task mesh sub-folders.")
    parser.add_argument("--vis", action="store_true", default=False,
                        help="Visualise observed vs mesh clouds for each demo.")
    parser.add_argument("--n_mesh_points", type=int, default=4096,
                        help="Points sampled from each mesh (default 4096).")
    parser.add_argument("--icp_max_dist", type=float, default=0.05,
                        help="ICP max correspondence distance in metres (default 0.05).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    mesh_dir = Path(args.mesh_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset not found: {dataset_dir}")
        sys.exit(1)

    excluded = {"processed", "raw", "__pycache__"}
    if args.task_name:
        task_dirs = [dataset_dir / args.task_name]
        if not task_dirs[0].exists():
            print(f"ERROR: task directory not found: {task_dirs[0]}")
            sys.exit(1)
    else:
        task_dirs = sorted(
            d for d in dataset_dir.iterdir()
            if d.is_dir() and d.name not in excluded
        )

    print("=" * 64)
    print("  FOCI Demo Pose-Alignment Quality Report  (ICP @ frame 0)")
    print(f"  Dataset  : {dataset_dir}")
    print(f"  Mesh dir : {mesh_dir}")
    print(f"  Tasks    : {len(task_dirs)}")
    print("=" * 64)

    for td in task_dirs:
        report_task(
            task_name=td.name,
            task_dir=td,
            mesh_dir=mesh_dir,
            visualise=args.vis,
            n_mesh_points=args.n_mesh_points,
            icp_max_dist=args.icp_max_dist,
        )

    print(f"\n{'=' * 64}\n")


if __name__ == "__main__":
    main()
