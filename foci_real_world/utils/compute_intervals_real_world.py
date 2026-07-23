#!/usr/bin/env python3
"""
Interval discovery for real-world FOCI demos.

Workflow:
1) Infer key frames from gripper state transitions.
2) Reuse change-point style logic from compute_interaction_intervals.
3) Compute grasp/manip intervals.
4) Visualize grasp/manip key frames (4 views) with Open3D.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _fallback_compute_velocity(positions: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
	if len(positions) <= 1:
		return np.array([]), np.array([])
	velocities = np.diff(positions, axis=0) / dt
	speeds = np.linalg.norm(velocities, axis=1)
	return velocities, speeds


def _fallback_compute_angular_velocity(quaternions: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
	if len(quaternions) <= 1:
		return np.array([]), np.array([])
	angular_velocities = []
	for i in range(len(quaternions) - 1):
		q1 = R.from_quat(quaternions[i])
		q2 = R.from_quat(quaternions[i + 1])
		q_rel = q2 * q1.inv()
		rotvec = q_rel.as_rotvec()
		angular_velocities.append(rotvec / dt)
	angular_velocities = np.array(angular_velocities)
	angular_speeds = np.linalg.norm(angular_velocities, axis=1)
	return angular_velocities, angular_speeds


def _fallback_normalize_and_combine_signals(
	linear: np.ndarray,
	angular: np.ndarray,
	distance: np.ndarray,
	return_magnitude: bool = False,
) -> np.ndarray:
	linear_norm = (linear - np.min(linear)) / (np.max(linear) - np.min(linear) + 1e-8)
	angular_norm = (angular - np.min(angular)) / (np.max(angular) - np.min(angular) + 1e-8)
	distance_norm = (distance - np.min(distance)) / (np.max(distance) - np.min(distance) + 1e-8)
	stacked = np.column_stack([linear_norm, angular_norm, distance_norm])
	if return_magnitude:
		return np.linalg.norm(stacked, axis=1)
	return stacked


def _fallback_detect_change_points_adaptive(signal: np.ndarray, min_size: int = 3, pen_factor: float = 1.0) -> List[int]:
	if len(signal) < min_size * 2:
		return []
	if signal.ndim == 1:
		signal = signal.reshape(-1, 1)
	n_samples = len(signal)
	try:
		import ruptures as rpt

		algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
		adjusted_pen = pen_factor * 0.5 if n_samples < 50 else pen_factor
		penalty_value = adjusted_pen * np.log(n_samples) * signal.shape[1]
		change_points = algo.predict(pen=penalty_value)
		if change_points and change_points[-1] == n_samples:
			change_points = change_points[:-1]
		if len(change_points) == 0 and n_samples >= 20:
			change_points = algo.predict(pen=penalty_value * 0.3)
			if change_points and change_points[-1] == n_samples:
				change_points = change_points[:-1]
		return change_points
	except Exception as exc:
		print(f"Change point detection failed: {exc}")
		return []


def _fallback_find_interaction_start_with_changepoints(keyframe_idx: int, change_points: List[int]) -> int:
	if keyframe_idx <= 0:
		return 0
	valid = [cp for cp in change_points if cp < keyframe_idx]
	if not valid:
		return 0
	return max(valid)


try:
	from foci_policy.utils.compute_interaction_intervals import (
		compute_angular_velocity,
		compute_velocity,
		detect_change_points_adaptive,
		find_interaction_start_with_changepoints,
		normalize_and_combine_signals,
	)
except Exception:
	# RLBench dependencies may be unavailable in real-world-only runs.
	compute_velocity = _fallback_compute_velocity
	compute_angular_velocity = _fallback_compute_angular_velocity
	normalize_and_combine_signals = _fallback_normalize_and_combine_signals
	detect_change_points_adaptive = _fallback_detect_change_points_adaptive
	find_interaction_start_with_changepoints = _fallback_find_interaction_start_with_changepoints


def load_json(path: Path) -> Dict:
	with open(path, "r") as f:
		return json.load(f)


def load_trajectory(path: Path) -> List[Dict]:
	with open(path, "r") as f:
		return json.load(f)


def load_camera_extrinsic(extrinsics_path: Path) -> np.ndarray:
	extrinsics = load_json(extrinsics_path)
	tx = extrinsics["translation"]["x"]
	ty = extrinsics["translation"]["y"]
	tz = extrinsics["translation"]["z"]
	qx = extrinsics["rotation"]["x"]
	qy = extrinsics["rotation"]["y"]
	qz = extrinsics["rotation"]["z"]
	qw = extrinsics["rotation"]["w"]
	rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

	extrinsic = np.eye(4)
	extrinsic[:3, :3] = rotation
	extrinsic[:3, 3] = np.array([tx, ty, tz])
	return extrinsic


def parse_gripper_pose(item: Dict) -> Tuple[np.ndarray, np.ndarray]:
	pos = item["gripper_pose"]["position"]
	ori = item["gripper_pose"]["orientation"]
	position = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
	quat = np.array([ori["x"], ori["y"], ori["z"], ori["w"]], dtype=np.float64)
	quat = quat / (np.linalg.norm(quat) + 1e-8)
	return position, quat


def pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
	mat = np.eye(4)
	mat[:3, :3] = R.from_quat(quaternion).as_matrix()
	mat[:3, 3] = position
	return mat


def infer_keyframes_from_gripper_state(trajectory: List[Dict]) -> Dict[str, int]:
	states = [str(item.get("gripper_state", "")).strip().lower() for item in trajectory]
	n = len(states)
	if n == 0:
		raise ValueError("Empty trajectory.")

	def is_open(state: str) -> bool:
		return state in ("open", "opened", "0", "false")

	pick_frame = None
	for i in range(1, n):
		if is_open(states[i - 1]) and not is_open(states[i]):
			pick_frame = i
			break

	if pick_frame is None:
		closed_indices = [i for i, s in enumerate(states) if not is_open(s)]
		pick_frame = closed_indices[0] if closed_indices else n // 3

	open_after_pick = None
	for i in range(pick_frame + 1, n):
		if not is_open(states[i - 1]) and is_open(states[i]):
			open_after_pick = i
			break

	if open_after_pick is not None:
		place_frame = max(pick_frame, open_after_pick - 1)
	else:
		closed_after_pick = [i for i in range(pick_frame, n) if not is_open(states[i])]
		place_frame = closed_after_pick[-1] if closed_after_pick else n - 1

	if place_frame <= pick_frame:
		place_frame = min(n - 1, max(pick_frame + 1, n - 1))

	return {
		"pick_frame": int(pick_frame),
		"place_frame": int(place_frame),
	}


def compute_bbox_center(points: np.ndarray) -> Optional[np.ndarray]:
	if points is None or len(points) == 0:
		return None
	bbox_min = np.min(points, axis=0)
	bbox_max = np.max(points, axis=0)
	return 0.5 * (bbox_min + bbox_max)


def image_to_world_points(
	rgb: np.ndarray,
	depth: np.ndarray,
	mask: Optional[np.ndarray],
	intrinsics: Dict,
	extrinsic: np.ndarray,
	depth_scale: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
	depth_m = depth.astype(np.float32) / depth_scale

	fx = intrinsics["K"][0]
	fy = intrinsics["K"][4]
	cx = intrinsics["K"][2]
	cy = intrinsics["K"][5]

	h, w = depth_m.shape
	if mask is None:
		valid = (depth_m > 0.01) & (depth_m < 3.0)
	else:
		if mask.shape != depth_m.shape:
			mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
		valid = (mask > 0) & (depth_m > 0.01) & (depth_m < 3.0)

	if not np.any(valid):
		return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

	v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
	v_valid = v[valid]
	u_valid = u[valid]
	z_valid = depth_m[valid]

	x = (u_valid - cx) * z_valid / fx
	y = (v_valid - cy) * z_valid / fy
	points_cam = np.stack([x, y, z_valid], axis=1)
	points_world = (extrinsic[:3, :3] @ points_cam.T).T + extrinsic[:3, 3]
	colors = rgb[valid].astype(np.float32) / 255.0
	return points_world, colors


def read_mask_for_object(demo_dir: Path, object_name: str, frame_idx: int) -> Optional[np.ndarray]:
	frame_name = f"{frame_idx:04d}.png"
	xmem_path = demo_dir / f"xmem_output_{object_name}" / "masks" / frame_name
	if xmem_path.exists():
		return cv2.imread(str(xmem_path), cv2.IMREAD_GRAYSCALE)

	raw_mask_path = demo_dir / "mask" / object_name / frame_name
	if raw_mask_path.exists():
		return cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)

	return None


def load_frame_data(
	demo_dir: Path,
	frame_idx: int,
	intrinsics: Dict,
	extrinsic: np.ndarray,
	base_object: str,
	moving_object: str,
) -> Dict[str, np.ndarray]:
	frame_name = f"{frame_idx:04d}.png"
	rgb_path = demo_dir / "color" / frame_name
	depth_path = demo_dir / "depth" / frame_name

	rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
	if rgb_bgr is None:
		raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
	rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

	depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
	if depth is None:
		raise FileNotFoundError(f"Depth frame not found: {depth_path}")

	base_mask = read_mask_for_object(demo_dir, base_object, frame_idx)
	moving_mask = read_mask_for_object(demo_dir, moving_object, frame_idx)

	scene_points, scene_colors = image_to_world_points(rgb, depth, None, intrinsics, extrinsic)
	base_points, base_colors = image_to_world_points(rgb, depth, base_mask, intrinsics, extrinsic)
	moving_points, moving_colors = image_to_world_points(rgb, depth, moving_mask, intrinsics, extrinsic)

	return {
		"scene_points": scene_points,
		"scene_colors": scene_colors,
		"base_points": base_points,
		"base_colors": base_colors,
		"moving_points": moving_points,
		"moving_colors": moving_colors,
	}


def extract_signals_from_realworld_demo(
	demo_dir: Path,
	trajectory: List[Dict],
	intrinsics: Dict,
	extrinsic: np.ndarray,
	base_object: str,
	moving_object: str,
) -> Dict[str, np.ndarray]:
	timesteps = []
	positions = []
	quats = []
	g2m_distances = []
	m2b_distances = []

	for frame_idx, item in enumerate(trajectory):
		timesteps.append(frame_idx)
		pos, quat = parse_gripper_pose(item)
		positions.append(pos)
		quats.append(quat)

		try:
			frame_data = load_frame_data(
				demo_dir=demo_dir,
				frame_idx=frame_idx,
				intrinsics=intrinsics,
				extrinsic=extrinsic,
				base_object=base_object,
				moving_object=moving_object,
			)
			moving_center = compute_bbox_center(frame_data["moving_points"])
			base_center = compute_bbox_center(frame_data["base_points"])

			if moving_center is not None:
				g2m_distances.append(float(np.linalg.norm(pos - moving_center)))
			else:
				g2m_distances.append(g2m_distances[-1] if g2m_distances else 1.0)

			if moving_center is not None and base_center is not None:
				m2b_distances.append(float(np.linalg.norm(moving_center - base_center)))
			else:
				m2b_distances.append(m2b_distances[-1] if m2b_distances else 1.0)
		except Exception:
			g2m_distances.append(g2m_distances[-1] if g2m_distances else 1.0)
			m2b_distances.append(m2b_distances[-1] if m2b_distances else 1.0)

	positions = np.array(positions)
	quats = np.array(quats)
	_, linear_speeds = compute_velocity(positions, dt=1.0)
	_, angular_speeds = compute_angular_velocity(quats, dt=1.0)

	if len(linear_speeds) > 0:
		linear_speeds = np.concatenate([linear_speeds, [linear_speeds[-1]]])
		angular_speeds = np.concatenate([angular_speeds, [angular_speeds[-1]]])
	else:
		linear_speeds = np.zeros(len(timesteps))
		angular_speeds = np.zeros(len(timesteps))

	return {
		"timesteps": np.array(timesteps),
		"gripper_positions": positions,
		"gripper_orientations": quats,
		"gripper_linear_speeds": linear_speeds,
		"gripper_angular_speeds": angular_speeds,
		"gripper_to_moving_distance": np.array(g2m_distances),
		"moving_to_base_distance": np.array(m2b_distances),
	}


def compute_intervals_realworld(
	signals: Dict[str, np.ndarray],
	pick_frame: int,
	place_frame: int,
	pen_factor: float = 0.5,
) -> Dict[str, int]:
	total_len = len(signals["timesteps"])

	grasp_end = min(pick_frame + 1, total_len)
	grasp_combined = normalize_and_combine_signals(
		signals["gripper_linear_speeds"][:grasp_end],
		signals["gripper_angular_speeds"][:grasp_end],
		signals["gripper_to_moving_distance"][:grasp_end],
	)
	grasp_cps = detect_change_points_adaptive(grasp_combined, min_size=3, pen_factor=pen_factor)
	grasp_start = find_interaction_start_with_changepoints(pick_frame, grasp_cps)

	manip_end = min(place_frame + 1, total_len)
	manip_start_search = max(pick_frame, 0)
	manip_combined = normalize_and_combine_signals(
		signals["gripper_linear_speeds"][manip_start_search:manip_end],
		signals["gripper_angular_speeds"][manip_start_search:manip_end],
		signals["moving_to_base_distance"][manip_start_search:manip_end],
	)
	manip_cps_local = detect_change_points_adaptive(manip_combined, min_size=3, pen_factor=pen_factor)
	manip_cps = [cp + manip_start_search for cp in manip_cps_local]
	manip_start = find_interaction_start_with_changepoints(place_frame, manip_cps)

	return {
		"pick_frame": int(pick_frame),
		"place_frame": int(place_frame),
		"grasp_start": int(grasp_start),
		"manip_start": int(manip_start),
		"grasp_interval": [int(grasp_start), int(pick_frame)],
		"manip_interval": [int(manip_start), int(place_frame)],
		"change_points_grasp": [int(v) for v in grasp_cps],
		"change_points_manip": [int(v) for v in manip_cps],
	}


def make_point_cloud(points: np.ndarray, colors: np.ndarray, brightness_gain: float = 1.0) -> o3d.geometry.PointCloud:
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	colors_out = np.clip(colors * brightness_gain, 0.0, 1.0)
	pcd.colors = o3d.utility.Vector3dVector(colors_out)
	return pcd


def add_gripper_frame(position: np.ndarray, quaternion: np.ndarray, size: float = 0.05) -> o3d.geometry.TriangleMesh:
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
	frame.transform(pose_to_matrix(position, quaternion))
	return frame


def build_single_frame_geometries(
	frame_data: Dict[str, np.ndarray],
	pose: Tuple[np.ndarray, np.ndarray],
) -> List[o3d.geometry.Geometry]:
	geoms: List[o3d.geometry.Geometry] = []

	world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
	geoms.append(world)

	scene = make_point_cloud(frame_data["scene_points"], frame_data["scene_colors"], brightness_gain=1.18)
	base = make_point_cloud(frame_data["base_points"], frame_data["base_colors"], brightness_gain=1.12)
	moving = make_point_cloud(frame_data["moving_points"], frame_data["moving_colors"], brightness_gain=1.12)
	geoms.extend([scene, base, moving])

	gripper = add_gripper_frame(pose[0], pose[1], size=0.065)
	gripper.paint_uniform_color([1.0, 0.0, 0.0])
	geoms.append(gripper)

	return geoms


def _visualize_single_state(
	demo_dir: Path,
	frame_name: str,
	frame_idx: int,
	trajectory: List[Dict],
	intrinsics: Dict,
	extrinsic: np.ndarray,
	base_object: str,
	moving_object: str,
	left: int,
	top: int,
) -> None:
	frame_data = load_frame_data(demo_dir, frame_idx, intrinsics, extrinsic, base_object, moving_object)
	pose = parse_gripper_pose(trajectory[frame_idx])
	geometries = build_single_frame_geometries(frame_data, pose)

	o3d.visualization.draw_geometries(
		geometries,
		window_name=f"{demo_dir.name} - {frame_name} (t={frame_idx})",
		width=1280,
		height=760,
		left=left,
		top=top,
	)


def visualize_four_intervals(
	demo_dir: Path,
	grasp_start_idx: int,
	pick_idx: int,
	manip_start_idx: int,
	place_idx: int,
	trajectory: List[Dict],
	intrinsics: Dict,
	extrinsic: np.ndarray,
	base_object: str,
	moving_object: str,
) -> None:
	print("\nVisualizing four key frames in 4 separate windows:")
	print(f"  1) Grasp Start : {grasp_start_idx}")
	print(f"  2) Grasp End   : {pick_idx}")
	print(f"  3) Manip Start : {manip_start_idx}")
	print(f"  4) Manip End   : {place_idx}")

	_visualize_single_state(
		demo_dir=demo_dir,
		frame_name="Grasp Start",
		frame_idx=grasp_start_idx,
		trajectory=trajectory,
		intrinsics=intrinsics,
		extrinsic=extrinsic,
		base_object=base_object,
		moving_object=moving_object,
		left=50,
		top=50,
	)
	_visualize_single_state(
		demo_dir=demo_dir,
		frame_name="Grasp End",
		frame_idx=pick_idx,
		trajectory=trajectory,
		intrinsics=intrinsics,
		extrinsic=extrinsic,
		base_object=base_object,
		moving_object=moving_object,
		left=1400,
		top=50,
	)
	_visualize_single_state(
		demo_dir=demo_dir,
		frame_name="Manip Start",
		frame_idx=manip_start_idx,
		trajectory=trajectory,
		intrinsics=intrinsics,
		extrinsic=extrinsic,
		base_object=base_object,
		moving_object=moving_object,
		left=50,
		top=50,
	)
	_visualize_single_state(
		demo_dir=demo_dir,
		frame_name="Manip End",
		frame_idx=place_idx,
		trajectory=trajectory,
		intrinsics=intrinsics,
		extrinsic=extrinsic,
		base_object=base_object,
		moving_object=moving_object,
		left=1400,
		top=50,
	)


def plot_interaction_analysis(result: Dict, task_name: str, output_dir: str = './analysis_results') -> None:
	"""
	Plot interaction analysis with signals, change points, and detected intervals.

	Creates two plots:
	1) Grasp phase: start -> pick
	2) Manipulation phase: pick -> place
	"""
	os.makedirs(output_dir, exist_ok=True)

	signals = result['signals']
	timesteps = signals['timesteps']
	pick_frame = result['pick_frame']
	place_frame = result['place_frame']
	grasp_start = result['grasp_start']
	manip_start = result['manip_start']
	demo_name = Path(result['demo_dir']).name

	# === Plot 1: Grasp phase ===
	fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
	grasp_end = pick_frame
	grasp_timesteps = timesteps[:grasp_end + 1]

	ax = axes[0]
	ax.plot(grasp_timesteps, signals['gripper_linear_speeds'][:grasp_end + 1], 'b-', linewidth=1.5, label='Linear Speed')
	ax.axvline(pick_frame, color='red', linestyle='--', linewidth=2, label='Pick Frame')
	ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3, label='Interaction Interval')
	for i, cp in enumerate(result.get('change_points_grasp', [])):
		if cp <= grasp_end:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
				label='Change Points' if i == 0 else '')
			ax.text(cp, ax.get_ylim()[1] * 0.95, f'{cp}', fontsize=8, ha='center', color='green')
	ax.set_ylabel('Linear Speed (m/s)', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)
	ax.set_title(f'Grasp Phase Analysis - {task_name} ({demo_name})', fontsize=13, fontweight='bold')

	ax = axes[1]
	ax.plot(grasp_timesteps, signals['gripper_angular_speeds'][:grasp_end + 1], 'g-', linewidth=1.5, label='Angular Speed')
	ax.axvline(pick_frame, color='red', linestyle='--', linewidth=2)
	ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3)
	for cp in result.get('change_points_grasp', []):
		if cp <= grasp_end:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax.set_ylabel('Angular Speed (rad/s)', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)

	ax = axes[2]
	ax.plot(grasp_timesteps, signals['gripper_to_moving_distance'][:grasp_end + 1], 'c-', linewidth=1.5, label='Gripper-Object Distance')
	ax.axvline(pick_frame, color='red', linestyle='--', linewidth=2)
	ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3)
	for cp in result.get('change_points_grasp', []):
		if cp <= grasp_end:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax.set_ylabel('Distance (m)', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)

	ax = axes[3]
	grasp_combined = normalize_and_combine_signals(
		signals['gripper_linear_speeds'][:grasp_end + 1],
		signals['gripper_angular_speeds'][:grasp_end + 1],
		signals['gripper_to_moving_distance'][:grasp_end + 1],
		return_magnitude=True,
	)
	ax.plot(grasp_timesteps, grasp_combined, 'purple', linewidth=2, label='Combined Signal')
	ax.axvline(pick_frame, color='red', linestyle='--', linewidth=2)
	ax.axvspan(grasp_start, pick_frame, color='yellow', alpha=0.3)
	for cp in result.get('change_points_grasp', []):
		if cp <= grasp_end:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax.set_ylabel('Signal Magnitude', fontsize=11)
	ax.set_xlabel('Frame', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	grasp_plot_path = os.path.join(output_dir, f'{task_name}_{demo_name}_grasp_phase.png')
	plt.savefig(grasp_plot_path, dpi=150, bbox_inches='tight')
	print(f"Grasp phase plot saved to {grasp_plot_path}")
	plt.close()

	# === Plot 2: Manipulation phase ===
	fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
	manip_timesteps = timesteps[pick_frame:place_frame + 1]

	ax = axes[0]
	ax.plot(manip_timesteps, signals['gripper_linear_speeds'][pick_frame:place_frame + 1], 'b-', linewidth=1.5, label='Linear Speed')
	ax.axvline(place_frame, color='red', linestyle='--', linewidth=2, label='Place Frame')
	ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3, label='Interaction Interval')
	for i, cp in enumerate(result.get('change_points_manip', [])):
		if pick_frame <= cp <= place_frame:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
				label='Change Points' if i == 0 else '')
			ax.text(cp, ax.get_ylim()[1] * 0.95, f'{cp}', fontsize=8, ha='center', color='green')
	ax.set_ylabel('Linear Speed (m/s)', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)
	ax.set_title(f'Manipulation Phase Analysis - {task_name} ({demo_name})', fontsize=13, fontweight='bold')

	ax = axes[1]
	ax.plot(manip_timesteps, signals['gripper_angular_speeds'][pick_frame:place_frame + 1], 'g-', linewidth=1.5, label='Angular Speed')
	ax.axvline(place_frame, color='red', linestyle='--', linewidth=2)
	ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3)
	for cp in result.get('change_points_manip', []):
		if pick_frame <= cp <= place_frame:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax.set_ylabel('Angular Speed (rad/s)', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)

	ax = axes[2]
	ax.plot(manip_timesteps, signals['moving_to_base_distance'][pick_frame:place_frame + 1], 'c-', linewidth=1.5, label='Object-Base Distance')
	ax.axvline(place_frame, color='red', linestyle='--', linewidth=2)
	ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3)
	for cp in result.get('change_points_manip', []):
		if pick_frame <= cp <= place_frame:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax.set_ylabel('Distance (m)', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)

	ax = axes[3]
	manip_combined = normalize_and_combine_signals(
		signals['gripper_linear_speeds'][pick_frame:place_frame + 1],
		signals['gripper_angular_speeds'][pick_frame:place_frame + 1],
		signals['moving_to_base_distance'][pick_frame:place_frame + 1],
		return_magnitude=True,
	)
	ax.plot(manip_timesteps, manip_combined, 'purple', linewidth=2, label='Combined Signal')
	ax.axvline(place_frame, color='red', linestyle='--', linewidth=2)
	ax.axvspan(manip_start, place_frame, color='yellow', alpha=0.3)
	for cp in result.get('change_points_manip', []):
		if pick_frame <= cp <= place_frame:
			ax.axvline(cp, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax.set_ylabel('Signal Magnitude', fontsize=11)
	ax.set_xlabel('Frame', fontsize=11)
	ax.legend(loc='upper right', fontsize=9)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	manip_plot_path = os.path.join(output_dir, f'{task_name}_{demo_name}_manip_phase.png')
	plt.savefig(manip_plot_path, dpi=150, bbox_inches='tight')
	print(f"Manipulation phase plot saved to {manip_plot_path}")
	plt.close()


def discover_intervals_for_demo(
	demo_dir: Path,
	pen_factor: float = 0.5,
	visualize: bool = True,
	plot_results: bool = False,
	output_dir: str = './analysis_results',
	save_json: Optional[Path] = None,
) -> Dict:
	if not demo_dir.exists():
		raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

	task_info_path = demo_dir / "task_info.json"
	traj_path = demo_dir / "trajectory.json"
	intr_path = demo_dir / "camera_intrinsics.json"
	ext_path = demo_dir / "camera_extrinsics.json"

	task_info = load_json(task_info_path)
	trajectory = load_trajectory(traj_path)
	intrinsics = load_json(intr_path)
	extrinsic = load_camera_extrinsic(ext_path)

	base_object = task_info["base_object"]
	moving_object = task_info["moving_object"]

	keyframes = infer_keyframes_from_gripper_state(trajectory)
	pick_frame = keyframes["pick_frame"]
	place_frame = keyframes["place_frame"]

	signals = extract_signals_from_realworld_demo(
		demo_dir=demo_dir,
		trajectory=trajectory,
		intrinsics=intrinsics,
		extrinsic=extrinsic,
		base_object=base_object,
		moving_object=moving_object,
	)

	interval_result = compute_intervals_realworld(
		signals=signals,
		pick_frame=pick_frame,
		place_frame=place_frame,
		pen_factor=pen_factor,
	)

	result = {
		"demo_dir": str(demo_dir),
		"task_name": task_info.get("task_name", "unknown"),
		"base_object": base_object,
		"moving_object": moving_object,
		**interval_result,
	}

	print("\n" + "=" * 70)
	print(f"Demo: {demo_dir.name}")
	print(f"Task: {result['task_name']}")
	print(f"Keyframes from gripper state: pick={pick_frame}, place={place_frame}")
	print(f"Grasp interval: {result['grasp_interval']}")
	print(f"Manip interval: {result['manip_interval']}")
	print(f"Grasp change points: {result['change_points_grasp']}")
	print(f"Manip change points: {result['change_points_manip']}")
	print("=" * 70)

	if plot_results:
		result['signals'] = signals
		plot_interaction_analysis(result, result['task_name'], output_dir)

	if save_json is not None:
		result_to_save = {k: v for k, v in result.items() if k != 'signals'}
		with open(save_json, "w") as f:
			json.dump(result_to_save, f, indent=2)
		print(f"Saved results to: {save_json}")

	if visualize:
		visualize_four_intervals(
			demo_dir=demo_dir,
			grasp_start_idx=result["grasp_interval"][0],
			pick_idx=result["grasp_interval"][1],
			manip_start_idx=result["manip_interval"][0],
			place_idx=result["manip_interval"][1],
			trajectory=trajectory,
			intrinsics=intrinsics,
			extrinsic=extrinsic,
			base_object=base_object,
			moving_object=moving_object,
		)

	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="Real-world interval discovery and visualization")
	parser.add_argument(
		"demo_dir",
		type=str,
		help="Path to a real-world demo directory under foci_real_world/dataset",
	)
	parser.add_argument(
		"--pen_factor",
		type=float,
		default=0.5,
		help="Penalty factor for adaptive change-point detection",
	)
	parser.add_argument(
		"--no-viz",
		action="store_true",
		help="Disable Open3D visualization windows",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Optional output JSON path for discovered intervals",
	)
	parser.add_argument(
		"--plot",
		action="store_true",
		help="Generate grasp/manip signal analysis plots",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default='./analysis_results',
		help="Output directory for plots",
	)

	args = parser.parse_args()

	demo_input = Path(args.demo_dir).expanduser()
	if demo_input.exists():
		demo_dir = demo_input.resolve()
	else:
		# Backward-compatible path resolution after script relocation.
		candidate = PROJECT_ROOT / "foci_real_world" / "dataset" / args.demo_dir
		demo_dir = candidate.resolve()

	if not demo_dir.exists():
		raise FileNotFoundError(
			f"Demo directory not found: {args.demo_dir} (also tried {PROJECT_ROOT / 'foci_real_world' / 'dataset' / args.demo_dir})"
		)

	output = Path(args.output).expanduser().resolve() if args.output else None
	discover_intervals_for_demo(
		demo_dir=demo_dir,
		pen_factor=args.pen_factor,
		visualize=not args.no_viz,
		plot_results=args.plot,
		output_dir=args.output_dir,
		save_json=output,
	)


if __name__ == "__main__":
	main()
