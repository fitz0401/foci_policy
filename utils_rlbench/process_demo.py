
import numpy as np
from typing import List
import os
import sys
import matplotlib.pyplot as plt
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from rlbench.utils import get_stored_demo


goal_frames_need_adjust_task_set = {
    'light_bulb_in',
    'close_jar',
}

class Demo(object):
    def __init__(self, observations, random_seed=None):
        self._observations = observations
        self.random_seed = random_seed

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    # print('==',obs.joint_velocities)
    jv = np.asarray(obs.joint_velocities, dtype=float)
    small_delta = np.allclose(jv, 0., atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def discover_keyframes(demo: Demo, task_name=None) -> List[int]:
    episode_keyframes = []
    prev_gripper_open = demo[0].gripper_open
    for i, obs in enumerate(demo):
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last):
            episode_keyframes.append(i)
        prev_gripper_open = obs.gripper_open
    # detect the last stop frame as goal (place) frame for certain tasks
    if task_name and task_name in goal_frames_need_adjust_task_set:
        stopped_buffer = 0
        stop_frames = []
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, delta=0.1)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            if stopped:
                stop_frames.append(i)
        if len(stop_frames) > 0:
            place_frame = stop_frames[-1]
            if episode_keyframes[-1] != place_frame:
                episode_keyframes.append(place_frame)
                episode_keyframes = sorted(episode_keyframes)
    if len(episode_keyframes) > 1 and (episode_keyframes[-1] - 1) == \
            episode_keyframes[-2]:
        episode_keyframes.pop(-2)
    print('Found %d keyframes.' % len(episode_keyframes), episode_keyframes)
    return episode_keyframes


def get_pick_place_keyframes(demo: Demo, task_name=None) -> List[int]:
    keyframes = discover_keyframes(demo, task_name=task_name)
    if len(keyframes) < 2:
        raise ValueError("Demo does not contain enough keyframes for pick and place.")
    if task_name == "put_item_in_drawer":
        return {'pick': keyframes[2], 'place': keyframes[3]}
    return {'pick': keyframes[0], 'place': keyframes[1]}


def visualize_keyframes(demo: Demo, keyframes: List[int], camera='front'):
    """Visualize RGB images at keyframes."""
    n_keyframes = len(keyframes)
    fig, axes = plt.subplots(1, n_keyframes, figsize=(5*n_keyframes, 5))
    if n_keyframes == 1:
        axes = [axes]
    
    for idx, kf_idx in enumerate(keyframes):
        obs = demo[kf_idx]
        rgb = getattr(obs, f'{camera}_rgb')
        gripper_state = 'OPEN' if obs.gripper_open > 0.5 else 'CLOSED'
        
        axes[idx].imshow(rgb)
        axes[idx].set_title(f'Frame {kf_idx}\nGripper: {gripper_state}', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def flat_pcd_image(obs_dict, CAMERAS, include_mask=True):
    #print(obs_dict)
    obs = []
    pcds = []
    masks = []
    others = {'demo':True}
    others.update(obs_dict)
    # print(list(others.keys()))
    for cname in CAMERAS:
        rgb = '%s_rgb' % cname
        pcd = '%s_point_cloud' % cname
        mask = '%s_mask' % cname
        depth = '%s_depth' % cname
        extrinsics = '%s_camera_extrinsics' % cname
        intrinsics = '%s_camera_intrinsics' % cname
        rgb_data = torch.from_numpy(others[rgb]).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
        #print(rgb_data.shape)
        ##
        rgb_data = (rgb_data.float() / 255.0)
        # import matplotlib.pyplot as plt
        # print(rgb_data.shape)
        # plt.imshow(rgb_data[0,0,:,:,:].permute(1,2,0).numpy())
        # plt.show()
        ##
        rgb_data = stack_on_channel(rgb_data)
        #print(rgb_data.shape)
        # rgb_data = _norm_rgb(rgb_data)

        pcd_data = stack_on_channel(torch.from_numpy(others[pcd]).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0))
        obs.append([rgb_data,pcd_data])
        pcds.append(pcd_data)
        if include_mask:
            mask_data = torch.from_numpy(others[mask].astype(np.float32)).to(torch.long).unsqueeze(dim=0).unsqueeze(dim=0)
            ###
            # print(mask_data.shape)
            # plt.imshow(mask_data[0,0,0,:,:].numpy())
            # plt.show()
            ###
            mask_data = stack_on_channel(mask_data)
            masks.append(mask_data)

    bs = obs[0][0].shape[0]
    pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcds], 1)

    image_features = [o[0] for o in obs]
    feat_size = image_features[0].shape[1]
    flat_imag_features = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in image_features], 1)
    flat_mask = None
    if include_mask:
        # flat_mask = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in masks], 1)
        # Masks are single-channel labels, not 3-channel RGB features.
        flat_mask = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 1) for p in masks], 1)
    
    return pcd_flat, flat_imag_features, flat_mask, pcds


if __name__ == "__main__":
    import argparse
    SCRIPT_DIR = os.path.dirname(__file__)
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

    parser = argparse.ArgumentParser(description='Inspect RLBench demo keyframes.')
    parser.add_argument('--task', type=str, default='sweep_to_dustpan_of_size',
                        help='RLBench task name.')
    parser.add_argument('--demo_index', type=int, default=0,
                        help='Index of the demo to load from the episodes folder.')
    parser.add_argument('--data_folder', type=str,
                        default=os.path.join(REPO_ROOT, 'data', 'rlbench_data'),
                        help='Root folder containing task demo folders.')
    parser.add_argument('--variation', type=int, default=0,
                        help='Variation number to load (default: 0).')
    args = parser.parse_args()

    task_name = args.task
    demo_index = args.demo_index
    data_path = os.path.join(
        args.data_folder,
        task_name,
        f'variation{args.variation}',
        'episodes',
    )
    demo = get_stored_demo(
        data_path=data_path,
        index=demo_index,
        init_matrix=False
    )
    
    # Discover all keyframes
    all_keyframes = discover_keyframes(demo, task_name=task_name)
    print(f"All keyframes: {all_keyframes}")
    print(f"Demo length: {len(demo)} frames")

    # Visualize all keyframes
    print("\n=== Visualizing all keyframes ===")
    visualize_keyframes(demo, all_keyframes, camera='front')

    keyframes_result = get_pick_place_keyframes(demo, task_name=task_name)
    print(f"Pick frame: {keyframes_result['pick']}, Place frame: {keyframes_result['place']}")
