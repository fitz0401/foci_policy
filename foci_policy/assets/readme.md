# Guidelines for using FOCI Policy on a new RLBench task

In this page, we provide instructions for using FOCI Policy on a new RLBench task. Taking `phone_on_base` as an example, the steps are as follows:

## 0. Configure the task

1. Add  task-revelant objects to the `task_object_dict`, this is used to obtain the object GT pose and mask during data collection in Step 1. The object names should match the names in the RLBench TTM file.
```bash
# file path: foci_policy/rlbench_env/utils/rlbench_objects.py
"phone_on_base": {
    "grasp_object_name": "phone",
    "target_object_name": "phone_case",
},
```

2. Add the task configuration to `task_obj_config.yaml`, this is used to determine the grasping and target objects, as well as the language instructions for the task.
```bash
# file path: foci_policy/config/task_obj_config.yaml
phone_on_base:
  grasp: phone
  target: phone_case
  lan_grasp: 'pick up the phone'
  lan_manip: 'place the phone on the base'
  mesh_names: [phone, phone_case]
```

3. Add the task to `dataloader.yaml`, this is used to load the dataset for training and evaluation. Also, add the task class to `config_utils.py`.
```bash
# file path: foci_policy/config/dataloader.yaml
task_list
  - phone_on_base

# file path: foci_policy/config/config_utils.py
TASK_CLASSES = {
    'phone_on_base': tasks.PhoneOnBase,
}
```

## 1. Prepare demonstrations
```bash
cd foci_policy/utils
python dataset_generator_per_var.py \
  --tasks phone_on_base --episodes_per_task=30 --variation=0 --processes=1
```

## 2. Prepare object meshes

For RLBench-18 tasks, you can download the RLBench object meshes from [Google Drive](https://drive.google.com/drive/folders/1kMlNjqbwJV7xUjhFtlR1oQS3gofbdF7N?usp=sharing), 

There are two ways to collect object meshes for new tasks:

- **Option 1: From CoppeliaSim (recommended if you have interactive display)**

`File` -> `Load Model` -> Select from `foci_policy/RLBench/rlbench/task_ttms/phone_on_base.ttm` -> Select `phone` / `phone_case` on left panel -> `File` -> `Export Selected shapes` (Up-vector: Z).

- **Option 2: From RLBench TTM**
```bash
# step 2.1: extract mesh from TTM
cd utils_rlbench
python extract_mesh_from_ttm.py --task phone_on_base --export
# step 2.2 (optional): combine mesh of multiple parts into a single mesh
python merge_mesh_parts.py --task phone_on_base --parts part_1 part_2 --output meshes/phone_on_base/xx.obj
# step 2.3 (optional): center mesh
cd ../foci_policy/utils
python mesh_validator.py --task phone_on_base --fix
```

Tips:

- Place `.obj` files under `foci_policy/assets/RLBench_mesh/phone_on_base/`.
- Ensure z-axis is up for the mesh. This is for better `FoundationPose` performance. You can use `MeshLab` to visualize and rotate the mesh if needed.
- The mesh names should match the names in `task_obj_config.yaml` in Step 0.

## 3. Check `FoundationPose`
You can have a quick check of the FoundationPose results on the collected demos.

```bash
cd foci_policy/scripts
python visualize_fp_poses.py --task_name phone_on_base --episode_idx 0 --postprocess
```

In practice, it fails on the following types of objects: small objects, regularly symmetrical objects, and partially visible objects.

## 4. Check the interaction intervals
```bash
cd foci_policy/utils
python compute_interaction_intervals.py --task phone_on_base --num_demos 1 --plot --viz3d
```

## 5. Preprocess demos into FOCI dataset
```bash
cd foci_policy/utils
python preprocess_raw_rlbench_demo.py --task_name phone_on_base --num_demos 5 --pose_method fp
## Tips: Use `--pose_method gt` to use GT poses instead of FoundationPose.
```

## 6. Check the FOCI dataset
```bash
cd foci_policy/utils
python visualize_dataset.py --task_name phone_on_base
```

## 7. Train and evaluate
```bash
cd foci_policy/scripts
python train_foci.py
python test_foci_simulator.py --disp --task phone_on_base --pose_method fp --debug --actor foci
```