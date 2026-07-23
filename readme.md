# FOCI Policy

[Paper](https://arxiv.org/abs/2609.08743) | [Website](https://fitz0401.github.io/foci-page/)

A one-shot object-centric learning framework for relational manipulation tasks.

<img src="assets/foci-highlight.gif" width="360" alt="FOCI highlight gif">

## 🛠️ Installation

**Step 1.** Create the `foci` environment

```bash
# Create a conda environment with Python 3.8
conda create -n foci python=3.8
conda activate foci
conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge eigen=3.4.0 boost=1.85.0 cmake ninja

# Clone this repo
cd foci_policy
pip install -r requirements.txt
pip install -e .
```

**Step 2.** Install CoppeliaSim, PyRep and RLBench

```bash
# Install CoppeliaSim V4.1.0
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
echo "export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> $HOME/.bashrc
echo 'export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"' >> $HOME/.bashrc
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> $HOME/.bashrc
source $HOME/.bashrc

# Install PyRep
conda activate foci
git clone https://github.com/stepjam/PyRep.git
cd PyRep && pip install -e . && cd ..

# Install RLBench
git clone https://github.com/MohitShridhar/RLBench.git
cd RLBench
git checkout -b foci --track origin/peract
# RLBench modifications (collision detection, demo I/O, etc.)
git apply ../patches/rlbench_peract_foci.patch
pip install -r requirements.txt
pip install -e .
cd ..
```

**Step 3.** Install `FoundationPose`

> Skip this step if you only want to run GT pose experiments.

```bash
# Change this to your CUDA 11.8 installation path!
# You can check your CUDA version with `nvcc --version`, or `ls -l /usr/local/cuda*` to see which version is installed.
export CUDA_HOME=/absolute/path/to/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
nvcc --version      # must report CUDA 11.8
"$CC" --version    # must report GCC 11 or older
pip install -r fp_requirements.txt
pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
# It takes a while to install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Build FoundationPose
cd foundation_pose
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
cd ..
```

Download the `FoundationPose` model weights from [Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) or the [FoundationPose repo](https://github.com/NVlabs/FoundationPose?tab=readme-ov-file). Place them under `data/model_weight/foundation_pose`.

## 📦 Data collection and preprocessing
**Step 1.** Prepare RLBench object meshes.

Download the RLBench object meshes from [Google Drive](https://drive.google.com/drive/folders/14RXpAN69jdVpdQYOX8jXgc06Gmmrs4ns?usp=drive_link), and place them under `foci_policy/assets/RLBench_mesh`. See this [page](foci_policy/assets/readme.md) for collecting meshes yourself.

**Step 2.** Collect and preprocess RLBench demos.

```bash
cd foci_policy/utils
# Collect RLBench demos (5 for training, 25 for testing)
python dataset_generator_per_var.py \
  --episodes_per_task=30 --variation=0 --processes=1

# Preprocess raw demos into the FOCI dataset
python preprocess_raw_rlbench_demo.py \
  --task_name all --num_demos 5 --pose_method fp
# Tips: Use `--pose_method gt` to use GT poses instead of FoundationPose.
```

Alternatively, download the preprocessed FOCI dataset from [Google Drive](https://drive.google.com/drive/folders/10JWRaD0zouXlXWtlNpGoCOSLok8h_8iD?usp=sharing). Evaluation results may vary slightly with different dataset versions.

## 🚀 Train and evaluate

Run training and evaluation from `foci_policy/scripts` because the scripts resolve config and dataset paths relative to that directory.

```bash
cd foci_policy/scripts
# Train the FOCI policy
python train_foci.py

# Evaluate the FOCI policy for a single task
python test_simulator.py \
  --task close_jar --n_tests 5 --pose_method fp
# Tips: Use `--disp` to launch an interactive simulator, `--debug` to visualize failures, and `--record` to save evaluation videos, `--pose_method gt` to use GT poses instead of FoundationPose.

# Run batch tests for all tasks
python batch_test_all_tasks.py
```

> A dedicated GPU with at least 16 GB VRAM is the practical minimum; 24 GB or more is recommended for running FoundationPose.

See this [page](foci_policy/assets/readme.md) if you want to visualize the FOCI dataset or try FOCI Policy on other RLBench tasks.

## 🦾 Real-world setup

For real-world experiments, see the separate repository:
https://github.com/fitz0401/foci_ros2_ws

The simulator pipeline does not require GroundingDINO, Segment Anything, or XMem2.
Install those only for `foci_real_world`:

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/IDEA-Research/GroundingDINO.git@126abe"
python -m pip install \
  "git+https://github.com/facebookresearch/segment-anything.git"
git clone https://github.com/mbzuai-metaverse/XMem2.git ../XMem2
python -m pip install -r ../XMem2/requirements.txt
(cd ../XMem2 && ./scripts/download_models.sh)

mkdir -p ckpts
wget -P ckpts \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P ckpts \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

export XMEM2_ROOT="$(cd ../XMem2 && pwd)"
export FOCI_GROUNDED_SAM_CKPT_DIR="$(pwd)/ckpts"
```

## 🙏 Acknowledgements
We gratefully acknowledge the following open-source projects that contributed to this work:

- Data collection and preprocessing pipeline from [Imagination Policy](https://github.com/HaojHuang/imagination-policy-cor24)
- Object-centric pose estimation from [Object-Centric Diffusion](https://github.com/NVlabs/object_centric_diffusion)

## 📄 Citing
If you use this work, please cite:

```bibtex
@misc{fu2026focipolicy,
      title={FOCI Policy: Focus on Object-Centric Interactions for Relational Manipulation Policies}, 
      author={Ze Fu and Pinhao Song and Yutong Hu and Renaud Detry},
      year={2026},
      eprint={2609.08743},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2609.08743}, 
}
```
