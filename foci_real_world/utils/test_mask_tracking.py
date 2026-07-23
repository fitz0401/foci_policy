import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(file_path, '../..'))
xmem_path = os.environ.get(
    'XMEM2_ROOT', os.path.abspath(os.path.join(project_root, '..', 'XMem2')))
sys.path.insert(0, xmem_path)

from inference.run_on_video import run_on_video
imgs_path = os.path.join(file_path, '../dataset/demo_001/color')
masks_path = os.path.join(file_path, '../dataset/demo_001/mask')   # Should contain annotation masks for frames in `frames_with_masks`
output_path = os.path.join(file_path, 'output')
frames_with_masks = [0]  # indices of frames for which there is an annotation mask
os.chdir(xmem_path)
run_on_video(imgs_path, masks_path, output_path, frames_with_masks)
