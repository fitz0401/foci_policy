"""
Adapted from: https://github.com/fitz0401/tasc/blob/main/tasc/vision_module/mask_generator.py
"""
import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import groundingdino.datasets.transforms as T
from segment_anything import build_sam, SamPredictor
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, predict, annotate
from torchvision.ops import nms
from PIL import Image
from pathlib import Path

file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add XMem2 to the import path. Keep it outside this repository by default so
# that it cannot be mistaken for vendored FOCI source.
PROJECT_ROOT = os.path.abspath(os.path.join(file_path, '../..'))
XMEM_PATH = os.environ.get(
    'XMEM2_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'XMem2')))
if os.path.exists(XMEM_PATH) and XMEM_PATH not in sys.path:
    sys.path.insert(0, XMEM_PATH)

class MaskGenerator:
    def __init__(self):
        self.ckpt_path = os.environ.get(
            'FOCI_GROUNDED_SAM_CKPT_DIR', os.path.join(PROJECT_ROOT, 'ckpts'))
        self.config_path = os.path.join(file_path, "../configs")
        self.device = device
        self.dino, self.sam = self._prepare_dino_and_sam()

    def _prepare_dino_and_sam(self):
        DINO_WEIGHTS_PATH = os.path.join(self.ckpt_path, "groundingdino_swint_ogc.pth")
        DINO_CONFIG_PATH = os.path.join(self.config_path, "grounding_dino.py")
        dino = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)
        SAM_WEIGHTS_PATH = os.path.join(self.ckpt_path, "sam_vit_h_4b8939.pth")
        sam = build_sam(checkpoint=SAM_WEIGHTS_PATH)
        sam.to(self.device)
        return dino, sam
    
    def _load_pil_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_np = np.asarray(image_pil)
        image_transformed, _ = transform(image_pil, None)
        self.H, self.W, _ = image_np.shape
        return image_np, image_transformed

    @staticmethod
    def _show_mask(masks, frame, random_color=True):
        annotated_frame_pil = Image.fromarray(frame).convert("RGBA")
        for i in range(masks.shape[0]):
            mask = masks[i][0]
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
            annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
        return np.array(annotated_frame_pil)
    
    def get_scene_object_bboxes(self, image, texts, visualize=False, logdir=None):
        # mask out gripper region
        image_np, image_transformed = self._load_pil_image(image)
        # get bounding box
        BOX_THRESHOLD = 0.3
        TEXT_THRESHOLD = 0.22 # can be tuned for different tasks
        all_boxes, all_logits, all_phrases = [], [], []
        missing_objects = []
        for text in texts:
            boxes, logits, phrases = predict(
                model=self.dino,
                image=image_transformed,
                caption=text,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            if logits is None or logits.numel() == 0:
                missing_objects.append(text)
                continue
            best_id = int(torch.argmax(logits).item())
            all_boxes.append(boxes[best_id])
            all_logits.append(logits[best_id])
            all_phrases.append(phrases[best_id])
        if len(all_boxes) == 0:
            raise RuntimeError(f"Failed to detect objects: {missing_objects}. Check object names or image quality.")

        boxes, logits, phrases = torch.stack(all_boxes, dim=0), torch.stack(all_logits, dim=0), all_phrases

        if visualize:
            annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
            # %matplotlib inline
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            annotated_frame_pil = Image.fromarray(annotated_frame)
            if logdir is not None:
                annotated_frame_pil.save(os.path.join(logdir, 'bbox.png'))
            plt.imshow(annotated_frame_pil)
            plt.show()
        return boxes, logits, phrases

    def filter_masks(self, boxes, scores, phrases, iou_threshold=0.5):
        # filter out masks by IOU
        device = boxes.device
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([self.W, self.H, self.W, self.H], device=device)
        keep_indices = nms(boxes_xyxy, scores, iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        phrases = [phrases[i] for i in keep_indices.tolist()]
        return boxes, scores, phrases

    def get_segmentation_masks(self, image, boxes, logits, phrases, visualize=False, save_path=None):
        image_np, _ = self._load_pil_image(image)
        if boxes is None or boxes.shape[0] == 0:
            raise RuntimeError(f"Cannot segment masks: no valid boxes. Objects: {phrases}")
        device = boxes.device
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([self.W, self.H, self.W, self.H], device=device)
        # get segmentation based on bounding box
        sam_predictor = SamPredictor(self.sam)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(self.device)
        sam_predictor.set_image(image_np)
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            masks = masks.cpu().numpy()
        if visualize:
            annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
            # %matplotlib inline
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            mask_annotated_img = self._show_mask(masks, annotated_frame)
            mask_annotated_img_pil = Image.fromarray(mask_annotated_img)
            plt.imshow(mask_annotated_img_pil)
            plt.show()
            if save_path is not None:
                mask_annotated_img_pil.save(save_path)
        return masks[:, 0]
    
    def track_masks_with_xmem(self, demo_dir, object_name, initial_mask, verbose=False):
        """ Run XMem++ tracking on a demo directory. """
        try:
            from inference.run_on_video import run_on_video
        except ImportError:
            if verbose:
                print("Warning: XMem++ not available, falling back to mask reuse")
            return {'masks': {}, 'success': False}
        
        demo_path = Path(demo_dir)
        imgs_path = str(demo_path / 'color')
        masks_path = str(demo_path / 'mask' / object_name)
        output_path = str(demo_path / f'xmem_output_{object_name}')
        
        # Create mask directory and save initial mask
        mask_dir = Path(masks_path)
        mask_dir.mkdir(exist_ok=True)
        
        # Save initial mask in XMem++ format (0000.png with object ID)
        initial_mask_path = mask_dir / '0000.png'
        # XMem uses object IDs (1, 2, 3...), 0 is background
        mask_with_id = (initial_mask > 0).astype(np.uint8)  # Convert to 1 for object
        cv2.imwrite(str(initial_mask_path), mask_with_id)
        
        if verbose:
            print(f"Running XMem++ tracking for {object_name}...")
            print(f"  Input images: {imgs_path}")
            print(f"  Initial mask: {initial_mask_path}")
            print(f"  Output: {output_path}")
        
        # Run XMem++ tracking with first frame annotation
        frames_with_masks = [0]  # Only annotate first frame
        
        # Change to XMem directory for inference
        original_dir = os.getcwd()
        try:
            os.chdir(XMEM_PATH)
            run_on_video(imgs_path, masks_path, output_path, frames_with_masks)
            os.chdir(original_dir)
        except Exception as e:
            os.chdir(original_dir)
            print(f"XMem++ tracking failed: {e}")
            return {'masks': {}, 'success': False}
        
        # Load tracked masks
        tracked_masks = {}
        output_mask_dir = Path(output_path) / 'Annotations'
        
        if not output_mask_dir.exists():
            if verbose:
                print(f"Warning: XMem++ output not found at {output_mask_dir}")
            return {'masks': {}, 'success': False}
        
        # Read all tracked masks
        mask_files = sorted(output_mask_dir.glob('*.png'))
        for mask_file in mask_files:
            frame_idx = int(mask_file.stem)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            # Convert from object ID (1) back to binary mask (0/1)
            mask_binary = (mask > 0).astype(np.uint8)
            tracked_masks[frame_idx] = mask_binary
        
        if verbose:
            print(f"  Loaded {len(tracked_masks)} tracked masks")
        
        return {'masks': tracked_masks, 'success': len(tracked_masks) > 0}
