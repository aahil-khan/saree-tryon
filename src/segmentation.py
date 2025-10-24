"""
Segmentation Module - SAM 2 wrapper for garment segmentation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GarmentSegmenter:
    """Segment saree and blouse from flat fabric images using SAM 2"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize SAM 2 segmentation model
        
        Args:
            checkpoint_path: Path to SAM 2 checkpoint
            device: Device to run on ("cuda" or "cpu")
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """Load SAM2 model from checkpoint"""
        try:
            logger.info("Loading SAM2 model...")
            try:
                # Try to import and use SAM2 if available
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                # Build SAM2 model
                self.model = build_sam2(
                    model_cfg="large",
                    ckpt_path=self.checkpoint_path,
                    device=self.device
                )
                
                # Initialize predictor
                self.predictor = SAM2ImagePredictor(self.model)
                self.model_type = "sam2"
                logger.info("SAM2 model loaded successfully")
            except ImportError:
                logger.warning("SAM2 not installed, trying SAM from controlnet_aux...")
                from controlnet_aux import SAMDetector
                
                self.predictor = SAMDetector.from_pretrained(
                    "facebook/sam-vit-large"
                )
                self.model_type = "sam"
                logger.info("SAM model loaded from controlnet_aux")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            self.predictor = None
            self.model_type = "none"
    
    def segment_garment(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment garment from fabric image using SAM2 or SAM
        
        Args:
            image_path: Path to garment image
            
        Returns:
            Tuple of (original_image, binary_mask)
        """
        try:
            logger.info(f"Segmenting garment from {image_path}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # If no model loaded, return full mask
            if self.predictor is None:
                logger.warning("No segmentation model available, using full image mask")
                mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255
                return image_rgb, mask
            
            # Use SAM2 or SAM to segment
            if self.model_type == "sam2":
                # SAM2 workflow
                self.predictor.set_image(image_rgb)
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    multimask_output=False,
                    return_logits=False
                )
                if len(masks) > 0:
                    mask = masks[0].astype(np.uint8) * 255
                else:
                    mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255
            else:
                # SAM (from controlnet_aux) workflow
                result = self.predictor(image_rgb)
                if isinstance(result, np.ndarray):
                    mask = result.astype(np.uint8) * 255
                elif hasattr(result, 'masks'):
                    masks = result.masks
                    if len(masks) > 0:
                        mask_areas = [np.sum(m) for m in masks]
                        largest_idx = np.argmax(mask_areas)
                        mask = masks[largest_idx].astype(np.uint8) * 255
                    else:
                        mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255
                else:
                    mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255
            
            logger.info("Segmentation completed successfully")
            return image_rgb, mask
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            # Fallback: return full mask
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255
                    return image_rgb, mask
            except:
                pass
            raise
    
    def save_mask(self, mask: np.ndarray, output_path: str):
        """Save mask as PNG file"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, mask)
            logger.info(f"Mask saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save mask: {e}")
            raise
    
    def get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from mask"""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return x_min, y_min, x_max, y_max


def segment_saree_and_blouse(saree_path: str, blouse_path: Optional[str] = None, 
                             checkpoint_path: str = "./models/sam2/sam2_hiera_large.pt",
                             device: str = "cuda") -> dict:
    """
    Main function to segment saree and optional blouse
    
    Args:
        saree_path: Path to saree image
        blouse_path: Path to blouse image (optional)
        checkpoint_path: Path to SAM 2 checkpoint
        device: Device to run on
        
    Returns:
        Dictionary with segmentation results
    """
    segmenter = GarmentSegmenter(checkpoint_path, device)
    
    # Segment saree
    saree_img, saree_mask = segmenter.segment_garment(saree_path)
    
    results = {
        "saree_image": saree_img,
        "saree_mask": saree_mask,
        "saree_bbox": segmenter.get_bounding_box(saree_mask),
    }
    
    # Segment blouse if provided
    if blouse_path:
        blouse_img, blouse_mask = segmenter.segment_garment(blouse_path)
        results.update({
            "blouse_image": blouse_img,
            "blouse_mask": blouse_mask,
            "blouse_bbox": segmenter.get_bounding_box(blouse_mask),
        })
    
    return results


if __name__ == "__main__":
    # Test segmentation
    logging.basicConfig(level=logging.INFO)
    print("Segmentation module loaded. Run from app.py")
