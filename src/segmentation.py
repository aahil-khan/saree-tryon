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
        """Load SAM 2 model and initialize predictor"""
        try:
            logger.info("Loading SAM 2 model...")
            # TODO: Implement SAM 2 model loading
            # from segment_anything_2 import build_sam2
            # self.model = build_sam2(checkpoint=self.checkpoint_path, device=self.device)
            # self.predictor = SamPredictor2(self.model)
            logger.info("SAM 2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM 2 model: {e}")
            raise
    
    def segment_garment(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment garment from flat fabric image
        
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # TODO: Run SAM 2 inference
            # masks = self.predictor.predict(image)
            
            # Placeholder: return dummy mask
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            
            logger.info("Segmentation completed successfully")
            return image, mask
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
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
