"""
Pose Extraction Module - Extract body pose using ControlNet OpenPose
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extract body pose keypoints from model photo using OpenPose"""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize OpenPose detector
        
        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.detector = None
        self.load_detector()
    
    def load_detector(self):
        """Load OpenPose detector via ControlNet"""
        try:
            logger.info("Loading OpenPose detector...")
            from controlnet_aux import OpenposeDetector
            
            # Load OpenPose detector (downloads model on first use)
            self.detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet-init",
                filename="body_pose_model.pth"
            )
            
            logger.info("OpenPose detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OpenPose detector: {e}")
            raise
    
    def extract_pose(self, image_path: str, target_size: Tuple[int, int] = (768, 1024)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pose keypoints from model image using OpenPose
        
        Args:
            image_path: Path to model image
            target_size: Target output size (width, height)
            
        Returns:
            Tuple of (original_image, pose_map)
        """
        try:
            logger.info(f"Extracting pose from {image_path}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image_resized = cv2.resize(image_rgb, target_size)
            
            # Run OpenPose detection to get pose map
            pose_map = self.detector(image_resized)
            
            # Ensure pose_map is correct format (numpy array with shape [H, W, 3])
            if isinstance(pose_map, np.ndarray):
                if pose_map.dtype != np.uint8:
                    pose_map = (pose_map * 255).astype(np.uint8)
                if len(pose_map.shape) == 2:
                    # Convert grayscale to RGB
                    pose_map = cv2.cvtColor(pose_map, cv2.COLOR_GRAY2RGB)
            else:
                # If detection returns PIL Image, convert to numpy
                import PIL
                pose_map = np.array(pose_map)
            
            logger.info("Pose extraction completed successfully")
            return image_resized, pose_map
            
        except Exception as e:
            logger.error(f"Error during pose extraction: {e}")
            raise
    
    def save_pose_map(self, pose_map: np.ndarray, output_path: str):
        """Save pose map as image"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pose_map_bgr = cv2.cvtColor(pose_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, pose_map_bgr)
            logger.info(f"Pose map saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save pose map: {e}")
            raise
    
    def extract_keypoints(self, image_path: str) -> dict:
        """
        Extract body keypoints
        
        Args:
            image_path: Path to model image
            
        Returns:
            Dictionary with keypoint information
        """
        try:
            logger.info(f"Extracting keypoints from {image_path}")
            
            # TODO: Implement keypoint extraction
            # Placeholder keypoints (example format)
            keypoints = {
                "shoulders": [(0, 0), (0, 0)],
                "elbows": [(0, 0), (0, 0)],
                "wrists": [(0, 0), (0, 0)],
                "hips": [(0, 0), (0, 0)],
                "knees": [(0, 0), (0, 0)],
                "ankles": [(0, 0), (0, 0)],
            }
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            raise


def extract_pose_from_model(image_path: str, target_size: Tuple[int, int] = (768, 1024),
                           device: str = "cuda") -> dict:
    """
    Main function to extract pose from model image
    
    Args:
        image_path: Path to model image
        target_size: Target output size
        device: Device to run on
        
    Returns:
        Dictionary with pose information
    """
    extractor = PoseExtractor(device)
    
    # Extract pose
    model_img, pose_map = extractor.extract_pose(image_path, target_size)
    
    # Extract keypoints
    keypoints = extractor.extract_keypoints(image_path)
    
    results = {
        "model_image": model_img,
        "pose_map": pose_map,
        "keypoints": keypoints,
    }
    
    return results


if __name__ == "__main__":
    # Test pose extraction
    logging.basicConfig(level=logging.INFO)
    print("Pose extraction module loaded. Run from app.py")
