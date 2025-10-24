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
        """Load pose detector using MediaPipe"""
        try:
            logger.info("Loading pose detector (MediaPipe)...")
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe pose detector loaded successfully")
                self.detector_type = "mediapipe"
            except ImportError:
                logger.warning("MediaPipe not installed, using fallback")
                self.detector_type = "fallback"
                self.detector = None
        except Exception as e:
            logger.error(f"Failed to load pose detector: {e}")
            self.detector_type = "fallback"
            self.detector = None
    
    def extract_pose(self, image_path: str, target_size: Tuple[int, int] = (768, 1024)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pose keypoints from model image
        
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
            
            # Create pose map (white background with pose skeleton)
            pose_map = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
            
            # Try to extract pose if detector is available
            if self.detector_type == "mediapipe" and self.detector is not None:
                try:
                    results = self.detector.process(image_resized)
                    if results.pose_landmarks:
                        # Draw pose landmarks on the map
                        self._draw_pose_landmarks(pose_map, results.pose_landmarks, target_size)
                except Exception as e:
                    logger.warning(f"Failed to extract pose with MediaPipe: {e}. Using blank pose map.")
            
            logger.info("Pose extraction completed successfully")
            return image_resized, pose_map
            
        except Exception as e:
            logger.error(f"Error during pose extraction: {e}")
            raise
    
    def _draw_pose_landmarks(self, image: np.ndarray, landmarks, target_size: Tuple[int, int]):
        """Draw pose landmarks on image"""
        try:
            # Draw skeleton connections and joints
            h, w = image.shape[:2]
            
            # landmarks is a NormalizedLandmarkList object
            # Access landmarks via .landmark attribute
            if hasattr(landmarks, 'landmark'):
                landmark_list = landmarks.landmark
            else:
                landmark_list = landmarks
            
            for landmark in landmark_list:
                if hasattr(landmark, 'visibility') and landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # Draw joint as circle
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        except Exception as e:
            logger.warning(f"Error drawing landmarks: {e}")
    
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
