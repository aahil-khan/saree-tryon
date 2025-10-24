"""
Utility Functions - Image processing and helper functions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (RGB)
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Loaded image: {image_path}, shape={image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise


def save_image(image: np.ndarray, output_path: Union[str, Path]):
    """
    Save image to file
    
    Args:
        image: Image as numpy array (RGB)
        output_path: Path to save image
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)
        logger.info(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size (width, height)
    
    Args:
        image: Input image
        target_size: Target size as (width, height)
        
    Returns:
        Resized image
    """
    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"Resized image to {target_size}")
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image (0-255 range)
        
    Returns:
        Normalized image (0-1 range)
    """
    try:
        normalized = image.astype(np.float32) / 255.0
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing image: {e}")
        raise


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image to [0, 255] range
    
    Args:
        image: Input image (0-1 range)
        
    Returns:
        Denormalized image (0-255 range)
    """
    try:
        denormalized = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return denormalized
    except Exception as e:
        logger.error(f"Error denormalizing image: {e}")
        raise


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply binary mask to image
    
    Args:
        image: Input image
        mask: Binary mask (same size as image)
        
    Returns:
        Masked image
    """
    try:
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        
        masked = image * (mask > 127).astype(np.uint8)
        return masked
    except Exception as e:
        logger.error(f"Error applying mask: {e}")
        raise


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blend factor (0-1)
        
    Returns:
        Blended image
    """
    try:
        blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        return blended
    except Exception as e:
        logger.error(f"Error blending images: {e}")
        raise


def create_output_directory(base_dir: Union[str, Path] = "./outputs") -> Path:
    """
    Create output directory structure
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Path to base output directory
    """
    try:
        output_dir = Path(base_dir)
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (output_dir / "poses").mkdir(parents=True, exist_ok=True)
        (output_dir / "results").mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directories at {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        raise


def setup_logging(log_level: str = "INFO", log_file: str = "tryon.log"):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Path to log file
    """
    try:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(f"logs/{log_file}"),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging setup complete (level={log_level})")
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """
    Validate if image path exists and is readable
    
    Args:
        image_path: Path to image
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image path does not exist: {image_path}")
            return False
        if not path.is_file():
            logger.warning(f"Path is not a file: {image_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating image path: {e}")
        return False


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about image
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    try:
        info = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min": image.min(),
            "max": image.max(),
            "mean": image.mean(),
        }
        return info
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        raise


if __name__ == "__main__":
    # Test utilities
    logging.basicConfig(level=logging.INFO)
    print("Utilities module loaded. Run from app.py")
