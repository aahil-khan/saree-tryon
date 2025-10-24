"""
Color Analysis Module - Extract dominant colors for blouse generation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ColorAnalyzer:
    """Extract colors from saree for blouse generation"""
    
    def __init__(self):
        """Initialize color analyzer"""
        pass
    
    def extract_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from image using K-means
        
        Args:
            image: Input image (RGB)
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors as (R, G, B) tuples
        """
        try:
            logger.info(f"Extracting {num_colors} dominant colors...")
            
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3).astype(np.float32)
            
            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers
            colors = centers.astype(int)
            
            logger.info(f"Extracted colors: {colors}")
            return [tuple(c) for c in colors]
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            raise
    
    def select_blouse_color(self, dominant_colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """
        Select appropriate blouse color from dominant saree colors
        
        Args:
            dominant_colors: List of dominant saree colors
            
        Returns:
            Selected blouse color as (R, G, B) tuple
        """
        try:
            logger.info("Selecting blouse color...")
            
            # For POC, use lightest color or create complementary color
            # Strategy: Pick the lightest color or average lighter version
            
            if not dominant_colors:
                logger.warning("No dominant colors provided, using default beige")
                return (220, 200, 180)  # Default beige
            
            # Calculate brightness for each color
            brightnesses = [sum(c) / 3 for c in dominant_colors]
            lightest_idx = np.argmax(brightnesses)
            lightest_color = dominant_colors[lightest_idx]
            
            # Make it slightly lighter and more saturated
            blouse_color = tuple(min(255, int(c * 1.1)) for c in lightest_color)
            
            logger.info(f"Selected blouse color: {blouse_color}")
            return blouse_color
            
        except Exception as e:
            logger.error(f"Error selecting blouse color: {e}")
            raise
    
    def generate_blouse_image(self, color: Tuple[int, int, int], 
                             size: Tuple[int, int] = (768, 1024)) -> np.ndarray:
        """
        Generate solid color blouse image
        
        Args:
            color: Blouse color as (R, G, B)
            size: Image size (height, width)
            
        Returns:
            Generated blouse image
        """
        try:
            logger.info(f"Generating blouse image with color {color}...")
            
            # Create solid color image
            blouse_img = np.full((size[0], size[1], 3), color, dtype=np.uint8)
            
            logger.info("Blouse image generated successfully")
            return blouse_img
            
        except Exception as e:
            logger.error(f"Error generating blouse image: {e}")
            raise
    
    def save_blouse(self, blouse_img: np.ndarray, output_path: str):
        """Save generated blouse image"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            blouse_bgr = cv2.cvtColor(blouse_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, blouse_bgr)
            logger.info(f"Blouse image saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save blouse image: {e}")
            raise


def generate_matching_blouse(saree_path: str, output_path: Optional[str] = None,
                            size: Tuple[int, int] = (768, 1024)) -> np.ndarray:
    """
    Main function to generate matching blouse from saree
    
    Args:
        saree_path: Path to saree image
        output_path: Path to save generated blouse (optional)
        size: Output size
        
    Returns:
        Generated blouse image
    """
    try:
        logger.info(f"Generating blouse for saree: {saree_path}")
        
        # Load saree image
        saree_img = cv2.imread(str(saree_path))
        if saree_img is None:
            raise ValueError(f"Failed to load saree image: {saree_path}")
        
        saree_img = cv2.cvtColor(saree_img, cv2.COLOR_BGR2RGB)
        
        # Resize to smaller size for color analysis
        saree_small = cv2.resize(saree_img, (256, 256))
        
        # Extract colors
        analyzer = ColorAnalyzer()
        dominant_colors = analyzer.extract_dominant_colors(saree_small, num_colors=5)
        
        # Select blouse color
        blouse_color = analyzer.select_blouse_color(dominant_colors)
        
        # Generate blouse image
        blouse_img = analyzer.generate_blouse_image(blouse_color, size)
        
        # Save if output path provided
        if output_path:
            analyzer.save_blouse(blouse_img, output_path)
        
        return blouse_img
        
    except Exception as e:
        logger.error(f"Error generating blouse: {e}")
        raise


if __name__ == "__main__":
    # Test color analysis
    logging.basicConfig(level=logging.INFO)
    print("Color analysis module loaded. Run from app.py")
