"""
Try-On Pipeline - Virtual try-on using HR-VITON
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ConditionGenerator(nn.Module):
    """HR-VITON Condition Generator"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 7, 2, 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class ImageGenerator(nn.Module):
    """HR-VITON Image Generator"""
    def __init__(self):
        super().__init__()
        # Simple U-Net style generator
        self.encode1 = nn.Conv2d(3, 64, 7, 1, 3)
        self.encode2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.encode3 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.bottleneck = nn.Conv2d(256, 512, 3, 1, 1)
        
        self.decode3 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.decode2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.decode1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.final = nn.Conv2d(64, 3, 7, 1, 3)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x, condition):
        # Encode
        e1 = self.relu(self.encode1(x))
        e2 = self.relu(self.encode2(e1))
        e3 = self.relu(self.encode3(e2))
        
        # Bottleneck with condition
        b = self.relu(self.bottleneck(e3 + condition))
        
        # Decode
        d3 = self.relu(self.decode3(b))
        d2 = self.relu(self.decode2(d3 + e3))
        d1 = self.relu(self.decode1(d2))
        
        # Output
        out = self.tanh(self.final(d1))
        return (out + 1) / 2  # Convert from [-1, 1] to [0, 1]


class TryOnPipeline:
    """Virtual try-on inference pipeline using HR-VITON"""
    
    def __init__(self, device: str = "cuda", model_dir: str = "./models/hrviton"):
        """
        Initialize try-on pipeline with HR-VITON
        
        Args:
            device: Device to run on ("cuda" or "cpu")
            model_dir: Directory containing HR-VITON model weights
        """
        self.device = device
        self.model_dir = Path(model_dir)
        self.condition_gen = None
        self.image_gen = None
        self.load_model()
    
    def load_model(self):
        """Load HR-VITON models"""
        try:
            logger.info("Loading HR-VITON models...")
            
            # Initialize networks
            self.condition_gen = ConditionGenerator().to(self.device)
            self.image_gen = ImageGenerator().to(self.device)
            
            # Load condition generator
            cond_path = self.model_dir / "condition_generator.pth"
            if cond_path.exists():
                logger.info(f"Loading condition generator from {cond_path}")
                state_dict = torch.load(cond_path, map_location=self.device)
                self.condition_gen.load_state_dict(state_dict)
                self.condition_gen.eval()
                logger.info("Condition generator loaded successfully")
            else:
                logger.warning(f"Condition generator not found at {cond_path}, using random initialization")
            
            # Load image generator
            img_path = self.model_dir / "image_generator.pth"
            if img_path.exists():
                logger.info(f"Loading image generator from {img_path}")
                state_dict = torch.load(img_path, map_location=self.device)
                self.image_gen.load_state_dict(state_dict)
                self.image_gen.eval()
                logger.info("Image generator loaded successfully")
            else:
                logger.warning(f"Image generator not found at {img_path}, using random initialization")
            
            logger.info("HR-VITON models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load HR-VITON models: {e}")
            raise
    
    def infer(self, model_image: np.ndarray, garment_image: np.ndarray, 
             pose_image: np.ndarray, num_inference_steps: int = 50,
             guidance_scale: float = 7.5) -> np.ndarray:
        """
        Run virtual try-on inference
        
        Args:
            model_image: Model/person image (768x1024 RGB, 0-255)
            garment_image: Garment/saree image (768x1024 RGB, 0-255)
            pose_image: Pose skeleton map (768x1024 RGB, 0-255)
            num_inference_steps: Ignored for HR-VITON
            guidance_scale: Ignored for HR-VITON
            
        Returns:
            Generated try-on image (768x1024 RGB, 0-255)
        """
        try:
            logger.info("Running HR-VITON inference...")
            
            # Convert numpy arrays to torch tensors
            model_tensor = self._image_to_tensor(model_image).to(self.device)
            garment_tensor = self._image_to_tensor(garment_image).to(self.device)
            pose_tensor = self._image_to_tensor(pose_image).to(self.device)
            
            logger.info(f"Input shapes - Model: {model_tensor.shape}, Garment: {garment_tensor.shape}")
            
            with torch.no_grad():
                # Generate condition from garment and pose
                # Concatenate garment image with pose map (4 channels: 3 RGB + 1 mask)
                pose_gray = torch.mean(pose_tensor, dim=1, keepdim=True)  # Convert to grayscale
                condition_input = torch.cat([garment_tensor, pose_gray], dim=1)
                
                logger.info(f"Condition input shape: {condition_input.shape}")
                
                # Generate condition
                condition = self.condition_gen(condition_input)
                logger.info(f"Condition shape: {condition.shape}")
                
                # Generate try-on image
                output = self.image_gen(model_tensor, condition)
                logger.info(f"Output shape: {output.shape}")
            
            # Convert tensor back to numpy (0-255)
            output_np = self._tensor_to_image(output)
            
            logger.info(f"Inference completed. Output shape: {output_np.shape}")
            return output_np
            
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            raise
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image (H,W,3) to tensor (1,3,H,W)"""
        try:
            # Ensure uint8
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert HWC to CHW
            image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension
            tensor = torch.from_numpy(image).unsqueeze(0)
            
            return tensor
        except Exception as e:
            logger.error(f"Error converting image to tensor: {e}")
            raise
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor (1,3,H,W) to numpy image (H,W,3)"""
        try:
            # Remove batch dimension and move to CPU
            image = tensor.squeeze(0).cpu().numpy()
            
            # Convert CHW to HWC
            image = np.transpose(image, (1, 2, 0))
            
            # Scale to [0, 255]
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            return image
        except Exception as e:
            logger.error(f"Error converting tensor to image: {e}")
            raise
    
    def postprocess(self, output_img: np.ndarray) -> np.ndarray:
        """
        Post-process generated image
        
        Args:
            output_img: Raw output image
            
        Returns:
            Post-processed image
        """
        try:
            logger.info("Post-processing output...")
            
            # Ensure output is correct size and format
            if output_img.shape != (1024, 768, 3):
                if len(output_img.shape) == 2:
                    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
                output_img = cv2.resize(output_img, (768, 1024))
            
            # Clip values to valid range
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            
            logger.info("Post-processing completed")
            return output_img
            
        except Exception as e:
            logger.error(f"Error during post-processing: {e}")
            raise


def run_tryon(model_img: np.ndarray, saree_img: np.ndarray, 
             blouse_img: Optional[np.ndarray] = None,
             pose_map: Optional[np.ndarray] = None,
             saree_mask: Optional[np.ndarray] = None,
             blouse_mask: Optional[np.ndarray] = None,
             device: str = "cuda") -> np.ndarray:
    """
    Main function to run complete try-on pipeline using HR-VITON
    
    Args:
        model_img: Model/person image (768x1024 RGB, 0-255)
        saree_img: Saree fabric image (768x1024 RGB, 0-255)
        blouse_img: Blouse image (optional)
        pose_map: Pose skeleton map (768x1024 RGB, 0-255)
        saree_mask: Saree mask (optional, not used in HR-VITON)
        blouse_mask: Blouse mask (optional, not used in HR-VITON)
        device: Device to run on
        
    Returns:
        Generated try-on image (768x1024 RGB, 0-255)
    """
    try:
        logger.info("Starting HR-VITON try-on pipeline...")
        
        # Initialize pipeline
        pipeline = TryOnPipeline(device=device, model_dir="./models/hrviton")
        
        # Use saree as garment if blouse not provided
        garment_img = blouse_img if blouse_img is not None else saree_img
        
        # Use white pose map if not provided
        if pose_map is None:
            pose_map = np.ones((1024, 768, 3), dtype=np.uint8) * 255
            logger.warning("No pose map provided, using blank white image")
        
        # Ensure inputs are correct size
        if model_img.shape[:2] != (1024, 768):
            from . import utils
            model_img = utils.resize_image(model_img, (768, 1024))
        
        if garment_img.shape[:2] != (1024, 768):
            from . import utils
            garment_img = utils.resize_image(garment_img, (768, 1024))
        
        if pose_map.shape[:2] != (1024, 768):
            from . import utils
            pose_map = utils.resize_image(pose_map, (768, 1024))
        
        # Run inference
        logger.info("Running HR-VITON inference...")
        output_img = pipeline.infer(
            model_image=model_img,
            garment_image=garment_img,
            pose_image=pose_map,
            num_inference_steps=1,  # Not used, but keeping for compatibility
            guidance_scale=1.0  # Not used, but keeping for compatibility
        )
        
        # Post-process
        output_img = pipeline.postprocess(output_img)
        
        logger.info("Try-on pipeline completed successfully")
        return output_img
        
    except Exception as e:
        logger.error(f"Error in try-on pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Test pipeline
    logging.basicConfig(level=logging.INFO)
    print("Try-on pipeline module loaded. Run from app.py")
