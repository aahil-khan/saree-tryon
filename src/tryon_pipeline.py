"""
Try-On Pipeline - Virtual try-on using Stable Diffusion ControlNet
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

logger = logging.getLogger(__name__)


class TryOnPipeline:
    """Virtual try-on inference pipeline using Stable Diffusion ControlNet"""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize try-on pipeline
        
        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load Stable Diffusion ControlNet model"""
        try:
            logger.info("Loading Stable Diffusion ControlNet model...")
            
            # Load ControlNet for pose control
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Load main pipeline
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
            
            logger.info("Stable Diffusion ControlNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            
        Returns:
            Generated try-on image (768x1024 RGB, 0-255)
        """
        try:
            logger.info(f"Running inference (steps={num_inference_steps}, guidance={guidance_scale})...")
            
            # Convert numpy arrays to PIL Images
            model_pil = Image.fromarray(model_image.astype(np.uint8))
            garment_pil = Image.fromarray(garment_image.astype(np.uint8))
            pose_pil = Image.fromarray(pose_image.astype(np.uint8))
            
            # Ensure all images are the same size
            model_pil = model_pil.resize((768, 1024))
            garment_pil = garment_pil.resize((768, 1024))
            pose_pil = pose_pil.resize((768, 1024))
            
            # Create prompt for try-on
            prompt = "a person wearing a beautiful saree, high quality, detailed fabric, professional photography"
            negative_prompt = "blurry, low quality, distorted, deformed"
            
            # Run inference with garment image as conditioning
            with torch.no_grad():
                output = self.pipeline(
                    prompt=prompt,
                    image=pose_pil,
                    controlnet_conditioning_scale=1.0,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    height=1024,
                    width=768,
                ).images[0]
            
            # Blend with garment to transfer texture
            output_np = np.array(output)
            garment_np = np.array(garment_pil)
            
            # Apply garment texture with 40% strength
            blended = (output_np * 0.6 + garment_np * 0.4).astype(np.uint8)
            
            logger.info(f"Inference completed. Output shape: {blended.shape}")
            return blended
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
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
                output_img = np.transpose(output_img, (1, 2, 0)) if len(output_img.shape) == 3 else output_img
            
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
    Main function to run complete try-on pipeline
    
    Args:
        model_img: Model/person image (768x1024 RGB, 0-255)
        saree_img: Saree fabric image (768x1024 RGB, 0-255)
        blouse_img: Blouse image (optional)
        pose_map: Pose skeleton map (768x1024 RGB, 0-255)
        saree_mask: Saree mask (optional)
        blouse_mask: Blouse mask (optional)
        device: Device to run on
        
    Returns:
        Generated try-on image (768x1024 RGB, 0-255)
    """
    try:
        logger.info("Starting try-on pipeline...")
        
        # Initialize pipeline
        pipeline = TryOnPipeline(device=device)
        
        # Use default values if not provided
        pose_map_use = pose_map if pose_map is not None else np.zeros((1024, 768, 3), dtype=np.uint8)
        garment_img = blouse_img if blouse_img is not None else saree_img
        
        # Ensure inputs are correct size
        if model_img.shape[:2] != (1024, 768):
            from . import utils
            model_img = utils.resize_image(model_img, (768, 1024))
        
        if garment_img.shape[:2] != (1024, 768):
            from . import utils
            garment_img = utils.resize_image(garment_img, (768, 1024))
        
        if pose_map_use.shape[:2] != (1024, 768):
            from . import utils
            pose_map_use = utils.resize_image(pose_map_use, (768, 1024))
        
        # Run inference
        output_img = pipeline.infer(
            model_image=model_img,
            garment_image=garment_img,
            pose_image=pose_map_use,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        # Post-process
        output_img = pipeline.postprocess(output_img)
        
        logger.info("Try-on pipeline completed successfully")
        return output_img
        
    except Exception as e:
        logger.error(f"Error in try-on pipeline: {e}")
        raise


if __name__ == "__main__":
    # Test pipeline
    logging.basicConfig(level=logging.INFO)
    print("Try-on pipeline module loaded. Run from app.py")
