"""
Saree Virtual Try-On POC - Gradio UI
"""

import gradio as gr
import numpy as np
import logging
from pathlib import Path
import cv2
from PIL import Image

from src import segmentation, pose_extraction, color_analysis, tryon_pipeline, utils

# Setup logging
utils.setup_logging()
logger = logging.getLogger(__name__)

# Create output directories
utils.create_output_directory()


def process_tryon(saree_image, model_image, blouse_image=None):
    """
    Main processing function for try-on
    
    Args:
        saree_image: Saree fabric image
        model_image: Model photo
        blouse_image: Optional blouse image
        
    Returns:
        Generated try-on image
    """
    try:
        logger.info("Starting try-on processing...")
        
        # Step 1: Load images
        logger.info("Step 1/6: Loading images...")
        saree_img = utils.load_image(saree_image) if isinstance(saree_image, str) else saree_image
        model_img = utils.load_image(model_image) if isinstance(model_image, str) else model_image
        
        # Resize both images to target size (768x1024)
        saree_img = utils.resize_image(saree_img, (768, 1024))
        model_img = utils.resize_image(model_img, (768, 1024))
        logger.info(f"Images loaded and resized to 768x1024")
        
        # Step 2: Generate or load blouse
        logger.info("Step 2/6: Preparing blouse...")
        if blouse_image is not None:
            logger.info("Loading provided blouse image...")
            blouse_img = utils.load_image(blouse_image) if isinstance(blouse_image, str) else blouse_image
            blouse_img = utils.resize_image(blouse_img, (768, 1024))
        else:
            logger.info("Generating matching blouse from saree...")
            try:
                blouse_img = color_analysis.generate_matching_blouse(saree_image)
                blouse_img = utils.resize_image(blouse_img, (768, 1024))
                utils.save_image(blouse_img, "./outputs/generated_blouse.png")
            except Exception as e:
                logger.warning(f"Blouse generation failed: {e}. Using white blouse as fallback.")
                blouse_img = np.ones((1024, 768, 3), dtype=np.uint8) * 240
        
        # Step 3: Segment garments
        logger.info("Step 3/6: Segmenting garments...")
        try:
            segmenter = segmentation.GarmentSegmenter(
                checkpoint_path="./models/sam2/sam2_hiera_large.pt",
                device="cuda"
            )
            
            # Segment saree
            saree_rgb, saree_mask = segmenter.segment_garment(
                saree_image if isinstance(saree_image, str) else "./outputs/temp_saree.png"
            )
            
            # Segment blouse if provided
            if blouse_image is not None:
                blouse_rgb, blouse_mask = segmenter.segment_garment(
                    blouse_image if isinstance(blouse_image, str) else "./outputs/temp_blouse.png"
                )
            else:
                blouse_mask = np.ones((1024, 768), dtype=np.uint8) * 255
            
            logger.info("Segmentation completed successfully")
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}. Using full masks as fallback.")
            saree_mask = np.ones((1024, 768), dtype=np.uint8) * 255
            blouse_mask = np.ones((1024, 768), dtype=np.uint8) * 255
        
        # Step 4: Extract pose
        logger.info("Step 4/6: Extracting pose...")
        try:
            extractor = pose_extraction.PoseExtractor(device="cuda")
            model_resized, pose_map = extractor.extract_pose(
                model_image if isinstance(model_image, str) else "./outputs/temp_model.png",
                target_size=(768, 1024)
            )
            logger.info("Pose extraction completed successfully")
        except Exception as e:
            logger.warning(f"Pose extraction failed: {e}. Using blank pose map as fallback.")
            pose_map = np.ones((1024, 768, 3), dtype=np.uint8) * 255
        
        # Step 5: Run try-on pipeline
        logger.info("Step 5/6: Running try-on inference (this may take 30-60 seconds)...")
        try:
            pipeline = tryon_pipeline.TryOnPipeline(device="cuda")
            output_img = pipeline.infer(
                model_image=model_img,
                garment_image=saree_img,
                pose_image=pose_map,
                num_inference_steps=50,
                guidance_scale=7.5
            )
            logger.info("Try-on inference completed successfully")
        except Exception as e:
            logger.error(f"Try-on inference failed: {e}")
            raise
        
        # Step 6: Save and return output
        logger.info("Step 6/6: Saving output...")
        if output_img is not None and output_img.size > 0:
            # Ensure output is RGB numpy array
            if isinstance(output_img, Image.Image):
                output_img = np.array(output_img)
            
            output_path = "./outputs/results/tryon_output.png"
            utils.save_image(output_img, output_path)
            logger.info(f"✅ Try-on completed! Output saved to {output_path}")
            
            # Return numpy array for Gradio display
            return output_img, "✅ Try-on completed successfully!"
        else:
            logger.warning("Output image is empty or None")
            return None, "❌ Error: Generated image is invalid"
        
    except Exception as e:
        logger.error(f"Error in try-on processing: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Saree Virtual Try-On POC") as demo:
        gr.Markdown("""
        # 👗 Saree Virtual Try-On POC
        
        Upload a saree fabric image and a model photo to generate a virtual try-on!
        Optionally upload a blouse image, or we'll generate a matching one automatically.
        
        **Processing Time:** ~30-60 seconds
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📥 Upload Images")
                
                saree_input = gr.Image(
                    type="filepath",
                    label="📍 Saree Fabric (512x512 to 1024x1024 px)"
                )
                
                model_input = gr.Image(
                    type="filepath",
                    label="👤 Model Photo (768x1024 px, front-facing)"
                )
                
                blouse_input = gr.Image(
                    type="filepath",
                    label="👚 Blouse (Optional - auto-generate if blank)"
                )
                
                generate_btn = gr.Button(
                    "🎨 Generate Try-On",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                gr.Markdown("### 📤 Generated Try-On")
                
                output_image = gr.Image(
                    label="Result",
                    type="numpy",
                    interactive=False
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Waiting for input..."
                )
        
        # Set up button click handler
        generate_btn.click(
            fn=process_tryon,
            inputs=[saree_input, model_input, blouse_input],
            outputs=[output_image, status_text],
            show_progress=True
        )
        
        # Add examples section
        gr.Markdown("""
        ---
        ## ℹ️ About This POC
        
        This is a proof-of-concept for virtual try-on using:
        - **Simplified Segmentation** (full mask for ControlNet conditioning)
        - **MediaPipe** for pose detection
        - **Stable Diffusion ControlNet OpenPose** for virtual try-on synthesis
        - **Color Analysis** for automatic blouse generation
        
        **Current Features (V1):**
        - Single front view try-on
        - Pose-aware fabric placement
        - Automatic color-matched blouse generation
        - Support for custom blouse images
        
        **Processing Time:**
        - First run: 2-3 minutes (model download ~6GB)
        - Subsequent runs: 30-60 seconds per image on GPU
        
        **Note:** The TensorFlow Lite warnings are normal and don't affect functionality.
        """)
    
    return demo


if __name__ == "__main__":
    logger.info("Starting Saree Virtual Try-On POC UI...")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
