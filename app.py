"""
Saree Virtual Try-On POC - Gradio UI
"""

import gradio as gr
import numpy as np
import logging
from pathlib import Path
import cv2

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
        logger.info("Loading images...")
        saree_img = utils.load_image(saree_image) if isinstance(saree_image, str) else saree_image
        model_img = utils.load_image(model_image) if isinstance(model_image, str) else model_image
        
        # Resize model image to target size
        model_img = utils.resize_image(model_img, (768, 1024))
        
        # Step 2: Generate or load blouse
        if blouse_image is not None:
            logger.info("Loading provided blouse image...")
            blouse_img = utils.load_image(blouse_image) if isinstance(blouse_image, str) else blouse_image
            blouse_img = utils.resize_image(blouse_img, (768, 1024))
        else:
            logger.info("Generating matching blouse from saree...")
            blouse_img = color_analysis.generate_matching_blouse(
                saree_image,
                size=(1024, 768)
            )
            # Save generated blouse for reference
            utils.save_image(blouse_img, "./outputs/generated_blouse.png")
        
        # Step 3: Segment garments (placeholder)
        logger.info("Segmenting garments...")
        saree_mask = np.ones((1024, 768), dtype=np.uint8) * 255
        blouse_mask = np.ones((1024, 768), dtype=np.uint8) * 255
        
        # Step 4: Extract pose (placeholder)
        logger.info("Extracting pose...")
        pose_map = np.zeros((1024, 768, 3), dtype=np.uint8)
        
        # Step 5: Run try-on
        logger.info("Running try-on pipeline...")
        output_img = tryon_pipeline.run_tryon(
            model_img=model_img,
            saree_img=saree_img,
            blouse_img=blouse_img,
            pose_map=pose_map,
            saree_mask=saree_mask,
            blouse_mask=blouse_mask,
            device="cuda"
        )
        
        # Step 6: Save output
        output_path = "./outputs/results/tryon_output.png"
        utils.save_image(output_img, output_path)
        logger.info(f"Try-on completed! Output saved to {output_path}")
        
        # Convert to BGR for display
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        return output_bgr, "✅ Try-on completed successfully!"
        
    except Exception as e:
        logger.error(f"Error in try-on processing: {e}")
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
            with gr.Column(label="Inputs"):
                gr.Markdown("### Upload Images")
                
                saree_input = gr.Image(
                    type="filepath",
                    label="📍 Saree Fabric",
                    info="Upload flat saree fabric image (512x512 to 1024x1024 px)"
                )
                
                model_input = gr.Image(
                    type="filepath",
                    label="👤 Model Photo",
                    info="Upload model photo (768x1024 px, front-facing)"
                )
                
                blouse_input = gr.Image(
                    type="filepath",
                    label="👚 Blouse (Optional)",
                    info="Upload blouse image or leave blank to auto-generate"
                )
                
                generate_btn = gr.Button(
                    "🎨 Generate Try-On",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(label="Output"):
                gr.Markdown("### Generated Try-On")
                
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
        - **SAM 2** for garment segmentation
        - **ControlNet OpenPose** for pose detection
        - **HR-VITON** for virtual try-on synthesis
        - **Color Analysis** for automatic blouse generation
        
        **Current Limitations (V1):**
        - Single front view only
        - Basic pattern preservation
        - No pose customization
        - Local processing only
        
        **Next Steps (V2):**
        - Multi-view generation
        - Advanced draping
        - Better pattern alignment
        - Cloud deployment
        """)
    
    return demo


if __name__ == "__main__":
    logger.info("Starting Saree Virtual Try-On POC UI...")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
