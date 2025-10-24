# Saree Virtual Try-On POC

A proof-of-concept for AI-driven virtual try-on of traditional Indian sarees using HR-VITON, SAM 2, and OpenPose.

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 11.7 support (RTX 3060 or better recommended)
- Python 3.10
- Conda or virtualenv

### Installation

1. **Create and activate environment:**
```bash
conda create -n saree-tryon python=3.10 -y
conda activate saree-tryon
```

2. **Install PyTorch with CUDA:**
```bash
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download model weights:**

   a. SAM 2:
   ```bash
   mkdir -p models/sam2
   cd models/sam2
   wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
   cd ../..
   ```
   
   b. HR-VITON:
   ```bash
   # Download from HuggingFace or official repository
   mkdir -p models/hrviton
   # Place weights in models/hrviton/
   ```
   
   c. ControlNet (auto-downloaded on first use)

### Running the Application

```bash
python app.py
```

The UI will be available at `http://localhost:7860`

## Usage

1. **Upload Saree:** Click the first input box and select a flat saree fabric image
2. **Upload Model:** Select a front-facing model photo
3. **Optional Blouse:** Upload a blouse image, or leave blank to auto-generate
4. **Generate:** Click the "Generate Try-On" button
5. **Download:** Right-click the output image to save

### Input Requirements

- **Saree Image:** 512x512 to 1024x1024 px, PNG/JPG format, flat fabric photo
- **Model Photo:** 768x1024 px preferred, 3:4 aspect ratio, clean background, front-facing
- **Blouse Image:** 512x512 px, PNG/JPG format (optional)

### Expected Output

- 768x1024 px image showing the model wearing the saree and blouse
- Processing time: 30-60 seconds on RTX 3060

## Project Structure

```
saree-tryon-poc/
├── app.py                    # Gradio UI entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── DEVELOPMENT_PLAN.md       # Development timeline
├── configs/
│   └── config.yaml          # Configuration file
├── src/
│   ├── segmentation.py      # SAM 2 wrapper
│   ├── pose_extraction.py   # OpenPose wrapper
│   ├── color_analysis.py    # Blouse color generation
│   ├── tryon_pipeline.py    # HR-VITON wrapper
│   └── utils.py             # Helper functions
├── models/                   # Model weights (not in git)
│   ├── sam2/
│   ├── hrviton/
│   └── controlnet/
├── data/                     # Sample images
│   ├── sample_sarees/
│   ├── sample_models/
│   └── sample_blouses/
└── outputs/                  # Generated results
    ├── masks/
    ├── poses/
    └── results/
```

## Components

### Segmentation (src/segmentation.py)
Uses SAM 2 (Segment Anything Model 2) to extract clean masks of saree and blouse from input images.

**Status:** Placeholder implementation, ready for SAM 2 integration

### Pose Extraction (src/pose_extraction.py)
Extracts body pose keypoints from model photo using ControlNet OpenPose.

**Status:** Placeholder implementation, ready for ControlNet integration

### Color Analysis (src/color_analysis.py)
Extracts dominant colors from saree and generates a matching solid-color blouse.

**Status:** Fully implemented, tested with K-means clustering

### Try-On Pipeline (src/tryon_pipeline.py)
Main inference pipeline that combines all components with HR-VITON for virtual try-on synthesis.

**Status:** Placeholder implementation, ready for HR-VITON integration

### Utilities (src/utils.py)
Helper functions for image processing, I/O, and logging.

**Status:** Fully implemented

## Configuration

Edit `configs/config.yaml` to customize:

- Model paths
- Image sizes
- Processing parameters
- Device settings (GPU/CPU)
- Output quality

## Development Status

### Completed ✅
- Project structure
- Configuration system
- Utility functions
- Color analysis module
- Gradio UI
- Documentation

### In Progress 🔄
- SAM 2 integration
- ControlNet OpenPose integration
- HR-VITON model loading
- End-to-end pipeline testing

### Planned 📋
- Advanced draping rules
- Multi-view generation
- Pose customization
- Performance optimization
- Docker containerization

## Performance Targets

- Generation time: < 60 seconds on RTX 3060
- Memory usage: < 12 GB VRAM
- Output quality: Recognizable garments with clear patterns

## Known Limitations

- **Single view only:** Front view only in this POC
- **Basic draping:** Natural draping from model, not custom Assamese-style
- **Pattern preservation:** Basic quality in POC, improved in V2
- **No pose editing:** Uses model's natural pose
- **Local only:** No cloud deployment in POC
- **Single GPU:** No multi-GPU support

## Troubleshooting

### CUDA/GPU Issues
```bash
# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"

# Fall back to CPU (slower)
# Edit config.yaml: device.use_gpu = false
```

### Out of Memory
```bash
# Option 1: Reduce resolution in config.yaml
# Option 2: Enable memory_efficient mode in config.yaml
# Option 3: Use CPU offloading
```

### Model Download Failures
- Check internet connection
- Download manually from:
  - SAM 2: https://github.com/facebookresearch/sam2
  - HR-VITON: HuggingFace Model Hub
- Place in `models/` directory

## Next Steps

1. Download all required model weights
2. Test each module individually
3. Run full end-to-end pipeline
4. Generate demo outputs
5. Gather client feedback
6. Plan V2 enhancements

## References

- **SAM 2:** https://github.com/facebookresearch/sam2
- **HR-VITON:** https://github.com/sangyun884/hr-viton
- **ControlNet:** https://github.com/lllyasviel/ControlNet
- **Gradio:** https://gradio.app

## License

This project is for demonstration and research purposes only.

## Contact

For questions or feedback, please contact the development team.

---

**Created:** October 24, 2025  
**Status:** Active Development  
**Version:** 0.1.0 (POC)
