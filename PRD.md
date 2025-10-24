# Product Requirements Document: Saree Virtual Try-On POC

## 1. Overview
**Product Name:** Saree Virtual Try-On POC
**Purpose:** Demonstrate basic feasibility of AI-driven virtual try-on for traditional Indian sarees
**Success Criteria:** Generate one front-view image of a model wearing uploaded saree with basic draping and matching blouse
**Timeline:** 3-4 days
**Scope:** Basic proof-of-concept only; focused on core try-on functionality with saree and blouse placement

## 2. Functional Requirements

### 2.1 Inputs
Two required image uploads:
* **Saree fabric image** (flat cloth, full saree or representative portion, 512x512 to 1024x1024 px, PNG/JPG)
* **Model photo** (person, 768x1024 px preferred, 3:4 aspect ratio, clean background, front-facing pose)

Optional input:
* **Blouse image** (either a fabric swatch or existing blouse photo, 512x512 px, PNG/JPG)
  * If not provided, system will generate a matching solid color blouse based on saree colors

### 2.2 Processing
* Segment saree from flat fabric image using SAM 2
* Extract pose keypoints from model photo using OpenPose via ControlNet
* Process blouse handling:
  * If blouse image provided: segment and process using SAM 2
  * If no blouse: extract dominant color from saree and generate matching solid color blouse
* Synthesize front-view output using HR-VITON:
  * Apply basic saree draping around body
  * Place blouse on upper body
  * Ensure natural fabric flow

### 2.3 Output
One 768x1024 px front-view image showing:
* Model wearing saree with basic draping
* Matching blouse on upper body
* Preserved saree patterns and colors
* Original model face and pose maintained

### 2.4 Non-Functional Requirements
* Generation time: under 60 seconds on a mid-range laptop GPU (RTX 3060/4060 or better)
* No multi-user support, authentication, or database required
* Local execution only; no cloud deployment
* Simple single-page UI (Gradio)

## 3. Technical Stack

### 3.1 Core Models
| Component | Model | Purpose | License |
| :--- | :--- | :--- | :--- |
| Try-on synthesis | HR-VITON | High-resolution virtual try-on with pattern preservation | Research use |
| Pose control | ControlNet OpenPose | Basic pose extraction and body keypoints | OpenRAIL-M |
| Segmentation | SAM 2 | Saree and blouse segmentation | Apache 2.0 |
| Color Analysis | Basic CV | Extract colors for matching blouse generation | N/A |

### 3.2 Dependencies
* Python 3.10
* PyTorch 2.0.0+cu117
* torchvision 0.15.1+cu117
* diffusers 0.20.2
* transformers 4.33.2
* controlnet_aux
* segment-anything (SAM 2)
* opencv-python 4.7.0.72
* gradio 4.x
* pillow
* numpy
* scikit-image
* omegaconf
* xformers 0.0.19

### 3.3 Pretrained Weights
* **HR-VITON:** Download from HuggingFace repository
* **ControlNet OpenPose:** `lllyasviel/sd-controlnet-openpose` from HuggingFace (auto-downloaded)
* **SAM 2:** Download checkpoint from Meta SAM 2 repo

## 4. Project Structure
```text
saree-tryon-poc/
├── app.py                     # Gradio UI entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Setup instructions
├── models/
│   ├── hrviton/              # HR-VITON weights and config
│   ├── controlnet/           # ControlNet OpenPose weights
│   └── sam2/                 # SAM 2 checkpoint
├── src/
│   ├── segmentation.py       # SAM 2 fabric segmentation
│   ├── pose_extraction.py    # OpenPose keypoint detection
│   ├── color_analysis.py     # Color extraction for blouse
│   ├── tryon_pipeline.py     # HR-VITON inference wrapper
│   └── utils.py              # Image preprocessing, postprocessing
├── data/
│   ├── sample_sarees/        # 3 sample saree textures
│   ├── sample_blouses/       # 3 sample blouse images (optional)
│   └── sample_models/        # 3 sample model photos
└── outputs/                   # Generated try-on results
```

## 5. Implementation Details

### 5.1 Segmentation Pipeline (src/segmentation.py)
**Objective:** Extract clean masks for saree and optional blouse from input images

**Steps:**
1. Load saree image (and blouse if provided)
2. Initialize SAM 2 with `sam2_hiera_large` checkpoint
3. Run automatic mask generation for saree fabric
4. Filter mask by size (keep largest contiguous region)
5. Save saree mask as binary PNG
6. If blouse provided, repeat steps 3-5 for blouse
7. Extract border regions using basic edge detection

**Input:** `saree.jpg`, `blouse.jpg` (optional)
**Output:** `saree_mask.png`, `blouse_mask.png` (if provided), cropped fabric regions

### 5.2 Pose Extraction (src/pose_extraction.py)
**Objective:** Extract body pose keypoints from model photo for ControlNet conditioning

**Steps:**
1. Load model image and resize to 768x1024 if needed
2. Use `controlnet_aux.OpenposeDetector` to extract pose
3. Extract keypoints (shoulders, arms, torso, waist)
4. Render pose skeleton as 768x1024 image for ControlNet input
5. Save pose map

**Input:** `model.jpg`
**Output:** `pose_map.png` (skeleton overlay)

### 5.3 Color Analysis Pipeline (src/color_analysis.py)
**Objective:** Extract dominant colors from saree for automatic blouse generation

**Steps:**
1. Load saree fabric image
2. Resize to manageable size (256x256)
3. Use K-means clustering to find 3-5 dominant colors
4. Select complementary color from palette (lighter shade or contrasting color)
5. Generate simple solid color blouse image
6. Save as blouse template

**Input:** `saree.jpg`
**Output:** `generated_blouse.png` (solid color image)
**Notes:** Only used when user doesn't provide a blouse image

### 5.4 Try-On Pipeline (src/tryon_pipeline.py)
**Objective:** Synthesize model wearing saree and blouse using HR-VITON

**Steps:**
1. Load HR-VITON model and weights
2. Prepare inputs:
   * Model image (768x1024)
   * Saree texture + mask
   * Blouse texture + mask (or generated)
   * Pose map from ControlNet OpenPose
3. Process through HR-VITON:
   * Encode saree and blouse features
   * Condition on pose map via ControlNet
   * Apply basic garment placement masks
   * Run inference with model parameters
4. Post-process output for clarity
5. Save result as 768x1024 PNG

**Input:** All preprocessed assets (model, saree, blouse, pose map)
**Output:** `output_front.png` (768x1024)

### 5.5 UI (app.py)
**Objective:** Simple upload-and-generate interface for saree try-on

**Framework:** Gradio

**Layout:**
```python
import gradio as gr
from src.tryon_pipeline import run_tryon

def generate(saree_img, model_img, blouse_img=None):
    output = run_tryon(saree_img, model_img, blouse_img)
    return output

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="filepath", label="Upload Saree"),
        gr.Image(type="filepath", label="Upload Model Photo"),
        gr.Image(type="filepath", label="Upload Blouse (Optional)")
    ],
    outputs=gr.Image(label="Output"),
    title="Saree Virtual Try-On POC",
    description="Upload saree and model image to generate virtual try-on")

demo.launch()
```

**UX Flow:**
1. User uploads saree fabric image
2. User uploads model photo
3. User optionally uploads blouse (or system generates one)
4. Click "Generate"
5. Processing runs (30–60s)
6. Output displayed inline
7. Download button enabled

## 6. Setup Instructions

### 6.1 Environment Setup
```bash
# Create conda environment
conda create -n saree-tryon python=3.10 -y
conda activate saree-tryon

# Install PyTorch with CUDA 11.7
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install core dependencies
pip install diffusers==0.20.2 transformers==4.33.2 xformers==0.0.19
pip install opencv-python pillow numpy omegaconf gradio scikit-image

# Install ControlNet auxiliary
pip install controlnet-aux

# Install SAM 2
pip install git+https://github.com/facebookresearch/sam2.git
```

### 6.2 Download Model Weights
```bash
# Create models directory
mkdir -p models/hrviton models/controlnet models/sam2

# HR-VITON (from HuggingFace or GitHub releases)
# Download and place in models/hrviton/

# ControlNet OpenPose (auto-downloaded by diffusers)
# Will be cached on first run

# SAM 2 checkpoint
cd models/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 6.3 Sample Data
Place 3 sample sarees and 3 model images in `data/` subdirectories for testing.

## 7. Acceptance Criteria

### 7.1 Minimum Viable Output
- [ ] Pipeline runs end-to-end without crashes
- [ ] Output image is 768x1024 px, clear, and centered
- [ ] Saree is visible with basic draping and recognizable pattern
- [ ] Solid color blouse visible on upper body
- [ ] Model's face and body proportions maintained
- [ ] Basic garment placement is correct and recognizable

### 7.2 Quality Benchmarks
- [ ] Fabric pattern visibility: Main saree patterns/colors are visible
- [ ] Pose fidelity: Body pose matches input model
- [ ] Basic placement: Saree at lower body, blouse at upper body

### 7.3 Performance
- [ ] Total generation time under 60 seconds on RTX 3060 or equivalent
- [ ] Memory usage under 12 GB VRAM at 768x1024 resolution

## 8. Out of Scope (V1)
* Multi-view generation (back, side views)
* Multiple pose variants
* User pose selection or editing
* Production deployment (Docker, API, cloud hosting)
* Authentication, user management
* Dataset collection or fine-tuning on saree images
* Super-resolution or 4K outputs
* Real-time inference or batching
* Advanced artifact cleanup (inpainting, refinement passes)

## 9. Risks & Mitigations
| Risk | Mitigation |
| :--- | :--- |
| Arm/waist occlusion artifacts | Acceptable in initial MVP; refinement in V2 |
| Pattern misalignment on complex fabrics | Use HR-VITON's pattern preservation; test with varied patterns |
| Pose mismatch with model input | Validate OpenPose keypoints visually |
| VRAM overflow on laptop GPU | Reduce resolution to 512x768 or use CPU offloading |
| Blouse color mismatch | Allow user to upload blouse for better control |

## 10. Testing Plan

### 10.1 Unit Tests
* `test_segmentation.py`: Verify SAM 2 produces valid masks for 3 sample sarees
* `test_pose.py`: Verify OpenPose extracts keypoints for 3 sample model photos
* `test_color.py`: Verify color extraction generates appropriate blouse colors

### 10.2 Integration Test
* Run full pipeline on one sample set (saree + model + optional blouse)
* Validate output dimensions, file format, and visual quality

### 10.3 Acceptance Test
* Run on 1-3 different sample sets
* Subjectively rate outputs against acceptance criteria
* Target: 1 passing quality demo for POC

## 11. Deliverables
* Working codebase in `saree-tryon-poc/` with all scripts and dependencies
* Gradio UI running locally with saree, model, and optional blouse upload
* 1-3 demo outputs showing saree virtual try-on with different fabrics
* `README.md` with setup, usage, and known limitations
* Short demo video (optional, 30–60s) showing upload → generate → output flow

## 12. Success Metrics
**Go/No-Go Decision Criteria:**
* If demo output shows basic saree placement with recognizable pattern and blouse → **Proceed to production planning**
* If output fails to show basic garment placement or has severe artifacts → **Revisit approach**

## 13. References & Resources
* **HR-VITON GitHub:** [https://github.com/sangyun884/hr-viton](https://github.com/sangyun884/hr-viton)
* **SAM 2 GitHub:** [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
* **ControlNet GitHub:** [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
* **ControlNet OpenPose Guide:** Stable Diffusion ControlNet

## 14. Implementation Checklist
- [ ] Set up conda environment and install dependencies
- [ ] Download HR-VITON, ControlNet, and SAM 2 weights
- [ ] Implement `src/segmentation.py` with SAM 2 masking
- [ ] Implement `src/pose_extraction.py` with OpenPose
- [ ] Implement `src/color_analysis.py` for blouse generation
- [ ] Implement `src/tryon_pipeline.py` integrating HR-VITON
- [ ] Build `app.py` Gradio UI with saree/model/blouse upload
- [ ] Test on sample sets and validate outputs
- [ ] Document setup in `README.md`
- [ ] Generate demo output for client presentation

***
*This PRD is optimized for a rapid 3-4 day POC implementation with clear technical choices and simple, achievable goals to demonstrate virtual try-on feasibility.*
