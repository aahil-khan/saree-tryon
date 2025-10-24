# ⚡ Quick Start Guide - Saree Virtual Try-On POC

## TL;DR - Get Running in 5 Steps

### 1️⃣ Create Environment (1 min)
```bash
conda create -n saree-tryon python=3.10 -y
conda activate saree-tryon
```

### 2️⃣ Install PyTorch (2 min)
```bash
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
```

### 3️⃣ Install Dependencies (3 min)
```bash
cd "c:/Users/aahil/OneDrive/Documents/Freelance/Saree MVP"
pip install -r requirements.txt
```

### 4️⃣ Download Models (30-60 min - mostly download time)
```bash
# SAM 2
mkdir -p models/sam2
cd models/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ../..

# HR-VITON - Download from: https://github.com/sangyun884/hr-viton
# Place in models/hrviton/
```

### 5️⃣ Run App (1 min)
```bash
python app.py
```

**Then open:** http://localhost:7860

---

## Project Files Created

✅ **Ready to Use:**
- `app.py` - Main Gradio UI
- `requirements.txt` - All dependencies
- `README.md` - Full setup guide
- `DEVELOPMENT_PLAN.md` - 3-4 day timeline

✅ **Core Modules (src/):**
- `color_analysis.py` - FULLY IMPLEMENTED ✅
- `utils.py` - FULLY IMPLEMENTED ✅
- `segmentation.py` - Ready for SAM 2 integration 🔄
- `pose_extraction.py` - Ready for OpenPose integration 🔄
- `tryon_pipeline.py` - Ready for HR-VITON integration 🔄

✅ **Configuration:**
- `configs/config.yaml` - All settings
- `.gitignore` - Git rules
- Folders: `models/`, `data/`, `outputs/`, `tests/`

---

## What's Working NOW

```python
# Color Analysis
from src.color_analysis import generate_matching_blouse
blouse = generate_matching_blouse("saree.jpg")

# Image Utils
from src.utils import load_image, resize_image, save_image
img = load_image("image.jpg")
img = resize_image(img, (768, 1024))
save_image(img, "output.jpg")

# Logging
from src.utils import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)
```

---

## What Needs Integration (Next 2-3 Days)

1. **SAM 2 Segmentation** (2-3 hours)
   - Edit `src/segmentation.py`
   - Load SAM 2 model
   - Implement mask generation

2. **OpenPose** (2-3 hours)
   - Edit `src/pose_extraction.py`
   - Load ControlNet OpenPose
   - Extract keypoints

3. **HR-VITON** (3-4 hours)
   - Edit `src/tryon_pipeline.py`
   - Load HR-VITON model
   - Implement inference

4. **Testing & Refinement** (2-3 hours)
   - Test each component
   - End-to-end testing
   - Quality refinement

---

## File Locations

```
📁 c:\Users\aahil\OneDrive\Documents\Freelance\Saree MVP
├── app.py                       # START HERE
├── requirements.txt             # Dependencies
├── README.md                    # Full guide
├── DEVELOPMENT_PLAN.md          # Timeline
├── SETUP_COMPLETE.md            # Setup details
├── PRD.md                       # Project requirements
├── 
├── 📁 src/
│   ├── __init__.py
│   ├── segmentation.py         # SAM 2 wrapper
│   ├── pose_extraction.py      # OpenPose wrapper
│   ├── color_analysis.py       # ✅ DONE
│   ├── tryon_pipeline.py       # HR-VITON wrapper
│   └── utils.py                # ✅ DONE
├── 
├── 📁 configs/
│   └── config.yaml             # Configuration
├── 
├── 📁 models/                  # (Download here)
│   ├── sam2/                   # SAM 2 checkpoint
│   ├── hrviton/                # HR-VITON weights
│   └── controlnet/             # (Auto-downloads)
├── 
├── 📁 data/                    # (Sample images)
├── 📁 outputs/                 # (Results go here)
└── 📁 tests/                   # (Unit tests)
```

---

## What You'll See

### After Running `python app.py`:
```
Running on local URL:  http://0.0.0.0:7860
```

### UI Layout:
```
┌─────────────────────────────────────────────────────┐
│         👗 Saree Virtual Try-On POC                 │
├─────────────────────┬───────────────────────────────┤
│                     │                               │
│  INPUTS:            │    OUTPUT:                    │
│                     │                               │
│  📍 Saree Fabric    │    Generated Try-On           │
│  👤 Model Photo     │    (Result image)             │
│  👚 Blouse (opt)    │                               │
│                     │    Status: Waiting...         │
│  [Generate Button]  │                               │
│                     │                               │
└─────────────────────┴───────────────────────────────┘
```

---

## Troubleshooting

### CUDA Issues?
```bash
# Test GPU
python -c "import torch; print(torch.cuda.is_available())"

# If false, check drivers or use CPU mode (slower)
```

### Import Errors?
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade

# Or specific package
pip install diffusers==0.20.2
```

### Model Download Failed?
- Check internet connection
- Download manually from GitHub
- Place in `models/` directory

---

## Development Workflow

**Day 1:** Environment + Model downloads
- [ ] Run setup steps 1-4 above
- [ ] Verify all imports work
- [ ] Test color_analysis module

**Day 2:** Integration
- [ ] Implement SAM 2 (segmentation.py)
- [ ] Implement OpenPose (pose_extraction.py)
- [ ] Test modules individually

**Day 3:** Pipeline
- [ ] Implement HR-VITON (tryon_pipeline.py)
- [ ] Run end-to-end tests
- [ ] Generate demo outputs

**Day 4:** Refinement
- [ ] Fix bugs
- [ ] Optimize performance
- [ ] Document results

---

## Key Configuration

Edit `configs/config.yaml` if needed:

```yaml
# Change resolution
image:
  target_size: [768, 1024]  # or [512, 768] for faster processing

# Enable CPU mode (if no GPU)
device:
  use_gpu: false

# Reduce VRAM usage
device:
  memory_efficient: true
```

---

## Performance Expectations

- **Setup:** 5-10 minutes (first time only)
- **Model Download:** 30-60 minutes
- **First Run:** 1-2 minutes (model loading)
- **Subsequent Runs:** 30-60 seconds per image

---

## Next Steps

1. ✅ You have the foundation
2. 🔄 Download models (Phase 2)
3. 🔄 Integrate SAM 2, OpenPose, HR-VITON (Phase 3)
4. 🔄 Test and refine (Phase 4)
5. ✅ Deploy and demo for client

---

## Support Resources

- **SAM 2:** https://github.com/facebookresearch/sam2
- **HR-VITON:** https://github.com/sangyun884/hr-viton
- **ControlNet:** https://github.com/lllyasviel/ControlNet
- **Gradio:** https://gradio.app/docs
- **OpenPose:** https://github.com/CMU-Perceptron/openpose

---

**Status:** ✅ Ready to Start  
**Est. Completion:** October 27-28, 2025  
**Total Development Time:** 20-24 hours

Good luck! 🚀
