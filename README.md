# Medical Chest X-Ray Image Enhancement Pipeline

## Goal
Build a lightweight multi-stage medical image enhancement pipeline for chest X-ray images.

## Pipeline Stages
1. **Denoising**: Remove noise from medical images.
2. **Super-Resolution**: Enhance resolution using GAN-based models (Real-ESRGAN).
3. **Classification**: Disease classification on enhanced images to verify clinical utility.

## Project Structure
- `data/`: Storage for original, degraded, and enhanced datasets.
- `models/`: PyTorch model definitions for each stage.
- `training/`: Training scripts.
- `evaluation/`: Scripts for calculating metrics (PSNR, SSIM) and evaluating classifier performance.
- `app/`: Streamlit web interface for demonstration.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
