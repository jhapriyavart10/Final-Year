# 🩻 Medical Chest X-Ray Enhancement Pipeline

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)

## 📖 Project Overview
Medical imaging is critical for early diagnosis, but image quality is often compromised by noise (from low-dose radiation for safety) or low resolution (hardware/storage constraints). Poor quality obscures fine details like nodules or fractures.

This project is a **complete end-to-end Deep Learning pipeline** designed to restore high-quality visual details from degraded Chest X-Rays. It not only enhances the images visually but also validates that these enhancements improve disease classification accuracy.

---

## ⚙️ Architecture & Pipeline WorkFlow

The pipeline runs in 3 main stages:
1. **Stage 1: Denoising (DnCNN)** - Removes Gaussian noise from the degraded 64x64 input.
2. **Stage 2: Super-Resolution (Real-ESRGAN/RRDBNet)** - Upscales the denoised image by 4x (to 256x256) while restoring high-frequency details.
3. **Stage 3: Clinical Validation (ResNet18)** - Classifies the image as "Normal" or "Abnormal". We compare classification accuracy on *Degraded* vs *Enhanced* images to prove our pipeline's medical value.

---

## 📂 Repository Structure (Where to find what)

To easily navigate the project, here is how the files are organized:

```text
Root/
+-- app/                        # 💻 User Interface
|   +-- streamlit_app.py        # Interactive web app to upload and test images
+-- data/                       # 📊 Dataset & Results
|   +-- original/               # High-quality ground truth images (256x256)
|   +-- degraded/               # Low-quality noisy inputs (64x64)
|   +-- enhanced/               # Pipeline output outputs
|   +-- evaluation_outputs/     # Visual results, heatmaps, and saved charts
+-- evaluation/                 # 🧪 Testing & Metrics Scripts
|   +-- evaluate_pipeline.py    # Calculates PSNR / SSIM of the whole pipeline
|   +-- evaluate_classifier.py  # Tests if enhanced images classify better than degraded
|   +-- visualize_heatmaps.py   # Generates heatmaps showing restored details
|   +-- ... (other analytic scripts)
+-- models/                     # 🧠 Neural Network Architectures & Weights
|   +-- denoiser/               # DnCNN model architecture and .pth weights
|   +-- super_resolution/       # RRDBNet / ESRGAN architecture and weights
|   +-- classifier/             # ResNet18 architecture and weights
+-- training/                   # 🛠️ Scripts to Train the Models
|   +-- train_denoiser.py       # Train the DnCNN model
|   +-- train_sr.py             # Train the Super Resolution model
|   +-- train_classifier.py     # Train the ResNet validation classifier
+-- setup_data.py               # (Step 1) Downloads/Organizes base dataset
+-- create_degraded_data.py     # (Step 2) Generates noisy/low-res inputs for training
+-- inference_pipeline.py       # Core logic that connects Denoiser -> SR model
```

---

## 🚀 Step-by-Step Guide to the Project

### Step 1: Installation & Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Data Preparation
If you are setting up from scratch, prepare the data:
1. **Setup base folders & data**: `python setup_data.py`
2. **Generate the degraded inputs**: `python create_degraded_data.py`
*(This creates the degraded 64x64 images from the original 256x256 images).*

### Step 3: Training (Optional - Pre-trained weights are provided)
If you want to retrain the models from scratch:
1. `python training/train_denoiser.py`
2. `python training/train_sr.py`
3. `python training/train_classifier.py`

### Step 4: Running Evaluations (Testing the system)
Verify how well the system works computationally:
- **Measure Image Quality (PSNR/SSIM):** `python evaluation/evaluate_pipeline.py`
- **Measure Clinical Accuracy:** `python evaluation/evaluate_classifier.py`
- **Generate Visual Error Heatmaps:** `python evaluation/visualize_heatmaps.py`
- **Explainable AI (Grad-CAM):** `python evaluation/visualize_gradcam.py` *(Shows where the AI is looking during diagnosis)*

### Step 5: Run the Interactive Web App (Demo)
Launch the user interface to visually test the pipeline on new X-Ray images:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Summary of Results

### 1. Image Quality Metrics
We evaluate the visual restoration quality of our pipeline by comparing the generated images against the original ground truth.

| Metric | Before (Bicubic Baseline) | After (Full Pipeline) | Improvement |
| :--- | :--- | :--- | :--- |
| **PSNR (dB)** | 25.2029 | 27.6412 | **+ 2.4383 dB** |
| **SSIM** | 0.5415 | 0.7728 | **+ 0.2313** |

*Result: The pipeline successfully reconstructs missing details, significantly boosting both peak signal-to-noise ratio and structural similarity.*

### 2. Clinical Validation & Ablation Study
To prove the medical utility of the restored images, we tested a ResNet18 classifier on the original degraded images versus the AI-enhanced outputs. We also performed an ablation study (testing the Super-Resolution model without the Denoiser) to validate the necessity of the multi-stage approach.

| Configuration | Accuracy | F1-Score | AUC |
| :--- | :--- | :--- | :--- |
| **Degraded Input (Baseline)** | 0.5896 | 0.5639 | 0.6022 |
| **ESRGAN Only (Ablation)** | 0.6848 | 0.6651 | 0.7599 |
| **Full Pipeline (DnCNN + ESRGAN)** | **0.7211** | **0.6854** | **0.7959** |

**Conclusion:** Enhancing the images leads to a massive **+13.15% jump in diagnostic accuracy**. Furthermore, the full pipeline strictly outperforms the SR-only model, proving that combining target-specific Denoising and Super-Resolution preserves the most clinically relevant medical features.

### ⏱️ Deployment & Efficiency Metrics
For a system to be viable in a real-world clinical setting, it must be lightweight and fast. We benchmarked the models on a standard CPU:
- **Total Pipeline Footprint**: `60.94 MB` (Very lightweight, easy to deploy on edge devices or standard hospital PCs)
  - *DnCNN*: 1.15 MB
  - *Real-ESRGAN*: 17.10 MB
  - *ResNet18*: 42.69 MB
- **Inference Latency**: `~303 ms` per image (CPU). The pipeline processes images at **~3.3 FPS** purely on CPU, meaning it can provide near real-time enhancement without requiring expensive GPU hardware.

## 💡 Notes for Future Development
- **Real-World Data:** Currently trained on synthetic Gaussian noise. Future iterations should test authentic low-dose/high-dose paired X-Rays.
- **Resource Constraints:** Ensure you have a CUDA-capable GPU for faster inference, though the pipeline will fall back to CPU if necessary.
