# Medical Chest X-Ray Enhancement Pipeline

## 📌 Problem Statement
Medical imaging is critical for early diagnosis, but image quality is often compromised by:
- **Noise**: Caused by low-dose acquisition (to ensure patient safety) or sensor limitations.
- **Low Resolution**: Due to hardware constraints or compression for storage/transmission.

Poor image quality can obscure fine details (e.g., nodules, fractures) and reduce the accuracy of both human radiologists and automated AI diagnostic tools. This project builds a lightweight, post-processing pipeline to restore high-quality visualizations from degraded inputs.

## 📂 Dataset
- **Source**: [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data).
- **Preprocessing**:
  - **Degraded Input**: 64x64 resolution + Gaussian Noise.
  - **Ground Truth**: 256x256 High-Quality Grayscale.
- **Structure**:
  - `data/original`: High-quality reference images.
  - `data/degraded`: Low-quality inputs used for testing.
  - `data/enhanced`: Pipeline outputs.

## ⚙️ Methodology
This project implements a multi-stage deep learning pipeline:

### Stage 1: Denoising (DnCNN)
- **Model**: Denoising Convolutional Neural Network (17 layers).
- **Goal**: Blind Gaussian noise removal.
- **Input**: Noisy Image $\rightarrow$ **Output**: Clean Image.

### Stage 2: Super-Resolution (Real-ESRGAN)
- **Model**: Residual-in-Residual Dense Block Network (RRDBNet) + UNet Discriminator.
- **Goal**: 4x Upscaling (64x64 $\rightarrow$ 256x256).
- **Losses**: Perceptual Loss (VGG), Adversarial Loss, and L1 Pixel Loss.

### Stage 3: Clinical Validation (Classification)
- **Model**: ResNet18 (Modified for 1-channel input).
- **Goal**: Classify images as "Normal" or "Abnormal" (Finding).
- **Hypothesis**: Classification accuracy should be higher on **Enhanced** images compared to **Degraded** ones.

## 📊 Results

### Image Quality Metrics
| Metric | Degraded (Input) | Enhanced (Output) | Improvement |
|:-------|:----------------:|:-----------------:|:-----------:|
| **PSNR** | ~25.40 dB | **~28.15 dB** | **+2.75 dB** |
| **SSIM** | ~0.55 | **~0.78** | **+0.23** |

*Note: Results averaged over local test set.*

### Visual Analysis
- **Heatmaps**: Difference maps (`visualize_heatmaps.py`) show the model recovering high-frequency edge details in the rib cage and lung interior.
- **Artifacts**: Minor GAN hallucinations observed in extremely noisy regions.

## ⚠️ Limitations
1.  **Synthetic Data**: Models trained on synthetic degradation (Gaussian noise) may not perfectly generalize to real-world quantum noise or motion blur.
2.  **Hallucinations**: Generative models can invent details. Clinical validation is essential.

## 🚀 Future Work
1.  **Real-world Pairs**: Use Low-Dose vs. Normal-Dose CT pairs.
2.  **Uncertainty Maps**: Visualize where the model is uncertain about the reconstruction.
3.  **Transformer Models**: Experiment with SwinIR for potentially better texture recovery.

## 🛠️ Usage

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
Launch the interactive web app to upload images and see the pipeline in action:
```bash
streamlit run app/streamlit_app.py
```

### 3. Evaluation
Generate metrics and heatmaps:
```bash
python evaluation/evaluate_pipeline.py
python evaluation/visualize_heatmaps.py
```
