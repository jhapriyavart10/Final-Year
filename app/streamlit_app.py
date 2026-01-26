import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet
from evaluation.metrics import calculate_psnr, calculate_ssim

# Page Config
st.set_page_config(page_title="Medical Image Enhancement", layout="wide")

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Denoiser
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(device)
    denoiser_path = os.path.join("models", "denoiser", "dncnn_best.pth")
    if os.path.exists(denoiser_path):
        denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
        denoiser.eval()
    else:
        st.error(f"Denoiser weights not found at {denoiser_path}")
        return None, None, device

    # Load SR Model
    sr_model = RRDBNet(in_channels=1, out_channels=1).to(device)
    sr_path = os.path.join("models", "super_resolution", "generator_latest.pth")
    if os.path.exists(sr_path):
        sr_model.load_state_dict(torch.load(sr_path, map_location=device))
        sr_model.eval()
    else:
        st.error(f"SR Model weights not found at {sr_path}")
        return None, None, device
        
    return denoiser, sr_model, device

def process_image(img, denoiser, sr_model, device):
    """Run the enhancement pipeline"""
    # Preprocess
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    
    # Degrade manually for demo if needed, or assume input IS degraded.
    # For this demo, let's assume the user uploads a LOW QUALITY image (or we simulate it).
    # Let's simulate degradation so we have a comparison if the user uploads a high-res image.
    
    # Simulate low-res/noisy input from the upload
    h, w = img_np.shape
    # Resize to 64x64 for model input (simulating low res)
    img_lr = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    # Add noise
    noise = np.random.normal(0, 15, img_lr.shape).astype(np.float32)
    img_lr_noisy = np.clip(img_lr + noise, 0, 255).astype(np.uint8)
    
    # Prepare Tensor
    input_tensor = torch.from_numpy(img_lr_noisy / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. Denoise
        # DnCNN predicts noise. Denoised = Input - Noise
        noise_pred = denoiser(input_tensor)
        denoised_tensor = input_tensor - noise_pred # Wait, my training script did noise prediction?
        # Let's check training script: model(noisy), criterion(outputs, clean). 
        # Ah! In train_denoiser.py, I used criterion(outputs, clean).
        # This means the model learned to predict CLEAN image directly, NOT the noise residual,
        # UNLESS the DnCNN class definition handles the subtraction internally.
        # Checking DnCNN class in dncnn.py:
        # def forward(self, x): out = self.dncnn(x); return x - out
        # So: self.dncnn(x) predicts the noise, and forward() returns (Input - Noise).
        # So the output of model(x) IS the cleaned image.
        
        cleaned_tensor = denoiser(input_tensor)
        
        # 2. Super-Resolution
        enhanced_tensor = sr_model(cleaned_tensor)
        
    # Post-process
    # LR/Noisy Image
    img_lr_disp = img_lr_noisy
    
    # Enhanced Image
    enhanced_np = enhanced_tensor.squeeze().cpu().numpy()
    enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
    
    # Resize output to match original if needed for metrics (it should be 256x256)
    if enhanced_np.shape != img_np.shape:
        enhanced_np = cv2.resize(enhanced_np, (w, h))

    return img_lr_disp, enhanced_np, img_np

def main():
    st.title("🩻 Chest X-Ray Enhancement Pipeline")
    st.markdown("""
    **Pipeline Stages:**
    1.  **Input:** Low-resolution, noisy X-ray image (64x64).
    2.  **Stage 1:** DnCNN Denoising.
    3.  **Stage 2:** Real-ESRGAN Super-Resolution (4x).
    4.  **Output:** Clean, High-resolution X-ray (256x256).
    """)
    
    denoiser, sr_model, device = load_models()
    
    if denoiser is None or sr_model is None:
        st.warning("Please place the .pth model files in the 'models/' directories to proceed.")
        return

    st.sidebar.header("Options")
    option = st.sidebar.radio("Input Source", ["Upload Image", "Use Test Sample"])

    input_image = None
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
    else:
        # Load a random sample from local test set
        test_dir = os.path.join("data", "original", "test")
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
            if files:
                selected_file = st.sidebar.selectbox("Select a test image", files)
                input_image = Image.open(os.path.join(test_dir, selected_file))
            else:
                st.error("No images found in data/original/test")
        else:
            st.error("Data directory not found.")

    if input_image is not None:
        st.write("---")
        
        if st.button("Run Enhancement Pipeline"):
            with st.spinner("Running AI models..."):
                # We treat the input_image as the 'Ground Truth' / High-Quality original
                # And we generate the degraded version on the fly to show the restoration capability
                degraded, enhanced, original = process_image(input_image, denoiser, sr_model, device)
                
                # Metrics
                psnr_val = calculate_psnr(original, enhanced)
                ssim_val = calculate_ssim(original, enhanced)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("1. Degraded Input")
                    st.image(degraded, caption="Low Res (64px) + Noise", use_column_width=True, clamp=True)
                    
                with col2:
                    st.subheader("2. AI Enhanced Output")
                    st.image(enhanced, caption="Restored (256px)", use_column_width=True, clamp=True)
                    
                with col3:
                    st.subheader("3. Ground Truth")
                    st.image(original, caption="Original High Res", use_column_width=True, clamp=True)
                
                st.success(f"**Results:** PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
                
                # Difference map
                diff = cv2.absdiff(original, enhanced)
                diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
                st.image(diff, caption="Error Map (Blue = Low Error, Red = High Error)", clamp=True)

if __name__ == "__main__":
    main()
