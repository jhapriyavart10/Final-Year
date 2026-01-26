import torch
import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet
from evaluation.metrics import calculate_psnr, calculate_ssim

def tensor_to_numpy(tensor):
    """Convert (1, C, H, W) tensor to (H, W) numpy array in range [0, 255]."""
    img = tensor.squeeze().cpu().detach().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def evaluate_pipeline():
    print("Evaluating Full Pipeline: Input -> Denoiser -> SR -> Output")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Paths
    test_degraded_dir = os.path.join("data", "degraded", "test")
    test_original_dir = os.path.join("data", "original", "test")
    
    # Load Models
    print("Loading models...")
    
    # 1. Denoiser
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(DEVICE)
    denoiser_path = os.path.join("models", "denoiser", "dncnn_best.pth")
    if os.path.exists(denoiser_path):
        denoiser.load_state_dict(torch.load(denoiser_path, map_location=DEVICE))
        print("✅ Denoiser weights loaded.")
    else:
        print("❌ Denoiser weights not found! Skipping evaluation.")
        return

    # 2. Super-Resolution
    sr_model = RRDBNet(in_channels=1, out_channels=1).to(DEVICE)
    sr_path = os.path.join("models", "super_resolution", "generator_latest.pth")
    if os.path.exists(sr_path):
        sr_model.load_state_dict(torch.load(sr_path, map_location=DEVICE))
        print("✅ SR Model weights loaded.")
    else:
        print("❌ SR Model weights not found! Skipping evaluation.")
        return

    denoiser.eval()
    sr_model.eval()

    files = glob.glob(os.path.join(test_degraded_dir, "*.png"))
    if not files:
        print("No test files found.")
        return

    psnr_values = []
    ssim_values = []

    print(f"Testing on {len(files)} images...")
    
    for f in tqdm(files):
        filename = os.path.basename(f)
        orig_path = os.path.join(test_original_dir, filename)
        
        # Load Images
        # Degraded (64x64)
        deg_img = Image.open(f).convert('L')
        deg_tensor = torch.from_numpy(np.array(deg_img) / 255.0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Original (256x256)
        orig_img = Image.open(orig_path).convert('L')
        orig_numpy = np.array(orig_img)

        with torch.no_grad():
            # Step 1: Denoise
            # Note: DnCNN expects noise to be added to input, and returns noise. 
            # Denoised = Input - PredictedNoise
            # Or if trained as Input->Clean, then direct output.
            # My training script used: loss(output, clean), so model predicts CLEAN image directly.
            # Wait, let's double check DnCNN implementation.
            # In dncnn.py: return x - out. (Residual learning).
            # So model predicts NOISE. Denoised = Input - Noise.
            # Correct.
            
            denoised_tensor = denoiser(deg_tensor)
            
            # Step 2: Super-Resolution (x4)
            sr_tensor = sr_model(denoised_tensor)
            
        # Convert to Numpy for metrics
        sr_numpy = tensor_to_numpy(sr_tensor)
        
        # Ensure sizes match (handling potential slight padding issues if any, though 64*4=256 should match)
        if sr_numpy.shape != orig_numpy.shape:
            sr_numpy = cv2.resize(sr_numpy, (orig_numpy.shape[1], orig_numpy.shape[0]))

        # Calculate Metrics
        p = calculate_psnr(orig_numpy, sr_numpy)
        s = calculate_ssim(orig_numpy, sr_numpy)
        
        psnr_values.append(p)
        ssim_values.append(s)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print("\nSame-Domain Evaluation Results:")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    evaluate_pipeline()
