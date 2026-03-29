import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet

def evaluate_configurations():
    print("Evaluating Three Configurations:")
    print("1. Degraded (Bicubic Upsampling)")
    print("2. SR Only (Real-ESRGAN without DnCNN)")
    print("3. Full Pipeline (DnCNN + Real-ESRGAN)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(device)
    denoiser_path = os.path.join("models", "denoiser", "dncnn_best.pth")
    if os.path.exists(denoiser_path):
        denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
        denoiser.eval()
    else:
        print(f"Warning: Denoiser weights not found at {denoiser_path}")
        return

    sr_model = RRDBNet(in_channels=1, out_channels=1).to(device)
    sr_path = os.path.join("models", "super_resolution", "generator_latest.pth")
    if os.path.exists(sr_path):
        sr_model.load_state_dict(torch.load(sr_path, map_location=device))
        sr_model.eval()
    else:
        print(f"Warning: SR Model weights not found at {sr_path}")
        return

    # Directories
    original_dir = os.path.join("data", "original", "test")
    degraded_dir = os.path.join("data", "degraded", "test")
    
    out_deg_dir = os.path.join("data", "evaluation_outputs", "1_degraded_bicubic")
    out_sr_only_dir = os.path.join("data", "evaluation_outputs", "2_sr_only")
    out_full_dir = os.path.join("data", "evaluation_outputs", "3_full_pipeline")
    
    os.makedirs(out_deg_dir, exist_ok=True)
    os.makedirs(out_sr_only_dir, exist_ok=True)
    os.makedirs(out_full_dir, exist_ok=True)
    
    import glob
    files = glob.glob(os.path.join(original_dir, "*.png"))
    
    if not files:
        print("No test files found in data/original/test.")
        return

    results = []
    processed_dataset = []

    print(f"Evaluating {len(files)} images...")
    
    with torch.no_grad():
        for f in tqdm(files):
            filename = os.path.basename(f)
            
            orig_path = f
            deg_path = os.path.join(degraded_dir, filename)
            
            if not os.path.exists(deg_path):
                continue
                
            orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            deg = cv2.imread(deg_path, cv2.IMREAD_GRAYSCALE)
            
            if orig is None or deg is None:
                continue
            
            target_h, target_w = orig.shape
            
            # 1. Config 1: Degraded (Bicubic Upsampling)
            bicubic = cv2.resize(deg, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Prepare tensor for models
            input_tensor = torch.from_numpy(deg / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # 2. Config 2: SR Only
            sr_only_tensor = sr_model(input_tensor)
            sr_only_np = sr_only_tensor.squeeze().cpu().numpy()
            sr_only_np = np.clip(sr_only_np * 255.0, 0, 255).astype(np.uint8)
            if sr_only_np.shape != orig.shape:
                 sr_only_np = cv2.resize(sr_only_np, (target_w, target_h))
                 
            # 3. Config 3: Full Pipeline
            cleaned_tensor = denoiser(input_tensor)
            full_tensor = sr_model(cleaned_tensor)
            full_np = full_tensor.squeeze().cpu().numpy()
            full_np = np.clip(full_np * 255.0, 0, 255).astype(np.uint8)
            if full_np.shape != orig.shape:
                 full_np = cv2.resize(full_np, (target_w, target_h))
            
            # Save Outputs
            cv2.imwrite(os.path.join(out_deg_dir, filename), bicubic)
            cv2.imwrite(os.path.join(out_sr_only_dir, filename), sr_only_np)
            cv2.imwrite(os.path.join(out_full_dir, filename), full_np)
            
            # Calculate metrics
            psnr_1 = psnr(orig, bicubic, data_range=255)
            ssim_1 = ssim(orig, bicubic, data_range=255)
            
            psnr_2 = psnr(orig, sr_only_np, data_range=255)
            ssim_2 = ssim(orig, sr_only_np, data_range=255)
            
            psnr_3 = psnr(orig, full_np, data_range=255)
            ssim_3 = ssim(orig, full_np, data_range=255)
            
            results.append({
                "Filename": filename,
                "PSNR_1_Degraded": psnr_1,
                "SSIM_1_Degraded": ssim_1,
                "PSNR_2_SROnly": psnr_2,
                "SSIM_2_SROnly": ssim_2,
                "PSNR_3_Full": psnr_3,
                "SSIM_3_Full": ssim_3
            })
            
            processed_dataset.append({
                "Filename": filename,
                "Original": orig,
                "Config_1": bicubic,
                "Config_2": sr_only_np,
                "Config_3": full_np
            })
            
    if not results:
        print("No valid image pairs found.")
        return processed_dataset

    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("               CONFIGURATION COMPARISON TABLE")
    print("="*70)
    
    summary = pd.DataFrame({
        "Configuration": ["1. Degraded (Bicubic)", "2. SR Only", "3. Full Pipeline"],
        "Average PSNR (dB)": [
            df['PSNR_1_Degraded'].mean(),
            df['PSNR_2_SROnly'].mean(),
            df['PSNR_3_Full'].mean()
        ],
        "Average SSIM": [
            df['SSIM_1_Degraded'].mean(),
            df['SSIM_2_SROnly'].mean(),
            df['SSIM_3_Full'].mean()
        ]
    })
    
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print("="*70)
    
    df.to_csv("evaluation/configuration_comparison.csv", index=False)
    print("Detailed results saved to evaluation/configuration_comparison.csv")
    print(f"Output images saved to data/evaluation_outputs/")
    
    return processed_dataset

if __name__ == "__main__":
    evaluate_configurations()
