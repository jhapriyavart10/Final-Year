import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob

def compare_results():
    print("COMPARISON: Degraded vs Original | Enhanced vs Original")
    
    original_dir = os.path.join("data", "original", "test")
    degraded_dir = os.path.join("data", "degraded", "test")
    enhanced_dir = os.path.join("data", "enhanced", "test")
    
    files = glob.glob(os.path.join(original_dir, "*.png"))
    
    if not files:
        print("No test files found.")
        return

    results = []

    print(f"Evaluating {len(files)} images...")
    
    for f in tqdm(files):
        filename = os.path.basename(f)
        
        orig_path = f
        deg_path = os.path.join(degraded_dir, filename)
        enh_path = os.path.join(enhanced_dir, filename)
        
        # Skip if enhanced doesn't exist (inference not run)
        if not os.path.exists(enh_path):
            continue
            
        if not os.path.exists(deg_path):
            continue
            
        # Load images
        orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        deg = cv2.imread(deg_path, cv2.IMREAD_GRAYSCALE)
        enh = cv2.imread(enh_path, cv2.IMREAD_GRAYSCALE)
        
        if orig is None or deg is None or enh is None:
            continue
        
        # Resize degraded to match original for valid comparison
        # This simulates "Bicubic Upsampling" baseline
        deg_resized = cv2.resize(deg, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Ensure enhanced matches original size
        if enh.shape != orig.shape:
             enh = cv2.resize(enh, (orig.shape[1], orig.shape[0]))

        # Calculate metrics
        # 1. Before (Degraded vs Original)
        psnr_before = psnr(orig, deg_resized, data_range=255)
        ssim_before = ssim(orig, deg_resized, data_range=255)
        
        # 2. After (Enhanced vs Original)
        psnr_after = psnr(orig, enh, data_range=255)
        ssim_after = ssim(orig, enh, data_range=255)
        
        results.append({
            "Filename": filename,
            "PSNR_Before": psnr_before,
            "SSIM_Before": ssim_before,
            "PSNR_After": psnr_after,
            "SSIM_After": ssim_after
        })
        
    if not results:
        print("No matching triplets found. Ensure inference has been run and data/enhanced/test is populated.")
        return

    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS TABLE")
    print("="*60)
    
    # Summary Table
    summary = pd.DataFrame({
        "Metric": ["PSNR (dB)", "SSIM"],
        "Before (Bicubic)": [df['PSNR_Before'].mean(), df['SSIM_Before'].mean()],
        "After (Enhanced)": [df['PSNR_After'].mean(), df['SSIM_After'].mean()],
        "Improvement": [
            df['PSNR_After'].mean() - df['PSNR_Before'].mean(),
            df['SSIM_After'].mean() - df['SSIM_Before'].mean()
        ]
    })
    
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print("="*60)
    
    # Save detailed results
    df.to_csv("evaluation/results_table.csv", index=False)
    print("Detailed results saved to evaluation/results_table.csv")

if __name__ == "__main__":
    compare_results()
