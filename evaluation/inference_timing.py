import os
import sys
import time
import torch
import cv2
import numpy as np
import glob
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet

def measure_inference_time():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Measuring CPU/GPU Inference Time on: {device.type.upper()}")
    
    # 1. Load Models
    print("Loading models...")
    
    # Denoiser
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(device)
    dncnn_path = os.path.join("models", "denoiser", "dncnn_best.pth")
    if os.path.exists(dncnn_path):
        denoiser.load_state_dict(torch.load(dncnn_path, map_location=device))
    denoiser.eval()
    
    # Super-Resolution (ESRGAN)
    sr_model = RRDBNet(in_channels=1, out_channels=1).to(device)
    sr_path = os.path.join("models", "super_resolution", "generator_latest.pth")
    if os.path.exists(sr_path):
        sr_model.load_state_dict(torch.load(sr_path, map_location=device))
    sr_model.eval()
    
    # 2. Get 50 images
    degraded_dir = os.path.join("data", "degraded", "test")
    all_files = glob.glob(os.path.join(degraded_dir, "*.png"))
    
    if not all_files:
        print(f"No images found in {degraded_dir}")
        return
        
    # Limit to maximum of 50 images
    test_files = all_files[:50]
    num_images = len(test_files)
    print(f"Running inference timing benchmark over {num_images} images...\n")
    
    # Pre-load image tensors to exclude disk I/O from inference calculations
    tensors = []
    for f in test_files:
        img_np = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_np is not None:
             t = torch.from_numpy(img_np / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
             tensors.append(t)
             
    if not tensors:
         print("Failed to load any valid tensors. Exiting.")
         return

    # Trackers for accumulated time
    dncnn_total_time = 0.0
    esrgan_total_time = 0.0
    full_pipe_total_time = 0.0

    # 3. Warm-up (Ensures GPU memory allocations don't affect initial time tracking)
    print("Warming up models...")
    with torch.no_grad():
        dummy = tensors[0]
        _ = denoiser(dummy)
        _ = sr_model(dummy)
        
    # Sync GPU to ensure precise tracking
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 4. Measure Inference Time
    print("Executing Benchmark...")
    with torch.no_grad():
        for t in tqdm(tensors, desc="Processing Images"):
            
            # A. DnCNN Timing
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()
            _ = denoiser(t)
            if device.type == 'cuda': torch.cuda.synchronize()
            dncnn_total_time += (time.time() - start)
            
            # B. ESRGAN Timing (Isolated)
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()
            _ = sr_model(t)
            if device.type == 'cuda': torch.cuda.synchronize()
            esrgan_total_time += (time.time() - start)
            
            # C. Full Pipeline Timing (Sequential)
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()
            clean = denoiser(t)
            _ = sr_model(clean)
            if device.type == 'cuda': torch.cuda.synchronize()
            full_pipe_total_time += (time.time() - start)

    # 5. Calculate & Console Output Average Times
    num_tensors = len(tensors)
    avg_dncnn = (dncnn_total_time / num_tensors) * 1000      # converted to ms
    avg_esrgan = (esrgan_total_time / num_tensors) * 1000    # converted to ms
    avg_full = (full_pipe_total_time / num_tensors) * 1000   # converted to ms
    
    print("\n" + "="*50)
    print(f"INFERENCE TIME BENCHMARK (Average over {num_images} images)")
    print("="*50)
    print(f"{'Component':<25} | {'Avg Time per Image':<20}")
    print("-" * 50)
    print(f"{'1. DnCNN (Denoise)':<25} | {avg_dncnn:>8.2f} ms")
    print(f"{'2. Real-ESRGAN (SuperRes)':<25} | {avg_esrgan:>8.2f} ms")
    print(f"{'3. Full Pipeline':<25} | {avg_full:>8.2f} ms")
    print("="*50)
    
    # Frame Rate (FPS) equivalent comparison
    fps_full = 1000.0 / avg_full
    print(f"Equivalent Throughput: ~{fps_full:.1f} Frames Per Second (FPS)")
    
if __name__ == "__main__":
    measure_inference_time()