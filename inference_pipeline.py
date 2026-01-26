import torch
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet

def run_inference():
    print("Running Inference Pipeline on ALL Sets (Train/Val/Test)...")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load Models
    print("Loading models...")
    
    # 1. Denoiser
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(DEVICE)
    denoiser_path = os.path.join("models", "denoiser", "dncnn_best.pth")
    if os.path.exists(denoiser_path):
        denoiser.load_state_dict(torch.load(denoiser_path, map_location=DEVICE))
        print("✅ Denoiser loaded.")
    else:
        print(f"❌ Denoiser weights not found at {denoiser_path}")
        return

    # 2. Super-Resolution
    sr_model = RRDBNet(in_channels=1, out_channels=1).to(DEVICE)
    sr_path = os.path.join("models", "super_resolution", "generator_latest.pth")
    if os.path.exists(sr_path):
        sr_model.load_state_dict(torch.load(sr_path, map_location=DEVICE))
        print("✅ SR Model loaded.")
    else:
        print(f"❌ SR weights not found at {sr_path}")
        return

    denoiser.eval()
    sr_model.eval()

    # Process all splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} set...")
        degraded_dir = os.path.join("data", "degraded", split)
        enhanced_dir = os.path.join("data", "enhanced", split)
        os.makedirs(enhanced_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(degraded_dir, "*.png"))
        if not files:
            print(f"No files found in {degraded_dir}")
            continue

        for f in tqdm(files, desc=f"Enhancing {split}"):
            filename = os.path.basename(f)
            
            # Load Degraded Image (64x64)
            img = Image.open(f).convert('L')
            img_np = np.array(img)
            
            # Preprocess
            tensor = torch.from_numpy(img_np / 255.0).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Step 1: Denoise
                cleaned_tensor = denoiser(tensor)
                
                # Step 2: Super-Resolution (x4)
                enhanced_tensor = sr_model(cleaned_tensor)
                
            # Post-process
            enhanced_np = enhanced_tensor.squeeze().cpu().numpy()
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
            
            # Save
            save_path = os.path.join(enhanced_dir, filename)
            cv2.imwrite(save_path, enhanced_np)

    print(f"\n✅ Inference complete! All datasets enriched.")

if __name__ == "__main__":
    run_inference()
