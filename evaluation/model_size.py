import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet
from models.classifier.resnet import Classifier

def calculate_model_sizes():
    print("Calculating Model Sizes...")
    print("====================================")
    
    # 1. Initialize models (using default architecture parameters)
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1)
    sr_model = RRDBNet(in_channels=1, out_channels=1)
    classifier = Classifier(num_classes=2, in_channels=1)
    
    models = {
        "DnCNN_Denoiser": denoiser,
        "RealESRGAN_SR": sr_model,
        "ResNet18_Classifier": classifier
    }
    
    total_size_mb = 0
    temp_dir = "temp_weights"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Print header
    print(f"{'Model Component':<25} | {'Parameters':<12} | {'File Size (MB)':<12}")
    print("-" * 55)
    
    for name, model in models.items():
        # Calculate number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        formatted_params = f"{total_params:,}"
        
        # Save dummy weights to measure actual disk footprint
        temp_path = os.path.join(temp_dir, f"{name}.pth")
        torch.save(model.state_dict(), temp_path)
        
        # Measure file size
        file_size_bytes = os.path.getsize(temp_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        total_size_mb += file_size_mb
        
        print(f"{name:<25} | {formatted_params:<12} | {file_size_mb:>9.2f} MB")
        
        # Clean up temporary file
        os.remove(temp_path)
        
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
        
    print("====================================")
    print(f"{'Total Pipeline Footprint':<25} | {'':<12} | {total_size_mb:>9.2f} MB")
    print("====================================")

if __name__ == "__main__":
    calculate_model_sizes()