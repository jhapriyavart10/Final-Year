import os
import glob
from PIL import Image
import random

def verify_data():
    base_dir = os.path.join("data", "original")
    splits = ["train", "val", "test"]
    
    total_images = 0
    
    print(f"{'Split':<10} | {'Count':<10} | {'Shape':<15} | {'Mode':<10}")
    print("-" * 55)
    
    for split in splits:
        dir_path = os.path.join(base_dir, split)
        if not os.path.exists(dir_path):
            print(f"{split:<10} | {'MISSING':<10} | {'-':<15} | {'-':<10}")
            continue
            
        files = glob.glob(os.path.join(dir_path, "*.png"))
        count = len(files)
        total_images += count
        
        sample_shape = "N/A"
        sample_mode = "N/A"
        
        if count > 0:
            # Check a random sample
            sample_file = random.choice(files)
            try:
                with Image.open(sample_file) as img:
                    sample_shape = str(img.size)
                    sample_mode = img.mode
            except Exception as e:
                sample_shape = "Error"
        
        print(f"{split:<10} | {count:<10} | {sample_shape:<15} | {sample_mode:<10}")

    print("-" * 55)
    print(f"Total images found: {total_images}")
    
    # Validation against requirements
    if total_images >= 1000 and total_images <= 3000:
        print("✅ Total image count in range (1000-3000).")
    else:
        print("❌ Total image count out of range.")
        
    # Check splits (approx 70/15/15)
    if total_images > 0:
        train_count = len(glob.glob(os.path.join(base_dir, "train", "*.png")))
        val_count = len(glob.glob(os.path.join(base_dir, "val", "*.png")))
        test_count = len(glob.glob(os.path.join(base_dir, "test", "*.png")))
        
        print(f"Split proportions: Train: {train_count/total_images:.2%}, Val: {val_count/total_images:.2%}, Test: {test_count/total_images:.2%}")

if __name__ == "__main__":
    verify_data()
