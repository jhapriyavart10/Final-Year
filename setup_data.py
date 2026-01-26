import os
import glob
import shutil
import random
import numpy as np
from PIL import Image
import kagglehub
from tqdm import tqdm

def setup_data():
    print("Downloading NIH Chest X-ray Sample dataset...")
    try:
        # Download sample dataset (~2GB or less usually for the sample version)
        # Using nih-chest-xrays/sample which contains ~5606 images
        path = kagglehub.dataset_download("nih-chest-xrays/sample")
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # specific path handling for the sample dataset structure
    # usually path/sample/images/*.png or path/images/*.png
    source_images = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
    
    if not source_images:
        print("No images found in the downloaded dataset.")
        return

    print(f"Found {len(source_images)} images.")
    
    # Limit to 3000 images as requested (or less if fewer available)
    num_images_to_process = min(3000, len(source_images))
    selected_images = random.sample(source_images, num_images_to_process)
    print(f"Selected {len(selected_images)} images for processing.")

    # Define paths
    base_dir = os.path.join("data", "original")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Calculate split indices
    n = len(selected_images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # n_test is the remainder

    train_files = selected_images[:n_train]
    val_files = selected_images[n_train:n_train+n_val]
    test_files = selected_images[n_train+n_val:]

    print(f"Splitting into: Train ({len(train_files)}), Val ({len(val_files)}), Test ({len(test_files)})")

    def process_and_save(files, dest_dir):
        for src_path in tqdm(files, desc=f"Processing to {dest_dir}"):
            try:
                filename = os.path.basename(src_path)
                with Image.open(src_path) as img:
                    # Convert to grayscale
                    img_gray = img.convert('L')
                    # Resize to 256x256
                    img_resized = img_gray.resize((256, 256), Image.Resampling.LANCZOS)
                    
                    dest_path = os.path.join(dest_dir, filename)
                    img_resized.save(dest_path)
            except Exception as e:
                print(f"Failed to process {src_path}: {e}")

    process_and_save(train_files, train_dir)
    process_and_save(val_files, val_dir)
    process_and_save(test_files, test_dir)

    print("Data processing complete!")

if __name__ == "__main__":
    setup_data()
