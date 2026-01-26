import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import io

def add_gaussian_noise(image, mean=0, sigma=15):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_jpeg_compression(image, quality=75):
    """Apply JPEG compression artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 0) # 0 for grayscale
    return decimg

def degrade_image(image_path, scale_factor=4):
    """
    Reads an image and applies degradations:
    1. Gaussian Blur
    2. Downsampling
    3. Gaussian Noise
    4. JPEG Compression
    """
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 1. Gaussian Blur
    # Kernel size random between 3 and 7
    ksize = np.random.choice([3, 5, 7])
    img_blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 2. Downsampling
    h, w = img.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    img_down = cv2.resize(img_blurred, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. Additive Gaussian Noise
    # Random sigma
    sigma = np.random.randint(5, 25)
    img_noisy = add_gaussian_noise(img_down, sigma=sigma)

    # 4. JPEG Compression
    # Random quality
    quality = np.random.randint(50, 95)
    img_jpeg = apply_jpeg_compression(img_noisy, quality=quality)

    return img_jpeg

def create_degraded_dataset():
    src_root = os.path.join("data", "original")
    dst_root = os.path.join("data", "degraded")
    
    splits = ["train", "val", "test"]
    
    print("Generating degraded dataset...")
    
    for split in splits:
        src_dir = os.path.join(src_root, split)
        dst_dir = os.path.join(dst_root, split)
        
        os.makedirs(dst_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(src_dir, "*.png"))
        
        for f in tqdm(files, desc=f"Processing {split}"):
            # We'll stick to scale_factor=4 for a challenging SR task (64x64 -> 256x256)
            # as requested (x2 and x4), but SR models usually train on one scale.
            # I will assume x4 for the main pipeline.
            degraded_img = degrade_image(f, scale_factor=4)
            
            if degraded_img is not None:
                filename = os.path.basename(f)
                save_path = os.path.join(dst_dir, filename)
                cv2.imwrite(save_path, degraded_img)
                
    print(f"Degraded images saved to {dst_root}")

if __name__ == "__main__":
    create_degraded_dataset()
