import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    img1, img2: numpy arrays of shape (H, W) or (H, W, C).
    """
    # Ensure images are in correct range/type if needed
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index).
    img1, img2: numpy arrays of shape (H, W) or (H, W, C).
    """
    return ssim(img1, img2, data_range=255)

