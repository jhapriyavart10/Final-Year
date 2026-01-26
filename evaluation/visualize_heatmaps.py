import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmaps(num_samples=10):
    # Paths
    degraded_dir = os.path.join("data", "degraded", "test")
    enhanced_dir = os.path.join("data", "enhanced", "test")
    output_dir = os.path.join("evaluation", "heatmaps")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files
    if not os.path.exists(enhanced_dir):
        print(f"Error: Enhanced directory not found: {enhanced_dir}")
        return

    files = glob.glob(os.path.join(enhanced_dir, "*.png"))
    if not files:
        print("No enhanced images found. Run inference_pipeline.py first.")
        return

    # Shuffle to get random samples
    np.random.shuffle(files)
    selected_files = files[:num_samples]
    
    print(f"Generating {len(selected_files)} heatmaps in {output_dir}...")
    
    for f_path in selected_files:
        filename = os.path.basename(f_path)
        deg_path = os.path.join(degraded_dir, filename)
        
        if not os.path.exists(deg_path):
            continue
            
        # Read images
        # Enhanced (256x256)
        img_enh = cv2.imread(f_path)
        img_enh = cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB)
        
        # Degraded (64x64)
        img_deg = cv2.imread(deg_path)
        img_deg = cv2.cvtColor(img_deg, cv2.COLOR_BGR2RGB)
        
        # Resize degraded to match enhanced for comparison (Input View)
        img_deg_resized = cv2.resize(img_deg, (img_enh.shape[1], img_enh.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Calculate Difference (Gray)
        gray_enh = cv2.cvtColor(img_enh, cv2.COLOR_RGB2GRAY)
        gray_deg = cv2.cvtColor(img_deg_resized, cv2.COLOR_RGB2GRAY)
        
        # Absolute difference: | Enhanced - Degraded |
        diff = cv2.absdiff(gray_enh, gray_deg)
        
        # Normalize diff for visualization (Contrast Stretch)
        diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Apply colormap (JET: Blue=Low Diff, Red=High Diff)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # 1. Degraded (Resized)
        axes[0].imshow(img_deg_resized)
        axes[0].set_title("Degraded Input (Upscaled)")
        axes[0].axis('off')
        
        # 2. Enhanced
        axes[1].imshow(img_enh)
        axes[1].set_title("Enhanced Output")
        axes[1].axis('off')
        
        # 3. Difference (Gray)
        axes[2].imshow(diff, cmap='gray')
        axes[2].set_title("Diff Magnitude")
        axes[2].axis('off')
        
        # 4. Heatmap
        axes[3].imshow(heatmap)
        axes[3].set_title("Change Heatmap (Blue->Red)")
        axes[3].axis('off')
        
        plt.suptitle(f"Enhancement Visualization: {filename}", fontsize=16)
        plt.tight_layout()
        
        save_file = os.path.join(output_dir, f"heatmap_{filename}")
        plt.savefig(save_file)
        plt.close()
        
        print(f"Saved: {save_file}")

if __name__ == "__main__":
    generate_heatmaps()
