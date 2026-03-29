import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet
from models.classifier.resnet import Classifier

def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Models
    print("Loading AI Models...")
    
    # Denoiser
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(device)
    denoiser.load_state_dict(torch.load(os.path.join("models", "denoiser", "dncnn_best.pth"), map_location=device))
    denoiser.eval()
    
    # Super-Resolution (ESRGAN)
    sr_model = RRDBNet(in_channels=1, out_channels=1).to(device)
    sr_model.load_state_dict(torch.load(os.path.join("models", "super_resolution", "generator_latest.pth"), map_location=device))
    sr_model.eval()
    
    # Classifier
    classifier = Classifier(num_classes=2, in_channels=1).to(device)
    cls_path = os.path.join("models", "classifier", "classifier_enhanced.pth")
    if not os.path.exists(cls_path):
        cls_path = os.path.join("models", "classifier", "classifier_degraded.pth")
    classifier.load_state_dict(torch.load(cls_path, map_location=device))
    classifier.eval()
    
    # 2. Load Labels
    labels_file = os.path.join("data", "sample_labels.csv")
    label_map = {}
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        for _, row in labels_df.iterrows():
            fname = row['Image Index']
            finding = row['Finding Labels']
            label_map[fname] = 0 if finding == 'No Finding' else 1
            
    # 3. Process Data
    original_files = glob.glob(os.path.join("data", "original", "test", "*.png"))
    degraded_dir = os.path.join("data", "degraded", "test")
    
    results = {
        'Without DnCNN (ESRGAN Only)': {'psnr': [], 'ssim': [], 'probs': []},
        'Without ESRGAN (DnCNN Only)': {'psnr': [], 'ssim': [], 'probs': []},
        'Full Pipeline': {'psnr': [], 'ssim': [], 'probs': []}
    }
    
    gt_labels = []
    
    print(f"Running Ablation Study on {len(original_files)} test images...")
    
    def prep_for_classifier(img_np):
        """Prepare raw numpy (H, W) for ResNet Classifier"""
        img_resized = cv2.resize(img_np, (224, 224))
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - 0.5) / 0.5
        return torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        for f in tqdm(original_files, desc="Evaluating Images"):
            filename = os.path.basename(f)
            deg_path = os.path.join(degraded_dir, filename)
            
            if not os.path.exists(deg_path) or filename not in label_map:
                continue
                
            orig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            deg = cv2.imread(deg_path, cv2.IMREAD_GRAYSCALE)
            
            target_h, target_w = orig.shape
            
            # Prepare tensor for enhancement models
            input_tensor = torch.from_numpy(deg / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # --- Config 1: Without DnCNN (ESRGAN Only) ---
            esrgan_tensor = sr_model(input_tensor)
            esrgan_np = np.clip(esrgan_tensor.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if esrgan_np.shape != orig.shape:
                esrgan_np = cv2.resize(esrgan_np, (target_w, target_h))
                
            # --- Config 2: Without ESRGAN (DnCNN Only) ---
            dncnn_tensor = denoiser(input_tensor)
            dncnn_np = np.clip(dncnn_tensor.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if dncnn_np.shape != orig.shape:
                # Upsample via bicubic since SR model is omitted
                dncnn_np = cv2.resize(dncnn_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                
            # --- Config 3: Full Pipeline (DnCNN + ESRGAN) ---
            full_tensor = sr_model(dncnn_tensor)
            full_np = np.clip(full_tensor.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if full_np.shape != orig.shape:
                full_np = cv2.resize(full_np, (target_w, target_h))
                
            # Compute Metrics & Predictions
            for conf_name, img_out in zip(
                ['Without DnCNN (ESRGAN Only)', 'Without ESRGAN (DnCNN Only)', 'Full Pipeline'],
                [esrgan_np, dncnn_np, full_np]
            ):
                # PSNR / SSIM
                results[conf_name]['psnr'].append(psnr_fn(orig, img_out, data_range=255))
                results[conf_name]['ssim'].append(ssim_fn(orig, img_out, data_range=255))
                
                # Classification Probability
                outputs = classifier(prep_for_classifier(img_out))
                prob = torch.softmax(outputs, dim=1)[0, 1].item()
                results[conf_name]['probs'].append(prob)
                
            gt_labels.append(label_map[filename])

    # 4. Aggregate Results into DataFrame
    if not gt_labels:
        print("No valid test images matching labels were found.")
        return

    summary = {}
    for conf in results.keys():
        psnr_avg = np.mean(results[conf]['psnr'])
        ssim_avg = np.mean(results[conf]['ssim'])
        
        try:
            auc = roc_auc_score(gt_labels, results[conf]['probs'])
        except Exception as e:
            auc = 0.5
            print(f"Failed to compute AUC for {conf}: {e}")
            
        summary[conf] = {
            'PSNR': psnr_avg,
            'SSIM': ssim_avg,
            'AUC': auc
        }
        
    df = pd.DataFrame(summary).T
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS")
    print("="*50)
    print(df.to_string(float_format="{:.4f}".format))
    
    # Save CSV
    csv_path = os.path.join("evaluation", "ablation_study_results.csv")
    df.to_csv(csv_path, index_label="Configuration")
    print(f"\nDetailed numerical results saved to {csv_path}")
    
    # 5. Plotting Bar Graphs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Ablation Study: Impact of Pipeline Components', fontsize=16, fontweight='bold')
    
    configs_display = ['ESRGAN Only\n(No Denoising)', 'DnCNN Only\n(No Super-Res)', 'Full Pipeline']
    metrics = ['PSNR', 'SSIM', 'AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = df[metric].values
        
        bars = ax.bar(configs_display, vals, color=colors[i], alpha=0.8, edgecolor='black')
        ax.set_title(metric, fontsize=14)
        
        # Scale Y axis
        if metric == 'PSNR':
            ax.set_ylim(0, max(vals) * 1.15)
            ax.set_ylabel("dB")
        else:
            ax.set_ylim(0, 1.1)
            
        ax.grid(axis='y', linestyle='--', alpha=0.4)
            
        # Value annotations
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            offset = 0.5 if metric == 'PSNR' else 0.02
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        ax.set_xticks(range(len(configs_display)))
        ax.set_xticklabels(configs_display, fontsize=10)
        
    plt.tight_layout()
    plot_path = os.path.join("evaluation", "ablation_study_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar graph plot to {plot_path}")

if __name__ == '__main__':
    run_ablation()
