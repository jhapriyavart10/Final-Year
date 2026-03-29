import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_all_metrics():
    print("Generating comprehensive metrics comparison plot...")
    
    config_csv = os.path.join("evaluation", "configuration_comparison.csv")
    class_csv = os.path.join("evaluation", "classifier_comparison_table.csv")
    
    if not os.path.exists(config_csv) or not os.path.exists(class_csv):
        print("Missing required CSV files. Please run compare_configurations.py and evaluate_classifier.py first.")
        return
        
    # Read Image Quality Metrics
    df_img = pd.read_csv(config_csv)
    psnr_degraded = df_img['PSNR_1_Degraded'].mean()
    ssim_degraded = df_img['SSIM_1_Degraded'].mean()
    
    psnr_esrgan = df_img['PSNR_2_SROnly'].mean()
    ssim_esrgan = df_img['SSIM_2_SROnly'].mean()
    
    psnr_full = df_img['PSNR_3_Full'].mean()
    ssim_full = df_img['SSIM_3_Full'].mean()
    
    # Read Classifier Metrics
    df_cls = pd.read_csv(class_csv).set_index('Configuration')
    
    acc_degraded = df_cls.loc['Degraded', 'Accuracy']
    auc_degraded = df_cls.loc['Degraded', 'AUC']
    
    acc_esrgan = df_cls.loc['ESRGAN Only', 'Accuracy']
    auc_esrgan = df_cls.loc['ESRGAN Only', 'AUC']
    
    # Check if 'Full Pipeline' exists, else try to find closest match or leave safe default
    # Looking for exact match first
    try:
        acc_full = df_cls.loc['Full Pipeline', 'Accuracy']
        auc_full = df_cls.loc['Full Pipeline', 'AUC']
    except KeyError:
        acc_full, auc_full = 0, 0
    
    # Compile Data
    labels = ['Degraded (Baseline)', 'ESRGAN Only', 'Full Pipeline (DnCNN+ESRGAN)']
    
    psnr_vals = [psnr_degraded, psnr_esrgan, psnr_full]
    ssim_vals = [ssim_degraded, ssim_esrgan, ssim_full]
    acc_vals = [acc_degraded, acc_esrgan, acc_full]
    auc_vals = [auc_degraded, auc_esrgan, auc_full]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AI Pipeline Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    metrics = [
        ('PSNR', psnr_vals, axes[0, 0], 'dB', '#1f77b4'),
        ('SSIM', ssim_vals, axes[0, 1], 'Index (0-1)', '#ff7f0e'),
        ('Classification Accuracy', acc_vals, axes[1, 0], 'Accuracy (0-1)', '#2ca02c'),
        ('Classification AUC', auc_vals, axes[1, 1], 'Area Under Curve (0-1)', '#d62728')
    ]
    
    for title, vals, ax, ylabel, color in metrics:
        bars = ax.bar(labels, vals, color=color, alpha=0.85, edgecolor='black', width=0.6)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Smart Y-axis scaling
        min_val = min(vals)
        max_val = max(vals)
        range_val = max_val - min_val
        
        if 'PSNR' in title:
            ax.set_ylim(min_val - (range_val * 2), max_val + (range_val * 1.5))
        else:
            ax.set_ylim(0, 1.1)  # Scale between 0 and 1 for percentages
            
        # Annotate bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            
            # Format nicely depending on metric
            if 'PSNR' in title:
                text = f'{val:.2f}'
            else:
                text = f'{val:.4f}'
                
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    text, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    plot_path = os.path.join("evaluation", "all_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot successfully saved to {plot_path}")

if __name__ == "__main__":
    plot_all_metrics()