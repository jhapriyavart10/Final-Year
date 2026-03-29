import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier.resnet import Classifier

def find_incorrect_predictions():
    print("Searching for images with incorrect predictions after enhancement...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loads Labels
    labels_file = os.path.join("data", "sample_labels.csv")
    if not os.path.exists(labels_file):
        print(f"Labels file missing: {labels_file}")
        return
        
    df = pd.read_csv(labels_file)
    label_map = {}
    for _, row in df.iterrows():
        fname = row['Image Index']
        finding = row['Finding Labels']
        label = 0 if finding == 'No Finding' else 1
        label_map[fname] = label
        
    # Directories
    orig_dir = os.path.join("data", "original", "test")
    deg_dir = os.path.join("data", "degraded", "test")
    enh_dir = os.path.join("data", "evaluation_outputs", "3_full_pipeline") # Run compare_configurations.py before this
    
    # Load Classifier
    classifier = Classifier(num_classes=2, in_channels=1).to(device)
    cls_path = os.path.join("models", "classifier", "classifier_enhanced.pth")
    if not os.path.exists(cls_path):
         cls_path = os.path.join("models", "classifier", "classifier_degraded.pth")
    if not os.path.exists(cls_path):
        print("No classifier models found.")
        return
        
    classifier.load_state_dict(torch.load(cls_path, map_location=device))
    classifier.eval()
    
    def predict(img_path):
        img = Image.open(img_path).convert('L')
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        t_img = t(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier(t_img)
            _, preds = torch.max(outputs, 1)
        return preds.item()

    incorrect_cases = []
    
    # Valid files
    import glob
    files = glob.glob(os.path.join(enh_dir, "*.png"))
    
    if not files:
        print(f"No enhanced images found in {enh_dir}. Run evaluation/compare_configurations.py first.")
        return
        
    for f in files:
        filename = os.path.basename(f)
        if filename not in label_map:
            continue
            
        true_label = label_map[filename]
        pred_label = predict(f) # Check prediction on the enhanced image
        
        if true_label != pred_label:
            incorrect_cases.append({
                "filename": filename,
                "true": true_label,
                "pred": pred_label
            })
            
        if len(incorrect_cases) >= 5:
            break
            
    if not incorrect_cases:
        print("Could not find any incorrect predictions! The classifier is extremely accurate.")
        return
        
    print(f"Found {len(incorrect_cases)} incorrect predictions. Plotting & Saving...")
    
    # Display & Save side-by-side
    out_img_dir = os.path.join("evaluation", "failure_cases")
    os.makedirs(out_img_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(incorrect_cases), 3, figsize=(15, 5 * len(incorrect_cases)))
    # Handle case where only 1 row is returned
    if len(incorrect_cases) == 1:
        axes = [axes]
        
    classes = {0: "Normal", 1: "Abnormal"}
        
    for idx, case in enumerate(incorrect_cases):
        filename = case['filename']
        true_class = classes[case['true']]
        pred_class = classes[case['pred']]
        
        p_orig = os.path.join(orig_dir, filename)
        p_deg = os.path.join(deg_dir, filename)
        p_enh = os.path.join(enh_dir, filename)
        
        orig_img = cv2.imread(p_orig, cv2.IMREAD_GRAYSCALE)
        deg_img = cv2.imread(p_deg, cv2.IMREAD_GRAYSCALE)
        enh_img = cv2.imread(p_enh, cv2.IMREAD_GRAYSCALE)
        
        # Ensure dimensions match for horizontal stacking (resize to original shape)
        target_h, target_w = orig_img.shape
        if deg_img.shape != orig_img.shape:
             deg_img = cv2.resize(deg_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if enh_img.shape != orig_img.shape:
             enh_img = cv2.resize(enh_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Save combination output image
        combined_img = np.hstack((deg_img, enh_img, orig_img))
        combined_out_path = os.path.join(out_img_dir, f"failure_{filename}")
        cv2.imwrite(combined_out_path, combined_img)
        
        # Plot to Matplotlib graph
        axes[idx][0].imshow(deg_img, cmap='gray')
        axes[idx][0].set_title(f"Degraded Input", fontsize=14)
        axes[idx][0].axis('off')
        
        axes[idx][1].imshow(enh_img, cmap='gray')
        axes[idx][1].set_title(f"Enhanced Mismatch\nPred: {pred_class} | True: {true_class}", fontsize=14, color='red')
        axes[idx][1].axis('off')
        
        axes[idx][2].imshow(orig_img, cmap='gray')
        axes[idx][2].set_title(f"Original GT", fontsize=14)
        axes[idx][2].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(out_img_dir, "failure_cases_grid.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved failure cases visual grid to: {plot_path}")
    print(f"Saved side-by-side joined image files individually to: {out_img_dir}")
    
if __name__ == "__main__":
    find_incorrect_predictions()