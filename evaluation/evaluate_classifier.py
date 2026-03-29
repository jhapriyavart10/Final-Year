import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import pandas as pd
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classifier.resnet import Classifier

class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels_file="data/sample_labels.csv"):
        self.img_dir = img_dir
        self.files = glob.glob(os.path.join(self.img_dir, "*.png"))
        
        # Load Labels
        if os.path.exists(labels_file):
            self.labels_df = pd.read_csv(labels_file)
            self.label_map = {}
            for idx, row in self.labels_df.iterrows():
                fname = row['Image Index']
                findings = row['Finding Labels']
                label = 0 if findings == 'No Finding' else 1
                self.label_map[fname] = label
        else:
             self.label_map = {}

        if self.label_map:
            self.valid_files = [f for f in self.files if os.path.basename(f) in self.label_map]
        else:
            self.valid_files = [] 
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_path = self.valid_files[idx]
        filename = os.path.basename(img_path)
        label = self.label_map[filename]
        img = Image.open(img_path).convert('L')
        img_t = self.transform(img)
        return img_t, torch.tensor(label, dtype=torch.long)

def evaluate_model(model, dataset_dir, DEVICE):
    print(f"\nEvaluating Classifier on {dataset_dir} data...")
    BATCH_SIZE = 32
    
    # Test Data
    try:
        test_dataset = CXRDataset(img_dir=dataset_dir)
        if len(test_dataset) == 0:
            print(f"No valid test files or labels found in {dataset_dir}.")
            return None
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    all_labels = []
    all_preds_prob = []
    all_preds_cls = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs) # Logits
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Class 1 (Disease)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.numpy())
            all_preds_prob.extend(probs.cpu().numpy())
            all_preds_cls.extend(preds.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds_cls)
    f1 = f1_score(all_labels, all_preds_cls, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_preds_prob)
    except:
        auc = 0.5 # Fail-safe
        
    print(f"Results for {dataset_dir}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    
    return {"Accuracy": acc, "F1": f1, "AUC": auc}

def compare():
    print("Comparing classification performance across configurations...")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model (We'll use classifier_enhanced as the baseline evaluator, or fallback to degraded)
    model = Classifier(num_classes=2, in_channels=1).to(DEVICE)
    model_path = os.path.join("models", "classifier", "classifier_enhanced.pth")
    if not os.path.exists(model_path):
         model_path = os.path.join("models", "classifier", "classifier_degraded.pth")
    
    if os.path.exists(model_path):
         model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
         print("Warning: No classifier weights found.")
         return
         
    # Directories from the compare_configurations script 
    dirs_to_evaluate = {
        "Degraded": os.path.join("data", "evaluation_outputs", "1_degraded_bicubic"),
        "ESRGAN Only": os.path.join("data", "evaluation_outputs", "2_sr_only"),
        "Full Pipeline": os.path.join("data", "evaluation_outputs", "3_full_pipeline")
    }
    
    results = {}
    for name, path in dirs_to_evaluate.items():
        if os.path.exists(path):
            res = evaluate_model(model, path, DEVICE)
            if res:
                results[name] = res
        else:
            print(f"Directory {path} not found. Ensure compare_configurations.py has run first.")
            
    if results:
        print("\n" + "="*70)
        print("                 CLASSIFICATION PERFORMANCE COMPARISON")
        print("="*70)
        
        df = pd.DataFrame(results).T
        print(df.to_string(float_format="{:.4f}".format))
        print("="*70)
        
        # Save to CSV
        output_csv = os.path.join("evaluation", "classifier_comparison_table.csv")
        df.to_csv(output_csv, index_label="Configuration")
        print(f"\nClassification results saved to {output_csv}")

if __name__ == "__main__":
    compare()

