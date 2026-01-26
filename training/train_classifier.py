import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob
from tqdm import tqdm
import sys
import pandas as pd
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier.resnet import Classifier

class CXRDataset(Dataset):
    def __init__(self, root_dir, dataset_type='original', split='train', labels_file="data/sample_labels.csv"):
        """
        dataset_type: 'original', 'degraded', or 'enhanced'
        reduce_classes: If True, maps to Binary (0: No Finding, 1: Disease)
        """
        self.img_dir = os.path.join(root_dir, dataset_type, split)
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Directory not found: {self.img_dir}")
            
        self.files = glob.glob(os.path.join(self.img_dir, "*.png"))
        
        # Load Labels
        if os.path.exists(labels_file):
            self.labels_df = pd.read_csv(labels_file)
            # Create map: filename -> label
            self.label_map = {}
            for idx, row in self.labels_df.iterrows():
                fname = row['Image Index']
                findings = row['Finding Labels']
                label = 0 if findings == 'No Finding' else 1
                self.label_map[fname] = label
        else:
            print(f"Warning: Labels file {labels_file} not found. Using Dummy labels.")
            self.label_map = {}

        # Filter files to only those present in the directory AND the CSV (if map exists)
        if self.label_map:
            self.valid_files = [f for f in self.files if os.path.basename(f) in self.label_map]
        else:
            self.valid_files = self.files

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # ResNet standard input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_path = self.valid_files[idx]
        filename = os.path.basename(img_path)
        
        if self.label_map:
            label = self.label_map[filename]
        else:
            label = 0 # Dummy
        
        img = Image.open(img_path).convert('L')
        img_t = self.transform(img)
        
        return img_t, torch.tensor(label, dtype=torch.long)

def train_model(dataset_type='degraded', epochs=10):
    print(f"Training Classifier on {dataset_type.upper()} images...")
    
    BATCH_SIZE = 32
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    root_dir = os.path.join("data")
    
    try:
        train_dataset = CXRDataset(root_dir, dataset_type=dataset_type, split='train')
        val_dataset = CXRDataset(root_dir, dataset_type=dataset_type, split='val')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model: ResNet18 (1 channel input, 2 classes)
    model = Classifier(num_classes=2, in_channels=1).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    save_dir = os.path.join("models", "classifier")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"classifier_{dataset_type}.pth")
    
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        train_acc = 100.*correct/total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100.*val_correct/val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model ({val_acc:.2f}%) to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="degraded", choices=["original", "degraded", "enhanced"], help="Dataset type to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    
    train_model(dataset_type=args.type, epochs=args.epochs)
