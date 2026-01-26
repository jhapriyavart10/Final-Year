import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import sys
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier.resnet import Classifier

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        # NOTE: In a real scenario, you would parse a CSV for labels.
        # Here, since we only have images without explicit class folders or CSV in the setup,
        # we will create DUMMY labels just to make the training loop runnable and functional.
        # User should replace this logic with actual label loading.
        
        self.img_dir = os.path.join(root_dir, 'original', split)
        self.files = glob.glob(os.path.join(self.img_dir, "*.png"))
        
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('L')
        img_t = self.to_tensor(img)
        img_t = self.transform(img_t)
        
        # Dummy Label Generation: 
        # Random binary label (0: Normal, 1: Abnormal) 
        # In production, use: label = self.labels[os.path.basename(img_path)]
        label = random.randint(0, 1) 
        
        return img_t, torch.tensor(label, dtype=torch.long)

def train_classifier():
    print("Training script for Classifier (ResNet)")
    
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    root_dir = os.path.join("data")
    if not os.path.exists(os.path.join(root_dir, 'original', 'train')):
        print("Error: Dataset not found.")
        return

    train_dataset = ClassificationDataset(root_dir, split='train')
    val_dataset = ClassificationDataset(root_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2 Classes (Binary)
    model = Classifier(num_classes=2, in_channels=1).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    save_dir = os.path.join("models", "classifier")
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
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
            
        print(f"Epoch {epoch+1} Train Acc: {100.*correct/total:.2f}%")
        
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
        
        print(f"Epoch {epoch+1} Val Acc: {100.*val_correct/val_total:.2f}%")
        
        torch.save(model.state_dict(), os.path.join(save_dir, "classifier_latest.pth"))

if __name__ == "__main__":
    train_classifier()
