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

# Add project root to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.degraded_dir = os.path.join(root_dir, 'degraded', split)
        self.clean_dir = os.path.join(root_dir, 'original', split)
        self.files = glob.glob(os.path.join(self.degraded_dir, "*.png"))
        
        # Transform: ToTensor converts 0-255 to 0.0-1.0
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        deg_path = self.files[idx]
        filename = os.path.basename(deg_path)
        clean_path = os.path.join(self.clean_dir, filename)
        
        # Load images
        # Degraded is already 64x64
        deg_img = Image.open(deg_path).convert('L')
        
        # Clean is 256x256, we need to resize it to 64x64 for DnCNN training
        # because DnCNN maps Noisy(H,W) -> Clean(H,W)
        clean_img = Image.open(clean_path).convert('L')
        clean_img = clean_img.resize(deg_img.size, Image.Resampling.BICUBIC)
        
        deg_tensor = self.transform(deg_img)
        clean_tensor = self.transform(clean_img)
        
        return deg_tensor, clean_tensor

def train_denoiser():
    print("Training script for Denoiser (DnCNN)")
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Data setup
    # In Colab/PyTorch, relative paths depend on where script is run from.
    # Assuming run from project root.
    root_dir = os.path.join("data")
    
    # Basic check to ensure data exists
    if not os.path.exists(os.path.join(root_dir, 'degraded', 'train')):
        print("Error: Dataset not found. Please run setup_data.py and create_degraded_data.py first.")
        return

    train_dataset = DenoisingDataset(root_dir, split='train')
    val_dataset = DenoisingDataset(root_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model setup
    # depth=17 is standard, but we'll use depth=10 for "lightweight"
    model = DnCNN(depth=10, n_channels=64, image_channels=1).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    save_dir = os.path.join("models", "denoiser")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dncnn_best.pth")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for noisy, clean in loop:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val loss: {best_val_loss:.6f}")

    print("Training complete!")

if __name__ == "__main__":
    train_denoiser()
