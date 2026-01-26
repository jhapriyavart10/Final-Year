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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.super_resolution.rrdbnet import RRDBNet
from models.super_resolution.discriminator import UNetDiscriminator

class SRDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.degraded_dir = os.path.join(root_dir, 'degraded', split)
        self.original_dir = os.path.join(root_dir, 'original', split)
        self.files = glob.glob(os.path.join(self.degraded_dir, "*.png"))
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        deg_path = self.files[idx]
        filename = os.path.basename(deg_path)
        orig_path = os.path.join(self.original_dir, filename)
        
        # Degraded 64x64
        deg_img = Image.open(deg_path).convert('L')
        # Original 256x256
        orig_img = Image.open(orig_path).convert('L')
        
        return self.transform(deg_img), self.transform(orig_img)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # Use first few layers for perceptual loss on grayscale images
        # We need to replicate grayscale to 3 channels
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        # Repeat channels to match VGG input
        sr_3c = sr.repeat(1, 3, 1, 1)
        hr_3c = hr.repeat(1, 3, 1, 1)
        loss = nn.MSELoss()(self.vgg_layers(sr_3c), self.vgg_layers(hr_3c))
        return loss

def train_sr():
    print("Training script for Super Resolution (Real-ESRGAN - Simplified)")
    
    BATCH_SIZE = 8
    EPOCHS = 10
    LR_G = 1e-4
    LR_D = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Data
    root_dir = os.path.join("data")
    if not os.path.exists(os.path.join(root_dir, 'degraded', 'train')):
        print("Error: Dataset not found.")
        return

    train_dataset = SRDataset(root_dir, split='train')
    val_dataset = SRDataset(root_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Models
    generator = RRDBNet(in_channels=1, out_channels=1).to(DEVICE)
    discriminator = UNetDiscriminator(in_channels=1).to(DEVICE)
    
    # Losses
    criterion_pixel = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    
    try:
        criterion_perceptual = PerceptualLoss().to(DEVICE)
        use_perceptual = True
    except:
        print("Warning: Could not load VGG for perceptual loss. Using only Pixel Loss.")
        use_perceptual = False

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D)
    
    save_dir = os.path.join("models", "super_resolution")
    os.makedirs(save_dir, exist_ok=True)
    
    # Training Loop
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        g_loss_accum = 0.0
        d_loss_accum = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (imgs_lr, imgs_hr) in enumerate(loop):
            imgs_lr = imgs_lr.to(DEVICE)
            imgs_hr = imgs_hr.to(DEVICE)
            
            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            
            # Generate HR
            gen_hr = generator(imgs_lr)
            
            # Pixel Loss
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            
            # Perceptual Loss
            loss_perceptual = criterion_perceptual(gen_hr, imgs_hr) if use_perceptual else 0.0
            
            # Adversarial Loss (Generator wants Discriminator to think it's Real)
            pred_fake = discriminator(gen_hr)
            # Label smoothing (real=1.0)
            valid = torch.ones_like(pred_fake).to(DEVICE)
            loss_gan = criterion_gan(pred_fake, valid)
            
            # Total Generator Loss
            # Weighted sum: 1.0 * Pixel + 0.01 * Perceptual + 0.005 * GAN (tuned for stability)
            loss_G = loss_pixel + (0.1 * loss_perceptual) + (0.005 * loss_gan)
            
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())
            
            valid = torch.ones_like(pred_real).to(DEVICE)
            fake = torch.zeros_like(pred_fake).to(DEVICE)
            
            loss_real = criterion_gan(pred_real, valid)
            loss_fake = criterion_gan(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            optimizer_D.step()
            
            g_loss_accum += loss_G.item()
            d_loss_accum += loss_D.item()
            
            loop.set_postfix(g_loss=loss_G.item(), d_loss=loss_D.item())
            
        print(f"Epoch {epoch+1} done. Avg G_Loss: {g_loss_accum/len(train_loader):.4f}")
        
        # Save checkpoints
        torch.save(generator.state_dict(), os.path.join(save_dir, "generator_latest.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator_latest.pth"))

if __name__ == "__main__":
    train_sr()
