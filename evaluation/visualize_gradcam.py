import os
import cv2
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier.resnet import Classifier

class SimpleGradCAM:
    """
    A simple implementation of Gradient-weighted Class Activation Mapping (Grad-CAM)
    for PyTorch models without needing external dependencies.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        # Use full_backward_hook for newer PyTorch versions
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        outputs = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()
            
        self.model.zero_grad()
        score = outputs[0, class_idx]
        score.backward()
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        # ReLU to only keep positive influences
        cam = F.relu(cam)
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().data.numpy(), class_idx

def overlay_cam_on_image(img_np, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlays the Grad-CAM heatmap onto the original image."""
    # Resize cam to match image dimensions
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    # Convert cam to uint8 0-255
    cam_uint8 = np.uint8(255 * cam_resized)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert grayscale image to RGB for overlay
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_np
        
    # Blend image and heatmap
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay

def generate_gradcam_visualizations(num_samples=10):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating Grad-CAM visualisations using {DEVICE}...")

    # Load Model
    model = Classifier(num_classes=2, in_channels=1).to(DEVICE)
    # We load the enhanced classifier weights to see what it learned
    model_path = os.path.join("models", "classifier", "classifier_enhanced.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "classifier", "classifier_degraded.pth")
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Classifier weights not found. Train classifier first.")
        return

    # Target the last convolutional layer of ResNet18
    target_layer = model.model.layer4[-1].conv2
    cam_extractor = SimpleGradCAM(model, target_layer)

    # Directories
    original_dir = os.path.join("data", "original", "test")
    enhanced_dir = os.path.join("data", "enhanced", "test")
    output_dir = os.path.join("evaluation", "gradcam")
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(enhanced_dir, "*.png"))
    if not files:
        print("No enhanced images found. Run inference_pipeline.py first.")
        return

    np.random.shuffle(files)
    selected_files = files[:num_samples]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    class_names = {0: "Normal", 1: "Abnormal"}

    for f_path in selected_files:
        filename = os.path.basename(f_path)
        orig_path = os.path.join(original_dir, filename)
        
        if not os.path.exists(orig_path):
             continue

        # Load images
        img_enh_pil = Image.open(f_path).convert('L')
        img_orig_pil = Image.open(orig_path).convert('L')
        
        # Prepare Tensors
        tensor_enh = transform(img_enh_pil).unsqueeze(0).to(DEVICE)
        tensor_orig = transform(img_orig_pil).unsqueeze(0).to(DEVICE)

        # Generate CAM for Enhanced
        cam_enh, pred_enh = cam_extractor(tensor_enh)
        # Generate CAM for Original
        cam_orig, pred_orig = cam_extractor(tensor_orig)

        # Convert PIL to Numpy for drawing (resize to 224x224 to match CAM input space visually)
        img_enh_np = np.array(img_enh_pil.resize((224, 224)))
        img_orig_np = np.array(img_orig_pil.resize((224, 224)))

        # Create Overlays
        overlay_enh = overlay_cam_on_image(img_enh_np, cam_enh)
        overlay_orig = overlay_cam_on_image(img_orig_np, cam_orig)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(overlay_orig)
        axes[0].set_title(f"Original (Pred: {class_names.get(pred_orig, 'Unk')})")
        axes[0].axis('off')
        
        axes[1].imshow(overlay_enh)
        axes[1].set_title(f"Enhanced (Pred: {class_names.get(pred_enh, 'Unk')})")
        axes[1].axis('off')

        plt.suptitle(f"Grad-CAM (Explainable AI) Focus Regions: {filename}", fontsize=14)
        plt.tight_layout()

        save_file = os.path.join(output_dir, f"gradcam_{filename}")
        plt.savefig(save_file)
        plt.close()
        
        print(f"Saved Grad-CAM: {save_file}")

if __name__ == "__main__":
    generate_gradcam_visualizations()
