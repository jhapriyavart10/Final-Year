import torch
import torch.nn as nn
from torchvision import models

class Classifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super(Classifier, self).__init__()
        # Use ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first layer for grayscale (1 channel)
        # ResNet conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify last layer
        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, num_classes)
        
    def forward(self, x):
        return self.model(x)
