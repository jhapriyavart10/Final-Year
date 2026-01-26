import torch.nn as nn
import torch

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, 
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Intermediate layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, 
                                    kernel_size=kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, momentum=0.9, eps=1e-04))
            layers.append(nn.ReLU(inplace=True))

        # Final layer
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        # Residual learning: Prediction is the noise, so Output = Input - Noise
        return x - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
