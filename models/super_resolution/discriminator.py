import torch
import torch.nn as nn

class UNetDiscriminator(nn.Module):
    """
    A lightweight U-Net based discriminator for Real-ESRGAN/SRGAN.
    Simplification: Using a standard VGG-style discriminator instead 
    of full U-Net Discriminator for reduced complexity in this template.
    """
    def __init__(self, in_channels=1, base_channels=64):
        super(UNetDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(discriminator_block(in_channels, base_channels, first_block=True))
        layers.extend(discriminator_block(base_channels, base_channels * 2))
        layers.extend(discriminator_block(base_channels * 2, base_channels * 4))
        layers.extend(discriminator_block(base_channels * 4, base_channels * 8))

        self.model = nn.Sequential(*layers)
        
        # Global Average Pooling then Dense
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(base_channels * 8, 1, 1)

    def forward(self, x):
        features = self.model(x)
        out = self.avg_pool(features)
        out = self.conv_last(out)
        return out.view(out.size(0), -1)  # (Batch, 1)
