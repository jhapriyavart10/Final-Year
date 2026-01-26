import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self, numberOfChannels=64, growthRate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(numberOfChannels, growthRate, 3, 1, 1)
        self.conv2 = nn.Conv2d(numberOfChannels + growthRate, growthRate, 3, 1, 1)
        self.conv3 = nn.Conv2d(numberOfChannels + 2 * growthRate, growthRate, 3, 1, 1)
        self.conv4 = nn.Conv2d(numberOfChannels + 3 * growthRate, growthRate, 3, 1, 1)
        self.conv5 = nn.Conv2d(numberOfChannels + 4 * growthRate, numberOfChannels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, numberOfChannels, growthRate=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(numberOfChannels, growthRate)
        self.RDB2 = ResidualDenseBlock(numberOfChannels, growthRate)
        self.RDB3 = ResidualDenseBlock(numberOfChannels, growthRate)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, numberOfChannels=64, numberOfRRDB=6):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, numberOfChannels, 3, 1, 1)
        self.RRDB_trunk = make_layer(RRDB, numberOfRRDB, numberOfChannels=numberOfChannels)
        self.trunk_conv = nn.Conv2d(numberOfChannels, numberOfChannels, 3, 1, 1)
        
        # Upsampling
        self.upconv1 = nn.Conv2d(numberOfChannels, numberOfChannels, 3, 1, 1)
        self.upconv2 = nn.Conv2d(numberOfChannels, numberOfChannels, 3, 1, 1)
        self.HRconv = nn.Conv2d(numberOfChannels, numberOfChannels, 3, 1, 1)
        self.conv_last = nn.Conv2d(numberOfChannels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        # Up 1 (x2)
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # Up 2 (x4)
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out
