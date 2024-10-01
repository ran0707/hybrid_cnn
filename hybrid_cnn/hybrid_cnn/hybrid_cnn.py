import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block (from ResNet)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out

# Squeeze-and-Excitation (SE) Block for Attention
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch_size, channels, 1, 1)
        x = x * excitation
        return x

# Hybrid CNN Model with Residual Blocks and Attention
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res_block1 = ResidualBlock(128, 256, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
        ))
        self.res_block2 = ResidualBlock(256, 512, downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
        ))

        # SE block
        self.se_block = SEBlock(512)

        # Adaptive pooling to make the output size fixed
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)  # Changed to 512 to match the output of Adaptive Pooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.se_block(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten the output from convolution layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
