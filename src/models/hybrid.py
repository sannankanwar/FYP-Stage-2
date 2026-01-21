import torch
import torch.nn as nn
from torchvision.models import resnet18
from src.models.layers.spectral import SpectralGating2d

class SpectralResNet(nn.Module):
    """
    Hybrid architecture: ResNet Stem + Spectral Gating + MLP Head.
    Optimized for high-resolution input (1024x1024) to predict metalens parameters.
    """
    def __init__(self, in_channels=2, modes=16):
        super().__init__()
        
        # 1. ResNet Backbone (Stem)
        # We load a standard resnet18
        base_model = resnet18(weights=None)
        
        # Modify first layer for 2-channel input (Cos/Sin phase)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool # /4 so far

        # Layer 1 (64 channels, /4)
        self.layer1 = base_model.layer1
        
        # Layer 2 (128 channels, /8)
        self.layer2 = base_model.layer2
        
        # Layer 3 (256 channels, /16)
        self.layer3 = base_model.layer3
        
        # At this point, 1024 -> 64. 
        # If we stop here, we have 64x64 feature maps.
        
        # 2. Spectral Layer (Global Mixer)
        # Input: [B, 256, 64, 64]
        # We mix globally here.
        self.spectral = SpectralGating2d(in_channels=256, out_channels=256, modes1=modes, modes2=modes)
        self.norm_spectral = nn.InstanceNorm2d(256)
        self.act_spectral = nn.GELU()

        # 3. Further Downsampling (Layer 4) -> /32
        self.layer4 = base_model.layer4
        
        # 4. Regressor Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3) # [xc, yc, fov]
        )

    def forward(self, x):
        # x: [B, 2, 1024, 1024]
        
        # CNN Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 256x256

        x = self.layer1(x)  # 256x256
        x = self.layer2(x)  # 128x128
        x = self.layer3(x)  # 64x64
        
        # Spectral Block (Global Context)
        # Skip connection around spectral block (optional but good for stability)
        x_spec = self.spectral(x)
        x_spec = self.norm_spectral(x_spec)
        x_spec = self.act_spectral(x_spec)
        x = x + x_spec 
        
        # Final Convolutional Stage
        x = self.layer4(x)  # 32x32, 512 channels
        
        # Regression Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
