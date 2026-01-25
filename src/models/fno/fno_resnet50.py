"""
FNO-ResNet50: ResNet50 backbone with Fourier Neural Operator integration.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
from src.models.layers.fno import FNOBlock
from src.models.layers.scaled_output import HybridScaledOutput


class FNOResNet50(nn.Module):
    """
    ResNet50 backbone with FNO block for global spectral mixing.
    
    Architecture:
        Input (2, H, W) → ResNet Stem → layer1-3 → FNO Block → layer4 → Pool → MLP → 5 params
    """
    def __init__(self, in_channels=2, output_dim=5, modes=32, fno_norm='instance', fno_activation='gelu',
                 xc_range=(-500, 500), yc_range=(-500, 500), fov_range=(1, 20),
                 wavelength_range=(0.4, 0.7), focal_length_range=(10, 100)):
        super().__init__()
        
        # Load ResNet50 backbone
        base = resnet50(weights=None)
        
        # Modified stem for 2-channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # Feature extraction layers
        self.layer1 = base.layer1  # 256 ch, /4
        self.layer2 = base.layer2  # 512 ch, /8
        self.layer3 = base.layer3  # 1024 ch, /16
        
        # FNO Block (global mixing at 1024 channels)
        self.fno = FNOBlock(channels=1024, modes=modes, norm=fno_norm, activation=fno_activation)
        
        # Final downsampling
        self.layer4 = base.layer4  # 2048 ch, /32
        
        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
        # Scaled output layer (outputs raw physical values)
        self.scaled_output = HybridScaledOutput(
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # FNO global mixing
        x = self.fno(x)
        
        # Final conv stage
        x = self.layer4(x)
        
        # Regression
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # Scale to physical values
        x = self.scaled_output(x)
        
        return x
