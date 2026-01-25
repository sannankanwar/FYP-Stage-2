"""
FNO-ResNet18: ResNet18 backbone with Fourier Neural Operator integration.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from src.models.layers.fno import FNOBlock
from src.models.layers.scaled_output import HybridScaledOutput


class FNOResNet18(nn.Module):
    """
    ResNet18 backbone with FNO block for global spectral mixing.
    
    Architecture:
        Input (2, H, W) → ResNet Stem → layer1-3 → FNO Block → layer4 → Pool → MLP → 5 params
    """
    def __init__(self, in_channels=2, output_dim=5, modes=32, fno_norm='instance', fno_activation='gelu',
                 input_resolution=256,
                 xc_range=(-500, 500), yc_range=(-500, 500), fov_range=(1, 20),
                 wavelength_range=(0.4, 0.7), focal_length_range=(10, 100)):
        super().__init__()
        
        # Resolution Adaptation
        # FNO backbone is designed for 256x256. If input is larger, downsample first.
        self.target_resolution = 256
        self.downsampler = None
        
        if input_resolution > self.target_resolution:
            factor = input_resolution // self.target_resolution
            print(f"FNOResNet18: Adapting input {input_resolution}x{input_resolution} -> {self.target_resolution}x{self.target_resolution} (AvgPool k={factor})")
            self.downsampler = nn.AvgPool2d(kernel_size=factor, stride=factor)
            
        # Load ResNet18 backbone
        base = resnet18(weights=None)
        
        # Modified stem for 2-channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # Feature extraction layers
        self.layer1 = base.layer1  # 64 ch, /4
        self.layer2 = base.layer2  # 128 ch, /8
        self.layer3 = base.layer3  # 256 ch, /16
        
        # FNO Block (global mixing at 256 channels)
        self.fno = FNOBlock(channels=256, modes=modes, norm=fno_norm, activation=fno_activation)
        
        # Final downsampling
        self.layer4 = base.layer4  # 512 ch, /32
        
        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        # Scaled output layer (outputs raw physical values)
        self.scaled_output = HybridScaledOutput(
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    
    def forward(self, x):
        # Adaptation
        if self.downsampler is not None:
            x = self.downsampler(x)

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
