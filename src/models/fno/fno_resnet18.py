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
                 xc_range=(-500, 500), yc_range=(-500, 500), S_range=(1, 40),
                 wavelength_range=(0.4, 0.7), focal_length_range=(10, 100)):
        super().__init__()
        
        # Resolution Adaptation
        # FNO backbone is designed for 256x256. If input is larger, downsample first.
        self.target_resolution = 256
        self.downsampler = None
        
        if input_resolution > self.target_resolution:
            factor = input_resolution // self.target_resolution
            print(f"FNOResNet18: Adapting input {input_resolution}x{input_resolution} -> {self.target_resolution}x{self.target_resolution} (Learnable Scaled Conv)")
            
            # Learnable Downsampling (Convolutional Adaptation)
            # Example for 1024 -> 256 (4x downsampling)
            # We use 2 stages of stride 2 to preserve more information than a single stride 4 conv.
            
            # Stage 1: 1024 -> 512. Expand channels to capture texture.
            # Stage 2: 512 -> 256. Contract channels back to in_channels for ResNet stem.
            
            # Note: We avoid ReLU on the final output to allow negative values (like sin/cos inputs) to pass to ResNet stem.
            
            if factor == 4:
                # Custom Pyramid Downsampler (1024 -> 256)
                # 2 -> 8 -> 16 channels to preserve phase information
                self.downsampler = nn.Sequential(
                    # 1024 -> 512: 2 -> 8 channels
                    nn.Conv2d(in_channels, 8, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(8),
                    nn.ReLU(inplace=True),
                    
                    # 512 -> 256: 8 -> 16 channels
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                )
            else:
                # Fallback
                if factor == 2:
                     self.downsampler = nn.Sequential(
                        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True)
                    )
                else:
                    self.downsampler = nn.AvgPool2d(kernel_size=factor, stride=factor)
            
        # Load ResNet18 backbone
        base = resnet18(weights=None)
        
        # Modified stem for 16-channel input (from downsampler)
        # If no downsampler (256x256 input), we project 2->16 first or just handle 2 channels?
        # If resolution=256, downsampler is None, input is 2 channels.
        # If resolution=1024, downsampler output is 16 channels.
        # We need to handle both.
        
        stem_in_channels = 16 if self.downsampler is not None else in_channels
        
        self.conv1 = nn.Conv2d(stem_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
            xc_range=xc_range, yc_range=yc_range, S_range=S_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
        
        # EXPLICIT CONTRACT: Model outputs physical units (microns)
        # Downstream tools verify this before attempting denormalization.
        self.output_space = "physical"
    
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
