"""
FNO-VGG19: VGG19 backbone with Fourier Neural Operator integration.
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
from src.models.layers.fno import FNOBlock


class FNOVGG19(nn.Module):
    """
    VGG19 backbone with FNO block for global spectral mixing.
    
    Architecture:
        Input (2, H, W) → VGG features[:18] → FNO Block → VGG features[18:] → Pool → MLP → 5 params
    """
    def __init__(self, in_channels=2, output_dim=5, modes=32, fno_norm='instance', fno_activation='gelu'):
        super().__init__()
        
        # Load VGG19 with batch norm
        base = vgg19_bn(weights=None)
        
        # VGG19_bn features breakdown:
        # 0-6: 64 ch, /2
        # 7-13: 128 ch, /4
        # 14-26: 256 ch, /8
        # 27-39: 512 ch, /16
        # 40-52: 512 ch, /32
        
        # Modify first conv for 2-channel input
        features = list(base.features)
        features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # First part of VGG (up to 256 channels, /8)
        self.features_pre = nn.Sequential(*features[:27])  # Up to 256 ch
        
        # FNO Block at 256 channels
        self.fno = FNOBlock(channels=256, modes=modes, norm=fno_norm, activation=fno_activation)
        
        # Second part of VGG (512 channels onwards)
        self.features_post = nn.Sequential(*features[27:])  # 512 ch, /32
        
        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        # First feature extraction block
        x = self.features_pre(x)
        
        # FNO global mixing
        x = self.fno(x)
        
        # Second feature extraction block
        x = self.features_post(x)
        
        # Regression
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
