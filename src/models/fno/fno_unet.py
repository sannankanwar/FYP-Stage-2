"""
FNO-UNet: U-Net architecture with Fourier Neural Operator in bottleneck.
Outputs pooled features for regression (not segmentation).
"""
import torch
import torch.nn as nn
from src.models.layers.fno import FNOBlock
from src.models.layers.scaled_output import HybridScaledOutput


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling: MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsampling: Upsample + Concat Skip + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + out_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FNOUNet(nn.Module):
    """
    U-Net with FNO in bottleneck for global context.
    Outputs 5 parameters via global pooling (regression mode).
    """
    def __init__(self, in_channels=2, output_dim=5, modes=32, fno_norm='instance', fno_activation='gelu',
                 xc_range=(-500, 500), yc_range=(-500, 500), fov_range=(1, 20),
                 wavelength_range=(0.4, 0.7), focal_length_range=(10, 100)):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck with FNO
        self.down4 = Down(512, 512)
        self.fno = FNOBlock(channels=512, modes=modes, norm=fno_norm, activation=fno_activation)
        
        # Decoder
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        # Scaled output layer
        self.scaled_output = HybridScaledOutput(
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    
    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck with FNO
        x5 = self.down4(x4)
        x5 = self.fno(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Regression
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.scaled_output(x)
        
        return x
