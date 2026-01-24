"""
GAN Inverter for Metalens Parameter Prediction.
Uses adversarial training to improve parameter prediction quality.

The discriminator learns to distinguish "good" predictions (that can reconstruct the input)
from "bad" predictions, providing a learned loss signal.
"""
import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    """
    Generator (Inverter): Phase Map → 5 Parameters
    
    Uses CNN encoder to extract features and predict parameters.
    This is essentially the same as our other inversion models.
    """
    def __init__(self, in_channels=2, output_dim=5):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 256 → 128
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 → 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 → 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 → 16
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 → 8
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 → 4
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Regressor
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output in [-1, 1] for normalized parameters
        )
    
    def forward(self, x):
        features = self.encoder(x)
        params = self.fc(features)
        return params


class Discriminator(nn.Module):
    """
    Discriminator: (Phase Map, Parameters) → Real/Fake
    
    Conditioned on the parameters. Learns to distinguish whether the
    parameters correctly describe the phase map.
    """
    def __init__(self, in_channels=2, param_dim=5, img_size=256):
        super().__init__()
        
        # Project parameters to spatial feature map
        self.param_proj = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, img_size * img_size),
        )
        self.img_size = img_size
        
        # Combined encoder (phase map + param map)
        self.encoder = nn.Sequential(
            # Input: 2 (phase) + 1 (param projection) = 3 channels
            nn.Conv2d(in_channels + 1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
            # No sigmoid - use with BCEWithLogitsLoss
        )
    
    def forward(self, phase_map, params):
        B = phase_map.size(0)
        
        # Project parameters to spatial map
        param_spatial = self.param_proj(params)
        param_spatial = param_spatial.view(B, 1, self.img_size, self.img_size)
        
        # Concatenate
        x = torch.cat([phase_map, param_spatial], dim=1)
        
        # Encode
        features = self.encoder(x)
        
        # Classify
        validity = self.fc(features)
        return validity


class GANInverter(nn.Module):
    """
    Combined GAN Inverter module.
    
    For inference, only the Generator is used.
    For training, both Generator and Discriminator are needed.
    """
    def __init__(self, in_channels=2, output_dim=5, img_size=256):
        super().__init__()
        
        self.generator = Generator(in_channels, output_dim)
        self.discriminator = Discriminator(in_channels, output_dim, img_size)
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Forward pass returns predicted parameters.
        For GAN training, use generator and discriminator separately.
        """
        return self.generator(x)
    
    def get_generator(self):
        return self.generator
    
    def get_discriminator(self):
        return self.discriminator
