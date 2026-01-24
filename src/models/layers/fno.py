"""
FNO Block - Wrapper around SpectralGating2d with normalization and skip connection.
This provides a clean interface for FNO integration into various backbones.
"""
import torch
import torch.nn as nn
from src.models.layers.spectral import SpectralGating2d


class FNOBlock(nn.Module):
    """
    Fourier Neural Operator Block.
    Combines spectral convolution with normalization and residual connection.
    """
    def __init__(self, channels, modes=32, norm='instance', activation='gelu'):
        """
        Args:
            channels: Number of input/output channels
            modes: Number of Fourier modes to keep (per dimension)
            norm: Normalization type ('instance', 'batch', 'layer', 'none')
            activation: Activation function ('gelu', 'relu', 'silu')
        """
        super().__init__()
        
        self.spectral = SpectralGating2d(
            in_channels=channels,
            out_channels=channels,
            modes1=modes,
            modes2=modes
        )
        
        # Normalization
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(channels)
        elif norm == 'layer':
            self.norm = nn.GroupNorm(1, channels)  # LayerNorm for 2D
        else:
            self.norm = nn.Identity()
        
        # Activation
        act_map = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(inplace=True),
            'silu': nn.SiLU(inplace=True),
        }
        self.activation = act_map.get(activation, nn.GELU())
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        x: (B, C, H, W)
        """
        residual = x
        x = self.spectral(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x + residual  # Skip connection
        return x
