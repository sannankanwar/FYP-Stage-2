import torch
import torch.nn as nn
import torch.fft

class SpectralGating2d(nn.Module):
    """
    Spectral Gating Layer.
     computes FFT2D, modulates freq components with learnable complex weights, then IFFT2D.
    
    This provides global receptice field mixing without N^2 complexity of Attention.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            modes1 (int): Number of Fourier modes to keep along dim 1 (height)
            modes2 (int): Number of Fourier modes to keep along dim 2 (width)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Scale factor for initialization
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable complex weights
        # We only learn the lower frequency modes (low-pass filtering inductive bias)
        # Shape: [in, out, modes1, modes2]
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y) -> (batch, in_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. FFT
        # x shape: [B, C, H, W]
        # x_ft shape: [B, C, H, W//2 + 1] (Real-valued FFT)
        x_ft = torch.fft.rfft2(x)

        # 2. Spectral Gating (Filtering)
        # We assume we only modify the lowest 'modes' frequencies.
        # This acts as a global convolution.
        
        # Init output spectrum
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Clip modes if input resolution is smaller than expected modes
        m1 = min(self.modes1, x_ft.size(-2) // 2)
        m2 = min(self.modes2, x_ft.size(-1))
        
        # Upper-Left corner (Low freqs in H, Low freqs in W)
        out_ft[:, :, :m1, :m2] = \
            self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
            
        # Lower-Left corner (High freqs in H - aliased negative, Low freqs in W)
        out_ft[:, :, -m1:, :m2] = \
            self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # 3. IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x
