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

        # Scale factor for initialization: 1/in_channels is more stable for large layers
        self.scale = (1 / in_channels)
        
        # Learnable complex weights
        # Learnable complex weights
        # We only learn the lower frequency modes (low-pass filtering inductive bias)
        # Shape: [in, out, modes1, modes2]
        # AMP Fix 2: Split into Real and Imaginary parts (Float32) to satisfy GradScaler
        
        scale_val = self.scale
        # Weights 1
        w1_init = scale_val * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        self.weights1_real = nn.Parameter(w1_init.real.float())
        self.weights1_imag = nn.Parameter(w1_init.imag.float())

        # Weights 2
        w2_init = scale_val * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        self.weights2_real = nn.Parameter(w2_init.real.float())
        self.weights2_imag = nn.Parameter(w2_init.imag.float())

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y) -> (batch, in_channel, x, y)
        # AMP Fix: ComplexHalf is not supported for einsum/baddbmm on CUDA.
        # We force execution in float32.
        with torch.cuda.amp.autocast(enabled=False):
            return torch.einsum("bixy,ioxy->boxy", input.cfloat(), weights.cfloat())

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
        
        # Reconstruct weights from Real/Imag parts
        weights1 = torch.complex(self.weights1_real, self.weights1_imag)
        weights2 = torch.complex(self.weights2_real, self.weights2_imag)
        
        # Upper-Left corner (Low freqs in H, Low freqs in W)
        out_ft[:, :, :m1, :m2] = \
            self.compl_mul2d(x_ft[:, :, :m1, :m2], weights1[:, :, :m1, :m2])
            
        # Lower-Left corner (High freqs in H - aliased negative, Low freqs in W)
        out_ft[:, :, -m1:, :m2] = \
            self.compl_mul2d(x_ft[:, :, -m1:, :m2], weights2[:, :, :m1, :m2])

        # 3. IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x
