"""
Swin Transformer for Metalens Parameter Inversion.
Adapted for regression task with 2-channel phase map input.
"""
import torch
import torch.nn as nn
from functools import partial


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=256, patch_size=4, in_chans=2, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Create position index
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij')).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer Block with window attention."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Partition windows
        x = self._window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        x = self.attn(x)
        
        # Reverse windows
        x = x.view(-1, self.window_size, self.window_size, C)
        x = self._window_reverse(x, self.window_size, H, W)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return x
    
    def _window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer (downsampling)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer for Metalens Inversion.
    
    Args:
        img_size: Input resolution (assumes square)
        patch_size: Patch size for tokenization
        in_chans: Input channels (2 for cos/sin phase)
        output_dim: Output parameters (5)
        embed_dim: Base embedding dimension
        depths: Number of blocks per stage
        num_heads: Number of attention heads per stage
        window_size: Window size for local attention
    """
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=2,
        output_dim=5,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        patches_resolution = img_size // patch_size
        self.patches_resolution = patches_resolution
        
        # Stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = embed_dim * (2 ** i_layer)
            resolution = patches_resolution // (2 ** i_layer)
            
            # Swin blocks for this stage
            blocks = nn.ModuleList([
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=min(window_size, resolution),
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio
                )
                for j in range(depths[i_layer])
            ])
            
            # Downsampling (except last stage)
            downsample = PatchMerging(dim) if i_layer < self.num_layers - 1 else None
            
            self.layers.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': downsample
            }))
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim * (2 ** (self.num_layers - 1)))
        
        # Regression head
        final_dim = embed_dim * (2 ** (self.num_layers - 1))
        self.head = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Bound output to [-1, 1] for normalized targets
        )
    
    def forward(self, x):
        x = self.patch_embed(x)
        H = W = self.patches_resolution
        
        for i_layer, layer in enumerate(self.layers):
            for block in layer['blocks']:
                x = block(x, H, W)
            
            if layer['downsample'] is not None:
                x = layer['downsample'](x, H, W)
                H = H // 2
                W = W // 2
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x
