import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2

class InverseMetalensWideResNet50(nn.Module):
    def __init__(self, output_dim=3, input_channels=2):
        super().__init__()
        
        # 1. Coordinate Injection (CoordConv)
        # We will add 2 extra channels (x_grid, y_grid) to the input
        self.input_channels = input_channels + 2 
        
        # 2. Backbone: WideResNet50_2
        # We modify the first layer to accept our custom channel count (4 instead of 3)
        self.backbone = wide_resnet50_2(pretrained=False) 
        
        # Replace the first conv layer to handle (Cos, Sin, X, Y)
        # Standard ResNet: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.input_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 3. Regression Head
        # WideResNet50's final FC layer is named 'fc' and has 2048 input features (unlike 512 in ResNet18)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim) # Output: [xc, yc, fov]
        )

    def forward(self, x):
        """
        Args:
            x: (B, 2, H, W) -> The Cos/Sin phase profiles
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Generate Coordinate Grids on the fly (batch-wise)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device),
            torch.linspace(-0.5, 0.5, W, device=device),
            indexing='ij'
        )
        
        # Expand to batch size: (B, 1, H, W)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Concatenate: Result is (B, 4, H, W)
        x_with_coords = torch.cat([x, grid_x, grid_y], dim=1)
        
        # Forward pass through Backbone
        pred_params = self.backbone(x_with_coords)
        
        return pred_params
