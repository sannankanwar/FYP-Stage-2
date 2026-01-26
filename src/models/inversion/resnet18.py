import torch
import torch.nn as nn
from torchvision.models import resnet18

class InverseMetalensModel(nn.Module):
    def __init__(self, output_dim=3, input_channels=2):
        super().__init__()
        
        # We will add 2 extra channels (x_grid, y_grid) to the input
        self.input_channels = input_channels + 2 
        
        # We modify the first layer to accept our custom channel count (4 instead of 3)
        self.backbone = resnet18(weights=None) # weights=None replaces deprecated pretrained=False 
        
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
        
        # ResNet18's final FC layer has 512 input features
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim) # Output: [xc, yc, fov]
        )
        
        # EXPLICIT CONTRACT: Model learns physical units due to loss function constraint
        self.output_space = "physical"

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
        
        # Forward pass through ResNet
        pred_params = self.backbone(x_with_coords)
        
        return pred_params
