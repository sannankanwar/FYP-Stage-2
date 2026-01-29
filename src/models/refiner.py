import torch
import torch.nn as nn
import torchvision.models as models

class ResNetRefiner(nn.Module):
    """
    Residual-Guided Parameter Refiner.
    
    Inputs:
        - Image Stream (B, 4, H, W): [cos_phi, sin_phi, res_cos, res_sin]
        - Conditioning (B, D_cond): Normalized parameter guess
        
    Outputs:
        - Delta Theta Norm (B, D_out): Normalized correction
    """
    def __init__(self, input_channels=4, condition_dim=5, output_dim=5, hidden_dim=64):
        super().__init__()
        
        # 1. Backbone: ResNet-18
        # We replace the first conv layer to accept 'input_channels'
        self.backbone = models.resnet18(weights=None)
        
        original_first_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 
            original_first_conv.out_channels, 
            kernel_size=original_first_conv.kernel_size, 
            stride=original_first_conv.stride, 
            padding=original_first_conv.padding, 
            bias=original_first_conv.bias
        )
        
        # Remove the final FC layer, we want the features before it
        # ResNet18 fc is linear(512 -> 1000). We replace it with Identity or handle forward manually.
        # Let's handle forward manually to tap into avgpool.
        del self.backbone.fc
        
        self.feature_dim = 512
        
        # 2. Conditioning Projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # 3. Fusion & Prediction Head
        # Concatenating Feature (512) + Condition (64) = 576
        fusion_dim = self.feature_dim + hidden_dim
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim)
        )
        
        # Initialize head to output near-zero
        # This helps stability at start (delta starts small)
        nn.init.constant_(self.head[-1].bias, 0.0)
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.001)

    def forward(self, img_input, condition_vec):
        """
        Args:
            img_input: (B, 4, H, W)
            condition_vec: (B, D_cond) normalized params
        """
        # Backbone Forward
        x = self.backbone.conv1(img_input)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        img_features = torch.flatten(x, 1) # (B, 512)
        
        # Conditioning
        cond_features = self.condition_proj(condition_vec) # (B, 64)
        
        # Fusion
        fused = torch.cat([img_features, cond_features], dim=1) # (B, 576)
        
        # Prediction
        delta = self.head(fused)
        
        # Clamping for stability (Steering Limit)
        # 2.0 sigma should be plenty for a correction
        delta = torch.clamp(delta, -2.0, 2.0)
        
        return delta
