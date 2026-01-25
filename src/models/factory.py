from src.models.hybrid import SpectralResNet
from src.models.inversion.resnet18 import InverseMetalensModel
from src.models.inversion.wideresnet50 import InverseMetalensWideResNet50
from src.models.fno import FNOResNet18, FNOResNet50, FNOVGG19, FNOUNet
from src.models.transformer import SwinTransformer
from src.models.gan import GANInverter

def get_model(config):
    """
    Factory function to get model instance based on configuration.
    
    Args:
        config (dict): Model configuration dictionary containing attributes like:
                       - name: Model architecture name
                       - input_channels: int (default 2)
                       - output_dim: int (default 5)
                       - modes: int (for FNO models, default 32)
                       
    Available models:
        - spectral_resnet: Original hybrid SpectralResNet
        - resnet18: Basic ResNet18 with coordinate channels
        - wideresnet50: WideResNet50 backbone
        - fno_resnet18: ResNet18 + FNO block
        - fno_resnet50: ResNet50 + FNO block
        - fno_vgg19: VGG19 + FNO block
        - fno_unet: U-Net + FNO in bottleneck
        - swin: Swin Transformer
        - gan_inverter: GAN-based inverter
                       
    Returns:
        nn.Module: The requested model instance.
    """
    name = config.get("name", "spectral_resnet")
    input_channels = config.get("input_channels", 2)
    output_dim = config.get("output_dim", 5)
    modes = config.get("modes", 32)
    
    # Extract parameter ranges for scaled output
    xc_range = tuple(config.get("xc_range", [-500, 500]))
    yc_range = tuple(config.get("yc_range", [-500, 500]))
    fov_range = tuple(config.get("fov_range", [1, 20]))
    wavelength_range = tuple(config.get("wavelength_range", [0.4, 0.7]))
    focal_length_range = tuple(config.get("focal_length_range", [10, 100]))
    
    # FNO models
    if name == "fno_resnet18":
        model = FNOResNet18(
            in_channels=input_channels, 
            output_dim=output_dim, 
            modes=modes,
            fno_norm=config.get("fno_norm", "instance"),
            fno_activation=config.get("fno_activation", "gelu"),
            input_resolution=config.get("resolution", 256),
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    elif name == "fno_resnet50":
        model = FNOResNet50(
            in_channels=input_channels, 
            output_dim=output_dim, 
            modes=modes,
            fno_norm=config.get("fno_norm", "instance"),
            fno_activation=config.get("fno_activation", "gelu"),
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    elif name == "fno_vgg19":
        model = FNOVGG19(
            in_channels=input_channels, 
            output_dim=output_dim, 
            modes=modes,
            fno_norm=config.get("fno_norm", "instance"),
            fno_activation=config.get("fno_activation", "gelu"),
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    elif name == "fno_unet":
        model = FNOUNet(
            in_channels=input_channels, 
            output_dim=output_dim, 
            modes=modes,
            fno_norm=config.get("fno_norm", "instance"),
            fno_activation=config.get("fno_activation", "gelu"),
            xc_range=xc_range, yc_range=yc_range, fov_range=fov_range,
            wavelength_range=wavelength_range, focal_length_range=focal_length_range
        )
    
    # Transformer models
    elif name == "swin":
        img_size = config.get("resolution", 256)
        model = SwinTransformer(
            img_size=img_size,
            patch_size=config.get("patch_size", 4),
            in_chans=input_channels,
            output_dim=output_dim,
            embed_dim=config.get("embed_dim", 96),
            depths=config.get("depths", (2, 2, 6, 2)),
            num_heads=config.get("num_heads", (3, 6, 12, 24)),
            window_size=config.get("window_size", 8)
        )
    
    # GAN models
    elif name == "gan_inverter":
        img_size = config.get("resolution", 256)
        model = GANInverter(
            in_channels=input_channels,
            output_dim=output_dim,
            img_size=img_size
        )
    
    # Legacy models
    elif name == "spectral_resnet":
        model = SpectralResNet(in_channels=input_channels, modes=modes, output_dim=output_dim)
    elif name == "resnet18":
        model = InverseMetalensModel(output_dim=output_dim, input_channels=input_channels)
    elif name == "wideresnet50":
        model = InverseMetalensWideResNet50(output_dim=output_dim, input_channels=input_channels)
    else:
        available = [
            "spectral_resnet", "resnet18", "wideresnet50",
            "fno_resnet18", "fno_resnet50", "fno_vgg19", "fno_unet",
            "swin", "gan_inverter"
        ]
        raise ValueError(f"Unknown model name: {name}. Available: {available}")

    # Activation Replacement Logic
    activation_name = config.get("activation")
    if activation_name:
        import torch.nn as nn
        from src.utils.model_utils import replace_activation
        
        act_map = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        
        new_act = act_map.get(activation_name.lower())
        if new_act:
            print(f"Replacing all ReLU with {new_act.__name__}")
            replace_activation(model, nn.ReLU, new_act)
        else:
            print(f"Warning: Unknown activation {activation_name}, skipping replacement.")
            
    return model

