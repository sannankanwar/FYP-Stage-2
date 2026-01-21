from src.models.hybrid import SpectralResNet
from src.models.inversion.resnet18 import InverseMetalensModel
from src.models.inversion.wideresnet50 import InverseMetalensWideResNet50

def get_model(config):
    """
    Factory function to get model instance based on configuration.
    
    Args:
        config (dict): Model configuration dictionary containing attributes like:
                       - name: "spectral_resnet", "resnet18", "wideresnet50"
                       - input_channels: int (default 2)
                       - output_dim: int (default 3)
                       - modes: int (for spectral models)
                       
    Returns:
        nn.Module: The requested model instance.
    """
    name = config.get("name", "spectral_resnet")
    input_channels = config.get("input_channels", 2)
    output_dim = config.get("output_dim", 3)
    
    if name == "spectral_resnet":
        modes = config.get("modes", 16)
        # SpectralResNet takes (in_channels, modes)
        # Output dim adjustment might be needed if not standard
        # But looking at hybrid.py, it hardcodes the FC to output 3.
        # FUTURE TODO: Make SpectralResNet output_dim configurable.
        return SpectralResNet(in_channels=input_channels, modes=modes)
        
    elif name == "resnet18":
        # InverseMetalensModel(output_dim=3, input_channels=2)
        return InverseMetalensModel(output_dim=output_dim, input_channels=input_channels)
        
    elif name == "wideresnet50":
        # InverseMetalensWideResNet50(output_dim=3, input_channels=2)
        return InverseMetalensWideResNet50(output_dim=output_dim, input_channels=input_channels)
        
    else:
        raise ValueError(f"Unknown model name: {name}. Available: spectral_resnet, resnet18, wideresnet50")
