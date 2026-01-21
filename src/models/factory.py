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
        model = SpectralResNet(in_channels=input_channels, modes=modes)
    elif name == "resnet18":
        model = InverseMetalensModel(output_dim=output_dim, input_channels=input_channels)
    elif name == "wideresnet50":
        model = InverseMetalensWideResNet50(output_dim=output_dim, input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model name: {name}. Available: spectral_resnet, resnet18, wideresnet50")

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
