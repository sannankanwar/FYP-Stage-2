import torch.nn as nn

def replace_activation(module, old_activation_type, new_activation_module):
    """
    Recursively replaces all instances of `old_activation_type` with `new_activation_module` 
    in the given `module`.
    
    Args:
        module (nn.Module): The model or module to modify.
        old_activation_type (type): The class of the activation to replace (e.g. nn.ReLU).
        new_activation_module (nn.Module or callable): Factory for the new activation (e.g. nn.SiLU()).
                                                        It should be a new instance or a function returning one.
                                                        
    Returns:
        nn.Module: The modified module (in-place).
    """
    
    for name, child in module.named_children():
        if isinstance(child, old_activation_type):
            # Found a match, replace it
            # Check if new_activation_module is a class or instance
            if isinstance(new_activation_module, type):
                new_act = new_activation_module()
            else:
                # Assuming it's an instance or a factory function, we need a fresh copy usually
                # But if it's stateless (like SiLU), reusing instance is often fine. 
                # For safety, we try to copy or instantiate if type. 
                # Simplest for now: pass an instance like nn.SiLU() and we use it (or a deepcopy if needed).
                # Since simple activations are stateless, assigning the same instance is fine.
                new_act = new_activation_module
            
            setattr(module, name, new_act)
        else:
            # Recurse
            replace_activation(child, old_activation_type, new_act_factory_check(new_activation_module))
            
    return module

def new_act_factory_check(proto):
    """Refreshes the new activation if it has state, otherwise returns as is."""
    # For standard activations (SiLU, Tanh, GELU), they are stateless, so we can pass them through.
    return proto
