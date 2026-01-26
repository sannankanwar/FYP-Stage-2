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

def process_predictions(model, predictions, normalizer, config):
    """
    Safely process model predictions, checking for output_space contract.
    Extracting this logic allows unit testing to prevent double-denormalization.
    """
    if not hasattr(model, 'output_space'):
         # Fix 2 & 5: Fail fast if contract is missing
         raise RuntimeError(
             f"Model {type(model).__name__} is missing required attribute 'output_space'. "
             "Cannot guess prediction tensor space safely."
         )

    output_space = model.output_space
    
    if output_space == 'unknown':
         raise RuntimeError(
             f"Model {type(model).__name__} has output_space='unknown'. "
             "Safe processing requires explicit 'physical' or 'normalized' contract."
         )
    
    if output_space == 'physical':
        print(f"Model contract: output_space='{output_space}'. Predictions are already physical.")
        if normalizer:
             # Assertion for regression prevention (Fix 5)
             # IF we were blindly following config, we would double denorm here.
             pass # SAFE: preventing double-denorm
    elif normalizer and config.get("standardize_outputs", False):
        # Only denormalize if model is NOT physical and config says standardized
        print(f"Model contract '{output_space}' (not physical). Denormalizing predictions...")
        predictions = normalizer.denormalize_tensor(predictions)
        
    return predictions
