import torch
import torch.optim as optim

class Muon(optim.Optimizer):
    """
    Muon - Momentumized Uhlenbeck-Ornstein Optimizer.
    Adapted from the reference implementation by Keller Jordan.
    
    Muon is a second-order-ish optimizer that uses Newton-Schulz iteration
    to orthogonalize updates. It is typically used for large 2D parameters (weights),
    while Adam is used for biases and embeddings.
    
    For simplicity in this project, we apply it to 2D tensors and fallback to SGD/Adam
    behavior (or just standard updates) for others if needed, but here we implement
    the core logic for all >=2D tensors and standard momentum for others.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, adamw_fallback=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, adamw_fallback=adamw_fallback)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            adamw_fallback = group['adamw_fallback']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                if update.ndim >= 2 and not update.is_complex():
                    # Newton-Schulz Iteration for >2D tensors
                    # Reshape to 2D for the operation
                    original_shape = update.shape
                    if update.ndim > 2:
                        # Flatten to (d1, rest) or similar suitable projection
                        # For Conv2d: (Out, In, H, W) -> (Out, In*H*W)
                        update_2d = update.view(update.size(0), -1)
                    else:
                        update_2d = update

                    row_norm = update_2d.shape[0] ** 0.5
                    col_norm = update_2d.shape[1] ** 0.5
                    
                    # Normalize
                    g = update_2d / (update_2d.norm() + 1e-16) # Global normalization placeholder, usually sophisticated
                    
                    # Instead of complex global norm, let's stick to the standard logic:
                    # X_k+1 = 1.5 * X_k - 0.5 * X_k * X_k^T * X_k
                    # We run this on the update matrix `g`
                    
                    # Pre-condition: g should have roughly spectral norm 1
                    g = update_2d
                    # Normalize by spectral norm estimate (using frobenius as proxy often or just simple trace)
                    # For stability:
                    g = g / (g.norm() + 1e-8) 

                    for _ in range(ns_steps):
                        # G_new = 1.5 * G - 0.5 * G @ G.T @ G
                        # Note: This is expensive. For efficiency we only do it if valid.
                        if g.shape[0] < g.shape[1]:
                             g = 1.5 * g - 0.5 * (g @ (g.T @ g))
                        else:
                             g = 1.5 * g - 0.5 * ((g @ g.T) @ g)
                    
                    # Scale update by learning rate and dimensional factors
                    update_final = g * lr 
                    
                    # Reshape back
                    p.add_(update_final.view(original_shape), alpha=-1.0)
                    
                else:
                    # Fallback for 1D vectors (biases, normalization)
                    # Use simple SGD Update or AdamW-like behavior
                    p.add_(update, alpha=-lr)
                    
        return loss
