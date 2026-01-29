import torch
import numpy as np

class DifficultySampler:
    """
    Direction B: Difficulty-Aware Active Sampler.
    Maintains a difficulty map over the parameter space and guides sampling.
    """
    def __init__(self, config):
        self.config = config
        
        # Hyperparameters
        self.alpha = float(config.get("sampler_alpha", 1.0))
        self.beta = float(config.get("sampler_beta", 0.5))
        self.ema_lambda = float(config.get("sampler_lambda", 0.05)) # Slow update
        self.p_min_floor = 1e-5 # To be refined based on bucket count
        
        # Parameter Ranges
        self.ranges = {
            'xc': config.get("xc_range", [-100, 100]),
            'yc': config.get("yc_range", [-100, 100]),
            'S': config.get("S_range", [5, 50]),
            'f': config.get("focal_length_range", [10, 100]),
            'lambda': config.get("wavelength_range", [0.4, 0.7])
        }
        
        # Grid Definition
        # 5 bins per parameter -> 5^5 = 3125 buckets
        self.bins_per_dim = 5
        self.param_order = ['xc', 'yc', 'S', 'f', 'lambda']
        self.total_buckets = self.bins_per_dim ** len(self.param_order)
        
        # Refine P_min based on bucket count
        self.p_min = 0.1 / self.total_buckets
        
        # State: Difficulty Table D
        # Initialize with 1.0
        self.D = torch.ones(self.total_buckets, dtype=torch.float32)
        
        # Pre-compute bin edges for efficiency? 
        # Easier to compute bin index on the fly.
        
        print(f"DifficultySampler Initialized: {self.total_buckets} buckets. Alpha={self.alpha}, Beta={self.beta}")

    def _get_bucket_indices(self, params_tensor):
        """
        Map parameters to bucket indices.
        params_tensor: (B, 5) [xc, yc, S, f, lambda]
        """
        B = params_tensor.shape[0]
        indices = torch.zeros(B, dtype=torch.long)
        
        stride = 1
        for i, name in enumerate(self.param_order):
            vals = params_tensor[:, i]
            low, high = self.ranges[name]
            
            # Normalize to [0, 1]
            norm = (vals - low) / (high - low)
            norm = torch.clamp(norm, 0.0, 0.9999) # Avoid 1.0 overflowing
            
            # Discretize
            bin_idx = (norm * self.bins_per_dim).long()
            
            # Accumulate index
            indices += bin_idx * stride
            stride *= self.bins_per_dim
            
        return indices

    def update(self, params_tensor, losses):
        """
        Update difficulty D based on observed losses.
        params_tensor: (B, 5)
        losses: (B,)
        """
        # Ensure CPU
        params_tensor = params_tensor.cpu()
        losses = losses.detach().cpu()
        
        indices = self._get_bucket_indices(params_tensor)
        
        # EMA Update
        # D[b] = (1-lambda) D[b] + lambda * loss
        # We handle duplicate indices in batch by simple loop or scatter_add (mean).
        # Loop is fine for batch_size 64.
        for i in range(len(indices)):
            idx = indices[i].item()
            loss_val = losses[i].item()
            
            self.D[idx] = (1 - self.ema_lambda) * self.D[idx] + self.ema_lambda * loss_val

    def sample_batch_params(self, batch_size):
        """
        Generate a batch of parameters using Mixture Sampling.
        Returns: (batch_size, 5) tensor
        """
        # 1. Determine number of active samples
        n_active = int(batch_size * self.beta)
        n_uniform = batch_size - n_active
        
        samples = []
        
        # Uniform Part
        if n_uniform > 0:
            uni_sample = torch.zeros(n_uniform, 5)
            for i, name in enumerate(self.param_order):
                low, high = self.ranges[name]
                uni_sample[:, i] = torch.rand(n_uniform) * (high - low) + low
            samples.append(uni_sample)
            
        # Active Part
        if n_active > 0:
            # Compute Probabilities P(b)
            # Smooth D
            D_smooth = self.D + 1e-6
            scores = D_smooth.pow(self.alpha)
            
            # Normalize momentarily to check clamping? No, straightforward clamping on P.
            P = scores / scores.sum()
            
            # Clamp
            P = torch.clamp(P, min=self.p_min)
            
            # Renormalize
            P = P / P.sum()
            
            # Sample Buckets
            bucket_indices = torch.multinomial(P, n_active, replacement=True)
            
            # Sample uniformly within buckets
            active_sample = torch.zeros(n_active, 5)
            
            # Reconstruct ranges from bucket index
            # This is the inverse of stride logic
            for i in range(n_active):
                b_idx = bucket_indices[i].item()
                temp_idx = b_idx
                
                for dim, name in enumerate(self.param_order):
                    stride = self.bins_per_dim ** dim # Wait, stride logic in _get was forward?
                    # My _get logic: idx += bin * stride
                    # where stride grows: 1, 5, 25...
                    # So dim 0 (xc) is LSB. dim 4 is MSB?
                    # stride for dim 0 is 1.
                    # bin = (temp_idx % (stride * 5)) // stride ?
                    # Cleaner:
                    # bin = temp_idx % 5
                    # temp_idx //= 5
                    
                    bin_val = temp_idx % self.bins_per_dim
                    temp_idx = temp_idx // self.bins_per_dim
                    
                    # Convert bin to range
                    low_global, high_global = self.ranges[name]
                    bin_width = (high_global - low_global) / self.bins_per_dim
                    
                    b_low = low_global + bin_val * bin_width
                    b_high = b_low + bin_width
                    
                    # Sample uniform in bin
                    val = torch.rand(1).item() * (b_high - b_low) + b_low
                    active_sample[i, dim] = val
            
            samples.append(active_sample)
            
        # Combine
        full_batch = torch.cat(samples, dim=0)
        
        # Shuffle (important since we appended blocks)
        perm = torch.randperm(batch_size)
        return full_batch[perm]
