import torch

class ParameterNormalizer:
    def __init__(self, ranges):
        """
        Args:
            ranges (dict): Dictionary of parameter ranges, e.g.,
                           {'width': [min, max], 'gap': [min, max], ...}
        """
        self.means = {}
        self.stds = {}
        self.param_names = []
        
        # We assume the model outputs parameters in a specific order.
        # Ideally, this should be consistent with how the model output is defined.
        # For this project, let's assume the order is: width, gap, height, length
        # based on previous data loader logic.
        self.param_order = ['xc', 'yc', 'fov']
        
        self._compute_stats(ranges)

    def _compute_stats(self, ranges):
        for name in self.param_order:
            if name in ranges:
                low, high = ranges[name]
                # Map [low, high] to roughly [-1, 1] or similar standard normal
                # Using mean and std of a uniform distribution U(low, high)
                # mean = (high + low) / 2
                # std = (high - low) / sqrt(12)
                # Or just MinMax scaling to [-1, 1]?
                # Standard score (z-score) is usually better for neural nets.
                
                mu = (high + low) / 2.0
                sigma = (high - low) / (2.0 * 1.732) # approx sqrt(3) is 1.732 for uniform distribution std dev
                # actually, let's just use simple scaling to mean 0, range [-1, 1]
                # val = (x - mu) / (range/2)
                
                self.means[name] = mu
                self.stds[name] = (high - low) / 2.0 # Scale to [-1, 1]
            else:
                # Default identity
                self.means[name] = 0.0
                self.stds[name] = 1.0

    def normalize(self, params_dict):
        """
        params_dict: dict of {name: tensor}
        Returns: tensor of shape (batch, num_params)
        """
        outputs = []
        for name in self.param_order:
            if name in params_dict:
                val = params_dict[name]
                norm_val = (val - self.means[name]) / self.stds[name]
                outputs.append(norm_val)
        return torch.stack(outputs, dim=1)

    def denormalize(self, tensor):
        """
        tensor: (batch, num_params)
        Returns: dict of {name: tensor}
        """
        outputs = {}
        for i, name in enumerate(self.param_order):
            if i < tensor.shape[1]:
                val = tensor[:, i]
                denorm_val = val * self.stds[name] + self.means[name]
                outputs[name] = denorm_val
        return outputs

    def normalize_tensor(self, tensor):
        """
        Assumes tensor is already stacked in order.
        """
        out = tensor.clone()
        for i, name in enumerate(self.param_order):
            if i < tensor.shape[1]:
                out[:, i] = (out[:, i] - self.means[name]) / self.stds[name]
        return out
        
    def denormalize_tensor(self, tensor):
        """
        Returns tensor in original scale, shape preserved.
        """
        out = tensor.clone()
        for i, name in enumerate(self.param_order):
            if i < tensor.shape[1]:
                out[:, i] = out[:, i] * self.stds[name] + self.means[name]
        return out
