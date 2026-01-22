import torch

class ParameterNormalizer:
    def __init__(self, ranges):
        """
        Args:
            ranges (dict): Dictionary of parameter ranges, e.g.,
                           {'xc': [min, max], 'yc': [min, max], ...}
        """
        self.means = {}
        self.stds = {}
        self.param_names = []
        
        # Order of parameters in the tensor
        self.param_order = ['xc', 'yc', 'fov', 'wavelength', 'focal_length']
        
        self._compute_stats(ranges)

    def _compute_stats(self, ranges):
        for name in self.param_order:
            if name in ranges:
                low, high = ranges[name]
                # Scale to [-1, 1]
                # val = (x - mean) / scale
                # where mean = (high + low) / 2
                # and scale = (high - low) / 2
                
                mu = (high + low) / 2.0
                sigma = (high - low) / 2.0 
                
                self.means[name] = mu
                self.stds[name] = sigma
            else:
                # Default identity
                # print(f"Warning: Range for {name} not provided to normalizer. Using Identity.")
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
