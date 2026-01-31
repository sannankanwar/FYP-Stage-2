
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import copy

from src.inversion.forward_model import compute_hyperbolic_phase, wrap_phase, get_2channel_representation

class MetalensRefinementEnv(gym.Env):
    """
    Gymnasium environment for refining metalens parameters.
    
    State:
        - Current Parameters [xc, yc, scale, wl, f] (Normalized or raw?) -> Let's use Relative Offset from initial guess?
        - Residual Map (Downsampled to 64x64 or 32x32 to save RAM/Compute for policy)
        
    Action:
        - Delta params [d_xc, d_yc, d_scale, d_wl, d_f]
        
    Reward:
        - Improvement in MSE (Old MSE - New MSE)
    """
    
    def __init__(self, config, dataset, sample_id_list=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.config = config
        self.dataset = dataset
        self.device = device
        
        # Determine subset of samples to train on
        if sample_id_list is None:
            self.sample_ids = list(range(len(dataset)))
        else:
            self.sample_ids = sample_id_list
            
        # Action Space: Continuous [-1, 1] corresponding to max_step_size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # Observation Space: Dict
        # 1. 'params': 5 floats
        # 2. 'residual': 1 channel, 64x64 image
        self.obs_size = 64
        self.observation_space = spaces.Dict({
            "params": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "residual": spaces.Box(low=-np.pi, high=np.pi, shape=(1, self.obs_size, self.obs_size), dtype=np.float32)
        })
        
        # Max step sizes for each param (Tuning required)
        # xc, yc (um), scale (ratio), wl (um), f (um)
        self.max_step = np.array([2.0, 2.0, 0.05, 0.05, 2.0], dtype=np.float32) 
        
        self.current_sample_idx = 0
        self.current_params = None
        self.target_params = None
        self.target_phase = None
        self.prev_mse = 0.0
        self.current_step_count = 0
        self.max_steps_per_episode = 20
        
        # Coordinate grid for physics (Fixed size based on config)
        # Infer resolution from dataset to match shapes
        if hasattr(dataset, 'N'):
            N = dataset.N
        else:
            # Inspection
            sample, _ = dataset[0]
            # Sample is [2, H, W]
            N = sample.shape[-1]
            
        L = config.get('physical_size', 100.0)
        # N = config.get('resolution', 1024) # Removed hardcoded default
        
        print(f"MetalensEnv: Simulation Resolution = {N}x{N}, Physical Size = {L}um")
        
        x = torch.linspace(-L/2, L/2, N).to(device)
        y = torch.linspace(-L/2, L/2, N).to(device)
        self.Y, self.X = torch.meshgrid(y, x, indexing='ij')
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick random sample
        self.current_sample_idx = np.random.choice(self.sample_ids)
        input_tensor, target_params = self.dataset[self.current_sample_idx]
        
        # Ground Truth Phase (Recover from input cos/sin)
        cos_map = input_tensor[0].to(self.device)
        sin_map = input_tensor[1].to(self.device)
        self.target_phase = torch.atan2(sin_map, cos_map)
        
        # Initial Guess: 
        # Ideally we run the Experiment9 model here. 
        # For efficiency, we can assume the Environment is initialized with PREDICTED params if available.
        # OR: We just perturb the GT params to simulate 'prediction error' for training purposes.
        # Let's start with Perturbed GT for faster training loop (avoids loading massive Exp9 ResNet).
        
        # Perturb by ~10%
        self.target_params = target_params.to(self.device) # [xc, yc, scale, wl, f]
        noise = torch.randn(5).to(self.device) * (self.target_params * 0.1) 
        # Add absolute noise for xy which might be 0
        noise[:2] += torch.randn(2).to(self.device) * 5.0 
        
        self.current_params = self.target_params + noise
        
        # Compute Initial State
        obs, mse = self._get_obs_and_mse()
        self.prev_mse = mse
        self.current_step_count = 0
        
        return obs, {}
        
    def step(self, action):
        self.current_step_count += 1
        
        # Update Params
        delta = torch.from_numpy(action).to(self.device) * torch.from_numpy(self.max_step).to(self.device)
        self.current_params += delta
        
        # Clip/Constrain?
        # Physics constraints (e.g. wl > 0, f > 0)
        self.current_params[2] = torch.clamp(self.current_params[2], 0.1, 5.0) # Scale
        self.current_params[3] = torch.clamp(self.current_params[3], 0.1, 2.0) # Wavelength
        self.current_params[4] = torch.clamp(self.current_params[4], 1.0, 1000.0) # Focal Length
        
        # Compute New State
        obs, mse = self._get_obs_and_mse()
        
        # Reward: Improvement
        reward = (self.prev_mse - mse) * 100.0 # Scale up for better gradient
        # Penalty for step?
        reward -= 0.1 
        
        self.prev_mse = mse
        
        # Done condition
        terminated = False
        truncated = False
        
        if self.current_step_count >= self.max_steps_per_episode:
            truncated = True
            
        if mse < 1e-4: # Converged
            terminated = True
            reward += 10.0
            
        return obs, reward, terminated, truncated, {}
        
    def _get_obs_and_mse(self):
        # 1. Forward Model
        # Unpack
        xc, yc, scale, wl, f = self.current_params
        
        # Coordinate Transform (matches Refiner/DE logic)
        # X_eff = X / scale + xc
        
        X_eff = self.X / scale + xc
        Y_eff = self.Y / scale + yc
        
        # Physics Phase
        k0 = 2.0 * torch.pi / wl
        R_sq = X_eff**2 + Y_eff**2
        phase_unwrapped = k0 * (torch.sqrt(R_sq + f**2) - f)
        
        # Wrap
        phase_sim = torch.angle(torch.exp(1j * phase_unwrapped))
        
        # 2. Residual
        # Wrap (Sim - Target)
        # Using complex diff is safer: angle(exp(1j*sim) / exp(1j*target)) = angle(exp(1j*(sim-target)))
        diff = torch.angle(torch.exp(1j * (phase_sim - self.target_phase)))
        
        mse = torch.mean(diff**2).item()
        
        # 3. Observation (Resize for CNN policy)
        # Diff is NxN (1024x1024). Resize to 64x64.
        # Add channel dim: [1, 1, 64, 64] for interpolate
        diff_small = torch.nn.functional.interpolate(
            diff.unsqueeze(0).unsqueeze(0), 
            size=(self.obs_size, self.obs_size), 
            mode='bilinear'
        ).squeeze(0) # [1, 64, 64]
        
        obs_params = self.current_params.detach().cpu().numpy().astype(np.float32)
        obs_residual = diff_small.detach().cpu().numpy().astype(np.float32)
        
        return {
            "params": obs_params,
            "residual": obs_residual
        }, mse
