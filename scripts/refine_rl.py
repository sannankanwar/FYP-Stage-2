
import argparse
import os
import sys
import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.envs.metalens_env import MetalensRefinementEnv
from data.loaders.simulation import OnTheFlyDataset

def make_env(config, dataset, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = MetalensRefinementEnv(config, dataset)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent for Refinement")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--steps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--output_dir", type=str, default="experiments/rl_refiner", help="Save dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Config & Dataset
    config = load_config(args.config)
    
    # Force some config for training speed if needed
    # config['resolution'] = 256 # Lower res for faster training loop?
    # Actually the Env uses config['resolution'] to generate coordinate grid. 
    # If we want faster training, we should probably set this low in the passed config dict.
    
    print("Initializing Dataset (for Environment usage)...")
    dataset = OnTheFlyDataset(config, length=1000) # Virtual length
    
    # 2. Setup Environment
    if args.n_envs > 1:
        # Multiprocessing
        env = SubprocVecEnv([make_env(config, dataset, i, args.seed) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(config, dataset, 0, args.seed)])
        
    # 3. Setup PPO Agent
    # MultiInputPolicy is required because our Observation Space is a Dict
    print("Initializing PPO Agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=os.path.join(args.output_dir, "logs")
    )
    
    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // args.n_envs,
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="ppo_refiner"
    )
    
    # 4. Train
    print(f"Starting Training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    
    # 5. Save Final Model
    final_path = os.path.join(args.output_dir, "ppo_refiner_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")
    
    # 6. Evaluation (Quick Smoke Test)
    print("\nRunning Verification Episode...")
    obs = env.reset()
    total_reward = 0
    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        # VecEnv returns array of done
        if isinstance(done, np.ndarray):
            done = done[0]
            
    print(f"Episode finished in {step} steps. Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
