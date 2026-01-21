import argparse
import yaml
import sys
import os

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train the function inverse model.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()

    print("Loading configurations...")
    # TODO: Load configs
    
    print("Initializing components...")
    # TODO: Initialize data, model, trainer

    print("Starting training...")
    # TODO: trainer.train()

if __name__ == "__main__":
    main()
