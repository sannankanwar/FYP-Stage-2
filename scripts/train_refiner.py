
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.models.factory import get_model
from src.models.refiner import ResNetRefiner
from src.training.refine_trainer import RefiningTrainer
from data.loaders.simulation import OnTheFlyDataset

def main():
    parser = argparse.ArgumentParser(description="Train Learned Optimizer (Refiner)")
    parser.add_argument("--config", type=str, required=True, help="Refinement experiment config")
    parser.add_argument("--baseline-config", type=str, required=True, help="Config of the baseline model (Exp9)")
    parser.add_argument("--baseline-checkpoint", type=str, required=True, help="Path to frozen baseline weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 1. Load Configs
    refine_config = load_config(args.config)
    baseline_config_raw = load_config(args.baseline_config)
    
    # Deep Merge Helper
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # Merge for Data Config (Refiner uses same data distribution as baseline usually)
    # But Refiner might want different training Parameters (lr, epochs)
    full_config = baseline_config_raw.copy()
    recursive_update(full_config, refine_config) # Deep merge to preserve 'data' ranges
    
    # Notification Helper
    from scripts.notify import send_telegram_message
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    def notify(title, body):
        print(f"Notification: {title} - {body}")
        if token and chat_id:
            # Escape underscores for Markdown to prevent broken formatting
            # Telegram Markdown V1 uses underscores for italics.
            safe_title = title.replace("_", "\\_")
            safe_body = body.replace("_", "\\_")
            send_telegram_message(token, chat_id, f"*{safe_title}*\n\n{safe_body}")

    experiment_name = full_config.get("experiment_name", "Refiner")
    notify(f"Refiner Training STARTED: {experiment_name}", f"Config: {args.config}")

    # 2. Setup Data
    print("Initializing Data...")
    try:
        train_dataset = OnTheFlyDataset(full_config, length=full_config.get("train_samples", 2000))
        train_loader = DataLoader(train_dataset, batch_size=full_config.get("batch_size", 64), shuffle=True, num_workers=0)
        
        val_dataset = OnTheFlyDataset(full_config, length=full_config.get("val_samples", 200)) # Validation requires different seed logic if fixed?
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 3. Load Baseline Model
        print("Loading Baseline Model...")
        
        # Load Weights First to check config
        if not os.path.exists(args.baseline_checkpoint):
             raise FileNotFoundError(f"Baseline checkpoint not found at {args.baseline_checkpoint}")
             
        ckpt = torch.load(args.baseline_checkpoint, map_location='cpu')
        
        # Helper to flatten config (matches main.py logic)
        def flatten_config(cfg):
            flat = cfg.copy()
            if 'model' in cfg and isinstance(cfg['model'], dict):
                flat.update(cfg['model'])
            if 'data' in cfg and isinstance(cfg['data'], dict):
                flat.update(cfg['data'])
            if 'training' in cfg and isinstance(cfg['training'], dict):
                flat.update(cfg['training'])
            return flat
    
        # Try to load config from checkpoint
        if 'config' in ckpt:
            print("Using configuration found in checkpoint for Baseline Model.")
            baseline_model_config = flatten_config(ckpt['config'])
        else:
            print("WARNING: No config in checkpoint, using provided YAML config.")
            baseline_model_config = flatten_config(baseline_config_raw)
    
        # Initialize Model with CORRECT config
        # Now that it is flattened, resolution=1024 will be visible, triggering Downsampler (16ch).
        baseline_model = get_model(baseline_model_config)
        
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
        try:
            baseline_model.load_state_dict(state)
            print(f"Loaded baseline from {args.baseline_checkpoint}")
        except Exception as e:
            print(f"Standard load failed ({e}), trying strict=False...")
            # If strict=False also fails on size mismatch, we catch it
            try:
                baseline_model.load_state_dict(state, strict=False)
                print("Loaded with strict=False.")
            except RuntimeError as re:
                print(f"CRITICAL ERROR: Failed to load baseline weights.")
                print(f"Mismatch Details: {re}")
                print("Possible Cause: The checkpoint was trained with a different architecture (e.g. 16 channels) than the current code expects.")
                print("Resolution: Please re-run the Baseline Experiment (Exp9) to generate a compatible checkpoint.")
                sys.exit(1)
                
        # Check Input Channels
        # Assuming first layer is conv1
        # For FNOResNet18 with Downsampler, conv1 is the stem conv.
        # The actual *Global* input channels for the model should be 2.
        # The Refiner inputs 2 channels (Cos/Sin) -> FNO (Downsampler handles -> 16).
        # This check needs to be smarter.
        # We check if the Model Class matches expectation.
        if hasattr(baseline_model, 'downsampler') and baseline_model.downsampler is not None:
             # It has downsampler, so it expects 2 channels input globally.
             pass
        elif hasattr(baseline_model, 'conv1'):
            in_ch = baseline_model.conv1.in_channels
            if in_ch != 4 and in_ch != 2:
                 print(f"WARNING: Baseline model conv1 has {in_ch} input channels. Might be fine if downsampler handles input.")
                 
        # 4. Init Refiner
        
        # 4. Init Refiner
        print("Initializing Refiner...")
        # Input channels = 4 (cos, sin, res_cos, res_sin)
        # Condition dim = 5 (xc, yc, S, f, lambda)
        # Output dim = 5 (deltas)
        refiner = ResNetRefiner(input_channels=4, condition_dim=5, output_dim=5)
        
        # Flatten Config so Trainer can find 'epochs' at top level
        full_config_flat = flatten_config(full_config)
    
        # 5. Training
        print("Starting Refiner Training...")
        trainer = RefiningTrainer(full_config_flat, baseline_model, refiner, train_loader, val_loader)
        trainer.train()
        
        notify(f"Refiner Training COMPLETED: {experiment_name}", "Process finished successfully.")

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"TRAINING FAILED: {e}")
        notify(f"Refiner Training FAILED: {experiment_name}", f"Error: {e}\n\nTraceback:\n{trace[-500:]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
