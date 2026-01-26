import os
import glob
import pandas as pd
import numpy as np
import json
import argparse

def analyze_experiment(exp_dir):
    """
    Analyzes a single experiment directory.
    Reads history.csv and returns best metrics.
    """
    history_path = os.path.join(exp_dir, "history.csv")
    if not os.path.exists(history_path):
        print(f"Skipping {exp_dir}: No history.csv found")
        return None
        
    try:
        df = pd.read_csv(history_path)
    except Exception as e:
        print(f"Error reading {history_path}: {e}")
        return None
        
    if df.empty:
        return None
        
    # Find Best Epoch (Lowest Val Loss)
    # Note: 'val_loss' in history.csv determines the best model checkpoint logic in Trainer
    best_row = df.loc[df['val_loss'].idxmin()]
    
    return {
        'exp_dir': exp_dir,
        'best_epoch': int(best_row['epoch']),
        'val_mse': best_row['val_loss'], # Assuming val_loss is MSE-based (it is for our losses usually, or proportional)
        # R2 might not be in history? Trainer logs train_loss, val_loss.
        # If R2 is missing, we rely on MSE.
        # Trainer doesn't calculate R2 in validation loop yet explicitly for history.csv
        # checks: Trainer._validate_epoch returns avg_val_loss.
    }

def select_best_run(suite_dir):
    """
    Selects best run from a suite directory containing multiple experiment folders.
    """
    exp_dirs = glob.glob(os.path.join(suite_dir, "exp5_*"))
    experiments = []
    
    for d in exp_dirs:
        if os.path.isdir(d):
            stats = analyze_experiment(d)
            if stats:
                experiments.append(stats)
                
    if not experiments:
        print("No valid experiments found.")
        return None
        
    print(f"Found {len(experiments)} experiments.")
    
    # Sort by Val MSE (Primary)
    # If we had R2, we would sort by R2 descending as tiebreaker.
    sorted_exps = sorted(experiments, key=lambda x: x['val_mse'])
    
    best = sorted_exps[0]
    
    print("\n--- Leaderboard ---")
    for i, exp in enumerate(sorted_exps):
        print(f"{i+1}. {os.path.basename(exp['exp_dir'])} | Val MSE: {exp['val_mse']:.6f} (Epoch {exp['best_epoch']})")
        
    print(f"\nWINNER: {os.path.basename(best['exp_dir'])}")
    
    # Save selection
    output_path = os.path.join(suite_dir, "best_run_selection.json")
    with open(output_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Selection saved to {output_path}")
    
    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, default="outputs_exp5")
    args = parser.parse_args()
    
    select_best_run(args.suite_dir)
