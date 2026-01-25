"""
Compare learning curves across experiments.
Reads history.csv from each experiment directory.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Directory containing experiment outputs")
    args = parser.parse_args()
    
    # Find all history.csv files
    # Structure: output_dir/EXP_NAME/history.csv
    exp_dirs = sorted(glob.glob(os.path.join(args.output_dir, "exp*")))
    
    data = []
    
    for d in exp_dirs:
        exp_name = os.path.basename(d)
        hist_path = os.path.join(d, "history.csv")
        
        if os.path.exists(hist_path):
            try:
                df = pd.read_csv(hist_path)
                df['experiment'] = exp_name
                data.append(df)
            except Exception as e:
                print(f"Error reading {hist_path}: {e}")
        else:
            print(f"Warning: No history.csv for {exp_name}")
            
    if not data:
        print("No data found to plot.")
        return

    full_df = pd.concat(data, ignore_index=True)
    
    # Setup plotting style
    sns.set_style("darkgrid")
    
    # 1. Overall Validation Loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_df, x='epoch', y='val_loss', hue='experiment', marker='o')
    plt.title("Validation Loss Comparison (Full Training)")
    plt.ylabel("Loss")
    plt.yscale('log') # Log scale often helps with loss
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison_val_loss_full.png"), dpi=150)
    plt.close()
    
    # 2. Zoomed Last 25 Epochs
    max_epoch = full_df['epoch'].max()
    zoom_start = max(0, max_epoch - 25)
    zoom_df = full_df[full_df['epoch'] > zoom_start]
    
    if not zoom_df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=zoom_df, x='epoch', y='val_loss', hue='experiment', marker='o')
        plt.title(f"Validation Loss (Last {max_epoch - zoom_start} Epochs)")
        plt.ylabel("Loss")
        # plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_val_loss_zoomed.png"), dpi=150)
        plt.close()
        
    print(f"Saved comparison plots to {args.output_dir}")

if __name__ == "__main__":
    main()
