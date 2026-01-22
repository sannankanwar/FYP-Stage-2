import re
import argparse
import matplotlib.pyplot as plt
import sys
import os

def parse_log_file(log_path):
    epochs = []
    train_losses = []
    val_losses = []

    # Regex to extract data
    pattern = re.compile(r"Epoch\s+(\d+)/\d+\s+\|.*Train Loss:\s+([\d\.e\+\-]+)\s+\|.*Val Loss:\s+([\d\.e\+\-]+)")

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None, None, None

    return epochs, train_losses, val_losses

def plot_losses(data_list, output_path, title=None, min_epoch=0, max_epoch=None):
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(14, 8))
    
    # distinct colors (Tableau 10)
    colors = plt.cm.tab10.colors
    
    for i, (name, epochs, train_losses, val_losses) in enumerate(data_list):
        # Filter by epoch range
        indices = [j for j, e in enumerate(epochs) if e >= min_epoch and (max_epoch is None or e <= max_epoch)]
        
        if not indices:
            print(f"Warning: No data for {name} in range [{min_epoch}, {max_epoch}]")
            continue
            
        filtered_epochs = [epochs[j] for j in indices]
        filtered_train = [train_losses[j] for j in indices]
        filtered_val = [val_losses[j] for j in indices]
        
        color = colors[i % len(colors)]
        # Plot Validation Loss (Solid line)
        plt.plot(filtered_epochs, filtered_val, label=f'{name} (Val)', linestyle='-', marker=None, linewidth=2.5, color=color, alpha=0.9)
        # Plot Train Loss (Dotted line, lighter)
        plt.plot(filtered_epochs, filtered_train, label=f'{name} (Train)', linestyle=':', linewidth=1.5, color=color, alpha=0.6)
    
    final_title = title if title else 'Training and Validation Loss Comparison'
    if min_epoch > 0:
        final_title += f" (Epochs {min_epoch}+)"
        
    plt.title(final_title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss from log file(s).")
    parser.add_argument("log_files", nargs='+', help="Path to the log file(s). You can specify multiple files to compare.")
    parser.add_argument("--labels", nargs='+', help="Custom labels for the log files (corresponding order).")
    parser.add_argument("--output", "-o", default="loss_plot.png", help="Path to save the plot image (default: loss_plot.png)")
    parser.add_argument("--title", type=str, default=None, help="Custom title for the plot")
    parser.add_argument("--min-epoch", type=int, default=0, help="Minimum epoch to plot (zoom in)")
    parser.add_argument("--max-epoch", type=int, default=None, help="Maximum epoch to plot")
    
    args = parser.parse_args()
    
    data_list = []
    
    for i, log_file in enumerate(args.log_files):
        print(f"Reading: {log_file}")
        epochs, train, val = parse_log_file(log_file)
        if epochs:
            # Determine Name/Label
            if args.labels and i < len(args.labels):
                name = args.labels[i]
            else:
                name = os.path.splitext(os.path.basename(log_file))[0]
            
            data_list.append((name, epochs, train, val))
        else:
            print(f"Skipping {log_file} (no data found)")
            
    if not data_list:
        print("No valid data found to plot.")
        return
        
    plot_losses(data_list, args.output, args.title, args.min_epoch, args.max_epoch)

if __name__ == "__main__":
    main()
