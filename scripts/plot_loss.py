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

def plot_losses(data_list, output_path):
    plt.figure(figsize=(12, 7))
    
    # distinct colors
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple']
    
    for i, (name, epochs, train_losses, val_losses) in enumerate(data_list):
        color = colors[i % len(colors)]
        # Plot Validation Loss (Solid line, more prominent)
        plt.plot(epochs, val_losses, label=f'{name} (Val)', linestyle='-', marker='s', markersize=4, linewidth=2, color=color)
        # Plot Train Loss (Dashed line, lighter)
        plt.plot(epochs, train_losses, label=f'{name} (Train)', linestyle='--', alpha=0.6, linewidth=1.5, color=color)
    
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss from log file(s).")
    parser.add_argument("log_files", nargs='+', help="Path to the log file(s). You can specify multiple files to compare.")
    parser.add_argument("--labels", nargs='+', help="Custom labels for the log files (corresponding order).")
    parser.add_argument("--output", "-o", default="loss_plot.png", help="Path to save the plot image (default: loss_plot.png)")
    
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
        
    plot_losses(data_list, args.output)

if __name__ == "__main__":
    main()
