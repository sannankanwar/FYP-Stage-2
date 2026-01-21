import re
import argparse
import matplotlib.pyplot as plt
import sys

def parse_log_file(log_path):
    epochs = []
    train_losses = []
    val_losses = []

    # Regex to extract data from lines like:
    # Epoch 94/100 | Time: 14.10s | Train Loss: 1769.868557 | Val Loss: 949.177473
    # It constructs a pattern that looks for 'Epoch', integer group, float group for Train Loss, float group for Val Loss
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
        sys.exit(1)

    return epochs, train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=4)
    
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Use log scale if standard deviation of loss is huge (optional, but good for losses starting high)
    # plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss from log file.")
    parser.add_argument("log_file", nargs='?', default="output.log", help="Path to the log file (default: output.log)")
    parser.add_argument("--output", "-o", default="loss_plot.png", help="Path to save the plot image (default: loss_plot.png)")
    
    args = parser.parse_args()
    
    print(f"Reading log file: {args.log_file}")
    epochs, train_losses, val_losses = parse_log_file(args.log_file)
    
    if not epochs:
        print("No valid epoch lines found in the log file.")
        return
        
    print(f"Found {len(epochs)} epochs.")
    plot_losses(epochs, train_losses, val_losses, args.output)

if __name__ == "__main__":
    main()
