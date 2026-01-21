import os
import shutil
import argparse
import datetime

def organize_experiment(name, description, output_root="outputs"):
    """
    Moves contents of outputs/checkpoints and outputs/logs into outputs/<name>/.
    Also moves loss_plot.png if it exists.
    Creates an experiment_info.md file.
    """
    
    # Define Target Paths
    target_dir = os.path.join(output_root, name)
    target_checkpoints = os.path.join(target_dir, "checkpoints")
    target_logs = os.path.join(target_dir, "logs")
    
    # Create Directories
    if os.path.exists(target_dir):
        print(f"Warning: Target directory {target_dir} already exists.")
        ans = input("Do you want to continue and potentially overwrite/merge? (y/n): ")
        if ans.lower() != 'y':
            print("Aborting.")
            return

    os.makedirs(target_checkpoints, exist_ok=True)
    os.makedirs(target_logs, exist_ok=True)
    
    # Source Directory (Root outputs)
    src_checkpoints = os.path.join(output_root, "checkpoints")
    src_logs = os.path.join(output_root, "logs")
    
    # Move Checkpoints
    if os.path.exists(src_checkpoints):
        print(f"Moving contents of {src_checkpoints} to {target_checkpoints}...")
        for item in os.listdir(src_checkpoints):
            s = os.path.join(src_checkpoints, item)
            d = os.path.join(target_checkpoints, item)
            if os.path.exists(d):
                print(f"  Skipping {item} (already exists in target)")
            else:
                shutil.move(s, d)
    else:
        print(f"Source checkpoints directory {src_checkpoints} not found (or empty).")
        
    # Move Logs
    if os.path.exists(src_logs):
        print(f"Moving contents of {src_logs} to {target_logs}...")
        for item in os.listdir(src_logs):
            s = os.path.join(src_logs, item)
            d = os.path.join(target_logs, item)
            if os.path.exists(d):
                 print(f"  Skipping {item} (already exists in target)")
            else:
                shutil.move(s, d)
    else:
        print(f"Source logs directory {src_logs} not found.")

    # Move Plot if exists
    # Check current directory and output directory for plot
    plot_name = "loss_plot.png" 
    # Usually script runs from root, so check root
    if os.path.exists(plot_name):
        print(f"Moving {plot_name} to {target_dir}...")
        shutil.move(plot_name, os.path.join(target_dir, plot_name))
    
    # Write Info File
    info_path = os.path.join(target_dir, "experiment_info.md")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(info_path, "w") as f:
        f.write(f"# Experiment: {name}\n\n")
        f.write(f"**Date**: {timestamp}\n\n")
        f.write(f"## Description\n{description}\n")
        
    print(f"\nSuccess! Experiment organized in {target_dir}")
    print(f"Info file created at {info_path}")

def main():
    parser = argparse.ArgumentParser(description="Organize experiment outputs into a named folder.")
    parser.add_argument("--name", required=True, help="Name of the experiment folder (e.g., experiment1)")
    parser.add_argument("--description", required=True, help="Description of the experiment conditions")
    
    args = parser.parse_args()
    
    organize_experiment(args.name, args.description)

if __name__ == "__main__":
    main()
