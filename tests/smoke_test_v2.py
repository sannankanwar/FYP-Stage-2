
import subprocess
import os
import shutil

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        print(result.stderr)
        return False
    return True

def smoke_test():
    print("=== Pipeline Smoke Test V2 ===")
    
    # 1. Create a dummy experiment config
    os.makedirs("configs/experiments", exist_ok=True)
    smoke_config_path = "configs/experiments/exp_smoke.yaml"
    with open(smoke_config_path, "w") as f:
        f.write("""
experiment_name: "test_smoke_v2"
description: "End-to-end smoke test"
epochs: 2
batch_size: 2
learning_rate: 0.001
optimizer: "adam"
standardize_outputs: true
loss_function: "weighted_standardized"
loss_weights: [1, 1, 1, 10, 10]
model:
  name: "spectral_resnet"
  modes: 2
  resolution: 64
""")

    # 2. Add local data override for speed in the main data config
    # We'll just rely on the defaults but maybe reduce train/val samples
    # Actually, OnTheFlyDataset uses lengths passed in train.py (1000/200)
    # 2 epochs * 1000 samples is a bit much for a smoke test.
    # I'll modify train.py to respect samples from config if present.

    # 3. Clean up
    if os.path.exists("outputs_2/test_smoke_v2"):
        shutil.rmtree("outputs_2/test_smoke_v2")

    # 4. Run the Pipeline steps manually (to check individual failures)
    exp_dir = "outputs_2/test_smoke_v2"
    log_file = f"{exp_dir}/logs/output.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # A. Training
    if not run_command(f"uv run scripts/train.py --config {smoke_config_path} > {log_file} 2>&1"):
        return

    # B. Loss Curves
    if not run_command(f"uv run python scripts/plot_loss.py {log_file} --output {exp_dir}/loss_plot_full.png"):
        return
    if not run_command(f"uv run python scripts/plot_loss.py {log_file} --output {exp_dir}/loss_plot_zoomed.png --min-epoch 1"):
        return

    # C. Evaluate (Scatter/Heatmap)
    if not run_command(f"uv run python scripts/evaluate.py --experiment_dir {exp_dir}"):
        return

    # D. Phase Reconstruction
    if not run_command(f"uv run python scripts/visualize_reconstruction.py --experiment_dir {exp_dir}"):
        return

    # 5. Verify Artifacts
    expected_files = [
        "checkpoints/best_model.pth",
        "logs/output.log",
        "loss_plot_full.png",
        "loss_plot_zoomed.png",
        "visualizations/reconstruction_comparison.png",
        "scatter_plots.png"
    ]
    
    all_present = True
    for f in expected_files:
        path = os.path.join(exp_dir, f)
        if os.path.exists(path):
            print(f" [OK] {f} exists.")
        else:
            print(f" [FAIL] {f} is missing at {path}")
            all_present = False

    if all_present:
        print("\n=== SMOKE TEST PASSED ===")
    else:
        print("\n=== SMOKE TEST FAILED ===")

if __name__ == "__main__":
    smoke_test()
