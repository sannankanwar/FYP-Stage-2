import os
import glob
import json
import argparse
import sys
import datetime

def analyze_experiment(exp_dir):
    """
    Analyzes a single experiment directory using the 'best_metrics.json' contract.
    """
    metrics_path = os.path.join(exp_dir, "best_metrics.json")
    
    if not os.path.exists(metrics_path):
        # Fallback: Check if history exists (maybe it crashed before best saves?)
        history = os.path.join(exp_dir, "metrics", "epoch_metrics.csv")
        # Legacy location fallback
        legacy_hist = os.path.join(exp_dir, "history.csv")
        
        status = "MISSING_METRICS"
        if os.path.exists(history) or os.path.exists(legacy_hist):
            status = "CRASHED_OR_INCOMPLETE"
            
        print(f"[WARN] {os.path.basename(exp_dir)}: {status} (No best_metrics.json)")
        return None
        
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            data['exp_dir'] = exp_dir
            return data
    except Exception as e:
        print(f"[ERROR] {os.path.basename(exp_dir)}: MALFORMED_JSON ({e})")
        return None

def select_best_run(suite_dir):
    print(f"Scanning suite directory: {os.path.abspath(suite_dir)}")
    
    if not os.path.exists(suite_dir):
        print(f"[FATAL] Suite directory not found: {suite_dir}")
        sys.exit(1)

    exp_dirs = glob.glob(os.path.join(suite_dir, "exp5_*"))
    valid_experiments = []
    
    print(f"Found {len(exp_dirs)} candidate directories.")
    
    for d in exp_dirs:
        if os.path.isdir(d):
            stats = analyze_experiment(d)
            if stats:
                valid_experiments.append(stats)
                
    if not valid_experiments:
        print("[FATAL] No valid experiments found. Check logs for crashes.")
        sys.exit(1)
        
    # Sort by Val MSE (Primary)
    # Tiebreaker: R2 (Descending) -> but currently R2 is 0.0 placeholder.
    # We sort by MSE Ascending.
    sorted_exps = sorted(valid_experiments, key=lambda x: x['best_val_mse'])
    
    best = sorted_exps[0]
    
    print("\n--- Leaderboard ---")
    for i, exp in enumerate(sorted_exps):
        name = os.path.basename(exp['exp_dir'])
        print(f"{i+1}. {name:<20} | Val MSE: {exp['best_val_mse']:.6f} (Epoch {exp['best_epoch']})")
        
    print(f"\nWINNER: {os.path.basename(best['exp_dir'])}")
    
    # Save selection atomically
    output_path = os.path.join(suite_dir, "best_run_selection.json")
    selection_data = {
        "winner_run_id": os.path.basename(best['exp_dir']),
        "winner_metrics": best,
        "candidates": sorted_exps,
        "timestamp": datetime.datetime.now().isoformat(),
        # "git_commit": ... (could inject if passed)
    }
    
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(selection_data, f, indent=2)
    os.rename(tmp_path, output_path)
    
    print(f"Selection saved to {output_path}")
    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, default="outputs_exp5")
    args = parser.parse_args()
    
    select_best_run(args.suite_dir)
