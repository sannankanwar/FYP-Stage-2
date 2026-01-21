import os
import torch
import numpy as np
from data.loaders.simulation import generate_grid_dataset

class MetalensEvaluator:
    """
    Handles tiered evaluation using deterministic grids to identify model blind spots.
    """
    def __init__(self, model, device="cpu", output_dir="outputs/evaluation"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

    @torch.no_grad()
    def run_grid_eval(self, xc_count, yc_count, fovs, N=128, stage_name="fast"):
        """
        Runs evaluation over a grid for multiple FOVs, clearing memory after each FOV.
        """
        all_results = {}

        for fov in fovs:
            print(f"[{stage_name}] Evaluating FOV: {fov} (Grid: {xc_count}x{yc_count})...")
            
            # 1. Generate grid data for this FOV
            X, y_true, metadata = generate_grid_dataset(
                xc_count=xc_count,
                yc_count=yc_count,
                fov=fov,
                N=N
            )

            # 2. Run Inference
            X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).to(self.device).float()
            y_pred_tensor = self.model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()

            # 3. Compute Metrics (Error Heatmap)
            # Targets are [xc, yc, fov]
            errors = np.abs(y_pred - y_true)
            
            # Reshape errors back to grid for heatmap visualization
            grid_shape = metadata["grid_shape"]
            error_heatmaps = {
                "xc_error": errors[:, 0].reshape(grid_shape),
                "yc_error": errors[:, 1].reshape(grid_shape),
                "fov_error": errors[:, 2].reshape(grid_shape),
                "total_mse": np.mean(errors**2, axis=1).reshape(grid_shape)
            }

            results = {
                "metadata": metadata,
                "heatmaps": error_heatmaps
            }

            # 4. Save results for this FOV
            save_path = os.path.join(self.output_dir, f"{stage_name}_fov_{fov}.npy")
            np.save(save_path, results)
            
            all_results[fov] = save_path

            # 5. Clear memory explicitly
            del X, X_tensor, y_pred, y_true, errors
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return all_results

    def run_fast_eval(self, fovs):
        return self.run_grid_eval(xc_count=10, yc_count=10, fovs=fovs, stage_name="fast")

    def run_slow_eval(self, fovs):
        return self.run_grid_eval(xc_count=25, yc_count=25, fovs=fovs, stage_name="slow")

    def run_final_eval(self, fovs):
        return self.run_grid_eval(xc_count=50, yc_count=50, fovs=fovs, stage_name="final")
