import unittest
import torch
import yaml
from src.models.fno.fno_resnet18 import FNOResNet18
from src.training.trainer import Trainer
from src.utils.normalization import ParameterNormalizer
from torch.utils.data import DataLoader, TensorDataset

class TestLossVariants(unittest.TestCase):
    def setUp(self):
        # Create Dummy Model
        self.model = FNOResNet18(in_channels=2, output_dim=5, modes=16, 
                                 xc_range=(-500,500), yc_range=(-500,500),
                                 S_range=(1,40), wavelength_range=(0.4,0.7),
                                 focal_length_range=(10,100))
        
        # Dummy Data
        self.B, self.H, self.W = 2, 64, 64
        self.inputs = torch.randn(self.B, 2, self.H, self.W)
        # Random physical targets
        self.targets = torch.tensor([
            [100.0, 100.0, 20.0, 0.55, 50.0],
            [-200.0, -200.0, 10.0, 0.65, 80.0]
        ])
        
        # Loader
        ds = TensorDataset(self.inputs, self.targets)
        # Attach range attributes to dataset for Trainer
        ds.xc_range = (-500,500)
        ds.yc_range = (-500,500)
        ds.S_range = (1,40)
        ds.wavelength_range = (0.4,0.7)
        ds.focal_length_range = (10,100)
        
        self.loader = DataLoader(ds, batch_size=2)
        
    def test_run_config(self, config_updates):
        """Helper to run one step of training with a specific config"""
        full_config = {
            "name": "fno_resnet18",
            "standardize_outputs": True,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "experiment_name": "test_loss",
            "output_dir": "test_outputs",
            "scheduler": None
        }
        full_config.update(config_updates)
        
        # Init Trainer
        trainer = Trainer(full_config, self.model, self.loader)
        
        # Force 1 step
        trainer.model.train()
        data, target = next(iter(self.loader))
        data, target = data.to(trainer.device), target.to(trainer.device)
        
        trainer.optimizer.zero_grad()
        output = trainer.model(data)
        
        # Loss
        loss_func = trainer.criterion
        # Check signature support
        # We need to manually call it like Trainer does to test Trainer's logic??
        # Usually Trainer handles the signature check.
        # But here we want to verifying gradients.
        
        # Replicate Trainer loop logic
        # Assuming Trainer loop logic is correct, we just call _train_epoch or run a manual forward/backward
        
        # Manual execution to inspect gradients:
        if config_updates['loss_function'] in ["gradient_consistency", "kendall", "pinn"]:
             loss, details = loss_func(output, target, data)
        else:
             loss, details = loss_func(output, target, data)
             
        loss.backward()
        trainer.optimizer.step()
        
        # Check Gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has no gradient!")
                
        # For Kendall, check loss params
        if config_updates['loss_function'] == "kendall":
             loss_params = list(loss_func.parameters())
             self.assertTrue(len(loss_params) > 0, "Kendall loss should have parameters")
             self.assertIsNotNone(loss_params[0].grad, "Kendall loss params have no gradient!")
             
        print(f"\n[PASS] Loss {config_updates['loss_function']} -> val={loss.item():.4f}")

    def test_all_variants(self):
        configs = [
            {"loss_function": "weighted_standardized"},
            {"loss_function": "gradient_consistency", "gradient_weight": 0.5},
            {"loss_function": "kendall", "init_log_var": 0.0},
            {"loss_function": "pinn", "pinn_weight": 0.1}
        ]
        
        for c in configs:
            with self.subTest(loss=c['loss_function']):
                # Re-init model each time to start fresh
                self.setUp()
                self.test_run_config(c)

if __name__ == '__main__':
    unittest.main()
