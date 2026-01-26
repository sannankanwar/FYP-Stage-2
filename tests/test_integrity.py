import unittest
import torch
import torch.nn as nn
from src.utils.model_utils import process_predictions
from src.training.loss import Naive5ParamMSELoss
from src.utils.normalization import ParameterNormalizer

class MockModel(nn.Module):
    def __init__(self, output_space):
        super().__init__()
        self.output_space = output_space

class TestSystemIntegrity(unittest.TestCase):
    
    def test_double_denormalization_prevention(self):
        """
        Ensures that if model.output_space == 'physical', we do NOT denormalize,
        even if config says standardize_outputs=True.
        """
        # Setup
        model = MockModel(output_space="physical")
        physical_pred = torch.tensor([[100.0, 100.0, 20.0, 0.5, 50.0]])
        
        # Create a normalizer that would "explode" values
        # Ranges ~ [-500, 500]. Sigma ~ 500. Mean ~ 0.
        # If denormalized: 100 * 500 = 50000.
        ranges = {
            'xc': [-500, 500], 'yc': [-500, 500], 'S': [1, 40],
            'wavelength': [0.4, 0.7], 'focal_length': [10, 100]
        }
        normalizer = ParameterNormalizer(ranges)
        config = {"standardize_outputs": True}
        
        # Execute
        result = process_predictions(model, physical_pred, normalizer, config)
        
        # Assert
        # Should remain 100.0 (Unchanged)
        self.assertTrue(torch.allclose(result, physical_pred), 
                        f"Prediction changed! Expected {physical_pred}, got {result}")
        print("\n[PASS] Physical model + Standardize Flag -> No Denormalization")

    def test_legacy_Unknown_model_denormalization(self):
        """
        Ensures that if model contact is unknown, we CRASH (Safety Fix #2).
        Previous behavior: Guess and denormalize.
        New behavior: RuntimeError.
        """
        class MockModel:
             output_space = "unknown" # Explicitly unknown or missing (setup sets unknown)
             
        model = MockModel()
        
        # Inputs (Normalized)
        norm_pred = torch.randn(2, 5)
        
        # Config says standardized
        config = {"standardize_outputs": True}
        
        # Define ranges for the normalizer
        self.ranges = {
            'xc': [-500, 500], 'yc': [-500, 500], 'S': [1, 40],
            'wavelength': [0.4, 0.7], 'focal_length': [10, 100]
        }
        normalizer = ParameterNormalizer(self.ranges)
        
        # Act & Assert
        with self.assertRaises(RuntimeError) as cm:
             process_predictions(model, norm_pred, normalizer, config)
             
        self.assertIn("Safe processing requires explicit", str(cm.exception))

    def test_naive_loss_is_disabled(self):
        """
        Ensures Naive5ParamMSELoss raises RuntimeError on init.
        """
        try:
            loss = Naive5ParamMSELoss()
            self.fail("Naive5ParamMSELoss did not raise RuntimeError upon instantiation!")
        except RuntimeError as e:
            self.assertIn("removed for safety", str(e))
            print("\n[PASS] Naive5ParamMSELoss correctly raises Tombstone Error")

if __name__ == '__main__':
    unittest.main()
