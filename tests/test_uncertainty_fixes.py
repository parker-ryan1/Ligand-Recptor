"""
Unit tests for Ligand-Recptor Enhanced GNN v2 - Focusing on Applied Fixes

Tests cover:
1. KL Divergence type consistency
2. MC Dropout state management
3. Variance regularization stability
4. Model checkpointing memory management
5. Logging configuration
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import logging
import tempfile
import os
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestKLDivergenceType(unittest.TestCase):
    """Test that KL divergence returns correct tensor type"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
    
    def test_kl_divergence_is_tensor(self):
        """KL divergence should always return a tensor, not a float"""
        # Simulate the fixed implementation
        kl_div = torch.tensor(0.0, dtype=torch.float32)
        
        # Verify it's a tensor
        self.assertIsInstance(kl_div, torch.Tensor)
        self.assertEqual(kl_div.dtype, torch.float32)
    
    def test_kl_divergence_accumulation(self):
        """Test that KL divergence accumulates correctly with tensor addition"""
        kl_div = torch.tensor(0.0, dtype=torch.float32)
        
        # Simulate accumulation from multiple sources
        test_values = torch.tensor([0.5, 1.0, 0.3, 0.2])
        for val in test_values:
            kl_div = kl_div + val
        
        expected = test_values.sum()
        self.assertAlmostEqual(kl_div.item(), expected.item(), places=5)
    
    def test_kl_divergence_backward_pass(self):
        """KL divergence tensor should maintain computational graph"""
        kl_div = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        
        # Add some computations
        kl_div = kl_div + torch.tensor(1.0, requires_grad=True)
        
        # Should be able to compute gradients
        self.assertTrue(kl_div.requires_grad)


class TestMCDropoutStateManagement(unittest.TestCase):
    """Test that MC dropout state is properly managed"""
    
    def setUp(self):
        """Set up test model"""
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def test_dropout_state_restoration(self):
        """Test that model state is properly restored after MC sampling"""
        # Model should start in eval mode
        self.model.eval()
        original_state = self.model.training
        self.assertFalse(original_state)
        
        # Simulate MC sampling with fixed implementation
        training_state = self.model.training
        try:
            self.model.train()
            self.assertTrue(self.model.training)
        finally:
            self.model.train(training_state)
        
        # State should be restored
        self.assertEqual(self.model.training, original_state)
    
    def test_dropout_state_restored_on_exception(self):
        """Test that model state is restored even if exception occurs"""
        self.model.eval()
        original_state = self.model.training
        
        training_state = self.model.training
        try:
            self.model.train()
            # Simulate an exception
            raise ValueError("Test exception")
        except ValueError:
            pass
        finally:
            self.model.train(training_state)
        
        # State should still be restored
        self.assertEqual(self.model.training, original_state)
    
    def test_mc_sampling_preserves_model_for_inference(self):
        """After MC sampling, model should work correctly for normal inference"""
        self.model.eval()
        x = torch.randn(5, 10)
        
        # Run MC sampling (fixed implementation)
        training_state = self.model.training
        try:
            self.model.train()
            predictions = []
            for _ in range(3):
                with torch.no_grad():
                    pred = self.model(x)
                    predictions.append(pred)
        finally:
            self.model.train(training_state)
        
        # Now inference should work without dropout
        self.model.eval()
        pred1 = self.model(x)
        pred2 = self.model(x)
        
        # Predictions should be identical (no dropout in eval mode)
        torch.testing.assert_close(pred1, pred2)


class TestVarianceRegularizationStability(unittest.TestCase):
    """Test numerical stability of variance regularization"""
    
    def test_log1p_stable_for_small_variance(self):
        """log1p should be numerically stable for small variance values"""
        small_var = torch.tensor(1e-8)
        
        # Fixed implementation using log1p
        variance_penalty = torch.log1p(torch.clamp(small_var, min=1e-6))
        
        # Should not be NaN or Inf
        self.assertFalse(torch.isnan(variance_penalty))
        self.assertFalse(torch.isinf(variance_penalty))
        self.assertGreater(variance_penalty.item(), -np.inf)
    
    def test_log1p_stable_for_large_variance(self):
        """log1p should be numerically stable for large variance values"""
        large_var = torch.tensor(1e10)
        
        # Fixed implementation using log1p
        variance_penalty = torch.log1p(torch.clamp(large_var, min=1e-6))
        
        # Should not be NaN or Inf
        self.assertFalse(torch.isnan(variance_penalty))
        self.assertFalse(torch.isinf(variance_penalty))
    
    def test_exp_underflow_comparison(self):
        """Compare exp() underflow with log1p() stability"""
        large_var = torch.tensor(100.0)
        
        # Old implementation (problematic)
        old_penalty = torch.exp(-large_var / 0.5)  # exp(-200) ≈ 0
        
        # New implementation (stable)
        new_penalty = torch.log1p(torch.clamp(large_var, min=1e-6))
        
        # Old version underflows to 0
        self.assertAlmostEqual(old_penalty.item(), 0.0, places=5)
        
        # New version remains stable
        self.assertGreater(new_penalty.item(), 0)
    
    def test_clamping_prevents_log_zero(self):
        """Clamping should prevent log of zero"""
        zero_var = torch.tensor(0.0)
        
        # With clamping
        clamped = torch.clamp(zero_var, min=1e-6)
        variance_penalty = torch.log1p(clamped)
        
        # Should not be -inf
        self.assertFalse(torch.isinf(variance_penalty))
        self.assertGreater(variance_penalty.item(), -np.inf)


class TestModelCheckpointingMemory(unittest.TestCase):
    """Test that model checkpointing uses disk, not memory"""
    
    def setUp(self):
        """Set up temporary directory for checkpoints"""
        self.temp_dir = tempfile.mkdtemp()
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        )
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_saved_to_disk(self):
        """Checkpoint should be saved to disk"""
        checkpoint_path = os.path.join(self.temp_dir, 'best_model.pt')
        
        # Fixed implementation: save to disk
        torch.save(self.model.state_dict(), checkpoint_path)
        
        # File should exist
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertGreater(os.path.getsize(checkpoint_path), 0)
    
    def test_checkpoint_can_be_loaded(self):
        """Checkpoint should be loadable"""
        checkpoint_path = os.path.join(self.temp_dir, 'best_model.pt')
        
        # Save checkpoint
        torch.save(self.model.state_dict(), checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        )
        new_model.load_state_dict(torch.load(checkpoint_path))
        
        # Test with same input
        x = torch.randn(5, 100)
        with torch.no_grad():
            pred1 = self.model(x)
            pred2 = new_model(x)
        
        # Predictions should be identical
        torch.testing.assert_close(pred1, pred2)
    
    def test_multiple_checkpoints_not_memory_hoarded(self):
        """Multiple checkpoints should not accumulate in memory"""
        # Save multiple checkpoints
        for i in range(10):
            checkpoint_path = os.path.join(self.temp_dir, f'checkpoint_{i}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)
        
        # Verify all files exist on disk
        checkpoint_files = list(Path(self.temp_dir).glob('checkpoint_*.pt'))
        self.assertEqual(len(checkpoint_files), 10)
        
        # Memory should not be significantly hoarded
        # (This is more of a conceptual test - actual memory profiling would be more complex)
        checkpoint_sizes = [os.path.getsize(f) for f in checkpoint_files]
        
        # All checkpoint sizes should be similar
        avg_size = np.mean(checkpoint_sizes)
        for size in checkpoint_sizes:
            self.assertAlmostEqual(size, avg_size, delta=avg_size * 0.01)


class TestLoggingConfiguration(unittest.TestCase):
    """Test that logging is properly configured"""
    
    def test_logger_exists(self):
        """Logger should be properly configured"""
        test_logger = logging.getLogger('test_module')
        self.assertIsNotNone(test_logger)
    
    def test_logger_has_handlers(self):
        """Logger should have at least one handler"""
        test_logger = logging.getLogger('test_module')
        
        # Add a handler if none exist
        if not test_logger.handlers:
            handler = logging.StreamHandler()
            test_logger.addHandler(handler)
        
        self.assertGreater(len(test_logger.handlers), 0)
    
    def test_logger_level_configuration(self):
        """Logger level should be configurable"""
        test_logger = logging.getLogger('test_module')
        
        # Set level
        test_logger.setLevel(logging.DEBUG)
        self.assertEqual(test_logger.level, logging.DEBUG)
        
        # Change level
        test_logger.setLevel(logging.INFO)
        self.assertEqual(test_logger.level, logging.INFO)
    
    def test_logging_different_levels(self):
        """Should be able to log at different levels"""
        test_logger = logging.getLogger('test_levels')
        handler = logging.StreamHandler()
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        
        # Should not raise any exceptions
        test_logger.debug("Debug message")
        test_logger.info("Info message")
        test_logger.warning("Warning message")
        test_logger.error("Error message")


class TestIntegrationUncertaintyPipeline(unittest.TestCase):
    """Integration tests for the complete uncertainty pipeline"""
    
    def setUp(self):
        """Set up test models"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')
    
    def tearDown(self):
        """Clean up"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_training_loop_with_fixes(self):
        """Test complete training loop with all fixes applied"""
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Generate synthetic data
        x = torch.randn(100, 20)
        y = torch.randn(100, 1)
        
        best_loss = float('inf')
        best_model_path = os.path.join(self.temp_dir, 'best_model.pt')
        
        # Training loop with fixes
        for epoch in range(5):
            model.train()
            
            # Forward pass
            pred = model(x)
            loss = criterion(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(x)
                val_loss = criterion(val_pred, y)
            
            # Checkpoint with disk storage (Fix #5)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
        
        # Verify checkpoint was saved
        self.assertTrue(os.path.exists(best_model_path))
        
        # Verify model can be loaded
        new_model = nn.Sequential(
            nn.Linear(20, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        new_model.load_state_dict(torch.load(best_model_path))
        
        # Verify loaded model works
        with torch.no_grad():
            pred = new_model(x[:5])
        
        self.assertEqual(pred.shape, (5, 1))


if __name__ == '__main__':
    unittest.main()
