"""
Unit Tests for Weights & Biases Integration

This module tests the wandb integration helper functions in scripts/utils.py
with proper mocking to avoid actual API calls.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path
import torch
import numpy as np
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import (
    wandb_available,
    init_wandb,
    log_metrics_wandb,
    log_images_wandb,
    log_confusion_matrix_wandb,
    log_gradients_wandb,
    log_model_artifact_wandb,
    finish_wandb
)


class TestWandbAvailability(unittest.TestCase):
    """Test wandb availability detection."""

    @patch('scripts.utils.wandb_available')
    def test_wandb_available_true(self, mock_available):
        """Test when wandb is available."""
        mock_available.return_value = True
        self.assertTrue(mock_available())

    @patch('scripts.utils.wandb_available')
    def test_wandb_available_false(self, mock_available):
        """Test when wandb is not available."""
        mock_available.return_value = False
        self.assertFalse(mock_available())


class TestWandbInitialization(unittest.TestCase):
    """Test wandb initialization with various scenarios."""

    @patch('wandb.init')
    @patch('wandb.run')
    def test_init_wandb_success(self, mock_run, mock_init):
        """Test successful wandb initialization."""
        mock_run_obj = MagicMock()
        mock_run_obj.name = 'test_run'
        mock_run_obj.url = 'http://test.url'
        mock_init.return_value = mock_run_obj
        mock_run.__bool__.return_value = True
        mock_run.name = 'test_run'
        mock_run.url = 'http://test.url'

        config = {'learning_rate': 0.001, 'batch_size': 16}
        result = init_wandb(
            config=config,
            project_name='test_project',
            run_name='test_run',
            tags=['test'],
            enable_wandb=True
        )

        self.assertTrue(result)
        mock_init.assert_called_once_with(
            project='test_project',
            name='test_run',
            config=config,
            tags=['test']
        )

    def test_init_wandb_disabled(self):
        """Test wandb initialization when disabled."""
        config = {'learning_rate': 0.001}
        result = init_wandb(
            config=config,
            enable_wandb=False
        )

        self.assertFalse(result)

    @patch('builtins.__import__', side_effect=ImportError("No module named 'wandb'"))
    def test_init_wandb_import_error(self, mock_import):
        """Test graceful handling of import error."""
        config = {'learning_rate': 0.001}
        result = init_wandb(
            config=config,
            enable_wandb=True
        )

        self.assertFalse(result)

    @patch('wandb.init', side_effect=Exception("Connection error"))
    def test_init_wandb_exception(self, mock_init):
        """Test graceful handling of other exceptions."""
        config = {'learning_rate': 0.001}
        result = init_wandb(
            config=config,
            enable_wandb=True
        )

        self.assertFalse(result)

    @patch('wandb.init')
    @patch('wandb.run')
    def test_init_wandb_default_params(self, mock_run, mock_init):
        """Test initialization with default parameters."""
        mock_run_obj = MagicMock()
        mock_run_obj.name = 'default_run'
        mock_run_obj.url = 'http://test.url'
        mock_init.return_value = mock_run_obj
        mock_run.name = 'default_run'

        config = {'test': 'value'}
        result = init_wandb(config=config, enable_wandb=True)

        self.assertTrue(result)
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        self.assertEqual(call_kwargs['project'], 'diabetic-retinopathy')
        self.assertEqual(call_kwargs['config'], config)


class TestWandbMetricsLogging(unittest.TestCase):
    """Test metrics logging to wandb."""

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.log')
    def test_log_metrics_success(self, mock_log, mock_available):
        """Test successful metrics logging."""
        metrics = {
            'train_loss': 0.5,
            'train_acc': 0.85,
            'val_loss': 0.6,
            'val_acc': 0.82
        }

        log_metrics_wandb(metrics, step=10)

        mock_log.assert_called_once_with(metrics, step=10)

    @patch('scripts.utils.wandb_available', return_value=False)
    def test_log_metrics_wandb_unavailable(self, mock_available):
        """Test logging when wandb unavailable."""
        metrics = {'loss': 0.5}
        # Should not raise error
        log_metrics_wandb(metrics, step=5)

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.log')
    def test_log_metrics_with_prefix(self, mock_log, mock_available):
        """Test metrics logging with prefix."""
        metrics = {'loss': 0.5, 'acc': 0.85}
        log_metrics_wandb(metrics, step=5, prefix='train/')

        expected_metrics = {'train/loss': 0.5, 'train/acc': 0.85}
        mock_log.assert_called_once_with(expected_metrics, step=5)

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.log', side_effect=Exception("API error"))
    def test_log_metrics_exception_handling(self, mock_log, mock_available):
        """Test graceful handling of logging exceptions."""
        metrics = {'loss': 0.5}
        # Should not raise error
        log_metrics_wandb(metrics, step=1)


class TestWandbImageLogging(unittest.TestCase):
    """Test image logging to wandb."""

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Image')
    @patch('wandb.log')
    def test_log_images_success(self, mock_log, mock_image, mock_available):
        """Test successful image logging."""
        mock_image.return_value = "mock_image"

        # Create dummy tensors
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        predictions = torch.tensor([0, 1, 3, 3])
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

        log_images_wandb(
            images=images,
            labels=labels,
            predictions=predictions,
            class_names=class_names,
            step=10,
            max_images=4
        )

        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][0]
        self.assertIn('predictions', call_args)

    @patch('scripts.utils.wandb_available', return_value=False)
    def test_log_images_wandb_unavailable(self, mock_available):
        """Test image logging when wandb unavailable."""
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        predictions = torch.tensor([0, 1])
        class_names = ['A', 'B']

        # Should not raise error
        log_images_wandb(images, labels, predictions, class_names)

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Image')
    @patch('wandb.log')
    def test_log_images_denormalization(self, mock_log, mock_image, mock_available):
        """Test image denormalization before logging."""
        mock_image.return_value = "mock_image"

        # Create normalized images
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        predictions = torch.tensor([0, 1])
        class_names = ['A', 'B']

        log_images_wandb(
            images=images,
            labels=labels,
            predictions=predictions,
            class_names=class_names,
            denormalize=True
        )

        # Verify Image was called (denormalization happened)
        self.assertTrue(mock_image.called)

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Image')
    @patch('wandb.log')
    def test_log_images_max_images_limit(self, mock_log, mock_image, mock_available):
        """Test that max_images limit is respected."""
        mock_image.return_value = "mock_image"

        images = torch.randn(10, 3, 224, 224)
        labels = torch.tensor([0] * 10)
        predictions = torch.tensor([0] * 10)
        class_names = ['A']

        log_images_wandb(
            images=images,
            labels=labels,
            predictions=predictions,
            class_names=class_names,
            max_images=5
        )

        # Should only log 5 images
        self.assertEqual(mock_image.call_count, 5)


class TestWandbConfusionMatrix(unittest.TestCase):
    """Test confusion matrix logging."""

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Image')
    @patch('wandb.log')
    @patch('scripts.utils.plt.close')
    @patch('scripts.utils.plt.savefig')
    @patch('scripts.utils.plt.tight_layout')
    @patch('scripts.utils.plt.subplots')
    @patch('scripts.utils.sns.heatmap')
    def test_log_confusion_matrix_success(self, mock_heatmap, mock_subplots, mock_tight,
                                         mock_savefig, mock_close, mock_log, mock_image,
                                         mock_available):
        """Test successful confusion matrix logging."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_image.return_value = "mock_image"

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        class_names = ['No DR', 'Mild', 'Moderate']

        log_confusion_matrix_wandb(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            step=100
        )

        mock_log.assert_called_once()
        mock_close.assert_called_once()

    @patch('scripts.utils.wandb_available', return_value=False)
    def test_log_confusion_matrix_wandb_unavailable(self, mock_available):
        """Test confusion matrix logging when wandb unavailable."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        class_names = ['A', 'B']

        # Should not raise error
        log_confusion_matrix_wandb(y_true, y_pred, class_names)

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Image')
    @patch('wandb.log')
    @patch('scripts.utils.plt.close')
    @patch('scripts.utils.plt.savefig')
    @patch('scripts.utils.plt.tight_layout')
    @patch('scripts.utils.plt.subplots')
    @patch('scripts.utils.sns.heatmap')
    def test_log_confusion_matrix_normalize(self, mock_heatmap, mock_subplots, mock_tight,
                                           mock_savefig, mock_close, mock_log, mock_image,
                                           mock_available):
        """Test confusion matrix with normalization."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_image.return_value = "mock_image"

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        class_names = ['A', 'B', 'C']

        log_confusion_matrix_wandb(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            normalize=True
        )

        # Verify logging happened
        mock_log.assert_called_once()


class TestWandbGradients(unittest.TestCase):
    """Test gradient logging."""

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.log')
    def test_log_gradients_success(self, mock_log, mock_available):
        """Test successful gradient logging."""
        # Create a simple model with gradients
        model = torch.nn.Linear(10, 5)
        # Simulate gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        log_gradients_wandb(model, step=5)

        # Verify metrics were logged
        mock_log.assert_called()

    @patch('scripts.utils.wandb_available', return_value=False)
    def test_log_gradients_wandb_unavailable(self, mock_available):
        """Test gradient logging when wandb unavailable."""
        model = torch.nn.Linear(10, 5)
        # Should not raise error
        log_gradients_wandb(model)

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.log')
    def test_log_gradients_no_gradients(self, mock_log, mock_available):
        """Test gradient logging when no gradients exist."""
        model = torch.nn.Linear(10, 5)
        # No gradients computed

        # Should not raise error
        log_gradients_wandb(model, step=1)


class TestWandbModelArtifacts(unittest.TestCase):
    """Test model artifact logging."""

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Artifact')
    @patch('wandb.log_artifact')
    def test_log_model_artifact_success(self, mock_log_artifact, mock_artifact_class, mock_available):
        """Test successful model artifact logging."""
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact

        # Create temporary model file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
            torch.save({'model': 'data'}, f.name)

        try:
            metadata = {'epoch': 10, 'val_acc': 0.85}
            log_model_artifact_wandb(
                model_path=model_path,
                artifact_name='best_model',
                metadata=metadata
            )

            mock_artifact_class.assert_called_once()
            mock_artifact.add_file.assert_called_once_with(model_path)
            mock_log_artifact.assert_called_once_with(mock_artifact)
        finally:
            # Cleanup
            Path(model_path).unlink()

    @patch('scripts.utils.wandb_available', return_value=False)
    def test_log_model_artifact_wandb_unavailable(self, mock_available):
        """Test artifact logging when wandb unavailable."""
        # Should not raise error
        log_model_artifact_wandb('fake_path.pth', 'artifact')

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.Artifact')
    def test_log_model_artifact_file_not_found(self, mock_artifact, mock_available):
        """Test artifact logging with non-existent file."""
        # Should not raise error (graceful handling)
        log_model_artifact_wandb('nonexistent.pth', 'artifact')


class TestWandbFinish(unittest.TestCase):
    """Test wandb finish/cleanup."""

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.finish')
    @patch('wandb.run', MagicMock())
    def test_finish_wandb_success(self, mock_finish, mock_available):
        """Test successful wandb finish."""
        finish_wandb()

        mock_finish.assert_called_once()

    @patch('scripts.utils.wandb_available', return_value=False)
    def test_finish_wandb_unavailable(self, mock_available):
        """Test finish when wandb unavailable."""
        # Should not raise error
        finish_wandb()

    @patch('scripts.utils.wandb_available', return_value=True)
    @patch('wandb.finish', side_effect=Exception("Finish error"))
    @patch('wandb.run', MagicMock())
    def test_finish_wandb_exception(self, mock_finish, mock_available):
        """Test graceful handling of finish exceptions."""
        # Should not raise error
        finish_wandb()


class TestWandbGracefulFallback(unittest.TestCase):
    """Test that all functions handle missing wandb gracefully."""

    @patch('scripts.utils.wandb_available')
    def test_all_functions_with_unavailable_wandb(self, mock_available):
        """Test that no function raises error when wandb unavailable."""
        mock_available.return_value = False

        # Test all functions - none should raise errors
        try:
            init_wandb({'test': 'config'}, enable_wandb=True)
            log_metrics_wandb({'loss': 0.5}, step=1)

            images = torch.randn(2, 3, 224, 224)
            labels = torch.tensor([0, 1])
            preds = torch.tensor([0, 1])
            log_images_wandb(images, labels, preds, ['A', 'B'])

            log_confusion_matrix_wandb(
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
                ['A', 'B']
            )

            model = torch.nn.Linear(10, 5)
            log_gradients_wandb(model)

            log_model_artifact_wandb('fake.pth', 'artifact')
            finish_wandb()

            # If we get here, all functions handled missing wandb gracefully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Function raised exception with unavailable wandb: {e}")


class TestWandbIntegrationE2E(unittest.TestCase):
    """End-to-end integration tests."""

    @patch('wandb.finish')
    @patch('wandb.Image')
    @patch('wandb.log')
    @patch('wandb.init')
    @patch('wandb.run')
    @patch('scripts.utils.plt.close')
    @patch('scripts.utils.plt.savefig')
    @patch('scripts.utils.plt.tight_layout')
    @patch('scripts.utils.plt.subplots')
    @patch('scripts.utils.sns.heatmap')
    def test_typical_training_workflow(self, mock_heatmap, mock_plt_subplots, mock_plt_tight,
                                      mock_plt_savefig, mock_plt_close, mock_run, mock_init,
                                      mock_log, mock_image, mock_finish):
        """Test a typical training workflow with wandb."""
        # Setup mocks
        mock_run_obj = MagicMock()
        mock_run_obj.name = 'test_run'
        mock_run_obj.url = 'http://test.url'
        mock_init.return_value = mock_run_obj
        mock_run.name = 'test_run'
        mock_image.return_value = "mock_image"

        # Mock matplotlib
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt_subplots.return_value = (mock_fig, mock_ax)

        # 1. Initialize
        config = {'lr': 0.001, 'batch_size': 16}
        init_result = init_wandb(config, enable_wandb=True)
        self.assertTrue(init_result)

        # 2. Log metrics for several epochs
        for epoch in range(3):
            metrics = {
                'train_loss': 0.5 - epoch * 0.1,
                'val_loss': 0.6 - epoch * 0.1,
                'train_acc': 0.7 + epoch * 0.05,
                'val_acc': 0.65 + epoch * 0.05
            }
            log_metrics_wandb(metrics, step=epoch)

        # 3. Log images
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        preds = torch.tensor([0, 1, 1, 3])
        log_images_wandb(images, labels, preds, ['A', 'B', 'C', 'D'])

        # 4. Log confusion matrix
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        log_confusion_matrix_wandb(y_true, y_pred, ['A', 'B', 'C'])

        # 5. Finish
        finish_wandb()

        # Verify all steps executed
        mock_init.assert_called_once()
        self.assertEqual(mock_log.call_count, 5)  # 3 metric logs + 1 image + 1 confusion matrix
        mock_finish.assert_called_once()


def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("Running Wandb Integration Tests")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWandbAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbMetricsLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbImageLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbConfusionMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbGradients))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbModelArtifacts))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbFinish))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbGracefulFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestWandbIntegrationE2E))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
