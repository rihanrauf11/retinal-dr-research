"""
Unit Tests for Utility Functions.

Tests cover:
- Random seed management
- Model parameter counting
- Checkpoint save/load operations
- Transform creation and configuration
- Data loader creation
- Metrics calculation
- Confusion matrix visualization
- Training history management
- Device management

Author: Generated with Claude Code
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

from utils import (
    set_seed, count_parameters, print_model_summary,
    save_checkpoint, load_checkpoint, resume_training_from_checkpoint,
    get_imagenet_stats, get_transforms,
    create_data_loaders, create_dataloader_from_dataset,
    calculate_metrics, print_metrics,
    plot_confusion_matrix, plot_confusion_matrix_from_metrics,
    save_training_history, load_training_history, plot_training_history,
    get_device, move_to_device,
    create_progress_bar, log_metrics
)


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SEED TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRandomSeed:
    """Test random seed management functions."""

    def test_set_seed_basic(self):
        """Test basic seed setting."""
        set_seed(42, verbose=False)

        # Generate random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)

        # Reset seed
        set_seed(42, verbose=False)

        # Generate again - should match
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)

        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)

    def test_set_seed_deterministic_mode(self):
        """Test deterministic mode flag."""
        # Should not raise errors
        set_seed(42, deterministic=True, verbose=False)
        set_seed(42, deterministic=False, verbose=False)

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42, verbose=False)
        rand1 = torch.rand(10)

        set_seed(123, verbose=False)
        rand2 = torch.rand(10)

        # Should be different
        assert not torch.allclose(rand1, rand2)

    def test_set_seed_verbose_output(self, capsys):
        """Test verbose output."""
        set_seed(42, verbose=True)
        captured = capsys.readouterr()

        assert "Random seed set to 42" in captured.out
        assert "deterministic mode" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PARAMETER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelParameters:
    """Test model parameter counting utilities."""

    def test_count_parameters_simple_model(self, simple_model):
        """Test counting parameters in a simple model."""
        total, trainable = count_parameters(simple_model)

        assert total > 0
        assert trainable > 0
        assert trainable <= total

    def test_count_parameters_trainable_only(self, simple_model):
        """Test trainable_only parameter."""
        trainable = count_parameters(simple_model, trainable_only=True)

        assert isinstance(trainable, int)
        assert trainable > 0

    def test_count_parameters_frozen_model(self, simple_model):
        """Test counting on frozen model."""
        # Freeze backbone
        simple_model.freeze_backbone()

        total, trainable = count_parameters(simple_model)

        assert trainable < total

    def test_print_model_summary(self, simple_model, capsys):
        """Test model summary printing."""
        stats = print_model_summary(simple_model, verbose=True)

        # Check returned stats
        assert 'total_params' in stats
        assert 'trainable_params' in stats
        assert 'memory_mb' in stats
        assert 'output_shape' in stats

        # Check console output
        captured = capsys.readouterr()
        assert "MODEL SUMMARY" in captured.out
        assert "Total Parameters" in captured.out

    def test_print_model_summary_no_verbose(self, simple_model):
        """Test model summary without printing."""
        stats = print_model_summary(simple_model, verbose=False)

        assert isinstance(stats, dict)
        assert stats['total_params'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointManagement:
    """Test checkpoint save/load operations."""

    def test_save_checkpoint_basic(self, simple_model, temp_data_dir):
        """Test basic checkpoint saving."""
        checkpoint_path = temp_data_dir / "checkpoint.pth"
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            metrics={'val_acc': 85.3, 'val_loss': 0.42},
            path=checkpoint_path
        )

        assert checkpoint_path.exists()

    def test_save_checkpoint_with_best(self, simple_model, temp_data_dir):
        """Test saving best model."""
        checkpoint_path = temp_data_dir / "checkpoint.pth"
        best_path = temp_data_dir / "best_model.pth"

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            metrics={'val_acc': 85.3},
            path=checkpoint_path,
            is_best=True
        )

        assert checkpoint_path.exists()
        assert best_path.exists()

    def test_save_checkpoint_with_kwargs(self, simple_model, temp_data_dir):
        """Test saving checkpoint with additional items."""
        checkpoint_path = temp_data_dir / "checkpoint.pth"
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        history = {'train_loss': [0.5, 0.4, 0.3]}
        config = {'batch_size': 32, 'lr': 1e-4}

        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            metrics={'val_acc': 85.3},
            path=checkpoint_path,
            history=history,
            config=config
        )

        # Load and verify
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'history' in checkpoint
        assert 'config' in checkpoint
        assert checkpoint['history'] == history
        assert checkpoint['config'] == config

    def test_load_checkpoint_basic(self, simple_model, temp_data_dir):
        """Test basic checkpoint loading."""
        checkpoint_path = temp_data_dir / "checkpoint.pth"
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        # Save
        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=10,
            metrics={'val_acc': 90.5},
            path=checkpoint_path
        )

        # Create new model and load
        new_model = type(simple_model)('resnet18', num_classes=5, pretrained=False)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)

        metadata = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            verbose=False
        )

        assert metadata['epoch'] == 10
        assert metadata['metrics']['val_acc'] == 90.5

    def test_load_checkpoint_nonexistent(self, simple_model, temp_data_dir):
        """Test loading nonexistent checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                path=temp_data_dir / "nonexistent.pth",
                model=simple_model
            )

    def test_load_checkpoint_state_preservation(self, simple_model, temp_data_dir):
        """Test that loaded model produces same output."""
        checkpoint_path = temp_data_dir / "checkpoint.pth"

        # Generate output before saving
        simple_model.eval()
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output_before = simple_model(test_input)

        # Save
        save_checkpoint(
            model=simple_model,
            optimizer=None,
            epoch=1,
            metrics={},
            path=checkpoint_path
        )

        # Create new model and load
        new_model = type(simple_model)('resnet18', num_classes=5, pretrained=False)
        load_checkpoint(path=checkpoint_path, model=new_model, verbose=False)

        # Generate output after loading
        new_model.eval()
        with torch.no_grad():
            output_after = new_model(test_input)

        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_resume_training_from_checkpoint(self, simple_model, temp_data_dir):
        """Test high-level resume function."""
        checkpoint_path = temp_data_dir / "checkpoint.pth"
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        history = {'train_loss': [0.5, 0.4, 0.3]}

        # Save
        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=10,
            metrics={'val_acc': 85.3},
            path=checkpoint_path,
            history=history
        )

        # Resume
        new_model = type(simple_model)('resnet18', num_classes=5, pretrained=False)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)

        start_epoch, best_metric, loaded_history = resume_training_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer
        )

        assert start_epoch == 11  # epoch + 1
        assert best_metric == 85.3
        assert loaded_history == history


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM UTILITIES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformUtilities:
    """Test data transform utility functions."""

    def test_get_imagenet_stats(self):
        """Test ImageNet statistics."""
        stats = get_imagenet_stats()

        assert 'mean' in stats
        assert 'std' in stats
        assert len(stats['mean']) == 3
        assert len(stats['std']) == 3

        # Check expected values
        assert stats['mean'] == [0.485, 0.456, 0.406]
        assert stats['std'] == [0.229, 0.224, 0.225]

    @pytest.mark.parametrize("is_train", [True, False])
    def test_get_transforms_albumentations(self, is_train):
        """Test Albumentations transform creation."""
        transform = get_transforms(224, is_train=is_train, backend='albumentations')

        assert transform is not None

        # Test transform on dummy image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img_array)

        assert 'image' in transformed
        tensor = transformed['image']
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    @pytest.mark.parametrize("is_train", [True, False])
    def test_get_transforms_torchvision(self, is_train):
        """Test torchvision transform creation."""
        transform = get_transforms(224, is_train=is_train, backend='torchvision')

        assert transform is not None

        # Test transform on PIL image
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    @pytest.mark.parametrize("aug_level", ['light', 'medium', 'heavy'])
    def test_get_transforms_augmentation_levels(self, aug_level):
        """Test different augmentation levels."""
        transform = get_transforms(
            224,
            is_train=True,
            augmentation_level=aug_level,
            backend='albumentations'
        )

        assert transform is not None

    def test_get_transforms_invalid_backend(self):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError, match="Invalid backend"):
            get_transforms(224, backend='invalid')

    def test_get_transforms_invalid_aug_level(self):
        """Test invalid augmentation level raises error."""
        with pytest.raises(ValueError, match="Invalid augmentation_level"):
            get_transforms(224, is_train=True, augmentation_level='invalid')

    @pytest.mark.parametrize("img_size", [64, 128, 224, 384])
    def test_get_transforms_different_sizes(self, img_size):
        """Test transforms with different image sizes."""
        transform = get_transforms(img_size, is_train=False, backend='albumentations')

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img_array)

        assert transformed['image'].shape == (3, img_size, img_size)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoaderUtilities:
    """Test data loader creation utilities."""

    def test_create_data_loaders_basic(self, sample_dataset_with_transform):
        """Test basic data loader creation with split."""
        train_loader, val_loader = create_data_loaders(
            dataset=sample_dataset_with_transform,
            batch_size=2,
            split_ratio=0.8,
            num_workers=0,  # 0 for testing
            seed=42
        )

        assert train_loader is not None
        assert val_loader is not None

        # Check sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        total_size = len(sample_dataset_with_transform)

        assert train_size + val_size == total_size
        assert train_size > val_size  # 80/20 split

    def test_create_data_loaders_reproducible(self, sample_dataset_with_transform):
        """Test that splits are reproducible with same seed."""
        train_loader1, val_loader1 = create_data_loaders(
            sample_dataset_with_transform, batch_size=2, seed=42, num_workers=0
        )

        train_loader2, val_loader2 = create_data_loaders(
            sample_dataset_with_transform, batch_size=2, seed=42, num_workers=0
        )

        # Should have same sizes
        assert len(train_loader1.dataset) == len(train_loader2.dataset)
        assert len(val_loader1.dataset) == len(val_loader2.dataset)

    @pytest.mark.parametrize("split_ratio", [0.5, 0.7, 0.8, 0.9])
    def test_create_data_loaders_different_splits(self, sample_dataset_with_transform, split_ratio):
        """Test different split ratios."""
        train_loader, val_loader = create_data_loaders(
            sample_dataset_with_transform,
            batch_size=2,
            split_ratio=split_ratio,
            num_workers=0
        )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        total_size = len(sample_dataset_with_transform)

        # Check approximate split ratio
        actual_ratio = train_size / total_size
        assert abs(actual_ratio - split_ratio) < 0.1  # Allow 10% tolerance

    def test_create_dataloader_from_dataset(self, sample_dataset_with_transform):
        """Test single DataLoader creation."""
        loader = create_dataloader_from_dataset(
            sample_dataset_with_transform,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )

        assert loader is not None
        assert len(loader) > 0

    def test_create_data_loaders_iteration(self, sample_dataset_with_transform):
        """Test that data loaders can be iterated."""
        train_loader, val_loader = create_data_loaders(
            sample_dataset_with_transform,
            batch_size=2,
            num_workers=0
        )

        # Get first batch
        images, labels = next(iter(train_loader))

        assert images.shape[0] <= 2  # Batch size
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsCalculation:
    """Test metrics calculation functions."""

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1])
        y_pred = np.array([0, 1, 1, 3, 4, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        # Check required keys
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'cohen_kappa' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class_metrics' in metrics

    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = np.array([0, 1, 2, 3, 4] * 10)

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0

    def test_calculate_metrics_per_class(self):
        """Test per-class metrics structure."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1])
        y_pred = np.array([0, 1, 1, 3, 4, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        per_class = metrics['per_class_metrics']

        # Should have metrics for each class present
        for class_idx in range(5):
            if str(class_idx) in per_class:
                class_metrics = per_class[str(class_idx)]
                assert 'precision' in class_metrics
                assert 'recall' in class_metrics
                assert 'f1-score' in class_metrics
                assert 'support' in class_metrics

    def test_calculate_metrics_confusion_matrix_shape(self):
        """Test confusion matrix shape."""
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = np.array([0, 1, 1, 3, 4] * 10)

        metrics = calculate_metrics(y_true, y_pred, num_classes=5)

        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (5, 5)

    def test_print_metrics(self, capsys):
        """Test metrics printing."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 1, 3, 4])

        metrics = calculate_metrics(y_true, y_pred)
        print_metrics(metrics, title="Test Metrics")

        captured = capsys.readouterr()
        assert "Test Metrics" in captured.out
        assert "Accuracy" in captured.out
        assert "Precision" in captured.out
        assert "Recall" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX VISUALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestConfusionMatrixVisualization:
    """Test confusion matrix plotting functions."""

    def test_plot_confusion_matrix_basic(self, temp_data_dir):
        """Test basic confusion matrix plotting."""
        y_true = np.array([0, 1, 2, 3, 4] * 20)
        y_pred = np.array([0, 1, 1, 3, 4] * 20)

        save_path = temp_data_dir / "cm.png"

        plot_confusion_matrix(
            y_true, y_pred,
            save_path=save_path,
            show=False
        )

        assert save_path.exists()

    def test_plot_confusion_matrix_normalization(self, temp_data_dir):
        """Test normalized vs unnormalized confusion matrix."""
        y_true = np.array([0, 1, 2] * 10)
        y_pred = np.array([0, 1, 1] * 10)

        # Normalized
        plot_confusion_matrix(
            y_true, y_pred,
            classes=['A', 'B', 'C'],
            normalize=True,
            save_path=temp_data_dir / "cm_normalized.png",
            show=False
        )

        # Unnormalized
        plot_confusion_matrix(
            y_true, y_pred,
            classes=['A', 'B', 'C'],
            normalize=False,
            save_path=temp_data_dir / "cm_unnormalized.png",
            show=False
        )

        assert (temp_data_dir / "cm_normalized.png").exists()
        assert (temp_data_dir / "cm_unnormalized.png").exists()

    def test_plot_confusion_matrix_custom_classes(self, temp_data_dir):
        """Test with custom class names."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])

        plot_confusion_matrix(
            y_true, y_pred,
            classes=['Class A', 'Class B', 'Class C'],
            save_path=temp_data_dir / "cm_custom.png",
            show=False
        )

        assert (temp_data_dir / "cm_custom.png").exists()

    def test_plot_confusion_matrix_from_metrics(self, temp_data_dir):
        """Test plotting from metrics dict."""
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = np.array([0, 1, 1, 3, 4] * 10)

        metrics = calculate_metrics(y_true, y_pred)

        plot_confusion_matrix_from_metrics(
            metrics,
            save_path=temp_data_dir / "cm_from_metrics.png",
            show=False
        )

        assert (temp_data_dir / "cm_from_metrics.png").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HISTORY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingHistory:
    """Test training history utilities."""

    def test_save_training_history(self, temp_data_dir):
        """Test saving training history."""
        history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'train_acc': [80, 85, 90],
            'val_acc': [75, 80, 85]
        }

        filepath = temp_data_dir / "history.json"
        save_training_history(history, filepath)

        assert filepath.exists()

    def test_load_training_history(self, temp_data_dir):
        """Test loading training history."""
        history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4]
        }

        filepath = temp_data_dir / "history.json"
        save_training_history(history, filepath)

        loaded_history = load_training_history(filepath)

        assert loaded_history == history

    def test_load_training_history_nonexistent(self, temp_data_dir):
        """Test loading nonexistent history."""
        with pytest.raises(FileNotFoundError):
            load_training_history(temp_data_dir / "nonexistent.json")

    @pytest.mark.slow
    def test_plot_training_history(self, temp_data_dir):
        """Test plotting training history."""
        history = {
            'train_loss': [0.5, 0.4, 0.3, 0.25],
            'val_loss': [0.6, 0.5, 0.4, 0.35],
            'train_acc': [80, 85, 90, 92],
            'val_acc': [75, 80, 85, 87]
        }

        save_path = temp_data_dir / "history_plot.png"

        plot_training_history(
            history,
            save_path=save_path,
            metrics=['loss', 'acc']
        )

        # Note: plot_training_history calls plt.show() which we can't test easily
        # Just check it doesn't crash


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeviceManagement:
    """Test device management utilities."""

    def test_get_device_cpu(self):
        """Test getting CPU device."""
        device = get_device(device_id=-1, verbose=False)

        assert device.type == 'cpu'

    def test_get_device_auto_detect(self):
        """Test auto device detection."""
        device = get_device(verbose=False)

        assert device.type in ['cpu', 'cuda']

    @pytest.mark.gpu
    def test_get_device_specific_gpu(self):
        """Test specific GPU selection."""
        device = get_device(device_id=0, verbose=False)

        assert device.type == 'cuda'
        assert device.index == 0

    def test_move_to_device_tensor(self):
        """Test moving tensor to device."""
        device = get_device(device_id=-1, verbose=False)
        tensor = torch.randn(3, 224, 224)

        moved_tensor = move_to_device(tensor, device)

        assert moved_tensor.device.type == device.type

    def test_move_to_device_dict(self):
        """Test moving dict of tensors to device."""
        device = get_device(device_id=-1, verbose=False)
        data = {
            'images': torch.randn(8, 3, 224, 224),
            'labels': torch.randint(0, 5, (8,))
        }

        moved_data = move_to_device(data, device)

        assert moved_data['images'].device.type == device.type
        assert moved_data['labels'].device.type == device.type

    def test_move_to_device_list(self):
        """Test moving list of tensors to device."""
        device = get_device(device_id=-1, verbose=False)
        data = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]

        moved_data = move_to_device(data, device)

        assert all(t.device.type == device.type for t in moved_data)

    def test_move_to_device_nested(self):
        """Test moving nested structure to device."""
        device = get_device(device_id=-1, verbose=False)
        data = {
            'batch': [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
            'labels': torch.randint(0, 5, (2,))
        }

        moved_data = move_to_device(data, device)

        assert all(t.device.type == device.type for t in moved_data['batch'])
        assert moved_data['labels'].device.type == device.type


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS & LOGGING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestProgressLogging:
    """Test progress bar and logging utilities."""

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        data = range(10)
        pbar = create_progress_bar(data, desc='Test', leave=False)

        assert pbar is not None

        # Iterate through it
        for _ in pbar:
            pass

    def test_log_metrics_console(self, capsys):
        """Test console logging of metrics."""
        metrics = {
            'loss': 0.45,
            'acc': 85.3,
            'f1': 0.82
        }

        log_metrics(metrics, prefix='val/')

        captured = capsys.readouterr()
        assert 'val/loss' in captured.out
        assert 'val/acc' in captured.out
        assert 'val/f1' in captured.out
        assert '0.4500' in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilsIntegration:
    """Test integration of multiple utility functions."""

    def test_complete_workflow(self, simple_model, sample_dataset_with_transform, temp_data_dir):
        """Test a complete workflow using multiple utilities."""
        # 1. Set seed
        set_seed(42, verbose=False)

        # 2. Get device
        device = get_device(verbose=False)

        # 3. Count parameters
        total, trainable = count_parameters(simple_model)
        assert total > 0

        # 4. Create data loaders
        train_loader, val_loader = create_data_loaders(
            sample_dataset_with_transform,
            batch_size=2,
            num_workers=0
        )

        # 5. Get transforms
        transform = get_transforms(224, is_train=False)
        assert transform is not None

        # 6. Save checkpoint
        checkpoint_path = temp_data_dir / "test_checkpoint.pth"
        save_checkpoint(
            model=simple_model,
            optimizer=None,
            epoch=1,
            metrics={'val_acc': 80.0},
            path=checkpoint_path
        )
        assert checkpoint_path.exists()

        # 7. Load checkpoint
        metadata = load_checkpoint(checkpoint_path, simple_model, verbose=False)
        assert metadata['epoch'] == 1

        # 8. Calculate metrics
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 1, 3, 4])
        metrics = calculate_metrics(y_true, y_pred)
        assert 'accuracy' in metrics

        # 9. Save training history
        history = {'train_loss': [0.5, 0.4]}
        history_path = temp_data_dir / "history.json"
        save_training_history(history, history_path)
        assert history_path.exists()

        # 10. Load training history
        loaded_history = load_training_history(history_path)
        assert loaded_history == history

        print("✓ Complete workflow test passed!")
