"""
Unit Tests for RetinalDataset Class.

Tests cover:
- Dataset initialization and validation
- Data loading (__len__, __getitem__)
- Transform application (torchvision and albumentations)
- Error handling
- Utility methods
- Integration with DataLoader

Author: Generated with Claude Code
"""

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

from dataset import RetinalDataset


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetInitialization:
    """Test RetinalDataset initialization and validation."""

    def test_dataset_creation_success(self, sample_csv, sample_images):
        """Test successful dataset creation with valid data."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=None
        )

        assert dataset is not None
        assert len(dataset) == 5
        assert dataset.csv_file == Path(sample_csv)
        assert dataset.img_dir == Path(sample_images)
        assert dataset.transform is None

    def test_dataset_missing_csv(self, temp_data_dir, sample_images):
        """Test that missing CSV file raises FileNotFoundError."""
        nonexistent_csv = temp_data_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            RetinalDataset(
                csv_file=str(nonexistent_csv),
                img_dir=str(sample_images)
            )

    def test_dataset_missing_imgdir(self, sample_csv, temp_data_dir):
        """Test that missing image directory raises FileNotFoundError."""
        nonexistent_dir = temp_data_dir / "nonexistent_dir"

        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            RetinalDataset(
                csv_file=str(sample_csv),
                img_dir=str(nonexistent_dir)
            )

    def test_dataset_invalid_columns(self, invalid_csv_missing_columns, sample_images):
        """Test that CSV with missing required columns raises ValueError."""
        with pytest.raises(ValueError, match="missing required columns"):
            RetinalDataset(
                csv_file=str(invalid_csv_missing_columns),
                img_dir=str(sample_images)
            )

    def test_dataset_invalid_diagnosis_values(self, invalid_csv_bad_diagnosis, sample_images):
        """Test that invalid diagnosis values raise ValueError."""
        with pytest.raises(ValueError, match="invalid diagnosis values"):
            RetinalDataset(
                csv_file=str(invalid_csv_bad_diagnosis),
                img_dir=str(sample_images)
            )

    def test_dataset_empty_csv(self, temp_data_dir, sample_images):
        """Test handling of empty CSV file."""
        empty_csv = temp_data_dir / "empty.csv"
        pd.DataFrame({'id_code': [], 'diagnosis': []}).to_csv(empty_csv, index=False)

        dataset = RetinalDataset(
            csv_file=str(empty_csv),
            img_dir=str(sample_images)
        )

        assert len(dataset) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetCoreFunctionality:
    """Test core dataset functionality (__len__, __getitem__)."""

    def test_dataset_len(self, sample_dataset):
        """Test that __len__ returns correct dataset size."""
        assert len(sample_dataset) == 5

    def test_dataset_len_large(self, sample_csv_large, sample_images):
        """Test __len__ with larger dataset."""
        # Note: sample_images only has 5 images, but we're testing __len__ only
        dataset = RetinalDataset(
            csv_file=str(sample_csv_large),
            img_dir=str(sample_images)
        )
        assert len(dataset) == 20

    def test_getitem_no_transform(self, sample_dataset):
        """Test loading image without transforms."""
        image, label = sample_dataset[0]

        # Check types
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)

        # Check values
        assert image.mode == 'RGB'
        assert image.size == (100, 100)  # Original size
        assert 0 <= label <= 4

    def test_getitem_all_samples(self, sample_dataset):
        """Test loading all samples in dataset."""
        for idx in range(len(sample_dataset)):
            image, label = sample_dataset[idx]

            assert isinstance(image, Image.Image)
            assert isinstance(label, int)
            assert image.mode == 'RGB'
            assert 0 <= label <= 4

    def test_getitem_with_torchvision_transform(self, sample_csv, sample_images, simple_transform):
        """Test loading with torchvision transforms."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=simple_transform
        )

        image, label = dataset[0]

        # Check types
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

        # Check shape and range
        assert image.shape == (3, 224, 224)
        assert image.min() >= 0.0 and image.max() <= 1.0

    def test_getitem_with_normalization(self, sample_csv, sample_images, normalize_transform):
        """Test loading with normalization transform."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=normalize_transform
        )

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        # Normalized values can be negative
        assert image.min() < 0.0

    def test_getitem_with_albumentations(self, sample_csv, sample_images, albumentation_transform):
        """Test loading with albumentations transforms."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=albumentation_transform
        )

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_getitem_invalid_index_negative(self, sample_dataset):
        """Test that negative index raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            _ = sample_dataset[-1]

    def test_getitem_invalid_index_too_large(self, sample_dataset):
        """Test that index >= len raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            _ = sample_dataset[100]

    def test_getitem_missing_image_file(self, temp_data_dir):
        """Test that missing image file raises FileNotFoundError."""
        # Create CSV with image that doesn't exist
        csv_path = temp_data_dir / "test.csv"
        pd.DataFrame({
            'id_code': ['missing_image'],
            'diagnosis': [0]
        }).to_csv(csv_path, index=False)

        img_dir = temp_data_dir / "images"
        img_dir.mkdir()

        dataset = RetinalDataset(str(csv_path), str(img_dir))

        with pytest.raises(FileNotFoundError, match="Image not found"):
            _ = dataset[0]

    def test_getitem_multiple_extensions(self, sample_csv, sample_images_mixed_formats):
        """Test loading images with different extensions (.png, .jpg, .jpeg)."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images_mixed_formats)
        )

        # Should successfully load all 5 images regardless of extension
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            assert isinstance(image, Image.Image)
            assert image.mode == 'RGB'

    def test_getitem_label_correctness(self, sample_dataset):
        """Test that labels match CSV values."""
        expected_labels = [0, 1, 2, 3, 4]

        for idx, expected_label in enumerate(expected_labels):
            _, actual_label = sample_dataset[idx]
            assert actual_label == expected_label


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY METHODS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetUtilityMethods:
    """Test utility methods (get_class_distribution, get_sample_info)."""

    def test_get_class_distribution(self, sample_dataset):
        """Test class distribution calculation."""
        distribution = sample_dataset.get_class_distribution()

        # Should have 1 sample per class
        assert distribution == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
        assert sum(distribution.values()) == 5

    def test_get_class_distribution_imbalanced(self, sample_csv_large, sample_images):
        """Test class distribution with imbalanced data."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv_large),
            img_dir=str(sample_images)
        )

        distribution = dataset.get_class_distribution()

        # Expected: [0]*8 + [1]*5 + [2]*3 + [3]*2 + [4]*2
        assert distribution[0] == 8
        assert distribution[1] == 5
        assert distribution[2] == 3
        assert distribution[3] == 2
        assert distribution[4] == 2

    def test_get_sample_info(self, sample_dataset):
        """Test getting sample metadata."""
        info = sample_dataset.get_sample_info(2)

        assert isinstance(info, dict)
        assert 'id_code' in info
        assert 'diagnosis' in info
        assert 'index' in info
        assert info['index'] == 2
        assert info['diagnosis'] == 2

    def test_get_sample_info_all_samples(self, sample_dataset):
        """Test getting info for all samples."""
        for idx in range(len(sample_dataset)):
            info = sample_dataset.get_sample_info(idx)

            assert info['index'] == idx
            assert isinstance(info['id_code'], str)
            assert 0 <= info['diagnosis'] <= 4


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetTransforms:
    """Test transform application and consistency."""

    def test_transform_resize(self, sample_dataset, simple_transform):
        """Test that resize transform works correctly."""
        dataset = RetinalDataset(
            csv_file=str(sample_dataset.csv_file),
            img_dir=str(sample_dataset.img_dir),
            transform=simple_transform
        )

        image, _ = dataset[0]
        assert image.shape == (3, 224, 224)

    def test_transform_consistency(self, sample_dataset, simple_transform):
        """Test that transforms are applied consistently."""
        dataset = RetinalDataset(
            csv_file=str(sample_dataset.csv_file),
            img_dir=str(sample_dataset.img_dir),
            transform=simple_transform
        )

        # Load same sample twice
        image1, label1 = dataset[0]
        image2, label2 = dataset[0]

        # Labels should match
        assert label1 == label2

        # Images should match (no random augmentation in simple_transform)
        assert torch.allclose(image1, image2)

    def test_transform_different_sizes(self, sample_csv, sample_images):
        """Test transforms with different output sizes."""
        sizes = [128, 224, 384]

        for size in sizes:
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()
            ])

            dataset = RetinalDataset(
                csv_file=str(sample_csv),
                img_dir=str(sample_images),
                transform=transform
            )

            image, _ = dataset[0]
            assert image.shape == (3, size, size)

    def test_albumentations_vs_torchvision(self, sample_csv, sample_images):
        """Test that both transform libraries work."""
        # Torchvision
        tv_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tv_dataset = RetinalDataset(str(sample_csv), str(sample_images), tv_transform)
        tv_image, _ = tv_dataset[0]

        # Albumentations
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        albu_transform = A.Compose([
            A.Resize(224, 224),
            ToTensorV2()
        ])
        albu_dataset = RetinalDataset(str(sample_csv), str(sample_images), albu_transform)
        albu_image, _ = albu_dataset[0]

        # Both should produce tensors of same shape
        assert tv_image.shape == albu_image.shape == (3, 224, 224)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetIntegration:
    """Test integration with PyTorch DataLoader and training workflows."""

    def test_dataset_with_dataloader(self, sample_dataset_with_transform):
        """Test that dataset works with DataLoader."""
        loader = DataLoader(
            sample_dataset_with_transform,
            batch_size=2,
            shuffle=False
        )

        # Get first batch
        images, labels = next(iter(loader))

        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_dataset_dataloader_batch_sizes(self, sample_dataset_with_transform):
        """Test DataLoader with different batch sizes."""
        batch_sizes = [1, 2, 5]

        for batch_size in batch_sizes:
            loader = DataLoader(
                sample_dataset_with_transform,
                batch_size=batch_size,
                shuffle=False
            )

            # Get first batch
            images, labels = next(iter(loader))

            expected_size = min(batch_size, len(sample_dataset_with_transform))
            assert images.shape[0] == expected_size
            assert labels.shape[0] == expected_size

    def test_dataset_dataloader_shuffle(self, sample_dataset_with_transform):
        """Test that shuffle works in DataLoader."""
        loader1 = DataLoader(sample_dataset_with_transform, batch_size=5, shuffle=False)
        loader2 = DataLoader(sample_dataset_with_transform, batch_size=5, shuffle=True)

        _, labels1 = next(iter(loader1))
        _, labels2 = next(iter(loader2))

        # With shuffle, order might be different (though not guaranteed)
        # At minimum, check that all labels are present
        assert sorted(labels1.tolist()) == sorted(labels2.tolist())

    def test_dataset_dataloader_iteration(self, sample_dataset_with_transform):
        """Test iterating through entire dataset with DataLoader."""
        loader = DataLoader(
            sample_dataset_with_transform,
            batch_size=2,
            shuffle=False
        )

        total_samples = 0
        for images, labels in loader:
            total_samples += images.shape[0]
            assert images.dim() == 4  # (batch, channels, h, w)
            assert labels.dim() == 1  # (batch,)

        assert total_samples == len(sample_dataset_with_transform)

    @pytest.mark.slow
    def test_dataset_dataloader_multiworker(self, sample_dataset_with_transform):
        """Test DataLoader with multiple workers (marked as slow)."""
        loader = DataLoader(
            sample_dataset_with_transform,
            batch_size=2,
            shuffle=False,
            num_workers=2
        )

        # Should work without errors
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 224, 224)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES AND ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_dataset_first_and_last_indices(self, sample_dataset):
        """Test loading first and last samples."""
        # First sample
        image0, label0 = sample_dataset[0]
        assert isinstance(image0, Image.Image)
        assert label0 == 0

        # Last sample
        last_idx = len(sample_dataset) - 1
        image_last, label_last = sample_dataset[last_idx]
        assert isinstance(image_last, Image.Image)
        assert label_last == 4

    def test_dataset_single_sample(self, temp_data_dir):
        """Test dataset with only one sample."""
        # Create CSV with 1 sample
        csv_path = temp_data_dir / "single.csv"
        pd.DataFrame({
            'id_code': ['img_001'],
            'diagnosis': [2]
        }).to_csv(csv_path, index=False)

        # Create image
        img_dir = temp_data_dir / "images"
        img_dir.mkdir()
        Image.new('RGB', (100, 100), color=(100, 100, 100)).save(img_dir / "img_001.png")

        # Create dataset
        dataset = RetinalDataset(str(csv_path), str(img_dir))

        assert len(dataset) == 1
        image, label = dataset[0]
        assert isinstance(image, Image.Image)
        assert label == 2

    def test_dataset_csv_with_extra_columns(self, temp_data_dir, sample_images):
        """Test that extra columns in CSV are ignored."""
        csv_path = temp_data_dir / "extra_cols.csv"
        pd.DataFrame({
            'id_code': [f'image_{i:03d}' for i in range(5)],
            'diagnosis': [0, 1, 2, 3, 4],
            'extra_col1': ['a', 'b', 'c', 'd', 'e'],
            'extra_col2': [10, 20, 30, 40, 50]
        }).to_csv(csv_path, index=False)

        # Should work fine with extra columns
        dataset = RetinalDataset(str(csv_path), str(sample_images))
        assert len(dataset) == 5

    def test_dataset_path_types(self, sample_csv, sample_images):
        """Test that both str and Path types work for paths."""
        # Test with str
        dataset1 = RetinalDataset(str(sample_csv), str(sample_images))
        assert len(dataset1) == 5

        # Test with Path
        dataset2 = RetinalDataset(sample_csv, sample_images)
        assert len(dataset2) == 5

    def test_dataset_class_constants(self):
        """Test that class constants are defined correctly."""
        assert RetinalDataset.VALID_EXTENSIONS == ['.png', '.jpg', '.jpeg']
        assert RetinalDataset.MIN_DIAGNOSIS == 0
        assert RetinalDataset.MAX_DIAGNOSIS == 4


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRIZED TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
    def test_load_each_sample(self, sample_dataset, idx):
        """Test loading each sample individually."""
        image, label = sample_dataset[idx]
        assert isinstance(image, Image.Image)
        assert label == idx

    @pytest.mark.parametrize("size", [64, 128, 224, 384, 512])
    def test_different_image_sizes(self, sample_csv, sample_images, size):
        """Test transforms with different output sizes."""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=transform
        )

        image, _ = dataset[0]
        assert image.shape == (3, size, size)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, sample_dataset_with_transform, batch_size):
        """Test DataLoader with various batch sizes."""
        loader = DataLoader(
            sample_dataset_with_transform,
            batch_size=batch_size,
            shuffle=False
        )

        images, labels = next(iter(loader))
        expected_size = min(batch_size, len(sample_dataset_with_transform))
        assert images.shape[0] == expected_size
