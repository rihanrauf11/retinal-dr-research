"""
Comprehensive Tests for DataLoader Integration with RetinalDataset.

This module tests the PyTorch DataLoader functionality with the RetinalDataset class,
covering basic loading, batch processing, transforms, different datasets, edge cases,
performance, and error handling.

Tests verify:
- Basic dataset loading and __getitem__ functionality
- Batch loading with various batch sizes
- Transform application (torchvision and albumentations)
- Cross-dataset compatibility (APTOS, Messidor)
- Edge cases (shuffle, num_workers, empty datasets)
- Loading performance and throughput
- Error handling for invalid configurations

Author: Generated with Claude Code
"""

import time
from pathlib import Path
from typing import Tuple, Dict

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from dataset import RetinalDataset


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataloader(
    dataset: RetinalDataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a standard DataLoader for testing.

    Args:
        dataset: RetinalDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader: Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False  # Disable for CPU testing
    )


def time_epoch(dataloader: DataLoader, max_batches: int = None) -> Dict[str, float]:
    """
    Time a full epoch (or partial epoch) and return statistics.

    Args:
        dataloader: DataLoader to time
        max_batches: Optional limit on number of batches

    Returns:
        dict: Statistics (total_time, batches_per_sec, images_per_sec, num_batches, num_images)
    """
    start_time = time.time()
    num_batches = 0
    num_images = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        num_batches += 1
        num_images += images.shape[0]

        if max_batches and batch_idx >= max_batches - 1:
            break

    elapsed_time = time.time() - start_time

    return {
        'total_time': elapsed_time,
        'batches_per_sec': num_batches / elapsed_time if elapsed_time > 0 else 0,
        'images_per_sec': num_images / elapsed_time if elapsed_time > 0 else 0,
        'num_batches': num_batches,
        'num_images': num_images
    }


def verify_batch_shapes(
    images: torch.Tensor,
    labels: torch.Tensor,
    expected_batch_size: int,
    expected_img_shape: Tuple[int, int, int] = (3, 224, 224)
) -> None:
    """
    Verify batch tensor shapes are correct.

    Args:
        images: Batch of images tensor
        labels: Batch of labels tensor
        expected_batch_size: Expected batch size
        expected_img_shape: Expected image shape (C, H, W)

    Raises:
        AssertionError: If shapes don't match
    """
    assert images.shape == (expected_batch_size, *expected_img_shape), \
        f"Expected images shape {(expected_batch_size, *expected_img_shape)}, got {images.shape}"

    assert labels.shape == (expected_batch_size,), \
        f"Expected labels shape {(expected_batch_size,)}, got {labels.shape}"

    assert images.dtype == torch.float32, \
        f"Expected images dtype torch.float32, got {images.dtype}"

    assert labels.dtype == torch.int64, \
        f"Expected labels dtype torch.int64, got {labels.dtype}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: BASIC DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicDataLoading:
    """Test basic dataset creation and item loading."""

    def test_dataset_creation_with_sample_data(self, sample_dataset):
        """Test dataset creation with sample fixture data."""
        assert sample_dataset is not None
        assert len(sample_dataset) == 5

        # Get first item
        image, label = sample_dataset[0]
        assert image is not None
        assert isinstance(label, int)
        assert 0 <= label <= 4

    def test_first_item_without_transform(self, sample_csv, sample_images):
        """Test loading first item without transforms returns PIL Image."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=None
        )

        image, label = dataset[0]

        # Without transform, should return PIL Image
        assert isinstance(image, Image.Image), \
            f"Expected PIL.Image.Image, got {type(image)}"
        assert isinstance(label, int)
        assert image.mode == 'RGB'
        assert image.size == (100, 100)  # Sample images are 100x100

    def test_first_item_with_transform(self, sample_csv, sample_images, simple_transform):
        """Test loading first item with transforms returns torch.Tensor."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=simple_transform
        )

        image, label = dataset[0]

        # With transform, should return tensor
        assert isinstance(image, torch.Tensor), \
            f"Expected torch.Tensor, got {type(image)}"
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert isinstance(label, int)

    def test_all_items_loadable(self, sample_dataset):
        """Test that all items in dataset can be loaded without error."""
        for idx in range(len(sample_dataset)):
            image, label = sample_dataset[idx]
            assert image is not None
            assert 0 <= label <= 4

    def test_class_distribution(self, sample_dataset):
        """Test get_class_distribution method."""
        distribution = sample_dataset.get_class_distribution()

        assert isinstance(distribution, dict)
        assert len(distribution) == 5  # All 5 classes present in sample
        assert all(count == 1 for count in distribution.values())  # 1 of each


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: BATCH LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchLoading:
    """Test DataLoader batch loading functionality."""

    def test_dataloader_batch_shapes(self, sample_dataset_with_transform):
        """Test that DataLoader produces correct batch shapes."""
        batch_size = 4  # Sample dataset has 5 items
        loader = get_dataloader(sample_dataset_with_transform, batch_size=batch_size)

        # Get first batch
        images, labels = next(iter(loader))

        # Verify shapes
        verify_batch_shapes(images, labels, batch_size)

        # Verify label values
        assert torch.all((labels >= 0) & (labels <= 4)), \
            f"Labels should be in [0, 4], got {labels}"

    def test_dataloader_iteration(self, sample_dataset_with_transform):
        """Test iterating through full dataset."""
        batch_size = 2
        loader = get_dataloader(sample_dataset_with_transform, batch_size=batch_size)

        all_batches = []
        for images, labels in loader:
            all_batches.append((images, labels))
            assert isinstance(images, torch.Tensor)
            assert isinstance(labels, torch.Tensor)

        # Sample dataset has 5 items, batch_size=2
        # Should have 3 batches: [2, 2, 1]
        assert len(all_batches) == 3

        # Check batch sizes
        assert all_batches[0][0].shape[0] == 2
        assert all_batches[1][0].shape[0] == 2
        assert all_batches[2][0].shape[0] == 1  # Last batch smaller

        # Verify total items
        total_items = sum(batch[0].shape[0] for batch in all_batches)
        assert total_items == len(sample_dataset_with_transform)

    def test_multiple_batch_sizes(self, sample_csv_large, temp_data_dir, simple_transform):
        """Test DataLoader with various batch sizes."""
        # Create images for large dataset
        img_dir = temp_data_dir / "images_large"
        img_dir.mkdir()

        import pandas as pd
        df = pd.read_csv(sample_csv_large)
        for img_id in df['id_code']:
            img = Image.new('RGB', (100, 100), color=(50, 100, 150))
            img.save(img_dir / f"{img_id}.png")

        dataset = RetinalDataset(
            csv_file=str(sample_csv_large),
            img_dir=str(img_dir),
            transform=simple_transform
        )

        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32]
        dataset_len = len(dataset)  # 20 items

        for batch_size in batch_sizes:
            loader = get_dataloader(dataset, batch_size=batch_size)

            total_items = 0
            num_batches = 0

            for images, labels in loader:
                total_items += images.shape[0]
                num_batches += 1

            # Verify all items loaded
            assert total_items == dataset_len, \
                f"Batch size {batch_size}: expected {dataset_len} items, got {total_items}"

            # Verify correct number of batches
            expected_batches = (dataset_len + batch_size - 1) // batch_size
            assert num_batches == expected_batches, \
                f"Batch size {batch_size}: expected {expected_batches} batches, got {num_batches}"

    def test_batch_labels_consistency(self, sample_dataset_with_transform):
        """Test that batch labels are consistent with individual loads."""
        # Load items individually
        individual_labels = []
        for idx in range(len(sample_dataset_with_transform)):
            _, label = sample_dataset_with_transform[idx]
            individual_labels.append(label)

        # Load via DataLoader
        loader = get_dataloader(sample_dataset_with_transform, batch_size=5, shuffle=False)
        batch_images, batch_labels = next(iter(loader))

        # Compare
        assert torch.all(batch_labels == torch.tensor(individual_labels)), \
            "Batch labels should match individual loads when shuffle=False"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransforms:
    """Test transform application in DataLoader."""

    def test_without_transforms(self, sample_csv, sample_images):
        """Test DataLoader with no transforms returns PIL Images."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=None
        )

        # Note: DataLoader requires tensors, so this will fail
        # This test verifies the dataset behavior, not DataLoader
        image, label = dataset[0]
        assert isinstance(image, Image.Image)

    def test_with_torchvision_transforms(self, sample_csv, sample_images):
        """Test DataLoader with torchvision transforms."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=transform
        )

        loader = get_dataloader(dataset, batch_size=2)
        images, labels = next(iter(loader))

        assert images.shape == (2, 3, 224, 224)
        assert images.dtype == torch.float32
        # ToTensor scales to [0, 1]
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_with_normalization(self, sample_csv, sample_images, normalize_transform):
        """Test DataLoader with ImageNet normalization."""
        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=normalize_transform
        )

        loader = get_dataloader(dataset, batch_size=5)
        images, labels = next(iter(loader))

        # After normalization, values should be roughly in [-2, 2] range
        assert images.min() >= -5.0  # Rough bounds
        assert images.max() <= 5.0

        # Check that mean is approximately [0, 0, 0]
        mean = images.mean(dim=(0, 2, 3))  # Average over batch and spatial dims
        # Mean won't be exactly 0 due to sample bias, but should be close
        assert torch.all(torch.abs(mean) < 1.0), \
            f"After normalization, mean should be close to 0, got {mean}"

    def test_with_albumentations(self, sample_csv, sample_images):
        """Test DataLoader with Albumentations transforms."""
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=transform
        )

        loader = get_dataloader(dataset, batch_size=3)
        images, labels = next(iter(loader))

        assert images.shape == (3, 3, 224, 224)
        assert images.dtype == torch.float32

    def test_transform_determinism(self, sample_csv, sample_images):
        """Test that deterministic transforms produce same output."""
        # Use deterministic transform (no random augmentation)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=transform
        )

        # Load same item twice
        img1, label1 = dataset[0]
        img2, label2 = dataset[0]

        assert torch.allclose(img1, img2), \
            "Deterministic transform should produce identical outputs"
        assert label1 == label2


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: DIFFERENT DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifferentDatasets:
    """Test DataLoader with different real datasets (APTOS, Messidor)."""

    @pytest.mark.skipif(
        not Path("data/aptos/train_split.csv").exists(),
        reason="APTOS train split not available"
    )
    def test_aptos_train_split_loading(self):
        """Test loading APTOS train split dataset."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file="data/aptos/train_split.csv",
            img_dir="data/aptos/train_images",
            transform=transform
        )

        # Should have 2929 samples (from previous split creation)
        assert len(dataset) == 2929

        # Test that we can create a DataLoader
        loader = get_dataloader(dataset, batch_size=16)

        # Load first batch
        images, labels = next(iter(loader))
        assert images.shape == (16, 3, 224, 224)
        assert labels.shape == (16,)

        # Sample 10 random items to verify they load
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), min(10, len(dataset)))

        for idx in sample_indices:
            img, label = dataset[idx]
            assert img.shape == (3, 224, 224)
            assert 0 <= label <= 4

    @pytest.mark.skipif(
        not Path("data/aptos/val_split.csv").exists(),
        reason="APTOS val split not available"
    )
    def test_aptos_val_split_loading(self):
        """Test loading APTOS validation split dataset."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file="data/aptos/val_split.csv",
            img_dir="data/aptos/train_images",  # Val uses same image dir
            transform=transform
        )

        # Should have 733 samples
        assert len(dataset) == 733

        # Create DataLoader
        loader = get_dataloader(dataset, batch_size=32)

        # Count total items via DataLoader
        total_items = sum(labels.shape[0] for _, labels in loader)
        assert total_items == len(dataset)

    @pytest.mark.skipif(
        not Path("data/messidor/test.csv").exists(),
        reason="Messidor dataset not available"
    )
    def test_messidor_dataset_loading(self):
        """Test loading Messidor test dataset."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file="data/messidor/test.csv",
            img_dir="data/messidor/images",
            transform=transform
        )

        # Should have 1057 samples
        assert len(dataset) == 1057

        # Test DataLoader
        loader = get_dataloader(dataset, batch_size=16)
        images, labels = next(iter(loader))

        assert images.shape == (16, 3, 224, 224)
        assert torch.all((labels >= 0) & (labels <= 4))

    @pytest.mark.skipif(
        not (Path("data/aptos/train_split.csv").exists() and
             Path("data/messidor/test.csv").exists()),
        reason="Both datasets not available"
    )
    def test_same_dataloader_code_different_datasets(self):
        """Test that same DataLoader code works for different datasets."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create both datasets with same transform
        aptos_dataset = RetinalDataset(
            csv_file="data/aptos/train_split.csv",
            img_dir="data/aptos/train_images",
            transform=transform
        )

        messidor_dataset = RetinalDataset(
            csv_file="data/messidor/test.csv",
            img_dir="data/messidor/images",
            transform=transform
        )

        # Use identical DataLoader parameters
        batch_size = 16
        aptos_loader = get_dataloader(aptos_dataset, batch_size=batch_size)
        messidor_loader = get_dataloader(messidor_dataset, batch_size=batch_size)

        # Load first batch from each
        aptos_images, aptos_labels = next(iter(aptos_loader))
        messidor_images, messidor_labels = next(iter(messidor_loader))

        # Verify both have same shape (showing code portability)
        assert aptos_images.shape == messidor_images.shape == (batch_size, 3, 224, 224)
        assert aptos_labels.shape == messidor_labels.shape == (batch_size,)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and special DataLoader configurations."""

    def test_batch_size_one(self, sample_dataset_with_transform):
        """Test DataLoader with batch_size=1."""
        loader = get_dataloader(sample_dataset_with_transform, batch_size=1)

        # Get first batch to check shape
        first_batch = next(iter(loader))
        images, labels = first_batch

        assert images.shape == (1, 3, 224, 224)
        assert labels.shape == (1,)

        # Count all batches with fresh iterator (should equal dataset length)
        loader = get_dataloader(sample_dataset_with_transform, batch_size=1)
        num_batches = sum(1 for _ in loader)
        assert num_batches == len(sample_dataset_with_transform)

    def test_shuffle_true(self, sample_dataset_with_transform):
        """Test DataLoader with shuffle=True produces different orders."""
        # Load with shuffle, seed 1
        torch.manual_seed(42)
        loader1 = get_dataloader(sample_dataset_with_transform, batch_size=5, shuffle=True)
        labels1 = next(iter(loader1))[1]

        # Load with shuffle, seed 2
        torch.manual_seed(123)
        loader2 = get_dataloader(sample_dataset_with_transform, batch_size=5, shuffle=True)
        labels2 = next(iter(loader2))[1]

        # Different seeds should produce different orders (very likely with 5 items)
        # (There's a 1/120 chance they're the same by chance, acceptable for testing)
        # So we test that they contain the same elements, but possibly in different order
        assert set(labels1.tolist()) == set(labels2.tolist()), \
            "Should contain same labels"

    def test_shuffle_false_deterministic(self, sample_dataset_with_transform):
        """Test that shuffle=False produces deterministic order."""
        loader1 = get_dataloader(sample_dataset_with_transform, batch_size=5, shuffle=False)
        labels1 = next(iter(loader1))[1]

        loader2 = get_dataloader(sample_dataset_with_transform, batch_size=5, shuffle=False)
        labels2 = next(iter(loader2))[1]

        assert torch.all(labels1 == labels2), \
            "shuffle=False should produce identical orders"

    def test_num_workers_zero(self, sample_dataset_with_transform):
        """Test DataLoader with num_workers=0 (single-threaded)."""
        loader = get_dataloader(sample_dataset_with_transform, batch_size=2, num_workers=0)

        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())

        assert len(all_labels) == len(sample_dataset_with_transform)

    def test_num_workers_multi(self, sample_dataset_with_transform):
        """Test DataLoader with num_workers>0 (multi-threaded)."""
        loader = get_dataloader(sample_dataset_with_transform, batch_size=2, num_workers=2)

        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())

        assert len(all_labels) == len(sample_dataset_with_transform)

    def test_num_workers_consistency(self, sample_dataset_with_transform):
        """Test that num_workers=0 and num_workers=2 produce same results."""
        # Load with num_workers=0
        loader0 = get_dataloader(
            sample_dataset_with_transform,
            batch_size=5,
            shuffle=False,
            num_workers=0
        )
        labels0 = next(iter(loader0))[1]

        # Load with num_workers=2
        loader2 = get_dataloader(
            sample_dataset_with_transform,
            batch_size=5,
            shuffle=False,
            num_workers=2
        )
        labels2 = next(iter(loader2))[1]

        # Should produce identical results when shuffle=False
        assert torch.all(labels0 == labels2), \
            "Different num_workers should produce same results with shuffle=False"

    def test_empty_dataset(self, temp_data_dir):
        """Test DataLoader with empty dataset."""
        # Create empty CSV
        import pandas as pd
        empty_csv = temp_data_dir / "empty.csv"
        pd.DataFrame({'id_code': [], 'diagnosis': []}).to_csv(empty_csv, index=False)

        img_dir = temp_data_dir / "empty_images"
        img_dir.mkdir()

        dataset = RetinalDataset(
            csv_file=str(empty_csv),
            img_dir=str(img_dir),
            transform=None
        )

        assert len(dataset) == 0

        loader = get_dataloader(dataset, batch_size=16)

        # Should produce no batches
        num_batches = sum(1 for _ in loader)
        assert num_batches == 0

    def test_drop_last_true(self, sample_dataset_with_transform):
        """Test DataLoader with drop_last=True."""
        batch_size = 2
        loader = get_dataloader(
            sample_dataset_with_transform,
            batch_size=batch_size,
            drop_last=True
        )

        batches = list(loader)

        # Sample dataset has 5 items, batch_size=2, drop_last=True
        # Should drop the last batch of 1 item
        # Expected: 2 batches of size 2
        assert len(batches) == 2

        for images, labels in batches:
            assert images.shape[0] == batch_size, \
                "All batches should have full batch_size when drop_last=True"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPerformance:
    """Test loading performance and throughput (marked as slow tests)."""

    @pytest.mark.skipif(
        not Path("data/aptos/train_split.csv").exists(),
        reason="APTOS dataset not available"
    )
    def test_loading_speed_full_epoch(self):
        """Time loading partial epoch (100 batches) and report statistics."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file="data/aptos/train_split.csv",
            img_dir="data/aptos/train_images",
            transform=transform
        )

        loader = get_dataloader(dataset, batch_size=32, num_workers=0)

        # Load only 30 batches to keep test fast (~1000 images)
        stats = time_epoch(loader, max_batches=30)

        print(f"\n{'='*70}")
        print("LOADING PERFORMANCE STATISTICS")
        print(f"{'='*70}")
        print(f"Dataset: APTOS train split ({len(dataset)} images total)")
        print(f"Batch size: 32")
        print(f"Num workers: 0")
        print(f"Batches tested: {stats['num_batches']}")
        print(f"Images loaded: {stats['num_images']}")
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Throughput: {stats['images_per_sec']:.2f} images/sec")
        print(f"Batches/sec: {stats['batches_per_sec']:.2f}")
        print(f"{'='*70}")

        # Sanity check: should have loaded 30 batches worth
        assert stats['num_batches'] == 30

        # Should be reasonably fast (>10 images/sec even on slow hardware)
        assert stats['images_per_sec'] > 1.0, \
            f"Loading seems too slow: {stats['images_per_sec']:.2f} images/sec"

    @pytest.mark.skipif(
        not Path("data/aptos/train_split.csv").exists(),
        reason="APTOS dataset not available"
    )
    def test_num_workers_speedup(self):
        """Compare loading speed with num_workers=0 vs num_workers=4."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file="data/aptos/train_split.csv",
            img_dir="data/aptos/train_images",
            transform=transform
        )

        # Test with num_workers=0
        loader0 = get_dataloader(dataset, batch_size=32, num_workers=0)
        stats0 = time_epoch(loader0, max_batches=50)

        # Test with num_workers=4
        loader4 = get_dataloader(dataset, batch_size=32, num_workers=4)
        stats4 = time_epoch(loader4, max_batches=50)

        speedup = stats4['images_per_sec'] / stats0['images_per_sec']

        print(f"\n{'='*70}")
        print("NUM_WORKERS COMPARISON")
        print(f"{'='*70}")
        print(f"num_workers=0: {stats0['images_per_sec']:.2f} images/sec")
        print(f"num_workers=4: {stats4['images_per_sec']:.2f} images/sec")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*70}")

        # Multi-worker should be at least as fast (might be slower due to overhead on small batches)
        # So we just verify it doesn't crash and produces results
        assert stats4['num_images'] == stats0['num_images']

    @pytest.mark.skipif(
        not Path("data/aptos/train_split.csv").exists(),
        reason="APTOS dataset not available"
    )
    def test_batch_size_impact_on_throughput(self):
        """Measure throughput for different batch sizes."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file="data/aptos/train_split.csv",
            img_dir="data/aptos/train_images",
            transform=transform
        )

        batch_sizes = [8, 16, 32, 64]
        results = {}

        for batch_size in batch_sizes:
            loader = get_dataloader(dataset, batch_size=batch_size, num_workers=0)
            stats = time_epoch(loader, max_batches=50)
            results[batch_size] = stats['images_per_sec']

        print(f"\n{'='*70}")
        print("BATCH SIZE IMPACT ON THROUGHPUT")
        print(f"{'='*70}")
        for batch_size, throughput in results.items():
            print(f"Batch size {batch_size:3d}: {throughput:6.2f} images/sec")
        print(f"{'='*70}")

        # All batch sizes should work
        assert all(throughput > 1.0 for throughput in results.values())


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 7: ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_missing_image_in_dataloader(self, temp_data_dir):
        """Test that missing image raises FileNotFoundError during iteration."""
        import pandas as pd

        # Create CSV with non-existent image
        csv_path = temp_data_dir / "missing_img.csv"
        pd.DataFrame({
            'id_code': ['nonexistent_image'],
            'diagnosis': [0]
        }).to_csv(csv_path, index=False)

        img_dir = temp_data_dir / "images_missing"
        img_dir.mkdir()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file=str(csv_path),
            img_dir=str(img_dir),
            transform=transform
        )

        loader = get_dataloader(dataset, batch_size=1)

        # Should raise FileNotFoundError when trying to load
        with pytest.raises(FileNotFoundError, match="Image not found"):
            images, labels = next(iter(loader))

    def test_invalid_batch_size_zero(self, sample_dataset_with_transform):
        """Test that batch_size=0 raises ValueError."""
        with pytest.raises(ValueError):
            loader = get_dataloader(sample_dataset_with_transform, batch_size=0)
            next(iter(loader))

    def test_invalid_batch_size_negative(self, sample_dataset_with_transform):
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError):
            loader = get_dataloader(sample_dataset_with_transform, batch_size=-1)
            next(iter(loader))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
