"""
Pytest Configuration and Shared Fixtures for DR Classification Tests.

This module provides reusable fixtures for all test files, including:
- Temporary directories and files
- Mock datasets and images
- Transform pipelines
- Model configurations

Author: Generated with Claude Code
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Callable

import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure matplotlib to use non-interactive backend (prevents hanging on visualization tests)
import matplotlib
matplotlib.use('Agg')

# Add scripts to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from dataset import RetinalDataset
from model import DRClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTORY AND FILE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='function')
def temp_data_dir():
    """
    Create a temporary directory for test data.

    Yields:
        Path: Path to temporary directory

    Cleanup:
        Automatically removes directory after test
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope='function')
def sample_csv(temp_data_dir):
    """
    Create a sample CSV file with valid DR data.

    Contains 5 samples with diagnoses 0-4 (one of each class).

    Args:
        temp_data_dir: Temporary directory fixture

    Returns:
        Path: Path to CSV file
    """
    csv_path = temp_data_dir / "test_data.csv"

    # Create sample data with all DR classes
    data = pd.DataFrame({
        'id_code': [f'image_{i:03d}' for i in range(5)],
        'diagnosis': [0, 1, 2, 3, 4]
    })

    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope='function')
def sample_csv_large(temp_data_dir):
    """
    Create a larger sample CSV with 20 samples.

    Args:
        temp_data_dir: Temporary directory fixture

    Returns:
        Path: Path to CSV file
    """
    csv_path = temp_data_dir / "test_data_large.csv"

    # Create 20 samples with imbalanced classes (realistic)
    data = pd.DataFrame({
        'id_code': [f'image_{i:03d}' for i in range(20)],
        'diagnosis': [0]*8 + [1]*5 + [2]*3 + [3]*2 + [4]*2
    })

    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope='function')
def sample_images(temp_data_dir, sample_csv):
    """
    Create dummy retinal images for testing.

    Creates 100x100 RGB images with different colors for each class.

    Args:
        temp_data_dir: Temporary directory fixture
        sample_csv: Sample CSV fixture

    Returns:
        Path: Path to image directory
    """
    img_dir = temp_data_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Read CSV to get image IDs
    df = pd.read_csv(sample_csv)

    # Create dummy images with class-specific colors
    for idx, row in df.iterrows():
        img_id = row['id_code']
        diagnosis = row['diagnosis']

        # Different color for each class
        color = (
            int(diagnosis * 50),
            100 + int(diagnosis * 30),
            200 - int(diagnosis * 40)
        )

        # Create and save image
        img = Image.new('RGB', (100, 100), color=color)
        img.save(img_dir / f"{img_id}.png")

    return img_dir


@pytest.fixture(scope='function')
def sample_images_mixed_formats(temp_data_dir, sample_csv):
    """
    Create images with mixed formats (.png, .jpg, .jpeg).

    Args:
        temp_data_dir: Temporary directory fixture
        sample_csv: Sample CSV fixture

    Returns:
        Path: Path to image directory
    """
    img_dir = temp_data_dir / "images_mixed"
    img_dir.mkdir(exist_ok=True)

    df = pd.read_csv(sample_csv)
    formats = ['.png', '.jpg', '.jpeg', '.png', '.jpg']

    for idx, row in df.iterrows():
        img_id = row['id_code']
        diagnosis = row['diagnosis']
        ext = formats[idx]

        color = (int(diagnosis * 50), 100, 200)
        img = Image.new('RGB', (100, 100), color=color)
        img.save(img_dir / f"{img_id}{ext}")

    return img_dir


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='function')
def sample_dataset(sample_csv, sample_images):
    """
    Create a RetinalDataset instance with test data.

    Args:
        sample_csv: Sample CSV fixture
        sample_images: Sample images fixture

    Returns:
        RetinalDataset: Initialized dataset
    """
    return RetinalDataset(
        csv_file=str(sample_csv),
        img_dir=str(sample_images),
        transform=None
    )


@pytest.fixture(scope='function')
def sample_dataset_with_transform(sample_csv, sample_images, simple_transform):
    """
    Create a RetinalDataset with transforms applied.

    Args:
        sample_csv: Sample CSV fixture
        sample_images: Sample images fixture
        simple_transform: Transform fixture

    Returns:
        RetinalDataset: Dataset with transforms
    """
    return RetinalDataset(
        csv_file=str(sample_csv),
        img_dir=str(sample_images),
        transform=simple_transform
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='function')
def simple_transform():
    """
    Create a basic torchvision transform pipeline.

    Returns:
        transforms.Compose: Simple transform (Resize + ToTensor)
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


@pytest.fixture(scope='function')
def normalize_transform():
    """
    Create transform with ImageNet normalization.

    Returns:
        transforms.Compose: Transform with normalization
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@pytest.fixture(scope='function')
def augmentation_transform():
    """
    Create transform with data augmentation.

    Returns:
        transforms.Compose: Transform with augmentations
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])


@pytest.fixture(scope='function')
def albumentation_transform():
    """
    Create Albumentations transform pipeline.

    Returns:
        A.Compose: Albumentations transform
    """
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


@pytest.fixture(scope='function')
def albumentation_augmentation():
    """
    Create Albumentations transform with heavy augmentation.

    Returns:
        A.Compose: Augmentation pipeline
    """
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='function')
def simple_model():
    """
    Create a simple DRClassifier model (ResNet18 for speed).

    Returns:
        DRClassifier: Small model for testing
    """
    return DRClassifier(
        model_name='resnet18',
        num_classes=5,
        pretrained=False,  # Faster without pretrained weights
        dropout_rate=0.3
    )


@pytest.fixture(scope='function')
def simple_model_pretrained():
    """
    Create a pretrained DRClassifier model.

    Returns:
        DRClassifier: Pretrained ResNet18
    """
    return DRClassifier(
        model_name='resnet18',
        num_classes=5,
        pretrained=True,
        dropout_rate=0.3
    )


@pytest.fixture(scope='session')
def device_fixture():
    """
    Auto-detect available device (CUDA or CPU).

    Returns:
        torch.device: Available device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='function')
def mock_image_tensor():
    """
    Create a mock image tensor for testing.

    Returns:
        torch.Tensor: Shape (3, 224, 224)
    """
    return torch.randn(3, 224, 224)


@pytest.fixture(scope='function')
def mock_batch_tensor():
    """
    Create a mock batch of images.

    Returns:
        torch.Tensor: Shape (4, 3, 224, 224)
    """
    return torch.randn(4, 3, 224, 224)


@pytest.fixture(scope='function')
def mock_labels():
    """
    Create mock labels for testing.

    Returns:
        torch.Tensor: Shape (4,) with values 0-4
    """
    return torch.tensor([0, 1, 2, 3, 4])[:4]


# ═══════════════════════════════════════════════════════════════════════════════
# INVALID DATA FIXTURES (for error testing)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='function')
def invalid_csv_missing_columns(temp_data_dir):
    """
    Create CSV with missing required columns.

    Args:
        temp_data_dir: Temporary directory fixture

    Returns:
        Path: Path to invalid CSV
    """
    csv_path = temp_data_dir / "invalid_columns.csv"
    data = pd.DataFrame({
        'wrong_column': [1, 2, 3],
        'another_wrong': ['a', 'b', 'c']
    })
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope='function')
def invalid_csv_bad_diagnosis(temp_data_dir):
    """
    Create CSV with invalid diagnosis values.

    Args:
        temp_data_dir: Temporary directory fixture

    Returns:
        Path: Path to invalid CSV
    """
    csv_path = temp_data_dir / "invalid_diagnosis.csv"
    data = pd.DataFrame({
        'id_code': ['img1', 'img2', 'img3'],
        'diagnosis': [0, 5, -1]  # 5 and -1 are invalid
    })
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope='function')
def corrupted_image(temp_data_dir):
    """
    Create a corrupted image file.

    Args:
        temp_data_dir: Temporary directory fixture

    Returns:
        Path: Path to corrupted image
    """
    img_path = temp_data_dir / "corrupted.png"
    with open(img_path, 'wb') as f:
        f.write(b'not a valid image file')
    return img_path


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='session')
def imagenet_stats():
    """
    ImageNet normalization statistics.

    Returns:
        dict: Mean and std values
    """
    return {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """
    Assert that tensor has expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple

    Raises:
        AssertionError: If shapes don't match
    """
    assert tensor.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float):
    """
    Assert that tensor values are in expected range.

    Args:
        tensor: Tensor to check
        min_val: Minimum expected value
        max_val: Maximum expected value

    Raises:
        AssertionError: If values out of range
    """
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    assert actual_min >= min_val and actual_max <= max_val, \
        f"Expected range [{min_val}, {max_val}], got [{actual_min}, {actual_max}]"


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
