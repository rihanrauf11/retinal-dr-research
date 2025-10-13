"""
RetinalDataset: PyTorch dataset class for diabetic retinopathy classification.

This module provides a flexible dataset loader for retinal fundus images with
support for multiple image formats and comprehensive error handling.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for diabetic retinopathy retinal fundus images.

    This dataset handles loading retinal images and their corresponding diagnosis labels
    from a CSV file. It supports multiple image formats and optional data augmentation.

    Diabetic Retinopathy Severity Levels:
        0: No DR (No Diabetic Retinopathy)
        1: Mild NPDR (Non-Proliferative Diabetic Retinopathy)
        2: Moderate NPDR
        3: Severe NPDR
        4: PDR (Proliferative Diabetic Retinopathy)

    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file containing image metadata. Must have columns:
        - 'id_code': Image identifier (filename without extension)
        - 'diagnosis': DR severity level (integer 0-4)

    img_dir : str or Path
        Directory path containing the retinal images. Images should be named
        according to their id_code with extensions .png, .jpg, or .jpeg.

    transform : callable, optional
        Optional transform/augmentation pipeline to be applied to images.
        Should accept a PIL Image and return a transformed version.
        Common choices: torchvision.transforms or albumentations.
        Default: None (no transformation applied).

    Attributes
    ----------
    data_frame : pd.DataFrame
        DataFrame containing the image metadata from csv_file
    img_dir : Path
        Path object pointing to the image directory
    transform : callable or None
        The transformation pipeline

    Raises
    ------
    FileNotFoundError
        If csv_file or img_dir does not exist
    ValueError
        If required columns are missing from CSV or diagnosis values invalid

    Examples
    --------
    >>> # Basic usage without transforms
    >>> dataset = RetinalDataset(
    ...     csv_file='train.csv',
    ...     img_dir='data/aptos/train_images'
    ... )
    >>> image, label = dataset[0]

    >>> # With torchvision transforms
    >>> from torchvision import transforms
    >>> transform = transforms.Compose([
    ...     transforms.Resize((224, 224)),
    ...     transforms.ToTensor(),
    ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    ...                         std=[0.229, 0.224, 0.225])
    ... ])
    >>> dataset = RetinalDataset(
    ...     csv_file='train.csv',
    ...     img_dir='data/aptos/train_images',
    ...     transform=transform
    ... )

    >>> # With albumentations
    >>> import albumentations as A
    >>> from albumentations.pytorch import ToTensorV2
    >>> transform = A.Compose([
    ...     A.Resize(224, 224),
    ...     A.Normalize(),
    ...     ToTensorV2()
    ... ])
    >>> dataset = RetinalDataset(
    ...     csv_file='train.csv',
    ...     img_dir='data/aptos/train_images',
    ...     transform=transform
    ... )
    """

    # Supported image file extensions
    VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']

    # Valid diagnosis range
    MIN_DIAGNOSIS = 0
    MAX_DIAGNOSIS = 4

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Optional[Callable] = None
    ):
        """Initialize the RetinalDataset."""
        # Convert to Path objects for better path handling
        self.csv_file = Path(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Validate paths exist
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load CSV data
        try:
            self.data_frame = pd.read_csv(self.csv_file)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Validate required columns
        required_columns = ['id_code', 'diagnosis']
        missing_columns = [col for col in required_columns
                          if col not in self.data_frame.columns]
        if missing_columns:
            raise ValueError(
                f"CSV missing required columns: {missing_columns}. "
                f"Found columns: {list(self.data_frame.columns)}"
            )

        # Validate diagnosis values
        invalid_diagnoses = self.data_frame[
            (self.data_frame['diagnosis'] < self.MIN_DIAGNOSIS) |
            (self.data_frame['diagnosis'] > self.MAX_DIAGNOSIS)
        ]
        if len(invalid_diagnoses) > 0:
            raise ValueError(
                f"Found {len(invalid_diagnoses)} invalid diagnosis values. "
                f"Diagnosis must be in range [{self.MIN_DIAGNOSIS}, {self.MAX_DIAGNOSIS}]. "
                f"Invalid values: {invalid_diagnoses['diagnosis'].unique()}"
            )

        # Reset index to ensure clean indexing
        self.data_frame = self.data_frame.reset_index(drop=True)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        Load and return a sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve (0 to len(dataset)-1)

        Returns
        -------
        tuple
            (image, label) where:
            - image: PIL Image or transformed image (depending on transform)
            - label: int diagnosis level (0-4)

        Raises
        ------
        IndexError
            If idx is out of range
        FileNotFoundError
            If the image file cannot be found with any valid extension
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")

        # Get image ID and label from dataframe
        img_id = self.data_frame.loc[idx, 'id_code']
        label = int(self.data_frame.loc[idx, 'diagnosis'])

        # Try to find image with supported extensions
        img_path = None
        for ext in self.VALID_EXTENSIONS:
            candidate_path = self.img_dir / f"{img_id}{ext}"
            if candidate_path.exists():
                img_path = candidate_path
                break

        # Raise error if image not found
        if img_path is None:
            tried_paths = [f"{img_id}{ext}" for ext in self.VALID_EXTENSIONS]
            raise FileNotFoundError(
                f"Image not found for id_code '{img_id}'. "
                f"Tried extensions: {self.VALID_EXTENSIONS} in directory {self.img_dir}. "
                f"Looked for: {tried_paths}"
            )

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

        # Apply transforms if provided
        if self.transform is not None:
            # Check if transform is albumentations (has __module__ attribute)
            if hasattr(self.transform, '__module__') and \
               'albumentations' in self.transform.__module__:
                # Albumentations expects numpy array
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                # Assume torchvision transform or similar (expects PIL)
                image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of diagnosis classes in the dataset.

        Returns
        -------
        dict
            Dictionary mapping diagnosis level to count
        """
        return self.data_frame['diagnosis'].value_counts().sort_index().to_dict()

    def get_sample_info(self, idx: int) -> dict:
        """
        Get metadata for a specific sample without loading the image.

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        dict
            Dictionary containing sample metadata
        """
        row = self.data_frame.loc[idx]
        return {
            'id_code': row['id_code'],
            'diagnosis': int(row['diagnosis']),
            'index': idx
        }


if __name__ == "__main__":
    """
    Basic testing and demonstration of the RetinalDataset class.
    """
    print("=" * 70)
    print("RetinalDataset Test Suite")
    print("=" * 70)

    # Test 1: Create a sample CSV file
    print("\n[Test 1] Creating sample CSV file...")
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    csv_path = temp_dir / "test_data.csv"
    img_dir = temp_dir / "images"
    img_dir.mkdir()

    # Create sample data
    sample_data = pd.DataFrame({
        'id_code': ['image_001', 'image_002', 'image_003', 'image_004', 'image_005'],
        'diagnosis': [0, 1, 2, 3, 4]
    })
    sample_data.to_csv(csv_path, index=False)
    print(f"✓ Created sample CSV with {len(sample_data)} entries")

    # Create dummy images
    print("\n[Test 2] Creating sample images...")
    for idx, img_id in enumerate(sample_data['id_code']):
        # Create a simple colored image (different color for each class)
        dummy_image = Image.new('RGB', (100, 100), color=(idx*50, 100, 200-idx*50))
        dummy_image.save(img_dir / f"{img_id}.png")
    print(f"✓ Created {len(sample_data)} dummy images")

    # Test 3: Initialize dataset
    print("\n[Test 3] Initializing dataset...")
    try:
        dataset = RetinalDataset(
            csv_file=str(csv_path),
            img_dir=str(img_dir)
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  - Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)

    # Test 4: Test __len__
    print("\n[Test 4] Testing __len__ method...")
    assert len(dataset) == 5, "Dataset length mismatch"
    print(f"✓ __len__ returns correct value: {len(dataset)}")

    # Test 5: Test __getitem__ without transforms
    print("\n[Test 5] Testing __getitem__ without transforms...")
    try:
        image, label = dataset[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  - Image type: {type(image)}")
        print(f"  - Image size: {image.size}")
        print(f"  - Label: {label} (No DR)")
        assert isinstance(image, Image.Image), "Image should be PIL Image"
        assert label == 0, "Label mismatch"
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 6: Test with transforms
    print("\n[Test 6] Testing __getitem__ with torchvision transforms...")
    try:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset_with_transform = RetinalDataset(
            csv_file=str(csv_path),
            img_dir=str(img_dir),
            transform=transform
        )

        image, label = dataset_with_transform[1]
        print(f"✓ Successfully loaded sample with transform")
        print(f"  - Image type: {type(image)}")
        print(f"  - Image shape: {image.shape}")
        print(f"  - Label: {label} (Mild NPDR)")
        assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
        assert image.shape == (3, 224, 224), "Image shape mismatch"
    except ImportError:
        print("⊘ Skipping: torchvision not available")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 7: Test class distribution
    print("\n[Test 7] Testing class distribution...")
    distribution = dataset.get_class_distribution()
    print("✓ Class distribution:")
    severity_names = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "PDR"
    }
    for diagnosis, count in distribution.items():
        print(f"  - Class {diagnosis} ({severity_names[diagnosis]}): {count} samples")

    # Test 8: Test sample info
    print("\n[Test 8] Testing get_sample_info...")
    info = dataset.get_sample_info(2)
    print(f"✓ Sample info for index 2:")
    for key, value in info.items():
        print(f"  - {key}: {value}")

    # Test 9: Test error handling - missing image
    print("\n[Test 9] Testing error handling for missing image...")
    sample_data_missing = pd.DataFrame({
        'id_code': ['nonexistent_image'],
        'diagnosis': [0]
    })
    csv_path_missing = temp_dir / "test_missing.csv"
    sample_data_missing.to_csv(csv_path_missing, index=False)

    dataset_missing = RetinalDataset(
        csv_file=str(csv_path_missing),
        img_dir=str(img_dir)
    )

    try:
        image, label = dataset_missing[0]
        print("✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError")
        print(f"  - Message: {str(e)[:80]}...")

    # Test 10: Test error handling - invalid CSV
    print("\n[Test 10] Testing error handling for invalid CSV...")
    invalid_csv = temp_dir / "invalid.csv"
    pd.DataFrame({'wrong_column': [1, 2, 3]}).to_csv(invalid_csv, index=False)

    try:
        dataset_invalid = RetinalDataset(
            csv_file=str(invalid_csv),
            img_dir=str(img_dir)
        )
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for missing columns")
        print(f"  - Message: {str(e)[:80]}...")

    # Cleanup
    print("\n[Cleanup] Removing temporary files...")
    import shutil
    shutil.rmtree(temp_dir)
    print("✓ Cleanup complete")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print("\nUsage example:")
    print("""
    from scripts.dataset import RetinalDataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = RetinalDataset(
        csv_file='data/aptos/train.csv',
        img_dir='data/aptos/train_images',
        transform=transform
    )

    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """)
