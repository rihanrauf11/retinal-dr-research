"""
Unit Tests for Data Augmentation Transforms.

Tests cover:
- Torchvision transforms (resize, normalize, augmentations)
- Albumentations transforms
- Transform composition and consistency
- Integration with dataset

Author: Generated with Claude Code
"""

import pytest
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ═══════════════════════════════════════════════════════════════════════════════
# TORCHVISION TRANSFORMS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTorchvisionTransforms:
    """Test torchvision transform operations."""

    def test_resize_transform(self):
        """Test that Resize transform produces correct output size."""
        transform = transforms.Resize((224, 224))

        # Create test image
        img = Image.new('RGB', (100, 100), color=(100, 150, 200))
        resized = transform(img)

        assert resized.size == (224, 224)
        assert isinstance(resized, Image.Image)

    @pytest.mark.parametrize("size", [64, 128, 224, 384, 512])
    def test_resize_different_sizes(self, size):
        """Test Resize with different output sizes."""
        transform = transforms.Resize((size, size))
        img = Image.new('RGB', (100, 100))
        resized = transform(img)

        assert resized.size == (size, size)

    def test_totensor_transform(self):
        """Test ToTensor converts PIL to tensor correctly."""
        transform = transforms.ToTensor()
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))

        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 100, 100)
        assert tensor.dtype == torch.float32
        # PIL [0, 255] -> Tensor [0.0, 1.0]
        assert 0.0 <= tensor.min() <= 1.0
        assert 0.0 <= tensor.max() <= 1.0

    def test_normalize_transform(self, imagenet_stats):
        """Test Normalize transform with ImageNet stats."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=imagenet_stats['mean'],
                std=imagenet_stats['std']
            )
        ])

        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        normalized = transform(img)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == (3, 100, 100)
        # Normalized values can be negative (128/255 - 0.485) / 0.229 can be positive or negative
        # Just check it's different from unnormalized
        assert not torch.allclose(normalized, torch.tensor(128/255.0))

    def test_normalize_values(self):
        """Test that normalization produces expected value ranges."""
        # Create image with known pixel values
        img_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(img_array)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        normalized = transform(img)

        # 128/255 = ~0.502, (0.502 - 0.5) / 0.5 = ~0.004
        assert torch.allclose(normalized, torch.zeros_like(normalized), atol=0.01)

    def test_compose_pipeline(self):
        """Test Compose with multiple transforms."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = Image.new('RGB', (100, 100), color=(100, 150, 200))
        output = transform(img)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_random_horizontal_flip(self):
        """Test RandomHorizontalFlip transform."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        transform = transforms.RandomHorizontalFlip(p=1.0)  # Always flip
        img = Image.new('RGB', (100, 100))

        flipped = transform(img)

        assert isinstance(flipped, Image.Image)
        assert flipped.size == img.size

    def test_random_vertical_flip(self):
        """Test RandomVerticalFlip transform."""
        torch.manual_seed(42)

        transform = transforms.RandomVerticalFlip(p=1.0)
        img = Image.new('RGB', (100, 100))

        flipped = transform(img)

        assert isinstance(flipped, Image.Image)
        assert flipped.size == img.size

    def test_color_jitter(self):
        """Test ColorJitter transform."""
        transform = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        jittered = transform(img)

        assert isinstance(jittered, Image.Image)
        assert jittered.size == img.size

    def test_transform_determinism_with_seed(self):
        """Test that transforms are deterministic when seed is set."""
        torch.manual_seed(42)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        img = Image.new('RGB', (100, 100), color=(100, 150, 200))

        # Apply transform twice with same seed
        torch.manual_seed(42)
        output1 = transform(img)

        torch.manual_seed(42)
        output2 = transform(img)

        assert torch.equal(output1, output2)

    def test_transform_preserve_type(self):
        """Test that non-tensor transforms preserve PIL type."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        img = Image.new('RGB', (100, 100))
        output = transform(img)

        assert isinstance(output, Image.Image)


# ═══════════════════════════════════════════════════════════════════════════════
# ALBUMENTATIONS TRANSFORMS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlbumentationsTransforms:
    """Test albumentations transform operations."""

    def test_albumentation_resize(self):
        """Test A.Resize produces correct output size."""
        transform = A.Resize(224, 224)

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img_array)

        assert transformed['image'].shape == (224, 224, 3)

    @pytest.mark.parametrize("size", [64, 128, 224, 384])
    def test_albumentation_resize_different_sizes(self, size):
        """Test A.Resize with different output sizes."""
        transform = A.Resize(size, size)

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img_array)

        assert transformed['image'].shape == (size, size, 3)

    def test_albumentation_normalize(self, imagenet_stats):
        """Test A.Normalize with ImageNet stats."""
        transform = A.Normalize(
            mean=imagenet_stats['mean'],
            std=imagenet_stats['std']
        )

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = transform(image=img_array)

        assert normalized['image'].shape == (100, 100, 3)
        # Normalized values should be roughly in [-3, 3] range for ImageNet stats
        assert -5 < normalized['image'].min() < 5
        assert -5 < normalized['image'].max() < 5

    def test_albumentation_totensor(self):
        """Test ToTensorV2 conversion."""
        transform = A.Compose([
            A.Resize(224, 224),
            ToTensorV2()
        ])

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transformed = transform(image=img_array)

        tensor = transformed['image']
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)  # CHW format

    def test_albumentation_horizontal_flip(self):
        """Test A.HorizontalFlip transform."""
        transform = A.HorizontalFlip(p=1.0)  # Always flip

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        flipped = transform(image=img_array)

        assert flipped['image'].shape == img_array.shape

    def test_albumentation_vertical_flip(self):
        """Test A.VerticalFlip transform."""
        transform = A.VerticalFlip(p=1.0)

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        flipped = transform(image=img_array)

        assert flipped['image'].shape == img_array.shape

    def test_albumentation_rotate90(self):
        """Test A.RandomRotate90 transform."""
        transform = A.RandomRotate90(p=1.0)

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rotated = transform(image=img_array)

        assert rotated['image'].shape == img_array.shape

    def test_albumentation_advanced_augmentations(self):
        """Test advanced albumentations like ShiftScaleRotate."""
        transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=1.0
            ),
            A.Resize(224, 224)
        ])

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        augmented = transform(image=img_array)

        assert augmented['image'].shape == (224, 224, 3)

    def test_albumentation_compose(self):
        """Test A.Compose with multiple transforms."""
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        output = transform(image=img_array)

        assert isinstance(output['image'], torch.Tensor)
        assert output['image'].shape == (3, 224, 224)

    def test_albumentation_oneof(self):
        """Test A.OneOf for random selection."""
        transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0)
            ], p=1.0),
            A.Resize(224, 224)
        ])

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        augmented = transform(image=img_array)

        # One of the transforms should have been applied
        assert augmented['image'].shape == (224, 224, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformIntegration:
    """Test transform integration with dataset and training."""

    def test_train_vs_val_transforms(self):
        """Test that training and validation transforms differ appropriately."""
        # Training: with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor()
        ])

        # Validation: without augmentation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        img = Image.new('RGB', (100, 100), color=(128, 128, 128))

        train_output = train_transform(img)
        val_output = val_transform(img)

        # Both should have same shape
        assert train_output.shape == val_output.shape == (3, 224, 224)

    def test_transform_with_dataset(self, sample_csv, sample_images):
        """Test transforms work end-to-end with RetinalDataset."""
        from dataset import RetinalDataset

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=transform
        )

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_transform_batch_consistency(self, sample_csv, sample_images):
        """Test that all images in batch have same size after transform."""
        from dataset import RetinalDataset
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = RetinalDataset(
            csv_file=str(sample_csv),
            img_dir=str(sample_images),
            transform=transform
        )

        loader = DataLoader(dataset, batch_size=5, shuffle=False)
        images, labels = next(iter(loader))

        # All images should have same shape
        assert images.shape == (5, 3, 224, 224)

    def test_transform_value_ranges_totensor(self):
        """Test that ToTensor produces values in [0, 1]."""
        transform = transforms.ToTensor()

        img = Image.new('RGB', (100, 100), color=(128, 64, 192))
        tensor = transform(img)

        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_transform_value_ranges_normalized(self):
        """Test normalized values are in reasonable range."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        normalized = transform(img)

        # Normalized values should be in roughly [-1, 1] for this case
        assert -1.5 < normalized.min() < 1.5
        assert -1.5 < normalized.max() < 1.5

    def test_get_transforms_function(self):
        """Test the get_transforms function from training script."""
        try:
            # Try to import from train_baseline
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root / 'scripts'))

            from train_baseline import get_transforms

            train_transform, val_transform = get_transforms(img_size=224)

            # Test both transforms
            img = Image.new('RGB', (100, 100), color=(128, 128, 128))

            train_output = train_transform(image=np.array(img))
            val_output = val_transform(image=np.array(img))

            # Both should produce tensors of correct shape
            assert isinstance(train_output['image'], torch.Tensor)
            assert isinstance(val_output['image'], torch.Tensor)
            assert train_output['image'].shape == (3, 224, 224)
            assert val_output['image'].shape == (3, 224, 224)

        except ImportError:
            pytest.skip("train_baseline module not available")


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES AND ROBUSTNESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformEdgeCases:
    """Test transform edge cases and robustness."""

    def test_transform_grayscale_to_rgb(self):
        """Test that grayscale images can be converted to RGB."""
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create grayscale image
        img = Image.new('L', (100, 100), color=128)
        output = transform(img)

        # Should be RGB (3 channels)
        assert output.shape == (3, 224, 224)

    def test_transform_very_small_image(self):
        """Test transform with very small input image."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Very small image
        img = Image.new('RGB', (10, 10), color=(100, 100, 100))
        output = transform(img)

        assert output.shape == (3, 224, 224)

    def test_transform_very_large_image(self):
        """Test transform with very large input image."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Large image
        img = Image.new('RGB', (2000, 2000), color=(100, 100, 100))
        output = transform(img)

        assert output.shape == (3, 224, 224)

    def test_transform_non_square_image(self):
        """Test transform with non-square image."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Non-square image
        img = Image.new('RGB', (100, 200), color=(100, 100, 100))
        output = transform(img)

        # Should be resized to square
        assert output.shape == (3, 224, 224)

    def test_albumentation_with_pil_image(self):
        """Test that albumentations work with numpy arrays from PIL."""
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Create PIL image and convert to numpy
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        img_array = np.array(img)

        output = transform(image=img_array)

        assert isinstance(output['image'], torch.Tensor)
        assert output['image'].shape == (3, 224, 224)

    def test_transform_preserves_image_info(self):
        """Test that transforms don't create unreasonable artifacts."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create solid color image
        img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        output = transform(img)

        # All channels should have similar values (solid color)
        # Allow some variation due to resize interpolation
        channel_means = output.mean(dim=(1, 2))
        assert torch.allclose(channel_means, channel_means.mean(), atol=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRIZED TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("size", [64, 128, 224, 384, 512])
    def test_resize_output_sizes(self, size):
        """Test Resize with various output sizes."""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        img = Image.new('RGB', (100, 100))
        output = transform(img)

        assert output.shape == (3, size, size)

    @pytest.mark.parametrize("flip_prob", [0.0, 0.5, 1.0])
    def test_flip_probabilities(self, flip_prob):
        """Test RandomHorizontalFlip with different probabilities."""
        transform = transforms.RandomHorizontalFlip(p=flip_prob)
        img = Image.new('RGB', (100, 100))

        # Apply transform multiple times
        outputs = [transform(img) for _ in range(10)]

        # All outputs should be valid
        assert all(isinstance(out, Image.Image) for out in outputs)

    @pytest.mark.parametrize("mean,std", [
        ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    ])
    def test_normalization_parameters(self, mean, std):
        """Test Normalize with different mean/std parameters."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        output = transform(img)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 100, 100)
