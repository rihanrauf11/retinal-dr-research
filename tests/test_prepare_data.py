"""
Unit Tests for Data Preparation Script (prepare_data.py)

This test suite covers all major functionality of the data preparation script
including verification, statistics, CSV operations, and dataset organization.

Author: Generated with Claude Code
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import pytest
import pandas as pd
import numpy as np
from PIL import Image

# Import functions from prepare_data
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_data import (
    verify_image,
    verify_dataset,
    validate_csv,
    calculate_image_quality_metrics,
    calculate_dataset_statistics,
    create_messidor_csv,
    create_sample_dataset,
    create_train_test_split,
    setup_logging,
    DR_CLASSES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_images(temp_dir):
    """Create sample images for testing."""
    img_dir = temp_dir / 'images'
    img_dir.mkdir()

    # Create 5 valid images (one per class)
    for i in range(5):
        img = Image.new('RGB', (224, 224), color=(i * 50, 100, 150))
        img.save(img_dir / f'image_{i:03d}.png')

    return img_dir


@pytest.fixture
def sample_csv(temp_dir, sample_images):
    """Create sample CSV file."""
    data = {
        'id_code': [f'image_{i:03d}' for i in range(5)],
        'diagnosis': [0, 1, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    csv_path = temp_dir / 'test.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def large_dataset(temp_dir):
    """Create a larger dataset for testing splits and sampling."""
    img_dir = temp_dir / 'images'
    img_dir.mkdir()

    # Create 50 images per class (250 total)
    data = []
    for class_num in range(5):
        for i in range(50):
            id_code = f'img_c{class_num}_n{i:03d}'
            img = Image.new('RGB', (512, 512), color=(class_num * 50, i * 5, 100))
            img.save(img_dir / f'{id_code}.png')
            data.append({'id_code': id_code, 'diagnosis': class_num})

    df = pd.DataFrame(data)
    csv_path = temp_dir / 'data.csv'
    df.to_csv(csv_path, index=False)

    return csv_path, img_dir


@pytest.fixture
def corrupted_image(temp_dir):
    """Create a corrupted image file."""
    img_dir = temp_dir / 'corrupted'
    img_dir.mkdir()

    # Create a file with invalid image data
    corrupted_file = img_dir / 'corrupted.png'
    with open(corrupted_file, 'wb') as f:
        f.write(b'This is not a valid image file')

    return corrupted_file


@pytest.fixture
def test_logger(temp_dir):
    """Create a test logger."""
    log_dir = temp_dir / 'logs'
    logger = setup_logging(log_dir, verbose=False)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageVerification:
    """Test image verification functionality."""

    def test_verify_valid_image(self, sample_images):
        """Test verification of a valid image."""
        img_path = list(sample_images.glob('*.png'))[0]
        is_valid, error = verify_image(img_path)
        assert is_valid is True
        assert error is None

    def test_verify_corrupted_image(self, corrupted_image):
        """Test verification of a corrupted image."""
        is_valid, error = verify_image(corrupted_image)
        assert is_valid is False
        assert error is not None
        assert isinstance(error, str)

    def test_verify_nonexistent_image(self, temp_dir):
        """Test verification of non-existent image."""
        fake_path = temp_dir / 'nonexistent.png'
        is_valid, error = verify_image(fake_path)
        assert is_valid is False
        assert error is not None


class TestDatasetVerification:
    """Test dataset verification functionality."""

    def test_verify_dataset_basic(self, sample_images, sample_csv, test_logger):
        """Test basic dataset verification."""
        results = verify_dataset(sample_images, sample_csv, test_logger)

        assert results['total_images'] == 5
        assert results['valid_images'] == 5
        assert len(results['corrupted_images']) == 0
        assert len(results['missing_images']) == 0
        assert results['csv_valid'] is True

    def test_verify_dataset_without_csv(self, sample_images, test_logger):
        """Test dataset verification without CSV file."""
        results = verify_dataset(sample_images, None, test_logger)

        assert results['total_images'] == 5
        assert results['valid_images'] == 5
        assert results['csv_valid'] is None

    def test_verify_dataset_with_missing_images(self, temp_dir, test_logger):
        """Test verification when images are missing."""
        # Create CSV with more entries than images
        img_dir = temp_dir / 'images'
        img_dir.mkdir()

        # Create only 3 images
        for i in range(3):
            img = Image.new('RGB', (100, 100))
            img.save(img_dir / f'image_{i}.png')

        # CSV references 5 images
        data = {
            'id_code': [f'image_{i}' for i in range(5)],
            'diagnosis': [0, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / 'test.csv'
        df.to_csv(csv_path, index=False)

        results = verify_dataset(img_dir, csv_path, test_logger)

        assert len(results['missing_images']) == 2
        assert 'image_3' in results['missing_images']
        assert 'image_4' in results['missing_images']

    def test_verify_dataset_expected_count(self, sample_images, sample_csv, test_logger):
        """Test verification with expected image count."""
        # Should pass with correct count
        results = verify_dataset(sample_images, sample_csv, test_logger, expected_count=5)
        assert results['total_images'] == 5

        # Should log warning with incorrect count (tested via logger, not asserted here)
        results = verify_dataset(sample_images, sample_csv, test_logger, expected_count=10)
        assert results['total_images'] == 5  # Still finds actual count


# ═══════════════════════════════════════════════════════════════════════════════
# CSV OPERATIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSVOperations:
    """Test CSV creation and validation."""

    def test_validate_csv_valid(self, sample_csv, test_logger):
        """Test validation of a valid CSV file."""
        is_valid = validate_csv(sample_csv, test_logger)
        assert is_valid is True

    def test_validate_csv_missing_columns(self, temp_dir, test_logger):
        """Test validation of CSV with missing columns."""
        # CSV missing 'diagnosis' column
        data = {'id_code': ['img1', 'img2']}
        df = pd.DataFrame(data)
        csv_path = temp_dir / 'invalid.csv'
        df.to_csv(csv_path, index=False)

        is_valid = validate_csv(csv_path, test_logger)
        assert is_valid is False

    def test_validate_csv_invalid_diagnosis(self, temp_dir, test_logger):
        """Test validation of CSV with invalid diagnosis values."""
        data = {
            'id_code': ['img1', 'img2', 'img3'],
            'diagnosis': [0, 5, 10]  # 5 and 10 are invalid
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / 'invalid.csv'
        df.to_csv(csv_path, index=False)

        is_valid = validate_csv(csv_path, test_logger)
        assert is_valid is False

    def test_create_messidor_csv(self, sample_images, temp_dir, test_logger):
        """Test creation of Messidor CSV template."""
        output_csv = temp_dir / 'messidor.csv'

        success = create_messidor_csv(sample_images, output_csv, test_logger, dry_run=False)

        assert success is True
        assert output_csv.exists()

        df = pd.read_csv(output_csv)
        assert len(df) == 5
        assert 'id_code' in df.columns
        assert 'diagnosis' in df.columns
        # All diagnoses should be 0 (placeholder)
        assert all(df['diagnosis'] == 0)

    def test_create_messidor_csv_no_images(self, temp_dir, test_logger):
        """Test CSV creation when no images exist."""
        empty_dir = temp_dir / 'empty'
        empty_dir.mkdir()
        output_csv = temp_dir / 'messidor.csv'

        success = create_messidor_csv(empty_dir, output_csv, test_logger, dry_run=False)

        assert success is False

    def test_create_messidor_csv_dry_run(self, sample_images, temp_dir, test_logger):
        """Test CSV creation in dry run mode."""
        output_csv = temp_dir / 'messidor.csv'

        success = create_messidor_csv(sample_images, output_csv, test_logger, dry_run=True)

        assert success is True
        assert not output_csv.exists()  # File should not be created in dry run


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatistics:
    """Test statistics calculation functionality."""

    def test_calculate_image_quality_metrics(self, sample_images):
        """Test calculation of image quality metrics."""
        img_path = list(sample_images.glob('*.png'))[0]
        metrics = calculate_image_quality_metrics(img_path)

        assert 'brightness' in metrics
        assert 'contrast' in metrics
        assert 'sharpness' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_calculate_dataset_statistics(self, sample_images, sample_csv, test_logger):
        """Test calculation of dataset statistics."""
        stats = calculate_dataset_statistics(sample_images, sample_csv, test_logger, sample_size=5)

        assert stats['total_images'] == 5
        assert 'class_distribution' in stats
        assert len(stats['class_distribution']) == 5

        # Check class distribution
        for class_num in range(5):
            assert class_num in stats['class_distribution']
            assert stats['class_distribution'][class_num]['count'] == 1
            assert stats['class_distribution'][class_num]['name'] == DR_CLASSES[class_num]

        # Check dimension summary
        assert 'dimension_summary' in stats
        assert 'width' in stats['dimension_summary']
        assert 'height' in stats['dimension_summary']

        # Check file size summary
        assert 'file_size_summary' in stats
        assert 'mean_mb' in stats['file_size_summary']

        # Check quality summary
        assert 'quality_summary' in stats

    def test_calculate_dataset_statistics_no_csv(self, sample_images, test_logger):
        """Test statistics calculation without CSV file."""
        stats = calculate_dataset_statistics(sample_images, None, test_logger, sample_size=5)

        assert stats['total_images'] == 5
        assert stats.get('class_distribution') == {}  # No CSV, no distribution

    def test_calculate_dataset_statistics_large_sample(self, temp_dir, test_logger):
        """Test statistics with sample size smaller than dataset."""
        # Create 20 images
        img_dir = temp_dir / 'images'
        img_dir.mkdir()

        for i in range(20):
            img = Image.new('RGB', (300, 300))
            img.save(img_dir / f'img_{i:03d}.png')

        stats = calculate_dataset_statistics(img_dir, None, test_logger, sample_size=10)

        assert stats['total_images'] == 20
        # Statistics should be based on 10 sample images


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATASET CREATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleDatasetCreation:
    """Test sample dataset creation functionality."""

    def test_create_sample_dataset_basic(self, large_dataset, temp_dir, test_logger):
        """Test basic sample dataset creation."""
        source_csv, source_img_dir = large_dataset
        output_csv = temp_dir / 'sample' / 'sample.csv'
        output_img_dir = temp_dir / 'sample' / 'images'

        success = create_sample_dataset(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_csv=output_csv,
            output_img_dir=output_img_dir,
            samples_per_class=10,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        assert success is True
        assert output_csv.exists()
        assert output_img_dir.exists()

        # Verify CSV
        df = pd.read_csv(output_csv)
        assert len(df) == 50  # 10 per class * 5 classes

        # Check class distribution
        for class_num in range(5):
            count = len(df[df['diagnosis'] == class_num])
            assert count == 10

        # Verify images were copied
        assert len(list(output_img_dir.glob('*.png'))) == 50

    def test_create_sample_dataset_insufficient_samples(self, temp_dir, test_logger):
        """Test sample creation when class has fewer samples than requested."""
        # Create dataset with unbalanced classes
        img_dir = temp_dir / 'images'
        img_dir.mkdir()

        data = []
        # Class 0: 5 samples (less than requested 10)
        for i in range(5):
            id_code = f'img_c0_n{i:03d}'
            img = Image.new('RGB', (100, 100))
            img.save(img_dir / f'{id_code}.png')
            data.append({'id_code': id_code, 'diagnosis': 0})

        # Class 1: 20 samples
        for i in range(20):
            id_code = f'img_c1_n{i:03d}'
            img = Image.new('RGB', (100, 100))
            img.save(img_dir / f'{id_code}.png')
            data.append({'id_code': id_code, 'diagnosis': 1})

        df = pd.DataFrame(data)
        csv_path = temp_dir / 'data.csv'
        df.to_csv(csv_path, index=False)

        output_csv = temp_dir / 'sample' / 'sample.csv'
        output_img_dir = temp_dir / 'sample' / 'images'

        success = create_sample_dataset(
            source_csv=csv_path,
            source_img_dir=img_dir,
            output_csv=output_csv,
            output_img_dir=output_img_dir,
            samples_per_class=10,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        assert success is True

        # Verify: class 0 should have all 5, class 1 should have 10
        sample_df = pd.read_csv(output_csv)
        assert len(sample_df[sample_df['diagnosis'] == 0]) == 5
        assert len(sample_df[sample_df['diagnosis'] == 1]) == 10

    def test_create_sample_dataset_dry_run(self, large_dataset, temp_dir, test_logger):
        """Test sample creation in dry run mode."""
        source_csv, source_img_dir = large_dataset
        output_csv = temp_dir / 'sample' / 'sample.csv'
        output_img_dir = temp_dir / 'sample' / 'images'

        success = create_sample_dataset(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_csv=output_csv,
            output_img_dir=output_img_dir,
            samples_per_class=10,
            logger=test_logger,
            dry_run=True,
            seed=42
        )

        assert success is True
        assert not output_csv.exists()
        assert not output_img_dir.exists()

    def test_create_sample_dataset_reproducible(self, large_dataset, temp_dir, test_logger):
        """Test that sample creation is reproducible with same seed."""
        source_csv, source_img_dir = large_dataset

        # Create first sample
        output_csv1 = temp_dir / 'sample1' / 'sample.csv'
        output_img_dir1 = temp_dir / 'sample1' / 'images'

        create_sample_dataset(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_csv=output_csv1,
            output_img_dir=output_img_dir1,
            samples_per_class=10,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        # Create second sample with same seed
        output_csv2 = temp_dir / 'sample2' / 'sample.csv'
        output_img_dir2 = temp_dir / 'sample2' / 'images'

        create_sample_dataset(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_csv=output_csv2,
            output_img_dir=output_img_dir2,
            samples_per_class=10,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        # Compare CSVs
        df1 = pd.read_csv(output_csv1)
        df2 = pd.read_csv(output_csv2)

        # Should have same id_codes (order might differ due to shuffling, so sort)
        assert set(df1['id_code']) == set(df2['id_code'])


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainTestSplit:
    """Test train/test split functionality."""

    def test_create_train_test_split_basic(self, large_dataset, temp_dir, test_logger):
        """Test basic train/test split creation."""
        source_csv, source_img_dir = large_dataset
        output_dir = temp_dir / 'split'

        success = create_train_test_split(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_dir=output_dir,
            split_ratio=0.8,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        assert success is True

        # Check files exist
        assert (output_dir / 'train.csv').exists()
        assert (output_dir / 'test.csv').exists()
        assert (output_dir / 'train_images').exists()
        assert (output_dir / 'test_images').exists()

        # Check split ratio
        train_df = pd.read_csv(output_dir / 'train.csv')
        test_df = pd.read_csv(output_dir / 'test.csv')

        total = len(train_df) + len(test_df)
        train_ratio = len(train_df) / total

        assert abs(train_ratio - 0.8) < 0.05  # Within 5% of target

    def test_create_train_test_split_stratified(self, large_dataset, temp_dir, test_logger):
        """Test that split maintains class distribution (stratified)."""
        source_csv, source_img_dir = large_dataset
        output_dir = temp_dir / 'split'

        create_train_test_split(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_dir=output_dir,
            split_ratio=0.8,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        train_df = pd.read_csv(output_dir / 'train.csv')
        test_df = pd.read_csv(output_dir / 'test.csv')

        # Check each class is proportionally split
        for class_num in range(5):
            train_count = len(train_df[train_df['diagnosis'] == class_num])
            test_count = len(test_df[test_df['diagnosis'] == class_num])

            # Should be approximately 40 train, 10 test (80/20 split of 50 per class)
            assert train_count == 40
            assert test_count == 10

    def test_create_train_test_split_different_ratios(self, large_dataset, temp_dir, test_logger):
        """Test split with different ratios."""
        source_csv, source_img_dir = large_dataset

        for split_ratio in [0.5, 0.7, 0.9]:
            output_dir = temp_dir / f'split_{split_ratio}'

            create_train_test_split(
                source_csv=source_csv,
                source_img_dir=source_img_dir,
                output_dir=output_dir,
                split_ratio=split_ratio,
                logger=test_logger,
                dry_run=False,
                seed=42
            )

            train_df = pd.read_csv(output_dir / 'train.csv')
            test_df = pd.read_csv(output_dir / 'test.csv')

            total = len(train_df) + len(test_df)
            actual_ratio = len(train_df) / total

            assert abs(actual_ratio - split_ratio) < 0.05

    def test_create_train_test_split_dry_run(self, large_dataset, temp_dir, test_logger):
        """Test split in dry run mode."""
        source_csv, source_img_dir = large_dataset
        output_dir = temp_dir / 'split'

        success = create_train_test_split(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_dir=output_dir,
            split_ratio=0.8,
            logger=test_logger,
            dry_run=True,
            seed=42
        )

        assert success is True
        assert not (output_dir / 'train.csv').exists()
        assert not (output_dir / 'test.csv').exists()

    def test_create_train_test_split_reproducible(self, large_dataset, temp_dir, test_logger):
        """Test that split is reproducible with same seed."""
        source_csv, source_img_dir = large_dataset

        # First split
        output_dir1 = temp_dir / 'split1'
        create_train_test_split(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_dir=output_dir1,
            split_ratio=0.8,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        # Second split with same seed
        output_dir2 = temp_dir / 'split2'
        create_train_test_split(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_dir=output_dir2,
            split_ratio=0.8,
            logger=test_logger,
            dry_run=False,
            seed=42
        )

        # Compare
        train_df1 = pd.read_csv(output_dir1 / 'train.csv')
        train_df2 = pd.read_csv(output_dir2 / 'train.csv')

        # Should have same id_codes
        assert set(train_df1['id_code']) == set(train_df2['id_code'])


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogging:
    """Test logging setup."""

    def test_setup_logging_creates_file(self, temp_dir):
        """Test that logging creates log file."""
        log_dir = temp_dir / 'logs'
        logger = setup_logging(log_dir, verbose=False)

        assert log_dir.exists()
        log_files = list(log_dir.glob('*.log'))
        assert len(log_files) == 1

        # Test logging works
        logger.info("Test message")

        # Check log file contains message
        with open(log_files[0], 'r') as f:
            content = f.read()
            assert "Test message" in content

    def test_setup_logging_verbose(self, temp_dir):
        """Test logging with verbose mode."""
        log_dir = temp_dir / 'logs'
        logger = setup_logging(log_dir, verbose=True)

        # Should have console handler at INFO level
        assert any(
            handler.level <= 20  # INFO level
            for handler in logger.handlers
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline_sample_creation(self, large_dataset, temp_dir, test_logger):
        """Test complete pipeline: verify -> stats -> sample."""
        source_csv, source_img_dir = large_dataset

        # 1. Verify dataset
        verify_results = verify_dataset(source_img_dir, source_csv, test_logger)
        assert verify_results['valid_images'] == 250

        # 2. Calculate statistics
        stats = calculate_dataset_statistics(source_img_dir, source_csv, test_logger)
        assert stats['total_images'] == 250

        # 3. Create sample
        output_csv = temp_dir / 'sample' / 'sample.csv'
        output_img_dir = temp_dir / 'sample' / 'images'

        success = create_sample_dataset(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_csv=output_csv,
            output_img_dir=output_img_dir,
            samples_per_class=10,
            logger=test_logger,
            dry_run=False,
            seed=42
        )
        assert success is True

        # 4. Verify sample
        sample_verify = verify_dataset(output_img_dir, output_csv, test_logger)
        assert sample_verify['valid_images'] == 50

    def test_full_pipeline_split_creation(self, large_dataset, temp_dir, test_logger):
        """Test complete pipeline: verify -> split -> verify splits."""
        source_csv, source_img_dir = large_dataset

        # 1. Verify original
        verify_results = verify_dataset(source_img_dir, source_csv, test_logger)
        assert verify_results['valid_images'] == 250

        # 2. Create split
        split_dir = temp_dir / 'split'
        success = create_train_test_split(
            source_csv=source_csv,
            source_img_dir=source_img_dir,
            output_dir=split_dir,
            split_ratio=0.8,
            logger=test_logger,
            dry_run=False,
            seed=42
        )
        assert success is True

        # 3. Verify train split
        train_verify = verify_dataset(
            split_dir / 'train_images',
            split_dir / 'train.csv',
            test_logger
        )
        assert train_verify['valid_images'] == 200

        # 4. Verify test split
        test_verify = verify_dataset(
            split_dir / 'test_images',
            split_dir / 'test.csv',
            test_logger
        )
        assert test_verify['valid_images'] == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
