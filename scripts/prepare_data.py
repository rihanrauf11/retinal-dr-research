#!/usr/bin/env python3
"""
Data Preparation Script for Diabetic Retinopathy Classification

This script provides comprehensive functionality for downloading, organizing,
verifying, and preparing diabetic retinopathy datasets (APTOS, Messidor).

Features:
- Download datasets from Kaggle API
- Verify dataset integrity and detect corrupted images
- Create standardized directory structure
- Generate CSV files in correct format
- Calculate dataset statistics
- Create sample subsets for testing
- Split datasets into train/test

Usage:
    # Download and prepare all datasets
    python scripts/prepare_data.py --datasets aptos messidor

    # Download APTOS only and create sample
    python scripts/prepare_data.py --aptos-only --create-sample

    # Verify existing datasets
    python scripts/prepare_data.py --verify-only

    # Calculate statistics only
    python scripts/prepare_data.py --stats-only

    # Dry run (no file operations)
    python scripts/prepare_data.py --datasets aptos --dry-run

Author: Generated with Claude Code
"""

import os
import sys
import json
import shutil
import zipfile
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

# Import utility functions
try:
    from scripts.utils import set_seed, create_progress_bar
except ImportError:
    # If running as script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.utils import set_seed, create_progress_bar


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Dataset metadata
DATASET_INFO = {
    'aptos': {
        'name': 'APTOS 2019 Blindness Detection',
        'kaggle_dataset': 'aptos2019-blindness-detection',
        'kaggle_competition': True,
        'train_count': 3662,
        'test_count': 1928,
        'format': 'png',
        'has_labels': True,
    },
    'messidor': {
        'name': 'Messidor Diabetic Retinopathy',
        'kaggle_dataset': None,  # Requires manual download
        'manual_download': True,
        'url': 'https://www.adcis.net/en/third-party/messidor2/',
        'format': 'tiff/png',
        'has_labels': True,
    }
}

# DR class names
DR_CLASSES = {
    0: 'No DR',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'Proliferative DR'
}

# Image extensions to check
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path, verbose: bool = True) -> logging.Logger:
    """
    Set up logging to file and console.

    Parameters
    ----------
    log_dir : Path
        Directory to save log files
    verbose : bool
        If True, set console logging to INFO level, else WARNING

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prepare_data_{timestamp}.log'

    # Create logger
    logger = logging.getLogger('prepare_data')
    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (INFO or WARNING level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_kaggle_api() -> bool:
    """
    Check if Kaggle API is installed and configured.

    Returns
    -------
    bool
        True if Kaggle API is available, False otherwise
    """
    try:
        import kaggle
        # Try to authenticate
        kaggle.api.authenticate()
        return True
    except ImportError:
        return False
    except Exception as e:
        logging.error(f"Kaggle API authentication failed: {e}")
        return False


def download_aptos_dataset(
    output_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """
    Download APTOS 2019 dataset from Kaggle.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded dataset
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual download

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if not check_kaggle_api():
        logger.error("Kaggle API not available. Please install: pip install kaggle")
        logger.error("And configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return False

    if dry_run:
        logger.info("[DRY RUN] Would download APTOS dataset from Kaggle")
        return True

    logger.info("Downloading APTOS 2019 dataset from Kaggle...")

    try:
        import kaggle

        # Create download directory
        download_dir = output_dir / 'downloads'
        download_dir.mkdir(parents=True, exist_ok=True)

        # Download competition files
        competition_name = 'aptos2019-blindness-detection'
        logger.info(f"Downloading competition: {competition_name}")

        kaggle.api.competition_download_files(
            competition=competition_name,
            path=str(download_dir),
            quiet=False
        )

        logger.info("Download completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error downloading APTOS dataset: {e}")
        return False


def download_messidor_instructions(logger: logging.Logger) -> None:
    """
    Display instructions for downloading Messidor dataset (manual download required).

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    """
    info = DATASET_INFO['messidor']

    logger.info("\n" + "=" * 80)
    logger.info("MESSIDOR DATASET - MANUAL DOWNLOAD REQUIRED")
    logger.info("=" * 80)
    logger.info("\nThe Messidor dataset requires manual download with signed agreement.")
    logger.info(f"\n📥 Download Instructions:")
    logger.info(f"1. Visit: {info['url']}")
    logger.info("2. Fill out the form with your information")
    logger.info("3. Download the dataset after receiving approval")
    logger.info("4. Extract files to: data/messidor/raw/")
    logger.info("5. Run this script again with --messidor-organize flag")
    logger.info("\n" + "=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE EXTRACTION AND ORGANIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_zip_files(
    zip_dir: Path,
    extract_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """
    Extract all ZIP files in a directory.

    Parameters
    ----------
    zip_dir : Path
        Directory containing ZIP files
    extract_dir : Path
        Directory to extract files to
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual extraction

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    zip_files = list(zip_dir.glob('*.zip'))

    if not zip_files:
        logger.warning(f"No ZIP files found in {zip_dir}")
        return False

    logger.info(f"Found {len(zip_files)} ZIP file(s) to extract")

    for zip_file in zip_files:
        if dry_run:
            logger.info(f"[DRY RUN] Would extract: {zip_file.name}")
            continue

        logger.info(f"Extracting: {zip_file.name}")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()

                # Extract with progress bar
                with create_progress_bar(
                    file_list,
                    desc=f"Extracting {zip_file.name}",
                    unit="file"
                ) as pbar:
                    for file in pbar:
                        zip_ref.extract(file, extract_dir)

            logger.info(f"Successfully extracted: {zip_file.name}")

        except Exception as e:
            logger.error(f"Error extracting {zip_file.name}: {e}")
            return False

    return True


def organize_aptos_dataset(
    raw_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """
    Organize APTOS dataset into standard directory structure.

    Expected raw structure:
        raw_dir/
        ├── train_images/
        ├── test_images/
        ├── train.csv
        └── test.csv (if available)

    Output structure:
        output_dir/
        ├── train_images/
        ├── test_images/
        ├── train.csv
        └── test.csv

    Parameters
    ----------
    raw_dir : Path
        Directory containing extracted APTOS files
    output_dir : Path
        Output directory for organized dataset
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual file operations

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if dry_run:
        logger.info("[DRY RUN] Would organize APTOS dataset")
        return True

    logger.info("Organizing APTOS dataset...")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy train CSV
        train_csv_src = raw_dir / 'train.csv'
        if train_csv_src.exists():
            shutil.copy2(train_csv_src, output_dir / 'train.csv')
            logger.info("Copied train.csv")
        else:
            logger.error(f"train.csv not found in {raw_dir}")
            return False

        # Copy test CSV if exists
        test_csv_src = raw_dir / 'test.csv'
        if test_csv_src.exists():
            shutil.copy2(test_csv_src, output_dir / 'test.csv')
            logger.info("Copied test.csv")

        # Copy train images
        train_img_src = raw_dir / 'train_images'
        train_img_dst = output_dir / 'train_images'
        if train_img_src.exists():
            if train_img_dst.exists():
                shutil.rmtree(train_img_dst)
            shutil.copytree(train_img_src, train_img_dst)
            logger.info(f"Copied train_images/ ({len(list(train_img_dst.glob('*')))} files)")
        else:
            logger.error(f"train_images/ not found in {raw_dir}")
            return False

        # Copy test images if exists
        test_img_src = raw_dir / 'test_images'
        test_img_dst = output_dir / 'test_images'
        if test_img_src.exists():
            if test_img_dst.exists():
                shutil.rmtree(test_img_dst)
            shutil.copytree(test_img_src, test_img_dst)
            logger.info(f"Copied test_images/ ({len(list(test_img_dst.glob('*')))} files)")

        logger.info("✓ APTOS dataset organized successfully")
        return True

    except Exception as e:
        logger.error(f"Error organizing APTOS dataset: {e}")
        return False


def organize_messidor_dataset(
    raw_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """
    Organize Messidor dataset into standard directory structure.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw Messidor files
    output_dir : Path
        Output directory for organized dataset
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual file operations

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if dry_run:
        logger.info("[DRY RUN] Would organize Messidor dataset")
        return True

    logger.info("Organizing Messidor dataset...")

    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        logger.info("Please download Messidor dataset manually first.")
        download_messidor_instructions(logger)
        return False

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_dir / 'images'
        img_dir.mkdir(exist_ok=True)

        # Find and copy all images
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(raw_dir.rglob(f'*{ext}'))

        if not image_files:
            logger.error(f"No image files found in {raw_dir}")
            return False

        logger.info(f"Found {len(image_files)} images")

        # Copy images with progress bar
        for img_file in create_progress_bar(image_files, desc="Copying images", unit="file"):
            dst_path = img_dir / img_file.name
            shutil.copy2(img_file, dst_path)

        logger.info("✓ Messidor dataset organized successfully")
        logger.info("⚠️  Note: You may need to create annotations.csv manually")
        return True

    except Exception as e:
        logger.error(f"Error organizing Messidor dataset: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify that an image can be loaded and is not corrupted.

    Parameters
    ----------
    image_path : Path
        Path to image file

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    try:
        # Try to open with PIL
        with Image.open(image_path) as img:
            img.verify()

        # Try to load with PIL again (verify() closes the file)
        with Image.open(image_path) as img:
            img.load()

        return True, None

    except Exception as e:
        return False, str(e)


def verify_dataset(
    img_dir: Path,
    csv_file: Optional[Path],
    logger: logging.Logger,
    expected_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Verify dataset integrity by checking images and CSV consistency.

    Parameters
    ----------
    img_dir : Path
        Directory containing images
    csv_file : Optional[Path]
        Path to CSV file with labels (if available)
    logger : logging.Logger
        Logger instance
    expected_count : Optional[int]
        Expected number of images

    Returns
    -------
    Dict[str, Any]
        Verification results dictionary
    """
    logger.info(f"Verifying dataset: {img_dir}")

    results = {
        'total_images': 0,
        'valid_images': 0,
        'corrupted_images': [],
        'missing_images': [],
        'csv_valid': None,
    }

    # Get all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(img_dir.glob(f'*{ext}'))

    results['total_images'] = len(image_files)
    logger.info(f"Found {results['total_images']} image files")

    if expected_count and results['total_images'] != expected_count:
        logger.warning(
            f"Image count mismatch! Expected: {expected_count}, Found: {results['total_images']}"
        )

    # Verify each image
    logger.info("Verifying image integrity...")
    for img_path in create_progress_bar(image_files, desc="Verifying images", unit="file"):
        is_valid, error = verify_image(img_path)
        if is_valid:
            results['valid_images'] += 1
        else:
            results['corrupted_images'].append({
                'path': str(img_path),
                'error': error
            })
            logger.warning(f"Corrupted image: {img_path.name} - {error}")

    # Verify CSV if provided
    if csv_file and csv_file.exists():
        try:
            df = pd.read_csv(csv_file)

            # Check required columns
            if 'id_code' not in df.columns or 'diagnosis' not in df.columns:
                results['csv_valid'] = False
                logger.error("CSV missing required columns: 'id_code' and/or 'diagnosis'")
            else:
                results['csv_valid'] = True

                # Check for missing images
                for id_code in df['id_code']:
                    # Try different extensions
                    found = False
                    for ext in IMAGE_EXTENSIONS:
                        img_path = img_dir / f"{id_code}{ext}"
                        if img_path.exists():
                            found = True
                            break

                    if not found:
                        results['missing_images'].append(id_code)

                if results['missing_images']:
                    logger.warning(f"Found {len(results['missing_images'])} images in CSV but not in directory")

                # Verify diagnosis values
                invalid_diagnoses = df[~df['diagnosis'].isin([0, 1, 2, 3, 4])]
                if len(invalid_diagnoses) > 0:
                    logger.warning(f"Found {len(invalid_diagnoses)} rows with invalid diagnosis values")

        except Exception as e:
            results['csv_valid'] = False
            logger.error(f"Error reading CSV: {e}")

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total images: {results['total_images']}")
    logger.info(f"Valid images: {results['valid_images']}")
    logger.info(f"Corrupted images: {len(results['corrupted_images'])}")
    logger.info(f"Missing images (in CSV but not found): {len(results['missing_images'])}")
    if results['csv_valid'] is not None:
        logger.info(f"CSV valid: {'✓' if results['csv_valid'] else '✗'}")
    logger.info("=" * 80 + "\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CSV GENERATION AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_csv(
    csv_file: Path,
    logger: logging.Logger
) -> bool:
    """
    Validate CSV file format and content.

    Parameters
    ----------
    csv_file : Path
        Path to CSV file
    logger : logging.Logger
        Logger instance

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_file)

        # Check required columns
        if 'id_code' not in df.columns:
            logger.error("CSV missing 'id_code' column")
            return False

        if 'diagnosis' not in df.columns:
            logger.error("CSV missing 'diagnosis' column")
            return False

        # Check diagnosis values
        valid_diagnoses = {0, 1, 2, 3, 4}
        invalid = df[~df['diagnosis'].isin(valid_diagnoses)]
        if len(invalid) > 0:
            logger.error(f"Found {len(invalid)} rows with invalid diagnosis values")
            logger.error(f"Valid values are: {valid_diagnoses}")
            return False

        logger.info(f"✓ CSV validated: {len(df)} rows")
        return True

    except Exception as e:
        logger.error(f"Error validating CSV: {e}")
        return False


def create_messidor_csv(
    img_dir: Path,
    output_csv: Path,
    logger: logging.Logger,
    dry_run: bool = False
) -> bool:
    """
    Create a CSV file for Messidor dataset from image filenames.

    Note: This creates a template CSV. Labels need to be filled in manually
    or imported from Messidor annotation files.

    Parameters
    ----------
    img_dir : Path
        Directory containing Messidor images
    output_csv : Path
        Path to save CSV file
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual file writing

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if dry_run:
        logger.info("[DRY RUN] Would create Messidor CSV")
        return True

    logger.info("Creating Messidor CSV template...")

    try:
        # Get all images
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(img_dir.glob(f'*{ext}'))

        if not image_files:
            logger.error(f"No images found in {img_dir}")
            return False

        # Create dataframe
        data = {
            'id_code': [img.stem for img in image_files],
            'diagnosis': [0] * len(image_files)  # Placeholder
        }
        df = pd.DataFrame(data)

        # Save CSV
        df.to_csv(output_csv, index=False)

        logger.info(f"✓ Created CSV with {len(df)} rows: {output_csv}")
        logger.warning("⚠️  Diagnosis values are placeholders (0). Update with actual labels.")
        return True

    except Exception as e:
        logger.error(f"Error creating Messidor CSV: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_image_quality_metrics(image_path: Path) -> Dict[str, float]:
    """
    Calculate quality metrics for an image.

    Parameters
    ----------
    image_path : Path
        Path to image file

    Returns
    -------
    Dict[str, float]
        Dictionary of quality metrics
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return {}

        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        metrics = {
            'brightness': float(np.mean(gray)),
            'contrast': float(np.std(gray)),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        }

        return metrics

    except Exception:
        return {}


def calculate_dataset_statistics(
    img_dir: Path,
    csv_file: Optional[Path],
    logger: logging.Logger,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a dataset.

    Parameters
    ----------
    img_dir : Path
        Directory containing images
    csv_file : Optional[Path]
        Path to CSV with labels
    logger : logging.Logger
        Logger instance
    sample_size : int
        Number of images to sample for quality metrics

    Returns
    -------
    Dict[str, Any]
        Statistics dictionary
    """
    logger.info("Calculating dataset statistics...")

    stats = {
        'total_images': 0,
        'class_distribution': {},
        'image_dimensions': {
            'widths': [],
            'heights': [],
        },
        'file_sizes': [],
        'quality_metrics': {
            'brightness': [],
            'contrast': [],
            'sharpness': [],
        }
    }

    # Get all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(img_dir.glob(f'*{ext}'))

    stats['total_images'] = len(image_files)

    # Class distribution from CSV
    if csv_file and csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            class_counts = df['diagnosis'].value_counts().to_dict()

            # Convert to class names and sort by class number
            for class_num in sorted(DR_CLASSES.keys()):
                count = class_counts.get(class_num, 0)
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                stats['class_distribution'][class_num] = {
                    'name': DR_CLASSES[class_num],
                    'count': int(count),
                    'percentage': round(percentage, 2)
                }

        except Exception as e:
            logger.warning(f"Could not calculate class distribution: {e}")

    # Sample images for detailed statistics
    sample_images = np.random.choice(
        image_files,
        size=min(sample_size, len(image_files)),
        replace=False
    )

    logger.info(f"Analyzing {len(sample_images)} sample images...")

    for img_path in create_progress_bar(sample_images, desc="Analyzing images", unit="file"):
        try:
            # Dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                stats['image_dimensions']['widths'].append(width)
                stats['image_dimensions']['heights'].append(height)

            # File size
            file_size = img_path.stat().st_size / (1024 * 1024)  # MB
            stats['file_sizes'].append(file_size)

            # Quality metrics
            metrics = calculate_image_quality_metrics(img_path)
            if metrics:
                stats['quality_metrics']['brightness'].append(metrics.get('brightness', 0))
                stats['quality_metrics']['contrast'].append(metrics.get('contrast', 0))
                stats['quality_metrics']['sharpness'].append(metrics.get('sharpness', 0))

        except Exception as e:
            logger.debug(f"Error analyzing {img_path.name}: {e}")
            continue

    # Calculate summary statistics
    if stats['image_dimensions']['widths']:
        stats['dimension_summary'] = {
            'width': {
                'min': int(np.min(stats['image_dimensions']['widths'])),
                'max': int(np.max(stats['image_dimensions']['widths'])),
                'mean': float(np.mean(stats['image_dimensions']['widths'])),
                'std': float(np.std(stats['image_dimensions']['widths'])),
            },
            'height': {
                'min': int(np.min(stats['image_dimensions']['heights'])),
                'max': int(np.max(stats['image_dimensions']['heights'])),
                'mean': float(np.mean(stats['image_dimensions']['heights'])),
                'std': float(np.std(stats['image_dimensions']['heights'])),
            }
        }

    if stats['file_sizes']:
        stats['file_size_summary'] = {
            'min_mb': round(np.min(stats['file_sizes']), 2),
            'max_mb': round(np.max(stats['file_sizes']), 2),
            'mean_mb': round(np.mean(stats['file_sizes']), 2),
            'total_mb': round(sum(stats['file_sizes']) * len(image_files) / len(sample_images), 2),
        }

    if stats['quality_metrics']['brightness']:
        stats['quality_summary'] = {
            'brightness': {
                'mean': round(np.mean(stats['quality_metrics']['brightness']), 2),
                'std': round(np.std(stats['quality_metrics']['brightness']), 2),
            },
            'contrast': {
                'mean': round(np.mean(stats['quality_metrics']['contrast']), 2),
                'std': round(np.std(stats['quality_metrics']['contrast']), 2),
            },
            'sharpness': {
                'mean': round(np.mean(stats['quality_metrics']['sharpness']), 2),
                'std': round(np.std(stats['quality_metrics']['sharpness']), 2),
            }
        }

    # Remove raw lists to reduce size
    del stats['image_dimensions']
    del stats['file_sizes']
    del stats['quality_metrics']

    return stats


def print_statistics(stats: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Print dataset statistics in a formatted table.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary
    logger : logging.Logger
        Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)

    logger.info(f"\n📊 Total Images: {stats['total_images']}")

    # Class distribution
    if stats.get('class_distribution'):
        logger.info("\n📈 Class Distribution:")
        logger.info("-" * 80)
        logger.info(f"{'Class':<5} {'Name':<20} {'Count':<10} {'Percentage':<10}")
        logger.info("-" * 80)

        for class_num, info in stats['class_distribution'].items():
            logger.info(
                f"{class_num:<5} {info['name']:<20} {info['count']:<10} "
                f"{info['percentage']:.2f}%"
            )

    # Dimensions
    if stats.get('dimension_summary'):
        logger.info("\n📐 Image Dimensions:")
        logger.info("-" * 80)
        dim = stats['dimension_summary']
        logger.info(f"Width  - Min: {dim['width']['min']}, Max: {dim['width']['max']}, "
                   f"Mean: {dim['width']['mean']:.1f}, Std: {dim['width']['std']:.1f}")
        logger.info(f"Height - Min: {dim['height']['min']}, Max: {dim['height']['max']}, "
                   f"Mean: {dim['height']['mean']:.1f}, Std: {dim['height']['std']:.1f}")

    # File sizes
    if stats.get('file_size_summary'):
        logger.info("\n💾 File Sizes:")
        logger.info("-" * 80)
        fs = stats['file_size_summary']
        logger.info(f"Min: {fs['min_mb']} MB, Max: {fs['max_mb']} MB, "
                   f"Mean: {fs['mean_mb']} MB")
        logger.info(f"Estimated total size: {fs['total_mb']} MB")

    # Quality metrics
    if stats.get('quality_summary'):
        logger.info("\n✨ Quality Metrics (from sample):")
        logger.info("-" * 80)
        qs = stats['quality_summary']
        logger.info(f"Brightness - Mean: {qs['brightness']['mean']:.2f}, "
                   f"Std: {qs['brightness']['std']:.2f}")
        logger.info(f"Contrast   - Mean: {qs['contrast']['mean']:.2f}, "
                   f"Std: {qs['contrast']['std']:.2f}")
        logger.info(f"Sharpness  - Mean: {qs['sharpness']['mean']:.2f}, "
                   f"Std: {qs['sharpness']['std']:.2f}")

    logger.info("\n" + "=" * 80 + "\n")


def save_statistics(
    stats: Dict[str, Any],
    output_file: Path,
    logger: logging.Logger
) -> None:
    """
    Save statistics to JSON file.

    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary
    output_file : Path
        Path to save JSON file
    logger : logging.Logger
        Logger instance
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"✓ Statistics saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error saving statistics: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATASET CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_sample_dataset(
    source_csv: Path,
    source_img_dir: Path,
    output_csv: Path,
    output_img_dir: Path,
    samples_per_class: int,
    logger: logging.Logger,
    dry_run: bool = False,
    seed: int = 42
) -> bool:
    """
    Create a balanced sample dataset with specified number of images per class.

    Parameters
    ----------
    source_csv : Path
        Source CSV file
    source_img_dir : Path
        Source image directory
    output_csv : Path
        Output CSV file
    output_img_dir : Path
        Output image directory
    samples_per_class : int
        Number of samples per class
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual file operations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would create sample dataset with {samples_per_class} samples per class")
        return True

    logger.info(f"Creating sample dataset ({samples_per_class} samples per class)...")

    try:
        # Read source CSV
        df = pd.read_csv(source_csv)

        # Set random seed
        np.random.seed(seed)

        # Sample from each class
        sampled_dfs = []
        for class_num in range(5):
            class_df = df[df['diagnosis'] == class_num]

            if len(class_df) < samples_per_class:
                logger.warning(
                    f"Class {class_num} has only {len(class_df)} samples "
                    f"(requested {samples_per_class}). Using all available."
                )
                sampled_dfs.append(class_df)
            else:
                sampled = class_df.sample(n=samples_per_class, random_state=seed)
                sampled_dfs.append(sampled)

        # Combine samples
        sample_df = pd.concat(sampled_dfs, ignore_index=True)

        # Shuffle
        sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        logger.info(f"Selected {len(sample_df)} samples")

        # Create output directory
        output_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        logger.info("Copying sample images...")
        copied = 0
        for _, row in create_progress_bar(
            sample_df.iterrows(),
            total=len(sample_df),
            desc="Copying images",
            unit="file"
        ):
            id_code = row['id_code']

            # Find source image (try different extensions)
            src_img = None
            for ext in IMAGE_EXTENSIONS:
                candidate = source_img_dir / f"{id_code}{ext}"
                if candidate.exists():
                    src_img = candidate
                    break

            if src_img is None:
                logger.warning(f"Image not found: {id_code}")
                continue

            # Copy to output
            dst_img = output_img_dir / src_img.name
            shutil.copy2(src_img, dst_img)
            copied += 1

        logger.info(f"Copied {copied} images")

        # Save CSV
        sample_df.to_csv(output_csv, index=False)
        logger.info(f"✓ Sample CSV saved: {output_csv}")

        # Print distribution
        logger.info("\nSample distribution:")
        for class_num in range(5):
            count = len(sample_df[sample_df['diagnosis'] == class_num])
            logger.info(f"  Class {class_num} ({DR_CLASSES[class_num]}): {count}")

        return True

    except Exception as e:
        logger.error(f"Error creating sample dataset: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def create_train_test_split(
    source_csv: Path,
    source_img_dir: Path,
    output_dir: Path,
    split_ratio: float,
    logger: logging.Logger,
    dry_run: bool = False,
    seed: int = 42
) -> bool:
    """
    Create train/test split with stratification by class.

    Parameters
    ----------
    source_csv : Path
        Source CSV file
    source_img_dir : Path
        Source image directory
    output_dir : Path
        Output directory for split datasets
    split_ratio : float
        Training set ratio (e.g., 0.8 for 80/20 split)
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, skip actual file operations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would create {split_ratio:.0%}/{(1-split_ratio):.0%} train/test split")
        return True

    logger.info(f"Creating {split_ratio:.0%}/{(1-split_ratio):.0%} train/test split...")

    try:
        from sklearn.model_selection import train_test_split

        # Read source CSV
        df = pd.read_csv(source_csv)

        # Stratified split
        train_df, test_df = train_test_split(
            df,
            train_size=split_ratio,
            stratify=df['diagnosis'],
            random_state=seed
        )

        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")

        # Create directories
        train_img_dir = output_dir / 'train_images'
        test_img_dir = output_dir / 'test_images'
        train_img_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy train images
        logger.info("Copying training images...")
        for _, row in create_progress_bar(
            train_df.iterrows(),
            total=len(train_df),
            desc="Copying train images",
            unit="file"
        ):
            id_code = row['id_code']
            for ext in IMAGE_EXTENSIONS:
                src_img = source_img_dir / f"{id_code}{ext}"
                if src_img.exists():
                    dst_img = train_img_dir / src_img.name
                    shutil.copy2(src_img, dst_img)
                    break

        # Copy test images
        logger.info("Copying test images...")
        for _, row in create_progress_bar(
            test_df.iterrows(),
            total=len(test_df),
            desc="Copying test images",
            unit="file"
        ):
            id_code = row['id_code']
            for ext in IMAGE_EXTENSIONS:
                src_img = source_img_dir / f"{id_code}{ext}"
                if src_img.exists():
                    dst_img = test_img_dir / src_img.name
                    shutil.copy2(src_img, dst_img)
                    break

        # Save CSVs
        train_csv = output_dir / 'train.csv'
        test_csv = output_dir / 'test.csv'
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        logger.info(f"✓ Train CSV saved: {train_csv}")
        logger.info(f"✓ Test CSV saved: {test_csv}")

        # Print distributions
        logger.info("\nClass distribution:")
        logger.info(f"{'Class':<6} {'Train':<10} {'Test':<10}")
        logger.info("-" * 30)
        for class_num in range(5):
            train_count = len(train_df[train_df['diagnosis'] == class_num])
            test_count = len(test_df[test_df['diagnosis'] == class_num])
            logger.info(f"{class_num:<6} {train_count:<10} {test_count:<10}")

        return True

    except Exception as e:
        logger.error(f"Error creating train/test split: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION AND CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Prepare diabetic retinopathy datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and prepare APTOS dataset
  python scripts/prepare_data.py --aptos-only

  # Download APTOS and create sample dataset
  python scripts/prepare_data.py --aptos-only --create-sample

  # Organize manually downloaded Messidor dataset
  python scripts/prepare_data.py --messidor-organize

  # Verify datasets only
  python scripts/prepare_data.py --verify-only

  # Calculate statistics only
  python scripts/prepare_data.py --stats-only

  # Full pipeline with all datasets
  python scripts/prepare_data.py --datasets aptos messidor --create-sample --create-split
        """
    )

    # Dataset selection
    dataset_group = parser.add_argument_group('Dataset Selection')
    dataset_group.add_argument(
        '--datasets',
        nargs='+',
        choices=['aptos', 'messidor'],
        help='Datasets to download and prepare'
    )
    dataset_group.add_argument(
        '--aptos-only',
        action='store_true',
        help='Only prepare APTOS dataset'
    )
    dataset_group.add_argument(
        '--messidor-organize',
        action='store_true',
        help='Organize manually downloaded Messidor dataset'
    )

    # Operations
    ops_group = parser.add_argument_group('Operations')
    ops_group.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets (no download/organization)'
    )
    ops_group.add_argument(
        '--stats-only',
        action='store_true',
        help='Only calculate and display statistics'
    )
    ops_group.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample dataset (50 images per class by default)'
    )
    ops_group.add_argument(
        '--samples-per-class',
        type=int,
        default=50,
        help='Number of samples per class for sample dataset (default: 50)'
    )
    ops_group.add_argument(
        '--create-split',
        action='store_true',
        help='Create train/test split from full dataset'
    )
    ops_group.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Train/test split ratio (default: 0.8)'
    )

    # Paths
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Root data directory (default: data/)'
    )
    path_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as data-dir)'
    )

    # Options
    opt_group = parser.add_argument_group('Options')
    opt_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (no actual file operations)'
    )
    opt_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    opt_group.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    opt_group.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Log directory (default: logs/)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Set seed
    set_seed(args.seed, deterministic=True, verbose=False)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    log_dir = Path(args.log_dir)

    # Setup logging
    logger = setup_logging(log_dir, verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("DIABETIC RETINOPATHY DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {args.seed}")
    if args.dry_run:
        logger.info("⚠️  DRY RUN MODE - No file operations will be performed")
    logger.info("")

    # Determine which datasets to process
    datasets_to_process = []
    if args.aptos_only:
        datasets_to_process = ['aptos']
    elif args.datasets:
        datasets_to_process = args.datasets
    elif args.messidor_organize:
        datasets_to_process = ['messidor']
    elif not (args.verify_only or args.stats_only):
        # If no dataset specified and not verification/stats only, show help
        logger.error("No datasets specified. Use --datasets, --aptos-only, or --messidor-organize")
        logger.info("Run with --help for usage information")
        return 1

    # ═══════════════════════════════════════════════════════════════════════════
    # DOWNLOAD AND ORGANIZE
    # ═══════════════════════════════════════════════════════════════════════════

    if not (args.verify_only or args.stats_only):
        # Process APTOS
        if 'aptos' in datasets_to_process:
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING APTOS DATASET")
            logger.info("=" * 80 + "\n")

            aptos_dir = data_dir / 'aptos'
            aptos_raw_dir = aptos_dir / 'raw'
            aptos_downloads_dir = data_dir / 'downloads'

            # Download
            success = download_aptos_dataset(data_dir, logger, args.dry_run)

            if success:
                # Extract
                success = extract_zip_files(
                    aptos_downloads_dir,
                    aptos_raw_dir,
                    logger,
                    args.dry_run
                )

                if success:
                    # Organize
                    success = organize_aptos_dataset(
                        aptos_raw_dir,
                        aptos_dir,
                        logger,
                        args.dry_run
                    )

        # Process Messidor
        if 'messidor' in datasets_to_process or args.messidor_organize:
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING MESSIDOR DATASET")
            logger.info("=" * 80 + "\n")

            messidor_dir = data_dir / 'messidor'
            messidor_raw_dir = messidor_dir / 'raw'

            if not args.messidor_organize:
                download_messidor_instructions(logger)
            else:
                success = organize_messidor_dataset(
                    messidor_raw_dir,
                    messidor_dir,
                    logger,
                    args.dry_run
                )

                if success:
                    # Create CSV template
                    img_dir = messidor_dir / 'images'
                    csv_file = messidor_dir / 'annotations.csv'
                    create_messidor_csv(img_dir, csv_file, logger, args.dry_run)

    # ═══════════════════════════════════════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════════════════════════════════════

    if args.verify_only or (datasets_to_process and not args.stats_only):
        logger.info("\n" + "=" * 80)
        logger.info("DATASET VERIFICATION")
        logger.info("=" * 80 + "\n")

        # Verify APTOS
        if args.verify_only or 'aptos' in datasets_to_process:
            aptos_dir = data_dir / 'aptos'

            # Verify train set
            if (aptos_dir / 'train_images').exists():
                logger.info("Verifying APTOS training set...")
                verify_dataset(
                    aptos_dir / 'train_images',
                    aptos_dir / 'train.csv',
                    logger,
                    expected_count=DATASET_INFO['aptos']['train_count']
                )

            # Verify test set
            if (aptos_dir / 'test_images').exists():
                logger.info("Verifying APTOS test set...")
                verify_dataset(
                    aptos_dir / 'test_images',
                    aptos_dir / 'test.csv',
                    logger,
                    expected_count=DATASET_INFO['aptos']['test_count']
                )

        # Verify Messidor
        if args.verify_only or 'messidor' in datasets_to_process:
            messidor_dir = data_dir / 'messidor'

            if (messidor_dir / 'images').exists():
                logger.info("Verifying Messidor dataset...")
                verify_dataset(
                    messidor_dir / 'images',
                    messidor_dir / 'annotations.csv' if (messidor_dir / 'annotations.csv').exists() else None,
                    logger
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════

    if args.stats_only or (datasets_to_process and not args.verify_only):
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATING STATISTICS")
        logger.info("=" * 80 + "\n")

        # APTOS statistics
        if args.stats_only or 'aptos' in datasets_to_process:
            aptos_dir = data_dir / 'aptos'

            if (aptos_dir / 'train_images').exists():
                logger.info("APTOS Training Set Statistics")
                logger.info("-" * 80)

                stats = calculate_dataset_statistics(
                    aptos_dir / 'train_images',
                    aptos_dir / 'train.csv',
                    logger
                )

                print_statistics(stats, logger)

                stats_file = output_dir / 'aptos' / 'train_statistics.json'
                save_statistics(stats, stats_file, logger)

        # Messidor statistics
        if args.stats_only or 'messidor' in datasets_to_process:
            messidor_dir = data_dir / 'messidor'

            if (messidor_dir / 'images').exists():
                logger.info("Messidor Dataset Statistics")
                logger.info("-" * 80)

                stats = calculate_dataset_statistics(
                    messidor_dir / 'images',
                    messidor_dir / 'annotations.csv' if (messidor_dir / 'annotations.csv').exists() else None,
                    logger
                )

                print_statistics(stats, logger)

                stats_file = output_dir / 'messidor' / 'statistics.json'
                save_statistics(stats, stats_file, logger)

    # ═══════════════════════════════════════════════════════════════════════════
    # SAMPLE DATASET CREATION
    # ═══════════════════════════════════════════════════════════════════════════

    if args.create_sample and not args.stats_only:
        logger.info("\n" + "=" * 80)
        logger.info("CREATING SAMPLE DATASET")
        logger.info("=" * 80 + "\n")

        # Create sample from APTOS if available
        aptos_dir = data_dir / 'aptos'
        sample_dir = data_dir / 'sample'

        if (aptos_dir / 'train.csv').exists() and (aptos_dir / 'train_images').exists():
            create_sample_dataset(
                source_csv=aptos_dir / 'train.csv',
                source_img_dir=aptos_dir / 'train_images',
                output_csv=sample_dir / 'sample.csv',
                output_img_dir=sample_dir / 'images',
                samples_per_class=args.samples_per_class,
                logger=logger,
                dry_run=args.dry_run,
                seed=args.seed
            )
        else:
            logger.warning("Cannot create sample: APTOS dataset not found")

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAIN/TEST SPLIT
    # ═══════════════════════════════════════════════════════════════════════════

    if args.create_split and not args.stats_only:
        logger.info("\n" + "=" * 80)
        logger.info("CREATING TRAIN/TEST SPLIT")
        logger.info("=" * 80 + "\n")

        # Create split for datasets that don't have official splits
        messidor_dir = data_dir / 'messidor'

        if (messidor_dir / 'annotations.csv').exists() and (messidor_dir / 'images').exists():
            logger.info("Creating train/test split for Messidor...")
            split_dir = messidor_dir / 'split'
            create_train_test_split(
                source_csv=messidor_dir / 'annotations.csv',
                source_img_dir=messidor_dir / 'images',
                output_dir=split_dir,
                split_ratio=args.split_ratio,
                logger=logger,
                dry_run=args.dry_run,
                seed=args.seed
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPLETION
    # ═══════════════════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 80)
    logger.info("✓ DATA PREPARATION COMPLETED")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("\n⚠️  This was a dry run. No actual file operations were performed.")

    logger.info("\nNext steps:")
    logger.info("1. Verify the organized datasets in data/")
    logger.info("2. Check CSV files have correct format (id_code, diagnosis)")
    logger.info("3. Review statistics and class distribution")
    logger.info("4. Use sample dataset for quick testing")
    logger.info("5. Begin training with: python scripts/train_baseline.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
