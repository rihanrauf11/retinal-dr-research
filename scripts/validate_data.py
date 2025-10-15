#!/usr/bin/env python3
"""
Comprehensive Data Validation Script for Diabetic Retinopathy Classification

This script performs thorough validation of all datasets including:
- Image file existence and integrity checks
- CSV format and value validation
- Duplicate detection
- Image dimension and quality statistics
- Cross-dataset comparisons

Generates detailed reports in:
- data/VALIDATION_REPORT.md (comprehensive markdown report)
- data/issues.txt (critical issues if found)

Usage:
    # Validate all datasets
    python scripts/validate_data.py

    # Validate specific dataset
    python scripts/validate_data.py --dataset aptos-train

    # Quick mode (skip duplicate detection)
    python scripts/validate_data.py --quick

    # Verbose output
    python scripts/validate_data.py --verbose

Author: Generated with Claude Code
"""

import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
VALID_DIAGNOSIS_VALUES = {0, 1, 2, 3, 4}

DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}

# Dataset configurations
DATASETS = {
    'aptos-train': {
        'name': 'APTOS Training Set',
        'csv': 'data/aptos/train.csv',
        'img_dir': 'data/aptos/train_images',
        'has_labels': True
    },
    'aptos-train-split': {
        'name': 'APTOS Training Split',
        'csv': 'data/aptos/train_split.csv',
        'img_dir': 'data/aptos/train_images',
        'has_labels': True
    },
    'aptos-val-split': {
        'name': 'APTOS Validation Split',
        'csv': 'data/aptos/val_split.csv',
        'img_dir': 'data/aptos/train_images',
        'has_labels': True
    },
    'aptos-test': {
        'name': 'APTOS Test Set',
        'csv': 'data/aptos/test.csv',
        'img_dir': 'data/aptos/test_images',
        'has_labels': False
    },
    'messidor': {
        'name': 'Messidor Test Set',
        'csv': 'data/messidor/test.csv',
        'img_dir': 'data/messidor/images',
        'has_labels': True
    },
    'sample-train': {
        'name': 'Sample Training Set',
        'csv': 'data/sample/train.csv',
        'img_dir': 'data/sample/images',
        'has_labels': True
    },
    'sample-val': {
        'name': 'Sample Validation Set',
        'csv': 'data/sample/val.csv',
        'img_dir': 'data/sample/images',
        'has_labels': True
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationResult:
    """Container for validation results."""

    def __init__(self, dataset_key: str, dataset_name: str):
        self.dataset_key = dataset_key
        self.dataset_name = dataset_name
        self.timestamp = datetime.now()

        # CSV validation
        self.csv_exists = False
        self.csv_valid = False
        self.csv_row_count = 0
        self.csv_errors = []

        # Image validation
        self.img_dir_exists = False
        self.total_images_on_disk = 0
        self.images_in_csv = 0
        self.missing_images = []  # In CSV but not on disk
        self.extra_images = []  # On disk but not in CSV
        self.corrupted_images = []  # Cannot be loaded
        self.valid_images = 0

        # Dimension statistics
        self.dimensions = {
            'widths': [],
            'heights': [],
            'unique_dimensions': set()
        }

        # Duplicate detection
        self.duplicates = []  # List of (hash, [file1, file2, ...])
        self.duplicate_count = 0

        # Label validation
        self.has_labels = False
        self.label_distribution = Counter()
        self.invalid_labels = []
        self.null_labels = 0

        # Quality metrics
        self.file_sizes = []  # In bytes
        self.formats = Counter()

        # Status
        self.passed = False
        self.warnings = []
        self.errors = []

    def add_error(self, message: str):
        """Add error message."""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add warning message."""
        self.warnings.append(message)

    def compute_final_status(self):
        """Compute final pass/fail status."""
        # Critical failures
        if not self.csv_exists or not self.img_dir_exists:
            self.passed = False
            return

        if not self.csv_valid:
            self.passed = False
            return

        if len(self.corrupted_images) > 0:
            self.passed = False
            return

        if len(self.missing_images) > 0:
            self.passed = False
            return

        if len(self.errors) > 0:
            self.passed = False
            return

        # If has labels, check for invalid values
        if self.has_labels and len(self.invalid_labels) > 0:
            self.passed = False
            return

        # All checks passed
        self.passed = True


# ═══════════════════════════════════════════════════════════════════════════════
# CSV VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_csv(csv_path: Path, has_labels: bool) -> Dict[str, Any]:
    """
    Validate CSV file format and content.

    Args:
        csv_path: Path to CSV file
        has_labels: Whether the CSV should have a 'diagnosis' column

    Returns:
        Dictionary with validation results
    """
    result = {
        'exists': False,
        'valid': False,
        'row_count': 0,
        'errors': [],
        'has_id_code': False,
        'has_diagnosis': False,
        'null_count': 0,
        'invalid_diagnoses': []
    }

    if not csv_path.exists():
        result['errors'].append(f"CSV file not found: {csv_path}")
        return result

    result['exists'] = True

    try:
        df = pd.read_csv(csv_path)
        result['row_count'] = len(df)

        # Check required columns
        if 'id_code' not in df.columns:
            result['errors'].append("Missing required column: 'id_code'")
            return result

        result['has_id_code'] = True

        # Check diagnosis column
        if has_labels:
            if 'diagnosis' not in df.columns:
                result['errors'].append("Missing required column: 'diagnosis'")
                return result

            result['has_diagnosis'] = True

            # Check for null values
            null_count = df['diagnosis'].isnull().sum()
            result['null_count'] = null_count

            if null_count > 0:
                result['errors'].append(f"Found {null_count} null diagnosis values")

            # Check for invalid diagnosis values
            valid_df = df[df['diagnosis'].notnull()]
            invalid = valid_df[~valid_df['diagnosis'].isin(VALID_DIAGNOSIS_VALUES)]

            if len(invalid) > 0:
                result['invalid_diagnoses'] = invalid['diagnosis'].unique().tolist()
                result['errors'].append(
                    f"Found {len(invalid)} rows with invalid diagnosis values: "
                    f"{result['invalid_diagnoses']}"
                )

        # Check for duplicate id_codes
        duplicate_ids = df[df.duplicated(subset=['id_code'], keep=False)]['id_code'].unique()
        if len(duplicate_ids) > 0:
            result['errors'].append(f"Found {len(duplicate_ids)} duplicate id_codes")

        # If no errors, mark as valid
        if len(result['errors']) == 0:
            result['valid'] = True

    except Exception as e:
        result['errors'].append(f"Error reading CSV: {str(e)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_image(img_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Verify that an image can be loaded and is not corrupted.

    Args:
        img_path: Path to image file

    Returns:
        (is_valid, error_message, dimensions)
    """
    try:
        with Image.open(img_path) as img:
            img.verify()

        # Reopen to get dimensions (verify closes the file)
        with Image.open(img_path) as img:
            dimensions = img.size  # (width, height)
            img.load()  # Force load to catch truncation errors

        return True, None, dimensions

    except Exception as e:
        return False, str(e), None


def compute_image_hash(img_path: Path) -> Optional[str]:
    """
    Compute MD5 hash of image file for duplicate detection.

    Args:
        img_path: Path to image file

    Returns:
        MD5 hash string or None if error
    """
    try:
        with open(img_path, 'rb') as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    except Exception:
        return None


def find_image_file(id_code: str, img_dir: Path) -> Optional[Path]:
    """
    Find image file with given id_code in directory.

    Tries multiple extensions.

    Args:
        id_code: Image identifier
        img_dir: Directory to search

    Returns:
        Path to image file or None if not found
    """
    for ext in IMAGE_EXTENSIONS:
        img_path = img_dir / f"{id_code}{ext}"
        if img_path.exists():
            return img_path
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE DATASET VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_dataset(
    dataset_key: str,
    config: Dict[str, Any],
    check_duplicates: bool = True,
    verbose: bool = False
) -> ValidationResult:
    """
    Perform comprehensive validation of a dataset.

    Args:
        dataset_key: Dataset identifier
        config: Dataset configuration
        check_duplicates: Whether to check for duplicate images
        verbose: Print verbose output

    Returns:
        ValidationResult object
    """
    result = ValidationResult(dataset_key, config['name'])
    result.has_labels = config['has_labels']

    csv_path = Path(config['csv'])
    img_dir = Path(config['img_dir'])

    if verbose:
        print(f"\n{'='*80}")
        print(f"Validating: {config['name']}")
        print(f"{'='*80}")

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. CSV Validation
    # ─────────────────────────────────────────────────────────────────────────────

    if verbose:
        print("\n[1/6] Validating CSV file...")

    csv_result = validate_csv(csv_path, config['has_labels'])

    result.csv_exists = csv_result['exists']
    result.csv_valid = csv_result['valid']
    result.csv_row_count = csv_result['row_count']
    result.csv_errors = csv_result['errors']

    if not result.csv_exists:
        result.add_error(f"CSV file not found: {csv_path}")
        result.compute_final_status()
        return result

    if not result.csv_valid:
        for error in csv_result['errors']:
            result.add_error(f"CSV validation: {error}")
        result.compute_final_status()
        return result

    # Load CSV for further checks
    df = pd.read_csv(csv_path)
    result.images_in_csv = len(df)

    # Extract label distribution
    if config['has_labels'] and 'diagnosis' in df.columns:
        result.label_distribution = Counter(df['diagnosis'].dropna().astype(int))
        result.null_labels = csv_result['null_count']
        result.invalid_labels = csv_result['invalid_diagnoses']

    if verbose:
        print(f"  ✓ CSV valid: {result.csv_row_count} rows")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Image Directory Check
    # ─────────────────────────────────────────────────────────────────────────────

    if verbose:
        print("\n[2/6] Checking image directory...")

    if not img_dir.exists():
        result.add_error(f"Image directory not found: {img_dir}")
        result.compute_final_status()
        return result

    result.img_dir_exists = True

    # Count images on disk
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(list(img_dir.glob(f"*{ext}")))

    result.total_images_on_disk = len(all_images)

    if verbose:
        print(f"  ✓ Found {result.total_images_on_disk} images on disk")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Image Existence Check (CSV vs Disk)
    # ─────────────────────────────────────────────────────────────────────────────

    if verbose:
        print("\n[3/6] Checking image existence (CSV vs disk)...")

    # Get all id_codes from disk (without extension)
    images_on_disk = set()
    for img_path in all_images:
        # Remove all possible extensions
        stem = img_path.stem
        images_on_disk.add(stem)

    # Get all id_codes from CSV
    images_in_csv = set(df['id_code'].astype(str))

    # Find missing images (in CSV but not on disk)
    result.missing_images = sorted(list(images_in_csv - images_on_disk))

    # Find extra images (on disk but not in CSV)
    result.extra_images = sorted(list(images_on_disk - images_in_csv))

    if len(result.missing_images) > 0:
        result.add_error(f"Found {len(result.missing_images)} images in CSV but not on disk")

    if len(result.extra_images) > 0:
        result.add_warning(f"Found {len(result.extra_images)} images on disk but not in CSV")

    if verbose:
        if len(result.missing_images) == 0:
            print(f"  ✓ All CSV images found on disk")
        else:
            print(f"  ✗ {len(result.missing_images)} missing images")

        if len(result.extra_images) > 0:
            print(f"  ! {len(result.extra_images)} extra images on disk")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Image Integrity Check (Corruption, Dimensions)
    # ─────────────────────────────────────────────────────────────────────────────

    if verbose:
        print("\n[4/6] Validating image integrity...")

    # Only check images that are in the CSV
    images_to_check = []
    for id_code in df['id_code']:
        img_path = find_image_file(str(id_code), img_dir)
        if img_path:
            images_to_check.append(img_path)

    # Validate images in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(verify_image, img_path): img_path
                   for img_path in images_to_check}

        progress = tqdm(as_completed(futures), total=len(futures),
                       desc="  Validating images", disable=not verbose)

        for future in progress:
            img_path = futures[future]
            is_valid, error, dimensions = future.result()

            if is_valid:
                result.valid_images += 1

                # Record dimensions
                if dimensions:
                    width, height = dimensions
                    result.dimensions['widths'].append(width)
                    result.dimensions['heights'].append(height)
                    result.dimensions['unique_dimensions'].add(dimensions)

                # Record file size
                result.file_sizes.append(img_path.stat().st_size)

                # Record format
                result.formats[img_path.suffix.lower()] += 1

            else:
                result.corrupted_images.append({
                    'path': str(img_path),
                    'id_code': img_path.stem,
                    'error': error
                })

    if len(result.corrupted_images) > 0:
        result.add_error(f"Found {len(result.corrupted_images)} corrupted images")

    if verbose:
        if len(result.corrupted_images) == 0:
            print(f"  ✓ All {result.valid_images} images are valid")
        else:
            print(f"  ✗ {len(result.corrupted_images)} corrupted images")

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Duplicate Detection
    # ─────────────────────────────────────────────────────────────────────────────

    if check_duplicates and len(images_to_check) > 0:
        if verbose:
            print("\n[5/6] Checking for duplicate images...")

        hash_to_files = defaultdict(list)

        # Compute hashes in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(compute_image_hash, img_path): img_path
                       for img_path in images_to_check}

            progress = tqdm(as_completed(futures), total=len(futures),
                           desc="  Computing hashes", disable=not verbose)

            for future in progress:
                img_path = futures[future]
                img_hash = future.result()

                if img_hash:
                    hash_to_files[img_hash].append(img_path)

        # Find duplicates
        for img_hash, files in hash_to_files.items():
            if len(files) > 1:
                result.duplicates.append((img_hash, [str(f) for f in files]))
                result.duplicate_count += len(files) - 1  # Don't count original

        if len(result.duplicates) > 0:
            result.add_warning(
                f"Found {len(result.duplicates)} sets of duplicate images "
                f"({result.duplicate_count} duplicates total)"
            )

        if verbose:
            if len(result.duplicates) == 0:
                print(f"  ✓ No duplicate images found")
            else:
                print(f"  ! {len(result.duplicates)} duplicate sets found")

    elif not check_duplicates:
        if verbose:
            print("\n[5/6] Skipping duplicate detection (quick mode)")

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Final Status Computation
    # ─────────────────────────────────────────────────────────────────────────────

    result.compute_final_status()

    if verbose:
        print("\n[6/6] Computing final validation status...")
        status_icon = "✓" if result.passed else "✗"
        status_text = "PASSED" if result.passed else "FAILED"
        print(f"  {status_icon} Validation {status_text}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(
    results: Dict[str, ValidationResult],
    output_path: Path,
    quick_mode: bool
):
    """
    Generate comprehensive markdown validation report.

    Args:
        results: Dictionary of validation results
        output_path: Path to output markdown file
        quick_mode: Whether quick mode was used
    """
    lines = []

    # Header
    lines.append("# Data Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Quick Mode:** {'Yes (duplicate detection skipped)' if quick_mode else 'No'}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    total_datasets = len(results)
    passed_datasets = sum(1 for r in results.values() if r.passed)
    failed_datasets = total_datasets - passed_datasets

    lines.append(f"- **Total Datasets Validated:** {total_datasets}")
    lines.append(f"- **Passed:** {passed_datasets} ✓")
    lines.append(f"- **Failed:** {failed_datasets} ✗")
    lines.append("")

    # Critical issues
    total_errors = sum(len(r.errors) for r in results.values())
    total_warnings = sum(len(r.warnings) for r in results.values())
    total_corrupted = sum(len(r.corrupted_images) for r in results.values())
    total_missing = sum(len(r.missing_images) for r in results.values())

    lines.append("### Critical Issues")
    lines.append("")
    lines.append(f"- **Total Errors:** {total_errors}")
    lines.append(f"- **Total Warnings:** {total_warnings}")
    lines.append(f"- **Corrupted Images:** {total_corrupted}")
    lines.append(f"- **Missing Images:** {total_missing}")
    lines.append("")

    # Per-dataset results
    lines.append("## Dataset Validation Results")
    lines.append("")

    for dataset_key, result in results.items():
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        lines.append(f"### {result.dataset_name} [{status}]")
        lines.append("")

        # Basic statistics
        lines.append("#### Basic Statistics")
        lines.append("")
        lines.append(f"- **CSV Rows:** {result.csv_row_count}")
        lines.append(f"- **Images in CSV:** {result.images_in_csv}")
        lines.append(f"- **Images on Disk:** {result.total_images_on_disk}")
        lines.append(f"- **Valid Images:** {result.valid_images}")
        lines.append(f"- **Corrupted Images:** {len(result.corrupted_images)}")
        lines.append(f"- **Missing Images:** {len(result.missing_images)}")
        lines.append(f"- **Extra Images:** {len(result.extra_images)}")
        lines.append("")

        # Image dimensions
        if result.dimensions['widths']:
            widths = result.dimensions['widths']
            heights = result.dimensions['heights']

            lines.append("#### Image Dimensions")
            lines.append("")
            lines.append(f"- **Width:** min={min(widths)}, max={max(widths)}, "
                        f"mean={int(np.mean(widths))}, std={int(np.std(widths))}")
            lines.append(f"- **Height:** min={min(heights)}, max={max(heights)}, "
                        f"mean={int(np.mean(heights))}, std={int(np.std(heights))}")
            lines.append(f"- **Unique Dimensions:** {len(result.dimensions['unique_dimensions'])}")
            lines.append("")

        # File sizes
        if result.file_sizes:
            sizes_mb = [s / (1024 * 1024) for s in result.file_sizes]
            lines.append("#### File Sizes")
            lines.append("")
            lines.append(f"- **Min:** {min(sizes_mb):.2f} MB")
            lines.append(f"- **Max:** {max(sizes_mb):.2f} MB")
            lines.append(f"- **Mean:** {np.mean(sizes_mb):.2f} MB")
            lines.append(f"- **Total:** {sum(sizes_mb):.2f} MB")
            lines.append("")

        # Image formats
        if result.formats:
            lines.append("#### Image Formats")
            lines.append("")
            for fmt, count in sorted(result.formats.items()):
                pct = (count / result.valid_images * 100) if result.valid_images > 0 else 0
                lines.append(f"- **{fmt}:** {count} ({pct:.1f}%)")
            lines.append("")

        # Label distribution
        if result.has_labels and result.label_distribution:
            lines.append("#### Class Distribution")
            lines.append("")
            lines.append("| Class | Name | Count | Percentage |")
            lines.append("|-------|------|-------|------------|")

            total = sum(result.label_distribution.values())
            for cls in sorted(result.label_distribution.keys()):
                count = result.label_distribution[cls]
                pct = (count / total * 100) if total > 0 else 0
                name = DR_CLASSES.get(cls, f"Unknown ({cls})")
                lines.append(f"| {cls} | {name} | {count} | {pct:.2f}% |")

            if result.null_labels > 0:
                lines.append(f"| - | NULL | {result.null_labels} | - |")

            lines.append("")

        # Duplicates
        if not quick_mode and result.duplicates:
            lines.append("#### Duplicate Images")
            lines.append("")
            lines.append(f"Found {len(result.duplicates)} sets of duplicate images:")
            lines.append("")

            for i, (img_hash, files) in enumerate(result.duplicates[:10], 1):
                lines.append(f"{i}. Hash: `{img_hash[:16]}...`")
                for f in files:
                    lines.append(f"   - `{f}`")
                lines.append("")

            if len(result.duplicates) > 10:
                lines.append(f"... and {len(result.duplicates) - 10} more duplicate sets")
                lines.append("")

        # Errors
        if result.errors:
            lines.append("#### Errors")
            lines.append("")
            for error in result.errors:
                lines.append(f"- ❌ {error}")
            lines.append("")

        # Warnings
        if result.warnings:
            lines.append("#### Warnings")
            lines.append("")
            for warning in result.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        # Corrupted images detail
        if result.corrupted_images:
            lines.append("#### Corrupted Images Detail")
            lines.append("")
            for img in result.corrupted_images[:20]:
                lines.append(f"- **{img['id_code']}**: {img['error']}")
            lines.append("")

            if len(result.corrupted_images) > 20:
                lines.append(f"... and {len(result.corrupted_images) - 20} more corrupted images")
                lines.append("")

        # Missing images detail
        if result.missing_images:
            lines.append("#### Missing Images Detail")
            lines.append("")
            lines.append(f"Images in CSV but not found on disk ({len(result.missing_images)} total):")
            lines.append("")
            for id_code in result.missing_images[:50]:
                lines.append(f"- `{id_code}`")
            lines.append("")

            if len(result.missing_images) > 50:
                lines.append(f"... and {len(result.missing_images) - 50} more missing images")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Cross-dataset comparison
    lines.append("## Cross-Dataset Comparison")
    lines.append("")

    lines.append("| Dataset | CSV Rows | Images | Valid | Corrupted | Missing |")
    lines.append("|---------|----------|--------|-------|-----------|---------|")

    for dataset_key, result in results.items():
        lines.append(
            f"| {result.dataset_name} | {result.csv_row_count} | "
            f"{result.total_images_on_disk} | {result.valid_images} | "
            f"{len(result.corrupted_images)} | {len(result.missing_images)} |"
        )

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    has_issues = any(not r.passed for r in results.values())

    if not has_issues:
        lines.append("✅ **All datasets passed validation!**")
        lines.append("")
        lines.append("Your data is ready for training. No issues detected.")
        lines.append("")
    else:
        lines.append("⚠️ **Issues detected that require attention:**")
        lines.append("")

        if total_corrupted > 0:
            lines.append(f"1. **Fix {total_corrupted} corrupted images:**")
            lines.append("   - These images cannot be loaded by PIL")
            lines.append("   - Either repair or remove them from the dataset")
            lines.append("   - See `data/issues.txt` for detailed list")
            lines.append("")

        if total_missing > 0:
            lines.append(f"2. **Resolve {total_missing} missing images:**")
            lines.append("   - These images are referenced in CSV but not found on disk")
            lines.append("   - Either add the images or remove entries from CSV")
            lines.append("   - See `data/issues.txt` for detailed list")
            lines.append("")

        if total_errors > 0:
            lines.append(f"3. **Address {total_errors} validation errors:**")
            lines.append("   - Review error messages in dataset sections above")
            lines.append("   - These are critical issues that prevent training")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/validate_data.py`*")
    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def generate_issues_file(
    results: Dict[str, ValidationResult],
    output_path: Path
):
    """
    Generate plain text file with critical issues only.

    Args:
        results: Dictionary of validation results
        output_path: Path to output issues file
    """
    lines = []

    lines.append("=" * 80)
    lines.append("CRITICAL DATA VALIDATION ISSUES")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    has_issues = False

    for dataset_key, result in results.items():
        if result.passed:
            continue

        has_issues = True

        lines.append("-" * 80)
        lines.append(f"Dataset: {result.dataset_name}")
        lines.append("-" * 80)
        lines.append("")

        # Errors
        if result.errors:
            lines.append("[CRITICAL ERRORS]")
            for error in result.errors:
                lines.append(f"  ✗ {error}")
            lines.append("")

        # Corrupted images
        if result.corrupted_images:
            lines.append(f"[CORRUPTED IMAGES] ({len(result.corrupted_images)} total)")
            for img in result.corrupted_images:
                lines.append(f"  - {img['id_code']}: {img['error']}")
            lines.append("")

        # Missing images
        if result.missing_images:
            lines.append(f"[MISSING IMAGES] ({len(result.missing_images)} total)")
            lines.append("  Images in CSV but not found on disk:")
            for id_code in result.missing_images[:100]:
                lines.append(f"  - {id_code}")
            if len(result.missing_images) > 100:
                lines.append(f"  ... and {len(result.missing_images) - 100} more")
            lines.append("")

    if not has_issues:
        lines.append("✓ No critical issues found!")
        lines.append("")
        lines.append("All datasets passed validation.")
        lines.append("")
    else:
        lines.append("=" * 80)
        lines.append("ACTION REQUIRED")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Please address the issues listed above before training.")
        lines.append("See data/VALIDATION_REPORT.md for detailed information.")
        lines.append("")

    # Write issues file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive data validation for DR classification datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate all datasets
    python scripts/validate_data.py

    # Validate specific dataset
    python scripts/validate_data.py --dataset aptos-train

    # Quick mode (skip duplicate detection)
    python scripts/validate_data.py --quick

    # Verbose output
    python scripts/validate_data.py --verbose
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        default='all',
        help='Dataset to validate (default: all)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip duplicate detection'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for reports (default: data)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE DATA VALIDATION")
    print("=" * 80)
    print()

    # Determine which datasets to validate
    if args.dataset == 'all':
        datasets_to_validate = DATASETS
    else:
        datasets_to_validate = {args.dataset: DATASETS[args.dataset]}

    print(f"Validating {len(datasets_to_validate)} dataset(s)...")
    print(f"Quick mode: {'Yes' if args.quick else 'No'}")
    print(f"Verbose: {'Yes' if args.verbose else 'No'}")
    print()

    # Validate each dataset
    results = {}
    for dataset_key, config in datasets_to_validate.items():
        # Skip if dataset doesn't exist
        csv_path = Path(config['csv'])
        if not csv_path.exists():
            print(f"⊗ Skipping {config['name']}: CSV not found")
            continue

        result = validate_dataset(
            dataset_key,
            config,
            check_duplicates=not args.quick,
            verbose=args.verbose
        )

        results[dataset_key] = result

        # Print summary
        status_icon = "✓" if result.passed else "✗"
        status_text = "PASSED" if result.passed else "FAILED"
        print(f"{status_icon} {config['name']}: {status_text}")

    print()
    print("=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    print()

    # Generate reports
    output_dir = Path(args.output_dir)

    report_path = output_dir / "VALIDATION_REPORT.md"
    generate_markdown_report(results, report_path, args.quick)
    print(f"✓ Generated validation report: {report_path}")

    issues_path = output_dir / "issues.txt"
    generate_issues_file(results, issues_path)
    print(f"✓ Generated issues file: {issues_path}")

    print()

    # Final summary
    total_datasets = len(results)
    passed_datasets = sum(1 for r in results.values() if r.passed)
    failed_datasets = total_datasets - passed_datasets

    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {total_datasets}")
    print(f"Passed: {passed_datasets} ✓")
    print(f"Failed: {failed_datasets} ✗")
    print()

    if failed_datasets == 0:
        print("✅ All datasets passed validation!")
        print("Your data is ready for training.")
        return 0
    else:
        print("⚠️ Some datasets failed validation.")
        print(f"Please review {report_path} and {issues_path} for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
