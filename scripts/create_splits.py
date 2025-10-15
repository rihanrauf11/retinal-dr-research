#!/usr/bin/env python3
"""
Create Stratified Train/Validation Splits for APTOS Dataset

This script creates stratified train/validation splits from the APTOS training data,
ensuring balanced class distribution across splits. This is critical for diabetic
retinopathy classification where class imbalance is significant.

Key Features:
    - Stratified splitting: maintains class distribution in both splits
    - Reproducible: uses fixed random seed (42)
    - Verification: checks for overlap, validates counts and distributions
    - Detailed reporting: prints comprehensive statistics

Usage:
    # Default: 80/20 split from data/aptos/train.csv
    python scripts/create_splits.py

    # Custom split ratio
    python scripts/create_splits.py --train-ratio 0.75

    # Custom paths
    python scripts/create_splits.py \\
        --input-csv data/aptos/train.csv \\
        --output-dir data/aptos \\
        --train-ratio 0.8

    # Verify existing splits
    python scripts/create_splits.py --verify-only

Author: Generated with Claude Code
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT_CSV = "data/aptos/train.csv"
DEFAULT_OUTPUT_DIR = "data/aptos"
DEFAULT_TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# DR severity levels for reference
DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV data and validate format.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with validated data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['id_code', 'diagnosis']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate diagnosis values (should be 0-4)
    valid_diagnoses = set(range(5))
    actual_diagnoses = set(df['diagnosis'].unique())
    invalid = actual_diagnoses - valid_diagnoses
    if invalid:
        raise ValueError(f"Invalid diagnosis values found: {invalid}")

    print(f"✓ Loaded {len(df)} samples from {csv_path}")
    return df


def get_class_distribution(df: pd.DataFrame) -> Dict[int, int]:
    """
    Get class distribution from DataFrame.

    Args:
        df: DataFrame with 'diagnosis' column

    Returns:
        Dictionary mapping class -> count
    """
    return df['diagnosis'].value_counts().sort_index().to_dict()


def print_class_distribution(df: pd.DataFrame, label: str = "Dataset") -> None:
    """
    Print detailed class distribution statistics.

    Args:
        df: DataFrame with 'diagnosis' column
        label: Label for the dataset being printed
    """
    dist = get_class_distribution(df)
    total = len(df)

    print(f"\n{label} Class Distribution:")
    print("=" * 60)
    print(f"{'Class':<8} {'Name':<18} {'Count':<8} {'Percentage':<12}")
    print("-" * 60)

    for cls in sorted(dist.keys()):
        count = dist[cls]
        pct = (count / total) * 100
        name = DR_CLASSES.get(cls, f"Unknown ({cls})")
        print(f"{cls:<8} {name:<18} {count:<8} {pct:>6.2f}%")

    print("-" * 60)
    print(f"{'Total':<27} {total:<8} {100.0:>6.2f}%")
    print()


def create_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split.

    Stratification ensures that the class distribution in the training and
    validation sets matches the overall distribution. This is critical for
    imbalanced datasets like APTOS where some classes (e.g., class 3) have
    very few samples.

    Args:
        df: Input DataFrame
        train_ratio: Fraction of data for training (default: 0.8)
        random_state: Random seed for reproducibility

    Returns:
        (train_df, val_df): Train and validation DataFrames
    """
    print(f"Creating {train_ratio:.0%}/{(1-train_ratio):.0%} stratified split...")
    print(f"Random seed: {random_state}")

    # Perform stratified split
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df['diagnosis'],  # This is the key for balanced splits!
        random_state=random_state,
        shuffle=True
    )

    # Reset indices for clean CSV output
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"✓ Created train split: {len(train_df)} samples")
    print(f"✓ Created val split: {len(val_df)} samples")

    return train_df, val_df


def verify_splits(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> bool:
    """
    Verify that splits are valid and properly stratified.

    Checks:
        1. No overlap between train and val
        2. Total samples = original samples
        3. All original IDs accounted for
        4. Class distributions are similar

    Args:
        original_df: Original full dataset
        train_df: Training split
        val_df: Validation split

    Returns:
        True if all checks pass, False otherwise
    """
    print("\nVerifying splits...")
    print("=" * 60)

    all_passed = True

    # Check 1: No overlap
    train_ids = set(train_df['id_code'])
    val_ids = set(val_df['id_code'])
    overlap = train_ids & val_ids

    if overlap:
        print(f"✗ FAIL: Found {len(overlap)} overlapping samples")
        all_passed = False
    else:
        print(f"✓ PASS: No overlap between train and val")

    # Check 2: Total count
    original_count = len(original_df)
    split_count = len(train_df) + len(val_df)

    if original_count != split_count:
        print(f"✗ FAIL: Total count mismatch ({split_count} vs {original_count})")
        all_passed = False
    else:
        print(f"✓ PASS: Total samples match ({split_count} = {original_count})")

    # Check 3: All IDs accounted for
    original_ids = set(original_df['id_code'])
    split_ids = train_ids | val_ids

    missing = original_ids - split_ids
    extra = split_ids - original_ids

    if missing or extra:
        print(f"✗ FAIL: ID mismatch (missing: {len(missing)}, extra: {len(extra)})")
        all_passed = False
    else:
        print(f"✓ PASS: All original IDs accounted for")

    # Check 4: Class distribution similarity
    print("\nClass Distribution Comparison:")
    print("-" * 60)
    print(f"{'Class':<8} {'Original %':<14} {'Train %':<14} {'Val %':<14}")
    print("-" * 60)

    original_dist = get_class_distribution(original_df)
    train_dist = get_class_distribution(train_df)
    val_dist = get_class_distribution(val_df)

    for cls in sorted(original_dist.keys()):
        orig_pct = (original_dist[cls] / len(original_df)) * 100
        train_pct = (train_dist.get(cls, 0) / len(train_df)) * 100
        val_pct = (val_dist.get(cls, 0) / len(val_df)) * 100

        # Check if percentages are within ±2% of original
        train_diff = abs(train_pct - orig_pct)
        val_diff = abs(val_pct - orig_pct)

        status = "✓" if (train_diff < 2.0 and val_diff < 2.0) else "✗"
        print(f"{status} {cls:<6} {orig_pct:>6.2f}%       {train_pct:>6.2f}%       {val_pct:>6.2f}%")

    print("=" * 60)

    if all_passed:
        print("\n✓ All verification checks passed!")
    else:
        print("\n✗ Some verification checks failed!")

    return all_passed


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Save train and validation splits to CSV files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        output_dir: Directory to save splits

    Returns:
        (train_path, val_path): Paths to saved CSV files
    """
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    train_path = output_dir / "train_split.csv"
    val_path = output_dir / "val_split.csv"

    # Save CSVs
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\n✓ Saved training split to: {train_path}")
    print(f"✓ Saved validation split to: {val_path}")

    return train_path, val_path


def print_summary(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> None:
    """
    Print comprehensive summary of the splits.

    Args:
        original_df: Original full dataset
        train_df: Training split
        val_df: Validation split
    """
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    print(f"\nOriginal samples: {len(original_df)}")
    print(f"Train samples:    {len(train_df)} ({len(train_df)/len(original_df)*100:.1f}%)")
    print(f"Val samples:      {len(val_df)} ({len(val_df)/len(original_df)*100:.1f}%)")

    # Print class distributions
    print_class_distribution(original_df, "Original")
    print_class_distribution(train_df, "Train")
    print_class_distribution(val_df, "Validation")


def verify_existing_splits(output_dir: Path) -> bool:
    """
    Verify existing split files without regenerating them.

    Args:
        output_dir: Directory containing split files

    Returns:
        True if splits exist and are valid
    """
    train_path = output_dir / "train_split.csv"
    val_path = output_dir / "val_split.csv"

    if not train_path.exists() or not val_path.exists():
        print("✗ Split files do not exist")
        return False

    print(f"Loading existing splits from {output_dir}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Try to load original for comparison
    original_path = output_dir / "train.csv"
    if original_path.exists():
        original_df = pd.read_csv(original_path)
        verify_splits(original_df, train_df, val_df)

    print_class_distribution(train_df, "Train Split")
    print_class_distribution(val_df, "Validation Split")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main function to create stratified splits."""
    parser = argparse.ArgumentParser(
        description="Create stratified train/validation splits for APTOS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create 80/20 split (default)
    python scripts/create_splits.py

    # Create 75/25 split
    python scripts/create_splits.py --train-ratio 0.75

    # Verify existing splits
    python scripts/create_splits.py --verify-only

    # Custom paths
    python scripts/create_splits.py \\
        --input-csv data/aptos/train.csv \\
        --output-dir data/aptos
        """
    )

    parser.add_argument(
        '--input-csv',
        type=str,
        default=DEFAULT_INPUT_CSV,
        help=f'Input CSV file (default: {DEFAULT_INPUT_CSV})'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for splits (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f'Training set ratio (default: {DEFAULT_TRAIN_RATIO})'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing splits without regenerating'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("APTOS Dataset Stratified Split Creation")
    print("=" * 60)

    # Verify-only mode
    if args.verify_only:
        success = verify_existing_splits(output_dir)
        return 0 if success else 1

    # Validate train ratio
    if not (0 < args.train_ratio < 1):
        print(f"✗ Error: train-ratio must be between 0 and 1 (got {args.train_ratio})")
        return 1

    try:
        # Load data
        print(f"\nInput CSV: {input_csv}")
        print(f"Output directory: {output_dir}")
        print()

        df = load_data(input_csv)

        # Print original distribution
        print_class_distribution(df, "Original Dataset")

        # Create splits
        train_df, val_df = create_stratified_split(
            df,
            train_ratio=args.train_ratio,
            random_state=args.seed
        )

        # Verify splits
        if not verify_splits(df, train_df, val_df):
            print("\n✗ Warning: Verification checks failed!")
            print("Proceeding anyway, but please review the output carefully.")

        # Save splits
        train_path, val_path = save_splits(train_df, val_df, output_dir)

        # Print summary
        print_summary(df, train_df, val_df)

        print("=" * 60)
        print("✓ Split creation complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Update your config files to use:")
        print(f"   train_csv: {train_path}")
        print(f"   val_csv: {val_path}")
        print(f"\n2. Remove or comment out 'val_split' in config")
        print(f"   (splits are now pre-defined)")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
