#!/usr/bin/env python3
"""
Data loading performance tests.

Profile data loading pipeline to identify bottlenecks and optimize training throughput.
Measures throughput, identifies optimal num_workers, and quantifies augmentation overhead.

Usage:
    python scripts/test_data_loading.py --config configs/retfound_lora_config.yaml
    python scripts/test_data_loading.py --config configs/retfound_lora_config.yaml --num-workers 0,2,4,8
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, validate_config
from scripts.dataset import RetinalDataset
from scripts.train_retfound_lora import get_transforms


class DataLoadingTester:
    """Test data loading performance and identify bottlenecks."""

    def __init__(self, config_path: str, verbose: bool = True):
        """
        Initialize the data loading tester.

        Args:
            config_path: Path to YAML config file
            verbose: Print detailed information
        """
        self.verbose = verbose
        self.config = Config.from_yaml(config_path)
        validate_config(self.config)

        # Determine device and pin_memory setting
        self.use_pin_memory = torch.cuda.is_available()
        self.device_type = "cuda" if torch.cuda.is_available() else \
                          ("mps" if torch.backends.mps.is_available() else "cpu")

        if self.verbose:
            print(f"\nDevice: {self.device_type}")
            print(f"pin_memory: {self.use_pin_memory}")

    def test_throughput(self, num_workers: int, num_batches: int = 50) -> Dict:
        """
        Measure dataloader throughput.

        Args:
            num_workers: Number of workers for DataLoader
            num_batches: Number of batches to measure (reduced default for faster testing)

        Returns:
            Dictionary with throughput metrics
        """
        if self.verbose:
            print(f"\nTesting num_workers={num_workers}...")

        try:
            # Create dataset
            train_transform, _ = get_transforms(
                img_size=self.config.image.input_size,
                model_variant=self.config.model.model_variant
            )

            dataset = RetinalDataset(
                csv_file=self.config.data.train_csv,
                img_dir=self.config.data.img_dir,
                transform=train_transform
            )

            if len(dataset) == 0:
                if self.verbose:
                    print(f"  ⚠️  Warning: Dataset is empty")
                return {}

            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=self.use_pin_memory,
                persistent_workers=(num_workers > 0)
            )

            if self.verbose:
                print(f"  Loading {num_batches} batches...")

            # Warmup with timeout protection
            warmup_count = 0
            for i, batch in enumerate(dataloader):
                warmup_count += 1
                if warmup_count >= 3:
                    break

            if warmup_count == 0:
                if self.verbose:
                    print(f"  ⚠️  Warning: Failed to load any batches")
                return {}

            # Measure throughput with progress indicator
            start_time = time.time()
            batch_times = []

            for i, batch in enumerate(dataloader):
                batch_time = time.time()
                batch_times.append(batch_time)

                if (i + 1) % max(1, num_batches // 5) == 0 and self.verbose:
                    print(f"    Loaded {i + 1}/{num_batches} batches...")

                if i >= num_batches - 1:
                    break

            elapsed = time.time() - start_time
            actual_batches = len(batch_times)
            throughput = actual_batches / elapsed if elapsed > 0 else 0

            # Calculate per-batch times
            if len(batch_times) > 1:
                per_batch_times = [batch_times[i] - batch_times[i-1] for i in range(1, len(batch_times))]
                avg_batch_time = sum(per_batch_times) / len(per_batch_times)
            else:
                avg_batch_time = elapsed / actual_batches if actual_batches > 0 else 0

            if self.verbose:
                print(f"  Throughput: {throughput:.2f} batches/sec")
                print(f"  Avg batch time: {avg_batch_time*1000:.1f} ms")

            return {
                'num_workers': num_workers,
                'throughput': throughput,
                'avg_batch_time_ms': avg_batch_time * 1000,
                'total_time': elapsed,
            }

        except Exception as e:
            if self.verbose:
                print(f"  ❌ Error: {str(e)[:100]}")
            return {}

    def compare_num_workers(self, worker_counts: List[int]) -> Dict:
        """
        Compare performance across different num_workers settings.

        Args:
            worker_counts: List of worker counts to test

        Returns:
            Dictionary mapping worker counts to performance metrics
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("NUM_WORKERS COMPARISON")
            print("=" * 60)

        results = {}
        for num_workers in worker_counts:
            # Skip num_workers > 0 on non-CUDA systems to avoid multiprocessing issues
            if num_workers > 0 and not torch.cuda.is_available():
                if self.verbose:
                    print(f"\nTesting num_workers={num_workers}...")
                    print(f"  ⚠️  Skipping num_workers > 0 on {self.device_type} (causes multiprocessing issues)")
                continue

            try:
                result = self.test_throughput(num_workers, num_batches=50)
                if result:
                    results[num_workers] = result
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Error testing num_workers={num_workers}: {e}")
                continue

        # Find optimal
        if results:
            best = max(results.items(), key=lambda x: x[1]['throughput'])
            if self.verbose:
                print(f"\n✅ Optimal num_workers: {best[0]} ({best[1]['throughput']:.2f} batches/sec)")

        return results

    def test_augmentation_overhead(self) -> Dict:
        """
        Measure augmentation overhead by comparing with/without augmentation.

        Returns:
            Dictionary with augmentation overhead metrics
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("AUGMENTATION OVERHEAD TEST")
            print("=" * 60)

        # Test with augmentation (train transforms)
        if self.verbose:
            print("\n[1/2] With augmentation...")

        train_transform, _ = get_transforms(
            img_size=self.config.image.input_size,
            model_variant=self.config.model.model_variant
        )

        dataset_aug = RetinalDataset(
            csv_file=self.config.data.train_csv,
            img_dir=self.config.data.img_dir,
            transform=train_transform
        )
        result_aug = self._measure_dataset(dataset_aug, num_batches=50)

        # Test without augmentation (val transforms)
        if self.verbose:
            print("\n[2/2] Without augmentation...")

        _, val_transform = get_transforms(
            img_size=self.config.image.input_size,
            model_variant=self.config.model.model_variant
        )

        dataset_no_aug = RetinalDataset(
            csv_file=self.config.data.train_csv,
            img_dir=self.config.data.img_dir,
            transform=val_transform
        )
        result_no_aug = self._measure_dataset(dataset_no_aug, num_batches=50)

        # Calculate overhead
        overhead_ms = result_aug['avg_batch_time_ms'] - result_no_aug['avg_batch_time_ms']
        overhead_pct = (overhead_ms / result_aug['avg_batch_time_ms']) * 100

        if self.verbose:
            print(f"\nAugmentation overhead: {overhead_ms:.1f} ms ({overhead_pct:.1f}% of total)")

            if overhead_pct > 30:
                print("⚠️  Warning: Augmentation overhead > 30%, consider simplifying")
            else:
                print("✅ Augmentation overhead acceptable")

        return {
            'with_aug': result_aug,
            'without_aug': result_no_aug,
            'overhead_ms': overhead_ms,
            'overhead_pct': overhead_pct,
        }

    def _measure_dataset(self, dataset, num_batches: int = 50) -> Dict:
        """
        Helper to measure dataset performance.

        Args:
            dataset: PyTorch Dataset to measure
            num_batches: Number of batches to measure

        Returns:
            Dictionary with throughput metrics
        """
        try:
            if len(dataset) == 0:
                if self.verbose:
                    print(f"  ⚠️  Warning: Dataset is empty")
                return {}

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=min(4, self.config.training.batch_size),  # Adaptive workers
                pin_memory=self.use_pin_memory
            )

            # Warmup with counter
            warmup_count = 0
            for i, batch in enumerate(dataloader):
                warmup_count += 1
                if warmup_count >= 3:
                    break

            if warmup_count == 0:
                if self.verbose:
                    print(f"  ⚠️  Warning: Failed to load any batches")
                return {}

            # Measure
            start = time.time()
            batch_count = 0
            for i, batch in enumerate(dataloader):
                batch_count += 1
                if batch_count >= num_batches:
                    break

            elapsed = time.time() - start

            if batch_count == 0:
                return {}

            throughput = batch_count / elapsed
            avg_batch_time = elapsed / batch_count

            if self.verbose:
                print(f"  Throughput: {throughput:.2f} batches/sec")
                print(f"  Avg batch time: {avg_batch_time*1000:.1f} ms")

            return {
                'throughput': throughput,
                'avg_batch_time_ms': avg_batch_time * 1000,
            }
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Error: {str(e)[:100]}")
            return {}

    def plot_results(self, results: Dict, output_path: str):
        """
        Plot performance comparison across num_workers settings.

        Args:
            results: Dictionary mapping worker counts to metrics
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        workers = sorted(results.keys())
        throughputs = [results[w]['throughput'] for w in workers]

        ax.plot(workers, throughputs, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('num_workers', fontsize=12)
        ax.set_ylabel('Throughput (batches/sec)', fontsize=12)
        ax.set_title('DataLoader Performance vs num_workers', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Mark optimal
        best_idx = throughputs.index(max(throughputs))
        ax.axvline(workers[best_idx], color='r', linestyle='--', alpha=0.5,
                   label=f'Optimal: {workers[best_idx]}')
        ax.legend()

        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if self.verbose:
            print(f"\n📊 Plot saved to: {output_path}")

        plt.close()

    def run_all_tests(self, worker_counts: List[int]):
        """
        Run complete data loading test suite.

        Args:
            worker_counts: List of worker counts to test
        """
        if self.verbose:
            print("=" * 60)
            print("DATA LOADING PERFORMANCE TESTS")
            print("=" * 60)

        # Test 1: Compare num_workers
        worker_results = self.compare_num_workers(worker_counts)

        if not worker_results:
            print("\n❌ All num_workers tests failed")
            print("   Possible reasons:")
            print("   • Dataset not found or is empty")
            print("   • CSV file missing or malformed")
            print("   • Image directory missing or no images")
            return

        # Test 2: Augmentation overhead
        aug_results = self.test_augmentation_overhead()

        # Plot results
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "data_loading_performance.png"
        self.plot_results(worker_results, str(plot_path))

        # Summary
        if self.verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)

            # Find valid results
            valid_results = {k: v for k, v in worker_results.items() if v}
            if not valid_results:
                print("\n⚠️  No valid test results collected")
                return

            best = max(valid_results.items(), key=lambda x: x[1]['throughput'])
            print(f"\nRecommendations:")
            print(f"  • Set num_workers: {best[0]}")
            print(f"  • Expected throughput: {best[1]['throughput']:.2f} batches/sec")

            if aug_results and 'overhead_pct' in aug_results:
                if aug_results['overhead_pct'] > 30:
                    print(f"  • Consider simplifying augmentation (current overhead: {aug_results['overhead_pct']:.1f}%)")
                else:
                    print(f"  • Augmentation overhead OK ({aug_results['overhead_pct']:.1f}%)")
            else:
                print(f"  • Augmentation tests could not be completed")


def main():
    parser = argparse.ArgumentParser(description="Test data loading performance")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--num-workers',
        type=str,
        default='0,2,4,8',
        help='Comma-separated num_workers to test (default: "0,2,4,8")'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )

    args = parser.parse_args()

    worker_counts = [int(w.strip()) for w in args.num_workers.split(',')]

    tester = DataLoadingTester(args.config, verbose=args.verbose)
    tester.run_all_tests(worker_counts)


if __name__ == '__main__':
    main()
