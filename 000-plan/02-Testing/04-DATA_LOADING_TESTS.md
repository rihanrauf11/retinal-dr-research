# Phase 4: Data Loading Tests

**Priority:** MEDIUM (Performance optimization)
**Time Estimate:** 4-5 hours
**Dependencies:** Phase 1 complete
**Deliverable:** `scripts/test_data_loading.py`

---

## Objective

Profile data loading pipeline to identify bottlenecks and optimize training throughput. Ensure GPU is not starved waiting for data.

---

## Rationale

**Problem:**
- Training takes 5 hours but could take 3 hours with optimized data loading
- GPU utilization only 60% because waiting for data
- Don't know optimal num_workers setting
- Augmentation pipeline might be bottleneck

**What we measure:**
- DataLoader throughput (batches/sec)
- GPU utilization during training
- Augmentation overhead
- Optimal num_workers setting
- I/O vs compute time

---

## Key Metrics

### 1. DataLoader Throughput
**Metric:** Batches per second
**Target:** > 10 batches/sec (for batch_size=32)
**Command:**
```python
start = time.time()
for i, batch in enumerate(dataloader):
    if i >= 100:
        break
elapsed = time.time() - start
throughput = 100 / elapsed
```

### 2. GPU Utilization
**Metric:** % time GPU is active
**Target:** > 90%
**Tool:** `nvidia-smi dmon` or PyTorch profiler

### 3. Augmentation Overhead
**Metric:** Time spent in augmentation vs total
**Target:** < 20% of total batch time

### 4. Optimal num_workers
**Test:** Compare num_workers = 0, 2, 4, 8, 16
**Find:** Sweet spot where throughput plateaus

---

## Implementation

```python
#!/usr/bin/env python3
"""
Data loading performance tests.

Usage:
    python scripts/test_data_loading.py --config configs/retfound_lora_config.yaml
    python scripts/test_data_loading.py --config configs/retfound_lora_config.yaml --num-workers 0,2,4,8
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config
from scripts.dataset import RetinalDataset, get_train_transforms, get_val_transforms


class DataLoadingTester:
    """Test data loading performance."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

    def test_throughput(self, num_workers: int, num_batches: int = 100) -> Dict:
        """Measure dataloader throughput."""
        print(f"\nTesting num_workers={num_workers}...")

        dataset = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_train_transforms()
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break

        # Measure throughput
        start_time = time.time()
        batch_times = []

        for i, batch in enumerate(dataloader):
            batch_time = time.time()
            batch_times.append(batch_time)

            if i >= num_batches:
                break

        elapsed = time.time() - start_time
        throughput = num_batches / elapsed

        # Calculate per-batch times
        if len(batch_times) > 1:
            per_batch_times = [batch_times[i] - batch_times[i-1] for i in range(1, len(batch_times))]
            avg_batch_time = sum(per_batch_times) / len(per_batch_times)
        else:
            avg_batch_time = elapsed / num_batches

        print(f"  Throughput: {throughput:.2f} batches/sec")
        print(f"  Avg batch time: {avg_batch_time*1000:.1f} ms")

        return {
            'num_workers': num_workers,
            'throughput': throughput,
            'avg_batch_time_ms': avg_batch_time * 1000,
            'total_time': elapsed,
        }

    def compare_num_workers(self, worker_counts: List[int]) -> Dict:
        """Compare different num_workers settings."""
        print("\n" + "=" * 60)
        print("NUM_WORKERS COMPARISON")
        print("=" * 60)

        results = {}
        for num_workers in worker_counts:
            result = self.test_throughput(num_workers, num_batches=50)
            results[num_workers] = result

        # Find optimal
        best = max(results.items(), key=lambda x: x[1]['throughput'])
        print(f"\n✅ Optimal num_workers: {best[0]} ({best[1]['throughput']:.2f} batches/sec)")

        return results

    def test_augmentation_overhead(self) -> Dict:
        """Measure augmentation overhead."""
        print("\n" + "=" * 60)
        print("AUGMENTATION OVERHEAD TEST")
        print("=" * 60)

        # Test with augmentation
        print("\n[1/2] With augmentation...")
        dataset_aug = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_train_transforms()
        )
        result_aug = self._measure_dataset(dataset_aug, num_batches=50)

        # Test without augmentation
        print("\n[2/2] Without augmentation...")
        dataset_no_aug = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_val_transforms()  # Only resize
        )
        result_no_aug = self._measure_dataset(dataset_no_aug, num_batches=50)

        # Calculate overhead
        overhead_ms = result_aug['avg_batch_time_ms'] - result_no_aug['avg_batch_time_ms']
        overhead_pct = (overhead_ms / result_aug['avg_batch_time_ms']) * 100

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
        """Helper to measure dataset performance."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break

        # Measure
        start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

        elapsed = time.time() - start
        throughput = num_batches / elapsed
        avg_batch_time = elapsed / num_batches

        print(f"  Throughput: {throughput:.2f} batches/sec")
        print(f"  Avg batch time: {avg_batch_time*1000:.1f} ms")

        return {
            'throughput': throughput,
            'avg_batch_time_ms': avg_batch_time * 1000,
        }

    def plot_results(self, results: Dict, output_path: str):
        """Plot performance comparison."""
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
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_path}")
        plt.close()

    def run_all_tests(self, worker_counts: List[int]):
        """Run complete test suite."""
        print("=" * 60)
        print("DATA LOADING PERFORMANCE TESTS")
        print("=" * 60)

        # Test 1: Compare num_workers
        worker_results = self.compare_num_workers(worker_counts)

        # Test 2: Augmentation overhead
        aug_results = self.test_augmentation_overhead()

        # Plot results
        output_dir = Path(self.config.system.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "data_loading_performance.png"
        self.plot_results(worker_results, str(plot_path))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        best = max(worker_results.items(), key=lambda x: x[1]['throughput'])
        print(f"\nRecommendations:")
        print(f"  • Set num_workers: {best[0]}")
        print(f"  • Expected throughput: {best[1]['throughput']:.2f} batches/sec")

        if aug_results['overhead_pct'] > 30:
            print(f"  • Consider simplifying augmentation (current overhead: {aug_results['overhead_pct']:.1f}%)")
        else:
            print(f"  • Augmentation overhead OK ({aug_results['overhead_pct']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test data loading performance")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num-workers', type=str, default='0,2,4,8',
                        help='Comma-separated num_workers to test')

    args = parser.parse_args()

    worker_counts = [int(w.strip()) for w in args.num_workers.split(',')]

    tester = DataLoadingTester(args.config)
    tester.run_all_tests(worker_counts)


if __name__ == '__main__':
    main()
```

---

## Usage

```bash
# Test default worker counts
python scripts/test_data_loading.py --config configs/retfound_lora_config.yaml

# Custom worker counts
python scripts/test_data_loading.py --config configs/retfound_lora_config.yaml --num-workers 0,4,8,16
```

---

## Expected Output

```
============================================================
DATA LOADING PERFORMANCE TESTS
============================================================

============================================================
NUM_WORKERS COMPARISON
============================================================

Testing num_workers=0...
  Throughput: 12.34 batches/sec
  Avg batch time: 81.0 ms

Testing num_workers=2...
  Throughput: 24.56 batches/sec
  Avg batch time: 40.7 ms

Testing num_workers=4...
  Throughput: 32.18 batches/sec
  Avg batch time: 31.1 ms

Testing num_workers=8...
  Throughput: 31.94 batches/sec
  Avg batch time: 31.3 ms

✅ Optimal num_workers: 4 (32.18 batches/sec)

============================================================
AUGMENTATION OVERHEAD TEST
============================================================

[1/2] With augmentation...
  Throughput: 28.45 batches/sec
  Avg batch time: 35.2 ms

[2/2] Without augmentation...
  Throughput: 42.18 batches/sec
  Avg batch time: 23.7 ms

Augmentation overhead: 11.5 ms (32.7% of total)
⚠️  Warning: Augmentation overhead > 30%, consider simplifying

============================================================
SUMMARY
============================================================

Recommendations:
  • Set num_workers: 4
  • Expected throughput: 32.18 batches/sec
  • Consider simplifying augmentation (current overhead: 32.7%)

📊 Plot saved to: results/retfound_lora/data_loading_performance.png
```

---

## Optimization Tips

### If Throughput Low (< 10 batches/sec)

**1. Increase num_workers:**
```yaml
data:
  num_workers: 4  # or 8
```

**2. Enable pin_memory:**
```python
pin_memory=True  # Faster data transfer to GPU
```

**3. Use persistent_workers:**
```python
persistent_workers=True  # Avoid worker recreation
```

### If Augmentation Overhead High (> 30%)

**1. Simplify augmentation:**
```python
# Remove expensive transforms:
# - ElasticTransform
# - GridDistortion
# Keep only:
# - RandomRotate90
# - HorizontalFlip
# - ColorJitter
```

**2. Reduce augmentation probability:**
```python
A.HorizontalFlip(p=0.3)  # Instead of p=0.5
```

---

## Next Steps

After Phase 4:
1. **Apply recommendations:** Update config with optimal num_workers
2. **Optional:** Proceed to [05-VISUAL_INSPECTION.md](05-VISUAL_INSPECTION.md)
3. **Or start training:** If satisfied with performance

---

## Success Criteria

- ✅ Script runs and measures throughput
- ✅ Identifies optimal num_workers
- ✅ Quantifies augmentation overhead
- ✅ Generates performance plots
- ✅ Provides actionable recommendations

