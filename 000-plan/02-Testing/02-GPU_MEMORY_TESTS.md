# Phase 2: GPU Memory Tests

**Priority:** CRITICAL (Must complete before training)
**Time Estimate:** 4-5 hours
**Dependencies:** Phase 1 (Sanity Tests) complete
**Deliverable:** `scripts/test_gpu_memory.py`

---

## Objective

Implement GPU memory profiling to predict peak memory usage and recommend optimal batch sizes BEFORE starting expensive training runs. Prevents OOM (Out Of Memory) crashes that waste GPU hours.

---

## Rationale

**Why this is critical:**
- Training crashes 80% through first epoch due to OOM
- Don't know if batch_size=32 will fit in 8GB GPU
- Can't compare RETFound Large (11GB) vs Green (6GB) memory needs
- No visibility into memory usage patterns during training

**What we gain:**
- Predict OOM before training starts
- Recommend maximum safe batch size
- Compare memory footprint of different models
- Optimize memory usage (gradient accumulation, mixed precision)

**Cost of NOT having this:**
- Restart training multiple times with different batch sizes
- Waste 30+ minutes per failed attempt
- Unable to plan multi-GPU or cloud GPU usage

---

## Memory Components to Profile

### 1. Model Parameters
**Formula:** `num_params × 4 bytes (float32)`

**What to measure:**
- Total parameters
- Trainable parameters (LoRA: only ~800K)
- Frozen parameters (backbone)

**Example:**
```
RETFound Large: 303M params × 4 bytes = 1,212 MB
RETFound Green: 21.3M params × 4 bytes = 85 MB
LoRA adapters: 800K params × 4 bytes = 3.2 MB
```

---

### 2. Optimizer State
**Formula:** `trainable_params × 8 bytes (AdamW keeps 2 states per param)`

**What to measure:**
- Optimizer state size
- Difference between AdamW (8 bytes/param) vs SGD (4 bytes/param)

**Example:**
```
RETFound + LoRA: 800K params × 8 bytes = 6.4 MB
Full fine-tuning: 303M params × 8 bytes = 2,424 MB
```

---

### 3. Activations (Forward Pass)
**Formula:** `batch_size × feature_maps × height × width × 4 bytes`

**What to measure:**
- Peak activation memory during forward pass
- How it scales with batch size

**Example:**
```
ViT-Large (224×224, batch=32):
  - Patch embeddings: 32 × 768 × 196 × 4 = ~19 MB
  - Attention maps: 32 × 12 × 24 × 197 × 197 × 4 = ~140 MB per layer
  - Total activations: ~2,000 MB (across all 24 layers)
```

---

### 4. Gradients (Backward Pass)
**Formula:** `trainable_params × 4 bytes`

**What to measure:**
- Gradient memory
- Peak memory during backward pass

**Example:**
```
RETFound + LoRA: 800K params × 4 bytes = 3.2 MB
Full fine-tuning: 303M params × 4 bytes = 1,212 MB
```

---

### 5. Peak Memory
**What to measure:**
- Absolute peak during training step
- Memory spikes during specific operations
- Difference between train vs eval mode

**Formula (rough estimate):**
```
Peak Memory = Model + Optimizer + Activations + Gradients + Buffer (20%)
```

---

## Implementation

### Complete Script: `scripts/test_gpu_memory.py`

```python
#!/usr/bin/env python3
"""
GPU memory profiling for training pipeline.
Measures peak memory usage and recommends optimal batch sizes.

Usage:
    # Profile RETFound + LoRA
    python scripts/test_gpu_memory.py --config configs/retfound_lora_config.yaml

    # Profile with different batch sizes
    python scripts/test_gpu_memory.py --config configs/retfound_lora_config.yaml --batch-sizes 8,16,32,64

    # Compare RETFound Large vs Green
    python scripts/test_gpu_memory.py --config configs/retfound_lora_config.yaml --output compare_memory.json
    python scripts/test_gpu_memory.py --config configs/retfound_green_lora_config.yaml --output compare_memory.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config
from scripts.dataset import RetinalDataset, get_train_transforms
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier
from scripts.utils import set_seed


class GPUMemoryProfiler:
    """Profile GPU memory usage during training."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        # Check if GPU available
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, skipping GPU memory profiling")
            self.device = None
        else:
            self.device = torch.device("cuda")
            # Get GPU properties
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {self.gpu_name}")
            print(f"Total Memory: {self.gpu_total_memory:.2f} GB")

    def reset_memory_stats(self):
        """Reset CUDA memory statistics."""
        if self.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics in MB."""
        if not self.device:
            return {}

        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1e6,
            'reserved_mb': torch.cuda.memory_reserved() / 1e6,
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1e6,
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1e6,
        }

    def profile_model_memory(self) -> Dict[str, float]:
        """Profile memory used by model parameters."""
        print("\n[1/5] Profiling model parameters...")

        if not self.device:
            return {}

        self.reset_memory_stats()
        set_seed(self.config.system.seed)

        # Create model
        if hasattr(self.config.model, 'lora_r'):
            model = RETFoundLoRA(
                checkpoint_path=self.config.model.checkpoint_path,
                num_classes=self.config.model.num_classes,
                lora_r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
            )
        else:
            model = DRClassifier(
                model_name=self.config.model.model_name,
                num_classes=self.config.model.num_classes,
                pretrained=self.config.model.pretrained,
            )

        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Get memory after model loaded
        stats = self.get_memory_stats()

        # Calculate theoretical memory (4 bytes per param for float32)
        theoretical_mb = total_params * 4 / 1e6
        trainable_mb = trainable_params * 4 / 1e6
        frozen_mb = frozen_params * 4 / 1e6

        print(f"  Total parameters: {total_params:,} ({theoretical_mb:.1f} MB)")
        print(f"  Trainable: {trainable_params:,} ({trainable_mb:.1f} MB)")
        print(f"  Frozen: {frozen_params:,} ({frozen_mb:.1f} MB)")
        print(f"  Actual GPU memory: {stats['allocated_mb']:.1f} MB")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'theoretical_mb': theoretical_mb,
            'actual_mb': stats['allocated_mb'],
            'model': model,  # Keep for next tests
        }

    def profile_optimizer_memory(self, model) -> Dict[str, float]:
        """Profile memory used by optimizer state."""
        print("\n[2/5] Profiling optimizer state...")

        if not self.device:
            return {}

        self.reset_memory_stats()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Trigger optimizer state creation (do a dummy step)
        dummy_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

        stats = self.get_memory_stats()

        # AdamW keeps 2 states per param (momentum + variance)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        theoretical_mb = trainable_params * 8 / 1e6  # 8 bytes = 2 states × 4 bytes

        print(f"  Optimizer: AdamW")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Theoretical memory: {theoretical_mb:.1f} MB (2 states × 4 bytes)")
        print(f"  Actual memory: {stats['allocated_mb']:.1f} MB")

        return {
            'optimizer': 'AdamW',
            'trainable_params': trainable_params,
            'theoretical_mb': theoretical_mb,
            'actual_mb': stats['allocated_mb'],
        }

    def profile_forward_pass(self, model, batch_size: int) -> Dict[str, float]:
        """Profile memory during forward pass."""
        print(f"\n[3/5] Profiling forward pass (batch_size={batch_size})...")

        if not self.device:
            return {}

        self.reset_memory_stats()
        model.eval()

        # Create dummy batch
        img_size = getattr(self.config.model, 'img_size', 224)
        dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=self.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_images)

        stats = self.get_memory_stats()

        # Calculate activation memory (approximate)
        input_mb = dummy_images.numel() * 4 / 1e6
        output_mb = outputs.numel() * 4 / 1e6
        activation_mb = stats['peak_allocated_mb'] - input_mb

        print(f"  Input: {dummy_images.shape} ({input_mb:.1f} MB)")
        print(f"  Output: {outputs.shape} ({output_mb:.1f} MB)")
        print(f"  Peak memory: {stats['peak_allocated_mb']:.1f} MB")
        print(f"  Activation memory: {activation_mb:.1f} MB")

        return {
            'batch_size': batch_size,
            'input_mb': input_mb,
            'output_mb': output_mb,
            'peak_mb': stats['peak_allocated_mb'],
            'activation_mb': activation_mb,
        }

    def profile_backward_pass(self, model, batch_size: int) -> Dict[str, float]:
        """Profile memory during backward pass."""
        print(f"\n[4/5] Profiling backward pass (batch_size={batch_size})...")

        if not self.device:
            return {}

        self.reset_memory_stats()
        model.train()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Create dummy batch
        img_size = getattr(self.config.model, 'img_size', 224)
        dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=self.device)
        dummy_labels = torch.randint(0, self.config.model.num_classes, (batch_size,), device=self.device)

        # Forward + backward pass
        outputs = model(dummy_images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, dummy_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = self.get_memory_stats()

        # Calculate gradient memory
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gradient_mb = trainable_params * 4 / 1e6

        print(f"  Batch size: {batch_size}")
        print(f"  Peak memory: {stats['peak_allocated_mb']:.1f} MB")
        print(f"  Gradient memory: {gradient_mb:.1f} MB")

        return {
            'batch_size': batch_size,
            'peak_mb': stats['peak_allocated_mb'],
            'gradient_mb': gradient_mb,
        }

    def profile_training_step(self, model, batch_size: int) -> Dict[str, float]:
        """Profile complete training step (most realistic)."""
        print(f"\n[5/5] Profiling complete training step (batch_size={batch_size})...")

        if not self.device:
            return {}

        self.reset_memory_stats()
        model.train()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Do a few training steps to get stable memory usage
        img_size = getattr(self.config.model, 'img_size', 224)

        for _ in range(3):  # Warmup
            dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=self.device)
            dummy_labels = torch.randint(0, self.config.model.num_classes, (batch_size,), device=self.device)

            outputs = model(dummy_images)
            loss = criterion(outputs, dummy_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure peak memory after warmup
        stats = self.get_memory_stats()

        print(f"  Batch size: {batch_size}")
        print(f"  Peak allocated: {stats['peak_allocated_mb']:.1f} MB")
        print(f"  Peak reserved: {stats['peak_reserved_mb']:.1f} MB")

        return {
            'batch_size': batch_size,
            'peak_allocated_mb': stats['peak_allocated_mb'],
            'peak_reserved_mb': stats['peak_reserved_mb'],
        }

    def recommend_batch_size(self, memory_profile: Dict) -> int:
        """Recommend maximum safe batch size."""
        if not self.device:
            return self.config.training.batch_size

        # Get peak memory for batch_size=1
        peak_per_sample = memory_profile['peak_allocated_mb'] / memory_profile['batch_size']

        # Calculate max batch size (use 90% of GPU memory for safety)
        available_memory_mb = self.gpu_total_memory * 1000 * 0.9
        max_batch_size = int(available_memory_mb / peak_per_sample)

        # Round down to nearest power of 2
        recommended = 1
        while recommended * 2 <= max_batch_size:
            recommended *= 2

        return recommended

    def profile_multiple_batch_sizes(self, model, batch_sizes: List[int]) -> Dict:
        """Profile memory for multiple batch sizes."""
        print("\n" + "=" * 60)
        print("BATCH SIZE COMPARISON")
        print("=" * 60)

        results = {}

        for batch_size in batch_sizes:
            try:
                print(f"\nTesting batch_size={batch_size}...")
                self.reset_memory_stats()

                profile = self.profile_training_step(model, batch_size)
                results[batch_size] = profile

                # Check if approaching memory limit
                memory_usage_pct = profile['peak_reserved_mb'] / (self.gpu_total_memory * 1000) * 100
                if memory_usage_pct > 90:
                    print(f"  ⚠️  Warning: Using {memory_usage_pct:.1f}% of GPU memory")
                    break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ❌ OOM at batch_size={batch_size}")
                    break
                else:
                    raise

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate memory profiling report."""
        report = []
        report.append("\n" + "=" * 60)
        report.append("MEMORY PROFILING REPORT")
        report.append("=" * 60)

        # GPU info
        if self.device:
            report.append(f"\nGPU: {self.gpu_name}")
            report.append(f"Total Memory: {self.gpu_total_memory:.2f} GB")

        # Model parameters
        if 'model_params' in results:
            mp = results['model_params']
            report.append(f"\nModel Parameters:")
            report.append(f"  Total: {mp['total_params']:,} ({mp['theoretical_mb']:.1f} MB)")
            report.append(f"  Trainable: {mp['trainable_params']:,} ({mp['trainable_params'] * 4 / 1e6:.1f} MB)")
            report.append(f"  Frozen: {mp['frozen_params']:,} ({mp['frozen_params'] * 4 / 1e6:.1f} MB)")

            # Calculate parameter efficiency (LoRA)
            if mp['trainable_params'] < mp['total_params']:
                efficiency = (1 - mp['trainable_params'] / mp['total_params']) * 100
                report.append(f"  Parameter efficiency: {efficiency:.1f}% reduction (LoRA)")

        # Optimizer state
        if 'optimizer' in results:
            op = results['optimizer']
            report.append(f"\nOptimizer State:")
            report.append(f"  Type: {op['optimizer']}")
            report.append(f"  Memory: {op['theoretical_mb']:.1f} MB")

        # Peak memory
        if 'training_step' in results:
            ts = results['training_step']
            report.append(f"\nPeak Memory (batch_size={ts['batch_size']}):")
            report.append(f"  Allocated: {ts['peak_allocated_mb']:.1f} MB")
            report.append(f"  Reserved: {ts['peak_reserved_mb']:.1f} MB")

            if self.device:
                usage_pct = ts['peak_reserved_mb'] / (self.gpu_total_memory * 1000) * 100
                report.append(f"  GPU Usage: {usage_pct:.1f}%")

        # Batch size comparison
        if 'batch_size_comparison' in results:
            bsc = results['batch_size_comparison']
            report.append(f"\nBatch Size Comparison:")
            for bs, profile in sorted(bsc.items()):
                memory_mb = profile['peak_reserved_mb']
                memory_gb = memory_mb / 1000
                report.append(f"  batch_size={bs:2d}: {memory_gb:.2f} GB")

        # Recommendation
        if 'recommended_batch_size' in results:
            rbs = results['recommended_batch_size']
            report.append(f"\nRecommendation:")
            report.append(f"  ✅ Maximum safe batch_size: {rbs}")
            report.append(f"  Current config batch_size: {self.config.training.batch_size}")

            if rbs < self.config.training.batch_size:
                report.append(f"  ⚠️  Warning: Config batch_size too large, will likely OOM")
                report.append(f"  💡 Suggestion: Reduce to batch_size={rbs}")
            else:
                report.append(f"  ✅ Config batch_size is safe")

        # Memory optimization tips
        report.append(f"\nMemory Optimization Tips:")
        if self.device and results.get('training_step', {}).get('peak_reserved_mb', 0) > self.gpu_total_memory * 1000 * 0.8:
            report.append("  • Consider using gradient accumulation:")
            report.append("    effective_batch_size = batch_size × accumulation_steps")
            report.append("  • Enable mixed precision training (float16):")
            report.append("    Can reduce memory by 40-50%")
            report.append("  • Reduce LoRA rank (r=8 → r=4)")
        else:
            report.append("  ✅ Memory usage is healthy, no optimization needed")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Profile GPU memory usage")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--batch-sizes',
        type=str,
        default=None,
        help='Comma-separated batch sizes to test (e.g., "8,16,32,64")'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    # Create profiler
    profiler = GPUMemoryProfiler(args.config)

    if not profiler.device:
        print("No GPU available, exiting")
        sys.exit(1)

    # Profile model
    model_results = profiler.profile_model_memory()
    model = model_results.pop('model')

    # Profile optimizer
    optimizer_results = profiler.profile_optimizer_memory(model)

    # Profile forward pass
    forward_results = profiler.profile_forward_pass(
        model,
        batch_size=profiler.config.training.batch_size
    )

    # Profile backward pass
    backward_results = profiler.profile_backward_pass(
        model,
        batch_size=profiler.config.training.batch_size
    )

    # Profile complete training step
    training_results = profiler.profile_training_step(
        model,
        batch_size=profiler.config.training.batch_size
    )

    # Profile multiple batch sizes if requested
    batch_size_results = {}
    if args.batch_sizes:
        batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(',')]
        batch_size_results = profiler.profile_multiple_batch_sizes(model, batch_sizes)

    # Recommend batch size
    recommended_batch_size = profiler.recommend_batch_size(training_results)

    # Compile results
    results = {
        'gpu_name': profiler.gpu_name,
        'gpu_memory_gb': profiler.gpu_total_memory,
        'model_params': model_results,
        'optimizer': optimizer_results,
        'forward_pass': forward_results,
        'backward_pass': backward_results,
        'training_step': training_results,
        'batch_size_comparison': batch_size_results,
        'recommended_batch_size': recommended_batch_size,
    }

    # Generate report
    report = profiler.generate_report(results)
    print(report)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            # Convert to JSON-serializable format
            json_results = {
                k: v for k, v in results.items()
                if k != 'batch_size_comparison' or not batch_size_results
            }
            if batch_size_results:
                json_results['batch_size_comparison'] = {
                    str(k): v for k, v in batch_size_results.items()
                }
            json.dump(json_results, f, indent=2)
        print(f"\n📊 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
```

---

## Usage Examples

### Basic Usage
```bash
# Profile default batch size
python scripts/test_gpu_memory.py --config configs/retfound_lora_config.yaml
```

### Compare Multiple Batch Sizes
```bash
# Test batch sizes 8, 16, 32, 64
python scripts/test_gpu_memory.py \
    --config configs/retfound_lora_config.yaml \
    --batch-sizes 8,16,32,64
```

### Compare RETFound Variants
```bash
# Profile RETFound Large
python scripts/test_gpu_memory.py \
    --config configs/retfound_lora_config.yaml \
    --output results/memory_large.json

# Profile RETFound Green
python scripts/test_gpu_memory.py \
    --config configs/retfound_green_lora_config.yaml \
    --output results/memory_green.json

# Compare results
diff results/memory_large.json results/memory_green.json
```

### Save Results for Documentation
```bash
# Generate report and save
python scripts/test_gpu_memory.py \
    --config configs/retfound_lora_config.yaml \
    --output memory_profile.json
```

---

## Expected Output

### Successful Profile
```
GPU: NVIDIA GeForce RTX 3090
Total Memory: 24.00 GB

[1/5] Profiling model parameters...
  Total parameters: 303,775,237 (1,215.1 MB)
  Trainable: 819,205 (3.3 MB)
  Frozen: 302,956,032 (1,211.8 MB)
  Actual GPU memory: 1,224.5 MB

[2/5] Profiling optimizer state...
  Optimizer: AdamW
  Trainable params: 819,205
  Theoretical memory: 6.6 MB (2 states × 4 bytes)
  Actual memory: 1,231.8 MB

[3/5] Profiling forward pass (batch_size=32)...
  Input: torch.Size([32, 3, 224, 224]) (19.3 MB)
  Output: torch.Size([32, 5]) (0.0 MB)
  Peak memory: 6,843.2 MB
  Activation memory: 5,612.1 MB

[4/5] Profiling backward pass (batch_size=32)...
  Batch size: 32
  Peak memory: 8,124.6 MB
  Gradient memory: 3.3 MB

[5/5] Profiling complete training step (batch_size=32)...
  Batch size: 32
  Peak allocated: 8,156.4 MB
  Peak reserved: 8,200.0 MB

============================================================
MEMORY PROFILING REPORT
============================================================

GPU: NVIDIA GeForce RTX 3090
Total Memory: 24.00 GB

Model Parameters:
  Total: 303,775,237 (1,215.1 MB)
  Trainable: 819,205 (3.3 MB)
  Frozen: 302,956,032 (1,211.8 MB)
  Parameter efficiency: 99.7% reduction (LoRA)

Optimizer State:
  Type: AdamW
  Memory: 6.6 MB

Peak Memory (batch_size=32):
  Allocated: 8,156.4 MB
  Reserved: 8,200.0 MB
  GPU Usage: 34.2%

Recommendation:
  ✅ Maximum safe batch_size: 64
  Current config batch_size: 32
  ✅ Config batch_size is safe

Memory Optimization Tips:
  ✅ Memory usage is healthy, no optimization needed

============================================================
```

### OOM Warning
```
[5/5] Profiling complete training step (batch_size=64)...
  Batch size: 64
  Peak allocated: 22,456.3 MB
  Peak reserved: 22,800.0 MB
  ⚠️  Warning: Using 95.0% of GPU memory

Recommendation:
  ✅ Maximum safe batch_size: 32
  Current config batch_size: 64
  ⚠️  Warning: Config batch_size too large, will likely OOM
  💡 Suggestion: Reduce to batch_size=32
```

---

## Batch Size Recommendations

### General Guidelines

**RETFound Large + LoRA (24GB GPU):**
```
batch_size=64:  ~16 GB (safe)
batch_size=32:  ~8 GB (very safe)
batch_size=16:  ~4 GB (conservative)
```

**RETFound Green + LoRA (8GB GPU):**
```
batch_size=32:  ~6 GB (safe)
batch_size=16:  ~3 GB (very safe)
batch_size=8:   ~1.5 GB (conservative)
```

**Full Fine-tuning (RETFound Large, 24GB GPU):**
```
batch_size=16:  ~18 GB (safe)
batch_size=8:   ~9 GB (very safe)
batch_size=4:   ~4.5 GB (conservative)
```

### Gradient Accumulation Alternative

If GPU memory limited, use gradient accumulation:
```python
effective_batch_size = batch_size × accumulation_steps

# Example: simulate batch_size=64 with 16GB GPU
batch_size = 16
accumulation_steps = 4
# Effective batch_size = 64
```

---

## Troubleshooting

### CUDA Out of Memory During Profiling

**Solution:**
```bash
# Start with smaller batch size
python scripts/test_gpu_memory.py \
    --config configs/retfound_lora_config.yaml \
    --batch-sizes 4,8,16
```

### Memory Leak Between Tests

**Solution:** Script already calls `torch.cuda.empty_cache()` between tests, but if issues persist:
```python
# Add to script
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Different Memory Usage Between Profiling and Training

**Expected:** Profiling uses dummy data, training uses actual data + DataLoader overhead
**Solution:** Profiling provides 90% safety margin

---

## Integration with Training

### Update Config Based on Results

```yaml
# After profiling recommends batch_size=32
training:
  batch_size: 32  # Updated from profiling
  accumulation_steps: 1
```

### Enable Mixed Precision (FP16)

```yaml
# If memory constrained
training:
  use_amp: true  # Automatic Mixed Precision
  # Can reduce memory by 40-50%
```

---

## Next Steps

After completing Phase 2:
1. **Document results:** Note recommended batch sizes for different GPUs
2. **Update configs:** Apply recommendations to training configs
3. **Move to Phase 3:** [03-OVERFITTING_TESTS.md](03-OVERFITTING_TESTS.md)

---

## Success Criteria

Phase 2 is complete when:
- ✅ `scripts/test_gpu_memory.py` exists and runs
- ✅ Accurate memory profiling (within 10% of actual)
- ✅ Batch size recommendations prevent OOM
- ✅ Comparison between RETFound variants documented
- ✅ Report generation works

---

**Ready to implement?** Copy the script to `scripts/test_gpu_memory.py` and profile your models!

**Have questions?** Check [99-IMPLEMENTATION_NOTES.md](99-IMPLEMENTATION_NOTES.md) for PyTorch profiling best practices.
