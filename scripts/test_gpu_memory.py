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
    python scripts/test_gpu_memory.py --config configs/retfound_lora_config.yaml --output memory_green.json
    python scripts/test_gpu_memory.py --config configs/retfound_large_lora_config.yaml --output memory_large.json
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, validate_config
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier
from scripts.utils import set_seed


class GPUMemoryProfiler:
    """Profile GPU memory usage during training."""

    def __init__(self, config_path: str, verbose: bool = True):
        """
        Initialize the memory profiler.

        Args:
            config_path: Path to YAML config file
            verbose: Print detailed information
        """
        self.verbose = verbose
        self.config = Config.from_yaml(config_path)
        validate_config(self.config)

        # Check if GPU available
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, skipping GPU memory profiling")
            self.device = None
            self.gpu_name = "CPU"
            self.gpu_total_memory = 0
        else:
            self.device = torch.device("cuda")
            # Get GPU properties
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"GPU: {self.gpu_name}")
                print(f"Total Memory: {self.gpu_total_memory:.2f} GB")
                print(f"{'='*60}")

    def reset_memory_stats(self):
        """Reset CUDA memory statistics."""
        if self.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

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

    def profile_model_memory(self) -> Dict:
        """Profile memory used by model parameters."""
        if self.verbose:
            print("\n[1/5] Profiling model parameters...")

        if not self.device:
            return {}

        self.reset_memory_stats()
        set_seed(self.config.system.seed)

        # Create model
        is_lora = hasattr(self.config.model, 'lora_r') and self.config.model.lora_r is not None

        if is_lora:
            model = RETFoundLoRA(
                checkpoint_path=self.config.model.pretrained_path,
                num_classes=self.config.model.num_classes,
                model_variant=self.config.model.model_variant,
                lora_r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                head_dropout=self.config.model.head_dropout,
                device=self.device
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

        if self.verbose:
            print(f"  Total parameters: {total_params:,} ({theoretical_mb:.1f} MB)")
            print(f"  Trainable: {trainable_params:,} ({trainable_mb:.1f} MB)")
            print(f"  Frozen: {frozen_params:,} ({frozen_mb:.1f} MB)")
            print(f"  Actual GPU memory: {stats['allocated_mb']:.1f} MB")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'theoretical_mb': theoretical_mb,
            'trainable_mb': trainable_mb,
            'frozen_mb': frozen_mb,
            'actual_mb': stats['allocated_mb'],
            'model': model,  # Keep for next tests
            'is_lora': is_lora,
        }

    def profile_optimizer_memory(self, model) -> Dict:
        """Profile memory used by optimizer state."""
        if self.verbose:
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

        if self.verbose:
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

    def profile_forward_pass(self, model, batch_size: int) -> Dict:
        """Profile memory during forward pass."""
        if self.verbose:
            print(f"\n[3/5] Profiling forward pass (batch_size={batch_size})...")

        if not self.device:
            return {}

        self.reset_memory_stats()
        model.eval()

        # Create dummy batch
        img_size = self.config.image.img_size
        dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=self.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_images)

        stats = self.get_memory_stats()

        # Calculate activation memory (approximate)
        input_mb = dummy_images.numel() * 4 / 1e6
        output_mb = outputs.numel() * 4 / 1e6
        activation_mb = max(0, stats['peak_allocated_mb'] - input_mb)

        if self.verbose:
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

    def profile_backward_pass(self, model, batch_size: int) -> Dict:
        """Profile memory during backward pass."""
        if self.verbose:
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
        img_size = self.config.image.img_size
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

        if self.verbose:
            print(f"  Batch size: {batch_size}")
            print(f"  Peak memory: {stats['peak_allocated_mb']:.1f} MB")
            print(f"  Gradient memory: {gradient_mb:.1f} MB")

        return {
            'batch_size': batch_size,
            'peak_mb': stats['peak_allocated_mb'],
            'gradient_mb': gradient_mb,
        }

    def profile_training_step(self, model, batch_size: int) -> Dict:
        """Profile complete training step (most realistic)."""
        if self.verbose:
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
        img_size = self.config.image.img_size

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

        if self.verbose:
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

        # Get peak memory for the current batch size
        peak_per_sample = memory_profile['peak_reserved_mb'] / memory_profile['batch_size']

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
        if self.verbose:
            print("\n" + "=" * 60)
            print("BATCH SIZE COMPARISON")
            print("=" * 60)

        results = {}

        for batch_size in batch_sizes:
            try:
                if self.verbose:
                    print(f"\nTesting batch_size={batch_size}...")
                self.reset_memory_stats()

                profile = self.profile_training_step(model, batch_size)
                results[batch_size] = profile

                # Check if approaching memory limit
                memory_usage_pct = profile['peak_reserved_mb'] / (self.gpu_total_memory * 1000) * 100
                if memory_usage_pct > 90:
                    if self.verbose:
                        print(f"  ⚠️  Warning: Using {memory_usage_pct:.1f}% of GPU memory")
                    break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.verbose:
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
            report.append(f"  Trainable: {mp['trainable_params']:,} ({mp['trainable_mb']:.1f} MB)")
            report.append(f"  Frozen: {mp['frozen_params']:,} ({mp['frozen_mb']:.1f} MB)")

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
            if bsc:
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
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )

    args = parser.parse_args()

    # Create profiler
    profiler = GPUMemoryProfiler(args.config, verbose=args.verbose)

    if not profiler.device:
        print("❌ No GPU available, exiting")
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        json_results = {
            k: v for k, v in results.items()
            if k != 'batch_size_comparison' or not batch_size_results
        }
        if batch_size_results:
            json_results['batch_size_comparison'] = {
                str(k): v for k, v in batch_size_results.items()
            }

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n📊 Results saved to: {output_path}")

    sys.exit(0)


if __name__ == '__main__':
    main()
