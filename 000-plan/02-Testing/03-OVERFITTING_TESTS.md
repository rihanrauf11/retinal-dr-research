# Phase 3: Overfitting Tests

**Priority:** HIGH (Complete before training)
**Time Estimate:** 3-4 hours
**Dependencies:** Phase 1 complete
**Deliverable:** `scripts/test_overfitting.py`

---

## Objective

Verify model can learn by intentionally overfitting on a tiny dataset (10 samples). This confirms the training loop works correctly before investing hours in full training.

---

## Rationale

**Why this matters:**
- Spending 5 hours training only to discover model can't learn
- Unclear if poor performance is data quality or model issue
- No confidence that optimizer/scheduler are configured correctly
- Can't distinguish between "model is learning slowly" vs "model can't learn"

**What we gain:**
- Proof that model has learning capacity
- Verification that loss converges to near-zero
- Confidence that gradients flow correctly
- Quick debugging (5 minutes vs 5 hours)

**The principle:** If a model can't overfit 10 samples, it won't generalize to 10,000.

---

## Test Strategy

### Create Tiny Dataset
```python
# Select 10 samples (2 per class)
# Class distribution: [2, 2, 2, 2, 2] for 5 classes
# Train until 100% accuracy achieved
```

### Expected Behavior
```
Initial accuracy: ~20% (random)
After 20 epochs: 100% accuracy
Final loss: < 0.01
```

### Success Criteria
- Loss decreases monotonically
- Reaches near-zero loss (< 0.01)
- Achieves 100% training accuracy
- Completes in < 5 minutes

---

## Implementation

```python
#!/usr/bin/env python3
"""
Overfitting test - verify model can learn on tiny dataset.

Usage:
    python scripts/test_overfitting.py --config configs/retfound_lora_config.yaml
    python scripts/test_overfitting.py --config configs/retfound_lora_config.yaml --num-samples 20
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config
from scripts.dataset import RetinalDataset, get_train_transforms
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier
from scripts.utils import set_seed


class OverfittingTester:
    """Test model's ability to overfit small dataset."""

    def __init__(self, config_path: str, num_samples: int = 10):
        self.config = load_config(config_path)
        self.num_samples = num_samples
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        set_seed(self.config.system.seed)

    def create_tiny_dataset(self) -> DataLoader:
        """Create dataset with only num_samples images."""
        print(f"\n[1/3] Creating tiny dataset ({self.num_samples} samples)...")

        # Load full dataset
        full_dataset = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_train_transforms()
        )

        # Select balanced samples (equal per class if possible)
        num_classes = self.config.model.num_classes
        samples_per_class = self.num_samples // num_classes

        indices = []
        class_counts = {i: 0 for i in range(num_classes)}

        for idx in range(len(full_dataset)):
            label = full_dataset.labels[idx]
            if class_counts[label] < samples_per_class:
                indices.append(idx)
                class_counts[label] += 1

            if len(indices) >= self.num_samples:
                break

        # Create subset
        tiny_dataset = Subset(full_dataset, indices)

        print(f"  ✓ Selected {len(tiny_dataset)} samples")
        print(f"  ✓ Class distribution: {dict(class_counts)}")

        # Create dataloader (no shuffle for reproducibility)
        dataloader = DataLoader(
            tiny_dataset,
            batch_size=min(4, len(tiny_dataset)),
            shuffle=False,  # Fixed order for reproducibility
            num_workers=0
        )

        return dataloader

    def create_model(self):
        """Create model for training."""
        print("\n[2/3] Creating model...")

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

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Model created: {type(model).__name__}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")

        return model

    def train_to_overfit(self, model, dataloader) -> dict:
        """Train model until it overfits (100% accuracy)."""
        print("\n[3/3] Training to overfit...")

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate * 2,  # Higher LR for faster overfitting
            weight_decay=0.0  # No regularization
        )

        # Training loop
        max_epochs = 50
        history = {
            'loss': [],
            'accuracy': [],
        }

        model.train()
        best_accuracy = 0.0

        progress = tqdm(range(max_epochs), desc="Overfitting")

        for epoch in progress:
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in dataloader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100.0 * correct / total

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.1f}%")

            # Check convergence
            if accuracy == 100.0 and avg_loss < 0.01:
                print(f"\n  ✓ Converged at epoch {epoch + 1}")
                print(f"  ✓ Final loss: {avg_loss:.6f}")
                print(f"  ✓ Final accuracy: {accuracy:.1f}%")
                best_accuracy = accuracy
                break

            best_accuracy = max(best_accuracy, accuracy)

        # Check if overfitting succeeded
        success = (best_accuracy == 100.0 and history['loss'][-1] < 0.1)

        return {
            'success': success,
            'epochs': len(history['loss']),
            'final_loss': history['loss'][-1],
            'final_accuracy': history['accuracy'][-1],
            'best_accuracy': best_accuracy,
            'history': history,
        }

    def plot_training_curves(self, history: dict, output_path: str = None):
        """Plot loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        ax1.plot(history['loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2.plot(history['accuracy'], 'g-', linewidth=2)
        ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Target: 100%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n  📊 Training curves saved to: {output_path}")

        plt.close()

    def run_test(self) -> bool:
        """Run complete overfitting test."""
        print("=" * 60)
        print("OVERFITTING TEST")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Target: {self.num_samples} samples, 100% accuracy, loss < 0.01")

        # Create tiny dataset
        dataloader = self.create_tiny_dataset()

        # Create model
        model = self.create_model()

        # Train to overfit
        results = self.train_to_overfit(model, dataloader)

        # Plot curves
        output_dir = Path(self.config.system.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "overfitting_test_curves.png"
        self.plot_training_curves(results['history'], str(plot_path))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Epochs trained: {results['epochs']}")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Final accuracy: {results['final_accuracy']:.1f}%")
        print(f"Best accuracy: {results['best_accuracy']:.1f}%")

        if results['success']:
            print("\n✅ SUCCESS: Model can overfit tiny dataset")
            print("   → Training loop works correctly")
            print("   → Model has learning capacity")
            print("   → Ready for full training")
            return True
        else:
            print("\n❌ FAILURE: Model failed to overfit")
            print("   → Check learning rate (might be too low)")
            print("   → Check model architecture")
            print("   → Check gradient flow (run Phase 1 sanity tests)")

            if results['best_accuracy'] < 50:
                print("   → Model barely learning, likely serious issue")
            elif results['best_accuracy'] < 90:
                print("   → Model learning slowly, try higher learning rate")
            else:
                print("   → Almost there, try more epochs or higher LR")

            return False


def main():
    parser = argparse.ArgumentParser(description="Test model overfitting capability")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples (default: 10)')

    args = parser.parse_args()

    tester = OverfittingTester(args.config, num_samples=args.num_samples)
    success = tester.run_test()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
```

---

## Usage

```bash
# Default (10 samples)
python scripts/test_overfitting.py --config configs/retfound_lora_config.yaml

# Larger test (20 samples)
python scripts/test_overfitting.py --config configs/retfound_lora_config.yaml --num-samples 20
```

---

## Expected Output

### Success Case
```
============================================================
OVERFITTING TEST
============================================================
Device: cuda
Target: 10 samples, 100% accuracy, loss < 0.01

[1/3] Creating tiny dataset (10 samples)...
  ✓ Selected 10 samples
  ✓ Class distribution: {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

[2/3] Creating model...
  ✓ Model created: RETFoundLoRA
  ✓ Trainable parameters: 819,205

[3/3] Training to overfit...
Overfitting: 100%|████████| 15/50 [00:03<00:00,  4.2it/s, loss=0.0023, acc=100.0%]

  ✓ Converged at epoch 15
  ✓ Final loss: 0.002341
  ✓ Final accuracy: 100.0%

  📊 Training curves saved to: results/retfound_lora/overfitting_test_curves.png

============================================================
SUMMARY
============================================================
Epochs trained: 15
Final loss: 0.002341
Final accuracy: 100.0%
Best accuracy: 100.0%

✅ SUCCESS: Model can overfit tiny dataset
   → Training loop works correctly
   → Model has learning capacity
   → Ready for full training
```

### Failure Case
```
[3/3] Training to overfit...
Overfitting: 100%|████████| 50/50 [00:10<00:00,  4.8it/s, loss=1.2345, acc=45.0%]

============================================================
SUMMARY
============================================================
Epochs trained: 50
Final loss: 1.234567
Final accuracy: 45.0%
Best accuracy: 45.0%

❌ FAILURE: Model failed to overfit
   → Check learning rate (might be too low)
   → Model barely learning, likely serious issue
```

---

## Troubleshooting

**Q: Model reaches 90% but not 100%**
- Try higher learning rate (2x or 5x)
- Increase max_epochs to 100
- Reduce dataset to 5 samples

**Q: Loss decreases but accuracy stays low**
- Model might be learning, but slowly
- Try 10x higher learning rate
- Check class balance

**Q: Loss becomes NaN**
- Learning rate too high
- Reduce by 10x

---

## Next Steps

After Phase 3:
1. **If test passes:** Move to full training or Phase 4
2. **If test fails:** Debug before proceeding
3. **Optional:** Proceed to [04-DATA_LOADING_TESTS.md](04-DATA_LOADING_TESTS.md)

---

## Success Criteria

- ✅ Script exists and runs
- ✅ Achieves 100% accuracy
- ✅ Loss < 0.01
- ✅ Completes in < 10 minutes
- ✅ Generates training curves

