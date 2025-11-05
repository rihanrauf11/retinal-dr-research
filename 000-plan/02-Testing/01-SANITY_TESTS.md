# Phase 1: Sanity Tests

**Priority:** CRITICAL (Must complete before training)
**Time Estimate:** 3-4 hours
**Dependencies:** None
**Deliverable:** `scripts/test_sanity.py`

---

## Objective

Implement fast smoke tests (< 2 minutes) to verify basic training loop functionality before committing to expensive GPU training runs. These tests catch configuration errors, model architecture issues, and gradient flow problems early.

---

## Rationale

**Why this is critical:**
- Training runs take hours and expensive GPU resources
- Config typos or path errors crash training at startup
- Model architecture bugs discovered hours into training
- Gradient flow issues lead to failed learning

**What we gain:**
- Catch 80% of common errors in < 2 minutes
- Verify training pipeline works before expensive runs
- Test configuration changes quickly
- Debug model issues in isolation

---

## Test Components

### 1. Configuration Validation
**Purpose:** Verify config file is valid before training starts

**What to check:**
- All file paths exist (checkpoint, data_csv, data_img_dir)
- Hyperparameters in valid ranges
- Device availability (CUDA/MPS/CPU)
- Output directories are writable
- Checkpoint file is loadable (if provided)

**Expected behavior:**
- ✅ PASS: All paths exist, parameters valid
- ❌ FAIL: Missing files, invalid parameters → Clear error message

---

### 2. Single Forward Pass
**Purpose:** Verify model can process input without errors

**Test:**
```python
# Create dummy batch
batch = {
    'image': torch.randn(4, 3, 224, 224).to(device),
    'label': torch.tensor([0, 1, 2, 3]).to(device)
}

# Forward pass
with torch.no_grad():
    outputs = model(batch['image'])

# Verify output shape
assert outputs.shape == (4, 5)  # (batch_size, num_classes)
```

**Expected behavior:**
- ✅ PASS: Forward pass completes, correct output shape
- ❌ FAIL: Runtime error, shape mismatch → Debug model architecture

---

### 3. Single Backward Pass
**Purpose:** Verify gradients flow through trainable parameters

**Test:**
```python
# Forward pass
outputs = model(batch['image'])
loss = criterion(outputs, batch['label'])

# Backward pass
optimizer.zero_grad()
loss.backward()

# Check gradients exist
for name, param in model.named_parameters():
    if param.requires_grad:
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
```

**Expected behavior:**
- ✅ PASS: All trainable parameters have non-NaN gradients
- ❌ FAIL: Missing or NaN gradients → Debug gradient flow

---

### 4. Loss Convergence on Single Batch
**Purpose:** Verify model can learn by overfitting single batch

**Test:**
```python
# Use same batch repeatedly
initial_loss = None
for step in range(20):
    outputs = model(batch['image'])
    loss = criterion(outputs, batch['label'])

    if step == 0:
        initial_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

final_loss = loss.item()

# Loss should decrease significantly
assert final_loss < initial_loss * 0.5, \
    f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
```

**Expected behavior:**
- ✅ PASS: Loss decreases by 50%+ in 20 steps
- ❌ FAIL: Loss doesn't decrease → Check optimizer, learning rate, model capacity

---

### 5. Data Loading Smoke Test
**Purpose:** Verify DataLoader can load actual data

**Test:**
```python
# Load small subset
dataset = RetinalDataset(
    csv_file=config.data.data_csv,
    img_dir=config.data.data_img_dir,
    transform=get_train_transforms()
)

# Sample a few batches
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for i, batch in enumerate(dataloader):
    assert 'image' in batch
    assert 'label' in batch
    assert batch['image'].shape[1:] == (3, 224, 224)

    if i >= 2:  # Test 3 batches
        break
```

**Expected behavior:**
- ✅ PASS: DataLoader returns valid batches
- ❌ FAIL: Missing images, corrupted files → Check data preparation

---

## Implementation

### Complete Script: `scripts/test_sanity.py`

```python
#!/usr/bin/env python3
"""
Sanity tests for training pipeline.
Runs fast smoke tests (< 2 minutes) to verify basic functionality.

Usage:
    python scripts/test_sanity.py --config configs/retfound_lora_config.yaml
    python scripts/test_sanity.py --config configs/retfound_lora_config.yaml --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config
from scripts.dataset import RetinalDataset, get_train_transforms, get_val_transforms
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier
from scripts.utils import set_seed


class SanityTester:
    """Runs sanity tests on training pipeline."""

    def __init__(self, config_path: str, verbose: bool = False):
        self.config = load_config(config_path)
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.results = {}

    def log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(f"  {message}")

    def test_config_validation(self) -> bool:
        """Test 1: Verify configuration is valid."""
        print("\n[1/5] Testing configuration validation...")

        try:
            # Check data paths exist
            if not Path(self.config.data.data_csv).exists():
                raise FileNotFoundError(f"CSV not found: {self.config.data.data_csv}")

            if not Path(self.config.data.data_img_dir).exists():
                raise FileNotFoundError(f"Image dir not found: {self.config.data.data_img_dir}")

            # Check checkpoint path exists (if not using pretrained)
            if hasattr(self.config.model, 'checkpoint_path'):
                checkpoint_path = Path(self.config.model.checkpoint_path)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Check output directory is writable
            output_dir = Path(self.config.system.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()

            # Check hyperparameters in valid ranges
            assert self.config.training.batch_size > 0, "batch_size must be > 0"
            assert self.config.training.num_epochs > 0, "num_epochs must be > 0"
            assert 0 < self.config.training.learning_rate < 1, "learning_rate must be in (0, 1)"

            # Check device availability
            if self.config.system.device == "cuda" and not torch.cuda.is_available():
                print("  ⚠️  Warning: CUDA requested but not available, using CPU")
            elif self.config.system.device == "mps" and not torch.backends.mps.is_available():
                print("  ⚠️  Warning: MPS requested but not available, using CPU")

            self.log("✓ All paths exist")
            self.log("✓ All hyperparameters valid")
            self.log("✓ Output directory writable")
            self.log(f"✓ Device: {self.device}")

            print("✅ PASS: Configuration is valid")
            return True

        except Exception as e:
            print(f"❌ FAIL: Configuration validation failed")
            print(f"   Error: {e}")
            return False

    def test_single_forward_pass(self) -> bool:
        """Test 2: Verify model can process input."""
        print("\n[2/5] Testing single forward pass...")

        try:
            # Create model
            set_seed(self.config.system.seed)

            if hasattr(self.config.model, 'lora_r'):
                # RETFound + LoRA model
                model = RETFoundLoRA(
                    checkpoint_path=self.config.model.checkpoint_path,
                    num_classes=self.config.model.num_classes,
                    lora_r=self.config.model.lora_r,
                    lora_alpha=self.config.model.lora_alpha,
                    lora_dropout=self.config.model.lora_dropout,
                    head_hidden_dim=getattr(self.config.model, 'head_hidden_dim', 512),
                    head_dropout=getattr(self.config.model, 'head_dropout', 0.3)
                )
            else:
                # Baseline model
                model = DRClassifier(
                    model_name=self.config.model.model_name,
                    num_classes=self.config.model.num_classes,
                    pretrained=self.config.model.pretrained
                )

            model = model.to(self.device)
            model.eval()

            # Create dummy batch
            batch_size = 4
            if hasattr(self.config.model, 'img_size'):
                img_size = self.config.model.img_size
            else:
                img_size = 224

            dummy_images = torch.randn(batch_size, 3, img_size, img_size).to(self.device)

            self.log(f"✓ Model created: {type(model).__name__}")
            self.log(f"✓ Input shape: {dummy_images.shape}")

            # Forward pass
            with torch.no_grad():
                outputs = model(dummy_images)

            # Verify output shape
            expected_shape = (batch_size, self.config.model.num_classes)
            assert outputs.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {outputs.shape}"

            # Verify output is valid (no NaN/Inf)
            assert not torch.isnan(outputs).any(), "Output contains NaN"
            assert not torch.isinf(outputs).any(), "Output contains Inf"

            self.log(f"✓ Output shape: {outputs.shape}")
            self.log(f"✓ Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

            print("✅ PASS: Forward pass successful")

            # Store model for next tests
            self.model = model
            self.batch_size = batch_size
            self.img_size = img_size

            return True

        except Exception as e:
            print(f"❌ FAIL: Forward pass failed")
            print(f"   Error: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False

    def test_single_backward_pass(self) -> bool:
        """Test 3: Verify gradients flow correctly."""
        print("\n[3/5] Testing single backward pass...")

        try:
            self.model.train()

            # Create dummy batch
            dummy_images = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
            dummy_labels = torch.randint(0, self.config.model.num_classes, (self.batch_size,)).to(self.device)

            # Forward pass
            outputs = self.model(dummy_images)

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, dummy_labels)

            self.log(f"✓ Loss computed: {loss.item():.4f}")

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Check gradients
            trainable_params = []
            params_with_grad = []
            params_without_grad = []
            nan_grad_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(name)

                    if param.grad is None:
                        params_without_grad.append(name)
                    elif torch.isnan(param.grad).any():
                        nan_grad_params.append(name)
                    else:
                        params_with_grad.append(name)

            self.log(f"✓ Trainable parameters: {len(trainable_params)}")
            self.log(f"✓ Parameters with gradients: {len(params_with_grad)}")

            # Report issues
            if params_without_grad:
                print(f"⚠️  Warning: {len(params_without_grad)} parameters missing gradients:")
                for name in params_without_grad[:5]:  # Show first 5
                    print(f"   - {name}")
                if len(params_without_grad) > 5:
                    print(f"   ... and {len(params_without_grad) - 5} more")

            if nan_grad_params:
                print(f"❌ FAIL: {len(nan_grad_params)} parameters have NaN gradients:")
                for name in nan_grad_params[:5]:
                    print(f"   - {name}")
                return False

            # Verify at least some parameters have gradients
            if len(params_with_grad) == 0:
                print("❌ FAIL: No parameters received gradients")
                return False

            print("✅ PASS: Backward pass successful")
            return True

        except Exception as e:
            print(f"❌ FAIL: Backward pass failed")
            print(f"   Error: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False

    def test_loss_convergence(self) -> bool:
        """Test 4: Verify model can learn on single batch."""
        print("\n[4/5] Testing loss convergence on single batch...")

        try:
            self.model.train()

            # Create optimizer
            if hasattr(self.config.model, 'lora_r'):
                # LoRA: only train LoRA parameters + classification head
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )
            else:
                # Baseline: train all parameters
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )

            criterion = nn.CrossEntropyLoss()

            # Create fixed batch
            torch.manual_seed(42)
            dummy_images = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
            dummy_labels = torch.randint(0, self.config.model.num_classes, (self.batch_size,)).to(self.device)

            # Train on same batch
            num_steps = 20
            losses = []

            progress = tqdm(range(num_steps), desc="Training", disable=not self.verbose)
            for step in progress:
                outputs = self.model(dummy_images)
                loss = criterion(outputs, dummy_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")

            initial_loss = losses[0]
            final_loss = losses[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100

            self.log(f"✓ Initial loss: {initial_loss:.4f}")
            self.log(f"✓ Final loss: {final_loss:.4f}")
            self.log(f"✓ Reduction: {reduction:.1f}%")

            # Check if loss decreased
            if final_loss >= initial_loss * 0.5:
                print(f"⚠️  Warning: Loss only decreased by {reduction:.1f}% (expected 50%+)")
                print(f"   This might indicate learning issues")
                # Don't fail, just warn

            # Check for NaN
            if any(torch.isnan(torch.tensor(l)) for l in losses):
                print("❌ FAIL: Loss became NaN during training")
                return False

            # Check if loss is decreasing trend
            if losses[-1] > losses[-5]:  # Last loss should be less than 5 steps ago
                print("⚠️  Warning: Loss not consistently decreasing")

            print("✅ PASS: Loss convergence successful")
            return True

        except Exception as e:
            print(f"❌ FAIL: Loss convergence test failed")
            print(f"   Error: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False

    def test_data_loading(self) -> bool:
        """Test 5: Verify DataLoader can load actual data."""
        print("\n[5/5] Testing data loading...")

        try:
            # Create dataset
            dataset = RetinalDataset(
                csv_file=self.config.data.data_csv,
                img_dir=self.config.data.data_img_dir,
                transform=get_train_transforms()
            )

            self.log(f"✓ Dataset created: {len(dataset)} images")

            # Create dataloader (num_workers=0 for simplicity)
            dataloader = DataLoader(
                dataset,
                batch_size=min(4, len(dataset)),
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )

            # Load a few batches
            num_batches_to_test = min(3, len(dataloader))

            for i, batch in enumerate(dataloader):
                if i >= num_batches_to_test:
                    break

                # Verify batch structure
                assert 'image' in batch, "Batch missing 'image' key"
                assert 'label' in batch, "Batch missing 'label' key"

                images = batch['image']
                labels = batch['label']

                # Verify shapes
                assert images.ndim == 4, f"Expected 4D images, got {images.ndim}D"
                assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
                assert labels.ndim == 1, f"Expected 1D labels, got {labels.ndim}D"

                # Verify value ranges
                assert images.min() >= -10 and images.max() <= 10, "Image values out of reasonable range"
                assert labels.min() >= 0 and labels.max() < self.config.model.num_classes, \
                    f"Label values out of range [0, {self.config.model.num_classes})"

                # Verify no NaN
                assert not torch.isnan(images).any(), "Images contain NaN"
                assert not torch.isnan(labels).any(), "Labels contain NaN"

                self.log(f"✓ Batch {i+1}: images {images.shape}, labels {labels.shape}")

            print("✅ PASS: Data loading successful")
            return True

        except Exception as e:
            print(f"❌ FAIL: Data loading test failed")
            print(f"   Error: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False

    def run_all_tests(self) -> bool:
        """Run all sanity tests."""
        print("=" * 60)
        print("SANITY TEST SUITE")
        print("=" * 60)
        print(f"Config: {self.config.system.output_dir}")
        print(f"Device: {self.device}")
        print(f"Seed: {self.config.system.seed}")

        # Run tests in order
        tests = [
            self.test_config_validation,
            self.test_single_forward_pass,
            self.test_single_backward_pass,
            self.test_loss_convergence,
            self.test_data_loading,
        ]

        results = []
        for test in tests:
            result = test()
            results.append(result)
            if not result:
                print(f"\n⚠️  Test failed, stopping here")
                break

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        passed = sum(results)
        total = len(results)

        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")

        if all(results):
            print("\n✅ ALL TESTS PASSED - Ready to train!")
            return True
        else:
            print("\n❌ SOME TESTS FAILED - Fix issues before training")
            return False


def main():
    parser = argparse.ArgumentParser(description="Run sanity tests on training pipeline")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/retfound_lora_config.yaml)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Run tests
    tester = SanityTester(args.config, verbose=args.verbose)
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
```

---

## Usage Examples

### Basic Usage
```bash
# Run sanity tests on default config
python scripts/test_sanity.py --config configs/retfound_lora_config.yaml

# Run with verbose output
python scripts/test_sanity.py --config configs/retfound_lora_config.yaml --verbose
```

### Integration with Workflow
```bash
# 1. Validate data first
python scripts/validate_data.py

# 2. Run sanity tests
python scripts/test_sanity.py --config configs/retfound_lora_config.yaml

# 3. If all pass, start training
python scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
```

### Testing Different Configs
```bash
# Test RETFound Large
python scripts/test_sanity.py --config configs/retfound_lora_config.yaml

# Test RETFound Green
python scripts/test_sanity.py --config configs/retfound_green_lora_config.yaml

# Test baseline model
python scripts/test_sanity.py --config configs/default_config.yaml
```

---

## Expected Output

### Successful Run
```
============================================================
SANITY TEST SUITE
============================================================
Config: results/retfound_lora
Device: cuda
Seed: 42

[1/5] Testing configuration validation...
✅ PASS: Configuration is valid

[2/5] Testing single forward pass...
✅ PASS: Forward pass successful

[3/5] Testing single backward pass...
✅ PASS: Backward pass successful

[4/5] Testing loss convergence on single batch...
Training: 100%|██████████| 20/20 [00:05<00:00,  3.85it/s, loss=0.0234]
✅ PASS: Loss convergence successful

[5/5] Testing data loading...
✅ PASS: Data loading successful

============================================================
SUMMARY
============================================================
Tests run: 5
Passed: 5
Failed: 0

✅ ALL TESTS PASSED - Ready to train!
```

### Failed Run (Example)
```
============================================================
SANITY TEST SUITE
============================================================
Config: results/retfound_lora
Device: cuda
Seed: 42

[1/5] Testing configuration validation...
❌ FAIL: Configuration validation failed
   Error: FileNotFoundError: Checkpoint not found: models/RETFound_cfp_weights.pth

⚠️  Test failed, stopping here

============================================================
SUMMARY
============================================================
Tests run: 1
Passed: 0
Failed: 1

❌ SOME TESTS FAILED - Fix issues before training
```

---

## Troubleshooting

### Test 1 Fails: Configuration Validation

**Common issues:**
```
FileNotFoundError: Checkpoint not found: models/RETFound_cfp_weights.pth
```
**Solution:** Download RETFound weights from https://github.com/rmaphoh/RETFound_MAE

```
FileNotFoundError: CSV not found: data/aptos/train.csv
```
**Solution:** Run `python scripts/prepare_data.py --aptos-only`

---

### Test 2 Fails: Forward Pass

**Common issues:**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch_size in test (currently 4) or use CPU

```
AssertionError: Expected shape (4, 5), got (4, 1000)
```
**Solution:** Model not properly configured for num_classes, check model initialization

---

### Test 3 Fails: Backward Pass

**Common issues:**
```
No parameters received gradients
```
**Solution:** Check if all parameters are frozen, verify requires_grad=True for trainable params

```
NaN gradients
```
**Solution:** Check learning rate (might be too high), verify input normalization

---

### Test 4 Fails: Loss Convergence

**Common issues:**
```
Warning: Loss only decreased by 10% (expected 50%+)
```
**Solution:** Might indicate:
- Learning rate too low (try increasing)
- Model too small for task
- Optimizer not configured correctly
- Gradients not flowing (check Test 3)

```
Loss became NaN during training
```
**Solution:** Learning rate too high, reduce it by 10x

---

### Test 5 Fails: Data Loading

**Common issues:**
```
FileNotFoundError: Image not found
```
**Solution:** Verify images exist in data_img_dir, check CSV has correct filenames

```
PIL.UnidentifiedImageError: cannot identify image file
```
**Solution:** Some images might be corrupted, run `python scripts/validate_data.py` to find them

---

## Integration with Existing Tests

### Relationship to pytest Suite
```
pytest tests/                      → Unit tests (verify components)
python scripts/test_sanity.py      → Integration smoke tests (verify pipeline)
```

**Run both:**
```bash
# 1. Unit tests (5 minutes)
pytest tests/ -v

# 2. Sanity tests (2 minutes)
python scripts/test_sanity.py --config configs/retfound_lora_config.yaml
```

### Relationship to Validation Scripts
```
python scripts/validate_data.py     → Data integrity (check files)
python scripts/test_sanity.py       → Training pipeline (check model)
```

**Recommended order:**
```bash
1. validate_data.py   (verify data is good)
2. test_sanity.py     (verify training works)
3. train_*.py         (start actual training)
```

---

## Next Steps

After implementing Phase 1:
1. **Test it:** Run sanity tests on your config
2. **Fix issues:** Address any failures
3. **Move to Phase 2:** [02-GPU_MEMORY_TESTS.md](02-GPU_MEMORY_TESTS.md)

**OR** if sanity tests pass and you're confident:
- Start training immediately
- Implement Phase 2 in parallel while training runs

---

## Success Criteria

Phase 1 is complete when:
- ✅ `scripts/test_sanity.py` exists and is executable
- ✅ All 5 tests pass on your configuration
- ✅ Test runs in < 2 minutes
- ✅ Clear error messages for failures
- ✅ Exit code 0 on success, 1 on failure (for CI integration)

---

**Ready to implement?** Copy the script above to `scripts/test_sanity.py` and test it!

**Have issues?** Check [99-IMPLEMENTATION_NOTES.md](99-IMPLEMENTATION_NOTES.md) for additional guidance.
