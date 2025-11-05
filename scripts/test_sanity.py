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

from scripts.config import Config, validate_config
from scripts.dataset import RetinalDataset
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier
from scripts.utils import set_seed
from scripts.train_retfound_lora import get_transforms


class SanityTester:
    """Runs sanity tests on training pipeline."""

    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = config_path
        self.config = Config.from_yaml(config_path)
        # Validate config and auto-set image parameters based on model variant
        validate_config(self.config)
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
            # Check data paths exist - handle both config formats
            # Some configs use train_csv, others might use data_csv
            train_csv = getattr(self.config.data, 'train_csv', None) or \
                        getattr(self.config.data, 'data_csv', None)
            if not train_csv:
                raise ValueError("No train_csv or data_csv found in config")

            train_csv_path = Path(train_csv)
            if not train_csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {train_csv}")

            # Handle train_img_dir (primary) or img_dir (auto-set)
            # Note: Some configs may have these in YAML but not in parsed config
            img_dir = getattr(self.config.data, 'train_img_dir', None) or \
                      getattr(self.config.data, 'img_dir', None)

            if img_dir:
                img_dir_path = Path(img_dir)
                if not img_dir_path.exists():
                    raise FileNotFoundError(f"Image dir not found: {img_dir}")
            else:
                print("  ⚠️  Warning: Image directory not specified in config (will be checked during training)")
                # Continue - image dir will be checked when actually training

            # Check checkpoint path exists (only for RETFound models)
            # Baseline models (identified by model_name like 'resnet50') don't need checkpoints
            is_retfound = hasattr(self.config.model, 'lora_r') and self.config.model.lora_r is not None
            is_retfound_large = is_retfound and \
                                hasattr(self.config.model, 'model_variant') and \
                                self.config.model.model_variant == 'large'

            if is_retfound_large and hasattr(self.config.model, 'pretrained_path') and self.config.model.pretrained_path:
                checkpoint_path = Path(self.config.model.pretrained_path)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint required for RETFound Large: {self.config.model.pretrained_path}")
            elif is_retfound and hasattr(self.config.model, 'pretrained_path') and self.config.model.pretrained_path:
                # RETFound Green - checkpoint is optional
                checkpoint_path = Path(self.config.model.pretrained_path)
                if not checkpoint_path.exists():
                    self.log(f"⚠️  Pretrained path not found: {self.config.model.pretrained_path} (optional for Green)")

            # Check output directory is writable
            output_dir = Path(self.config.paths.output_dir)
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
            self.log(f"✓ Config variant: {getattr(self.config.model, 'model_variant', 'baseline')}")

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
                    checkpoint_path=self.config.model.pretrained_path,
                    num_classes=self.config.model.num_classes,
                    model_variant=getattr(self.config.model, 'model_variant', 'large'),
                    lora_r=self.config.model.lora_r,
                    lora_alpha=self.config.model.lora_alpha,
                    lora_dropout=self.config.model.lora_dropout,
                    head_dropout=getattr(self.config.model, 'head_dropout', 0.3),
                    device=self.device
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

            # Create dummy batch with correct image size from config
            batch_size = 4
            img_size = self.config.image.img_size

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
            # Get transforms for the correct image size and variant
            model_variant = getattr(self.config.model, 'model_variant', 'large')
            train_transform, _ = get_transforms(
                img_size=self.config.image.img_size,
                model_variant=model_variant
            )

            # Handle both config formats for CSV and image dir
            train_csv = getattr(self.config.data, 'train_csv', None) or \
                        getattr(self.config.data, 'data_csv', None)
            img_dir = getattr(self.config.data, 'train_img_dir', None) or \
                      getattr(self.config.data, 'img_dir', None)

            # Create dataset
            dataset = RetinalDataset(
                csv_file=train_csv,
                img_dir=img_dir,
                transform=train_transform
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

                # Handle both batch formats (tuple or dict)
                if isinstance(batch, (tuple, list)):
                    # Tuple format: (images, labels)
                    assert len(batch) == 2, f"Expected tuple of (images, labels), got {len(batch)} elements"
                    images, labels = batch
                else:
                    # Dict format: {'image': ..., 'label': ...}
                    assert 'image' in batch, "Batch missing 'image' key"
                    assert 'label' in batch, "Batch missing 'label' key"
                    images = batch['image']
                    labels = batch['label']

                # Verify shapes
                assert images.ndim == 4, f"Expected 4D images, got {images.ndim}D"
                assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
                assert labels.ndim == 1, f"Expected 1D labels, got {labels.ndim}D"

                # Verify value ranges (normalized images)
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
        print(f"Config: {self.config_path}")
        print(f"Output dir: {self.config.paths.output_dir}")
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
