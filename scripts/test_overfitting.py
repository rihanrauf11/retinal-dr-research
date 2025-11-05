#!/usr/bin/env python3
"""
Overfitting test - verify model can learn on tiny dataset.

This test trains a model to overfit on a small dataset (default 10 samples)
to verify that the training loop works correctly and the model has learning capacity.

Usage:
    python scripts/test_overfitting.py --config configs/retfound_lora_config.yaml
    python scripts/test_overfitting.py --config configs/retfound_lora_config.yaml --num-samples 20
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, validate_config
from scripts.dataset import RetinalDataset
from scripts.train_retfound_lora import get_transforms
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier
from scripts.utils import set_seed


class OverfittingTester:
    """Test model's ability to overfit small dataset."""

    def __init__(self, config_path: str, num_samples: int = 10, verbose: bool = True):
        """
        Initialize the overfitting tester.

        Args:
            config_path: Path to YAML config file
            num_samples: Number of samples to use for overfitting (default: 10)
            verbose: Print detailed information
        """
        self.verbose = verbose
        self.num_samples = num_samples
        self.config = Config.from_yaml(config_path)
        validate_config(self.config)

        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        set_seed(self.config.system.seed)

    def create_tiny_dataset(self) -> DataLoader:
        """
        Create dataset with only num_samples images.

        Returns balanced samples (equal number per class if possible).
        """
        if self.verbose:
            print(f"\n[1/3] Creating tiny dataset ({self.num_samples} samples)...")

        # Load full dataset
        train_transform, _ = get_transforms(
            img_size=self.config.image.input_size,
            model_variant=self.config.model.model_variant
        )

        full_dataset = RetinalDataset(
            csv_file=self.config.data.train_csv,
            img_dir=self.config.data.img_dir,
            transform=train_transform
        )

        # Select balanced samples (equal per class if possible)
        num_classes = self.config.model.num_classes
        samples_per_class = self.num_samples // num_classes

        indices = []
        class_counts = {i: 0 for i in range(num_classes)}

        # Iterate through dataset and select balanced samples
        for idx in range(len(full_dataset)):
            # Access label from dataframe
            label = int(full_dataset.data_frame.loc[full_dataset.data_frame.index[idx], 'diagnosis'])

            if class_counts[label] < samples_per_class:
                indices.append(idx)
                class_counts[label] += 1

            if len(indices) >= self.num_samples:
                break

        # Create subset
        tiny_dataset = Subset(full_dataset, indices)

        if self.verbose:
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
        if self.verbose:
            print("\n[2/3] Creating model...")

        # Detect if LoRA model
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

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.verbose:
            print(f"  ✓ Model created: {type(model).__name__}")
            print(f"  ✓ Trainable parameters: {trainable_params:,}")

        return model

    def train_to_overfit(self, model, dataloader) -> Dict:
        """
        Train model until it overfits (100% accuracy).

        Args:
            model: Model to train
            dataloader: DataLoader with tiny dataset

        Returns:
            Dictionary with training results
        """
        if self.verbose:
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
                # Handle both tuple and dict formats
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    # Tuple format: (images, labels)
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

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
                if self.verbose:
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

    def plot_training_curves(self, history: Dict, output_path: str = None):
        """
        Plot loss and accuracy curves.

        Args:
            history: Training history with 'loss' and 'accuracy' keys
            output_path: Path to save the plot (optional)
        """
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
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"\n  📊 Training curves saved to: {output_path}")

        plt.close()

    def run_test(self) -> bool:
        """
        Run complete overfitting test.

        Returns:
            True if test passed, False otherwise
        """
        if self.verbose:
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
        output_dir = Path(self.config.paths.output_dir)
        plot_path = output_dir / "overfitting_test_curves.png"
        self.plot_training_curves(results['history'], str(plot_path))

        # Summary
        if self.verbose:
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

        return results['success']


def main():
    parser = argparse.ArgumentParser(description="Test model overfitting capability")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to overfit (default: 10)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )

    args = parser.parse_args()

    tester = OverfittingTester(args.config, num_samples=args.num_samples, verbose=args.verbose)
    success = tester.run_test()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
