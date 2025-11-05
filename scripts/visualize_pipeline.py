#!/usr/bin/env python3
"""
Visual inspection tools for data pipeline and model predictions.

Create visual debugging tools to inspect data augmentation, model predictions,
and training progress. Helps catch subtle issues that numeric metrics miss.

Usage:
    # Visualize augmentations
    python scripts/visualize_pipeline.py --config configs/retfound_lora_config.yaml --mode augmentation

    # Visualize batch samples
    python scripts/visualize_pipeline.py --config configs/retfound_lora_config.yaml --mode batch

    # Visualize predictions (requires trained model)
    python scripts/visualize_pipeline.py \
        --config configs/retfound_lora_config.yaml \
        --mode predictions \
        --checkpoint results/retfound_lora/checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, validate_config
from scripts.dataset import RetinalDataset
from scripts.train_retfound_lora import get_transforms
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier


class PipelineVisualizer:
    """Visualize data pipeline and model predictions."""

    def __init__(self, config_path: str, verbose: bool = True):
        """
        Initialize the pipeline visualizer.

        Args:
            config_path: Path to YAML config file
            verbose: Print detailed information
        """
        self.verbose = verbose
        self.config = Config.from_yaml(config_path)
        validate_config(self.config)

        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Class names for Diabetic Retinopathy
        self.class_names = [
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative"
        ]

        if self.verbose:
            print(f"\nDevice: {self.device}")

    def visualize_augmentation(self, num_examples: int = 3, aug_per_image: int = 6):
        """
        Show multiple augmentations of same images.

        Args:
            num_examples: Number of different images to show
            aug_per_image: Number of augmented versions per image
        """
        if self.verbose:
            print(f"\n[1/1] Generating augmentation gallery ({num_examples} examples)...")

        # Get transforms (returns tuple!)
        train_transform, val_transform = get_transforms(
            img_size=self.config.image.img_size,
            model_variant=self.config.model.model_variant
        )

        # Load dataset with augmentation
        dataset = RetinalDataset(
            csv_file=self.config.data.train_csv,
            img_dir=self.config.data.img_dir,
            transform=train_transform
        )

        if len(dataset) == 0:
            print("❌ Error: Dataset is empty")
            return

        # Select random images
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)

        fig, axes = plt.subplots(len(indices), aug_per_image + 1,
                                figsize=(3 * (aug_per_image + 1), 3 * len(indices)))

        if len(indices) == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices):
            # Get original image (no augmentation)
            dataset_no_aug = RetinalDataset(
                csv_file=self.config.data.train_csv,
                img_dir=self.config.data.img_dir,
                transform=val_transform
            )
            orig_image, label = dataset_no_aug[idx]  # Tuple unpacking
            orig_img = self._tensor_to_image(orig_image)

            # Show original
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original\n{self.class_names[label]}", fontsize=10)
            axes[i, 0].axis('off')

            # Show augmentations
            for j in range(aug_per_image):
                aug_image, _ = dataset[idx]  # Tuple unpacking
                aug_img = self._tensor_to_image(aug_image)

                axes[i, j + 1].imshow(aug_img)
                axes[i, j + 1].set_title(f"Aug {j+1}", fontsize=10)
                axes[i, j + 1].axis('off')

        plt.suptitle("Augmentation Gallery", fontsize=16, y=0.995)
        plt.tight_layout()

        output_path = Path(self.config.paths.output_dir) / "augmentation_gallery.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if self.verbose:
            print(f"✅ Saved to: {output_path}")

        plt.close()

    def visualize_batch(self, num_batches: int = 2):
        """
        Visualize random training batches.

        Args:
            num_batches: Number of batches to visualize
        """
        if self.verbose:
            print(f"\n[1/1] Visualizing {num_batches} training batches...")

        # Get transforms
        train_transform, _ = get_transforms(
            img_size=self.config.image.img_size,
            model_variant=self.config.model.model_variant
        )

        dataset = RetinalDataset(
            csv_file=self.config.data.train_csv,
            img_dir=self.config.data.img_dir,
            transform=train_transform
        )

        if len(dataset) == 0:
            print("❌ Error: Dataset is empty")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=min(16, len(dataset)),
            shuffle=True,
            num_workers=0
        )

        for batch_idx, (images, labels) in enumerate(dataloader):  # Tuple unpacking
            if batch_idx >= num_batches:
                break

            # Create grid
            grid_size = int(np.ceil(np.sqrt(len(images))))
            fig, axes = plt.subplots(grid_size, grid_size,
                                    figsize=(3 * grid_size, 3 * grid_size))
            axes = axes.flatten()

            for i in range(len(images)):
                img = self._tensor_to_image(images[i])
                label = labels[i].item()  # Tensor to Python int

                axes[i].imshow(img)
                axes[i].set_title(f"{self.class_names[label]}", fontsize=10)
                axes[i].axis('off')

            # Hide unused subplots
            for i in range(len(images), len(axes)):
                axes[i].axis('off')

            plt.suptitle(f"Training Batch {batch_idx + 1}", fontsize=16)
            plt.tight_layout()

            output_path = Path(self.config.paths.output_dir) / f"batch_{batch_idx+1}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

            if self.verbose:
                print(f"✅ Saved to: {output_path}")

            plt.close()

    def visualize_predictions(self, checkpoint_path: str, num_samples: int = 16):
        """
        Visualize model predictions vs ground truth.

        Args:
            checkpoint_path: Path to trained model checkpoint
            num_samples: Number of samples to visualize
        """
        if self.verbose:
            print(f"\n[1/2] Loading model from checkpoint...")

        # Load model
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
                pretrained=False,
            )

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model = model.to(self.device)
            model.eval()
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return

        if self.verbose:
            print("✅ Model loaded")

        # Get predictions
        if self.verbose:
            print(f"[2/2] Generating predictions ({num_samples} samples)...")

        # Get validation transforms
        _, val_transform = get_transforms(
            img_size=self.config.image.img_size,
            model_variant=self.config.model.model_variant
        )

        dataset = RetinalDataset(
            csv_file=self.config.data.train_csv,
            img_dir=self.config.data.img_dir,
            transform=val_transform
        )

        if len(dataset) == 0:
            print("❌ Error: Dataset is empty")
            return

        # Random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        grid_size = int(np.ceil(np.sqrt(len(indices))))
        fig, axes = plt.subplots(grid_size, grid_size,
                                figsize=(4 * grid_size, 4 * grid_size))
        axes = axes.flatten()

        with torch.no_grad():
            for i, idx in enumerate(indices):
                image, label = dataset[idx]  # Tuple unpacking
                image_batch = image.unsqueeze(0).to(self.device)

                # Predict
                outputs = model(image_batch)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1).item()
                confidence = probs[0, pred].item()

                # Visualize
                img = self._tensor_to_image(image)
                axes[i].imshow(img)

                # Color code: green if correct, red if wrong
                color = 'green' if pred == label else 'red'
                axes[i].set_title(
                    f"True: {self.class_names[label]}\n"
                    f"Pred: {self.class_names[pred]} ({confidence:.2f})",
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        plt.suptitle("Model Predictions", fontsize=16)
        plt.tight_layout()

        output_path = Path(self.config.paths.output_dir) / "predictions.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        if self.verbose:
            print(f"✅ Saved to: {output_path}")

        plt.close()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to displayable image with variant-aware denormalization.

        Args:
            tensor: Image tensor (C, H, W)

        Returns:
            NumPy array suitable for imshow (H, W, C)
        """
        # Get normalization from config (variant-dependent)
        mean = np.array(self.config.image.mean)
        std = np.array(self.config.image.std)

        # Denormalize
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        return img


def main():
    parser = argparse.ArgumentParser(description="Visualize data pipeline and predictions")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['augmentation', 'batch', 'predictions'],
        help='Visualization mode'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (required for predictions mode)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=16,
        help='Number of samples to visualize (default: 16)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )

    args = parser.parse_args()

    # Validate predictions mode requirements
    if args.mode == 'predictions' and not args.checkpoint:
        print("❌ Error: --checkpoint required for predictions mode")
        sys.exit(1)

    # Create visualizer
    visualizer = PipelineVisualizer(args.config, verbose=args.verbose)

    # Run visualization
    print("=" * 60)
    print(f"VISUAL INSPECTION: {args.mode.upper()}")
    print("=" * 60)

    try:
        if args.mode == 'augmentation':
            visualizer.visualize_augmentation(num_examples=3, aug_per_image=6)
        elif args.mode == 'batch':
            visualizer.visualize_batch(num_batches=2)
        elif args.mode == 'predictions':
            visualizer.visualize_predictions(args.checkpoint, num_samples=args.num_samples)

        print("\n✅ Visualization complete")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
