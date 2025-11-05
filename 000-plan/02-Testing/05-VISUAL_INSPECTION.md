# Phase 5: Visual Inspection

**Priority:** MEDIUM (Debugging aid)
**Time Estimate:** 5-6 hours
**Dependencies:** Phase 1 complete
**Deliverable:** `scripts/visualize_pipeline.py`

---

## Objective

Create visual debugging tools to inspect data augmentation, model predictions, and training progress. Helps catch subtle issues that numeric metrics miss.

---

## Rationale

**What visual inspection reveals:**
- Augmentation bugs (over-aggressive transforms)
- Data quality issues (corrupted images, wrong labels)
- Model behavior (confidence patterns, failure modes)
- Training anomalies (loss spikes, gradient issues)

**Examples of caught issues:**
- Augmentation making images too dark/bright
- Label mismatches in dataset
- Model confidently wrong on certain classes
- Training curves showing overfitting early

---

## Visualizations to Implement

### 1. Augmentation Gallery
**Show:** Grid of same image with different augmentations
**Purpose:** Verify augmentations are reasonable

### 2. Batch Samples
**Show:** Random training batches with labels
**Purpose:** Verify data loading works correctly

### 3. Prediction Samples
**Show:** Model predictions vs ground truth
**Purpose:** Understand model behavior

### 4. Training Curves
**Show:** Loss, accuracy, learning rate over time
**Purpose:** Monitor training progress

### 5. Confusion Matrix
**Show:** Predicted vs actual class distribution
**Purpose:** Identify systematic errors

---

## Implementation

```python
#!/usr/bin/env python3
"""
Visual inspection tools for data and model.

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
from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config
from scripts.dataset import RetinalDataset, get_train_transforms, get_val_transforms
from scripts.retfound_lora import RETFoundLoRA
from scripts.model import DRClassifier


class PipelineVisualizer:
    """Visualize data pipeline and model predictions."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        # Class names for DR
        self.class_names = [
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative"
        ]

    def visualize_augmentation(self, num_examples: int = 3, aug_per_image: int = 6):
        """Show multiple augmentations of same images."""
        print(f"\n[1/1] Generating augmentation gallery...")

        # Load dataset
        dataset = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_train_transforms()
        )

        # Select random images
        indices = np.random.choice(len(dataset), num_examples, replace=False)

        fig, axes = plt.subplots(num_examples, aug_per_image + 1,
                                 figsize=(3 * (aug_per_image + 1), 3 * num_examples))

        if num_examples == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices):
            # Get original image (no augmentation)
            dataset_no_aug = RetinalDataset(
                csv_file=self.config.data.data_csv,
                img_dir=self.config.data.data_img_dir,
                transform=get_val_transforms()
            )
            orig_batch = dataset_no_aug[idx]
            orig_img = self._tensor_to_image(orig_batch['image'])
            label = orig_batch['label'].item()

            # Show original
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original\n{self.class_names[label]}", fontsize=10)
            axes[i, 0].axis('off')

            # Show augmentations
            for j in range(aug_per_image):
                batch = dataset[idx]
                aug_img = self._tensor_to_image(batch['image'])

                axes[i, j + 1].imshow(aug_img)
                axes[i, j + 1].set_title(f"Aug {j+1}", fontsize=10)
                axes[i, j + 1].axis('off')

        plt.suptitle("Augmentation Gallery", fontsize=16, y=0.995)
        plt.tight_layout()

        output_path = Path(self.config.system.output_dir) / "augmentation_gallery.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")
        plt.close()

    def visualize_batch(self, num_batches: int = 2):
        """Visualize random training batches."""
        print(f"\n[1/1] Visualizing training batches...")

        dataset = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_train_transforms()
        )

        dataloader = DataLoader(
            dataset,
            batch_size=min(16, self.config.training.batch_size),
            shuffle=True,
            num_workers=0
        )

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images = batch['image']
            labels = batch['label']

            # Create grid
            grid_size = int(np.ceil(np.sqrt(len(images))))
            fig, axes = plt.subplots(grid_size, grid_size,
                                     figsize=(3 * grid_size, 3 * grid_size))
            axes = axes.flatten()

            for i in range(len(images)):
                img = self._tensor_to_image(images[i])
                label = labels[i].item()

                axes[i].imshow(img)
                axes[i].set_title(f"{self.class_names[label]}", fontsize=10)
                axes[i].axis('off')

            # Hide unused subplots
            for i in range(len(images), len(axes)):
                axes[i].axis('off')

            plt.suptitle(f"Training Batch {batch_idx + 1}", fontsize=16)
            plt.tight_layout()

            output_path = Path(self.config.system.output_dir) / f"batch_{batch_idx+1}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to: {output_path}")
            plt.close()

    def visualize_predictions(self, checkpoint_path: str, num_samples: int = 16):
        """Visualize model predictions."""
        print(f"\n[1/2] Loading model from checkpoint...")

        # Load model
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
                pretrained=False,
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print("✅ Model loaded")

        # Get predictions
        print(f"[2/2] Generating predictions...")

        dataset = RetinalDataset(
            csv_file=self.config.data.data_csv,
            img_dir=self.config.data.data_img_dir,
            transform=get_val_transforms()
        )

        # Random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size,
                                 figsize=(4 * grid_size, 4 * grid_size))
        axes = axes.flatten()

        with torch.no_grad():
            for i, idx in enumerate(indices):
                batch = dataset[idx]
                image = batch['image'].unsqueeze(0).to(self.device)
                label = batch['label'].item()

                # Predict
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1).item()
                confidence = probs[0, pred].item()

                # Visualize
                img = self._tensor_to_image(batch['image'])
                axes[i].imshow(img)

                # Color code: green if correct, red if wrong
                color = 'green' if pred == label else 'red'
                axes[i].set_title(
                    f"True: {self.class_names[label]}\n"
                    f"Pred: {self.class_names[pred]} ({confidence:.2f})",
                    fontsize=10,
                    color=color
                )
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle("Model Predictions", fontsize=16)
        plt.tight_layout()

        output_path = Path(self.config.system.output_dir) / "predictions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")
        plt.close()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image."""
        # Denormalize if needed (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        return img


def main():
    parser = argparse.ArgumentParser(description="Visualize data pipeline and predictions")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True,
                        choices=['augmentation', 'batch', 'predictions'],
                        help='Visualization mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (for predictions mode)')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples to visualize')

    args = parser.parse_args()

    if args.mode == 'predictions' and not args.checkpoint:
        print("Error: --checkpoint required for predictions mode")
        sys.exit(1)

    # Create visualizer
    visualizer = PipelineVisualizer(args.config)

    # Run visualization
    print("=" * 60)
    print(f"VISUAL INSPECTION: {args.mode.upper()}")
    print("=" * 60)

    if args.mode == 'augmentation':
        visualizer.visualize_augmentation(num_examples=3, aug_per_image=6)
    elif args.mode == 'batch':
        visualizer.visualize_batch(num_batches=2)
    elif args.mode == 'predictions':
        visualizer.visualize_predictions(args.checkpoint, num_samples=args.num_samples)

    print("\n✅ Visualization complete")


if __name__ == '__main__':
    main()
```

---

## Usage

```bash
# Visualize augmentations
python scripts/visualize_pipeline.py \
    --config configs/retfound_lora_config.yaml \
    --mode augmentation

# Visualize training batches
python scripts/visualize_pipeline.py \
    --config configs/retfound_lora_config.yaml \
    --mode batch

# Visualize predictions (requires trained model)
python scripts/visualize_pipeline.py \
    --config configs/retfound_lora_config.yaml \
    --mode predictions \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --num-samples 24
```

---

## Expected Outputs

### 1. Augmentation Gallery
- Shows original image + 6 augmented versions
- Verifies augmentations are reasonable
- File: `augmentation_gallery.png`

### 2. Batch Visualization
- Shows 16 images from training batch with labels
- Verifies data loading works
- File: `batch_1.png`, `batch_2.png`

### 3. Prediction Visualization
- Shows images with true label and predicted label
- Green border = correct, red border = wrong
- Includes confidence scores
- File: `predictions.png`

---

## What to Look For

### Augmentation Issues
❌ **Bad:** Images too dark/bright, completely distorted
✅ **Good:** Images still recognizable, reasonable variation

### Data Issues
❌ **Bad:** Wrong labels, corrupted images, artifacts
✅ **Good:** Clear images, labels match visual inspection

### Model Issues
❌ **Bad:** Confident but wrong, systematic errors (always predicts class 0)
✅ **Good:** Reasonable confidence, errors make sense

---

## Next Steps

After Phase 5:
1. **Fix issues found:** Adjust augmentation, fix data labels
2. **Document findings:** Note any systematic patterns
3. **Proceed to Phase 6:** [06-PREFLIGHT_ORCHESTRATOR.md](06-PREFLIGHT_ORCHESTRATOR.md)
4. **Or start training:** If everything looks good

---

## Success Criteria

- ✅ Script generates all visualizations
- ✅ Augmentations look reasonable
- ✅ No corrupted images in batches
- ✅ Predictions make qualitative sense
- ✅ High-quality PNG outputs

