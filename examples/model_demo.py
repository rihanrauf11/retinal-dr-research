#!/usr/bin/env python3
"""
DRClassifier Model Demo
=======================
Comprehensive demonstration of using the DRClassifier model with the
configuration system and dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.config import Config
from scripts.model import DRClassifier, get_model_summary


def demo_basic_usage():
    """Demonstrate basic model creation and usage."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Model Usage")
    print("=" * 80)

    # Create model
    model = DRClassifier(
        model_name='resnet50',
        num_classes=5,
        pretrained=True,
        dropout_rate=0.3
    )

    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    print(f"\nInput shape: {tuple(images.shape)}")

    # Forward pass
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Predictions: {predictions.tolist()}")
    print(f"Sample probabilities: {probabilities[0].tolist()}")

    # DR severity mapping
    severity_map = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "PDR"
    }

    print("\nPredicted severity levels:")
    for i, pred in enumerate(predictions):
        print(f"  Image {i+1}: {severity_map[pred.item()]}")


def demo_config_integration():
    """Demonstrate integration with configuration system."""
    print("\n" + "=" * 80)
    print("DEMO 2: Configuration Integration")
    print("=" * 80)

    # Load configuration
    config = Config.from_yaml('configs/default_config.yaml')
    print(f"\nLoaded config: {config.model.model_name}")

    # Create model from config
    model = DRClassifier.from_config(config.model)

    # Set device from config
    device = torch.device(config.system.device)
    model = model.to(device)
    print(f"Model moved to device: {device}")

    # Create data with correct image size
    images = torch.randn(2, 3, config.image.img_size, config.image.img_size)
    images = images.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(images)

    print(f"\nInput size: {config.image.img_size}x{config.image.img_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Output shape: {tuple(output.shape)}")


def demo_transfer_learning():
    """Demonstrate transfer learning workflow."""
    print("\n" + "=" * 80)
    print("DEMO 3: Transfer Learning Workflow")
    print("=" * 80)

    # Create model
    model = DRClassifier('resnet50', num_classes=5, pretrained=True)

    print("\n[Step 1] Initial state:")
    total, trainable = model.get_num_params()
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    print("\n[Step 2] Freeze backbone for classifier-only training:")
    model.freeze_backbone()

    print("\n[Step 3] After training classifier, unfreeze for fine-tuning:")
    model.unfreeze_backbone()

    print("\n✓ Transfer learning workflow demonstrated")


def demo_different_architectures():
    """Demonstrate using different model architectures."""
    print("\n" + "=" * 80)
    print("DEMO 4: Different Model Architectures")
    print("=" * 80)

    architectures = [
        ('resnet50', 224, "Standard CNN baseline"),
        ('efficientnet_b0', 224, "Efficient and compact"),
        ('mobilenetv3_small_100', 224, "Mobile-optimized"),
        ('resnet34', 224, "Lighter ResNet variant"),
    ]

    print("\nTesting different architectures:\n")

    for model_name, img_size, description in architectures:
        try:
            model = DRClassifier(
                model_name=model_name,
                num_classes=5,
                pretrained=False  # Faster for demo
            )

            # Test forward pass
            images = torch.randn(2, 3, img_size, img_size)
            with torch.no_grad():
                output = model(images)

            total_params, _ = model.get_num_params()

            print(f"✓ {model_name:30s} | {total_params:>12,} params | {description}")

        except Exception as e:
            print(f"✗ {model_name:30s} | Failed: {e}")


def demo_model_analysis():
    """Demonstrate model analysis and inspection."""
    print("\n" + "=" * 80)
    print("DEMO 5: Model Analysis")
    print("=" * 80)

    model = DRClassifier('resnet50', num_classes=5, pretrained=True)

    # Get model summary
    print("\n[Model Summary]")
    get_model_summary(model, input_size=(4, 3, 224, 224))

    # Inspect feature dimension
    print(f"Feature dimension: {model.get_feature_dim()}")

    # Model representation
    print("\n[Model Representation]")
    print(model)


def demo_batch_prediction():
    """Demonstrate batch prediction with proper preprocessing."""
    print("\n" + "=" * 80)
    print("DEMO 6: Batch Prediction")
    print("=" * 80)

    # Create model
    model = DRClassifier('resnet50', num_classes=5, pretrained=False)
    model.eval()

    # Simulate batch of images
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)

    print(f"\nProcessing batch of {batch_size} images...")

    # Batch prediction
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        confidences, _ = torch.max(probabilities, dim=1)

    # Display results
    severity_map = {
        0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"
    }

    print("\nPrediction Results:")
    print("-" * 60)
    print(f"{'Image':<10} {'Prediction':<15} {'Confidence':<15}")
    print("-" * 60)

    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = confidences[i].item()
        severity = severity_map[pred_class]

        print(f"{i+1:<10} {severity:<15} {confidence:>6.2%}")

    print("-" * 60)


def demo_mixed_precision():
    """Demonstrate mixed precision training setup."""
    print("\n" + "=" * 80)
    print("DEMO 7: Mixed Precision Setup")
    print("=" * 80)

    model = DRClassifier('resnet50', num_classes=5, pretrained=False)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("\n✓ CUDA available - setting up mixed precision")

        # Move model to GPU
        model = model.to(device)

        # Create dummy data
        images = torch.randn(2, 3, 224, 224).to(device)

        # Use autocast for mixed precision
        from torch.cuda.amp import autocast

        with autocast():
            output = model(images)

        print(f"  Output dtype: {output.dtype}")
        print("✓ Mixed precision forward pass successful")
    else:
        print("\n⚠ CUDA not available, skipping mixed precision demo")
        print("  This demo would show performance benefits on GPU")


def demo_model_persistence():
    """Demonstrate saving and loading models."""
    print("\n" + "=" * 80)
    print("DEMO 8: Model Persistence")
    print("=" * 80)

    # Create and train a model (simulated)
    print("\n[Step 1] Creating model...")
    model = DRClassifier('resnet50', num_classes=5, pretrained=False)

    # Save model checkpoint
    print("\n[Step 2] Saving model checkpoint...")
    import tempfile
    import os

    checkpoint_path = tempfile.mktemp(suffix='.pth')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'dropout_rate': model.dropout_rate,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"  ✓ Saved to: {checkpoint_path}")

    # Load model checkpoint
    print("\n[Step 3] Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    loaded_model = DRClassifier(
        model_name=checkpoint['model_name'],
        num_classes=checkpoint['num_classes'],
        pretrained=False
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    print("  ✓ Model loaded successfully")

    # Verify models are equivalent
    print("\n[Step 4] Verifying loaded model...")
    test_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output1 = model(test_input)
        output2 = loaded_model(test_input)

    diff = torch.abs(output1 - output2).max().item()
    print(f"  Maximum difference: {diff:.10f}")

    if diff < 1e-6:
        print("  ✓ Models produce identical outputs")
    else:
        print("  ⚠ Models produce different outputs")

    # Cleanup
    os.remove(checkpoint_path)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("DRClassifier Model Demonstrations")
    print("=" * 80)

    demos = [
        demo_basic_usage,
        demo_config_integration,
        demo_transfer_learning,
        demo_different_architectures,
        demo_model_analysis,
        demo_batch_prediction,
        demo_mixed_precision,
        demo_model_persistence,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)

    print("\n📚 For more information:")
    print("  - Model implementation: scripts/model.py")
    print("  - Configuration system: scripts/config.py")
    print("  - Dataset loader: scripts/dataset.py")
    print("  - Config examples: configs/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
