#!/usr/bin/env python3
"""
Configuration System Demo
=========================
This script demonstrates various ways to use the configuration system.
"""

import sys
from pathlib import Path

# Add parent directory to path to import scripts module
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, DataConfig, ModelConfig, TrainingConfig


def demo_basic_usage():
    """Demonstrate basic configuration usage."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Configuration Usage")
    print("=" * 70)

    # Create default configuration
    config = Config()
    print("\nDefault configuration:")
    print(config)


def demo_custom_config():
    """Demonstrate creating custom configuration."""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Configuration")
    print("=" * 70)

    config = Config(
        data=DataConfig(
            train_csv="data/aptos/train.csv",
            train_img_dir="data/aptos/train_images"
        ),
        model=ModelConfig(
            model_name="efficientnet_b3",
            num_classes=5,
            pretrained=True
        ),
        training=TrainingConfig(
            batch_size=32,
            num_epochs=50,
            learning_rate=3e-4
        )
    )

    print("\nCustom configuration:")
    print(f"Model: {config.model.model_name}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")


def demo_yaml_loading():
    """Demonstrate loading from YAML files."""
    print("\n" + "=" * 70)
    print("DEMO 3: Loading from YAML")
    print("=" * 70)

    configs_dir = Path(__file__).parent

    # Load different configurations
    yaml_files = [
        "default_config.yaml",
        "vit_large_config.yaml",
        "efficientnet_config.yaml"
    ]

    for yaml_file in yaml_files:
        yaml_path = configs_dir / yaml_file
        if yaml_path.exists():
            print(f"\nLoading {yaml_file}:")
            config = Config.from_yaml(yaml_path)
            print(f"  Model: {config.model.model_name}")
            print(f"  Image size: {config.image.img_size}x{config.image.img_size}")
            print(f"  Batch size: {config.training.batch_size}")
            print(f"  Device: {config.system.device}")


def demo_config_modification():
    """Demonstrate modifying existing configuration."""
    print("\n" + "=" * 70)
    print("DEMO 4: Configuration Modification")
    print("=" * 70)

    # Load base configuration
    base_config = Config()
    print("\nOriginal configuration:")
    print(f"  Model: {base_config.model.model_name}")
    print(f"  Batch size: {base_config.training.batch_size}")

    # Create modified version
    modified_config = base_config.update(
        model={'model_name': 'resnet101'},
        training={'batch_size': 48, 'num_epochs': 30}
    )

    print("\nModified configuration:")
    print(f"  Model: {modified_config.model.model_name}")
    print(f"  Batch size: {modified_config.training.batch_size}")
    print(f"  Epochs: {modified_config.training.num_epochs}")

    print("\nOriginal config unchanged:")
    print(f"  Model: {base_config.model.model_name}")
    print(f"  Batch size: {base_config.training.batch_size}")


def demo_validation():
    """Demonstrate configuration validation."""
    print("\n" + "=" * 70)
    print("DEMO 5: Configuration Validation")
    print("=" * 70)

    # Valid configuration
    valid_config = Config()
    print("\nValidating valid configuration...")
    try:
        valid_config.validate(create_dirs=True)
        print("✓ Validation successful!")
    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Invalid configuration (negative batch size)
    print("\nValidating invalid configuration (negative batch size)...")
    invalid_config = Config(
        training=TrainingConfig(batch_size=-1)
    )
    try:
        invalid_config.validate(create_dirs=False)
        print("✗ Should have failed validation!")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")


def demo_saving():
    """Demonstrate saving configuration to YAML."""
    print("\n" + "=" * 70)
    print("DEMO 6: Saving Configuration")
    print("=" * 70)

    # Create custom configuration
    config = Config(
        model=ModelConfig(
            model_name="vit_base_patch16_224",
            num_classes=5
        ),
        training=TrainingConfig(
            batch_size=24,
            num_epochs=40,
            learning_rate=2e-4
        )
    )

    # Save to temporary file
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        delete=False
    )
    temp_path = temp_file.name
    temp_file.close()

    print(f"\nSaving configuration to: {temp_path}")
    config.to_yaml(temp_path)

    # Load it back
    print("\nLoading saved configuration...")
    loaded_config = Config.from_yaml(temp_path)
    print(f"  Model: {loaded_config.model.model_name}")
    print(f"  Batch size: {loaded_config.training.batch_size}")

    # Cleanup
    import os
    os.remove(temp_path)
    print(f"\n✓ Configuration save/load successful!")


def demo_dict_conversion():
    """Demonstrate dictionary conversion."""
    print("\n" + "=" * 70)
    print("DEMO 7: Dictionary Conversion")
    print("=" * 70)

    # Create configuration
    config = Config(
        model=ModelConfig(model_name="resnet50"),
        training=TrainingConfig(batch_size=32)
    )

    # Convert to dictionary
    config_dict = config.to_dict()
    print("\nConfiguration as dictionary:")
    print(f"  Model name: {config_dict['model']['model_name']}")
    print(f"  Batch size: {config_dict['training']['batch_size']}")
    print(f"  Device: {config_dict['system']['device']}")

    # Create from dictionary
    new_dict = {
        'model': {'model_name': 'efficientnet_b0'},
        'training': {'batch_size': 64}
    }
    new_config = Config.from_dict(new_dict)
    print("\nConfiguration from dictionary:")
    print(f"  Model name: {new_config.model.model_name}")
    print(f"  Batch size: {new_config.training.batch_size}")


def demo_device_detection():
    """Demonstrate device detection."""
    print("\n" + "=" * 70)
    print("DEMO 8: Device Detection")
    print("=" * 70)

    import torch

    config = Config()
    print(f"\nAuto-detected device: {config.system.device}")

    print("\nAvailable devices:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")

    if hasattr(torch.backends, 'mps'):
        print(f"  MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")

    print(f"  CPU always available: True")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Configuration System Demonstrations")
    print("=" * 70)

    demos = [
        demo_basic_usage,
        demo_custom_config,
        demo_yaml_loading,
        demo_config_modification,
        demo_validation,
        demo_saving,
        demo_dict_conversion,
        demo_device_detection
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - configs/README.md")
    print("  - scripts/config.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
