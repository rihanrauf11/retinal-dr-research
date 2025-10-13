# Configuration Management System Guide

## Overview

The configuration management system provides a type-safe, flexible, and maintainable way to manage experiment configurations for diabetic retinopathy classification research.

## Features

✓ **Type-Safe**: Full type hints using Python dataclasses
✓ **YAML Support**: Load/save configurations from/to YAML files
✓ **Validation**: Automatic validation of parameters and paths
✓ **Auto-Creation**: Automatically creates output directories
✓ **Device Detection**: Automatically detects available hardware (CUDA/MPS/CPU)
✓ **Immutable Updates**: Create new configs without modifying originals
✓ **Extensible**: Easy to add new parameters or configuration groups

## Quick Start

### 1. Using Default Configuration

```python
from scripts.config import Config

# Create with defaults
config = Config()

# Validate and create directories
config.validate()

# Print configuration
print(config)
```

### 2. Loading from YAML

```python
from scripts.config import Config

# Load pre-defined configuration
config = Config.from_yaml('configs/default_config.yaml')

# Validate
config.validate()
```

### 3. Creating Custom Configuration

```python
from scripts.config import Config, DataConfig, ModelConfig, TrainingConfig

config = Config(
    data=DataConfig(
        train_csv='data/aptos/train.csv',
        train_img_dir='data/aptos/train_images'
    ),
    model=ModelConfig(
        model_name='efficientnet_b3',
        num_classes=5,
        pretrained=True
    ),
    training=TrainingConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=3e-4
    )
)

# Save for future use
config.to_yaml('configs/my_experiment.yaml')
```

## Configuration Structure

### DataConfig
```python
@dataclass
class DataConfig:
    train_csv: Optional[str] = None          # Training CSV path
    train_img_dir: Optional[str] = None      # Training images directory
    test_csv: Optional[str] = None           # Test CSV path
    test_img_dir: Optional[str] = None       # Test images directory
```

### ModelConfig
```python
@dataclass
class ModelConfig:
    model_name: str = "resnet50"    # Model architecture
    num_classes: int = 5            # Number of classes (DR: 0-4)
    pretrained: bool = True         # Use pretrained weights
```

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    batch_size: int = 16            # Batch size
    num_epochs: int = 20            # Number of epochs
    learning_rate: float = 1e-4     # Learning rate
    weight_decay: float = 1e-4      # L2 regularization
```

### ImageConfig
```python
@dataclass
class ImageConfig:
    img_size: int = 224             # Target image size (NxN)
```

### SystemConfig
```python
@dataclass
class SystemConfig:
    num_workers: int = 4            # Data loading workers
    seed: int = 42                  # Random seed
    device: str = "cuda"            # Device: cuda/cpu/mps
```

### PathConfig
```python
@dataclass
class PathConfig:
    checkpoint_dir: str = "results/checkpoints"  # Checkpoint directory
    log_dir: str = "results/logs"                # Log directory
```

## Available Configurations

### 1. Default Configuration (`default_config.yaml`)
- **Model**: ResNet50
- **Image Size**: 224x224
- **Batch Size**: 16
- **Use Case**: General purpose, good starting point

### 2. ViT Large Configuration (`vit_large_config.yaml`)
- **Model**: Vision Transformer Large
- **Image Size**: 384x384
- **Batch Size**: 8
- **Use Case**: High accuracy requirements, larger models

### 3. EfficientNet Configuration (`efficientnet_config.yaml`)
- **Model**: EfficientNet-B3
- **Image Size**: 300x300
- **Batch Size**: 32
- **Use Case**: Balance of speed and accuracy

## Common Patterns

### Pattern 1: Training Script Integration

```python
from scripts.config import Config
from scripts.dataset import RetinalDataset
from torch.utils.data import DataLoader
import torch

# Load configuration
config = Config.from_yaml('configs/experiment.yaml')
config.validate()

# Set random seed
torch.manual_seed(config.system.seed)

# Create dataset
train_dataset = RetinalDataset(
    csv_file=config.data.train_csv,
    img_dir=config.data.train_img_dir,
    transform=your_transform
)

# Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    num_workers=config.system.num_workers,
    shuffle=True
)

# Set device
device = torch.device(config.system.device)

# Training loop
for epoch in range(config.training.num_epochs):
    for batch in train_loader:
        # Your training code
        pass
```

### Pattern 2: Hyperparameter Sweep

```python
from scripts.config import Config

base_config = Config.from_yaml('configs/default_config.yaml')

# Sweep over learning rates
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

for lr in learning_rates:
    # Create modified configuration
    config = base_config.update(
        training={'learning_rate': lr},
        paths={
            'checkpoint_dir': f'results/checkpoints/lr_{lr}',
            'log_dir': f'results/logs/lr_{lr}'
        }
    )

    # Save configuration
    config.to_yaml(f'configs/sweep_lr_{lr}.yaml')

    # Run training with this config
    # train_model(config)
```

### Pattern 3: Multi-Dataset Training

```python
from scripts.config import Config, DataConfig

# APTOS dataset
config_aptos = Config(
    data=DataConfig(
        train_csv='data/aptos/train.csv',
        train_img_dir='data/aptos/train_images'
    )
)

# Messidor dataset
config_messidor = Config(
    data=DataConfig(
        train_csv='data/messidor/train.csv',
        train_img_dir='data/messidor/train_images'
    )
)

# Train on both
configs = [config_aptos, config_messidor]
for i, config in enumerate(configs):
    print(f"Training on dataset {i+1}")
    # train_model(config)
```

### Pattern 4: Configuration Comparison

```python
from scripts.config import Config

# Load multiple configurations
configs = {
    'resnet50': Config.from_yaml('configs/default_config.yaml'),
    'vit_large': Config.from_yaml('configs/vit_large_config.yaml'),
    'efficientnet': Config.from_yaml('configs/efficientnet_config.yaml')
}

# Compare
for name, config in configs.items():
    print(f"\n{name}:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Image size: {config.image.img_size}")
    print(f"  Batch size: {config.training.batch_size}")
```

## Validation

The configuration system includes comprehensive validation:

```python
config = Config.from_yaml('configs/experiment.yaml')

try:
    config.validate()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except FileNotFoundError as e:
    print(f"Missing file/directory: {e}")
```

**What gets validated:**
- ✓ Positive values for `num_classes`, `batch_size`, `num_epochs`, etc.
- ✓ Valid device names (`cuda`, `cpu`, `mps`)
- ✓ Existence of input files (CSV, image directories)
- ✓ Output directories are created if they don't exist

## Device Selection

The system automatically detects available hardware:

```python
config = Config()  # Auto-detects device

# Manual override
config = Config(system=SystemConfig(device='mps'))  # Apple Silicon
config = Config(system=SystemConfig(device='cuda:0'))  # Specific GPU
config = Config(system=SystemConfig(device='cpu'))  # CPU
```

**Device Priority:**
1. CUDA (if available)
2. MPS (if on Apple Silicon)
3. CPU (fallback)

## Tips and Best Practices

### 1. Model Selection

```python
import timm

# List all available pretrained models
models = timm.list_models(pretrained=True)

# Search for specific models
resnet_models = timm.list_models('resnet*', pretrained=True)
vit_models = timm.list_models('vit*', pretrained=True)
```

### 2. Batch Size Guidelines

| Model | Image Size | Recommended Batch Size (16GB GPU) |
|-------|-----------|-----------------------------------|
| ResNet50 | 224x224 | 32-64 |
| ResNet101 | 224x224 | 24-48 |
| EfficientNet-B3 | 300x300 | 32-48 |
| ViT Base | 224x224 | 32-48 |
| ViT Large | 384x384 | 8-16 |

### 3. Learning Rate Selection

| Scenario | Recommended LR |
|----------|---------------|
| Training from scratch | 1e-3 to 3e-4 |
| Fine-tuning (small model) | 1e-4 to 5e-5 |
| Fine-tuning (large model) | 5e-5 to 1e-5 |
| Transfer learning | 1e-4 to 3e-4 |

### 4. Configuration Naming

Use descriptive names for your configurations:

```
configs/
  ├── resnet50_224_bs32.yaml       # Model_ImageSize_BatchSize
  ├── vit_large_384_lr5e5.yaml     # Model_ImageSize_LearningRate
  ├── efficientnet_b3_messidor.yaml # Model_Dataset
  └── experiment_2024_01_15.yaml   # Date-based
```

### 5. Version Control

Add to `.gitignore`:
```
# Keep template configs
!configs/*_config.yaml
!configs/README.md

# Ignore experiment-specific configs
configs/experiment_*.yaml
configs/sweep_*.yaml
```

## Testing

Run the test suite:

```bash
# Test configuration system
python scripts/config.py

# Run demonstrations
python configs/config_demo.py
```

## Troubleshooting

### Issue: "Configuration file not found"
```python
# Check if file exists
from pathlib import Path
config_path = Path('configs/my_config.yaml')
print(f"Exists: {config_path.exists()}")
```

### Issue: "CUDA not available"
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use CPU instead
config = Config(system=SystemConfig(device='cpu'))
```

### Issue: "Invalid model name"
```python
import timm

# Check if model exists
model_name = 'resnet50'
if model_name in timm.list_models():
    print(f"{model_name} is available")
else:
    print(f"{model_name} not found")
    print("Available models:", timm.list_models('resnet*'))
```

## API Reference

### Config Methods

- `Config()` - Create with default values
- `Config.from_yaml(path)` - Load from YAML file
- `Config.from_dict(dict)` - Create from dictionary
- `config.to_yaml(path)` - Save to YAML file
- `config.to_dict()` - Convert to dictionary
- `config.validate()` - Validate configuration
- `config.update(**kwargs)` - Create updated copy

## Examples

See:
- `configs/config_demo.py` - Comprehensive demonstrations
- `scripts/config.py` - Implementation and tests
- `configs/README.md` - Configuration file documentation

## Support

For issues or questions:
1. Check `configs/README.md`
2. Run `python configs/config_demo.py`
3. Examine test suite in `scripts/config.py`
