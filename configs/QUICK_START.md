# Configuration System - Quick Start

## 5-Minute Guide

### 1. Basic Usage (30 seconds)

```python
from scripts.config import Config

# Create default config
config = Config()
config.validate()
```

### 2. Load from YAML (1 minute)

```python
from scripts.config import Config

# Load pre-configured setup
config = Config.from_yaml('configs/default_config.yaml')
config.validate()

# Print configuration
print(config)
```

### 3. Custom Configuration (2 minutes)

```python
from scripts.config import Config, DataConfig, TrainingConfig

# Create custom config
config = Config(
    data=DataConfig(
        train_csv='data/aptos/train.csv',
        train_img_dir='data/aptos/train_images'
    ),
    training=TrainingConfig(
        batch_size=32,
        num_epochs=50
    )
)

# Save for reuse
config.to_yaml('configs/my_experiment.yaml')
```

### 4. Use in Training (2 minutes)

```python
from scripts.config import Config
from scripts.dataset import RetinalDataset
from torch.utils.data import DataLoader
import torch

# Load config
config = Config.from_yaml('configs/default_config.yaml')
config.validate()

# Set seed
torch.manual_seed(config.system.seed)

# Create dataset
dataset = RetinalDataset(
    csv_file=config.data.train_csv,
    img_dir=config.data.train_img_dir,
    transform=your_transform
)

# Create loader
loader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    num_workers=config.system.num_workers,
    shuffle=True
)

# Set device
device = torch.device(config.system.device)

# Train
for epoch in range(config.training.num_epochs):
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        # Your training code here
```

## Common Commands

```python
# Load configuration
config = Config.from_yaml('path/to/config.yaml')

# Validate and create directories
config.validate()

# Update parameters
new_config = config.update(training={'batch_size': 64})

# Save configuration
config.to_yaml('path/to/save.yaml')

# Convert to dictionary
config_dict = config.to_dict()
```

## Available Configs

| Config File | Model | Image Size | Batch Size | Use Case |
|-------------|-------|-----------|-----------|----------|
| `default_config.yaml` | ResNet50 | 224 | 16 | General purpose |
| `vit_large_config.yaml` | ViT Large | 384 | 8 | High accuracy |
| `efficientnet_config.yaml` | EfficientNet-B3 | 300 | 32 | Speed/accuracy balance |

## Quick Tips

1. **Always validate**: `config.validate()`
2. **Set seed**: `torch.manual_seed(config.system.seed)`
3. **Check device**: Device is auto-detected
4. **Save configs**: Save each experiment's config for reproducibility

## Need More Help?

- Full guide: `CONFIGURATION_GUIDE.md`
- Usage examples: `configs/README.md`
- Run demos: `python configs/config_demo.py`
- Run tests: `python scripts/config.py`

## One-Liner Setup

```bash
python3 -c "from scripts.config import Config; Config().to_yaml('configs/my_config.yaml')"
```
