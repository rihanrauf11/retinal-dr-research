# Configuration Files

This directory contains YAML configuration files for different training experiments.

## Available Configurations

### 1. `default_config.yaml`
Default configuration using ResNet50 with standard settings.
- Model: ResNet50
- Image size: 224x224
- Batch size: 16
- Good starting point for most experiments

### 2. `vit_large_config.yaml`
Configuration for Vision Transformer (ViT) Large model.
- Model: ViT Large (16x16 patches)
- Image size: 384x384
- Batch size: 8 (smaller due to model size)
- Best for high-accuracy requirements

### 3. `efficientnet_config.yaml`
Configuration for EfficientNet-B3.
- Model: EfficientNet-B3
- Image size: 300x300
- Batch size: 32 (larger due to efficiency)
- Good balance of speed and accuracy

## Usage

### Loading a Configuration

```python
from scripts.config import Config

# Load from YAML file
config = Config.from_yaml('configs/default_config.yaml')

# Validate and create output directories
config.validate()

# Print configuration
print(config)
```

### Creating Custom Configuration

```python
from scripts.config import Config, DataConfig, TrainingConfig

# Create with custom parameters
config = Config(
    data=DataConfig(
        train_csv='data/aptos/train.csv',
        train_img_dir='data/aptos/train_images'
    ),
    training=TrainingConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=3e-4
    )
)

# Save to file
config.to_yaml('configs/my_experiment.yaml')
```

### Modifying Existing Configuration

```python
# Load base configuration
config = Config.from_yaml('configs/default_config.yaml')

# Update specific parameters
new_config = config.update(
    model={'model_name': 'resnet101'},
    training={'batch_size': 32, 'num_epochs': 50}
)

# Save modified configuration
new_config.to_yaml('configs/resnet101_large_batch.yaml')
```

### Using in Training Scripts

```python
from scripts.config import Config
from scripts.dataset import RetinalDataset
from torch.utils.data import DataLoader
import torch

# Load configuration
config = Config.from_yaml('configs/default_config.yaml')
config.validate()

# Set random seed for reproducibility
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
    # Your training code here
    pass
```

## Configuration Structure

Each configuration file has the following sections:

### Data
- `train_csv`: Path to training CSV
- `train_img_dir`: Training images directory
- `test_csv`: Path to test/validation CSV
- `test_img_dir`: Test images directory

### Model
- `model_name`: Model architecture name (see timm for options)
- `num_classes`: Number of output classes (5 for DR)
- `pretrained`: Use pretrained weights

### Training
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization strength

### Image
- `img_size`: Target image size (NxN)

### System
- `num_workers`: Number of data loading workers
- `seed`: Random seed for reproducibility
- `device`: Training device ('cuda', 'cpu', or 'mps')

### Paths
- `checkpoint_dir`: Directory for saving checkpoints
- `log_dir`: Directory for saving logs

## Creating New Configurations

To create a new configuration:

1. Copy an existing config file:
   ```bash
   cp configs/default_config.yaml configs/my_experiment.yaml
   ```

2. Edit the parameters:
   ```yaml
   model:
     model_name: vit_base_patch16_224
     num_classes: 5
     pretrained: true

   training:
     batch_size: 24
     num_epochs: 30
     learning_rate: 0.0002
   ```

3. Load in your script:
   ```python
   config = Config.from_yaml('configs/my_experiment.yaml')
   ```

## Tips

1. **Model Selection**: Use `timm.list_models()` to see available models:
   ```python
   import timm
   models = timm.list_models(pretrained=True)
   print(models)
   ```

2. **Batch Size**: Adjust based on GPU memory:
   - ResNet50 @ 224: 32-64
   - ViT Large @ 384: 8-16
   - EfficientNet-B3 @ 300: 32-48

3. **Learning Rate**: Common starting points:
   - From scratch: 1e-3
   - Fine-tuning: 1e-4 to 5e-5
   - Large models: 5e-5 to 1e-5

4. **Device Selection**:
   - NVIDIA GPU: `cuda` or `cuda:0`, `cuda:1`, etc.
   - Apple Silicon: `mps`
   - CPU: `cpu`

## Validation

Always validate your configuration before training:

```python
try:
    config.validate()
    print("Configuration is valid!")
except (ValueError, FileNotFoundError) as e:
    print(f"Configuration error: {e}")
```

This will:
- Check that parameters are valid (positive values, etc.)
- Verify that input files/directories exist
- Create output directories if they don't exist
