# Utils API Documentation

## Overview

The `scripts.utils` module provides a comprehensive collection of utility functions for deep learning workflows, including seed management, checkpoint handling, data transforms, metrics calculation, visualization, and Weights & Biases integration.

**Key Features:**
- Reproducibility through seed management
- Model parameter counting and summaries
- Checkpoint save/load with full state
- Data transform pipelines (torchvision & albumentations)
- DataLoader creation utilities
- Comprehensive metrics calculation
- Confusion matrix visualization
- Training history management
- Device management and utilities
- Complete W&B integration

**Source File:**
- [scripts/utils.py](../scripts/utils.py:1) - Complete utilities module

---

## Table of Contents

1. [Random Seed Management](#random-seed-management)
2. [Model Parameter Utilities](#model-parameter-utilities)
3. [Checkpoint Management](#checkpoint-management)
4. [Data Transform Utilities](#data-transform-utilities)
5. [Data Loader Utilities](#data-loader-utilities)
6. [Metrics Calculation](#metrics-calculation)
7. [Visualization](#visualization)
8. [Training History](#training-history)
9. [Device Management](#device-management)
10. [Weights & Biases Integration](#weights--biases-integration)

---

## Random Seed Management

### set_seed

Set random seeds for reproducibility across all libraries.

**Signature:**
```python
def set_seed(seed: int, deterministic: bool = True, verbose: bool = True) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | Required | Random seed value (e.g., 42, 123) |
| `deterministic` | `bool` | `True` | Make CUDNN deterministic (slower but reproducible) |
| `verbose` | `bool` | `True` | Print confirmation message |

**Sets seeds for:**
- Python `random` module
- NumPy
- PyTorch (CPU and all CUDA devices)
- CUDNN backend

**Example:**

```python
from scripts.utils import set_seed

# Full reproducibility (deterministic mode)
set_seed(42, deterministic=True)
# Output: ✓ Random seed set to 42 (deterministic mode: ON)

# Faster training with minor variation
set_seed(42, deterministic=False)
# Output: ✓ Random seed set to 42 (deterministic mode: OFF)
```

---

## Model Parameter Utilities

### count_parameters

Count parameters in a PyTorch model.

**Signature:**
```python
def count_parameters(model: nn.Module, trainable_only: bool = False) -> Union[int, Tuple[int, int]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | PyTorch model |
| `trainable_only` | `bool` | `False` | Return format |

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Number of trainable parameters (if `trainable_only=True`) |
| `Tuple[int, int]` | `(total_params, trainable_params)` (if `trainable_only=False`) |

**Example:**

```python
from scripts.utils import count_parameters
from scripts.model import DRClassifier

model = DRClassifier('resnet50', num_classes=5)

# Get both counts
total, trainable = count_parameters(model)
print(f"Total: {total:,}, Trainable: {trainable:,}")
# Output: Total: 23,516,997, Trainable: 23,516,997

# Get only trainable
trainable = count_parameters(model, trainable_only=True)
print(f"Trainable: {trainable:,}")
```

---

### print_model_summary

Print comprehensive model summary with architecture and parameter info.

**Signature:**
```python
def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    verbose: bool = True
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | Model to summarize |
| `input_size` | `Tuple[int, ...]` | `(1, 3, 224, 224)` | Input tensor shape (batch, channels, height, width) |
| `verbose` | `bool` | `True` | Print summary to console |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Dictionary with `total_params`, `trainable_params`, `non_trainable_params`, `memory_mb`, `input_shape`, `output_shape`, `forward_pass_ok` |

**Example:**

```python
from scripts.utils import print_model_summary

model = DRClassifier('resnet50', num_classes=5)
stats = print_model_summary(model, input_size=(2, 3, 224, 224))

# Output:
# ╔═══════════════════════════════════════════════════════╗
# ║              MODEL SUMMARY                            ║
# ╚═══════════════════════════════════════════════════════╝
# Total Parameters:        23,516,997
# Trainable Parameters:    23,516,997
# Non-trainable Parameters:           0
# Memory Size:                 89.67 MB
# Input Shape:          (2, 3, 224, 224)
# Output Shape:                   (2, 5)
# ✓ Forward pass successful

# Access stats
print(f"Memory: {stats['memory_mb']:.2f} MB")
```

---

## Checkpoint Management

### save_checkpoint

Save model checkpoint with comprehensive state.

**Signature:**
```python
def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    path: Union[str, Path],
    is_best: bool = False,
    **kwargs
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | Model to save |
| `optimizer` | `optim.Optimizer` or `None` | Required | Optimizer state (can be None) |
| `epoch` | `int` | Required | Current epoch number |
| `metrics` | `Dict[str, Any]` | Required | Metrics dict (e.g., `{'val_acc': 85.3}`) |
| `path` | `str` or `Path` | Required | Save path |
| `is_best` | `bool` | `False` | Also save as `best_model.pth` |
| `**kwargs` | `Any` | - | Additional items (scheduler, history, config, etc.) |

**Checkpoint Structure:**
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,  # if optimizer provided
    'metrics': dict,
    'model_class': str,
    **kwargs  # any additional items
}
```

**Example:**

```python
from scripts.utils import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'val_acc': 85.3, 'val_loss': 0.42},
    path='checkpoints/epoch_10.pth',
    is_best=True,
    scheduler=scheduler.state_dict(),
    history=training_history,
    config=config.to_dict()
)

# Output:
# ✓ Checkpoint saved: checkpoints/epoch_10.pth
# ✓ Best model saved: checkpoints/best_model.pth
```

---

### load_checkpoint

Load model checkpoint.

**Signature:**
```python
def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str = 'cpu',
    strict: bool = True,
    verbose: bool = True
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | Required | Path to checkpoint |
| `model` | `nn.Module` | Required | Model to load state into |
| `optimizer` | `optim.Optimizer` | `None` | Optimizer to load state into (optional) |
| `map_location` | `str` | `'cpu'` | Device mapping ('cpu', 'cuda', 'cuda:0', etc.) |
| `strict` | `bool` | `True` | Require exact key match |
| `verbose` | `bool` | `True` | Print loading info |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Checkpoint metadata (epoch, metrics, etc.) |

**Example:**

```python
from scripts.utils import load_checkpoint
import torch

model = DRClassifier('resnet50', num_classes=5)
optimizer = torch.optim.Adam(model.parameters())

metadata = load_checkpoint(
    path='checkpoints/best_model.pth',
    model=model,
    optimizer=optimizer,
    map_location='cuda'
)

# Output:
# ✓ Checkpoint loaded: checkpoints/best_model.pth
# ✓ Resuming from epoch 10
# ✓ Best validation accuracy: 85.30%

print(f"Epoch: {metadata['epoch']}")
print(f"Metrics: {metadata['metrics']}")
```

---

### resume_training_from_checkpoint

High-level function to resume training.

**Signature:**
```python
def resume_training_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = 'cpu'
) -> Tuple[int, float, Dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | `str` or `Path` | Required | Path to checkpoint |
| `model` | `nn.Module` | Required | Model to load |
| `optimizer` | `optim.Optimizer` | `None` | Optimizer to load |
| `scheduler` | `Any` | `None` | LR scheduler to load |
| `map_location` | `str` | `'cpu'` | Device mapping |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[int, float, Dict]` | `(start_epoch, best_metric, history)` |

**Example:**

```python
from scripts.utils import resume_training_from_checkpoint

model = DRClassifier('resnet50')
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

start_epoch, best_acc, history = resume_training_from_checkpoint(
    checkpoint_path='checkpoints/latest.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    map_location='cuda'
)

print(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")

# Continue training
for epoch in range(start_epoch, num_epochs + 1):
    # Training loop...
    pass
```

---

## Data Transform Utilities

### get_imagenet_stats

Get ImageNet normalization statistics.

**Signature:**
```python
def get_imagenet_stats() -> Dict[str, List[float]]
```

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, List[float]]` | Dict with `'mean'` and `'std'` keys containing RGB values |

**Example:**

```python
from scripts.utils import get_imagenet_stats

stats = get_imagenet_stats()
print(stats)
# Output: {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

# Use in transforms
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(**stats)
])
```

---

### get_transforms

Create data transformation pipeline with configurable augmentation.

**Signature:**
```python
def get_transforms(
    img_size: int = 224,
    is_train: bool = True,
    augmentation_level: str = 'medium',
    backend: str = 'albumentations'
) -> Union[A.Compose, transforms.Compose]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_size` | `int` | `224` | Target image size (square) |
| `is_train` | `bool` | `True` | Apply augmentation if True |
| `augmentation_level` | `str` | `'medium'` | One of `'light'`, `'medium'`, `'heavy'` |
| `backend` | `str` | `'albumentations'` | `'albumentations'` or `'torchvision'` |

**Augmentation Levels:**

| Level | Augmentations |
|-------|--------------|
| `'light'` | Flips only |
| `'medium'` | Flips + rotation + color jitter (recommended) |
| `'heavy'` | Flips + rotation + color + coarse dropout |

**Returns:**

| Type | Description |
|------|-------------|
| `A.Compose` or `transforms.Compose` | Transform pipeline |

**Example:**

```python
from scripts.utils import get_transforms

# Training transforms with medium augmentation
train_transform = get_transforms(224, is_train=True, augmentation_level='medium')

# Validation transforms (no augmentation)
val_transform = get_transforms(224, is_train=False)

# Heavy augmentation for small datasets
heavy_transform = get_transforms(224, is_train=True, augmentation_level='heavy')

# Using torchvision backend
tv_transform = get_transforms(224, is_train=True, backend='torchvision')
```

---

## Data Loader Utilities

### create_data_loaders

Create train/validation data loaders with automatic splitting.

**Signature:**
```python
def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | Required | PyTorch Dataset to split |
| `batch_size` | `int` | `32` | Batch size for both loaders |
| `split_ratio` | `float` | `0.8` | Training fraction (0.8 = 80% train, 20% val) |
| `num_workers` | `int` | `4` | Data loading workers |
| `pin_memory` | `bool` | `True` | Use pinned memory (faster for CUDA) |
| `seed` | `int` | `42` | Random seed for split |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[DataLoader, DataLoader]` | `(train_loader, val_loader)` |

**Example:**

```python
from scripts.utils import create_data_loaders, get_transforms
from scripts.dataset import RetinalDataset

# Create dataset
transform = get_transforms(224, is_train=True)
dataset = RetinalDataset('train.csv', 'images/', transform=transform)

# Create loaders with 80/20 split
train_loader, val_loader = create_data_loaders(
    dataset=dataset,
    batch_size=32,
    split_ratio=0.8,
    num_workers=4
)

# Output: ✓ Created data loaders: train=2930, val=732

print(f"Train batches: {len(train_loader)}")  # 92
print(f"Val batches: {len(val_loader)}")      # 23
```

---

### create_dataloader_from_dataset

Create a single DataLoader from dataset.

**Signature:**
```python
def create_dataloader_from_dataset(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | Required | PyTorch Dataset |
| `batch_size` | `int` | `32` | Batch size |
| `shuffle` | `bool` | `True` | Shuffle data |
| `num_workers` | `int` | `4` | Worker processes |
| `pin_memory` | `bool` | `True` | Pinned memory |

**Example:**

```python
from scripts.utils import create_dataloader_from_dataset

test_loader = create_dataloader_from_dataset(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False  # Don't shuffle test data
)
```

---

## Metrics Calculation

### calculate_metrics

Calculate comprehensive classification metrics.

**Signature:**
```python
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `np.ndarray` | Required | Ground truth labels `[n_samples]` |
| `y_pred` | `np.ndarray` | Required | Predicted labels `[n_samples]` |
| `num_classes` | `int` | `5` | Number of classes |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Comprehensive metrics dictionary |

**Return Dictionary Structure:**
```python
{
    'accuracy': float,
    'precision_macro': float,
    'precision_weighted': float,
    'recall_macro': float,
    'recall_weighted': float,
    'f1_macro': float,
    'f1_weighted': float,
    'cohen_kappa': float,
    'confusion_matrix': List[List[int]],
    'per_class_metrics': {
        '0': {'precision': float, 'recall': float, 'f1-score': float, 'support': int},
        '1': {...},
        ...
    }
}
```

**Example:**

```python
from scripts.utils import calculate_metrics
import numpy as np

y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
y_pred = np.array([0, 1, 1, 3, 4, 0, 1, 2, 2, 4])

metrics = calculate_metrics(y_true, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

# Per-class metrics
for class_id, class_metrics in metrics['per_class_metrics'].items():
    print(f"Class {class_id}: Precision={class_metrics['precision']:.4f}, "
          f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}")
```

---

### print_metrics

Pretty-print metrics dictionary.

**Signature:**
```python
def print_metrics(metrics: Dict[str, Any], title: str = "Metrics", precision: int = 4) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `Dict[str, Any]` | Required | Metrics from `calculate_metrics()` |
| `title` | `str` | `"Metrics"` | Title for report |
| `precision` | `int` | `4` | Decimal places |

**Example:**

```python
from scripts.utils import calculate_metrics, print_metrics

metrics = calculate_metrics(y_true, y_pred)
print_metrics(metrics, title="Validation Metrics")

# Output:
# ╔════════════════════════════════════════════════╗
# ║         Validation Metrics                     ║
# ╚════════════════════════════════════════════════╝
# Accuracy:              0.8000
# Precision (macro):     0.8000
# Precision (weighted):  0.8000
# Recall (macro):        0.8000
# Recall (weighted):     0.8000
# F1-Score (macro):      0.8000
# F1-Score (weighted):   0.8000
# Cohen's Kappa:         0.7500
```

---

## Visualization

### plot_confusion_matrix

Plot and optionally save confusion matrix.

**Signature:**
```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    show: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `np.ndarray` | Required | Ground truth labels |
| `y_pred` | `np.ndarray` | Required | Predicted labels |
| `classes` | `List[str]` | `None` | Class names (defaults to DR classes) |
| `save_path` | `str` or `Path` | `None` | Path to save figure |
| `normalize` | `bool` | `True` | Normalize by row (show proportions) |
| `title` | `str` | `'Confusion Matrix'` | Plot title |
| `figsize` | `Tuple[int, int]` | `(10, 8)` | Figure size |
| `cmap` | `str` | `'Blues'` | Colormap |
| `show` | `bool` | `True` | Display plot |

**Example:**

```python
from scripts.utils import plot_confusion_matrix
import numpy as np

y_true = np.array([0, 1, 2, 3, 4] * 20)
y_pred = np.array([0, 1, 1, 3, 4] * 20)

plot_confusion_matrix(
    y_true, y_pred,
    classes=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
    save_path='confusion_matrix.png',
    normalize=True,
    title='Test Set Confusion Matrix'
)
# Output: ✓ Confusion matrix saved: confusion_matrix.png
```

---

### plot_training_history

Plot training history (loss and accuracy curves).

**Signature:**
```python
def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `history` | `Dict[str, List]` | Required | Training history |
| `save_path` | `str` or `Path` | `None` | Save path |
| `metrics` | `List[str]` | `['loss', 'acc']` | Metrics to plot |
| `figsize` | `Tuple[int, int]` | `(14, 5)` | Figure size |

**Example:**

```python
from scripts.utils import plot_training_history

history = {
    'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35],
    'train_acc': [65, 75, 80, 85, 87],
    'val_loss': [0.9, 0.7, 0.6, 0.5, 0.48],
    'val_acc': [60, 70, 75, 80, 82]
}

plot_training_history(
    history,
    save_path='training_curves.png',
    metrics=['loss', 'acc']
)
# Output: ✓ Training history plot saved: training_curves.png
```

---

## Training History

### save_training_history

Save training history to JSON file.

**Signature:**
```python
def save_training_history(history: Dict[str, List], filepath: Union[str, Path]) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `history` | `Dict[str, List]` | Training history |
| `filepath` | `str` or `Path` | Save path |

**Example:**

```python
from scripts.utils import save_training_history

history = {
    'train_loss': [0.8, 0.6, 0.5],
    'train_acc': [65, 75, 80],
    'val_loss': [0.9, 0.7, 0.6],
    'val_acc': [60, 70, 75]
}

save_training_history(history, 'logs/training_history.json')
# Output: ✓ Training history saved: logs/training_history.json
```

---

### load_training_history

Load training history from JSON file.

**Signature:**
```python
def load_training_history(filepath: Union[str, Path]) -> Dict[str, List]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` or `Path` | Path to JSON file |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, List]` | Training history dictionary |

**Example:**

```python
from scripts.utils import load_training_history

history = load_training_history('logs/training_history.json')
print(f"Trained for {len(history['train_loss'])} epochs")
# Output:
# ✓ Training history loaded: logs/training_history.json
# Trained for 20 epochs
```

---

## Device Management

### get_device

Get PyTorch device (auto-detect or specific GPU).

**Signature:**
```python
def get_device(device_id: Optional[int] = None, verbose: bool = True) -> torch.device
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device_id` | `int` | `None` | GPU ID (0, 1, etc.) or None for auto, -1 for CPU |
| `verbose` | `bool` | `True` | Print device info |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.device` | PyTorch device object |

**Example:**

```python
from scripts.utils import get_device

# Auto-detect
device = get_device()
# Output: ✓ Using device: cuda:0 (NVIDIA GeForce RTX 3090)

# Force CPU
device = get_device(device_id=-1)
# Output: ✓ Using device: cpu

# Specific GPU
device = get_device(device_id=1)
# Output: ✓ Using device: cuda:1 (NVIDIA GeForce RTX 3090)
```

---

### move_to_device

Recursively move tensors/models to device.

**Signature:**
```python
def move_to_device(obj: Any, device: torch.device) -> Any
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `obj` | `Any` | Object to move (tensor, model, dict, list, tuple) |
| `device` | `torch.device` | Target device |

**Returns:**

| Type | Description |
|------|-------------|
| `Any` | Object on target device |

**Example:**

```python
from scripts.utils import move_to_device, get_device
import torch

device = get_device()

# Move tensor
tensor = torch.randn(3, 224, 224)
tensor = move_to_device(tensor, device)

# Move dict of tensors
data = {
    'images': torch.randn(8, 3, 224, 224),
    'labels': torch.randint(0, 5, (8,))
}
data = move_to_device(data, device)

# Move list of tensors
batch = [torch.randn(3, 224, 224) for _ in range(8)]
batch = move_to_device(batch, device)
```

---

## Weights & Biases Integration

### wandb_available

Check if Weights & Biases is available.

**Signature:**
```python
def wandb_available() -> bool
```

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if wandb can be imported |

**Example:**

```python
from scripts.utils import wandb_available

if wandb_available():
    print("W&B is available")
    # Use W&B logging
else:
    print("W&B not installed")
    # Skip W&B logging
```

---

### init_wandb

Initialize Weights & Biases run with graceful fallback.

**Signature:**
```python
def init_wandb(
    config: Dict[str, Any],
    project_name: str = 'diabetic-retinopathy',
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    enable_wandb: bool = True,
    **kwargs
) -> bool
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Dict[str, Any]` | Required | Configuration to log |
| `project_name` | `str` | `'diabetic-retinopathy'` | W&B project name |
| `run_name` | `str` | `None` | Run name (auto-generated if None) |
| `tags` | `List[str]` | `None` | Tags for the run |
| `enable_wandb` | `bool` | `True` | Whether to enable W&B |
| `**kwargs` | `Any` | - | Additional arguments for `wandb.init()` |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if initialized successfully |

**Example:**

```python
from scripts.utils import init_wandb

config = {
    'model': 'resnet50',
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 20
}

wandb_enabled = init_wandb(
    config=config,
    project_name='dr-classification',
    run_name='resnet50_exp1',
    tags=['baseline', 'resnet50']
)

if wandb_enabled:
    print("W&B logging enabled")
else:
    print("Training without W&B")

# Output:
# ✓ W&B initialized: resnet50_exp1 (Project: dr-classification)
#   View at: https://wandb.ai/username/dr-classification/runs/abc123
```

---

### log_metrics_wandb

Log metrics to Weights & Biases.

**Signature:**
```python
def log_metrics_wandb(
    metrics: Dict[str, Union[float, int]],
    step: Optional[int] = None,
    prefix: str = ''
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `Dict[str, Union[float, int]]` | Required | Metrics to log |
| `step` | `int` | `None` | Training step/epoch |
| `prefix` | `str` | `''` | Prefix for metric names |

**Example:**

```python
from scripts.utils import log_metrics_wandb

# Log training metrics
log_metrics_wandb(
    metrics={'loss': 0.45, 'accuracy': 85.3},
    step=10,
    prefix='train/'
)

# Log validation metrics
log_metrics_wandb(
    metrics={'loss': 0.52, 'accuracy': 82.1},
    step=10,
    prefix='val/'
)
```

---

### log_images_wandb

Log sample predictions with images to W&B.

**Signature:**
```python
def log_images_wandb(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    step: Optional[int] = None,
    max_images: int = 8,
    denormalize: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | `torch.Tensor` | Required | Images `[B, C, H, W]` |
| `labels` | `torch.Tensor` | Required | Ground truth labels `[B]` |
| `predictions` | `torch.Tensor` | Required | Predicted labels `[B]` |
| `class_names` | `List[str]` | Required | Class names |
| `step` | `int` | `None` | Step/epoch |
| `max_images` | `int` | `8` | Max images to log |
| `denormalize` | `bool` | `True` | Denormalize images |

**Example:**

```python
from scripts.utils import log_images_wandb

class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

# Get sample batch
with torch.no_grad():
    sample_images, sample_labels = next(iter(val_loader))
    sample_images = sample_images.to(device)
    sample_preds = model(sample_images).argmax(dim=1).cpu()
    sample_labels = sample_labels.cpu()

log_images_wandb(
    images=sample_images.cpu(),
    labels=sample_labels,
    predictions=sample_preds,
    class_names=class_names,
    step=10,
    max_images=8
)
```

---

### log_confusion_matrix_wandb

Log confusion matrix as image to W&B.

**Signature:**
```python
def log_confusion_matrix_wandb(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    step: Optional[int] = None,
    normalize: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `np.ndarray` | Required | Ground truth labels |
| `y_pred` | `np.ndarray` | Required | Predicted labels |
| `class_names` | `List[str]` | Required | Class names |
| `step` | `int` | `None` | Step/epoch |
| `normalize` | `bool` | `True` | Normalize matrix |

**Example:**

```python
from scripts.utils import log_confusion_matrix_wandb
import numpy as np

# Collect all predictions
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Log confusion matrix
log_confusion_matrix_wandb(
    y_true=np.array(all_labels),
    y_pred=np.array(all_preds),
    class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
    step=20,
    normalize=True
)
```

---

### log_model_artifact_wandb

Save model as W&B artifact.

**Signature:**
```python
def log_model_artifact_wandb(
    model_path: Union[str, Path],
    artifact_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    artifact_type: str = 'model'
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` or `Path` | Required | Path to model checkpoint |
| `artifact_name` | `str` | Required | Artifact name |
| `metadata` | `Dict[str, Any]` | `None` | Optional metadata |
| `artifact_type` | `str` | `'model'` | Artifact type |

**Example:**

```python
from scripts.utils import log_model_artifact_wandb

log_model_artifact_wandb(
    model_path='checkpoints/best_model.pth',
    artifact_name='resnet50_best_epoch_10',
    metadata={
        'epoch': 10,
        'val_acc': 85.3,
        'val_loss': 0.42,
        'model_name': 'resnet50',
        'num_classes': 5
    }
)
# Output: ✓ Model saved as W&B artifact: resnet50_best_epoch_10
```

---

### finish_wandb

Finish W&B run gracefully.

**Signature:**
```python
def finish_wandb() -> None
```

**Example:**

```python
from scripts.utils import finish_wandb

# At end of training
finish_wandb()
# Output: ✓ W&B run finished
```

---

## Complete Usage Example

```python
#!/usr/bin/env python3
"""Complete workflow using utils module."""

from scripts.utils import (
    set_seed, get_device, get_transforms, create_data_loaders,
    count_parameters, save_checkpoint, calculate_metrics,
    plot_confusion_matrix, plot_training_history,
    init_wandb, log_metrics_wandb, log_confusion_matrix_wandb, finish_wandb
)
from scripts.dataset import RetinalDataset
from scripts.model import DRClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Setup
set_seed(42)
device = get_device()

# 2. Initialize W&B
config = {
    'model': 'resnet50',
    'lr': 1e-4,
    'batch_size': 32,
    'epochs': 10
}

wandb_enabled = init_wandb(
    config=config,
    project_name='dr-classification',
    run_name='complete_example',
    tags=['tutorial']
)

# 3. Data
train_tf, val_tf = get_transforms(224, is_train=True), get_transforms(224, is_train=False)
dataset = RetinalDataset('train.csv', 'images/', transform=train_tf)
train_loader, val_loader = create_data_loaders(dataset, batch_size=32)

# 4. Model
model = DRClassifier('resnet50', num_classes=5).to(device)
total, trainable = count_parameters(model)
print(f"Parameters: {total:,} total, {trainable:,} trainable")

# 5. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 6. Training loop
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(1, 11):
    # Training...
    # (Training code here)

    train_loss, train_acc = 0.5, 80.0  # Example values
    val_loss, val_acc = 0.6, 75.0      # Example values

    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Log to W&B
    if wandb_enabled:
        log_metrics_wandb({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, step=epoch)

    # Save checkpoint
    if epoch % 5 == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'val_acc': val_acc, 'val_loss': val_loss},
            path=f'checkpoints/epoch_{epoch}.pth'
        )

# 7. Evaluation
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 8. Metrics
metrics = calculate_metrics(
    y_true=np.array(all_labels),
    y_pred=np.array(all_preds)
)

print("\nFinal Metrics:")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1-Score: {metrics['f1_macro']:.4f}")
print(f"Kappa: {metrics['cohen_kappa']:.4f}")

# 9. Visualizations
plot_training_history(history, save_path='training_curves.png')
plot_confusion_matrix(
    y_true=np.array(all_labels),
    y_pred=np.array(all_preds),
    save_path='confusion_matrix.png'
)

# 10. Log final confusion matrix to W&B
if wandb_enabled:
    log_confusion_matrix_wandb(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
        step=10
    )
    finish_wandb()

print("\nTraining complete!")
```

---

## See Also

- [Dataset API Documentation](dataset_api.md) - Dataset loading
- [Models API Documentation](models_api.md) - Model architectures
- [Training API Documentation](training_api.md) - Training workflows
- [Evaluation API Documentation](evaluation_api.md) - Model evaluation

---

**Generated with Claude Code** | Last Updated: 2024
