# Training API Documentation

## Overview

The training module provides complete training pipelines for diabetic retinopathy classification models, including baseline models and parameter-efficient LoRA fine-tuning.

**Key Features:**
- Complete training loop with validation
- Automatic checkpointing and model saving
- Learning rate scheduling
- Data augmentation pipelines
- Weights & Biases integration
- Resume training from checkpoints
- Mixed precision training (for LoRA)

**Source Files:**
- [scripts/train_baseline.py](../scripts/train_baseline.py:1) - Baseline training
- [scripts/train_retfound_lora.py](../scripts/train_retfound_lora.py:1) - LoRA training

---

## Table of Contents

1. [Core Training Functions](#core-training-functions)
2. [Data Preparation](#data-preparation)
3. [Training Utilities](#training-utilities)
4. [Complete Workflow Examples](#complete-workflow-examples)
5. [Command Line Usage](#command-line-usage)

---

## Core Training Functions

### train

Main training function for complete training workflow.

**Signature:**
```python
def train(
    config: Config,
    resume_checkpoint: Optional[str] = None,
    enable_wandb: bool = False,
    wandb_project: str = 'diabetic-retinopathy',
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None
) -> Dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Config` | Required | Configuration object with model, data, and training settings |
| `resume_checkpoint` | `str` | `None` | Path to checkpoint to resume training from |
| `enable_wandb` | `bool` | `False` | Enable Weights & Biases logging |
| `wandb_project` | `str` | `'diabetic-retinopathy'` | W&B project name |
| `wandb_run_name` | `str` | `None` | W&B run name (auto-generated if None) |
| `wandb_tags` | `List[str]` | `None` | Tags for W&B run |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict` | Training history with keys: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `learning_rate`, `epoch_time` |

**Example:**

```python
from scripts.config import Config
from scripts.train_baseline import train

# Load configuration
config = Config.from_yaml('configs/default_config.yaml')

# Train model
history = train(
    config=config,
    enable_wandb=True,
    wandb_project='dr-classification',
    wandb_tags=['baseline', 'resnet50']
)

# Access results
print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
print(f"Best val accuracy: {max(history['val_acc']):.2f}%")
```

---

### train_one_epoch

Train model for one epoch.

**Signature:**
```python
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to train |
| `train_loader` | `DataLoader` | Training data loader |
| `criterion` | `nn.Module` | Loss function (e.g., CrossEntropyLoss) |
| `optimizer` | `optim.Optimizer` | Optimizer (e.g., Adam, AdamW) |
| `device` | `torch.device` | Device for training ('cuda' or 'cpu') |
| `epoch` | `int` | Current epoch number (for progress display) |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, float]` | Dictionary with keys `train_loss` and `train_acc` |

**Example:**

```python
import torch.nn as nn
import torch.optim as optim

model = DRClassifier('resnet50').to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train for one epoch
metrics = train_one_epoch(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epoch=1
)

print(f"Train Loss: {metrics['train_loss']:.4f}")
print(f"Train Acc: {metrics['train_acc']:.2f}%")
```

---

### validate

Validate model on validation set.

**Signature:**
```python
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to validate |
| `val_loader` | `DataLoader` | Validation data loader |
| `criterion` | `nn.Module` | Loss function |
| `device` | `torch.device` | Device for validation |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, float]` | Dictionary with keys `val_loss`, `val_acc`, and `class_i_acc` for each class |

**Example:**

```python
model.eval()
val_metrics = validate(
    model=model,
    val_loader=val_loader,
    criterion=criterion,
    device=device
)

print(f"Val Loss: {val_metrics['val_loss']:.4f}")
print(f"Val Acc: {val_metrics['val_acc']:.2f}%")

# Per-class accuracy
for i in range(5):
    class_acc = val_metrics.get(f'class_{i}_acc', 0)
    print(f"  Class {i}: {class_acc:.2f}%")
```

---

## Data Preparation

### get_transforms

Create training and validation transforms using Albumentations.

**Signature:**
```python
def get_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|---------|
| `img_size` | `int` | Target image size (square) |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[A.Compose, A.Compose]` | `(train_transform, val_transform)` |

**Training Augmentations:**
- Resize to target size
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- RandomRotate90 (p=0.5)
- ShiftScaleRotate (shift=0.1, scale=0.1, rotate=45°, p=0.5)
- Color jitter / brightness contrast (p=0.5)
- CoarseDropout (8 holes, 32×32, p=0.3)
- ImageNet normalization

**Validation Transforms:**
- Resize to target size
- ImageNet normalization only

**Example:**

```python
from scripts.train_baseline import get_transforms

# Get transforms for 224×224 images
train_transform, val_transform = get_transforms(224)

# Use with dataset
train_dataset = RetinalDataset(
    csv_file='train.csv',
    img_dir='images/',
    transform=train_transform
)

val_dataset = RetinalDataset(
    csv_file='val.csv',
    img_dir='images/',
    transform=val_transform
)
```

---

### create_data_loaders

Create training and validation data loaders with 80/20 split.

**Signature:**
```python
def create_data_loaders(
    config: Config,
    train_transform: A.Compose,
    val_transform: A.Compose
) -> Tuple[DataLoader, DataLoader]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Config` | Configuration object |
| `train_transform` | `A.Compose` | Training augmentation pipeline |
| `val_transform` | `A.Compose` | Validation augmentation pipeline |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[DataLoader, DataLoader]` | `(train_loader, val_loader)` |

**Example:**

```python
from scripts.config import Config
from scripts.train_baseline import get_transforms, create_data_loaders

config = Config.from_yaml('configs/default_config.yaml')
train_transform, val_transform = get_transforms(config.image.img_size)

train_loader, val_loader = create_data_loaders(
    config=config,
    train_transform=train_transform,
    val_transform=val_transform
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

---

## Training Utilities

### set_seed

Set random seeds for reproducibility.

**Signature:**
```python
def set_seed(seed: int) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `int` | Random seed value (e.g., 42) |

**Sets seeds for:**
- Python random
- NumPy
- PyTorch (CPU and CUDA)
- CUDNN (deterministic mode)

**Example:**

```python
from scripts.train_baseline import set_seed

set_seed(42)  # For reproducible results
```

---

### save_checkpoint

Save model checkpoint.

**Signature:**
```python
def save_checkpoint(
    checkpoint_dict: Dict,
    filepath: Path,
    is_best: bool = False
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dict` | `Dict` | Required | Dictionary containing model state, optimizer state, and metadata |
| `filepath` | `Path` | Required | Path to save checkpoint |
| `is_best` | `bool` | `False` | If True, also save as 'best_model.pth' |

**Checkpoint Structure:**
```python
checkpoint_dict = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'train_loss': float,
    'train_acc': float,
    'val_loss': float,
    'val_acc': float,
    'best_acc': float,
    'history': dict,
    'config': dict,
    'model_name': str,
    'num_classes': int
}
```

**Example:**

```python
from scripts.train_baseline import save_checkpoint
from pathlib import Path

checkpoint_dict = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 0.35,
    'train_acc': 87.5,
    'val_loss': 0.42,
    'val_acc': 85.3,
    'best_acc': 85.3,
    'history': history,
    'config': config.to_dict(),
    'model_name': 'resnet50',
    'num_classes': 5
}

save_checkpoint(
    checkpoint_dict=checkpoint_dict,
    filepath=Path('checkpoints/checkpoint_epoch_10.pth'),
    is_best=True
)
# Saves to:
#   - checkpoints/checkpoint_epoch_10.pth
#   - checkpoints/best_model.pth (if is_best=True)
```

---

### save_training_history

Save training history to JSON file.

**Signature:**
```python
def save_training_history(history: Dict, filepath: Path) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `history` | `Dict` | Training history dictionary |
| `filepath` | `Path` | Path to save JSON file |

**Example:**

```python
from scripts.train_baseline import save_training_history
from pathlib import Path

history = {
    'train_loss': [0.8, 0.6, 0.5, 0.4],
    'train_acc': [65.0, 75.0, 80.0, 85.0],
    'val_loss': [0.9, 0.7, 0.6, 0.5],
    'val_acc': [60.0, 70.0, 75.0, 80.0],
    'learning_rate': [1e-4, 1e-4, 5e-5, 5e-5],
    'epoch_time': [120, 118, 115, 117]
}

save_training_history(
    history=history,
    filepath=Path('logs/training_history.json')
)
```

---

## Complete Workflow Examples

### Example 1: Training Baseline Model from Scratch

```python
#!/usr/bin/env python3
"""Complete baseline training workflow."""

from scripts.config import Config
from scripts.train_baseline import train
from scripts.model import DRClassifier
import torch

# Step 1: Load configuration
config = Config.from_yaml('configs/default_config.yaml')

# Step 2: Override settings if needed
config.training.num_epochs = 20
config.training.batch_size = 32
config.training.learning_rate = 1e-4
config.model.model_name = 'resnet50'

# Step 3: Train model
print("Starting training...")
history = train(
    config=config,
    enable_wandb=True,
    wandb_project='dr-classification',
    wandb_run_name='resnet50_baseline',
    wandb_tags=['baseline', 'resnet50', 'aptos']
)

# Step 4: Analyze results
print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)

best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
best_acc = max(history['val_acc'])
final_acc = history['val_acc'][-1]

print(f"Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch})")
print(f"Final validation accuracy: {final_acc:.2f}%")
print(f"Total training time: {sum(history['epoch_time']) / 60:.1f} minutes")

# Step 5: Load best model for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DRClassifier('resnet50', num_classes=5)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"\nBest model loaded from epoch {checkpoint['epoch']}")
print(f"Ready for inference!")
```

### Example 2: Training RETFound with LoRA

```python
#!/usr/bin/env python3
"""Complete LoRA training workflow."""

from scripts.config import Config
from scripts.retfound_lora import RETFoundLoRA
from scripts.dataset import RetinalDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
CHECKPOINT_PATH = 'models/RETFound_cfp_weights.pth'
NUM_EPOCHS = 15
BATCH_SIZE = 48
LEARNING_RATE = 1e-4
LORA_R = 8
LORA_ALPHA = 32

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Create LoRA model
print("\n[1/5] Creating LoRA model...")
model = RETFoundLoRA(
    checkpoint_path=CHECKPOINT_PATH,
    num_classes=5,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    device=device
)

# Show parameter efficiency
model.print_parameter_summary()

# Step 2: Prepare data
print("\n[2/5] Preparing data...")

# Transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# Dataset
full_dataset = RetinalDataset(
    csv_file='data/aptos/train.csv',
    img_dir='data/aptos/train_images',
    transform=None
)

# Split
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train samples: {train_size}, Val samples: {val_size}")

# Step 3: Training setup
print("\n[3/5] Setting up training...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    verbose=True
)

# Step 4: Training loop
print(f"\n[4/5] Training for {NUM_EPOCHS} epochs...")

best_val_acc = 0.0
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-" * 50)

    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{train_loss / (pbar.n + 1):.4f}',
            'acc': f'{100.0 * train_correct / train_total:.2f}%'
        })

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # Calculate metrics
    train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Print results
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%", end='')

    # Save best model
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        model.save_lora_adapters('checkpoints/lora_best.pth')
        print(" ✓ (Best!)")
    else:
        print()

    # Update learning rate
    scheduler.step(val_acc)

    # Save checkpoint
    if epoch % 5 == 0:
        model.save_lora_adapters(f'checkpoints/lora_epoch_{epoch}.pth')

# Step 5: Final results
print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Final validation accuracy: {val_acc:.2f}%")
print(f"\nBest model saved to: checkpoints/lora_best.pth")
```

### Example 3: Resume Training from Checkpoint

```python
from scripts.config import Config
from scripts.train_baseline import train

# Load configuration
config = Config.from_yaml('configs/default_config.yaml')

# Resume from interrupted checkpoint
history = train(
    config=config,
    resume_checkpoint='checkpoints/checkpoint_interrupted.pth',
    enable_wandb=True,
    wandb_project='dr-classification',
    wandb_run_name='resnet50_resumed'
)

# Training will continue from the saved epoch
print(f"Resumed training complete!")
print(f"Final accuracy: {history['val_acc'][-1]:.2f}%")
```

---

## Command Line Usage

### Training Baseline Model

```bash
# Basic training
python scripts/train_baseline.py \
    --config configs/default_config.yaml

# With W&B logging
python scripts/train_baseline.py \
    --config configs/default_config.yaml \
    --wandb \
    --wandb-project dr-classification \
    --wandb-run-name resnet50_exp1 \
    --wandb-tags baseline resnet50

# Resume training
python scripts/train_baseline.py \
    --config configs/default_config.yaml \
    --resume checkpoints/checkpoint_epoch_10.pth \
    --wandb

# Debug mode (print full config)
python scripts/train_baseline.py \
    --config configs/default_config.yaml \
    --debug
```

### Training RETFound with LoRA

```bash
# Basic LoRA training
python scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --config configs/retfound_lora_config.yaml

# Custom LoRA parameters
python scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --lora_r 16 \
    --lora_alpha 64 \
    --batch_size 48 \
    --epochs 15 \
    --learning_rate 1e-4

# With W&B logging
python scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --config configs/retfound_lora_config.yaml \
    --wandb \
    --wandb-project dr-lora \
    --wandb-run-name lora_r8_exp1

# Resume LoRA training
python scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --resume checkpoints/lora_epoch_5.pth
```

---

## See Also

- [Models API Documentation](models_api.md) - Model architectures
- [Dataset API Documentation](dataset_api.md) - Dataset loading
- [Evaluation API Documentation](evaluation_api.md) - Model evaluation
- [Utils API Documentation](utils_api.md) - Utility functions including W&B integration

---

**Generated with Claude Code** | Last Updated: 2024
