# RETFound Foundation Model Guide

## Overview

This guide covers using the RETFound foundation model for diabetic retinopathy classification. RETFound is a self-supervised Vision Transformer (ViT-Large) pre-trained on 1.6 million retinal images, specifically designed for ophthalmology tasks.

## Table of Contents

1. [What is RETFound?](#what-is-retfound)
2. [Quick Start](#quick-start)
3. [Installation & Setup](#installation--setup)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [Training with RETFound](#training-with-retfound)
7. [**LoRA: Parameter-Efficient Fine-Tuning**](#lora-parameter-efficient-fine-tuning) ⭐ NEW
8. [Comparison with Baselines](#comparison-with-baselines)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## What is RETFound?

**RETFound** is a foundation model for retinal image analysis developed by Zhou et al. (Nature, 2023). Key features:

### Advantages over ImageNet Pre-training

| Feature | ImageNet Models | RETFound |
|---------|----------------|----------|
| **Pre-training Data** | Natural images (cats, dogs, cars) | Retinal images (1.6M fundus photos) |
| **Domain Relevance** | Generic visual features | Ophthalmology-specific features |
| **Transfer Learning** | Requires more data to adapt | Faster convergence on retinal tasks |
| **Performance** | Good baseline | State-of-the-art on retinal datasets |

### Architecture

- **Model**: Vision Transformer Large (ViT-L/16)
- **Parameters**: ~303 million
- **Patch Size**: 16×16 pixels
- **Embedding Dimension**: 1024
- **Transformer Blocks**: 24
- **Attention Heads**: 16
- **Input Size**: 224×224 or 384×384 pixels

### Pre-training Details

- **Dataset**: 1.6M retinal images from multiple sources
- **Method**: Masked Autoencoding (MAE) - self-supervised learning
- **Modalities**: Color Fundus Photography (CFP) and Optical Coherence Tomography (OCT)

---

## Quick Start

### 1. Download RETFound Weights

```bash
# Download from official repository
# Note: Replace with actual download URL
wget https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_cfp_weights.pth

# Move to models directory
mv RETFound_cfp_weights.pth models/
```

### 2. Load and Use the Model

```python
from scripts.retfound_model import load_retfound_model
import torch

# Load model with pretrained weights
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,  # For 5-grade DR classification
    device=torch.device('cuda')
)

# Use for inference
model.eval()
with torch.no_grad():
    outputs = model(images)  # (batch_size, 5)
    predictions = torch.argmax(outputs, dim=1)
```

---

## Installation & Setup

### Prerequisites

Ensure you have the required packages installed:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (should already be installed)
pip install torch torchvision timm
```

### Download RETFound Checkpoints

RETFound provides two types of weights:

1. **CFP (Color Fundus Photography)** - For standard fundus images
2. **OCT (Optical Coherence Tomography)** - For OCT images

For diabetic retinopathy classification, use the CFP weights:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Download CFP weights
# Visit: https://github.com/rmaphoh/RETFound_MAE
# Download: RETFound_cfp_weights.pth
# Place in: models/RETFound_cfp_weights.pth
```

---

## Basic Usage

### Example 1: Load Model for Classification

```python
from scripts.retfound_model import load_retfound_model
import torch

# Load model
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

print(f"Model loaded with {model.get_num_params():,} parameters")
```

### Example 2: Create Model from Scratch

```python
from scripts.retfound_model import get_retfound_vit_large

# Create ViT-Large model without pretrained weights
model = get_retfound_vit_large(
    img_size=224,
    num_classes=5,
    drop_rate=0.1,
    attn_drop_rate=0.1
)
```

### Example 3: Feature Extraction

```python
from scripts.retfound_model import load_retfound_model
import torch

# Load model without classification head
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=0,  # No classification head
    device=torch.device('cuda')
)

# Extract features
model.eval()
with torch.no_grad():
    features = model(images)  # (batch_size, 1024)

# Use features for downstream tasks
# e.g., clustering, visualization, custom classifiers
```

### Example 4: Global Average Pooling

```python
# Use global average pooling instead of CLS token
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    use_global_pool=True  # Average all patch tokens
)
```

---

## Advanced Usage

### Fine-tuning Strategies

#### 1. Full Fine-tuning

Train all parameters:

```python
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5
)

# All parameters are trainable
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

#### 2. Linear Probing

Freeze backbone, train only classification head:

```python
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5
)

# Freeze all parameters except classification head
for name, param in model.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

# Only train classification head
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
```

#### 3. Gradual Unfreezing

Start with frozen backbone, gradually unfreeze layers:

```python
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5
)

# Phase 1: Train only head (5 epochs)
for param in model.parameters():
    param.requires_grad = False
model.head.weight.requires_grad = True
model.head.bias.requires_grad = True

# Phase 2: Unfreeze last 6 blocks (5 epochs)
for i in range(18, 24):  # Blocks 18-23
    for param in model.blocks[i].parameters():
        param.requires_grad = True

# Phase 3: Unfreeze all (10 epochs)
for param in model.parameters():
    param.requires_grad = True
```

#### 4. Layer-wise Learning Rate Decay

Use different learning rates for different layers:

```python
def get_parameter_groups(model, base_lr=1e-4, decay_rate=0.9):
    """Create parameter groups with layer-wise LR decay."""
    parameter_groups = []

    # Classification head (highest LR)
    parameter_groups.append({
        'params': model.head.parameters(),
        'lr': base_lr
    })

    # Transformer blocks (decreasing LR)
    num_blocks = len(model.blocks)
    for i, block in enumerate(reversed(model.blocks)):
        layer_lr = base_lr * (decay_rate ** (num_blocks - i))
        parameter_groups.append({
            'params': block.parameters(),
            'lr': layer_lr
        })

    # Embedding layers (lowest LR)
    parameter_groups.append({
        'params': [model.cls_token, model.pos_embed],
        'lr': base_lr * (decay_rate ** num_blocks)
    })
    parameter_groups.append({
        'params': model.patch_embed.parameters(),
        'lr': base_lr * (decay_rate ** num_blocks)
    })

    return parameter_groups

# Create optimizer with layer-wise LR
param_groups = get_parameter_groups(model, base_lr=1e-4)
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

### Different Image Sizes

RETFound can handle different input sizes:

```python
# Standard size (224x224)
model_224 = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5
)

# Larger size (384x384) - better quality, slower training
from scripts.retfound_model import get_retfound_vit_large

model_384 = get_retfound_vit_large(
    img_size=384,
    num_classes=5
)

# Load pretrained weights (positional embeddings will be interpolated)
checkpoint = torch.load('models/RETFound_cfp_weights.pth')
model_384.load_state_dict(checkpoint['model'], strict=False)
```

---

## Training with RETFound

### Create Training Configuration

Create `configs/retfound_config.yaml`:

```yaml
# RETFound Configuration for DR Classification
model:
  model_name: retfound_vit_large
  num_classes: 5
  pretrained: true
  pretrained_path: "models/RETFound_cfp_weights.pth"
  dropout_rate: 0.1

training:
  batch_size: 16  # Reduce if OOM (ViT-Large is memory intensive)
  num_epochs: 20
  learning_rate: 0.0001  # Lower LR for fine-tuning
  weight_decay: 0.01
  warmup_epochs: 2

  # Optimizer settings
  optimizer: adamw
  scheduler: cosine

  # Early stopping
  patience: 5
  min_delta: 0.001

image:
  input_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

data:
  train_csv: "data/train.csv"
  test_csv: "data/test.csv"
  img_dir: "data/train_images"
  val_split: 0.2
  num_workers: 4

paths:
  output_dir: "results/retfound_baseline"
  checkpoint_dir: "results/retfound_baseline/checkpoints"
  log_dir: "results/retfound_baseline/logs"

system:
  device: "cuda"
  seed: 42
  mixed_precision: true
```

### Training Script for RETFound

Create `scripts/train_retfound.py`:

```python
"""Training script for RETFound model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from scripts.retfound_model import load_retfound_model
from scripts.dataset import RetinalDataset
from scripts.config import Config
from scripts.train_baseline import train_one_epoch, validate, get_transforms

def train_retfound(config: Config):
    """
    Train RETFound model for DR classification.

    Args:
        config: Configuration object
    """
    # Set device
    device = torch.device(config.system.device)

    # Load RETFound model
    print("Loading RETFound model...")
    model = load_retfound_model(
        checkpoint_path=config.model.pretrained_path,
        num_classes=config.model.num_classes,
        device=device
    )

    # Create datasets
    train_transform, val_transform = get_transforms(config)

    train_dataset = RetinalDataset(
        csv_file=config.data.train_csv,
        img_dir=config.data.img_dir,
        transform=train_transform
    )

    val_dataset = RetinalDataset(
        csv_file=config.data.val_csv,
        img_dir=config.data.img_dir,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs
    )

    # Training loop
    best_val_acc = 0.0

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, config.paths.checkpoint_dir / 'best_model.pth')
            print(f"Saved best model with val_acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    config = Config.from_yaml("configs/retfound_config.yaml")
    config.validate(create_dirs=True)
    train_retfound(config)
```

### Run Training

```bash
# Activate environment
source venv/bin/activate

# Train RETFound model
python scripts/train_retfound.py --config configs/retfound_config.yaml
```

---

## LoRA: Parameter-Efficient Fine-Tuning

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that enables training large models with <1% of the original parameters. Instead of updating all weights, LoRA adds small trainable "adapter" matrices that modify the model's behavior.

#### Key Advantages

| Metric | Full Fine-Tuning | LoRA Fine-Tuning |
|--------|------------------|------------------|
| **Trainable Parameters** | 303M (100%) | ~800K (0.26%) |
| **Training Memory** | 4.5 GB | 0.01 GB |
| **Training Speed** | 1x | 2-3x faster |
| **Storage per Checkpoint** | 1.2 GB | 3 MB |
| **Parameter Efficiency** | 1x | **383x** |

#### How LoRA Works

Instead of updating weight matrix W directly:
```
W_new = W_old + ΔW
```

LoRA decomposes the update into low-rank matrices:
```
W_new = W_old + B × A
```

Where:
- `W`: Original weights (frozen)
- `B, A`: Low-rank matrices (trainable)
- `rank r << dimension d`

For a 1024×3072 attention layer:
- Original: 3,145,728 parameters
- LoRA (r=8): 16,384 parameters (0.5%!)

### Quick Start with LoRA

#### 1. Create LoRA Model

```python
from scripts.retfound_lora import RETFoundLoRA

# Create LoRA-adapted RETFound
model = RETFoundLoRA(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    lora_r=8,              # Rank (capacity)
    lora_alpha=32,         # Scaling factor
    lora_dropout=0.1       # Regularization
)

# Shows: "trainable params: 793,605 || all params: 304,095,237 || trainable%: 0.26%"
```

#### 2. Train with LoRA

```python
import torch.optim as optim

# Only LoRA adapters and classifier are trainable
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,              # Higher LR than full fine-tuning
    weight_decay=0.01
)

# Training loop (same as usual)
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

#### 3. Use Configuration File

```bash
# Train with LoRA configuration
python3 scripts/train_baseline.py --config configs/retfound_lora_config.yaml
```

### LoRA Hyperparameters

#### Rank (r)

Controls the capacity of LoRA adapters:

| Rank | Trainable Params | Use Case |
|------|-----------------|----------|
| r=4  | ~400K (0.13%)  | Large datasets, limited compute |
| **r=8** | **~800K (0.26%)** | **Recommended for most cases** |
| r=16 | ~1.6M (0.52%)  | Small datasets, complex tasks |
| r=32 | ~3.2M (1.03%)  | Maximum capacity, near full fine-tuning |

#### Alpha (α)

Scaling factor for LoRA updates:
- Common practice: `alpha = 4 × rank`
- Higher α = stronger LoRA influence
- Lower α = more conservative updates

```python
# Conservative (alpha = 2*r)
model = RETFoundLoRA(lora_r=8, lora_alpha=16)

# Balanced (alpha = 4*r) - RECOMMENDED
model = RETFoundLoRA(lora_r=8, lora_alpha=32)

# Aggressive (alpha = 8*r)
model = RETFoundLoRA(lora_r=8, lora_alpha=64)
```

#### Target Modules

Which layers to adapt:

```python
# QKV only (default, most efficient)
model = RETFoundLoRA(target_modules=["qkv"])
# ~800K params, focuses on attention

# QKV + output projection (more capacity)
model = RETFoundLoRA(target_modules=["qkv", "proj"])
# ~1.05M params, adapts full attention
```

### Training Strategies with LoRA

#### Strategy 1: Direct LoRA Fine-Tuning (Recommended)

Train LoRA adapters and classifier from scratch:

```python
model = RETFoundLoRA(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    lora_r=8,
    lora_alpha=32
)

# All LoRA + classifier parameters are trainable by default
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Train for 15-20 epochs
```

**Advantages:**
- Simple and effective
- Fast convergence (LoRA adapters learn quickly)
- Best for most use cases

#### Strategy 2: Two-Stage Training

First train classifier, then add LoRA:

```python
# Stage 1: Train classifier only (5 epochs)
for name, param in model.backbone.named_parameters():
    param.requires_grad = False  # Freeze all backbone

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
# Train...

# Stage 2: Unfreeze LoRA and continue (10 epochs)
for name, param in model.backbone.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
# Continue training...
```

**Advantages:**
- Stable classifier initialization
- Good for noisy or imbalanced datasets

#### Strategy 3: Progressive Rank Increase

Start with low rank, gradually increase:

```python
# Phase 1: r=4 (5 epochs)
model = RETFoundLoRA(lora_r=4, lora_alpha=16)
# Train...

# Phase 2: r=8 (5 epochs)
model = RETFoundLoRA(lora_r=8, lora_alpha=32)
model.load_state_dict(checkpoint)  # Load from phase 1
# Continue training...

# Phase 3: r=16 (5 epochs)
model = RETFoundLoRA(lora_r=16, lora_alpha=64)
# Continue training...
```

**Advantages:**
- Prevents overfitting
- Explores different capacity levels

### LoRA Configuration Example

Complete configuration for LoRA training (`configs/retfound_lora_config.yaml`):

```yaml
model:
  model_name: retfound_lora
  num_classes: 5
  pretrained_path: "models/RETFound_cfp_weights.pth"

  # LoRA parameters
  lora_r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  head_dropout: 0.3
  target_modules: ["qkv"]

training:
  batch_size: 32              # Larger batch (LoRA uses less memory)
  num_epochs: 20
  learning_rate: 0.0005       # Higher LR than full fine-tuning
  weight_decay: 0.01

  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 2

  mixed_precision: true       # Faster training

image:
  input_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

data:
  train_csv: "data/train.csv"
  val_split: 0.2
  img_dir: "data/train_images"

paths:
  output_dir: "results/retfound_lora"
  checkpoint_dir: "results/retfound_lora/checkpoints"
```

### Saving and Loading LoRA Models

#### Save Only LoRA Adapters (Efficient)

```python
# Save only LoRA adapters (~3 MB vs 1.2 GB full model)
model.save_lora_adapters('checkpoints/lora_dr_epoch_20.pth')
```

#### Load LoRA Adapters

```python
# Create base model
model = RETFoundLoRA(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    lora_r=8
)

# Load LoRA adapters
checkpoint = torch.load('checkpoints/lora_dr_epoch_20.pth')
model.backbone.load_state_dict(checkpoint['lora_adapters'], strict=False)
model.classifier.load_state_dict(checkpoint['classifier'])
```

#### Share Multiple LoRA Adapters

You can train multiple LoRA adapters for different tasks and switch between them:

```python
# Base RETFound model (1.2 GB)
base_checkpoint = 'models/RETFound_cfp_weights.pth'

# LoRA adapters for different datasets (3 MB each)
lora_aptos = 'lora_aptos.pth'
lora_messidor = 'lora_messidor.pth'
lora_eyepacs = 'lora_eyepacs.pth'

# Switch adapters on the fly
model = RETFoundLoRA(base_checkpoint, lora_r=8)
model.load_lora_adapters(lora_aptos)     # Use APTOS adapter
# Inference on APTOS data...

model.load_lora_adapters(lora_messidor)  # Switch to Messidor
# Inference on Messidor data...
```

### Performance Comparison: LoRA vs Full Fine-Tuning

#### Expected Accuracy

On diabetic retinopathy classification:

| Method | Accuracy | Parameters | Training Time | Memory |
|--------|----------|-----------|---------------|---------|
| Linear Probing | ~82% | 5K | 0.5x | 0.5x |
| LoRA (r=4) | ~85% | 400K | 1.5x | 0.7x |
| **LoRA (r=8)** | **~87%** | **800K** | **1.0x** | **1.0x** |
| LoRA (r=16) | ~88% | 1.6M | 1.2x | 1.5x |
| Full Fine-Tuning | ~88-89% | 303M | 3x | 4.5x |

**Observation:** LoRA (r=8) achieves 97% of full fine-tuning performance with 0.26% of parameters!

#### Training Speed

On NVIDIA V100 GPU, batch size optimized for each method:

| Method | Batch Size | Time/Epoch | GPU Memory | Convergence |
|--------|-----------|-----------|------------|-------------|
| Full FT | 16 | 8 min | 12 GB | 20 epochs |
| LoRA (r=8) | 32 | 5 min | 8 GB | 15 epochs |
| LoRA (r=16) | 24 | 6 min | 10 GB | 18 epochs |

### When to Use LoRA

#### ✅ Use LoRA When:

- Limited GPU memory (< 16 GB)
- Want to train multiple adapters for different datasets
- Need fast iteration during experimentation
- Working with small datasets (LoRA prevents overfitting)
- Want to share models efficiently (base model + adapters)

#### ❌ Use Full Fine-Tuning When:

- Have abundant compute resources (>= 32 GB GPU)
- Need absolute maximum performance (88% vs 87%)
- Dataset is very large (>100K images)
- Task is very different from pretraining (unlikely for DR)

### Best Practices for LoRA

1. **Start with r=8**: Good balance between capacity and efficiency
2. **Use higher learning rate**: 2-5x higher than full fine-tuning
3. **Larger batch sizes**: LoRA uses less memory, so increase batch size
4. **Warm up learning rate**: 1-2 epochs of warmup for stable training
5. **Monitor overfitting**: LoRA can overfit on very small datasets, use dropout
6. **Save adapters separately**: Much more storage-efficient
7. **Experiment with rank**: Try r=4, 8, 16 to find best trade-off

### Troubleshooting LoRA

#### Issue 1: LoRA Not Learning

**Symptoms**: Training loss doesn't decrease, validation accuracy stays low

**Solutions**:
```python
# 1. Increase learning rate
config.training.learning_rate = 1e-3  # Try 2-10x higher

# 2. Increase LoRA rank
model = RETFoundLoRA(lora_r=16, lora_alpha=64)

# 3. Check gradients are flowing
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

#### Issue 2: Overfitting Quickly

**Symptoms**: Training accuracy high, validation accuracy low

**Solutions**:
```python
# 1. Increase dropout
model = RETFoundLoRA(lora_dropout=0.2, head_dropout=0.5)

# 2. Use lower rank
model = RETFoundLoRA(lora_r=4, lora_alpha=16)

# 3. Add weight decay
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

# 4. Use more data augmentation
config.image.augmentation.coarse_dropout.p = 0.7
```

#### Issue 3: Performance Below Expectations

**Symptoms**: LoRA accuracy significantly below full fine-tuning

**Solutions**:
```python
# 1. Increase rank gradually
model = RETFoundLoRA(lora_r=16, lora_alpha=64)  # or r=32

# 2. Target more modules
model = RETFoundLoRA(target_modules=["qkv", "proj"])

# 3. Two-stage training
# Stage 1: Train classifier only
# Stage 2: Add LoRA adapters

# 4. Longer training
config.training.num_epochs = 30  # LoRA might need more epochs
```

---

## Comparison with Baselines

### Performance Comparison

Expected performance on diabetic retinopathy classification:

| Model | Parameters | ImageNet Pre-trained | Retinal Pre-trained | Expected Accuracy |
|-------|-----------|---------------------|---------------------|-------------------|
| ResNet50 | 25M | ✓ | ✗ | ~75-80% |
| EfficientNet-B4 | 19M | ✓ | ✗ | ~78-82% |
| ViT-Base | 86M | ✓ | ✗ | ~80-84% |
| **RETFound (ViT-Large)** | **303M** | ✗ | **✓** | **~85-90%** |

### Training Time Comparison

On NVIDIA V100 GPU with batch size 16:

| Model | Training Time per Epoch | Inference Time (batch=16) |
|-------|------------------------|---------------------------|
| ResNet50 | ~2 minutes | ~50ms |
| EfficientNet-B4 | ~4 minutes | ~80ms |
| ViT-Base | ~5 minutes | ~100ms |
| RETFound (ViT-Large) | ~8 minutes | ~150ms |

### Memory Requirements

| Model | Training Memory (batch=16) | Inference Memory (batch=16) |
|-------|---------------------------|----------------------------|
| ResNet50 | ~4 GB | ~2 GB |
| EfficientNet-B4 | ~6 GB | ~3 GB |
| ViT-Base | ~8 GB | ~4 GB |
| RETFound (ViT-Large) | ~12 GB | ~6 GB |

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# 1. Reduce batch size
config.training.batch_size = 8  # or 4

# 2. Use gradient accumulation
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. Use smaller image size
config.image.input_size = 224  # instead of 384
```

### Issue 2: Checkpoint Loading Errors

**Error**: `Missing keys in checkpoint` or `Unexpected keys`

**Solutions**:

```python
# 1. Use strict=False (default in load_retfound_model)
model = load_retfound_model(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    strict=False  # Ignore missing/unexpected keys
)

# 2. Manual checkpoint loading
checkpoint = torch.load('models/RETFound_cfp_weights.pth')
state_dict = checkpoint['model']  # or checkpoint['state_dict']

# Remove prefixes if needed
from scripts.retfound_model import _clean_state_dict_keys
state_dict = _clean_state_dict_keys(state_dict)

model.load_state_dict(state_dict, strict=False)
```

### Issue 3: Slow Training

**Problem**: Training takes too long

**Solutions**:

1. **Enable Mixed Precision**:
   ```python
   config.system.mixed_precision = True
   ```

2. **Increase Batch Size** (if memory allows):
   ```python
   config.training.batch_size = 32
   ```

3. **Use Multiple GPUs**:
   ```python
   model = nn.DataParallel(model)
   ```

4. **Optimize Data Loading**:
   ```python
   config.data.num_workers = 8  # More workers
   config.data.pin_memory = True
   config.data.persistent_workers = True
   ```

### Issue 4: Poor Convergence

**Problem**: Model not learning or overfitting

**Solutions**:

```python
# 1. Adjust learning rate
config.training.learning_rate = 1e-5  # Lower for fine-tuning

# 2. Use warmup
warmup_epochs = 2
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=warmup_epochs
)

# 3. Add regularization
config.model.dropout_rate = 0.2  # Increase dropout
config.training.weight_decay = 0.05  # Increase weight decay

# 4. Use data augmentation
from scripts.train_baseline import get_transforms
train_transform, _ = get_transforms(config, augment=True)

# 5. Try different fine-tuning strategy
# Start with linear probing, then full fine-tuning
```

---

## References

### Papers

1. **RETFound Paper**:
   - Zhou, Y., et al. (2023). "A Foundation Model for Generalizable Disease Detection from Retinal Images." *Nature*.
   - [Link to paper](https://www.nature.com/articles/s41586-023-06555-x)

2. **Vision Transformer**:
   - Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
   - [Link to paper](https://arxiv.org/abs/2010.11929)

3. **Masked Autoencoding**:
   - He, K., et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*.
   - [Link to paper](https://arxiv.org/abs/2111.06377)

### Code & Resources

- **Official RETFound Repository**: [https://github.com/rmaphoh/RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE)
- **Timm Library**: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

### Datasets

- **Kaggle Diabetic Retinopathy Detection**: [https://www.kaggle.com/c/diabetic-retinopathy-detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **APTOS 2019 Blindness Detection**: [https://www.kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **MESSIDOR**: Public diabetic retinopathy dataset

---

## Best Practices

### 1. Data Preparation

- **Image Quality**: Ensure high-quality retinal images (minimal artifacts)
- **Consistency**: Use images from same modality (CFP or OCT)
- **Preprocessing**: Apply CLAHE for contrast enhancement
- **Cropping**: Remove black borders and focus on retinal region

### 2. Training Strategy

- **Start Small**: Begin with linear probing (freeze backbone)
- **Gradual Unfreezing**: Slowly unfreeze layers from top to bottom
- **Low Learning Rate**: Use 10-100x smaller LR than training from scratch
- **Warmup**: Use 1-2 epochs of warmup for stable training

### 3. Hyperparameters

- **Batch Size**: 16-32 (depends on GPU memory)
- **Learning Rate**: 1e-5 to 1e-4 for fine-tuning
- **Weight Decay**: 0.01-0.05
- **Dropout**: 0.1-0.2
- **Epochs**: 10-30 (foundation models converge faster)

### 4. Validation

- **Cross-validation**: Use k-fold CV for robust evaluation
- **Stratification**: Ensure balanced class distribution
- **Multiple Metrics**: Track accuracy, F1, AUC, sensitivity, specificity
- **Confusion Matrix**: Analyze per-class performance

---

## Next Steps

After implementing RETFound:

1. **Download Weights**: Get official RETFound checkpoint
2. **Prepare Data**: Organize your DR dataset
3. **Baseline Training**: Train baseline models (ResNet, EfficientNet)
4. **RETFound Training**: Train RETFound with different strategies
5. **Comparison**: Compare performance, training time, and efficiency
6. **Hyperparameter Tuning**: Optimize for your specific dataset
7. **Ensemble**: Combine RETFound with other models
8. **Deployment**: Export best model for clinical use

---

**For more information**:
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for general training tips
- See [MODEL_GUIDE.md](MODEL_GUIDE.md) for baseline model usage
- See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for config management
