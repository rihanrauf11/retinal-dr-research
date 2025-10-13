# Training Guide for Diabetic Retinopathy Classification

## Overview

This guide provides comprehensive instructions for training diabetic retinopathy classification models using the baseline training script. The training pipeline integrates all project components (dataset, model, configuration) into a production-ready workflow.

## Quick Start

### 1. Prepare Your Data

Organize your data as follows:

```
data/
└── aptos/  (or your dataset name)
    ├── train.csv          # CSV with 'id_code' and 'diagnosis' columns
    └── train_images/      # Directory containing .png/.jpg images
```

**CSV Format:**
```csv
id_code,diagnosis
image_001,0
image_002,2
image_003,1
...
```

### 2. Create Configuration

Copy and edit the example configuration:

```bash
cp configs/train_example.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your paths and hyperparameters
```

### 3. Start Training

```bash
python scripts/train_baseline.py --config configs/my_experiment.yaml
```

## Training Script Features

### ✓ Complete Pipeline
- Automatic train/validation split (80/20)
- Data augmentation for training
- Progress bars with real-time metrics
- Automatic checkpointing
- Best model tracking
- Training history logging

### ✓ Data Augmentation

**Training Augmentations:**
- Resize to target size
- Random horizontal/vertical flips
- Random 90-degree rotations
- Shift, scale, and rotate
- Color jittering
- Random brightness/contrast
- Coarse dropout (cutout)

**Validation:**
- Resize only (no augmentation)

### ✓ Checkpointing

Automatically saves:
- Checkpoint every epoch: `checkpoint_epoch_N.pth`
- Best model: `best_model.pth`
- On interrupt: `checkpoint_interrupted.pth`

Each checkpoint contains:
- Model weights
- Optimizer state
- Current epoch
- Best accuracy
- Training history
- Configuration snapshot

### ✓ Training History

Tracks per epoch:
- Training loss and accuracy
- Validation loss and accuracy
- Per-class accuracy
- Learning rate
- Epoch time

Saved to: `results/logs/training_history_TIMESTAMP.json`

## Usage Examples

### Basic Training

```bash
python scripts/train_baseline.py --config configs/default_config.yaml
```

### Resume Training

```bash
python scripts/train_baseline.py \
    --config configs/my_experiment.yaml \
    --resume results/checkpoints/checkpoint_epoch_10.pth
```

### Debug Mode

```bash
python scripts/train_baseline.py \
    --config configs/my_experiment.yaml \
    --debug
```

## Configuration Guide

### Model Selection

Choose based on your requirements:

| Model | Parameters | Speed | Accuracy | Memory |
|-------|-----------|-------|----------|--------|
| `resnet34` | 21M | Fast | Good | Low |
| `resnet50` | 24M | Medium | Good | Medium |
| `efficientnet_b0` | 4M | Fast | Good | Low |
| `efficientnet_b3` | 11M | Medium | Better | Medium |
| `vit_base_patch16_224` | 87M | Slow | Best | High |

### Hyperparameter Guidelines

#### Batch Size
- **16**: Safe default for most GPUs
- **32**: If you have 8GB+ GPU memory
- **8**: For large models or limited memory

#### Learning Rate
- **1e-3**: Training from scratch
- **1e-4**: Fine-tuning pretrained models (recommended)
- **5e-5**: Fine-tuning large models

#### Number of Epochs
- **20-30**: Quick experiments
- **30-50**: Standard training
- **50-100**: Maximum performance

#### Image Size
- **224**: ResNet, EfficientNet-B0-B2
- **300**: EfficientNet-B3-B5
- **384**: ViT, larger models
- **512**: High-resolution training

## Training Strategies

### Strategy 1: Standard Training

Use pretrained weights and train end-to-end:

```yaml
model:
  model_name: resnet50
  pretrained: true

training:
  batch_size: 16
  num_epochs: 30
  learning_rate: 0.0001
```

```bash
python scripts/train_baseline.py --config configs/standard.yaml
```

### Strategy 2: Two-Phase Training

**Phase 1: Train classifier head (freeze backbone)**

Modify the training script or use lower learning rate for backbone layers.

**Phase 2: Fine-tune entire model**

Use lower learning rate for full fine-tuning:

```yaml
training:
  learning_rate: 0.00001  # Lower LR for fine-tuning
  num_epochs: 20
```

### Strategy 3: Different Architectures

Try multiple architectures:

```bash
# ResNet50
python scripts/train_baseline.py --config configs/resnet50.yaml

# EfficientNet-B3
python scripts/train_baseline.py --config configs/efficientnet_b3.yaml

# Vision Transformer
python scripts/train_baseline.py --config configs/vit_base.yaml
```

### Strategy 4: Hyperparameter Search

Create configs with different hyperparameters:

```bash
# Different learning rates
for lr in 1e-5 5e-5 1e-4 5e-4; do
    # Modify config
    python scripts/train_baseline.py --config configs/lr_${lr}.yaml
done
```

## Output Structure

After training, you'll have:

```
results/
├── checkpoints/
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   ├── ...
│   └── best_model.pth              # Best model
└── logs/
    └── training_history_*.json     # Training metrics
```

## Training Output

### Console Output Example

```
╔══════════════════════════════════════════════════════════════════════════╗
║         Diabetic Retinopathy Classification - Training                   ║
╚══════════════════════════════════════════════════════════════════════════╝

[INFO] Random seed set to: 42
[INFO] Using device: cuda

[INFO] Loading dataset...
[INFO] Total samples: 3662
[INFO] Train/Val split: 2929/733 (80/20)

[INFO] Class distribution:
  Class 0 (No DR          ):  1805 (61.6%)
  Class 1 (Mild NPDR      ):   370 (12.6%)
  Class 2 (Moderate NPDR  ):   999 (34.1%)
  Class 3 (Severe NPDR    ):   193 ( 6.6%)
  Class 4 (PDR            ):   295 (10.1%)

[INFO] Creating model: resnet50
✓ Created DRClassifier with backbone: resnet50
  - Feature dimension: 2048
  - Output classes: 5
  - Dropout rate: 0.3
  - Pretrained: True
[INFO] Total parameters: 23,518,277
[INFO] Trainable parameters: 23,518,277

[INFO] Starting training for 30 epochs...
================================================================================

Epoch 1/30
--------------------------------------------------------------------------------
Epoch 1 [Train]: 100%|████████| 184/184 [02:15<00:00, 1.36it/s, loss=1.234, acc=56.78%]
Validating: 100%|████████████████| 46/46 [00:18<00:00, 2.50it/s, loss=1.123, acc=58.45%]

Results:
  Train Loss: 1.2340  Train Acc: 56.78%
  Val Loss:   1.1230  Val Acc:   58.45% ✓ (Best!)
  Epoch Time: 153.2s
  ✓ Saved best model: results/checkpoints/best_model.pth

[... continuing for all epochs ...]

================================================================================
Training Complete!
================================================================================

Best Validation Accuracy: 82.45%
Final Training Accuracy:  85.23%
Final Validation Accuracy: 81.98%

Total Training Time: 76.4 minutes
Average Time per Epoch: 152.8 seconds

Training history saved to: results/logs/training_history_20241012_203045.json
================================================================================
```

## Monitoring Training

### Real-time Monitoring

The training script shows:
- Progress bars for training and validation
- Real-time loss and accuracy
- Per-epoch summaries
- Best model tracking

### Training History Analysis

After training, analyze the JSON history file:

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('results/logs/training_history_*.json') as f:
    history = json.load(f)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solutions:**
```yaml
# 1. Reduce batch size
training:
  batch_size: 8  # or 4

# 2. Use smaller image size
image:
  img_size: 224  # instead of 384 or 512

# 3. Use smaller model
model:
  model_name: efficientnet_b0  # instead of resnet50
```

### Issue: Training Too Slow

**Solutions:**
```yaml
# 1. Increase num_workers
system:
  num_workers: 8  # more parallel data loading

# 2. Enable pin_memory (automatically done for CUDA)

# 3. Use smaller model for quick experiments
model:
  model_name: resnet34
```

### Issue: Poor Convergence

**Solutions:**
```yaml
# 1. Lower learning rate
training:
  learning_rate: 0.00005

# 2. Use pretrained weights
model:
  pretrained: true

# 3. Train longer
training:
  num_epochs: 50

# 4. Try different model
model:
  model_name: efficientnet_b3
```

### Issue: Overfitting

**Solutions:**
```yaml
# 1. Increase weight decay
training:
  weight_decay: 0.001  # more regularization

# 2. Use more aggressive augmentation
# (edit in train_baseline.py)

# 3. Use dropout in model
# (automatic with DRClassifier, dropout=0.3)
```

### Issue: Data Loading Errors

**Check:**
1. CSV file exists and has correct columns (`id_code`, `diagnosis`)
2. Image directory exists and contains images
3. Image filenames match `id_code` in CSV
4. Images have valid extensions (`.png`, `.jpg`, `.jpeg`)

```bash
# Verify data
python -c "
from scripts.dataset import RetinalDataset
dataset = RetinalDataset('data/aptos/train.csv', 'data/aptos/train_images')
print(f'Dataset size: {len(dataset)}')
print(f'Class distribution: {dataset.get_class_distribution()}')
"
```

## Best Practices

### 1. Start with Baseline

Use default configuration first:

```bash
python scripts/train_baseline.py --config configs/default_config.yaml
```

### 2. Experiment Systematically

Change one thing at a time:
- First: Try different models
- Then: Tune learning rate
- Then: Adjust batch size
- Finally: Train longer

### 3. Track Experiments

Use descriptive names for configs and outputs:

```yaml
paths:
  checkpoint_dir: results/checkpoints/resnet50_lr1e4_bs32
  log_dir: results/logs/resnet50_lr1e4_bs32
```

### 4. Save Best Models

Always keep the best model:
- Automatically saved as `best_model.pth`
- Can resume training from any checkpoint

### 5. Monitor Validation Accuracy

- Training accuracy can be misleading (overfitting)
- **Validation accuracy** is the true performance metric
- Early stopping based on validation accuracy

## Advanced Usage

### Custom Data Split

Modify `create_data_loaders()` in `train_baseline.py`:

```python
# Change split ratio
train_size = int(0.9 * total_size)  # 90/10 instead of 80/20
val_size = total_size - train_size
```

### Custom Augmentation

Modify `get_transforms()` in `train_baseline.py`:

```python
# Add more aggressive augmentation
train_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Add your custom augmentations here
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])
```

### Learning Rate Scheduling

The script includes `ReduceLROnPlateau` by default. Modify in `train()`:

```python
# Change scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)
```

### Early Stopping

Add to training loop in `train()`:

```python
# After validation
if val_metrics['val_acc'] < best_acc - 0.01:  # No improvement
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered")
        break
else:
    patience_counter = 0
```

## Performance Benchmarks

Approximate training times on different hardware:

| Model | GPU | Batch Size | Time/Epoch | Total (30 epochs) |
|-------|-----|-----------|-----------|-------------------|
| ResNet50 | V100 | 32 | 2 min | 1 hour |
| ResNet50 | T4 | 16 | 3 min | 1.5 hours |
| ResNet50 | CPU | 16 | 30 min | 15 hours |
| EfficientNet-B3 | V100 | 24 | 2.5 min | 1.25 hours |
| ViT-Base | V100 | 16 | 4 min | 2 hours |

*Times are approximate for dataset of ~4000 images*

## Next Steps

After training:

1. **Evaluate on test set**
   - Use trained model to predict on test data
   - Calculate final metrics

2. **Analyze results**
   - Plot training curves
   - Examine per-class performance
   - Identify failure cases

3. **Iterate**
   - Try different models
   - Adjust hyperparameters
   - Add more data augmentation

4. **Deploy**
   - Export best model
   - Create inference pipeline
   - Test on new data

## Support

For issues or questions:
1. Check this training guide
2. See `scripts/train_baseline.py` implementation
3. Review `MODEL_GUIDE.md` for model details
4. Check `CONFIGURATION_GUIDE.md` for config help

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
# Place your data in data/aptos/

# 3. Create configuration
cp configs/train_example.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml

# 4. Train model
python scripts/train_baseline.py --config configs/my_experiment.yaml

# 5. Monitor training
# Watch console output for progress

# 6. Evaluate results
# Best model saved at results/checkpoints/best_model.pth
# Training history at results/logs/training_history_*.json

# 7. Resume if needed
python scripts/train_baseline.py \
    --config configs/my_experiment.yaml \
    --resume results/checkpoints/checkpoint_epoch_15.pth
```

Happy training! 🚀
