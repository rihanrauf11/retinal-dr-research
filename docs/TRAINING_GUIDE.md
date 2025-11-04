# Training Guide: RETFound + LoRA

Complete step-by-step guide for training diabetic retinopathy classification models using RETFound and LoRA.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Training with RETFound Large](#training-with-retfound-large)
5. [Training with RETFound_Green](#training-with-retfoundgreen)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Cross-Dataset Evaluation](#cross-dataset-evaluation)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Prerequisites

### System Requirements

- **GPU Memory**: 6GB minimum (Green), 11GB recommended (Large)
- **Python**: 3.8 or newer
- **CUDA**: 11.8+ (for GPU training)
- **Disk Space**: 50GB minimum (for datasets and checkpoints)

### Software Dependencies

```bash
# Core dependencies
PyTorch >= 2.0
torchvision >= 0.15
timm >= 0.9.0
peft >= 0.4.0
albumentations >= 1.3.0

# Optional but recommended
wandb >= 0.14.0      # Experiment tracking
optuna >= 3.0.0      # Hyperparameter optimization
tensorboard >= 2.12  # Training visualization
```

### Verify Installation

```bash
python3 << 'EOF'
import torch
import torchvision
import timm
import peft
import albumentations

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ TorchVision: {torchvision.__version__}")
print(f"✓ timm: {timm.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ Albumentations: {albumentations.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/diabetic-retinopathy-classification.git
cd diabetic-retinopathy-classification
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Data Directories

```bash
mkdir -p data/aptos/train_images
mkdir -p data/aptos/val_images
mkdir -p data/aptos/test_images
mkdir -p models
mkdir -p results
```

### Step 5: Download Pre-trained Weights

**For RETFound Large**:
```bash
wget -O models/RETFound_cfp_weights.pth \
  https://github.com/rmaphoh/RETFound_MAE/releases/download/weights/RETFound_cfp_weights.pth
```

**For RETFound_Green**:
```bash
wget -O models/retfoundgreen_statedict.pth \
  https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
```

---

## Data Preparation

### Understanding the Data Format

The project expects data in the following structure:

```
data/aptos/
├── train.csv          # Training labels: id_code, diagnosis
├── val.csv            # Validation labels
├── test.csv           # Test labels (optional)
├── train_images/      # Training fundus images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── val_images/        # Validation images
└── test_images/       # Test images (optional)
```

### CSV Format

Each CSV file should have exactly two columns:

```csv
id_code,diagnosis
image_001,0
image_002,2
image_003,1
...
```

Where:
- `id_code`: Image filename without extension
- `diagnosis`: DR severity level (0-4)
  - 0: No DR
  - 1: Mild DR
  - 2: Moderate DR
  - 3: Severe DR
  - 4: Proliferative DR

### Creating Train/Val/Test Splits

If you have a single dataset, create stratified splits:

```bash
python3 scripts/prepare_data.py \
    --input-csv data/raw_labels.csv \
    --input-dir data/raw_images \
    --output-dir data/aptos \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --stratify \
    --seed 42
```

### Verify Data Integrity

```bash
python3 << 'EOF'
import pandas as pd
from pathlib import Path

# Check training data
train_csv = pd.read_csv('data/aptos/train.csv')
train_dir = Path('data/aptos/train_images')

print(f"Training samples: {len(train_csv)}")
print(f"Missing images: {sum(~train_dir.joinpath(f'{idx}.png').exists() for idx in train_csv['id_code'])}")
print(f"Label distribution:\n{train_csv['diagnosis'].value_counts().sort_index()}")

# Check validation data
val_csv = pd.read_csv('data/aptos/val.csv')
val_dir = Path('data/aptos/val_images')

print(f"\nValidation samples: {len(val_csv)}")
print(f"Missing images: {sum(~val_dir.joinpath(f'{idx}.png').exists() for idx in val_csv['id_code'])}")
print(f"Label distribution:\n{val_csv['diagnosis'].value_counts().sort_index()}")
EOF
```

---

## Training with RETFound Large

### Step 1: Verify RETFound Large Weights

```bash
ls -lh models/RETFound_cfp_weights.pth
# Should show: ~1.2GB
```

### Step 2: Review Configuration

Edit `configs/retfound_lora_config.yaml`:

```yaml
model:
  model_variant: large        # Ensure this is set to 'large'
  num_classes: 5
  pretrained_path: "models/RETFound_cfp_weights.pth"

training:
  num_epochs: 20              # Typically 20-30 epochs
  batch_size: 32              # Adjust based on GPU memory
  learning_rate: 0.0005       # LoRA learning rate

  lora_r: 8                   # Rank (can try 4, 8, 16)
  lora_alpha: 32              # 4x rank

system:
  device: cuda                # Use GPU
  seed: 42                    # For reproducibility
```

### Step 3: Dry Run (Optional but Recommended)

Test data loading without training:

```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --num-epochs 1 \
    --log-level DEBUG
```

### Step 4: Start Training

```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml
```

**Expected Output**:
```
================================================================================
RETFOUND + LORA TRAINING
================================================================================
Config: configs/retfound_lora_config.yaml
Device: cuda
Output: results/retfound_lora
...
[Epoch 1/20] Train Loss: 1.234, Train Acc: 45.2% | Val Loss: 0.987, Val Acc: 52.3%
[Epoch 2/20] Train Loss: 0.876, Train Acc: 65.4% | Val Loss: 0.654, Val Acc: 68.1%
...
[BEST] Epoch 15: Validation Accuracy: 88.2%
...
Training completed in 4.5 hours
```

### Step 5: Resume Training (if interrupted)

```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --resume results/retfound_lora/checkpoints/checkpoint_epoch_10.pth
```

### Step 6: Monitor with TensorBoard (Optional)

```bash
tensorboard --logdir results/retfound_lora/tensorboard
# Open browser to http://localhost:6006
```

### Step 7: Monitor with Weights & Biases (Optional)

```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --wandb \
    --wandb-project "retfound-research" \
    --wandb-run-name "retfound_large_aptos"
```

---

## Training with RETFound_Green

### Step 1: Verify RETFound_Green Weights

```bash
ls -lh models/retfoundgreen_statedict.pth
# Should show: ~50MB
```

### Step 2: Review Configuration

Edit `configs/retfound_green_lora_config.yaml`:

```yaml
model:
  model_variant: green        # Ensure this is set to 'green'
  num_classes: 5
  pretrained_path: "models/retfoundgreen_statedict.pth"

image:
  input_size: 392             # Green uses 392x392
  mean: [0.5, 0.5, 0.5]      # Custom normalization
  std: [0.5, 0.5, 0.5]

training:
  num_epochs: 20              # Converges faster
  batch_size: 32              # Can be same or larger
  learning_rate: 0.0005

system:
  device: cuda
  seed: 42
```

### Step 3: Start Training

```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_green_lora_config.yaml
```

**Expected Output** (faster than Large):
```
[Epoch 1/20] Train Loss: 1.156, Train Acc: 48.3% | Val Loss: 0.923, Val Acc: 55.2%
[Epoch 2/20] Train Loss: 0.812, Train Acc: 67.1% | Val Loss: 0.598, Val Acc: 70.4%
...
[BEST] Epoch 12: Validation Accuracy: 86.5%
...
Training completed in 45 minutes (vs 4.5 hours for Large)
```

### Step 4: Compare Training Time

```bash
# RETFound Large
echo "Training RETFound Large..."
time python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml

# RETFound_Green
echo "Training RETFound_Green..."
time python3 scripts/train_retfound_lora.py \
    --config configs/retfound_green_lora_config.yaml

# Green should complete ~3-4x faster
```

---

## Hyperparameter Optimization

### Quick Tuning (5-10 Trials)

For rapid iteration:

```bash
python3 scripts/hyperparameter_search.py \
    --checkpoint-path models/retfoundgreen_statedict.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --model-variant green \
    --n-trials 10 \
    --num-epochs 5
```

### Full Tuning (50 Trials)

For comprehensive optimization:

```bash
python3 scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --model-variant large \
    --n-trials 50 \
    --num-epochs 10
```

### Extract Best Parameters

```bash
cat results/optuna/retfound_lora_search/best_params.json
# Output:
# {
#   "lora_r": 8,
#   "lora_alpha": 32,
#   "learning_rate": 0.00047,
#   "batch_size": 32,
#   "dropout": 0.15,
#   "best_value": 88.34
# }
```

### Apply Best Parameters

Create `configs/retfound_lora_tuned.yaml` with:

```yaml
model:
  model_variant: large
  lora_r: 8              # From best params
  lora_alpha: 32

training:
  batch_size: 32         # From best params
  learning_rate: 0.00047 # From best params
  head_dropout: 0.15     # From best params
```

Then train with:

```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_tuned.yaml
```

---

## Cross-Dataset Evaluation

### Single Dataset Evaluation

```bash
# Evaluate RETFound Large
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --model_variant large \
    --datasets APTOS:data/aptos/test.csv:data/aptos/test_images
```

### Cross-Dataset Evaluation

Measure generalization gap:

```bash
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/test_images \
        Messidor:data/messidor/test.csv:data/messidor/test_images \
        IDRiD:data/idrid/test.csv:data/idrid/test_images
```

### Compare Models Side-by-Side

```bash
# Create comparison script
python3 << 'EOF'
import subprocess
import json

models = {
    'retfound_large': 'results/retfound_lora/checkpoints/best_model.pth',
    'retfound_green': 'results/retfound_green_lora/checkpoints/best_model.pth'
}

variants = {
    'retfound_large': 'large',
    'retfound_green': 'green'
}

datasets = [
    'APTOS:data/aptos/test.csv:data/aptos/test_images',
    'Messidor:data/messidor/test.csv:data/messidor/test_images'
]

for model_name, checkpoint in models.items():
    variant = variants[model_name]
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} (variant: {variant})")
    print('='*60)

    for dataset in datasets:
        cmd = [
            'python3', 'scripts/evaluate_cross_dataset.py',
            '--checkpoint', checkpoint,
            '--model_type', 'lora',
            '--model_variant', variant,
            '--datasets', dataset
        ]
        subprocess.run(cmd)
EOF
```

---

## Monitoring and Logging

### TensorBoard Visualization

```bash
# Launch TensorBoard
tensorboard --logdir results/retfound_lora/tensorboard

# Open browser to http://localhost:6006
# Monitor:
# - Training/validation loss
# - Accuracy curves
# - Learning rate schedule
# - Gradient norms
```

### Weights & Biases Integration

Enable W&B logging during training:

```bash
# First time setup
wandb login

# Then train with W&B
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --wandb \
    --wandb-project "retfound-research" \
    --wandb-run-name "large_baseline_aptos" \
    --wandb-notes "Testing RETFound Large with default hyperparams"
```

W&B tracks:
- Hyperparameters
- Training curves
- GPU memory usage
- System metrics
- Model checkpoints
- Configuration files

### Check Training Logs

```bash
# View training history
cat results/retfound_lora/training_history.json | python3 -m json.tool

# Plot learning curves
python3 << 'EOF'
import json
import matplotlib.pyplot as plt

with open('results/retfound_lora/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 3, 3)
plt.plot(history['lr'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
EOF
```

---

## Troubleshooting

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 16  # From 32
   ```

2. **Reduce LoRA rank**:
   ```yaml
   model:
     lora_r: 4  # From 8
   ```

3. **Clear GPU cache**:
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

4. **Use gradient checkpointing** (slower but saves memory):
   ```yaml
   training:
     gradient_checkpointing: true
   ```

### Data Loading Errors

**Error**: `FileNotFoundError: Image file not found`

**Solution**: Verify CSV and image directory match:
```bash
# Check if image files exist
python3 << 'EOF'
import pandas as pd
from pathlib import Path

train_csv = pd.read_csv('data/aptos/train.csv')
img_dir = Path('data/aptos/train_images')

for idx in train_csv['id_code'][:10]:
    img_path = img_dir / f'{idx}.png'
    if not img_path.exists():
        print(f"Missing: {img_path}")
    else:
        print(f"✓ {img_path}")
EOF
```

### Image Dimensions Mismatch

**Error**: `Expected input with shape [*, 384], but got input of size [2, 0]`

**Solutions**:

1. **Check normalization**:
   - Large: `[0.485, 0.456, 0.406]`
   - Green: `[0.5, 0.5, 0.5]`

2. **Check input size**:
   - Large: 224×224
   - Green: 392×392

3. **Verify image format**: PNG, JPG, or JPEG only

### Poor Training Performance

**Problem**: Accuracy not improving or stuck

**Debugging**:
```bash
# 1. Check data quality
python3 scripts/dataset.py  # Tests data loading

# 2. Check if model is training
python3 << 'EOF'
import torch
from scripts.retfound_lora import RETFoundLoRA

model = RETFoundLoRA(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5
)

# Check trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Should see ~800K trainable params for Large
EOF

# 3. Try different learning rate
# 4. Check if using correct variant normalization
# 5. Increase training epochs
```

---

## Best Practices

### 1. Data Management

- **Always stratify splits**: Use `--stratify` flag when creating splits
- **Verify class balance**: Check that train/val distributions are similar
- **Use consistent image format**: Either all PNG or all JPG
- **Document data source**: Note where images come from

### 2. Reproducibility

```yaml
system:
  seed: 42                      # Set seed
  cudnn_deterministic: true     # Deterministic CUDA
  cudnn_benchmark: false        # No autotuning
```

Save configuration with results:
```bash
cp configs/retfound_lora_config.yaml results/config_used.yaml
```

### 3. Hyperparameter Selection

- **Start with defaults**: Before tuning
- **Search systematically**: Use Optuna or grid search
- **Validate on test set**: Not just validation set
- **Cross-validate**: Use multiple folds

### 4. Checkpoint Management

```yaml
checkpoint:
  save_best_only: true          # Only save best model
  monitor_metric: accuracy       # What to monitor
  keep_last_n: 3                # Keep last 3 checkpoints
```

Clean up old checkpoints:
```bash
find results/retfound_lora/checkpoints -name "checkpoint_*.pth" -delete
ls results/retfound_lora/checkpoints  # Verify
```

### 5. Evaluation Best Practices

- **Test on held-out set**: Never use validation set
- **Report multiple metrics**: Accuracy, F1, AUC, Cohen's Kappa
- **Show confusion matrix**: Understand misclassification patterns
- **Cross-dataset validation**: Critical for generalization

### 6. Documentation

Keep training log:
```bash
# At start of training
echo "Starting training: $(date)" >> training_log.txt
echo "Config: configs/retfound_lora_config.yaml" >> training_log.txt
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> training_log.txt

# During training
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    2>&1 | tee -a training_log.txt
```

### 7. Variant Selection for Different Scenarios

**Quick Prototyping** (hours):
- Use RETFound_Green
- Small dataset (1000 images)
- Batch size: 32, epochs: 10

**Full Training** (days):
- Use RETFound Large
- Full dataset
- Batch size: 32, epochs: 20-30

**Hyperparameter Search**:
- Use RETFound_Green for iteration
- Run 50 trials to find best params
- Apply to RETFound Large for final model

**Production Deployment**:
- Use RETFound_Green
- Optimized for latency
- Monitor inference time

### 8. Version Control

Track important experiments:
```bash
# Create descriptive branch
git checkout -b experiment/retfound-green-baseline

# Document experiment
echo "# RETFound_Green Baseline" > experiment_notes.md
echo "## Setup" >> experiment_notes.md
echo "- Model: RETFound_Green" >> experiment_notes.md
echo "- Config: retfound_green_lora_config.yaml" >> experiment_notes.md

# Commit results
git add results/retfound_green_lora/
git commit -m "Add RETFound_Green baseline results"
```

---

## Quick Reference

### Training Command Templates

```bash
# RETFound Large - Full Training
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml

# RETFound_Green - Fast Training
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_green_lora_config.yaml

# With W&B Logging
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --wandb --wandb-run-name "my_experiment"

# Resume from Checkpoint
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --resume results/retfound_lora/checkpoints/checkpoint_epoch_10.pth

# Hyperparameter Search
python3 scripts/hyperparameter_search.py \
    --checkpoint-path models/retfoundgreen_statedict.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --n-trials 50

# Cross-Dataset Evaluation
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --model_variant large \
    --datasets APTOS:data/aptos/test.csv:data/aptos/test_images
```

### Typical Training Timeline

**RETFound Large**:
- Data loading: 5 min
- Model initialization: 2 min
- Training (20 epochs): 3-4 hours
- Evaluation: 5 min
- **Total**: 3.5-4.5 hours

**RETFound_Green**:
- Data loading: 5 min
- Model initialization: 1 min
- Training (20 epochs): 30-45 min
- Evaluation: 5 min
- **Total**: 45 min - 1 hour

---

## Next Steps

After training:

1. **Evaluate**: Run `scripts/evaluate_cross_dataset.py`
2. **Analyze**: Check confusion matrix and per-class metrics
3. **Compare**: Test both RETFound Large and Green
4. **Document**: Save results and configuration
5. **Deploy**: Use best model for production

See [RETFOUND_GUIDE.md](RETFOUND_GUIDE.md) for more details on model comparison and selection.

