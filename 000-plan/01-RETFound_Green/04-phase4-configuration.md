# Phase 4: Configuration & Documentation

## Overview

**Objective**: Create configuration templates and update documentation for complete RETFound_Green integration.

**Files Modified/Created**: Config files, documentation

**Estimated Effort**: 1 day

**Risk Level**: **LOW** - Documentation only, no code changes

**Validation**: Configs are valid YAML, documentation is complete and accurate

---

## File 1: configs/retfound_green_lora_config.yaml (NEW)

Create this new configuration file as a template for RETFound_Green training:

```yaml
# RETFound_Green (ViT-Small, 21.3M params) with LoRA Fine-tuning
#
# This configuration uses the efficient RETFound_Green foundation model,
# which is 14x smaller than RETFound while maintaining competitive performance.
# Trained on 75K retinal images with token reconstruction approach.
#
# Key Differences from RETFound Large:
# - Parameters: 21.3M vs 303M (93% reduction)
# - Input size: 392×392 vs 224×224
# - Normalization: 0.5 mean/std vs ImageNet
# - Training time: ~3-5 min/epoch vs 10-15 min/epoch
# - GPU memory: ~6-8GB vs 11-12GB (batch=32)
#
# Recommended use cases:
# - Resource-constrained research environments
# - Rapid prototyping and hyperparameter optimization
# - Production deployment (lower inference latency)
# - Comparative studies on model size effects
#
# References:
# - RETFound_Green: https://github.com/justinengelmann/RETFound_Green
# - Weights: https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth

system:
  seed: 42
  device: cuda  # 'cuda', 'mps', or 'cpu'
  num_workers: 4
  logging_level: INFO
  mixed_precision: false  # Set true for faster training on supported GPUs

model:
  model_name: retfound_lora
  model_variant: green  # Key difference: use 'green' for RETFound_Green
  num_classes: 5
  pretrained: true
  # Download from: https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
  pretrained_path: "models/retfoundgreen_statedict.pth"

  # LoRA Hyperparameters (tuned for RETFound_Green)
  # Note: Same LoRA structure as Large variant, but different embed_dim (384 vs 1024)
  lora_r: 8           # Rank of low-rank matrices (8 recommended for Green, can try 4, 8, 16)
  lora_alpha: 32      # Scaling factor (typically 4x rank)
  head_dropout: 0.1   # Dropout in classification head

image:
  # RETFound_Green expects 392×392 input with 0.5 mean/std normalization
  input_size: 392  # Key difference: larger than RETFound Large (224)
  mean: [0.5, 0.5, 0.5]  # Key difference: not ImageNet norm
  std: [0.5, 0.5, 0.5]

  # Data augmentation
  max_rotation: 10
  brightness_contrast_limit: 0.2
  augmentation_prob: 0.5
  gaussian_noise_std: 0.02

data:
  # Training dataset
  train_csv: "data/aptos/train.csv"
  train_img_dir: "data/aptos/train_images"

  # Validation dataset
  val_csv: "data/aptos/val.csv"
  val_img_dir: "data/aptos/val_images"

  # Optional: test dataset
  test_csv: "data/aptos/test.csv"
  test_img_dir: "data/aptos/test_images"

training:
  num_epochs: 20  # Reduced from 30 for Large (converges faster)
  batch_size: 32  # Can be larger due to memory efficiency
  learning_rate: 0.0005  # Slightly higher than Large (adapters are small)
  weight_decay: 1e-4

  # Optimization
  optimizer: adam  # 'adam', 'adamw', or 'sgd'
  warmup_epochs: 2
  warmup_lr_init: 1e-6

  # Learning rate scheduling
  lr_scheduler: cosine  # 'cosine', 'linear', or 'exponential'
  min_lr: 1e-6

  # Validation
  val_freq: 1  # Validate every N epochs
  save_freq: 1  # Save checkpoint every N epochs

  # Early stopping
  early_stop_patience: 10  # Stop if val_acc doesn't improve for N epochs
  early_stop_metric: accuracy  # 'accuracy', 'auc', or 'loss'

checkpoint:
  save_dir: "results/retfound_green_lora"
  save_best_only: true
  monitor_metric: accuracy

wandb:
  enabled: false
  project: "retfound-green"
  run_name: "retfound_green_lora_baseline"
  notes: "RETFound_Green with LoRA on APTOS dataset"
  tags: ["retfound_green", "lora", "aptos"]

# Notes:
# 1. Download RETFound_Green weights before running:
#    wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
#    mv retfoundgreen_statedict.pth models/
#
# 2. Training command:
#    python3 scripts/train_retfound_lora.py --config configs/retfound_green_lora_config.yaml
#
# 3. Expected performance (on APTOS validation):
#    - Accuracy: 85-88%
#    - Training time: ~1-2 hours (20 epochs on RTX 3090)
#    - GPU memory: ~6GB (batch_size=32)
#
# 4. Cross-dataset evaluation:
#    python3 scripts/evaluate_cross_dataset.py \
#        --checkpoint results/retfound_green_lora/checkpoints/best_model.pth \
#        --datasets APTOS:data/aptos/test.csv:data/aptos/test_images \
#                   Messidor:data/messidor/test.csv:data/messidor/test_images
```

---

## File 2: configs/retfound_lora_config.yaml (UPDATE)

Update the existing config to explicitly document the Large variant:

**Changes**:
- Add `model_variant: large` to make it explicit
- Update comments to explain the difference from Green
- Add reference to RETFound_Green config

**Location**: Update the model section:

```yaml
# RETFound (ViT-Large, 303M params) with LoRA Fine-tuning
#
# This is the original configuration using the full-size RETFound model.
# For a more efficient alternative, see configs/retfound_green_lora_config.yaml
#
# Key Features:
# - 303M parameters in backbone
# - Proven cross-dataset generalization
# - Higher accuracy but requires more resources

system:
  seed: 42
  device: cuda
  num_workers: 4
  logging_level: INFO
  mixed_precision: false

model:
  model_name: retfound_lora
  model_variant: large  # UPDATED: Make explicit
  num_classes: 5
  pretrained: true
  pretrained_path: "models/RETFound_cfp_weights.pth"

  # LoRA Hyperparameters (tuned for RETFound Large)
  lora_r: 8           # Rank
  lora_alpha: 32      # Scaling factor
  head_dropout: 0.1

image:
  # RETFound uses 224×224 with ImageNet normalization
  input_size: 224
  mean: [0.485, 0.456, 0.406]  # ImageNet normalization
  std: [0.229, 0.224, 0.225]

  # Data augmentation
  max_rotation: 10
  brightness_contrast_limit: 0.2
  augmentation_prob: 0.5
  gaussian_noise_std: 0.02

data:
  train_csv: "data/aptos/train.csv"
  train_img_dir: "data/aptos/train_images"
  val_csv: "data/aptos/val.csv"
  val_img_dir: "data/aptos/val_images"
  test_csv: "data/aptos/test.csv"
  test_img_dir: "data/aptos/test_images"

training:
  num_epochs: 30
  batch_size: 32
  learning_rate: 0.00035  # Slightly lower than Green
  weight_decay: 1e-4

  optimizer: adam
  warmup_epochs: 2
  warmup_lr_init: 1e-6

  lr_scheduler: cosine
  min_lr: 1e-6

  val_freq: 1
  save_freq: 1
  early_stop_patience: 10
  early_stop_metric: accuracy

checkpoint:
  save_dir: "results/retfound_lora"
  save_best_only: true
  monitor_metric: accuracy

wandb:
  enabled: false
  project: "retfound"
  run_name: "retfound_lora_baseline"
  tags: ["retfound", "lora", "aptos"]

# MIGRATION NOTE:
# If you want to use the smaller, more efficient RETFound_Green model instead,
# use: configs/retfound_green_lora_config.yaml
```

---

## Documentation Updates

### File 3: RETFOUND_GUIDE.md (UPDATE)

Add new section comparing the two models:

**Add this section after the introduction**:

```markdown
## RETFound Variants: Large vs Green

This project now supports two RETFound model variants:

### RETFound (Large)
- **Architecture**: ViT-Large with 303M parameters
- **Input**: 224×224 pixels
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Training Data**: 1.6M images from DDR, ODIR, AIROGS
- **Training Method**: Masked Autoencoding (MAE)
- **Proven**: ✅ Validated in original RETFound paper

**When to use Large**:
- When maximum accuracy is the priority
- For published benchmarks and comparisons
- When resources allow (~12GB GPU memory)
- For comprehensive cross-dataset studies

### RETFound_Green
- **Architecture**: ViT-Small with 21.3M parameters (93% smaller)
- **Input**: 392×392 pixels (larger, captures more context)
- **Normalization**: Custom (mean=0.5, std=0.5)
- **Training Data**: 75K images from DDR, ODIR, AIROGS (50% less data)
- **Training Method**: Token Reconstruction with dual corruption
- **Efficiency**: 400× less compute to train

**When to use Green**:
- When computational resources are limited
- For rapid prototyping and experimentation
- For hyperparameter optimization (faster iteration)
- For production deployment (lower latency)
- When you want to validate the approach on limited budgets

### Performance Comparison

| Metric | RETFound Large | RETFound_Green |
|--------|---|---|
| Parameters | 303M | 21.3M |
| Training Time | 10-15 min/epoch | 3-5 min/epoch |
| GPU Memory (batch=32) | 11-12GB | 6-8GB |
| Inference Speed | ~100-200ms | ~30-50ms |
| APTOS Accuracy | 88-90% | 85-88% |
| Training Data | 1.6M images | 75K images |
| Cross-Dataset Gap | ~3% | ~4-5% (estimated) |

### Getting Started

**Using RETFound Large** (default, proven approach):
```bash
python3 scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
```

**Using RETFound_Green** (efficient alternative):
```bash
# 1. Download weights
wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
mv retfoundgreen_statedict.pth models/

# 2. Train with Green config
python3 scripts/train_retfound_lora.py --config configs/retfound_green_lora_config.yaml
```

### Choosing Between Variants

**Use a decision matrix**:

| Consideration | Use Large | Use Green |
|---|---|---|
| **Priority: Accuracy** | ✅ | ❌ |
| **Priority: Speed** | ❌ | ✅ |
| **Available GPU RAM** | >10GB | >6GB |
| **Time Budget** | Hours-Days | Minutes-Hours |
| **Published Comparison** | ✅ | Consider both |
| **Production Deploy** | ❌ | ✅ |
| **Research/Validation** | ✅ | ✅ |

For most practitioners, we recommend:
1. **Start with Green** for development and testing (faster iteration)
2. **Use Large** for final submission/publication (proven performance)
3. **Compare both** to understand model size vs accuracy tradeoff
```

### File 4: CLAUDE.md (UPDATE)

Update the architecture section to mention both models:

**Update the "High-Level Architecture" section**:

```markdown
## High-Level Architecture

### Three-Tier Model Architecture

1. **Baseline Models** (`scripts/model.py` - `DRClassifier`)
   - Wraps any `timm` model (1000+ architectures)
   - Standard fine-tuning approach (all parameters trainable)
   - Used for comparison benchmarks

2. **RETFound Foundation Models** - Now supports two variants:

   **Option A: RETFound (Large)** (`scripts/retfound_model.py`)
   - Vision Transformer (ViT-Large) with 303M parameters
   - Pre-trained on 1.6M retinal images using masked autoencoding
   - Input: 224×224 pixels
   - Use this for maximum accuracy and proven benchmarks

   **Option B: RETFound_Green (Small)** - NEW
   - Vision Transformer (ViT-Small) with 21.3M parameters
   - Pre-trained on 75K retinal images using token reconstruction
   - Input: 392×392 pixels (larger context)
   - Use this for efficient training and resource-constrained environments
   - GitHub: https://github.com/justinengelmann/RETFound_Green

3. **RETFound + LoRA** (`scripts/retfound_lora.py` - `RETFoundLoRA`)
   - **Primary research approach** - parameter-efficient fine-tuning
   - Works with both RETFound Large and Green variants
   - Freezes foundation backbone, adds trainable low-rank adapters
   - Only ~800K trainable parameters (0.26% of Large, 3.8% of Green)
   - Achieved through PEFT library's LoRA implementation
   - Better cross-dataset generalization than full fine-tuning
```

### File 5: README.md (UPDATE)

Add a section highlighting the new RETFound_Green option:

**Add after "Quick Start"**:

```markdown
### Model Variants

This project supports two RETFound variants. Choose based on your constraints:

**RETFound Large** (Original)
- 303M parameters, 224×224 input
- Best accuracy, proven benchmarks
- Requires 12GB GPU memory

**RETFound_Green** (Efficient)
- 21.3M parameters, 392×392 input
- 400× less compute, 93% smaller
- Requires 6GB GPU memory
- Faster training and inference

See [RETFOUND_GUIDE.md](docs/RETFOUND_GUIDE.md) for detailed comparison.
```

---

## Documentation Structure

The documentation hierarchy should be:

```
README.md (top-level overview)
├── Quick start for both variants
└── Link to RETFOUND_GUIDE.md

RETFOUND_GUIDE.md (detailed model comparison)
├── RETFound Large explanation
├── RETFound_Green explanation
├── Performance comparison table
└── Decision matrix for choosing variant

TRAINING_GUIDE.md (updated)
├── Example: Train with Large (original)
└── Example: Train with Green (new)

CLAUDE.md (project context for Claude)
└── Updated architecture description (both variants)

configs/retfound_lora_config.yaml (Large)
configs/retfound_green_lora_config.yaml (Green - NEW)
```

---

## Implementation Checklist

### Configuration Files
- [ ] Create `configs/retfound_green_lora_config.yaml`
- [ ] Update `configs/retfound_lora_config.yaml` with explicit `model_variant: large`
- [ ] Add comments explaining differences between configs
- [ ] Verify YAML syntax is valid

### Documentation
- [ ] Add "RETFound Variants" section to `RETFOUND_GUIDE.md`
- [ ] Create performance comparison table
- [ ] Create decision matrix
- [ ] Add getting started instructions for both variants
- [ ] Update `CLAUDE.md` architecture section
- [ ] Update `README.md` with model variant section
- [ ] Update `TRAINING_GUIDE.md` with new examples
- [ ] Add references to RETFound_Green GitHub repository

### Testing
- [ ] Verify both config files load without errors
- [ ] Verify all YAML is valid
- [ ] Test training with both configs

### Validation
- [ ] All documentation is accurate and consistent
- [ ] All code examples are tested
- [ ] No broken links or references
- [ ] Decision matrix is clear and helpful

---

## Documentation Examples to Add

### Example 1: Getting Started with Green
```markdown
#### Quick Start with RETFound_Green

1. Download the pretrained weights:
   ```bash
   wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
   mv retfoundgreen_statedict.pth models/
   ```

2. Train with the Green configuration:
   ```bash
   python3 scripts/train_retfound_lora.py \
       --config configs/retfound_green_lora_config.yaml
   ```

3. Evaluate on cross-dataset:
   ```bash
   python3 scripts/evaluate_cross_dataset.py \
       --checkpoint results/retfound_green_lora/checkpoints/best_model.pth \
       --datasets APTOS:data/aptos/test.csv:data/aptos/test_images
   ```
```

### Example 2: Comparing Models
```markdown
#### Comparing RETFound Large vs Green

Train both models and compare results:

```bash
# Train RETFound Large
time python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml

# Train RETFound_Green
time python3 scripts/train_retfound_lora.py \
    --config configs/retfound_green_lora_config.yaml

# Evaluate both on cross-dataset
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --datasets APTOS:data/aptos/test.csv:data/aptos/test_images

python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_green_lora/checkpoints/best_model.pth \
    --datasets APTOS:data/aptos/test.csv:data/aptos/test_images
```

Compare training time, memory usage, and accuracy to see the efficiency tradeoff.
```

---

## Validation Criteria for Phase 4

When this phase is complete, you should have:

1. ✅ Valid YAML configuration for RETFound_Green
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/retfound_green_lora_config.yaml'))"
   ```

2. ✅ Clear comparison between variants in documentation

3. ✅ Updated README with variant information

4. ✅ Complete getting started guide for both models

5. ✅ Decision matrix to help users choose

6. ✅ Code examples for both training and evaluation

7. ✅ All documentation links are consistent and working

---

## Transition to Phase 5+

After Phase 4, the implementation is complete! Proceed to Phase 5 for:
- Risk analysis and mitigation
- Comprehensive validation
- Final testing before deployment

See `05-risks-and-mitigations.md` and `06-validation-checklist.md` for details.
