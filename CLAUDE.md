# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research project** for diabetic retinopathy (DR) classification using parameter-efficient fine-tuning of vision foundation models. The project focuses on cross-dataset generalization using **RETFound + LoRA** (Low-Rank Adaptation), achieving 99.7% parameter reduction while maintaining competitive accuracy.

**Key Research Goal:** Improve cross-dataset generalization for DR screening models deployed across different imaging equipment and patient populations.

## Essential Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Training Commands

**Baseline model training (ResNet, EfficientNet, etc.):**
```bash
python3 scripts/train_baseline.py --config configs/default_config.yaml
```

**RETFound + LoRA training (recommended):**
```bash
# Download RETFound weights first from: https://github.com/rmaphoh/RETFound_MAE
# Place at: models/RETFound_cfp_weights.pth

python3 scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --config configs/retfound_lora_config.yaml
```

**Resume training from checkpoint:**
```bash
python3 scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --resume results/retfound_lora/checkpoints/checkpoint_epoch_5.pth
```

### Hyperparameter Optimization
```bash
# Run Optuna search (50 trials)
python3 scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --n-trials 50
```

### Cross-Dataset Evaluation
```bash
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/images \
        Messidor:data/messidor/test.csv:data/messidor/images
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dataset.py -v
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

### Data Preparation
```bash
# Download APTOS dataset (requires Kaggle API configured)
python scripts/prepare_data.py --aptos-only

# Create sample dataset for testing
python scripts/prepare_data.py --aptos-only --create-sample

# Verify existing datasets
python scripts/prepare_data.py --verify-only
```

## High-Level Architecture

### Three-Tier Model Architecture

1. **Baseline Models** (`scripts/model.py` - `DRClassifier`)
   - Wraps any `timm` model (1000+ architectures)
   - Standard fine-tuning approach (all parameters trainable)
   - Used for comparison benchmarks
   - Example: ResNet50 (23M params), EfficientNet-B3 (11M params)

2. **RETFound Foundation Model** (`scripts/retfound_model.py`)
   - Vision Transformer (ViT-Large) with 303M parameters
   - Pre-trained on 1.6M retinal images using masked autoencoding
   - Domain-specific for ophthalmology tasks
   - Checkpoint must be downloaded separately

3. **RETFound + LoRA** (`scripts/retfound_lora.py` - `RETFoundLoRA`)
   - **Primary research approach** - parameter-efficient fine-tuning
   - Freezes RETFound backbone, adds trainable low-rank adapters
   - Only ~800K trainable parameters (0.26% of full model)
   - Achieved through PEFT library's LoRA implementation
   - Better cross-dataset generalization than full fine-tuning

### Training Pipeline Flow

```
Data Loading (scripts/dataset.py)
    ↓
Configuration (scripts/config.py - YAML-based)
    ↓
Model Creation (scripts/model.py or scripts/retfound_lora.py)
    ↓
Training Loop (scripts/train_baseline.py or scripts/train_retfound_lora.py)
    ├── Augmentation (albumentations)
    ├── Forward pass
    ├── Loss computation (CrossEntropyLoss)
    ├── Backward pass + optimizer step
    └── Validation + checkpointing
    ↓
Evaluation (scripts/evaluate_cross_dataset.py)
```

### Key Design Patterns

**Configuration System:**
- All experiments configured via YAML files in `configs/`
- Type-safe dataclass-based config (`scripts/config.py`)
- Supports command-line overrides
- Automatic device detection (CUDA/MPS/CPU)
- Validation ensures paths exist and parameters are valid

**Dataset Architecture:**
- `RetinalDataset` class in `scripts/dataset.py`
- CSV-based labels with format: `id_code,diagnosis`
- Supports .png, .jpg, .jpeg images
- Albumentations for training augmentation
- torchvision transforms for validation (resize only)
- Robust error handling for corrupted images

**Checkpoint Strategy:**
- Saves every epoch: `checkpoint_epoch_N.pth`
- Tracks best validation accuracy: `best_model.pth`
- On interrupt: `checkpoint_interrupted.pth`
- Each checkpoint includes: model weights, optimizer state, epoch, best accuracy, training history

**LoRA Integration:**
- Uses HuggingFace PEFT library
- Applies LoRA to QKV attention projections in ViT
- Configurable rank (r=4,8,16,32) and alpha scaling
- Separate classification head (always trainable)
- Can load/merge/save adapters independently

## Critical Implementation Details

### LoRA Configuration
**Key hyperparameters in `configs/retfound_lora_config.yaml`:**
- `lora_r`: Rank of low-rank matrices (8 is recommended default)
- `lora_alpha`: Scaling factor (typically 4x rank, so 32 for r=8)
- `learning_rate`: Higher than full fine-tuning (0.0005 vs 0.0001)
- `batch_size`: Can be larger due to memory savings (32 vs 16)

**Why these matter:**
- Rank controls adapter capacity: r=8 for most cases, r=16 for small datasets
- Alpha controls update magnitude: higher = stronger LoRA influence
- Higher LR needed because only adapters are trained, not frozen backbone

### Data Format Requirements
**CSV structure:**
```csv
id_code,diagnosis
image_001,0
image_002,2
```
- Column names MUST be exactly `id_code` and `diagnosis`
- `id_code`: filename without extension
- `diagnosis`: integer 0-4 (DR severity level)
- Images should be in separate directory with matching filenames

**Image organization:**
```
data/
└── aptos/
    ├── train.csv
    ├── test.csv
    ├── train_images/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    └── test_images/
```

### RETFound Weight Loading
**Critical:** RETFound weights are NOT in this repo and must be downloaded:
1. Visit: https://github.com/rmaphoh/RETFound_MAE
2. Download: `RETFound_cfp_weights.pth` (CFP = Color Fundus Photography)
3. Place in: `models/RETFound_cfp_weights.pth`
4. The loading logic in `scripts/retfound_lora.py` handles key mismatches

### Cross-Dataset Evaluation Pattern
The `evaluate_cross_dataset.py` script requires specific format:
```bash
--datasets \
    DATASET_NAME:path/to/csv:path/to/images \
    DATASET_NAME2:path/to/csv2:path/to/images2
```

This enables measuring **generalization gap** - the key research metric showing how well models transfer across different imaging equipment and populations.

## Directory Structure Semantics

**`scripts/`** - All executable Python modules (training, evaluation, utilities)
- Files are both importable modules AND runnable scripts
- Running `python scripts/module.py` often includes self-tests
- Core modules: `dataset.py`, `model.py`, `retfound_lora.py`, `utils.py`, `config.py`

**`configs/`** - YAML configuration files
- Each config represents a complete experiment specification
- `retfound_lora_config.yaml` is the primary config for research
- `train_example.yaml` has detailed annotations

**`tests/`** - pytest test suite (300+ tests)
- Comprehensive coverage of dataset, model, transforms, utils
- Fast execution (~80 seconds total)
- Run before making significant changes

**`results/`** - Training outputs (NOT in git)
- Organized by model type: `baseline/`, `retfound_lora/`, `optuna/`
- Contains checkpoints, logs, training history JSON

**`data/`** - Datasets (NOT in git)
- Expected structure: `data/aptos/`, `data/messidor/`, `data/sample/`
- Use `scripts/prepare_data.py` to download and organize

**`models/`** - Pre-trained weights (NOT in git)
- Place RETFound checkpoint here: `models/RETFound_cfp_weights.pth`

**`docs/`** - Extensive documentation
- `RETFOUND_GUIDE.md` - Foundation model details
- `TRAINING_GUIDE.md` - Step-by-step training instructions
- `HYPERPARAMETER_OPTIMIZATION_GUIDE.md` - Optuna search guide
- `CONFIGURATION_GUIDE.md` - Config system reference

## Common Workflows

### Starting a New Experiment
1. Copy relevant config: `cp configs/retfound_lora_config.yaml configs/my_experiment.yaml`
2. Edit paths and hyperparameters in `my_experiment.yaml`
3. Ensure RETFound weights are downloaded
4. Run training: `python3 scripts/train_retfound_lora.py --config configs/my_experiment.yaml`
5. Monitor progress in console output
6. Best model saved at: `results/retfound_lora/checkpoints/best_model.pth`

### Debugging Training Issues
1. **OOM errors:** Reduce `batch_size` in config or reduce `lora_r`
2. **Slow convergence:** Increase `learning_rate` (LoRA needs higher LR)
3. **Poor generalization:** Increase `head_dropout`, add more augmentation
4. **Dataset errors:** Run `python scripts/dataset.py` to test loading

### Running Hyperparameter Search
1. Use `scripts/hyperparameter_search.py` with Optuna
2. Searches over: lora_r, lora_alpha, learning_rate, batch_size, dropout
3. Results saved to: `results/optuna/<study_name>/`
4. Best params in: `best_params.json`
5. Apply best params to new training run

### Adding Support for New Dataset
1. Organize data in same CSV + images format
2. Update config file with new paths
3. Run `scripts/prepare_data.py --verify-only` to check
4. No code changes needed - dataset loader is generic

## Integration with Weights & Biases (Optional)

Training scripts support W&B for experiment tracking:
```bash
# First time setup
pip install wandb
wandb login

# Use with training
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --wandb \
    --wandb-project my-project \
    --wandb-run-name experiment-1
```

This logs: hyperparameters, metrics per epoch, sample predictions, confusion matrices, training curves.

## Performance Expectations

**Training time (on RTX 3090, 3662 images):**
- Baseline ResNet50: ~4 hours (30 epochs)
- RETFound + LoRA (r=8): ~5 hours (20 epochs)
- Full RETFound fine-tuning: ~12 hours (20 epochs)

**Memory usage:**
- RETFound + LoRA: ~6-8GB GPU (batch_size=32)
- Full fine-tuning: ~11-12GB GPU (batch_size=32)
- Baseline ResNet50: ~4-5GB GPU (batch_size=32)

**Expected accuracies (on APTOS validation set):**
- Baseline ResNet50: 82-85%
- RETFound + LoRA (r=8): 88-89%
- Full RETFound fine-tuning: 89-90%

**Cross-dataset generalization gap:**
- ResNet50: ~7% drop on new datasets
- RETFound + LoRA: ~3% drop (50% reduction in gap)

## Important Notes

1. **This is research code** - prioritize reproducibility and experimentation
2. **Random seed is critical** - set `system.seed: 42` in config for reproducible results
3. **LoRA is the primary approach** - not baseline models (those are for comparison)
4. **Cross-dataset evaluation is the key metric** - not just single-dataset accuracy
5. **Configuration files should be versioned** - commit configs with results
6. **Training can be resumed** - always save checkpoints, use `--resume` flag
7. **Test suite should pass** - run `pytest tests/` before committing major changes

## Research Context

This project investigates the **domain shift problem** in DR screening:
- Models trained on one dataset (e.g., APTOS) perform poorly on different datasets (e.g., Messidor)
- Root causes: different imaging equipment, demographics, annotation protocols
- Solution: Use domain-specific foundation model (RETFound) + parameter-efficient fine-tuning (LoRA)
- Hypothesis: LoRA prevents overfitting to source domain, improves generalization

The code is structured to facilitate:
- Systematic comparison of different architectures
- Hyperparameter optimization with Optuna
- Cross-dataset evaluation to measure generalization
- Reproducible experiments with configuration management

## Dependencies

**Core framework:** PyTorch 2.0+, torchvision 0.15+
**Model libraries:** timm (baseline models), transformers + PEFT (LoRA)
**Data augmentation:** albumentations
**Hyperparameter search:** optuna
**Experiment tracking:** wandb (optional)
**Testing:** pytest

See `requirements.txt` for complete list and versions.
