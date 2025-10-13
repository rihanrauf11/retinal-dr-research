# API Documentation Index

## Overview

This directory contains comprehensive API documentation for the Diabetic Retinopathy Classification project. The documentation covers all major modules, classes, functions, and workflows.

**Documentation Status:** ✅ Complete

---

## Documentation Files

### 1. [Dataset API](dataset_api.md)
**RetinalDataset Class**

Documentation for loading and processing retinal fundus images for diabetic retinopathy classification.

**Key Topics:**
- `RetinalDataset` class initialization
- Image loading and transformations
- Class distribution analysis
- Integration with DataLoader
- Troubleshooting common issues

**Use this when:** Loading datasets, understanding data formats, debugging data loading issues

---

### 2. [Models API](models_api.md)
**Model Architectures**

Complete documentation for all model architectures including baseline models and parameter-efficient LoRA fine-tuning.

**Key Topics:**
- `DRClassifier` - Baseline models (ResNet, EfficientNet, ViT)
- `VisionTransformer` - RETFound foundation model architecture
- `RETFoundLoRA` - Parameter-efficient fine-tuning with LoRA
- Model loading and initialization
- Parameter counting and model summaries
- Model comparison and selection

**Use this when:** Creating models, understanding architectures, choosing between models, implementing LoRA

---

### 3. [Training API](training_api.md)
**Training Workflows**

Documentation for training pipelines including baseline training and LoRA fine-tuning.

**Key Topics:**
- `train()` - Main training function
- `train_one_epoch()` - Single epoch training
- `validate()` - Validation loop
- Data augmentation pipelines
- Checkpoint management
- Resume training functionality
- Command-line usage examples

**Use this when:** Training models, setting up experiments, configuring training pipelines, resuming interrupted training

---

### 4. [Evaluation API](evaluation_api.md)
**Model Evaluation**

Complete evaluation workflows including cross-dataset generalization assessment.

**Key Topics:**
- `evaluate_dataset()` - Single dataset evaluation
- `evaluate_cross_dataset()` - Multi-dataset evaluation
- Model loading utilities
- Comprehensive metrics calculation
- Cross-dataset generalization assessment
- Automatic visualization generation

**Use this when:** Evaluating trained models, assessing generalization, comparing model performance, generating reports

---

### 5. [Utils API](utils_api.md)
**Utility Functions**

Comprehensive collection of utility functions organized by category.

**Key Topics:**
- **Random Seed Management** - `set_seed()`
- **Model Utilities** - `count_parameters()`, `print_model_summary()`
- **Checkpoint Management** - `save_checkpoint()`, `load_checkpoint()`, `resume_training_from_checkpoint()`
- **Data Transforms** - `get_transforms()`, `get_imagenet_stats()`
- **Data Loaders** - `create_data_loaders()`, `create_dataloader_from_dataset()`
- **Metrics** - `calculate_metrics()`, `print_metrics()`
- **Visualization** - `plot_confusion_matrix()`, `plot_training_history()`
- **Device Management** - `get_device()`, `move_to_device()`
- **W&B Integration** - Complete Weights & Biases logging utilities

**Use this when:** Using utility functions, setting up experiments, calculating metrics, creating visualizations, integrating W&B

---

## Quick Reference

### Common Use Cases

#### Starting a New Project
1. Read [Dataset API](dataset_api.md) - Understand data loading
2. Read [Models API](models_api.md) - Choose model architecture
3. Read [Training API](training_api.md) - Set up training pipeline
4. Read [Utils API](utils_api.md) - Leverage utility functions

#### Training a Model
```python
# See Training API for complete examples
from scripts.config import Config
from scripts.train_baseline import train

config = Config.from_yaml('configs/default_config.yaml')
history = train(config, enable_wandb=True)
```

#### Evaluating a Model
```python
# See Evaluation API for complete examples
from scripts.evaluate_cross_dataset import evaluate_cross_dataset, load_baseline_model
import torch

device = torch.device('cuda')
model = load_baseline_model('checkpoints/best_model.pth', device)

datasets = [
    ('APTOS', 'data/aptos/test.csv', 'data/aptos/test_images'),
    ('Messidor', 'data/messidor/test.csv', 'data/messidor/images'),
]

results = evaluate_cross_dataset(model, datasets, device, 'results/eval')
```

#### Using LoRA for Efficient Fine-Tuning
```python
# See Models API and Training API for complete examples
from scripts.retfound_lora import RETFoundLoRA

model = RETFoundLoRA(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    lora_r=8,
    lora_alpha=32
)

# Only trains ~800K parameters (0.26% of model)
model.print_parameter_summary()
```

---

## Documentation Features

Each API documentation file includes:

✅ **Comprehensive Function Signatures** - With type hints and default values
✅ **Parameter Tables** - Detailed parameter descriptions
✅ **Return Value Documentation** - Types and descriptions
✅ **Exception Handling** - What exceptions can be raised and when
✅ **Complete Examples** - Copy-paste ready code snippets
✅ **Best Practices** - Tips and recommendations
✅ **Troubleshooting** - Common issues and solutions
✅ **Cross-References** - Links to related documentation

---

## Code Examples

All documentation includes complete, runnable code examples:

- **Minimal Examples** - Quick start snippets
- **Complete Workflows** - End-to-end examples
- **Command-Line Usage** - Bash command examples
- **Error Handling** - Exception handling patterns
- **Best Practices** - Production-ready code

---

## Contributing to Documentation

When adding new features to the codebase:

1. Update the relevant API documentation file
2. Add function signature with type hints
3. Document all parameters and return values
4. Include at least one complete example
5. Add any relevant exceptions
6. Update this README if adding new documentation files

---

## Additional Resources

### Project Documentation
- [README.md](../README.md) - Project overview and setup
- [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - Training workflow guide
- [MODEL_GUIDE.md](../MODEL_GUIDE.md) - Model architecture guide
- [RETFOUND_GUIDE.md](../RETFOUND_GUIDE.md) - RETFound integration guide
- [CONFIGURATION_GUIDE.md](../CONFIGURATION_GUIDE.md) - Configuration system guide

### Configuration Files
- [configs/](../configs/) - Configuration examples
- [configs/default_config.yaml](../configs/default_config.yaml) - Default configuration
- [configs/retfound_lora_config.yaml](../configs/retfound_lora_config.yaml) - LoRA configuration

### Source Code
- [scripts/](../scripts/) - All source code modules

---

## Getting Help

### Documentation Issues
If you find errors or have suggestions for improving the documentation:
1. Check the source code for the most up-to-date information
2. Review the complete examples in each documentation file
3. Consult related documentation files
4. Check the main [README.md](../README.md) for project-level information

### Code Usage Questions
1. Start with the relevant API documentation
2. Review the complete examples section
3. Check the troubleshooting section
4. Refer to the source code comments

---

## Documentation Statistics

| File | Size | Functions/Classes Documented | Examples |
|------|------|------------------------------|----------|
| [dataset_api.md](dataset_api.md) | 18 KB | 5 methods | 9+ |
| [models_api.md](models_api.md) | 30 KB | 20+ methods | 15+ |
| [training_api.md](training_api.md) | 19 KB | 10+ functions | 12+ |
| [evaluation_api.md](evaluation_api.md) | 19 KB | 5+ functions | 8+ |
| [utils_api.md](utils_api.md) | 33 KB | 30+ functions | 25+ |
| **Total** | **119 KB** | **70+ documented** | **69+ examples** |

---

**Generated with Claude Code** | Last Updated: 2024

---

## Quick Navigation

- [← Back to Project README](../README.md)
- [Dataset API →](dataset_api.md)
- [Models API →](models_api.md)
- [Training API →](training_api.md)
- [Evaluation API →](evaluation_api.md)
- [Utils API →](utils_api.md)
