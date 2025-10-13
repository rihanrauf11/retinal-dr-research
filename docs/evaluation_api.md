# Evaluation API Documentation

## Overview

The evaluation module provides comprehensive model evaluation capabilities including cross-dataset generalization assessment, detailed metrics calculation, and result visualization.

**Key Features:**
- Cross-dataset evaluation for generalization assessment
- Comprehensive metrics (accuracy, precision, recall, F1, Cohen's Kappa)
- Per-class performance analysis
- Confusion matrix visualization
- Automatic model type detection
- Support for baseline and LoRA models
- JSON output for further analysis

**Source File:**
- [scripts/evaluate_cross_dataset.py](../scripts/evaluate_cross_dataset.py:1)

---

## Table of Contents

1. [Model Loading Functions](#model-loading-functions)
2. [Evaluation Functions](#evaluation-functions)
3. [Metrics and Visualization](#metrics-and-visualization)
4. [Complete Examples](#complete-examples)
5. [Command Line Usage](#command-line-usage)

---

## Model Loading Functions

### detect_model_type

Auto-detect model type from checkpoint structure.

**Signature:**
```python
def detect_model_type(checkpoint_path: str) -> str
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | Path to checkpoint file |

**Returns:**

| Type | Description |
|------|-------------|
| `str` | Model type: `'lora'`, `'baseline'`, or `'retfound'` |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If checkpoint format is unknown |

**Detection Logic:**
- If `'lora_adapters'` or `'lora_config'` exists → `'lora'`
- If `'model_state_dict'` exists → `'baseline'`
- If `'model'` key exists → `'retfound'`

**Example:**

```python
from scripts.evaluate_cross_dataset import detect_model_type

model_type = detect_model_type('checkpoints/best_model.pth')
print(f"Detected model type: {model_type}")
# Output: Detected model type: baseline

lora_type = detect_model_type('checkpoints/lora_best.pth')
print(f"Detected model type: {lora_type}")
# Output: Detected model type: lora
```

---

### load_baseline_model

Load DRClassifier model from checkpoint.

**Signature:**
```python
def load_baseline_model(
    checkpoint_path: str,
    device: torch.device
) -> nn.Module
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | Path to checkpoint file |
| `device` | `torch.device` | Device to load model on |

**Returns:**

| Type | Description |
|------|-------------|
| `nn.Module` | Loaded model in evaluation mode |

**Example:**

```python
from scripts.evaluate_cross_dataset import load_baseline_model
import torch

device = torch.device('cuda')
model = load_baseline_model(
    checkpoint_path='checkpoints/resnet50_best.pth',
    device=device
)

# Output:
# [INFO] Loading baseline model...
# [INFO] Model: resnet50
# [INFO] Classes: 5
# [INFO] Parameters: 23,516,997 total, 23,516,997 trainable
```

---

### load_lora_model

Load RETFoundLoRA model from checkpoint.

**Signature:**
```python
def load_lora_model(
    checkpoint_path: str,
    device: torch.device
) -> nn.Module
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | Path to LoRA checkpoint |
| `device` | `torch.device` | Device to load model on |

**Returns:**

| Type | Description |
|------|-------------|
| `nn.Module` | Loaded LoRA model in evaluation mode |

**Example:**

```python
from scripts.evaluate_cross_dataset import load_lora_model
import torch

device = torch.device('cuda')
model = load_lora_model(
    checkpoint_path='checkpoints/lora_best.pth',
    device=device
)

# Output:
# [INFO] Loading LoRA model...
# LoRA configuration loaded
# Model ready for evaluation
```

---

## Evaluation Functions

### evaluate_dataset

Evaluate model on a single dataset.

**Signature:**
```python
def evaluate_dataset(
    model: nn.Module,
    dataset_name: str,
    csv_file: str,
    img_dir: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | Model to evaluate (in eval mode) |
| `dataset_name` | `str` | Required | Name of dataset for reporting |
| `csv_file` | `str` | Required | Path to test CSV |
| `img_dir` | `str` | Required | Path to test images |
| `device` | `torch.device` | Required | Device for evaluation |
| `batch_size` | `int` | `64` | Batch size for evaluation |
| `num_workers` | `int` | `4` | Number of data loading workers |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Dictionary with metrics, predictions, and metadata |

**Return Dictionary Structure:**
```python
{
    'dataset_name': str,
    'accuracy': float,
    'precision_macro': float,
    'precision_weighted': float,
    'recall_macro': float,
    'recall_weighted': float,
    'f1_macro': float,
    'f1_weighted': float,
    'cohen_kappa': float,
    'confusion_matrix': List[List[int]],
    'per_class_metrics': Dict[str, Dict],
    'predictions': List[int],
    'ground_truth': List[int],
    'num_samples': int
}
```

**Example:**

```python
from scripts.evaluate_cross_dataset import evaluate_dataset, load_baseline_model
import torch

# Load model
device = torch.device('cuda')
model = load_baseline_model('checkpoints/best_model.pth', device)

# Evaluate on test set
results = evaluate_dataset(
    model=model,
    dataset_name='APTOS Test',
    csv_file='data/aptos/test.csv',
    img_dir='data/aptos/test_images',
    device=device,
    batch_size=64
)

# Print results
print(f"Dataset: {results['dataset_name']}")
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"F1-Score (macro): {results['f1_macro']:.4f}")
print(f"Cohen's Kappa: {results['cohen_kappa']:.4f}")

# Per-class metrics
print("\nPer-class metrics:")
for class_id, metrics in results['per_class_metrics'].items():
    print(f"  Class {class_id}: Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
```

---

### evaluate_cross_dataset

Evaluate model across multiple datasets for generalization assessment.

**Signature:**
```python
def evaluate_cross_dataset(
    model: nn.Module,
    datasets: List[Tuple[str, str, str]],
    device: torch.device,
    output_dir: str,
    batch_size: int = 64
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | Model to evaluate |
| `datasets` | `List[Tuple[str, str, str]]` | Required | List of (name, csv_file, img_dir) tuples |
| `device` | `torch.device` | Required | Device for evaluation |
| `output_dir` | `str` | Required | Directory to save results and visualizations |
| `batch_size` | `int` | `64` | Batch size for evaluation |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Comprehensive results for all datasets |

**Return Dictionary Structure:**
```python
{
    'results_by_dataset': {
        'dataset1': {...metrics...},
        'dataset2': {...metrics...},
        ...
    },
    'summary': {
        'mean_accuracy': float,
        'std_accuracy': float,
        'generalization_gap': float,  # max - min accuracy
        'datasets_evaluated': int
    },
    'output_dir': str
}
```

**Example:**

```python
from scripts.evaluate_cross_dataset import evaluate_cross_dataset, load_lora_model
import torch

# Load model
device = torch.device('cuda')
model = load_lora_model('checkpoints/lora_best.pth', device)

# Define datasets for cross-dataset evaluation
datasets = [
    ('APTOS', 'data/aptos/test.csv', 'data/aptos/test_images'),
    ('Messidor-2', 'data/messidor/test.csv', 'data/messidor/images'),
    ('IDRiD', 'data/idrid/test.csv', 'data/idrid/images'),
]

# Evaluate across datasets
results = evaluate_cross_dataset(
    model=model,
    datasets=datasets,
    device=device,
    output_dir='results/cross_dataset_evaluation',
    batch_size=64
)

# Print summary
print("Cross-Dataset Evaluation Summary:")
print("=" * 60)
for dataset_name, metrics in results['results_by_dataset'].items():
    print(f"{dataset_name:15s}: {metrics['accuracy']:>6.2%}  "
          f"F1={metrics['f1_macro']:.4f}  Kappa={metrics['cohen_kappa']:.4f}")

print("\nGeneralization:")
print(f"  Mean accuracy: {results['summary']['mean_accuracy']:.2%}")
print(f"  Std deviation: {results['summary']['std_accuracy']:.2%}")
print(f"  Generalization gap: {results['summary']['generalization_gap']:.2%}")
print(f"\nResults saved to: {results['output_dir']}")

# Output files created:
#   - results/cross_dataset_evaluation/results.json
#   - results/cross_dataset_evaluation/confusion_matrix_APTOS.png
#   - results/cross_dataset_evaluation/confusion_matrix_Messidor-2.png
#   - results/cross_dataset_evaluation/confusion_matrix_IDRiD.png
#   - results/cross_dataset_evaluation/comparison_plot.png
```

---

## Metrics and Visualization

### Metrics Calculated

For each dataset, the following metrics are calculated:

**Overall Metrics:**
- **Accuracy**: Proportion of correct predictions
- **Precision** (macro/weighted): Average precision across classes
- **Recall** (macro/weighted): Average recall across classes
- **F1-Score** (macro/weighted): Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement between predictions and ground truth (accounting for chance)

**Per-Class Metrics:**
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Support**: Number of samples in each class

**Confusion Matrix:**
- Row: Ground truth labels
- Column: Predicted labels
- Cell (i, j): Number of samples with true label i predicted as j

---

### Visualization Outputs

When running cross-dataset evaluation, the following visualizations are automatically generated:

1. **Confusion Matrices** (`confusion_matrix_{dataset}.png`)
   - Normalized heatmap showing prediction patterns
   - Row-normalized to show percentages
   - Class labels on axes

2. **Comparison Plot** (`comparison_plot.png`)
   - Bar plot comparing accuracy across datasets
   - Error bars showing confidence intervals
   - Generalization gap highlighted

3. **Results JSON** (`results.json`)
   - Complete results in machine-readable format
   - Can be loaded for further analysis

---

## Complete Examples

### Example 1: Single Dataset Evaluation

```python
#!/usr/bin/env python3
"""Evaluate model on single test dataset."""

from scripts.evaluate_cross_dataset import detect_model_type, load_baseline_model, evaluate_dataset
import torch
import json

# Configuration
CHECKPOINT_PATH = 'checkpoints/resnet50_best.pth'
TEST_CSV = 'data/aptos/test.csv'
TEST_IMG_DIR = 'data/aptos/test_images'
OUTPUT_FILE = 'results/test_evaluation.json'

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Detect and load model
model_type = detect_model_type(CHECKPOINT_PATH)
print(f"Model type: {model_type}")

if model_type == 'baseline':
    model = load_baseline_model(CHECKPOINT_PATH, device)
elif model_type == 'lora':
    from scripts.evaluate_cross_dataset import load_lora_model
    model = load_lora_model(CHECKPOINT_PATH, device)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# Step 2: Evaluate
print(f"\nEvaluating on test set...")
results = evaluate_dataset(
    model=model,
    dataset_name='APTOS Test',
    csv_file=TEST_CSV,
    img_dir=TEST_IMG_DIR,
    device=device,
    batch_size=64,
    num_workers=4
)

# Step 3: Print results
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nDataset: {results['dataset_name']}")
print(f"Samples: {results['num_samples']}")

print(f"\nOverall Metrics:")
print(f"  Accuracy:         {results['accuracy']:>7.2%}")
print(f"  Precision (macro):{results['precision_macro']:>7.4f}")
print(f"  Recall (macro):   {results['recall_macro']:>7.4f}")
print(f"  F1-Score (macro): {results['f1_macro']:>7.4f}")
print(f"  Cohen's Kappa:    {results['cohen_kappa']:>7.4f}")

print(f"\nPer-Class Performance:")
class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
print("-" * 60)

for class_id, metrics in results['per_class_metrics'].items():
    class_idx = int(class_id)
    print(f"{class_names[class_idx]:<15} "
          f"{metrics['precision']:<10.4f} "
          f"{metrics['recall']:<10.4f} "
          f"{metrics['f1-score']:<10.4f} "
          f"{metrics['support']:<8}")

# Step 4: Save results
import os
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {OUTPUT_FILE}")

# Step 5: Visualize confusion matrix
from scripts.utils import plot_confusion_matrix
import numpy as np

plot_confusion_matrix(
    y_true=np.array(results['ground_truth']),
    y_pred=np.array(results['predictions']),
    classes=class_names,
    save_path='results/confusion_matrix.png',
    normalize=True,
    title='Confusion Matrix - APTOS Test Set'
)
print("Confusion matrix saved to: results/confusion_matrix.png")
```

### Example 2: Cross-Dataset Generalization Assessment

```python
#!/usr/bin/env python3
"""Assess model generalization across multiple datasets."""

from scripts.evaluate_cross_dataset import (
    detect_model_type, load_baseline_model, load_lora_model, evaluate_cross_dataset
)
import torch
import json
import pandas as pd

# Configuration
CHECKPOINT_PATH = 'checkpoints/lora_best.pth'
OUTPUT_DIR = 'results/generalization_assessment'

# Define test datasets
DATASETS = [
    ('APTOS (Val)', 'data/aptos/val.csv', 'data/aptos/train_images'),
    ('Messidor-2', 'data/messidor/test.csv', 'data/messidor/images'),
    ('EyePACS', 'data/eyepacs/test.csv', 'data/eyepacs/images'),
    ('IDRiD', 'data/idrid/test.csv', 'data/idrid/images'),
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load model
print("Loading model...")
model_type = detect_model_type(CHECKPOINT_PATH)

if model_type == 'lora':
    model = load_lora_model(CHECKPOINT_PATH, device)
elif model_type == 'baseline':
    model = load_baseline_model(CHECKPOINT_PATH, device)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

print(f"Model type: {model_type}\n")

# Evaluate across datasets
print("Starting cross-dataset evaluation...")
print("=" * 70)

results = evaluate_cross_dataset(
    model=model,
    datasets=DATASETS,
    device=device,
    output_dir=OUTPUT_DIR,
    batch_size=64
)

# Print detailed results
print("\n" + "=" * 70)
print("CROSS-DATASET EVALUATION RESULTS")
print("=" * 70)

# Results table
print(f"\n{'Dataset':<20} {'Accuracy':<10} {'F1-Macro':<10} {'Kappa':<10} {'Samples':<8}")
print("-" * 70)

for dataset_name in DATASETS:
    name = dataset_name[0]
    metrics = results['results_by_dataset'][name]
    print(f"{name:<20} "
          f"{metrics['accuracy']:<10.2%} "
          f"{metrics['f1_macro']:<10.4f} "
          f"{metrics['cohen_kappa']:<10.4f} "
          f"{metrics['num_samples']:<8}")

# Summary statistics
print(f"\n{'SUMMARY STATISTICS':<20}")
print("-" * 70)
print(f"Mean Accuracy:        {results['summary']['mean_accuracy']:.2%}")
print(f"Std Deviation:        {results['summary']['std_accuracy']:.2%}")
print(f"Generalization Gap:   {results['summary']['generalization_gap']:.2%}")
print(f"Datasets Evaluated:   {results['summary']['datasets_evaluated']}")

# Calculate drop from validation set
val_acc = results['results_by_dataset']['APTOS (Val)']['accuracy']
min_acc = min(m['accuracy'] for m in results['results_by_dataset'].values())
max_drop = val_acc - min_acc

print(f"\nPerformance Drop:")
print(f"  Validation accuracy:  {val_acc:.2%}")
print(f"  Minimum accuracy:     {min_acc:.2%}")
print(f"  Maximum drop:         {max_drop:.2%}")

# Save summary to CSV for easy viewing
summary_data = []
for dataset_name in DATASETS:
    name = dataset_name[0]
    metrics = results['results_by_dataset'][name]
    summary_data.append({
        'Dataset': name,
        'Accuracy': metrics['accuracy'],
        'F1-Score (Macro)': metrics['f1_macro'],
        'Precision (Macro)': metrics['precision_macro'],
        'Recall (Macro)': metrics['recall_macro'],
        "Cohen's Kappa": metrics['cohen_kappa'],
        'Samples': metrics['num_samples']
    })

df = pd.DataFrame(summary_data)
csv_path = f"{OUTPUT_DIR}/summary.csv"
df.to_csv(csv_path, index=False)
print(f"\nSummary saved to: {csv_path}")

print("\n" + "=" * 70)
print(f"All results and visualizations saved to: {OUTPUT_DIR}")
print("=" * 70)
```

---

## Command Line Usage

### Evaluate Baseline Model

```bash
# Single dataset evaluation
python scripts/evaluate_cross_dataset.py \
    --checkpoint checkpoints/resnet50_best.pth \
    --datasets APTOS:data/aptos/test.csv:data/aptos/test_images \
    --output_dir results/evaluation/resnet50

# Cross-dataset evaluation
python scripts/evaluate_cross_dataset.py \
    --checkpoint checkpoints/resnet50_best.pth \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/test_images \
        Messidor:data/messidor/test.csv:data/messidor/images \
        IDRiD:data/idrid/test.csv:data/idrid/images \
    --output_dir results/cross_dataset/resnet50 \
    --batch_size 64
```

### Evaluate LoRA Model

```bash
# Cross-dataset evaluation with LoRA
python scripts/evaluate_cross_dataset.py \
    --checkpoint checkpoints/lora_best.pth \
    --model_type lora \
    --datasets \
        "APTOS Val:data/aptos/val.csv:data/aptos/train_images" \
        "Messidor-2:data/messidor/test.csv:data/messidor/images" \
        "EyePACS:data/eyepacs/test.csv:data/eyepacs/images" \
    --output_dir results/generalization/lora_r8 \
    --batch_size 64
```

### Compare Multiple Models

```bash
#!/bin/bash
# Compare baseline and LoRA models

# Evaluate ResNet50
python scripts/evaluate_cross_dataset.py \
    --checkpoint checkpoints/resnet50_best.pth \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/test_images \
        Messidor:data/messidor/test.csv:data/messidor/images \
    --output_dir results/comparison/resnet50

# Evaluate EfficientNet
python scripts/evaluate_cross_dataset.py \
    --checkpoint checkpoints/efficientnet_best.pth \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/test_images \
        Messidor:data/messidor/test.csv:data/messidor/images \
    --output_dir results/comparison/efficientnet

# Evaluate LoRA
python scripts/evaluate_cross_dataset.py \
    --checkpoint checkpoints/lora_best.pth \
    --model_type lora \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/test_images \
        Messidor:data/messidor/test.csv:data/messidor/images \
    --output_dir results/comparison/lora

echo "Evaluation complete! Check results/comparison/ for all results."
```

---

## See Also

- [Models API Documentation](models_api.md) - Model architectures
- [Training API Documentation](training_api.md) - Training workflows
- [Utils API Documentation](utils_api.md) - Utility functions including metrics
- [Dataset API Documentation](dataset_api.md) - Dataset loading

---

**Generated with Claude Code** | Last Updated: 2024
