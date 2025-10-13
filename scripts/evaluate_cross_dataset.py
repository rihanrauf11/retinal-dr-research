#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Script for Diabetic Retinopathy Models

This script evaluates trained DR models across multiple test datasets to assess
cross-dataset generalization performance. It supports both baseline models
(DRClassifier) and LoRA-adapted models (RETFoundLoRA).

Key Features:
    - Evaluate on multiple datasets (APTOS, Messidor, IDRiD, etc.)
    - Comprehensive metrics: accuracy, precision, recall, F1, Cohen's Kappa
    - Per-class performance analysis
    - Confusion matrix visualization
    - Cross-dataset comparison plots
    - Generalization gap calculation
    - JSON output for further analysis

Usage:
    # Evaluate baseline model
    python scripts/evaluate_cross_dataset.py \\
        --checkpoint results/baseline/checkpoints/best_model.pth \\
        --datasets \\
            APTOS:data/aptos/test.csv:data/aptos/images \\
            Messidor:data/messidor/test.csv:data/messidor/images \\
        --output_dir results/evaluation/baseline

    # Evaluate LoRA model
    python scripts/evaluate_cross_dataset.py \\
        --checkpoint results/retfound_lora/checkpoints/checkpoint_best.pth \\
        --model_type lora \\
        --datasets \\
            APTOS:data/aptos/test.csv:data/aptos/images \\
        --batch_size 64

Author: Generated with Claude Code
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project modules
try:
    from scripts.dataset import RetinalDataset
    from scripts.model import DRClassifier
    from scripts.retfound_lora import RETFoundLoRA
except ModuleNotFoundError:
    # Handle direct execution from scripts directory
    from dataset import RetinalDataset
    from model import DRClassifier
    from retfound_lora import RETFoundLoRA


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_model_type(checkpoint_path: str) -> str:
    """
    Auto-detect model type from checkpoint structure.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Model type: 'lora', 'baseline', or 'retfound'

    Raises:
        ValueError: If checkpoint format is unknown
    """
    print(f"[INFO] Detecting model type...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Check for LoRA model
    if 'lora_adapters' in checkpoint or 'lora_config' in checkpoint:
        print(f"[INFO] Detected model type: LoRA")
        return 'lora'

    # Check for baseline model
    if 'model_state_dict' in checkpoint:
        print(f"[INFO] Detected model type: Baseline")
        return 'baseline'

    # Check for raw RETFound model
    if 'model' in checkpoint or isinstance(checkpoint, dict):
        print(f"[INFO] Detected model type: RETFound (raw)")
        return 'retfound'

    raise ValueError(
        f"Unknown checkpoint format. Expected keys: 'lora_adapters', "
        f"'model_state_dict', or 'model'. Found: {list(checkpoint.keys())}"
    )


def load_baseline_model(
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Load DRClassifier model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in evaluation mode
    """
    print(f"[INFO] Loading baseline model...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model configuration
    model_name = checkpoint.get('model_name', 'resnet50')
    num_classes = checkpoint.get('num_classes', 5)

    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Classes: {num_classes}")

    # Create model
    model = DRClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False  # We're loading weights
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Get parameter counts
    if hasattr(model, 'get_num_params'):
        total, trainable = model.get_num_params()
        print(f"[INFO] Parameters: {total:,} total, {trainable:,} trainable")

    return model


def load_lora_model(
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Load RETFoundLoRA model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in evaluation mode
    """
    print(f"[INFO] Loading LoRA model...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get LoRA configuration
    lora_config = checkpoint.get('lora_config', {})
    retfound_path = lora_config.get('checkpoint_path')
    lora_r = lora_config.get('r', 8)
    lora_alpha = lora_config.get('alpha', 32)
    num_classes = checkpoint.get('num_classes', 5)

    if not retfound_path:
        raise ValueError(
            "LoRA checkpoint does not contain RETFound checkpoint path. "
            "Please provide it manually."
        )

    print(f"[INFO] RETFound checkpoint: {retfound_path}")
    print(f"[INFO] LoRA rank: {lora_r}, alpha: {lora_alpha}")
    print(f"[INFO] Classes: {num_classes}")

    # Create model
    model = RETFoundLoRA(
        checkpoint_path=retfound_path,
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        device=device
    )

    # Load LoRA adapters
    if 'lora_adapters' in checkpoint:
        model.backbone.load_state_dict(checkpoint['lora_adapters'], strict=False)

    # Load classifier
    if 'classifier_state' in checkpoint:
        model.classifier.load_state_dict(checkpoint['classifier_state'])

    # Set to eval mode
    model.eval()

    # Get parameter counts
    total = model.get_num_params(trainable_only=False)
    trainable = model.get_num_params(trainable_only=True)
    print(f"[INFO] Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.3f}%)")

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: str = 'auto',
    device: str = 'cuda'
) -> Tuple[nn.Module, str]:
    """
    Load model from checkpoint with automatic type detection.

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Model type ('auto', 'baseline', 'lora')
        device: Device to load model on

    Returns:
        (model, detected_model_type)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Auto-detect model type if needed
    if model_type == 'auto':
        model_type = detect_model_type(checkpoint_path)

    # Load appropriate model
    if model_type == 'lora':
        model = load_lora_model(checkpoint_path, device)
    elif model_type == 'baseline':
        model = load_baseline_model(checkpoint_path, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, model_type


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_evaluation_transform(img_size: int = 224) -> A.Compose:
    """
    Create evaluation transform (no augmentation).

    Args:
        img_size: Target image size

    Returns:
        Albumentations transform
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return transform


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing all metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Macro and weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Format results
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'cohen_kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {}
    }

    # Per-class metrics with class names
    class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']

    for i in range(min(len(class_names), len(precision_per_class))):
        metrics['per_class_metrics'][str(i)] = {
            'class_name': class_names[i],
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }

    return metrics


def evaluate_dataset(
    model: nn.Module,
    csv_path: str,
    img_dir: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict:
    """
    Evaluate model on a single dataset.

    Args:
        model: Model to evaluate
        csv_path: Path to CSV file with labels
        img_dir: Directory containing images
        dataset_name: Name of dataset for logging
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers

    Returns:
        Dictionary containing metrics and predictions
    """
    print(f"\n[INFO] Evaluating on {dataset_name}...")

    # Create transform
    transform = get_evaluation_transform()

    # Create dataset
    try:
        dataset = RetinalDataset(
            csv_file=csv_path,
            img_dir=img_dir,
            transform=transform
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset {dataset_name}: {e}")
        return None

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"[INFO] Dataset size: {len(dataset)} samples")

    # Inference
    all_predictions = []
    all_labels = []
    all_logits = []

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  Processing {dataset_name}"):
            # Move to device
            images = images.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = outputs.max(1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(outputs.cpu())

    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    logits = torch.cat(all_logits, dim=0).numpy()

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Add dataset-specific info
    metrics['dataset_name'] = dataset_name
    metrics['num_samples'] = len(y_true)
    metrics['predictions'] = y_pred.tolist()
    metrics['ground_truth'] = y_true.tolist()
    metrics['logits'] = logits.tolist()

    # Print summary
    print(f"  [RESULTS] {dataset_name}:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"    Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    dataset_name: str,
    save_path: Path,
    normalize: bool = True
) -> None:
    """
    Create and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        dataset_name: Name of dataset
        save_path: Path to save figure
        normalize: Whether to normalize by row (true labels)
    """
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    else:
        cm_normalized = cm

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        vmin=0,
        vmax=1 if normalize else None,
        ax=ax,
        square=True
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Confusion Matrix - {dataset_name}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cross_dataset_comparison(
    results_dict: Dict,
    output_dir: Path
) -> None:
    """
    Create bar plot comparing performance across datasets.

    Args:
        results_dict: Dictionary mapping dataset names to metrics
        output_dir: Directory to save plot
    """
    datasets = list(results_dict.keys())
    metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

    # Extract data
    data = {
        metric: [results_dict[ds][metric] for ds in datasets]
        for metric in metrics_to_plot
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.2

    # Colors for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        offset = width * (i - len(metrics_to_plot)/2 + 0.5)
        bars = ax.bar(
            x + offset,
            data[metric],
            width,
            label=label,
            color=colors[i],
            alpha=0.8
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(
        'Cross-Dataset Performance Comparison',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_heatmap(
    results_dict: Dict,
    output_dir: Path
) -> None:
    """
    Create heatmap of per-class F1 scores across datasets.

    Args:
        results_dict: Dictionary mapping dataset names to metrics
        output_dir: Directory to save plot
    """
    datasets = list(results_dict.keys())
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

    # Create matrix: rows=datasets, cols=classes
    f1_matrix = []
    for ds in datasets:
        row = []
        for class_idx in range(5):
            f1 = results_dict[ds]['per_class_metrics'].get(
                str(class_idx), {}
            ).get('f1', 0.0)
            row.append(f1)
        f1_matrix.append(row)

    f1_matrix = np.array(f1_matrix)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(datasets) * 1.2)))

    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=datasets,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'F1 Score'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title(
        'Per-Class F1 Scores Across Datasets',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_generalization_analysis(
    results_dict: Dict,
    summary: Dict,
    output_dir: Path
) -> None:
    """
    Create visualization showing generalization gap.

    Args:
        results_dict: Dictionary mapping dataset names to metrics
        summary: Summary statistics
        output_dir: Directory to save plot
    """
    datasets = list(results_dict.keys())
    accuracies = [results_dict[ds]['accuracy'] for ds in datasets]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracies
    x = np.arange(len(datasets))
    bars = ax.bar(x, accuracies, color='steelblue', alpha=0.7)

    # Highlight best and worst
    best_idx = accuracies.index(max(accuracies))
    worst_idx = accuracies.index(min(accuracies))

    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.8)
    bars[worst_idx].set_color('red')
    bars[worst_idx].set_alpha(0.8)

    # Add mean line
    ax.axhline(
        summary['mean_accuracy'],
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f"Mean: {summary['mean_accuracy']:.4f}"
    )

    # Add std bounds
    ax.axhline(
        summary['mean_accuracy'] + summary['std_accuracy'],
        color='orange',
        linestyle=':',
        linewidth=1,
        alpha=0.5
    )
    ax.axhline(
        summary['mean_accuracy'] - summary['std_accuracy'],
        color='orange',
        linestyle=':',
        linewidth=1,
        alpha=0.5
    )

    # Annotations
    ax.text(
        best_idx,
        accuracies[best_idx] + 0.02,
        f'Best\n{accuracies[best_idx]:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='green'
    )
    ax.text(
        worst_idx,
        accuracies[worst_idx] + 0.02,
        f'Worst\n{accuracies[worst_idx]:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='red'
    )

    # Add generalization gap annotation
    ax.annotate(
        '',
        xy=(worst_idx, accuracies[worst_idx]),
        xytext=(best_idx, accuracies[best_idx]),
        arrowprops=dict(arrowstyle='<->', color='purple', lw=2)
    )
    mid_x = (best_idx + worst_idx) / 2
    mid_y = (accuracies[best_idx] + accuracies[worst_idx]) / 2
    ax.text(
        mid_x,
        mid_y,
        f'Gap: {summary["generalization_gap"]:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='purple',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Generalization Analysis\nGap: {summary["generalization_gap"]:.4f} '
        f'(σ: {summary["std_accuracy"]:.4f})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'generalization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY & REPORTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_summary_statistics(results_dict: Dict) -> Dict:
    """
    Calculate cross-dataset summary statistics.

    Args:
        results_dict: Dictionary mapping dataset names to metrics

    Returns:
        Summary statistics dictionary
    """
    accuracies = [results_dict[ds]['accuracy'] for ds in results_dict]
    f1_scores = [results_dict[ds]['f1_macro'] for ds in results_dict]

    summary = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'generalization_gap': float(np.max(accuracies) - np.min(accuracies)),
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'best_dataset': max(results_dict.keys(), key=lambda k: results_dict[k]['accuracy']),
        'worst_dataset': min(results_dict.keys(), key=lambda k: results_dict[k]['accuracy']),
        'num_datasets': len(results_dict),
        'total_samples': sum(results_dict[ds]['num_samples'] for ds in results_dict)
    }

    return summary


def print_detailed_report(
    results_dict: Dict,
    summary: Dict,
    model_info: Dict
) -> None:
    """
    Print comprehensive evaluation report to console.

    Args:
        results_dict: Dictionary mapping dataset names to metrics
        summary: Summary statistics
        model_info: Model information
    """
    print("\n" + "=" * 80)
    print("CROSS-DATASET EVALUATION REPORT")
    print("=" * 80)

    # Model information
    print(f"\nModel Information:")
    print(f"  Checkpoint: {model_info['checkpoint_path']}")
    print(f"  Model Type: {model_info['model_type']}")
    if model_info.get('total_params'):
        total = model_info['total_params']
        trainable = model_info.get('trainable_params', total)
        print(f"  Parameters: {total:,} total, {trainable:,} trainable")
        if trainable < total:
            print(f"  Efficiency: {100 * trainable / total:.3f}% trainable")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Datasets Evaluated: {summary['num_datasets']}")
    print(f"  Total Samples: {summary['total_samples']:,}")
    print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f} (±{summary['std_accuracy']:.4f})")
    print(f"  Mean F1-Score: {summary['mean_f1']:.4f} (±{summary['std_f1']:.4f})")
    print(f"  Best Dataset: {summary['best_dataset']} ({summary['max_accuracy']:.4f})")
    print(f"  Worst Dataset: {summary['worst_dataset']} ({summary['min_accuracy']:.4f})")
    print(f"  Generalization Gap: {summary['generalization_gap']:.4f}")

    # Per-dataset results
    for dataset_name, metrics in sorted(results_dict.items()):
        print(f"\n{'-' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'-' * 80}")
        print(f"  Samples: {metrics['num_samples']:,}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

        # Per-class metrics
        print(f"\n  Per-Class Metrics:")
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print(f"  {'-' * 70}")
        for class_id, class_metrics in sorted(metrics['per_class_metrics'].items()):
            print(
                f"  {class_metrics['class_name']:<20} "
                f"{class_metrics['precision']:>10.4f} "
                f"{class_metrics['recall']:>10.4f} "
                f"{class_metrics['f1']:>10.4f} "
                f"{class_metrics['support']:>10}"
            )

    print("\n" + "=" * 80)


def save_predictions_csv(
    results_dict: Dict,
    output_dir: Path
) -> None:
    """
    Save predictions to CSV files.

    Args:
        results_dict: Dictionary mapping dataset names to metrics
        output_dir: Directory to save CSV files
    """
    import pandas as pd

    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, metrics in results_dict.items():
        # Create DataFrame
        df = pd.DataFrame({
            'ground_truth': metrics['ground_truth'],
            'prediction': metrics['predictions'],
            'correct': [
                gt == pred for gt, pred in
                zip(metrics['ground_truth'], metrics['predictions'])
            ]
        })

        # Add logits
        logits = np.array(metrics['logits'])
        for i in range(logits.shape[1]):
            df[f'logit_class_{i}'] = logits[:, i]

        # Save
        csv_path = predictions_dir / f'{dataset_name}_predictions.csv'
        df.to_csv(csv_path, index=False)
        print(f"  [INFO] Predictions saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for cross-dataset evaluation.
    """
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Evaluation for Diabetic Retinopathy Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate baseline model on multiple datasets
  python scripts/evaluate_cross_dataset.py \\
      --checkpoint results/baseline/checkpoints/best_model.pth \\
      --datasets \\
          APTOS:data/aptos/test.csv:data/aptos/images \\
          Messidor:data/messidor/test.csv:data/messidor/images \\
      --output_dir results/evaluation/baseline

  # Evaluate LoRA model
  python scripts/evaluate_cross_dataset.py \\
      --checkpoint results/retfound_lora/checkpoints/checkpoint_best.pth \\
      --model_type lora \\
      --datasets \\
          APTOS:data/aptos/test.csv:data/aptos/images \\
      --batch_size 64 \\
      --save_predictions

Dataset format: name:csv_path:image_directory
        """
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='Dataset specifications (format: name:csv_path:img_dir)'
    )

    # Optional arguments
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['auto', 'baseline', 'lora'],
        default='auto',
        help='Model type (default: auto-detect)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results (default: results/evaluation)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu, default: cuda)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )

    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save individual predictions to CSV files'
    )

    args = parser.parse_args()

    # Setup
    print("\n" + "=" * 80)
    print("CROSS-DATASET EVALUATION")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    # Load model
    print(f"\n[INFO] Loading model from: {args.checkpoint}")
    model, model_type = load_model_from_checkpoint(
        args.checkpoint,
        args.model_type,
        args.device
    )

    # Parse dataset specifications
    datasets = []
    for ds_spec in args.datasets:
        parts = ds_spec.split(':')
        if len(parts) != 3:
            print(f"[WARNING] Invalid dataset specification: {ds_spec}")
            print(f"          Expected format: name:csv_path:img_dir")
            continue

        name, csv, img_dir = parts
        datasets.append({
            'name': name,
            'csv': csv,
            'img_dir': img_dir
        })

    print(f"\n[INFO] Evaluating on {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"  - {ds['name']}: {ds['csv']}")

    # Evaluate on each dataset
    results = {}
    for ds in datasets:
        metrics = evaluate_dataset(
            model=model,
            csv_path=ds['csv'],
            img_dir=ds['img_dir'],
            dataset_name=ds['name'],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        if metrics is not None:
            results[ds['name']] = metrics

            # Plot confusion matrix
            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(
                cm,
                ds['name'],
                output_dir / f'confusion_matrix_{ds["name"]}.png'
            )

    if not results:
        print("[ERROR] No datasets were successfully evaluated")
        return

    # Calculate summary statistics
    print(f"\n[INFO] Calculating summary statistics...")
    summary = calculate_summary_statistics(results)

    # Create visualizations
    print(f"[INFO] Creating visualizations...")
    plot_cross_dataset_comparison(results, output_dir)
    plot_per_class_heatmap(results, output_dir)
    plot_generalization_analysis(results, summary, output_dir)

    # Prepare final results
    model_info = {
        'checkpoint_path': args.checkpoint,
        'model_type': model_type,
        'total_params': model.get_num_params(trainable_only=False) if hasattr(model, 'get_num_params') else None,
        'trainable_params': model.get_num_params(trainable_only=True) if hasattr(model, 'get_num_params') else None
    }

    final_results = {
        'model_info': model_info,
        'datasets': results,
        'summary': summary,
        'evaluation_time': datetime.now().isoformat()
    }

    # Save results to JSON
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"[INFO] Results saved to: {json_path}")

    # Save predictions if requested
    if args.save_predictions:
        print(f"\n[INFO] Saving predictions...")
        save_predictions_csv(results, output_dir)

    # Print detailed report
    print_detailed_report(results, summary, model_info)

    print(f"\n[INFO] Evaluation complete!")
    print(f"[INFO] All outputs saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
