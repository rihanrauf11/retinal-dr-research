"""
Utility Functions for Diabetic Retinopathy Classification.

This module provides reusable utility functions for:
- Random seed management
- Model parameter counting
- Checkpoint saving/loading
- Data transforms and loaders
- Metrics calculation
- Visualization
- Training history management
- Device management

All functions include proper type hints and comprehensive docstrings.

Author: Generated with Claude Code
"""

import os
import json
import random
from pathlib import Path
from typing import (
    Dict, List, Tuple, Optional, Any, Union, Callable, Iterable
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SEED MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int, deterministic: bool = True, verbose: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDNN (optionally deterministic)

    Args:
        seed: Random seed value (typically 42, 123, etc.)
        deterministic: If True, makes CUDNN deterministic (slower but reproducible)
        verbose: If True, print confirmation message

    Example:
        >>> from scripts.utils import set_seed
        >>> set_seed(42, deterministic=True)
        ✓ Random seed set to 42 (deterministic mode: ON)

    Note:
        Deterministic mode may reduce performance but ensures full reproducibility.
        Set deterministic=False for faster training at cost of minor variation.
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # CUDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if verbose:
        det_status = "ON" if deterministic else "OFF"
        print(f"✓ Random seed set to {seed} (deterministic mode: {det_status})")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PARAMETER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module, trainable_only: bool = False) -> Union[int, Tuple[int, int]]:
    """
    Count parameters in a PyTorch model.

    Args:
        model: PyTorch model
        trainable_only: If True, return only trainable parameters
                       If False, return (total, trainable) tuple

    Returns:
        int: Number of trainable parameters (if trainable_only=True)
        Tuple[int, int]: (total_params, trainable_params) (if trainable_only=False)

    Example:
        >>> from scripts.utils import count_parameters
        >>> total, trainable = count_parameters(model)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
        Total: 25,557,032, Trainable: 25,557,032

        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Trainable: {trainable:,}")
        Trainable: 25,557,032
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable_only:
        return trainable_params
    else:
        return total_params, trainable_params


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Print comprehensive model summary.

    Args:
        model: PyTorch model
        input_size: Input tensor shape (batch, channels, height, width)
        verbose: If True, print detailed summary

    Returns:
        Dict containing model statistics

    Example:
        >>> from scripts.utils import print_model_summary
        >>> stats = print_model_summary(model, input_size=(1, 3, 224, 224))
        ╔═══════════════════════════════════════════════════════╗
        ║              MODEL SUMMARY                            ║
        ╚═══════════════════════════════════════════════════════╝
        Total Parameters:        25,557,032
        Trainable Parameters:    25,557,032
        Non-trainable Parameters: 0
        Memory Size:             97.49 MB
        Input Shape:             (1, 3, 224, 224)
        Output Shape:            (1, 5)
    """
    total_params, trainable_params = count_parameters(model)
    non_trainable = total_params - trainable_params

    # Estimate memory (float32 = 4 bytes per parameter)
    memory_mb = (total_params * 4) / (1024 ** 2)

    # Test forward pass to get output shape
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            dummy_input = torch.randn(*input_size).to(device)
            output = model(dummy_input)
            output_shape = tuple(output.shape)
            forward_pass_ok = True
    except Exception as e:
        output_shape = "Error"
        forward_pass_ok = False

    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable,
        'memory_mb': memory_mb,
        'input_shape': input_size,
        'output_shape': output_shape,
        'forward_pass_ok': forward_pass_ok
    }

    if verbose:
        print("╔" + "═" * 59 + "╗")
        print("║" + " " * 15 + "MODEL SUMMARY" + " " * 31 + "║")
        print("╚" + "═" * 59 + "╝")
        print(f"Total Parameters:        {total_params:>15,}")
        print(f"Trainable Parameters:    {trainable_params:>15,}")
        print(f"Non-trainable Parameters:{non_trainable:>15,}")
        print(f"Memory Size:             {memory_mb:>12.2f} MB")
        print(f"Input Shape:             {str(input_size):>15s}")
        print(f"Output Shape:            {str(output_shape):>15s}")

        if forward_pass_ok:
            print("✓ Forward pass successful")
        else:
            print("✗ Forward pass failed")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    path: Union[str, Path],
    is_best: bool = False,
    **kwargs
) -> None:
    """
    Save model checkpoint with comprehensive state.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer (can be None)
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'val_acc': 0.85, 'val_loss': 0.45})
        path: Path to save checkpoint
        is_best: If True, also save as 'best_model.pth' in same directory
        **kwargs: Additional items to save (scheduler, history, config, etc.)

    Saved checkpoint structure:
        {
            'epoch': int,
            'model_state_dict': OrderedDict,
            'optimizer_state_dict': OrderedDict (if optimizer provided),
            'metrics': dict,
            'model_class': str,
            **kwargs (any additional items)
        }

    Example:
        >>> from scripts.utils import save_checkpoint
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'val_acc': 85.3, 'val_loss': 0.42},
        ...     path='checkpoints/epoch_10.pth',
        ...     is_best=True,
        ...     scheduler=scheduler.state_dict(),
        ...     history=training_history,
        ...     config=config
        ... )
        ✓ Checkpoint saved: checkpoints/epoch_10.pth
        ✓ Best model saved: checkpoints/best_model.pth
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'model_class': model.__class__.__name__,
    }

    # Add optimizer if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Add any additional items
    checkpoint.update(kwargs)

    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")

    # Save as best model if requested
    if is_best:
        best_path = path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved: {best_path}")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str = 'cpu',
    strict: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        map_location: Device to map checkpoint ('cpu', 'cuda', 'cuda:0', etc.)
        strict: If True, require exact key match in state_dict
        verbose: If True, print loading info

    Returns:
        Dict containing checkpoint metadata (epoch, metrics, etc.)

    Example:
        >>> from scripts.utils import load_checkpoint
        >>> metadata = load_checkpoint(
        ...     path='checkpoints/epoch_10.pth',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     map_location='cuda'
        ... )
        ✓ Checkpoint loaded: checkpoints/epoch_10.pth
        ✓ Resuming from epoch 10
        ✓ Best validation accuracy: 85.30%

        >>> start_epoch = metadata['epoch'] + 1
        >>> best_acc = metadata['metrics']['val_acc']
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if verbose:
            print(f"✓ Checkpoint loaded: {path}")
    except RuntimeError as e:
        if not strict:
            print(f"⚠ Warning: Some keys mismatch (strict=False): {e}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Print info
    if verbose:
        if 'epoch' in checkpoint:
            print(f"✓ Resuming from epoch {checkpoint['epoch']}")

        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if 'val_acc' in metrics:
                print(f"✓ Best validation accuracy: {metrics['val_acc']:.2f}%")
            elif 'accuracy' in metrics:
                print(f"✓ Accuracy: {metrics['accuracy']:.2f}%")

    # Return metadata
    metadata = {k: v for k, v in checkpoint.items()
               if k not in ['model_state_dict', 'optimizer_state_dict']}

    return metadata


def resume_training_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = 'cpu'
) -> Tuple[int, float, Dict]:
    """
    High-level function to resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load
        optimizer: Optimizer to load (optional)
        scheduler: Learning rate scheduler to load (optional)
        map_location: Device to map to

    Returns:
        Tuple of (start_epoch, best_metric, history)
        - start_epoch: Epoch to resume from (loaded_epoch + 1)
        - best_metric: Best validation metric achieved
        - history: Training history dict

    Example:
        >>> from scripts.utils import resume_training_from_checkpoint
        >>> start_epoch, best_acc, history = resume_training_from_checkpoint(
        ...     checkpoint_path='checkpoints/latest.pth',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler
        ... )
        >>> print(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    """
    metadata = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        map_location=map_location,
        verbose=True
    )

    # Extract resume information
    start_epoch = metadata.get('epoch', 0) + 1
    metrics = metadata.get('metrics', {})
    best_metric = metrics.get('val_acc', 0.0)
    history = metadata.get('history', {})

    # Load scheduler if provided
    if scheduler is not None and 'scheduler_state_dict' in metadata:
        scheduler.load_state_dict(metadata['scheduler_state_dict'])

    return start_epoch, best_metric, history


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_imagenet_stats() -> Dict[str, List[float]]:
    """
    Get ImageNet normalization statistics.

    Returns:
        Dict with 'mean' and 'std' keys containing RGB values

    Example:
        >>> from scripts.utils import get_imagenet_stats
        >>> stats = get_imagenet_stats()
        >>> print(stats)
        {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    """
    return {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }


def get_transforms(
    img_size: int = 224,
    is_train: bool = True,
    augmentation_level: str = 'medium',
    backend: str = 'albumentations'
) -> Union[A.Compose, transforms.Compose]:
    """
    Create data transformation pipeline with configurable augmentation.

    Args:
        img_size: Target image size (will be square: img_size x img_size)
        is_train: If True, apply augmentation; if False, only resize/normalize
        augmentation_level: One of 'light', 'medium', 'heavy' (ignored if is_train=False)
        backend: 'albumentations' or 'torchvision'

    Returns:
        Transform pipeline (A.Compose or transforms.Compose depending on backend)

    Augmentation Levels:
        - light: Basic flips only
        - medium: Flips + rotation + color jitter (default)
        - heavy: Flips + rotation + color + coarse dropout

    Example:
        >>> from scripts.utils import get_transforms
        >>> # Training transforms with medium augmentation
        >>> train_transform = get_transforms(224, is_train=True, augmentation_level='medium')
        >>>
        >>> # Validation transforms (no augmentation)
        >>> val_transform = get_transforms(224, is_train=False)
        >>>
        >>> # Heavy augmentation for small datasets
        >>> heavy_transform = get_transforms(224, is_train=True, augmentation_level='heavy')
    """
    stats = get_imagenet_stats()
    mean, std = stats['mean'], stats['std']

    if backend == 'albumentations':
        if is_train:
            # Training transforms with augmentation
            if augmentation_level == 'light':
                transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2()
                ])

            elif augmentation_level == 'medium':
                transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=45,
                        p=0.5
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=0.5
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2()
                ])

            elif augmentation_level == 'heavy':
                transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=45,
                        p=0.5
                    ),
                    A.OneOf([
                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                        A.RandomBrightnessContrast(p=1.0),
                    ], p=0.5),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        fill_value=0,
                        p=0.3
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2()
                ])
            else:
                raise ValueError(f"Invalid augmentation_level: {augmentation_level}. "
                               "Choose from: 'light', 'medium', 'heavy'")

        else:
            # Validation transforms (no augmentation)
            transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    elif backend == 'torchvision':
        if is_train:
            # Training transforms
            if augmentation_level == 'light':
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

            elif augmentation_level == 'medium':
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

            elif augmentation_level == 'heavy':
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:
                raise ValueError(f"Invalid augmentation_level: {augmentation_level}")

        else:
            # Validation transforms
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    else:
        raise ValueError(f"Invalid backend: {backend}. Choose from: 'albumentations', 'torchvision'")

    return transform


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/validation data loaders with automatic splitting.

    Args:
        dataset: PyTorch Dataset to split
        batch_size: Batch size for both loaders
        split_ratio: Fraction for training (e.g., 0.8 = 80% train, 20% val)
        num_workers: Number of data loading workers
        pin_memory: If True, use pinned memory (faster for CUDA)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> from scripts.utils import create_data_loaders
        >>> from scripts.dataset import RetinalDataset
        >>>
        >>> dataset = RetinalDataset('train.csv', 'images/', transform=train_transform)
        >>> train_loader, val_loader = create_data_loaders(
        ...     dataset=dataset,
        ...     batch_size=32,
        ...     split_ratio=0.8,
        ...     num_workers=4
        ... )
        >>> print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        Train batches: 250, Val batches: 63
    """
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    # Auto-detect pin_memory
    if pin_memory == True and not torch.cuda.is_available():
        pin_memory = False

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"✓ Created data loaders: train={train_size}, val={val_size}")

    return train_loader, val_loader


def create_dataloader_from_dataset(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a single DataLoader from dataset.

    Convenience wrapper for cleaner code.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Use pinned memory for CUDA

    Returns:
        DataLoader

    Example:
        >>> from scripts.utils import create_dataloader_from_dataset
        >>> loader = create_dataloader_from_dataset(
        ...     dataset=test_dataset,
        ...     batch_size=64,
        ...     shuffle=False
        ... )
    """
    # Auto-detect pin_memory
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (shape: [n_samples])
        y_pred: Predicted labels (shape: [n_samples])
        num_classes: Number of classes

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision_macro/weighted: Macro and weighted precision
        - recall_macro/weighted: Macro and weighted recall
        - f1_macro/weighted: Macro and weighted F1-score
        - cohen_kappa: Cohen's Kappa coefficient
        - confusion_matrix: Confusion matrix as nested list
        - per_class_metrics: Dict mapping class index to metrics

    Example:
        >>> from scripts.utils import calculate_metrics
        >>> import numpy as np
        >>>
        >>> y_true = np.array([0, 1, 2, 3, 4, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 3, 4, 0, 1])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>>
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
        >>> print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    """
    # Overall accuracy
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

    # Build results dictionary
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

    # Add per-class metrics
    for i in range(len(precision_per_class)):
        metrics['per_class_metrics'][str(i)] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1-score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }

    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics", precision: int = 4) -> None:
    """
    Pretty-print metrics dictionary.

    Args:
        metrics: Metrics dictionary from calculate_metrics()
        title: Title for the metrics report
        precision: Number of decimal places

    Example:
        >>> from scripts.utils import calculate_metrics, print_metrics
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print_metrics(metrics, title="Validation Metrics")

        ╔════════════════════════════════════════════════╗
        ║         Validation Metrics                     ║
        ╚════════════════════════════════════════════════╝
        Accuracy:              0.8571
        Precision (macro):     0.8333
        Precision (weighted):  0.8452
        Recall (macro):        0.8000
        Recall (weighted):     0.8571
        F1-Score (macro):      0.8095
        F1-Score (weighted):   0.8452
        Cohen's Kappa:         0.8095
    """
    width = 56
    print("╔" + "═" * (width - 2) + "╗")
    print("║" + f" {title}".ljust(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")

    # Main metrics
    print(f"Accuracy:              {metrics['accuracy']:.{precision}f}")
    print(f"Precision (macro):     {metrics['precision_macro']:.{precision}f}")
    print(f"Precision (weighted):  {metrics['precision_weighted']:.{precision}f}")
    print(f"Recall (macro):        {metrics['recall_macro']:.{precision}f}")
    print(f"Recall (weighted):     {metrics['recall_weighted']:.{precision}f}")
    print(f"F1-Score (macro):      {metrics['f1_macro']:.{precision}f}")
    print(f"F1-Score (weighted):   {metrics['f1_weighted']:.{precision}f}")
    print(f"Cohen's Kappa:         {metrics['cohen_kappa']:.{precision}f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

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
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: Class names for axis labels (default: DR classes)
        save_path: Path to save figure (optional)
        normalize: If True, normalize by row (show proportions)
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap ('Blues', 'Greens', 'Reds', etc.)
        show: If True, display plot (set False when saving only)

    Example:
        >>> from scripts.utils import plot_confusion_matrix
        >>> import numpy as np
        >>>
        >>> y_true = np.array([0, 1, 2, 3, 4] * 100)
        >>> y_pred = np.array([0, 1, 1, 3, 4] * 100)
        >>>
        >>> plot_confusion_matrix(
        ...     y_true, y_pred,
        ...     classes=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
        ...     save_path='confusion_matrix.png',
        ...     normalize=True
        ... )
    """
    # Default classes for DR
    if classes is None:
        classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    else:
        cm_display = cm

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        vmin=0,
        vmax=1 if normalize else None,
        ax=ax,
        square=True
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")

    # Show plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix_from_metrics(
    metrics: Dict[str, Any],
    classes: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Plot confusion matrix from metrics dictionary.

    Convenience wrapper for plot_confusion_matrix() that extracts
    confusion matrix from calculate_metrics() output.

    Args:
        metrics: Metrics dictionary from calculate_metrics()
        classes: Class names
        save_path: Path to save figure
        **kwargs: Additional arguments passed to plot_confusion_matrix()

    Example:
        >>> from scripts.utils import calculate_metrics, plot_confusion_matrix_from_metrics
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> plot_confusion_matrix_from_metrics(metrics, save_path='cm.png')
    """
    cm = np.array(metrics['confusion_matrix'])

    # Create dummy arrays for plotting (already have CM)
    # Reconstruct y_true and y_pred from confusion matrix
    n_classes = cm.shape[0]
    y_true, y_pred = [], []

    for i in range(n_classes):
        for j in range(n_classes):
            count = int(cm[i, j])
            y_true.extend([i] * count)
            y_pred.extend([j] * count)

    plot_confusion_matrix(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        classes=classes,
        save_path=save_path,
        **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HISTORY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_training_history(history: Dict[str, List], filepath: Union[str, Path]) -> None:
    """
    Save training history to JSON file.

    Args:
        history: Dictionary with lists of metrics per epoch
                 Example: {'train_loss': [...], 'val_loss': [...], ...}
        filepath: Path to save JSON file

    Example:
        >>> from scripts.utils import save_training_history
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'train_acc': [80, 85, 90],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ...     'val_acc': [75, 80, 85]
        ... }
        >>> save_training_history(history, 'logs/training_history.json')
        ✓ Training history saved: logs/training_history.json
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✓ Training history saved: {filepath}")


def load_training_history(filepath: Union[str, Path]) -> Dict[str, List]:
    """
    Load training history from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with training history

    Example:
        >>> from scripts.utils import load_training_history
        >>> history = load_training_history('logs/training_history.json')
        >>> print(f"Trained for {len(history['train_loss'])} epochs")
        ✓ Training history loaded: logs/training_history.json
        Trained for 20 epochs
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Training history not found: {filepath}")

    with open(filepath, 'r') as f:
        history = json.load(f)

    print(f"✓ Training history loaded: {filepath}")

    return history


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
        metrics: List of metric pairs to plot (default: ['loss', 'acc'])
        figsize: Figure size

    Example:
        >>> from scripts.utils import load_training_history, plot_training_history
        >>> history = load_training_history('logs/training_history.json')
        >>> plot_training_history(history, save_path='training_curves.png')
    """
    if metrics is None:
        metrics = ['loss', 'acc']

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history.get(f'train_{metrics[0]}', [])) + 1)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            ax.plot(epochs, history[train_key], label=f'Train {metric.capitalize()}',
                   marker='o', markersize=4, linewidth=2)

        if val_key in history:
            ax.plot(epochs, history[val_key], label=f'Val {metric.capitalize()}',
                   marker='s', markersize=4, linewidth=2)

            # Mark best validation
            best_val = min(history[val_key]) if 'loss' in metric else max(history[val_key])
            best_epoch = history[val_key].index(best_val) + 1
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                      label=f'Best (epoch {best_epoch})')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Curves', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved: {save_path}")

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_device(device_id: Optional[int] = None, verbose: bool = True) -> torch.device:
    """
    Get PyTorch device (auto-detect or specific GPU).

    Args:
        device_id: Specific GPU ID (0, 1, etc.) or None for auto-detection
        verbose: If True, print device information

    Returns:
        torch.device

    Example:
        >>> from scripts.utils import get_device
        >>> device = get_device()
        ✓ Using device: cuda:0 (NVIDIA GeForce RTX 3090)

        >>> # Force CPU
        >>> device = get_device(device_id=-1)
        ✓ Using device: cpu

        >>> # Specific GPU
        >>> device = get_device(device_id=1)
        ✓ Using device: cuda:1
    """
    # Force CPU if device_id == -1
    if device_id == -1:
        device = torch.device('cpu')

    # Specific GPU
    elif device_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            print("⚠ CUDA not available, falling back to CPU")
            device = torch.device('cpu')

    # Auto-detect
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    if verbose:
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            print(f"✓ Using device: {device} ({gpu_name})")
        else:
            print(f"✓ Using device: {device}")

    return device


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Recursively move tensors/models to device.

    Handles tensors, models, dicts, lists, and tuples.

    Args:
        obj: Object to move (tensor, model, dict, list, tuple)
        device: Target device

    Returns:
        Object on target device

    Example:
        >>> from scripts.utils import move_to_device, get_device
        >>> device = get_device()
        >>>
        >>> # Move tensor
        >>> tensor = torch.randn(3, 224, 224)
        >>> tensor = move_to_device(tensor, device)
        >>>
        >>> # Move dict of tensors
        >>> data = {'images': torch.randn(8, 3, 224, 224), 'labels': torch.randint(0, 5, (8,))}
        >>> data = move_to_device(data, device)
    """
    if isinstance(obj, (torch.Tensor, nn.Module)):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS & LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def create_progress_bar(
    iterable: Iterable,
    desc: str = '',
    total: Optional[int] = None,
    leave: bool = True,
    **kwargs
) -> tqdm:
    """
    Create consistent progress bar with tqdm.

    Args:
        iterable: Iterable to wrap
        desc: Description text
        total: Total iterations (auto-detect if None)
        leave: Keep progress bar after completion
        **kwargs: Additional tqdm arguments

    Returns:
        tqdm progress bar

    Example:
        >>> from scripts.utils import create_progress_bar
        >>> for batch in create_progress_bar(train_loader, desc='Training'):
        ...     # Training code here
        ...     pass
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        ncols=100,
        **kwargs
    )


def log_metrics(
    metrics: Dict[str, float],
    prefix: str = '',
    logger: Optional[Any] = None
) -> None:
    """
    Log metrics to console or logger.

    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix for metric names (e.g., 'train/', 'val/')
        logger: Optional logger object (e.g., tensorboard)

    Example:
        >>> from scripts.utils import log_metrics
        >>> metrics = {'loss': 0.45, 'acc': 85.3, 'f1': 0.82}
        >>> log_metrics(metrics, prefix='val/')
        val/loss: 0.4500
        val/acc:  85.30
        val/f1:   0.8200
    """
    # Console logging
    max_name_len = max(len(prefix + k) for k in metrics.keys())

    for name, value in metrics.items():
        full_name = prefix + name
        if isinstance(value, float):
            print(f"{full_name:<{max_name_len+2}}: {value:>7.4f}")
        else:
            print(f"{full_name:<{max_name_len+2}}: {value}")

    # Logger (e.g., TensorBoard)
    if logger is not None:
        for name, value in metrics.items():
            logger.add_scalar(prefix + name, value)


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHTS & BIASES (WANDB) INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def wandb_available() -> bool:
    """
    Check if Weights & Biases is available.

    Returns:
        bool: True if wandb can be imported, False otherwise

    Example:
        >>> from scripts.utils import wandb_available
        >>> if wandb_available():
        ...     print("wandb is available")
    """
    try:
        import wandb
        return True
    except ImportError:
        return False


def init_wandb(
    config: Dict[str, Any],
    project_name: str = 'diabetic-retinopathy',
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    enable_wandb: bool = True,
    **kwargs
) -> bool:
    """
    Initialize Weights & Biases run with graceful fallback.

    Args:
        config: Configuration dictionary to log
        project_name: W&B project name (default: 'diabetic-retinopathy')
        run_name: Optional run name (auto-generated if None)
        tags: List of tags for the run
        enable_wandb: Whether to enable W&B (if False, returns False immediately)
        **kwargs: Additional arguments passed to wandb.init()

    Returns:
        bool: True if wandb initialized successfully, False otherwise

    Example:
        >>> from scripts.utils import init_wandb
        >>> config = {'lr': 0.001, 'batch_size': 32}
        >>> wandb_enabled = init_wandb(config, project_name='my-project')
        >>> if wandb_enabled:
        ...     print("W&B initialized successfully")
    """
    if not enable_wandb:
        return False

    try:
        import wandb

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=tags or [],
            **kwargs
        )

        print(f"✓ W&B initialized: {wandb.run.name} (Project: {project_name})")
        print(f"  View at: {wandb.run.url}")
        return True

    except ImportError:
        print("[INFO] wandb not available. Install with: pip install wandb")
        print("[INFO] Continuing without W&B logging...")
        return False

    except Exception as e:
        print(f"[WARNING] Failed to initialize wandb: {e}")
        print("[INFO] Continuing without W&B logging...")
        return False


def log_metrics_wandb(
    metrics: Dict[str, Union[float, int]],
    step: Optional[int] = None,
    prefix: str = ''
) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics: Dictionary of metric name -> value
        step: Training step/epoch number
        prefix: Optional prefix for metric names (e.g., 'train/')

    Example:
        >>> from scripts.utils import log_metrics_wandb
        >>> metrics = {'loss': 0.5, 'accuracy': 0.85}
        >>> log_metrics_wandb(metrics, step=10, prefix='train/')
    """
    if not wandb_available():
        return

    try:
        import wandb

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log metrics
        wandb.log(metrics, step=step)

    except Exception as e:
        # Silently ignore wandb errors to not interrupt training
        pass


def log_images_wandb(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    step: Optional[int] = None,
    max_images: int = 8,
    denormalize: bool = True
) -> None:
    """
    Log sample predictions with images to Weights & Biases.

    Args:
        images: Tensor of images [B, C, H, W]
        labels: Tensor of ground truth labels [B]
        predictions: Tensor of predicted labels [B]
        class_names: List of class names
        step: Training step/epoch number
        max_images: Maximum number of images to log (default: 8)
        denormalize: Whether to denormalize images (default: True)

    Example:
        >>> from scripts.utils import log_images_wandb
        >>> log_images_wandb(
        ...     images=sample_images,
        ...     labels=sample_labels,
        ...     predictions=sample_preds,
        ...     class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
        ...     step=10
        ... )
    """
    if not wandb_available():
        return

    try:
        import wandb

        # Limit number of images
        images = images[:max_images]
        labels = labels[:max_images]
        predictions = predictions[:max_images]

        # Denormalize images if needed (ImageNet stats)
        if denormalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            images = images * std + mean
            images = torch.clamp(images, 0, 1)

        # Convert to numpy and transpose to [B, H, W, C]
        images_np = images.cpu().numpy()
        images_np = np.transpose(images_np, (0, 2, 3, 1))

        # Create wandb images with captions
        wandb_images = []
        for idx in range(len(images_np)):
            img = images_np[idx]
            true_label = class_names[labels[idx].item()]
            pred_label = class_names[predictions[idx].item()]

            # Mark correct/incorrect predictions
            if labels[idx] == predictions[idx]:
                caption = f"✓ True: {true_label} | Pred: {pred_label}"
            else:
                caption = f"✗ True: {true_label} | Pred: {pred_label}"

            wandb_images.append(
                wandb.Image(img, caption=caption)
            )

        # Log images
        wandb.log({"predictions": wandb_images}, step=step)

    except Exception as e:
        # Silently ignore wandb errors
        pass


def log_confusion_matrix_wandb(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    step: Optional[int] = None,
    normalize: bool = True
) -> None:
    """
    Log confusion matrix as image to Weights & Biases.

    Args:
        y_true: Array of ground truth labels
        y_pred: Array of predicted labels
        class_names: List of class names
        step: Training step/epoch number
        normalize: Whether to normalize confusion matrix (default: True)

    Example:
        >>> from scripts.utils import log_confusion_matrix_wandb
        >>> log_confusion_matrix_wandb(
        ...     y_true=np.array([0, 1, 2, 3, 4]),
        ...     y_pred=np.array([0, 1, 1, 3, 4]),
        ...     class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
        ...     step=10
        ... )
    """
    if not wandb_available():
        return

    try:
        import wandb
        from io import BytesIO

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize if requested
        if normalize:
            cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        else:
            cm_display = cm

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm_display,
            annot=True,
            fmt='.2%' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'},
            ax=ax
        )

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()

        # Convert to image and log
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        wandb.log({
            "confusion_matrix": wandb.Image(buf),
        }, step=step)

        plt.close(fig)

    except Exception as e:
        # Silently ignore wandb errors
        pass


def log_gradients_wandb(
    model: nn.Module,
    step: Optional[int] = None
) -> None:
    """
    Log model gradient statistics to Weights & Biases.

    Args:
        model: PyTorch model
        step: Training step/epoch number

    Example:
        >>> from scripts.utils import log_gradients_wandb
        >>> log_gradients_wandb(model, step=10)
    """
    if not wandb_available():
        return

    try:
        import wandb

        grad_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[f"gradients/{name}_mean"] = param.grad.mean().item()
                grad_stats[f"gradients/{name}_std"] = param.grad.std().item()
                grad_stats[f"gradients/{name}_max"] = param.grad.max().item()
                grad_stats[f"gradients/{name}_min"] = param.grad.min().item()

        if grad_stats:
            wandb.log(grad_stats, step=step)

    except Exception as e:
        # Silently ignore wandb errors
        pass


def log_model_artifact_wandb(
    model_path: Union[str, Path],
    artifact_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    artifact_type: str = 'model'
) -> None:
    """
    Save model as Weights & Biases artifact.

    Args:
        model_path: Path to model checkpoint file
        artifact_name: Name for the artifact
        metadata: Optional metadata dictionary
        artifact_type: Type of artifact (default: 'model')

    Example:
        >>> from scripts.utils import log_model_artifact_wandb
        >>> log_model_artifact_wandb(
        ...     model_path='checkpoints/best_model.pth',
        ...     artifact_name='best_model_epoch_10',
        ...     metadata={'epoch': 10, 'val_acc': 85.3}
        ... )
    """
    if not wandb_available():
        return

    try:
        import wandb

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata or {}
        )

        # Add file to artifact
        artifact.add_file(str(model_path))

        # Log artifact
        wandb.log_artifact(artifact)

        print(f"  ✓ Model saved as W&B artifact: {artifact_name}")

    except Exception as e:
        print(f"[WARNING] Failed to log model artifact: {e}")


def finish_wandb() -> None:
    """
    Finish Weights & Biases run gracefully.

    Example:
        >>> from scripts.utils import finish_wandb
        >>> finish_wandb()
    """
    if not wandb_available():
        return

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            print("✓ W&B run finished")

    except Exception as e:
        # Silently ignore wandb errors
        pass


if __name__ == "__main__":
    """
    Demo and testing of utility functions.
    """
    print("=" * 70)
    print("DR Classification Utility Functions - Demo")
    print("=" * 70)

    # Test set_seed
    print("\n[1] Setting random seed...")
    set_seed(42, deterministic=True)

    # Test get_device
    print("\n[2] Detecting device...")
    device = get_device()

    # Test get_transforms
    print("\n[3] Creating transforms...")
    train_transform = get_transforms(224, is_train=True, augmentation_level='medium')
    val_transform = get_transforms(224, is_train=False)
    print("✓ Train and validation transforms created")

    # Test calculate_metrics
    print("\n[4] Calculating metrics...")
    y_true = np.array([0, 1, 2, 3, 4, 0, 1] * 10)
    y_pred = np.array([0, 1, 1, 3, 4, 0, 1] * 10)
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, title="Demo Metrics")

    # Test plot_confusion_matrix
    print("\n[5] Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, show=False)
    print("✓ Confusion matrix plotted")

    print("\n" + "=" * 70)
    print("Demo completed! All utility functions are working.")
    print("=" * 70)

    print("\n📖 Usage examples:")
    print("""
    from scripts.utils import (
        set_seed, get_transforms, create_data_loaders,
        save_checkpoint, calculate_metrics, plot_confusion_matrix
    )

    # Set seed
    set_seed(42)

    # Get transforms
    train_tf = get_transforms(224, is_train=True)
    val_tf = get_transforms(224, is_train=False)

    # Create loaders
    train_loader, val_loader = create_data_loaders(dataset, batch_size=32)

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch=10, metrics={'val_acc': 85.3},
                   path='checkpoints/epoch_10.pth', is_best=True)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
    """)
