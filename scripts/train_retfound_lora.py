#!/usr/bin/env python3
"""
RETFound + LoRA Training Script for Diabetic Retinopathy Classification

This script demonstrates parameter-efficient fine-tuning using Low-Rank Adaptation
(LoRA) on the RETFound foundation model. LoRA enables training with <1% of the
original parameters while maintaining competitive performance.

Key Advantages of LoRA Fine-Tuning:
    - Trains only ~800K parameters (vs 303M in full fine-tuning)
    - 2-3x faster training with significantly less memory
    - Storage efficient: 3 MB adapter checkpoints vs 1.2 GB full models
    - Prevents catastrophic forgetting of pretrained knowledge
    - Enables training multiple task-specific adapters

Usage:
    # Basic training
    python scripts/train_retfound_lora.py \\
        --checkpoint_path models/RETFound_cfp_weights.pth \\
        --config configs/retfound_lora_config.yaml

    # Custom LoRA settings
    python scripts/train_retfound_lora.py \\
        --checkpoint_path models/RETFound_cfp_weights.pth \\
        --lora_r 16 --lora_alpha 64 \\
        --batch_size 48 --epochs 15

    # Resume training
    python scripts/train_retfound_lora.py \\
        --checkpoint_path models/RETFound_cfp_weights.pth \\
        --resume results/retfound_lora/checkpoints/checkpoint_epoch_5.pth

Author: Generated with Claude Code
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project modules
try:
    from scripts.config import Config
    from scripts.dataset import RetinalDataset
    from scripts.retfound_lora import RETFoundLoRA
    # Wandb utilities
    from scripts.utils import (
        init_wandb, log_metrics_wandb, log_images_wandb,
        log_confusion_matrix_wandb, log_model_artifact_wandb,
        finish_wandb, wandb_available
    )
except ModuleNotFoundError:
    # Handle direct execution from scripts directory
    from config import Config
    from dataset import RetinalDataset
    from retfound_lora import RETFoundLoRA
    from utils import (
        init_wandb, log_metrics_wandb, log_images_wandb,
        log_confusion_matrix_wandb, log_model_artifact_wandb,
        finish_wandb, wandb_available
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Make CUDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]:
    """
    Create training and validation transforms using Albumentations.

    For LoRA fine-tuning, we use standard augmentations that don't distort
    the learned features too much, as the pretrained model already has strong
    representations.

    Args:
        img_size: Target image size (typically 224 for RETFound)

    Returns:
        (train_transform, val_transform)
    """
    # ImageNet normalization (standard for vision transformers)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with moderate augmentation
    # NOTE: LoRA benefits from augmentation but doesn't need as aggressive
    # augmentation as training from scratch
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),
        A.CoarseDropout(
            max_holes=4,
            max_height=16,
            max_width=16,
            p=0.3
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_data_loaders(
    config: Config,
    train_transform: A.Compose,
    val_transform: A.Compose
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        config: Configuration object
        train_transform: Training transformations
        val_transform: Validation transformations

    Returns:
        (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = RetinalDataset(
        csv_file=config.data.train_csv,
        img_dir=config.data.img_dir,
        transform=None  # We'll apply transforms separately
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(config.data.val_split * total_size)
    train_size = total_size - val_size

    # Split dataset with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(config.system.seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    # NOTE: LoRA uses less memory, so we can afford larger batch sizes
    # and more workers for faster data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.data.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.data.num_workers > 0 else False
    )

    print(f"[INFO] Dataset split: {train_size} train, {val_size} val")

    return train_loader, val_loader


def save_checkpoint(
    checkpoint_dict: Dict,
    filepath: Path,
    is_best: bool = False
) -> None:
    """
    Save training checkpoint.

    Args:
        checkpoint_dict: Dictionary containing training state
        filepath: Path to save checkpoint
        is_best: If True, also save as 'best_model.pth'
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict, filepath)

    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        torch.save(checkpoint_dict, best_path)


def save_training_history(history: Dict, filepath: Path) -> None:
    """
    Save training history to JSON file.

    Args:
        history: Training history dictionary
        filepath: Path to save JSON file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: RETFoundLoRA,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train model for one epoch with LoRA adapters.

    Key Notes for LoRA Training:
    - Only LoRA adapters and classification head are trainable
    - Base model weights remain frozen
    - Gradients only flow through ~0.26% of parameters
    - This enables faster training and prevents overfitting

    Args:
        model: RETFoundLoRA model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer (only updates trainable params)
        device: Device to use for training
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision training

    Returns:
        Dictionary containing training metrics
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass (with mixed precision if enabled)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Check for NaN/Inf
                if not torch.isfinite(loss):
                    print(f"\n[WARNING] Non-finite loss at batch {batch_idx}, skipping...")
                    continue

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping (helps with LoRA stability)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Check for NaN/Inf
                if not torch.isfinite(loss):
                    print(f"\n[WARNING] Non-finite loss at batch {batch_idx}, skipping...")
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"\n[ERROR] CUDA out of memory at batch {batch_idx}")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    # Calculate final metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {
        'train_loss': avg_loss,
        'train_acc': accuracy
    }


def validate(
    model: RETFoundLoRA,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: RETFoundLoRA model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use for validation

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class accuracy (for DR: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=PDR)
    num_classes = model.num_classes
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # Progress bar
    pbar = tqdm(val_loader, desc='Validating', leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

            # Update progress bar
            avg_loss = running_loss / (pbar.n + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

    # Calculate final metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)

    return {
        'val_loss': avg_loss,
        'val_acc': accuracy,
        'class_accuracies': class_accuracies
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    config: Config,
    checkpoint_path: str,
    lora_r: int,
    lora_alpha: int,
    resume_checkpoint: Optional[str] = None,
    use_mixed_precision: bool = True,
    enable_wandb: bool = False,
    wandb_project: str = 'diabetic-retinopathy',
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None
) -> None:
    """
    Main training function for RETFound + LoRA.

    This function demonstrates parameter-efficient fine-tuning where:
    1. RETFound backbone weights are frozen
    2. Only LoRA adapters (~800K params) and classifier are trained
    3. Results in 99.7% parameter reduction vs full fine-tuning
    4. Achieves competitive performance with 2-3x faster training

    Args:
        config: Configuration object
        checkpoint_path: Path to RETFound pretrained weights
        lora_r: LoRA rank (controls adapter capacity)
        lora_alpha: LoRA alpha scaling factor
        resume_checkpoint: Optional path to resume training
        use_mixed_precision: Use automatic mixed precision for faster training
        enable_wandb: Whether to enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)
        wandb_tags: W&B tags for the run
    """
    print("\n" + "=" * 80)
    print("RETFOUND + LORA TRAINING")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────
    set_seed(config.system.seed)

    # Device setup
    device = torch.device(config.system.device)
    print(f"\n[INFO] Using device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create output directories
    output_dir = Path(config.paths.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    adapter_dir = output_dir / 'adapters'  # For LoRA-only checkpoints
    log_dir = output_dir / 'logs'

    for directory in [output_dir, checkpoint_dir, adapter_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Weights & Biases Initialization
    # ─────────────────────────────────────────────────────────────────────
    wandb_enabled = False
    if enable_wandb:
        # Generate run name if not provided
        if wandb_run_name is None:
            wandb_run_name = f"retfound_lora_r{lora_r}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set default tags
        if wandb_tags is None:
            wandb_tags = ['retfound', 'lora', f'rank_{lora_r}']

        # Prepare config dict with LoRA-specific info
        wandb_config = config.to_dict()
        wandb_config.update({
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'use_mixed_precision': use_mixed_precision
        })

        # Initialize wandb
        wandb_enabled = init_wandb(
            config=wandb_config,
            project_name=wandb_project,
            run_name=wandb_run_name,
            tags=wandb_tags,
            enable_wandb=enable_wandb
        )

    # ─────────────────────────────────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading data...")
    train_transform, val_transform = get_transforms(config.image.input_size)
    train_loader, val_loader = create_data_loaders(
        config, train_transform, val_transform
    )

    # ─────────────────────────────────────────────────────────────────────
    # Model Setup (RETFound + LoRA)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Creating RETFound + LoRA model...")
    print(f"[INFO] RETFound checkpoint: {checkpoint_path}")
    print(f"[INFO] LoRA configuration:")
    print(f"  - Rank (r): {lora_r}")
    print(f"  - Alpha: {lora_alpha}")
    print(f"  - Target modules: attention QKV projections")

    model = RETFoundLoRA(
        checkpoint_path=checkpoint_path,
        num_classes=config.model.num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        head_dropout=0.3,
        device=device
    )

    # Display parameter efficiency
    total_params = model.get_num_params(trainable_only=False)
    trainable_params = model.get_num_params(trainable_only=True)
    trainable_pct = 100.0 * trainable_params / total_params

    print(f"\n[INFO] Parameter Efficiency:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.3f}%)")
    print(f"  Parameter reduction: {total_params / trainable_params:.1f}x")
    print(f"  Memory savings: ~99.7% compared to full fine-tuning")

    # Log LoRA-specific metrics to wandb
    if wandb_enabled:
        log_metrics_wandb({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': trainable_pct,
            'parameter_reduction_factor': total_params / trainable_params
        }, step=0)

    # ─────────────────────────────────────────────────────────────────────
    # Training Setup
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Setting up training...")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW with lower learning rate for fine-tuning
    # NOTE: We only optimize parameters with requires_grad=True
    # This automatically filters to LoRA adapters + classifier
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999)
    )

    print(f"[INFO] Optimizer: AdamW")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Weight decay: {config.training.weight_decay}")
    print(f"  - Note: Only LoRA adapters and classifier are optimized")

    # Learning rate scheduler: Cosine annealing
    # LoRA benefits from gradual learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=1e-6
    )

    # Mixed precision training (for faster training on GPUs)
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print(f"[INFO] Mixed precision training enabled")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'epoch_time': [],
        'class_accuracies': []
    }

    start_epoch = 1
    best_acc = 0.0

    # Resume from checkpoint if provided
    if resume_checkpoint:
        print(f"\n[INFO] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)

        # Load LoRA adapters
        if 'lora_adapters' in checkpoint:
            model.backbone.load_state_dict(checkpoint['lora_adapters'], strict=False)

        # Load classifier
        if 'classifier_state' in checkpoint:
            model.classifier.load_state_dict(checkpoint['classifier_state'])

        # Load optimizer
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Load training state
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        history = checkpoint.get('history', history)

        print(f"[INFO] Resumed from epoch {start_epoch-1}, best acc: {best_acc:.2f}%")

    # ─────────────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Starting training for {config.training.num_epochs} epochs...")
    print("=" * 80)

    total_start_time = time.time()

    try:
        for epoch in range(start_epoch, config.training.num_epochs + 1):
            epoch_start_time = time.time()

            # Print epoch header
            print(f"\nEpoch {epoch}/{config.training.num_epochs}")
            print("-" * 80)

            # Train one epoch
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, scaler
            )

            # Validate
            val_metrics = validate(
                model, val_loader, criterion, device
            )

            # Update learning rate
            scheduler.step()

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_time)
            history['class_accuracies'].append(val_metrics['class_accuracies'])

            # Print epoch summary
            print(f"\nResults:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}  "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}  "
                  f"Val Acc:   {val_metrics['val_acc']:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}  "
                  f"Time: {epoch_time:.2f}s")

            # Per-class accuracy
            print(f"  Per-class accuracy:")
            class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
            for i, (name, acc) in enumerate(zip(class_names[:model.num_classes],
                                                val_metrics['class_accuracies'])):
                print(f"    {name}: {acc:.2f}%")

            # ─────────────────────────────────────────────────────────────────
            # Log to Weights & Biases
            # ─────────────────────────────────────────────────────────────────
            if wandb_enabled:
                # Log metrics
                wandb_metrics = {
                    'train_loss': train_metrics['train_loss'],
                    'train_acc': train_metrics['train_acc'],
                    'val_loss': val_metrics['val_loss'],
                    'val_acc': val_metrics['val_acc'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                }

                # Add per-class accuracies
                for i, acc in enumerate(val_metrics['class_accuracies']):
                    wandb_metrics[f'class_{i}_acc'] = acc

                log_metrics_wandb(wandb_metrics, step=epoch)

                # Log sample predictions every 5 epochs
                if epoch % 5 == 0:
                    try:
                        with torch.no_grad():
                            # Get a batch of validation samples
                            sample_images, sample_labels = next(iter(val_loader))
                            sample_images = sample_images[:8].to(device)
                            sample_labels = sample_labels[:8]

                            # Get predictions
                            model.eval()
                            sample_outputs = model(sample_images)
                            sample_preds = sample_outputs.argmax(dim=1).cpu()
                            model.train()

                            # Log to wandb
                            log_images_wandb(
                                sample_images.cpu(),
                                sample_labels,
                                sample_preds,
                                class_names=['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR'],
                                step=epoch,
                                max_images=8
                            )
                    except Exception as e:
                        print(f"[WARNING] Failed to log sample predictions: {e}")

            # Save checkpoint every epoch
            checkpoint_dict = {
                'epoch': epoch,
                'lora_adapters': model.get_lora_state_dict(),
                'classifier_state': model.classifier.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lora_config': {
                    'r': lora_r,
                    'alpha': lora_alpha,
                    'checkpoint_path': checkpoint_path
                },
                'best_acc': best_acc,
                'history': history,
                'config': config.to_dict() if hasattr(config, 'to_dict') else None
            }

            checkpoint_path_save = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(checkpoint_dict, checkpoint_path_save)

            # Save LoRA adapters separately (storage efficient: ~3 MB)
            adapter_path = adapter_dir / f'lora_adapters_epoch_{epoch}.pth'
            model.save_lora_adapters(adapter_path)

            # Save best model
            if val_metrics['val_acc'] > best_acc:
                best_acc = val_metrics['val_acc']
                checkpoint_dict['best_acc'] = best_acc

                best_checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'
                save_checkpoint(checkpoint_dict, best_checkpoint_path, is_best=True)

                best_adapter_path = adapter_dir / 'lora_adapters_best.pth'
                model.save_lora_adapters(best_adapter_path)

                print(f"  ✓ New best model saved (acc: {best_acc:.2f}%)")

                # Log best model to wandb
                if wandb_enabled:
                    log_model_artifact_wandb(
                        model_path=best_checkpoint_path,
                        artifact_name=f'best_model_epoch_{epoch}',
                        metadata={
                            'epoch': epoch,
                            'val_acc': val_metrics['val_acc'],
                            'val_loss': val_metrics['val_loss'],
                            'train_acc': train_metrics['train_acc'],
                            'train_loss': train_metrics['train_loss'],
                            'lora_r': lora_r,
                            'lora_alpha': lora_alpha,
                            'trainable_params': trainable_params,
                            'trainable_pct': trainable_pct
                        }
                    )

            # Save training history
            history_path = log_dir / 'training_history.json'
            save_training_history(history, history_path)

            print()

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Training interrupted by user. Saving checkpoint...")

        # Save checkpoint on interruption
        interrupt_checkpoint = {
            'epoch': epoch,
            'lora_adapters': model.get_lora_state_dict(),
            'classifier_state': model.classifier.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'lora_config': {
                'r': lora_r,
                'alpha': lora_alpha,
                'checkpoint_path': checkpoint_path
            },
            'best_acc': best_acc,
            'history': history
        }

        interrupt_path = checkpoint_dir / 'checkpoint_interrupted.pth'
        save_checkpoint(interrupt_checkpoint, interrupt_path)

        adapter_interrupt_path = adapter_dir / 'lora_adapters_interrupted.pth'
        model.save_lora_adapters(adapter_interrupt_path)

        print(f"[INFO] Checkpoint saved to: {interrupt_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Training Complete
    # ─────────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start_time
    total_hours = total_time / 3600

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nTraining Summary:")
    print(f"  Total time: {total_hours:.2f} hours")
    print(f"  Best validation accuracy: {best_acc:.2f}%")
    print(f"  Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"LoRA adapters saved to: {adapter_dir}")
    print(f"Training history saved to: {history_path}")
    print("\nParameter Efficiency Summary:")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.3f}%)")
    print(f"  Parameter reduction: {total_params / trainable_params:.1f}x")
    print(f"  Adapter checkpoint size: ~3 MB (vs 1.2 GB full model)")

    # ─────────────────────────────────────────────────────────────────────
    # Log Final Confusion Matrix to wandb
    # ─────────────────────────────────────────────────────────────────────
    if wandb_enabled:
        try:
            print("\n[INFO] Calculating final confusion matrix for W&B...")
            all_preds = []
            all_labels = []

            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())

            log_confusion_matrix_wandb(
                y_true=np.array(all_labels),
                y_pred=np.array(all_preds),
                class_names=['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR'],
                step=config.training.num_epochs
            )
            print("[INFO] ✓ Confusion matrix logged to W&B")

        except Exception as e:
            print(f"[WARNING] Failed to log confusion matrix: {e}")

        # Finish wandb run
        finish_wandb()

    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for RETFound + LoRA training script.
    """
    parser = argparse.ArgumentParser(
        description='Train RETFound with LoRA for Diabetic Retinopathy Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with config file
  python scripts/train_retfound_lora.py \\
      --checkpoint_path models/RETFound_cfp_weights.pth \\
      --config configs/retfound_lora_config.yaml

  # Custom LoRA settings
  python scripts/train_retfound_lora.py \\
      --checkpoint_path models/RETFound_cfp_weights.pth \\
      --lora_r 16 --lora_alpha 64 \\
      --batch_size 48 --epochs 15 \\
      --lr 1e-4

  # Resume training
  python scripts/train_retfound_lora.py \\
      --checkpoint_path models/RETFound_cfp_weights.pth \\
      --resume results/retfound_lora/checkpoints/checkpoint_epoch_5.pth

For more information, see RETFOUND_GUIDE.md
        """
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to RETFound pretrained checkpoint (.pth file)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/retfound_lora_config.yaml',
        help='Path to configuration YAML file (default: configs/retfound_lora_config.yaml)'
    )

    # LoRA hyperparameters
    parser.add_argument(
        '--lora_r',
        type=int,
        default=None,
        help='LoRA rank (4, 8, 16, 32). Higher = more capacity. Default: from config or 8'
    )

    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=None,
        help='LoRA alpha scaling factor. Common: 4*rank. Default: from config or 32'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (LoRA can use larger batches). Default: from config or 32'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs. Default: from config or 10'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate for fine-tuning. Default: from config or 5e-5'
    )

    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    # Mixed precision
    parser.add_argument(
        '--no_mixed_precision',
        action='store_true',
        help='Disable mixed precision training (slower but more stable)'
    )

    # Weights & Biases
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='diabetic-retinopathy',
        help='W&B project name (default: diabetic-retinopathy)'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='W&B run name (default: auto-generated)'
    )
    parser.add_argument(
        '--wandb-tags',
        nargs='+',
        default=None,
        help='W&B tags for the run'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    # Override config with command-line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    if args.lr is not None:
        config.training.learning_rate = args.lr

    # LoRA hyperparameters (use args or config or defaults)
    lora_r = args.lora_r
    if lora_r is None:
        lora_r = getattr(config.model, 'lora_r', 8)

    lora_alpha = args.lora_alpha
    if lora_alpha is None:
        lora_alpha = getattr(config.model, 'lora_alpha', 32)

    # Validate configuration
    config.validate(create_dirs=True)

    # Print configuration summary
    print("\nTraining Configuration:")
    print(f"  Dataset: {config.data.train_csv}")
    print(f"  Output: {config.paths.output_dir}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  LoRA rank: {lora_r}")
    print(f"  LoRA alpha: {lora_alpha}")

    # Start training
    train(
        config=config,
        checkpoint_path=args.checkpoint_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        resume_checkpoint=args.resume,
        use_mixed_precision=not args.no_mixed_precision,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags
    )


if __name__ == "__main__":
    main()
