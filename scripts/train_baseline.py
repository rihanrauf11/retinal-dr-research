#!/usr/bin/env python3
"""
Baseline Training Script for Diabetic Retinopathy Classification

This script provides a complete training pipeline for diabetic retinopathy
classification using the DRClassifier model with proper data loading,
augmentation, training loop, validation, checkpointing, and logging.

Usage:
    python scripts/train_baseline.py --config configs/default_config.yaml
    python scripts/train_baseline.py --config configs/default_config.yaml --resume checkpoint.pth
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Data augmentation
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project modules
from scripts.config import Config
from scripts.dataset import RetinalDataset
from scripts.model import DRClassifier

# Wandb utilities
from scripts.utils import (
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

    Parameters
    ----------
    seed : int
        Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Make CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]:
    """
    Create training and validation transforms using Albumentations.

    Parameters
    ----------
    img_size : int
        Target image size

    Returns
    -------
    tuple
        (train_transform, val_transform)
    """
    # ImageNet normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with augmentation
    train_transform = A.Compose([
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
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=1.0
            ),
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

    # Validation transforms (minimal)
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
    Create training and validation data loaders with 80/20 split.

    Parameters
    ----------
    config : Config
        Configuration object
    train_transform : A.Compose
        Training augmentation pipeline
    val_transform : A.Compose
        Validation augmentation pipeline

    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    print("\n[INFO] Loading dataset...")

    # Load full dataset
    full_dataset = RetinalDataset(
        csv_file=config.data.train_csv,
        img_dir=config.data.train_img_dir,
        transform=None  # We'll apply transforms per split
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    print(f"[INFO] Total samples: {total_size}")
    print(f"[INFO] Train/Val split: {train_size}/{val_size} (80/20)")

    # Split dataset
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.system.seed)
    )

    # Create separate datasets with different transforms
    train_dataset = RetinalDataset(
        csv_file=config.data.train_csv,
        img_dir=config.data.train_img_dir,
        transform=train_transform
    )
    train_dataset.data_frame = train_dataset.data_frame.iloc[train_indices.indices].reset_index(drop=True)

    val_dataset = RetinalDataset(
        csv_file=config.data.train_csv,
        img_dir=config.data.train_img_dir,
        transform=val_transform
    )
    val_dataset.data_frame = val_dataset.data_frame.iloc[val_indices.indices].reset_index(drop=True)

    # Get class distribution
    print("\n[INFO] Class distribution:")
    severity_names = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "PDR"
    }

    train_dist = train_dataset.get_class_distribution()
    for cls, count in sorted(train_dist.items()):
        pct = 100.0 * count / train_size
        print(f"  Class {cls} ({severity_names[cls]:15s}): {count:4d} ({pct:5.1f}%)")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def save_checkpoint(
    checkpoint_dict: Dict,
    filepath: Path,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.

    Parameters
    ----------
    checkpoint_dict : dict
        Dictionary containing model state, optimizer state, and metadata
    filepath : Path
        Path to save checkpoint
    is_best : bool
        If True, also save as 'best_model.pth'
    """
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint_dict, filepath)

    # Also save as best model if applicable
    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        torch.save(checkpoint_dict, best_path)


def save_training_history(history: Dict, filepath: Path) -> None:
    """
    Save training history to JSON file.

    Parameters
    ----------
    history : dict
        Training history dictionary
    filepath : Path
        Path to save JSON file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : torch.device
        Device to use for training
    epoch : int
        Current epoch number

    Returns
    -------
    dict
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
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(f"\n[WARNING] Non-finite loss detected at batch {batch_idx}, skipping...")
                continue

            # Backward pass
            loss.backward()
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
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Parameters
    ----------
    model : nn.Module
        Model to validate
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to use for validation

    Returns
    -------
    dict
        Dictionary containing validation metrics
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class accuracy
    class_correct = [0] * 5
    class_total = [0] * 5

    # Progress bar
    pbar = tqdm(val_loader, desc='Validating', leave=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

    # Calculate final metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    # Calculate per-class accuracy
    class_acc = {}
    for i in range(5):
        if class_total[i] > 0:
            class_acc[f'class_{i}_acc'] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_acc[f'class_{i}_acc'] = 0.0

    return {
        'val_loss': avg_loss,
        'val_acc': accuracy,
        **class_acc
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    config: Config,
    resume_checkpoint: Optional[str] = None,
    enable_wandb: bool = False,
    wandb_project: str = 'diabetic-retinopathy',
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None
) -> Dict:
    """
    Main training function.

    Parameters
    ----------
    config : Config
        Configuration object
    resume_checkpoint : str, optional
        Path to checkpoint to resume from
    enable_wandb : bool
        Whether to enable Weights & Biases logging
    wandb_project : str
        W&B project name
    wandb_run_name : str, optional
        W&B run name
    wandb_tags : list, optional
        W&B tags for the run

    Returns
    -------
    dict
        Training history
    """
    # ─────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Diabetic Retinopathy Classification - Training")
    print("=" * 80)

    # Set seed for reproducibility
    set_seed(config.system.seed)
    print(f"\n[INFO] Random seed set to: {config.system.seed}")

    # Setup device
    device = torch.device(config.system.device)
    print(f"[INFO] Using device: {device}")

    # Create output directories
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    log_dir = Path(config.paths.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Weights & Biases Initialization
    # ─────────────────────────────────────────────────────────────────────
    wandb_enabled = False
    if enable_wandb:
        # Generate run name if not provided
        if wandb_run_name is None:
            wandb_run_name = f"{config.model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set default tags
        if wandb_tags is None:
            wandb_tags = ['baseline', config.model.model_name]

        # Initialize wandb
        wandb_enabled = init_wandb(
            config=config.to_dict(),
            project_name=wandb_project,
            run_name=wandb_run_name,
            tags=wandb_tags,
            enable_wandb=enable_wandb
        )

    # ─────────────────────────────────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────────────────────────────────
    train_transform, val_transform = get_transforms(config.image.img_size)
    train_loader, val_loader = create_data_loaders(
        config,
        train_transform,
        val_transform
    )

    # ─────────────────────────────────────────────────────────────────────
    # Model Setup
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Creating model: {config.model.model_name}")
    model = DRClassifier.from_config(config.model)
    model = model.to(device)

    total_params, trainable_params = model.get_num_params()
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    # ─────────────────────────────────────────────────────────────────────
    # Training Setup
    # ─────────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'epoch_time': []
    }

    start_epoch = 1
    best_acc = 0.0

    # Resume from checkpoint if provided
    if resume_checkpoint:
        print(f"\n[INFO] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
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
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Validate
            val_metrics = validate(
                model, val_loader, criterion, device
            )

            # Update learning rate
            scheduler.step(val_metrics['val_acc'])

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_time)

            # Print epoch summary
            print(f"\nResults:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}  "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}  "
                  f"Val Acc:   {val_metrics['val_acc']:.2f}%", end='')

            # Check if best model
            is_best = val_metrics['val_acc'] > best_acc
            if is_best:
                best_acc = val_metrics['val_acc']
                print(" ✓ (Best!)")
            else:
                print()

            print(f"  Epoch Time: {epoch_time:.1f}s")

            # ─────────────────────────────────────────────────────────────────
            # Log to Weights & Biases
            # ─────────────────────────────────────────────────────────────────
            if wandb_enabled:
                # Log metrics
                log_metrics_wandb({
                    'train_loss': train_metrics['train_loss'],
                    'train_acc': train_metrics['train_acc'],
                    'val_loss': val_metrics['val_loss'],
                    'val_acc': val_metrics['val_acc'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                    **{k: v for k, v in val_metrics.items() if k.startswith('class_')}
                }, step=epoch)

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

            # Save checkpoint
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['train_loss'],
                'train_acc': train_metrics['train_acc'],
                'val_loss': val_metrics['val_loss'],
                'val_acc': val_metrics['val_acc'],
                'best_acc': best_acc,
                'history': history,
                'config': config.to_dict(),
                'model_name': config.model.model_name,
                'num_classes': config.model.num_classes
            }

            # Save regular checkpoint
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(checkpoint_dict, checkpoint_path, is_best=is_best)

            if is_best:
                print(f"  ✓ Saved best model: {checkpoint_dir / 'best_model.pth'}")

                # Log best model to wandb
                if wandb_enabled:
                    log_model_artifact_wandb(
                        model_path=checkpoint_dir / 'best_model.pth',
                        artifact_name=f'best_model_epoch_{epoch}',
                        metadata={
                            'epoch': epoch,
                            'val_acc': val_metrics['val_acc'],
                            'val_loss': val_metrics['val_loss'],
                            'train_acc': train_metrics['train_acc'],
                            'train_loss': train_metrics['train_loss'],
                            'model_name': config.model.model_name,
                            'num_classes': config.model.num_classes
                        }
                    )
            else:
                print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user")
        print("[INFO] Saving checkpoint...")

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'config': config.to_dict(),
            'interrupted': True
        }
        checkpoint_path = checkpoint_dir / 'checkpoint_interrupted.pth'
        save_checkpoint(checkpoint_dict, checkpoint_path)
        print(f"[INFO] Checkpoint saved to: {checkpoint_path}")

        return history

    except Exception as e:
        print(f"\n[ERROR] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

    # ─────────────────────────────────────────────────────────────────────
    # Post-Training
    # ─────────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Final Training Accuracy:  {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")

    print(f"\nTotal Training Time: {total_time/60:.1f} minutes")
    print(f"Average Time per Epoch: {total_time/config.training.num_epochs:.1f} seconds")

    # Save training history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_path = log_dir / f'training_history_{timestamp}.json'
    save_training_history(history, history_path)
    print(f"\nTraining history saved to: {history_path}")

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

    print("=" * 80 + "\n")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train diabetic retinopathy classification model'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
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
    try:
        config = Config.from_yaml(args.config)
        print(f"✓ Configuration loaded from: {args.config}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    # Print configuration
    if args.debug:
        print("\n" + "=" * 80)
        print("Configuration")
        print("=" * 80)
        print(config)

    # Train model
    try:
        history = train(
            config,
            resume_checkpoint=args.resume,
            enable_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
