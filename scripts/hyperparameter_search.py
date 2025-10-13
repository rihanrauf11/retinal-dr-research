#!/usr/bin/env python3
"""
Hyperparameter Optimization for RETFound + LoRA using Optuna

This script performs automated hyperparameter search for diabetic retinopathy
classification using Optuna's TPE (Tree-structured Parzen Estimator) sampler.

Features:
    - Optimizes LoRA rank, alpha, learning rate, batch size, dropout
    - Early stopping to save compute time
    - Pruning of unpromising trials
    - Comprehensive logging and visualization
    - Resumable studies
    - GPU-efficient with mixed precision training

Search Space:
    - LoRA rank (r): [4, 8, 16, 32]
    - LoRA alpha: [16, 32, 64]
    - Learning rate: [1e-5, 1e-3] (log-uniform)
    - Batch size: [8, 16, 32]
    - Dropout rate: [0.1, 0.5] (uniform)

Usage:
    # Basic search (50 trials)
    python scripts/hyperparameter_search.py \\
        --checkpoint-path models/RETFound_cfp_weights.pth \\
        --data-csv data/aptos/train.csv \\
        --data-img-dir data/aptos/train_images \\
        --n-trials 50

    # Quick search with timeout
    python scripts/hyperparameter_search.py \\
        --checkpoint-path models/RETFound_cfp_weights.pth \\
        --data-csv data/aptos/train.csv \\
        --data-img-dir data/aptos/train_images \\
        --n-trials 20 \\
        --timeout-hours 6 \\
        --num-epochs 8

    # Resume existing study
    python scripts/hyperparameter_search.py \\
        --checkpoint-path models/RETFound_cfp_weights.pth \\
        --data-csv data/aptos/train.csv \\
        --data-img-dir data/aptos/train_images \\
        --resume-study results/optuna/my_study/study.pkl \\
        --n-trials 30

Author: Generated with Claude Code
"""

import os
import sys
import json
import pickle
import argparse
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# Optuna for hyperparameter optimization
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Project modules
try:
    from scripts.config import Config
    from scripts.dataset import RetinalDataset
    from scripts.retfound_lora import RETFoundLoRA
    from scripts.utils import (
        set_seed, get_device, count_parameters,
        init_wandb, log_metrics_wandb, finish_wandb
    )
except ModuleNotFoundError:
    from config import Config
    from dataset import RetinalDataset
    from retfound_lora import RETFoundLoRA
    from utils import (
        set_seed, get_device, count_parameters,
        init_wandb, log_metrics_wandb, finish_wandb
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptunaConfig:
    """
    Configuration for Optuna hyperparameter search.

    Attributes:
        study_name: Name for the Optuna study
        n_trials: Number of optimization trials to run
        timeout_hours: Maximum hours to run (None for no limit)

        checkpoint_path: Path to RETFound pretrained weights
        data_csv: Path to training CSV file
        data_img_dir: Path to training images directory
        output_dir: Directory to save optimization results

        num_epochs: Maximum epochs per trial (reduced for speed)
        early_stopping_patience: Epochs to wait before early stopping
        img_size: Input image size (224 for RETFound)
        num_workers: DataLoader workers
        seed: Random seed for reproducibility

        lora_r_choices: LoRA rank options to search
        lora_alpha_choices: LoRA alpha options to search
        lr_range: Learning rate search range (min, max)
        batch_size_choices: Batch size options to search
        dropout_range: Dropout rate search range (min, max)

        enable_wandb: Enable W&B logging for trials
        wandb_project: W&B project name
    """
    # Study settings
    study_name: str = "retfound_lora_search"
    n_trials: int = 50
    timeout_hours: Optional[float] = None

    # Base paths
    checkpoint_path: str = ""
    data_csv: str = ""
    data_img_dir: str = ""
    output_dir: str = "results/optuna"

    # Fixed training settings
    num_epochs: int = 10  # Reduced for faster trials
    early_stopping_patience: int = 3
    img_size: int = 224
    num_workers: int = 4
    seed: int = 42

    # Hyperparameter search spaces
    lora_r_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    lora_alpha_choices: List[int] = field(default_factory=lambda: [16, 32, 64])
    lr_range: Tuple[float, float] = (1e-5, 1e-3)
    batch_size_choices: List[int] = field(default_factory=lambda: [8, 16, 32])
    dropout_range: Tuple[float, float] = (0.1, 0.5)

    # Wandb settings
    enable_wandb: bool = False
    wandb_project: str = "diabetic-retinopathy-hpo"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: Path) -> None:
        """Save config to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_transforms(img_size: int, is_train: bool = True) -> A.Compose:
    """
    Get data augmentation transforms.

    Args:
        img_size: Target image size
        is_train: Whether training or validation transforms

    Returns:
        Albumentations composition
    """
    if is_train:
        transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=4,
                max_height=20,
                max_width=20,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    return transform


def create_data_loaders(
    config: OptunaConfig,
    batch_size: int,
    seed: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        config: Optuna configuration
        batch_size: Batch size for loaders
        seed: Random seed for splitting

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    full_dataset = RetinalDataset(
        csv_file=config.data_csv,
        img_dir=config.data_img_dir,
        transform=None  # Will be set per split
    )

    # Split dataset (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Set transforms
    train_transform = get_transforms(config.img_size, is_train=True)
    val_transform = get_transforms(config.img_size, is_train=False)

    # Apply transforms (access underlying dataset)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0
    )

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: RETFoundLoRA,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: RETFoundLoRA model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scaler: Gradient scaler for mixed precision

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(
    model: RETFoundLoRA,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model.

    Args:
        model: RETFoundLoRA model
        loader: Validation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_objective(config: OptunaConfig):
    """
    Create objective function for Optuna optimization.

    This factory function captures the config and returns the objective function
    that Optuna will call for each trial.

    Args:
        config: Optuna configuration

    Returns:
        Objective function for Optuna
    """

    def objective(trial: Trial) -> float:
        """
        Objective function to maximize validation accuracy.

        Args:
            trial: Optuna trial object

        Returns:
            Best validation accuracy achieved
        """
        try:
            # ═══════════════════════════════════════════════════════════════
            # 1. SUGGEST HYPERPARAMETERS
            # ═══════════════════════════════════════════════════════════════

            lora_r = trial.suggest_categorical('lora_r', config.lora_r_choices)
            lora_alpha = trial.suggest_categorical('lora_alpha', config.lora_alpha_choices)
            learning_rate = trial.suggest_float('learning_rate', *config.lr_range, log=True)
            batch_size = trial.suggest_categorical('batch_size', config.batch_size_choices)
            dropout = trial.suggest_float('dropout', *config.dropout_range)

            # Log hyperparameters
            print(f"\n[Trial {trial.number}] Hyperparameters:")
            print(f"  LoRA rank: {lora_r}")
            print(f"  LoRA alpha: {lora_alpha}")
            print(f"  Learning rate: {learning_rate:.6f}")
            print(f"  Batch size: {batch_size}")
            print(f"  Dropout: {dropout:.3f}")

            # ═══════════════════════════════════════════════════════════════
            # 2. SETUP
            # ═══════════════════════════════════════════════════════════════

            # Set seed for reproducibility
            set_seed(config.seed + trial.number)

            # Get device
            device = get_device()

            # Create data loaders with suggested batch_size
            train_loader, val_loader = create_data_loaders(
                config, batch_size, config.seed + trial.number
            )

            # Initialize model with suggested hyperparameters
            model = RETFoundLoRA(
                checkpoint_path=config.checkpoint_path,
                num_classes=5,  # DR has 5 classes
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,  # Fixed
                head_dropout=dropout,  # Suggested
                device=device
            )

            # Count parameters
            total_params, trainable_params = count_parameters(model)
            trainable_pct = 100.0 * trainable_params / total_params
            print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")

            # Setup optimizer with suggested learning rate
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )

            # Loss function
            criterion = nn.CrossEntropyLoss()

            # Mixed precision training
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

            # Initialize wandb for this trial if enabled
            wandb_enabled = False
            if config.enable_wandb:
                trial_name = f"trial_{trial.number:03d}"
                wandb_enabled = init_wandb(
                    config={
                        'lora_r': lora_r,
                        'lora_alpha': lora_alpha,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'dropout': dropout,
                        'trial_number': trial.number
                    },
                    project_name=config.wandb_project,
                    run_name=trial_name,
                    tags=['optuna', 'hyperparameter_search'],
                    enable_wandb=True
                )

            # ═══════════════════════════════════════════════════════════════
            # 3. TRAINING LOOP
            # ═══════════════════════════════════════════════════════════════

            best_val_acc = 0.0
            best_val_loss = float('inf')
            epochs_no_improve = 0

            start_time = time.time()

            for epoch in range(config.num_epochs):
                epoch_start = time.time()

                # Train
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, scaler
                )

                # Validate
                val_loss, val_acc = validate(
                    model, val_loader, criterion, device
                )

                epoch_time = time.time() - epoch_start

                # Log to console
                print(f"  Epoch {epoch+1}/{config.num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
                      f"({epoch_time:.1f}s)")

                # Log to wandb
                if wandb_enabled:
                    log_metrics_wandb({
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'epoch_time': epoch_time
                    }, step=epoch)

                # Report intermediate value to Optuna (for pruning)
                trial.report(val_acc, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    print(f"  Trial {trial.number} pruned at epoch {epoch+1}")
                    if wandb_enabled:
                        finish_wandb()
                    raise optuna.TrialPruned()

                # Track best validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve >= config.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(no improvement for {config.early_stopping_patience} epochs)")
                    break

            total_time = time.time() - start_time

            # Final summary
            print(f"  Trial {trial.number} completed:")
            print(f"    Best Val Acc: {best_val_acc:.2f}%")
            print(f"    Best Val Loss: {best_val_loss:.4f}")
            print(f"    Total Time: {total_time:.1f}s")

            # Log final metrics to wandb
            if wandb_enabled:
                log_metrics_wandb({
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                    'total_time': total_time
                })
                finish_wandb()

            # Clean up
            del model
            del optimizer
            del train_loader
            del val_loader
            torch.cuda.empty_cache()

            return best_val_acc

        except torch.cuda.OutOfMemoryError:
            print(f"  [ERROR] Trial {trial.number} failed: CUDA out of memory")
            print(f"  Tried batch_size={batch_size}, lora_r={lora_r}")
            torch.cuda.empty_cache()
            return 0.0  # Return worst score for failed trials

        except Exception as e:
            print(f"  [ERROR] Trial {trial.number} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            raise  # Re-raise to let Optuna handle it

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_trial_to_csv(trial: Trial, study_dir: Path) -> None:
    """
    Save trial results to CSV file.

    Args:
        trial: Completed Optuna trial
        study_dir: Study output directory
    """
    csv_path = study_dir / 'trials.csv'

    # Extract trial data
    trial_data = {
        'trial_number': trial.number,
        'lora_r': trial.params.get('lora_r', None),
        'lora_alpha': trial.params.get('lora_alpha', None),
        'learning_rate': trial.params.get('learning_rate', None),
        'batch_size': trial.params.get('batch_size', None),
        'dropout': trial.params.get('dropout', None),
        'best_val_acc': trial.value if trial.value is not None else 0.0,
        'state': trial.state.name,
        'duration_seconds': trial.duration.total_seconds() if trial.duration else 0.0,
        'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else '',
        'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else ''
    }

    # Append to CSV
    df = pd.DataFrame([trial_data])

    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)


def save_best_params(study: optuna.Study, study_dir: Path) -> None:
    """
    Save best hyperparameters to JSON.

    Args:
        study: Optuna study
        study_dir: Study output directory
    """
    best_params = study.best_params.copy()
    best_params['best_value'] = study.best_value
    best_params['best_trial'] = study.best_trial.number

    json_path = study_dir / 'best_params.json'
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"\n[INFO] Best parameters saved to: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_optimization_history(study: optuna.Study, save_path: Path) -> None:
    """
    Plot optimization history showing objective value over trials.

    Args:
        study: Optuna study
        save_path: Path to save plot
    """
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved optimization history plot: {save_path}")


def plot_param_importance(study: optuna.Study, save_path: Path) -> None:
    """
    Plot hyperparameter importance.

    Args:
        study: Optuna study
        save_path: Path to save plot
    """
    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved parameter importance plot: {save_path}")
    except Exception as e:
        print(f"[WARNING] Could not create param importance plot: {e}")


def plot_parallel_coordinate(study: optuna.Study, save_path: Path) -> None:
    """
    Plot parallel coordinate showing relationship between params and objective.

    Args:
        study: Optuna study
        save_path: Path to save plot
    """
    try:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved parallel coordinate plot: {save_path}")
    except Exception as e:
        print(f"[WARNING] Could not create parallel coordinate plot: {e}")


def plot_slice(study: optuna.Study, save_path: Path) -> None:
    """
    Plot slice plots showing how each param affects objective.

    Args:
        study: Optuna study
        save_path: Path to save plot
    """
    try:
        fig = optuna.visualization.matplotlib.plot_slice(study)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved slice plot: {save_path}")
    except Exception as e:
        print(f"[WARNING] Could not create slice plot: {e}")


def generate_summary_report(study: optuna.Study, config: OptunaConfig, study_dir: Path) -> None:
    """
    Generate markdown summary report.

    Args:
        study: Optuna study
        config: Optuna configuration
        study_dir: Study output directory
    """
    report_path = study_dir / 'summary_report.md'

    # Get statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    best_trial = study.best_trial
    best_params = study.best_params
    best_value = study.best_value

    # Top 10 trials
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]

    # Calculate statistics of top 10
    if len(top_trials) >= 2:
        top_values = [t.value for t in top_trials]
        mean_top = np.mean(top_values)
        std_top = np.std(top_values)
    else:
        mean_top = top_values[0] if top_trials else 0
        std_top = 0

    # Write report
    with open(report_path, 'w') as f:
        f.write("# Hyperparameter Optimization Summary\n\n")
        f.write(f"**Study Name:** {study.study_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Study Statistics\n\n")
        f.write(f"- **Total Trials:** {len(study.trials)}\n")
        f.write(f"- **Completed:** {len(completed_trials)}\n")
        f.write(f"- **Pruned:** {len(pruned_trials)}\n")
        f.write(f"- **Failed:** {len(failed_trials)}\n\n")

        f.write("## Best Trial\n\n")
        f.write(f"- **Trial Number:** {best_trial.number}\n")
        f.write(f"- **Validation Accuracy:** {best_value:.2f}%\n")
        f.write(f"- **Duration:** {best_trial.duration.total_seconds():.1f}s\n\n")

        f.write("### Best Hyperparameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(best_params, indent=2))
        f.write("\n```\n\n")

        f.write("## Top 10 Trials\n\n")
        f.write("| Trial | Val Acc | LoRA r | LoRA α | LR | Batch | Dropout |\n")
        f.write("|-------|---------|--------|--------|----|----|-------|\n")
        for t in top_trials:
            f.write(f"| {t.number} | {t.value:.2f}% | {t.params['lora_r']} | "
                    f"{t.params['lora_alpha']} | {t.params['learning_rate']:.2e} | "
                    f"{t.params['batch_size']} | {t.params['dropout']:.3f} |\n")

        f.write(f"\n**Mean of Top 10:** {mean_top:.2f}% ± {std_top:.2f}%\n\n")

        f.write("## Search Space\n\n")
        f.write(f"- **LoRA Rank:** {config.lora_r_choices}\n")
        f.write(f"- **LoRA Alpha:** {config.lora_alpha_choices}\n")
        f.write(f"- **Learning Rate:** [{config.lr_range[0]:.2e}, {config.lr_range[1]:.2e}]\n")
        f.write(f"- **Batch Size:** {config.batch_size_choices}\n")
        f.write(f"- **Dropout:** [{config.dropout_range[0]}, {config.dropout_range[1]}]\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Epochs per Trial:** {config.num_epochs}\n")
        f.write(f"- **Early Stopping Patience:** {config.early_stopping_patience}\n")
        f.write(f"- **Random Seed:** {config.seed}\n\n")

    print(f"[INFO] Saved summary report: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_hyperparameter_search(config: OptunaConfig) -> optuna.Study:
    """
    Run hyperparameter optimization study.

    Args:
        config: Optuna configuration

    Returns:
        Completed Optuna study
    """
    # Create output directory
    study_dir = Path(config.output_dir) / config.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = study_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Save configuration
    config.save(study_dir / 'config.json')

    # Set random seed
    set_seed(config.seed)

    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION FOR RETFOUND + LORA")
    print("=" * 80)
    print(f"Study Name: {config.study_name}")
    print(f"Number of Trials: {config.n_trials}")
    print(f"Output Directory: {study_dir}")
    print(f"Checkpoint Path: {config.checkpoint_path}")
    print(f"Data CSV: {config.data_csv}")
    print(f"Data Images: {config.data_img_dir}")
    print("=" * 80)

    # Create or load study
    study_path = study_dir / 'study.pkl'

    if study_path.exists():
        print(f"\n[INFO] Loading existing study from {study_path}")
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f"[INFO] Resuming study with {len(study.trials)} existing trials")
    else:
        print(f"\n[INFO] Creating new study")
        study = optuna.create_study(
            study_name=config.study_name,
            direction="maximize",  # Maximize validation accuracy
            sampler=TPESampler(seed=config.seed),
            pruner=MedianPruner(
                n_startup_trials=5,  # Don't prune first 5 trials
                n_warmup_steps=3     # Prune after 3 epochs
            )
        )

    # Create objective function
    objective = create_objective(config)

    # Create callback to save after each trial
    def save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Save study and trial results after each trial."""
        # Save study object
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        # Save trial to CSV
        save_trial_to_csv(trial, study_dir)

        # Save best parameters
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            save_best_params(study, study_dir)

    # Run optimization
    print(f"\n[INFO] Starting optimization...")
    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout_hours * 3600 if config.timeout_hours else None,
            callbacks=[save_callback],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Optimization interrupted by user")

    total_time = time.time() - start_time

    # Save final results
    print(f"\n[INFO] Optimization completed in {total_time/3600:.2f} hours")

    # Save study
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    # Generate visualizations
    print("\n[INFO] Generating visualizations...")
    plot_optimization_history(study, plots_dir / 'optimization_history.png')
    plot_param_importance(study, plots_dir / 'param_importance.png')
    plot_parallel_coordinate(study, plots_dir / 'parallel_coordinate.png')
    plot_slice(study, plots_dir / 'slice_plots.png')

    # Generate summary report
    generate_summary_report(study, config, study_dir)

    # Print final summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Validation Accuracy: {study.best_value:.2f}%")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        if key == 'learning_rate':
            print(f"  {key}: {value:.2e}")
        elif key == 'dropout':
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nResults saved to: {study_dir}")
    print("=" * 80)

    return study


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for RETFound + LoRA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='Path to RETFound pretrained weights'
    )
    parser.add_argument(
        '--data-csv',
        type=str,
        required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--data-img-dir',
        type=str,
        required=True,
        help='Path to training images directory'
    )

    # Study settings
    parser.add_argument(
        '--study-name',
        type=str,
        default='retfound_lora_search',
        help='Name for the Optuna study'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--timeout-hours',
        type=float,
        default=None,
        help='Maximum hours to run (None for no limit)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/optuna',
        help='Output directory for results'
    )

    # Training settings
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Maximum epochs per trial'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=3,
        help='Epochs to wait before early stopping'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Wandb settings
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable W&B logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='diabetic-retinopathy-hpo',
        help='W&B project name'
    )

    # Resume study
    parser.add_argument(
        '--resume-study',
        type=str,
        default=None,
        help='Path to existing study.pkl to resume'
    )

    args = parser.parse_args()

    # Create configuration
    config = OptunaConfig(
        study_name=args.study_name,
        n_trials=args.n_trials,
        timeout_hours=args.timeout_hours,
        checkpoint_path=args.checkpoint_path,
        data_csv=args.data_csv,
        data_img_dir=args.data_img_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project
    )

    # Handle resume
    if args.resume_study:
        # Load existing config from resumed study
        resume_dir = Path(args.resume_study).parent
        config_path = resume_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            # Update only trial count and timeout
            config.n_trials = args.n_trials
            config.timeout_hours = args.timeout_hours
            print(f"[INFO] Resuming study from {args.resume_study}")

    # Run optimization
    study = run_hyperparameter_search(config)

    return study


if __name__ == '__main__':
    main()
