# Phase 2: Training Pipeline Integration

## Overview

**Objective**: Integrate RETFound_Green into training and configuration systems with proper transform handling for different normalization and input sizes.

**Files Modified**: `scripts/train_retfound_lora.py`, `scripts/config.py`

**Estimated Effort**: 1-2 days

**Risk Level**: **LOW** - Backward compatible with existing training code

**Validation**: Training runs without errors for both variants, transforms match model requirements

---

## File 1: scripts/config.py

### Current State
The file defines configuration dataclasses for:
- `SystemConfig` - Device, seed, logging
- `ImageConfig` - Input preprocessing (size, normalization)
- `ModelConfig` - Model architecture (name, pretrained, etc.)
- `TrainingConfig` - Learning rate, batch size, epochs, etc.

### Required Changes

#### 1. Add model_variant field to ModelConfig

Modify the `ModelConfig` dataclass:

```python
@dataclass
class ModelConfig:
    """Model configuration for training and evaluation."""

    model_name: str = "retfound_lora"
    model_variant: str = "large"  # NEW FIELD: 'large' or 'green'
    num_classes: int = 5
    pretrained: bool = True

    # LoRA-specific configurations
    lora_r: int = 8
    lora_alpha: int = 32
    head_dropout: float = 0.1

    # Checkpoint path for loading pretrained weights
    pretrained_path: str = "models/RETFound_cfp_weights.pth"

    # ... rest of fields unchanged
```

#### 2. Update validation method

Add validation in `ModelConfig.__post_init__()`:

```python
def __post_init__(self):
    """Validate configuration after initialization."""
    # Validate model_variant
    if self.model_variant not in ["large", "green"]:
        raise ValueError(
            f"model_variant must be 'large' or 'green', got '{self.model_variant}'"
        )

    # Validate that LoRA parameters are reasonable
    if self.lora_r < 1 or self.lora_r > 64:
        raise ValueError(
            f"lora_r must be between 1 and 64, got {self.lora_r}"
        )

    if self.lora_alpha < 1:
        raise ValueError(
            f"lora_alpha must be positive, got {self.lora_alpha}"
        )

    # Validate checkpoint path based on variant
    if self.model_variant == "large":
        if "green" in self.pretrained_path.lower():
            logger.warning(
                f"model_variant is 'large' but pretrained_path contains 'green': "
                f"{self.pretrained_path}"
            )
    elif self.model_variant == "green":
        if "green" not in self.pretrained_path.lower():
            logger.warning(
                f"model_variant is 'green' but pretrained_path doesn't contain 'green': "
                f"{self.pretrained_path}"
            )
```

#### 3. Update ImageConfig for variant support

The `ImageConfig` controls image size and normalization. Update to be variant-aware:

```python
@dataclass
class ImageConfig:
    """Image preprocessing configuration."""

    input_size: int = 224  # Will be overridden based on model_variant
    # Normalization values - will be set based on variant
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation parameters
    augmentation_prob: float = 0.5
    max_rotation: int = 10
    # ... rest of fields unchanged

    def update_for_variant(self, model_variant: str):
        """Update image config based on model variant."""
        if model_variant == "large":
            # RETFound (ViT-Large)
            self.input_size = 224
            self.mean = [0.485, 0.456, 0.406]  # ImageNet
            self.std = [0.229, 0.224, 0.225]
        elif model_variant == "green":
            # RETFound_Green (ViT-Small)
            self.input_size = 392
            self.mean = [0.5, 0.5, 0.5]  # Custom
            self.std = [0.5, 0.5, 0.5]
        else:
            raise ValueError(f"Unknown model_variant: {model_variant}")
```

#### 4. Add config validation function

```python
def validate_config(config: Config) -> None:
    """
    Validate entire configuration for consistency.

    Checks:
    - model_variant and image transforms match
    - paths exist or will be created
    - hyperparameters are reasonable
    """
    # Update image config based on model variant
    config.image.update_for_variant(config.model.model_variant)

    # Validate checkpoint path
    checkpoint_path = Path(config.model.pretrained_path)
    if not checkpoint_path.exists():
        logger.warning(
            f"Pretrained checkpoint not found: {checkpoint_path}. "
            f"Model will need to be downloaded."
        )

    # Log configuration
    logger.info(
        f"Configuration validated: "
        f"variant={config.model.model_variant}, "
        f"img_size={config.image.input_size}, "
        f"lora_r={config.model.lora_r}"
    )
```

### Backward Compatibility
- ✅ New field has default `model_variant="large"`
- ✅ Old configs without this field still work (loads default)
- ✅ New configs can override with `model_variant: green`

---

## File 2: scripts/train_retfound_lora.py

### Current State
Training script that:
- Loads config from YAML
- Creates model with RETFoundLoRA
- Runs training loop with validation
- Saves checkpoints
- Logs metrics

### Required Changes

#### 1. Add Command-Line Argument

Update argument parser to accept variant override:

```python
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RETFound (or RETFound_Green) with LoRA adaptation'
    )

    # ... existing arguments ...

    # NEW: Model variant argument
    parser.add_argument(
        '--model_variant',
        type=str,
        choices=['large', 'green'],
        default=None,  # None means use config value
        help='Foundation model variant: large (ViT-L, 303M) or green (ViT-S, 21.3M). '
             'Overrides value in config if specified.'
    )

    # ... rest of arguments ...

    return parser.parse_args()
```

#### 2. Add Transform Generation Function

Create a variant-aware function to generate albumentations transforms:

```python
def get_transforms(
    image_size: int,
    model_variant: str,
    augmentation: bool = True
) -> Tuple[A.Compose, A.Compose]:
    """
    Create augmentation and normalization transforms.

    Args:
        image_size: Input image size (224 for Large, 392 for Green)
        model_variant: 'large' (ImageNet norm) or 'green' (0.5 norm)
        augmentation: If True, apply augmentation for training; else just resize/norm

    Returns:
        (train_transform, val_transform)
    """
    # Determine normalization based on variant
    if model_variant == "large":
        # ImageNet normalization for RETFound
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_variant == "green":
        # Custom normalization for RETFound_Green
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")

    if augmentation:
        # Training augmentation (with data augmentation)
        train_transform = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=None)
    else:
        # Training without augmentation
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=None)

    # Validation transform (no augmentation, just resize and normalize)
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=None)

    return train_transform, val_transform
```

#### 3. Update Training Configuration Setup

Modify the main training function to handle variants:

```python
def main():
    """Main training function."""
    args = parse_arguments()

    # Load configuration from YAML
    config = load_config(args.config)

    # Override model_variant from command-line if specified
    if args.model_variant is not None:
        config.model.model_variant = args.model_variant
        logger.info(f"Overriding model_variant with command-line arg: {args.model_variant}")

    # Validate configuration (updates image config based on variant)
    validate_config(config)

    # Set seeds for reproducibility
    set_seed(config.system.seed)

    # Determine device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    logger.info(f"Using device: {device}")

    # Create data transforms based on variant
    train_transform, val_transform = get_transforms(
        image_size=config.image.input_size,
        model_variant=config.model.model_variant,
        augmentation=True
    )

    logger.info(f"Transforms configured for {config.model.model_variant} variant:")
    logger.info(f"  Image size: {config.image.input_size}×{config.image.input_size}")
    logger.info(f"  Normalization mean: {config.image.mean}")
    logger.info(f"  Normalization std: {config.image.std}")

    # ... rest of main function (loading data, creating model, etc.) ...
```

#### 4. Update Model Creation

When creating RETFoundLoRA, pass the variant:

```python
def create_model(config: Config, device: torch.device) -> RETFoundLoRA:
    """Create RETFoundLoRA model with variant support."""
    logger.info(f"Creating RETFoundLoRA with variant='{config.model.model_variant}'...")

    model = RETFoundLoRA(
        checkpoint_path=config.model.pretrained_path,
        num_classes=config.model.num_classes,
        model_variant=config.model.model_variant,  # Pass variant here
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        head_dropout=config.model.head_dropout,
        device=device,
    )

    # Print trainable parameter summary
    model.print_trainable_summary()

    return model
```

#### 5. Update Checkpoint Saving

Ensure variant is saved in checkpoint metadata:

```python
def save_checkpoint(
    model: RETFoundLoRA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    training_history: dict,
    config: Config,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save training checkpoint with variant metadata."""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'training_history': training_history,
        # NEW: Store variant information
        'lora_config': {
            'r': config.model.lora_r,
            'alpha': config.model.lora_alpha,
            'variant': config.model.model_variant,  # Store variant here
            'embed_dim': model.embed_dim,
            'checkpoint_path': str(config.model.pretrained_path),
        },
        'config': asdict(config),  # Store full config
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_best:
        checkpoint_path = checkpoint_dir / 'best_model.pth'
    else:
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    if is_best:
        logger.info(f"New best accuracy: {best_acc:.4f}")
```

#### 6. Log Training Configuration

Add detailed logging of configuration at training start:

```python
def log_training_config(config: Config) -> None:
    """Log detailed training configuration."""
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)

    logger.info(f"\nModel Configuration:")
    logger.info(f"  Model Name: {config.model.model_name}")
    logger.info(f"  Variant: {config.model.model_variant}")
    logger.info(f"  Pretrained Path: {config.model.pretrained_path}")
    logger.info(f"  Number of Classes: {config.model.num_classes}")

    logger.info(f"\nLoRA Configuration:")
    logger.info(f"  LoRA Rank (r): {config.model.lora_r}")
    logger.info(f"  LoRA Alpha: {config.model.lora_alpha}")
    logger.info(f"  Head Dropout: {config.model.head_dropout}")

    logger.info(f"\nImage Configuration:")
    logger.info(f"  Input Size: {config.image.input_size}×{config.image.input_size}")
    logger.info(f"  Normalization Mean: {config.image.mean}")
    logger.info(f"  Normalization Std: {config.image.std}")

    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Batch Size: {config.training.batch_size}")
    logger.info(f"  Learning Rate: {config.training.learning_rate}")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Optimizer: {config.training.optimizer}")

    logger.info(f"\nSystem Configuration:")
    logger.info(f"  Seed: {config.system.seed}")
    logger.info(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info("=" * 70 + "\n")
```

### Backward Compatibility
- ✅ `--model_variant` is optional (uses config value if not specified)
- ✅ Default in config is `model_variant="large"` (preserves existing behavior)
- ✅ Old training commands work unchanged:
  ```bash
  python3 scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
  ```

---

## Configuration File Updates

### Update: configs/retfound_lora_config.yaml

Modify to explicitly set variant and update comments:

```yaml
# RETFound (ViT-Large, 303M) LoRA Configuration
# This is the baseline configuration using the full-size RETFound model.

model:
  model_name: retfound_lora
  model_variant: large  # NEW: explicitly set to 'large'
  num_classes: 5
  pretrained: true
  pretrained_path: "models/RETFound_cfp_weights.pth"

  # LoRA Hyperparameters (tuned for RETFound Large)
  lora_r: 8           # Rank (8 recommended for Large)
  lora_alpha: 32      # Scaling factor (4x rank)
  head_dropout: 0.1

image:
  # RETFound uses 224×224 with ImageNet normalization
  input_size: 224
  mean: [0.485, 0.456, 0.406]  # ImageNet normalization
  std: [0.229, 0.224, 0.225]

  # Augmentation parameters
  max_rotation: 10
  brightness_contrast_limit: 0.2
  augmentation_prob: 0.5

# ... rest of config unchanged ...
```

---

## Implementation Checklist

### Code Implementation
- [ ] Add `model_variant` field to `ModelConfig` in `config.py`
- [ ] Add `update_for_variant()` method to `ImageConfig`
- [ ] Add `__post_init__()` validation to `ModelConfig`
- [ ] Add `validate_config()` function
- [ ] Add `--model_variant` argument to training script
- [ ] Implement `get_transforms()` function with variant-aware normalization
- [ ] Update model creation to pass `model_variant` to `RETFoundLoRA`
- [ ] Update checkpoint saving to include variant in metadata
- [ ] Add `log_training_config()` function
- [ ] Update `configs/retfound_lora_config.yaml` to explicitly set variant

### Testing
- [ ] Test config loading with and without `model_variant` field
- [ ] Test `update_for_variant("large")` sets correct image size and normalization
- [ ] Test `update_for_variant("green")` sets correct image size and normalization
- [ ] Test training with `--model_variant large` (existing workflow)
- [ ] Test training with `--model_variant green` (new workflow)
- [ ] Test checkpoint loading includes variant metadata
- [ ] Test config validation catches invalid variants

### Validation
- [ ] Training runs without errors for `model_variant="large"`
- [ ] Training runs without errors for `model_variant="green"`
- [ ] Image preprocessing matches model expectations
- [ ] Normalization is correct for each variant
- [ ] Checkpoint metadata includes variant and embed_dim
- [ ] Backward compatibility: old configs still work
- [ ] Model output shapes are consistent

---

## Example Usage

### Training with RETFound (Large) - Original Workflow
```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --checkpoint_path models/RETFound_cfp_weights.pth
# Uses model_variant="large" from config
```

### Training with RETFound_Green (New Workflow)
```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_green_lora_config.yaml \
    --checkpoint_path models/retfoundgreen_statedict.pth \
    --model_variant green
# Uses model_variant="green" from command line
```

### Override Variant via CLI
```bash
python3 scripts/train_retfound_lora.py \
    --config configs/retfound_lora_config.yaml \
    --model_variant green
# Uses green variant despite config saying "large"
```

---

## Validation Criteria for Phase 2

When this phase is complete, you should be able to:

1. ✅ Load configuration with model_variant field
   ```python
   config = load_config('configs/retfound_lora_config.yaml')
   assert config.model.model_variant == 'large'
   ```

2. ✅ Update image config based on variant
   ```python
   config.image.update_for_variant('green')
   assert config.image.input_size == 392
   assert config.image.mean == [0.5, 0.5, 0.5]
   ```

3. ✅ Generate transforms for each variant
   ```python
   train_tf, val_tf = get_transforms(224, 'large')
   train_tf, val_tf = get_transforms(392, 'green')
   ```

4. ✅ Train with both variants
   ```bash
   python3 scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
   python3 scripts/train_retfound_lora.py --config configs/retfound_green_lora_config.yaml
   ```

5. ✅ Checkpoint includes variant metadata
   ```python
   checkpoint = torch.load('results/checkpoint_epoch_001.pth')
   assert checkpoint['lora_config']['variant'] in ['large', 'green']
   ```

6. ✅ Backward compatibility maintained
   ```bash
   # Old workflow still works
   python3 scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
   ```

---

## Potential Issues & Solutions

### Issue 1: Normalization Mismatch
**Problem**: Using wrong mean/std for variant

**Solution**:
- ImageNet norm for Large: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- Custom norm for Green: `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`
- Verify in config and transforms

### Issue 2: Image Size Mismatch
**Problem**: Feeding 224×224 images to Green (expects 392×392)

**Solution**:
- Large: input_size=224
- Green: input_size=392
- `update_for_variant()` handles this automatically

### Issue 3: Config Not Updated
**Problem**: Old config file doesn't have model_variant field

**Solution**:
- Set default: `model_variant = "large"`
- Code handles missing field gracefully
- Update config files explicitly for clarity

---

## Transition to Phase 3

Once Phase 2 is complete and validated, proceed to Phase 3 which will:
- Enable cross-dataset evaluation with both variants
- Add variant detection in evaluation scripts
- Support hyperparameter optimization for Green
- Validate generalization performance

See `03-phase3-evaluation.md` for details.
