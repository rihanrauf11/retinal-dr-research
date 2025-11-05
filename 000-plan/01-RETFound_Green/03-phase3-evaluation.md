# Phase 3: Evaluation & Cross-Dataset Support

## Overview

**Objective**: Enable evaluation and hyperparameter optimization for both model variants with robust variant detection.

**Files Modified**: `scripts/evaluate_cross_dataset.py`, `scripts/hyperparameter_search.py`

**Estimated Effort**: 1-2 days

**Risk Level**: **MEDIUM** - Variant detection must be robust

**Validation**: Cross-dataset evaluation works with both variants, variant auto-detection is reliable

---

## File 1: scripts/evaluate_cross_dataset.py

### Current State
The script:
- Loads trained checkpoints
- Evaluates on multiple datasets
- Computes metrics (accuracy, AUC, etc.)
- Generates confusion matrices
- Detects model type (baseline, LoRA, etc.)

### Required Changes

#### 1. Add Variant Detection Functions

Add these functions to detect and handle model variants:

```python
def detect_model_variant(checkpoint_path: Union[str, Path]) -> str:
    """
    Detect model variant from checkpoint metadata.

    Strategy:
    1. Check explicit 'variant' field in lora_config
    2. Check embedding dimension (1024=Large, 384=Green)
    3. Check parameter count in state dict
    4. Default to 'large' if uncertain

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        'large' or 'green'

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False
        )
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        raise

    # Strategy 1: Explicit variant metadata
    if isinstance(checkpoint, dict) and 'lora_config' in checkpoint:
        lora_config = checkpoint['lora_config']

        if 'variant' in lora_config:
            variant = lora_config['variant']
            logger.info(f"Detected variant from metadata: {variant}")
            return variant

        # Strategy 2: Use embed_dim if available
        if 'embed_dim' in lora_config:
            embed_dim = lora_config['embed_dim']
            if embed_dim == 1024:
                logger.info(f"Detected variant from embed_dim=1024: large")
                return 'large'
            elif embed_dim == 384:
                logger.info(f"Detected variant from embed_dim=384: green")
                return 'green'

    # Strategy 3: Infer from state dict parameter count
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if isinstance(state_dict, dict):
        # Count parameters in state dict
        param_count = sum(
            v.numel() if isinstance(v, torch.Tensor) else 1
            for v in state_dict.values()
        )

        # RETFound Large: ~303M params, Green: ~21.3M params
        if param_count > 50_000_000:  # >50M likely Large
            logger.info(f"Detected variant from param_count={param_count:,}: large")
            return 'large'
        else:
            logger.info(f"Detected variant from param_count={param_count:,}: green")
            return 'green'

    # Default to 'large' if all detection strategies fail
    logger.warning(
        f"Could not detect variant from {checkpoint_path}. "
        f"Defaulting to 'large'. "
        f"Consider adding 'variant' field to checkpoint metadata."
    )
    return 'large'


def get_evaluation_transforms(
    variant: str,
    image_size: int = None
) -> Tuple[A.Compose, None]:
    """
    Get evaluation transforms for a specific model variant.

    Args:
        variant: 'large' or 'green'
        image_size: Override default image size (optional)

    Returns:
        Albumentations transform pipeline
    """
    if variant == "large":
        if image_size is None:
            image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif variant == "green":
        if image_size is None:
            image_size = 392
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return transform


def get_model_config_from_checkpoint(
    checkpoint_path: Union[str, Path]
) -> dict:
    """
    Extract model configuration from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Dictionary with keys: variant, embed_dim, lora_r, lora_alpha, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )

    variant = detect_model_variant(checkpoint_path)

    if isinstance(checkpoint, dict) and 'lora_config' in checkpoint:
        lora_config = checkpoint['lora_config']
    else:
        lora_config = {}

    config = {
        'variant': variant,
        'embed_dim': lora_config.get('embed_dim', 1024 if variant == 'large' else 384),
        'lora_r': lora_config.get('r', 8),
        'lora_alpha': lora_config.get('alpha', 32),
        'checkpoint_path': str(lora_config.get('checkpoint_path', 'unknown')),
    }

    return config
```

#### 2. Update Evaluation Function

Modify the main evaluation function to detect and use correct variant:

```python
def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    test_datasets: Dict[str, Tuple[str, str]],  # name: (csv_path, img_dir)
    model_type: str = None,  # 'lora', 'baseline', etc.
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device = None,
) -> dict:
    """
    Evaluate a checkpoint on multiple test datasets.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_datasets: Dict mapping dataset name to (csv_path, image_dir)
        model_type: Type of model ('lora', 'baseline', 'retfound')
                   Auto-detected if None
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        device: Device to run evaluation on

    Returns:
        Dictionary with results for each dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(checkpoint_path)

    # Auto-detect model variant
    variant = detect_model_variant(checkpoint_path)
    logger.info(f"Evaluating checkpoint with variant: {variant}")

    # Get model configuration
    model_config = get_model_config_from_checkpoint(checkpoint_path)
    logger.info(f"Model config: {model_config}")

    # Load model with correct architecture
    logger.info(f"Loading model checkpoint: {checkpoint_path}")

    try:
        if model_type == 'lora' or 'lora' in str(checkpoint_path):
            model = load_lora_checkpoint(
                checkpoint_path,
                model_variant=variant,
                device=device
            )
        else:
            model = load_baseline_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    model.eval()

    # Get evaluation transforms for this variant
    eval_transform = get_evaluation_transforms(variant)

    results = {}

    # Evaluate on each dataset
    for dataset_name, (csv_path, img_dir) in test_datasets.items():
        logger.info(f"\nEvaluating on {dataset_name}...")

        try:
            dataset = RetinalDataset(
                csv_file=csv_path,
                img_dir=img_dir,
                transform=eval_transform
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            # Run evaluation
            metrics = evaluate_dataset(
                model=model,
                dataloader=dataloader,
                device=device
            )

            metrics['dataset_name'] = dataset_name
            metrics['num_samples'] = len(dataset)
            results[dataset_name] = metrics

            logger.info(
                f"{dataset_name}: Accuracy={metrics['accuracy']:.4f}, "
                f"AUC={metrics['auc']:.4f}"
            )

        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}

    return results


def load_lora_checkpoint(
    checkpoint_path: Union[str, Path],
    model_variant: str,
    device: torch.device = None,
) -> RETFoundLoRA:
    """
    Load LoRA checkpoint with correct model variant.

    Args:
        checkpoint_path: Path to checkpoint
        model_variant: 'large' or 'green'
        device: Device to place model on

    Returns:
        Loaded RETFoundLoRA model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Extract configuration
    lora_config = checkpoint.get('lora_config', {})

    # Recreate model with correct variant
    model = RETFoundLoRA(
        checkpoint_path=lora_config.get('checkpoint_path', 'models/RETFound_cfp_weights.pth'),
        num_classes=5,  # Infer from checkpoint or hardcode
        model_variant=model_variant,
        lora_r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('alpha', 32),
        device=device
    )

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device)
```

#### 3. Update Argument Parsing

Add variant information to evaluation arguments:

```python
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate model on cross-dataset benchmark'
    )

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
        help='Datasets to evaluate on. Format: NAME:csv:images NAME2:csv2:images2'
    )

    # NEW: Optional variant override
    parser.add_argument(
        '--model_variant',
        type=str,
        choices=['large', 'green'],
        default=None,
        help='Override detected model variant (for debugging)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    # ... rest of arguments ...

    return parser.parse_args()
```

#### 4. Update Main Evaluation Pipeline

```python
def main():
    """Main evaluation pipeline."""
    args = parse_arguments()

    # Setup logging
    setup_logging(log_file='evaluation.log')

    logger.info("Starting cross-dataset evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Auto-detect model variant
    if args.model_variant:
        variant = args.model_variant
        logger.info(f"Using user-specified variant: {variant}")
    else:
        variant = detect_model_variant(args.checkpoint)
        logger.info(f"Auto-detected variant: {variant}")

    # Parse dataset arguments
    test_datasets = {}
    for dataset_spec in args.datasets:
        parts = dataset_spec.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid dataset spec: {dataset_spec}")
        name, csv_path, img_dir = parts
        test_datasets[name] = (csv_path, img_dir)

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_datasets=test_datasets,
        model_type='lora',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Print results
    print_evaluation_results(results, variant=variant)

    # Save results
    results_file = Path('evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
```

### Backward Compatibility
- ✅ Existing evaluation code continues to work
- ✅ Variant auto-detection is transparent (no user action needed)
- ✅ Optional `--model_variant` flag for debugging/override

---

## File 2: scripts/hyperparameter_search.py

### Current State
The script:
- Uses Optuna to search hyperparameter space
- Trains multiple models with different parameters
- Tracks best parameters
- Saves optimization results

### Required Changes

#### 1. Add Variant Parameter to Search Space

```python
def get_optuna_search_space(model_variant: str = 'large') -> dict:
    """
    Define Optuna search space based on model variant.

    Smaller models (Green) may benefit from different hyperparameters
    than larger models (Large).

    Args:
        model_variant: 'large' or 'green'

    Returns:
        Dictionary with parameter ranges for Optuna
    """
    if model_variant == 'large':
        # Search space tuned for RETFound Large (303M params)
        search_space = {
            'lora_r': [4, 8, 16, 32],  # Rank
            'lora_alpha': [16, 32, 64],  # Alpha (2-4x rank)
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
            'batch_size': [16, 32],
            'head_dropout': [0.0, 0.1, 0.2],
        }
    elif model_variant == 'green':
        # Search space tuned for RETFound_Green (21.3M params)
        # Smaller model may need different parameters
        search_space = {
            'lora_r': [4, 8, 16],  # Smaller ranks sufficient
            'lora_alpha': [16, 32],  # Slightly lower alpha
            'learning_rate': [1e-4, 5e-4, 1e-3],  # Higher LR for smaller model
            'batch_size': [32, 64],  # Can use larger batches
            'head_dropout': [0.0, 0.1],  # Less dropout needed
        }
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")

    return search_space


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        train_csv: str,
        train_img_dir: str,
        val_csv: str,
        val_img_dir: str,
        model_variant: str = 'large',
        checkpoint_path: str = None,
        n_trials: int = 50,
        output_dir: str = 'results/optuna',
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            train_csv: Path to training CSV
            train_img_dir: Path to training images
            val_csv: Path to validation CSV
            val_img_dir: Path to validation images
            model_variant: 'large' or 'green'
            checkpoint_path: Path to foundation model weights
            n_trials: Number of trials to run
            output_dir: Directory to save results
        """
        self.train_csv = train_csv
        self.train_img_dir = train_img_dir
        self.val_csv = val_csv
        self.val_img_dir = val_img_dir
        self.model_variant = model_variant
        self.checkpoint_path = checkpoint_path
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get search space for this variant
        self.search_space = get_optuna_search_space(model_variant)

        logger.info(
            f"Initialized optimizer for {model_variant} variant "
            f"with {n_trials} trials"
        )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Validation accuracy (higher is better)
        """
        # Suggest hyperparameters
        params = {
            'lora_r': trial.suggest_categorical('lora_r', self.search_space['lora_r']),
            'lora_alpha': trial.suggest_categorical('lora_alpha', self.search_space['lora_alpha']),
            'learning_rate': trial.suggest_categorical('learning_rate', self.search_space['learning_rate']),
            'batch_size': trial.suggest_categorical('batch_size', self.search_space['batch_size']),
            'head_dropout': trial.suggest_categorical('head_dropout', self.search_space['head_dropout']),
        }

        logger.info(f"\nTrial {trial.number}: {params}")

        try:
            # Train model with these parameters
            val_acc = self._train_and_evaluate(params)
            logger.info(f"Trial {trial.number} validation accuracy: {val_acc:.4f}")
            return val_acc

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst score on failure

    def _train_and_evaluate(self, params: dict) -> float:
        """
        Train model with given parameters and return validation accuracy.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Validation accuracy
        """
        # Create model
        model = RETFoundLoRA(
            checkpoint_path=self.checkpoint_path,
            num_classes=5,
            model_variant=self.model_variant,
            lora_r=params['lora_r'],
            lora_alpha=params['lora_alpha'],
            head_dropout=params['head_dropout'],
        )

        # Create data loaders (abbreviated for clarity)
        train_loader = self._get_dataloader(
            self.train_csv,
            self.train_img_dir,
            params['batch_size'],
            augment=True
        )
        val_loader = self._get_dataloader(
            self.val_csv,
            self.val_img_dir,
            params['batch_size'],
            augment=False
        )

        # Training loop (abbreviated)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate']
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Train for fixed number of epochs
        for epoch in range(5):  # Short training for quick evaluation
            # ... training code ...
            pass

        # Evaluate on validation set
        val_acc = self._evaluate(model, val_loader)

        return val_acc

    def optimize(self) -> dict:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting hyperparameter optimization for {self.model_variant}")

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'retfound_{self.model_variant}_search'
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=1,  # Use 1 job (don't parallelize to avoid GPU conflicts)
            show_progress_bar=True
        )

        # Extract best trial
        best_trial = study.best_trial
        logger.info(f"\nBest trial: {best_trial.number}")
        logger.info(f"Best validation accuracy: {best_trial.value:.4f}")
        logger.info(f"Best parameters: {best_trial.params}")

        # Save results
        results = {
            'model_variant': self.model_variant,
            'best_trial_number': best_trial.number,
            'best_accuracy': best_trial.value,
            'best_params': best_trial.params,
            'n_trials': self.n_trials,
            'search_space': self.search_space,
        }

        results_file = self.output_dir / f'results_{self.model_variant}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        return results
```

#### 2. Update CLI Arguments

```python
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization with Optuna'
    )

    # ... existing arguments ...

    # NEW: Model variant argument
    parser.add_argument(
        '--model_variant',
        type=str,
        choices=['large', 'green'],
        default='large',
        help='Model variant to optimize (large or green)'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of Optuna trials'
    )

    # ... rest of arguments ...

    return parser.parse_args()


def main():
    """Main hyperparameter search function."""
    args = parse_arguments()

    logger.info(f"Starting hyperparameter search for {args.model_variant}")

    optimizer = HyperparameterOptimizer(
        train_csv=args.train_csv,
        train_img_dir=args.train_img_dir,
        val_csv=args.val_csv,
        val_img_dir=args.val_img_dir,
        model_variant=args.model_variant,  # Pass variant
        checkpoint_path=args.checkpoint_path,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )

    results = optimizer.optimize()

    print_summary(results)
```

### Backward Compatibility
- ✅ Default `model_variant='large'` preserves existing behavior
- ✅ New `--model_variant green` enables search for smaller model
- ✅ Search space adapts automatically based on variant

---

## Implementation Checklist

### Code Implementation
- [ ] Add `detect_model_variant()` function
- [ ] Add `get_evaluation_transforms()` function
- [ ] Add `get_model_config_from_checkpoint()` function
- [ ] Add `load_lora_checkpoint()` with variant support
- [ ] Update evaluation function to use variant detection
- [ ] Add `--model_variant` CLI argument for evaluation
- [ ] Add `get_optuna_search_space()` for variant-aware search
- [ ] Update `HyperparameterOptimizer` with variant support
- [ ] Update hyperparameter search CLI with variant argument

### Testing
- [ ] Test variant detection on Large checkpoint
- [ ] Test variant detection on Green checkpoint
- [ ] Test evaluation transforms for both variants
- [ ] Test cross-dataset evaluation with both variants
- [ ] Test hyperparameter search with variant='large'
- [ ] Test hyperparameter search with variant='green'
- [ ] Test backward compatibility (old checkpoints still work)

### Validation
- [ ] Checkpoint variant auto-detection is 100% reliable
- [ ] Evaluation transforms match model expectations
- [ ] Cross-dataset evaluation works for both variants
- [ ] Results are comparable between variants
- [ ] Hyperparameter optimization converges for both models
- [ ] Backward compatibility maintained

---

## Example Usage

### Evaluate RETFound Large on Multiple Datasets
```bash
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/images \
        Messidor:data/messidor/test.csv:data/messidor/images
# Auto-detects variant=large
```

### Evaluate RETFound_Green on Multiple Datasets
```bash
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_green_lora/checkpoints/best_model.pth \
    --datasets \
        APTOS:data/aptos/test.csv:data/aptos/images \
        Messidor:data/messidor/test.csv:data/messidor/images
# Auto-detects variant=green
```

### Run Hyperparameter Search for RETFound_Green
```bash
python3 scripts/hyperparameter_search.py \
    --checkpoint-path models/retfoundgreen_statedict.pth \
    --model_variant green \
    --n_trials 50
# Uses Green-specific search space
```

---

## Validation Criteria for Phase 3

When this phase is complete, you should be able to:

1. ✅ Auto-detect model variant from checkpoint
   ```python
   variant = detect_model_variant('checkpoint.pth')
   assert variant in ['large', 'green']
   ```

2. ✅ Get evaluation transforms for each variant
   ```python
   tf_large = get_evaluation_transforms('large')
   tf_green = get_evaluation_transforms('green')
   ```

3. ✅ Evaluate both model types on cross-dataset
   ```bash
   python3 scripts/evaluate_cross_dataset.py --checkpoint large_checkpoint.pth --datasets ...
   python3 scripts/evaluate_cross_dataset.py --checkpoint green_checkpoint.pth --datasets ...
   ```

4. ✅ Run hyperparameter search for both variants
   ```bash
   python3 scripts/hyperparameter_search.py --model_variant large --n_trials 50
   python3 scripts/hyperparameter_search.py --model_variant green --n_trials 50
   ```

5. ✅ Compare results between variants
   - Training accuracy
   - Cross-dataset generalization gap
   - Training speed
   - Memory usage

---

## Potential Issues & Solutions

### Issue 1: Variant Detection Fails
**Problem**: Cannot determine variant from old checkpoint without metadata

**Solution**:
- Fallback strategies: param count → embed_dim → default to 'large'
- User can override with `--model_variant` flag
- Update checkpoints to include variant in metadata

### Issue 2: Wrong Transform Applied
**Problem**: Evaluation uses wrong normalization/size for variant

**Solution**:
- Verify variant detection is working
- Check checkpoint metadata
- Use explicit `--model_variant` override if needed

### Issue 3: Hyperparameter Search Stuck
**Problem**: Green model converges differently than Large

**Solution**:
- Use variant-specific search space
- Adjust learning rates (Green may need higher LR)
- Reduce training epochs for quick feedback

---

## Transition to Phase 4

Once Phase 3 is complete and validated, proceed to Phase 4 which will:
- Create RETFound_Green config template
- Update all documentation
- Final validation and testing

See `04-phase4-configuration.md` for details.
