# Hyperparameter Optimization Guide

## Overview

This guide explains how to use the automated hyperparameter optimization script (`scripts/hyperparameter_search.py`) to find the best hyperparameters for RETFound + LoRA training on diabetic retinopathy classification.

**Key Features:**
- Automated search using Optuna's TPE sampler
- Early stopping to save compute time
- Pruning of unpromising trials
- Comprehensive logging and visualization
- Resumable studies
- GPU-efficient with mixed precision training

## Installation

The hyperparameter optimization script requires Optuna:

```bash
pip install optuna>=3.0.0 optuna-dashboard>=0.9.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Search Space

The script optimizes the following hyperparameters:

| Hyperparameter | Type | Search Space | Description |
|----------------|------|--------------|-------------|
| **lora_r** | Categorical | [4, 8, 16, 32] | LoRA rank - controls adapter capacity |
| **lora_alpha** | Categorical | [16, 32, 64] | LoRA scaling factor |
| **learning_rate** | Log-uniform | [1e-5, 1e-3] | Learning rate for optimizer |
| **batch_size** | Categorical | [8, 16, 32] | Training batch size |
| **dropout** | Uniform | [0.1, 0.5] | Dropout rate before classifier |

## Quick Start

### 1. Basic Search (50 trials)

```bash
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --n-trials 50
```

**Expected Time:** ~12-16 hours on RTX 3090
**Output:** `results/optuna/retfound_lora_search/`

### 2. Quick Search (20 trials with timeout)

For faster experimentation:

```bash
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --n-trials 20 \
    --timeout-hours 6 \
    --num-epochs 8
```

**Expected Time:** ~5-7 hours on RTX 3090

### 3. Resume Existing Study

Continue a previously interrupted study:

```bash
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --resume-study results/optuna/retfound_lora_search/study.pkl \
    --n-trials 30
```

## Command-Line Arguments

### Required Arguments

- `--checkpoint-path` - Path to RETFound pretrained weights (.pth file)
- `--data-csv` - Path to training CSV file
- `--data-img-dir` - Path to training images directory

### Study Settings

- `--study-name` - Name for the Optuna study (default: `retfound_lora_search`)
- `--n-trials` - Number of optimization trials (default: 50)
- `--timeout-hours` - Maximum hours to run (default: None, no limit)
- `--output-dir` - Output directory (default: `results/optuna`)

### Training Settings

- `--num-epochs` - Maximum epochs per trial (default: 10)
  - *Reduced for faster trials*
- `--early-stopping-patience` - Patience for early stopping (default: 3)
  - *Stops if no improvement for 3 epochs*
- `--num-workers` - DataLoader workers (default: 4)
- `--seed` - Random seed (default: 42)

### Wandb Integration

- `--wandb` - Enable Weights & Biases logging
- `--wandb-project` - W&B project name (default: `diabetic-retinopathy-hpo`)

### Resume

- `--resume-study` - Path to existing `study.pkl` to resume

## Output Structure

After running optimization, you'll find:

```
results/optuna/<study_name>/
├── config.json                    # Configuration used
├── study.pkl                      # Optuna study object (resumable)
├── trials.csv                     # All trials with hyperparams and results
├── best_params.json              # Best hyperparameters
├── summary_report.md             # Human-readable summary
└── plots/
    ├── optimization_history.png   # Objective value over trials
    ├── param_importance.png       # Which params matter most
    ├── parallel_coordinate.png    # Param relationships
    └── slice_plots.png            # How each param affects objective
```

### trials.csv Format

CSV file containing all trial results:

| Column | Description |
|--------|-------------|
| `trial_number` | Trial number |
| `lora_r` | LoRA rank used |
| `lora_alpha` | LoRA alpha used |
| `learning_rate` | Learning rate used |
| `batch_size` | Batch size used |
| `dropout` | Dropout rate used |
| `best_val_acc` | Best validation accuracy achieved |
| `state` | Trial state (COMPLETE, PRUNED, FAIL) |
| `duration_seconds` | Time taken for trial |
| `datetime_start` | When trial started |
| `datetime_complete` | When trial completed |

### best_params.json Format

```json
{
  "lora_r": 16,
  "lora_alpha": 64,
  "learning_rate": 0.0002,
  "batch_size": 32,
  "dropout": 0.25,
  "best_value": 87.53,
  "best_trial": 23
}
```

## Interpreting Results

### 1. Optimization History Plot

Shows validation accuracy over trials. Look for:
- **Upward trend:** Search is finding better configs
- **Plateau:** Search has converged
- **Best value line:** Shows best accuracy found

### 2. Parameter Importance Plot

Shows which hyperparameters have the most impact:
- **High importance:** Focus on tuning these
- **Low importance:** Less critical, can use defaults

### 3. Parallel Coordinate Plot

Shows relationships between parameters and objective:
- **Lines colored by objective value**
- **Identify patterns** in successful trials

### 4. Slice Plots

Shows how each parameter affects validation accuracy:
- **Steep slopes:** Parameter has strong effect
- **Flat lines:** Parameter has weak effect

### 5. Summary Report

Markdown file with:
- Study statistics (completed, pruned, failed trials)
- Best trial details
- Top 10 trials table
- Mean and std of top 10
- Search space used

## Usage Examples

### Example 1: Full Search for Production

Run a comprehensive search for production deployment:

```bash
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --study-name production_search \
    --n-trials 100 \
    --num-epochs 15 \
    --output-dir results/optuna_production
```

### Example 2: Rapid Prototyping

Quickly test if optimization is working:

```bash
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --study-name quick_test \
    --n-trials 10 \
    --num-epochs 5 \
    --timeout-hours 2
```

### Example 3: Focused Search

After initial search, focus on promising regions:

```python
# Modify script's OptunaConfig to narrow search space:
lora_r_choices = [8, 16]  # Only test r=8,16
lora_alpha_choices = [32, 64]  # Only test alpha=32,64
lr_range = (1e-4, 5e-4)  # Narrower LR range
```

### Example 4: With Wandb Tracking

Track all trials in Weights & Biases:

```bash
# First login to wandb
wandb login

# Run with wandb enabled
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --n-trials 50 \
    --wandb \
    --wandb-project my-dr-hpo
```

Each trial will be logged as a separate run in W&B dashboard.

## Performance Optimization

### Speed Tips

1. **Reduce epochs per trial:**
   ```bash
   --num-epochs 8  # Instead of 10
   ```

2. **Use timeout instead of fixed trials:**
   ```bash
   --timeout-hours 12  # Run for 12 hours
   ```

3. **Early stopping:**
   - Already enabled by default (patience=3)
   - Stops trials that aren't improving

4. **Pruning:**
   - Already enabled with MedianPruner
   - Stops unpromising trials early
   - Saves ~30-40% compute time

### GPU Memory Tips

If you encounter CUDA OOM errors:

1. **Reduce batch size options:**
   ```python
   batch_size_choices = [8, 16]  # Remove 32
   ```

2. **Reduce LoRA rank options:**
   ```python
   lora_r_choices = [4, 8, 16]  # Remove 32
   ```

3. **Use gradient accumulation** (modify script):
   ```python
   accumulation_steps = 2
   effective_batch_size = batch_size * accumulation_steps
   ```

## Time Estimates

Based on RTX 3090 GPU:

| Configuration | Epochs/Trial | Time/Trial | Total Time (50 trials) |
|---------------|--------------|------------|------------------------|
| **Fast** | 8 | ~15 min | ~6-8 hours |
| **Standard** | 10 | ~20 min | ~12-16 hours |
| **Thorough** | 15 | ~30 min | ~18-25 hours |

**Notes:**
- Early stopping reduces average time by 30-40%
- Pruning stops unpromising trials early
- CUDA OOM failures are quick (~30 seconds)

## Best Practices

### 1. Start Small

```bash
# First run: 10 trials to test
python scripts/hyperparameter_search.py ... --n-trials 10

# Second run: 50 trials for real search
python scripts/hyperparameter_search.py ... --n-trials 50
```

### 2. Check Results Periodically

Monitor `trials.csv` during optimization:

```bash
# View latest trials
tail -n 5 results/optuna/retfound_lora_search/trials.csv

# Count completed trials
grep COMPLETE results/optuna/retfound_lora_search/trials.csv | wc -l
```

### 3. Resume if Interrupted

If optimization stops (power outage, etc.):

```bash
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --resume-study results/optuna/retfound_lora_search/study.pkl \
    --n-trials 50  # Will continue until 50 total
```

### 4. Validate Best Parameters

After optimization, validate the best hyperparameters:

```bash
# Train with best params on full data
python scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --config configs/retfound_lora_config.yaml \
    --lora_r 16 \
    --lora_alpha 64 \
    --batch_size 32 \
    --lr 0.0002 \
    --epochs 30  # Full training
```

### 5. Cross-Dataset Validation

Test best hyperparameters on other datasets:

```bash
# Evaluate on Messidor
python scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --datasets Messidor:data/messidor/test.csv:data/messidor/images
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Reduce batch size options
```python
batch_size_choices = [8, 16]  # Remove 32
```

**Solution 2:** Reduce LoRA rank options
```python
lora_r_choices = [4, 8]  # Remove 16, 32
```

**Solution 3:** Use smaller model (modify script)

### Issue: Trials Taking Too Long

**Solution 1:** Reduce epochs per trial
```bash
--num-epochs 8
```

**Solution 2:** More aggressive early stopping
```bash
--early-stopping-patience 2
```

**Solution 3:** Use timeout
```bash
--timeout-hours 8
```

### Issue: Study Not Finding Good Results

**Solution 1:** Increase number of trials
```bash
--n-trials 100
```

**Solution 2:** Expand search space
```python
lora_r_choices = [4, 8, 16, 32, 64]
lr_range = (1e-6, 1e-2)
```

**Solution 3:** Check data quality
- Verify CSV and images are correct
- Check class distribution
- Ensure proper data splits

### Issue: Want to Resume but Lost study.pkl

**Solution:** Can't resume, but you can check `trials.csv` for completed trials and manually set best params.

## Advanced Usage

### Custom Search Space

Edit `scripts/hyperparameter_search.py` to customize:

```python
@dataclass
class OptunaConfig:
    # Add new hyperparameters
    weight_decay_range: Tuple[float, float] = (1e-5, 1e-2)
    warmup_epochs_choices: List[int] = field(default_factory=lambda: [0, 2, 5])
```

Then in `objective()` function:

```python
weight_decay = trial.suggest_float('weight_decay', *config.weight_decay_range, log=True)
warmup_epochs = trial.suggest_categorical('warmup_epochs', config.warmup_epochs_choices)
```

### Multi-Objective Optimization

Optimize for both accuracy and parameter count:

```python
# In objective function, return tuple
return best_val_acc, -trainable_params  # Maximize acc, minimize params

# Create study with multiple objectives
study = optuna.create_study(
    directions=["maximize", "minimize"]
)
```

### Conditional Search Spaces

Make some hyperparameters depend on others:

```python
# In objective function
if lora_r >= 16:
    # Higher ranks need higher alpha
    lora_alpha = trial.suggest_categorical('lora_alpha', [32, 64, 128])
else:
    lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32])
```

## Example Workflow

### Complete Hyperparameter Optimization Workflow

```bash
# Step 1: Quick test (2 trials)
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --study-name test_run \
    --n-trials 2 \
    --num-epochs 3

# Step 2: Small search (20 trials)
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --study-name pilot_search \
    --n-trials 20 \
    --timeout-hours 6

# Step 3: Review results
cat results/optuna/pilot_search/summary_report.md
# Check plots in results/optuna/pilot_search/plots/

# Step 4: Full search (50 trials)
python scripts/hyperparameter_search.py \
    --checkpoint-path models/RETFound_cfp_weights.pth \
    --data-csv data/aptos/train.csv \
    --data-img-dir data/aptos/train_images \
    --study-name production_search \
    --n-trials 50 \
    --wandb

# Step 5: Train with best hyperparameters
# Get best params from results/optuna/production_search/best_params.json
python scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --config configs/retfound_lora_config.yaml \
    --lora_r 16 \
    --lora_alpha 64 \
    --batch_size 32 \
    --lr 0.0002 \
    --epochs 30

# Step 6: Evaluate on test set
python scripts/evaluate_cross_dataset.py \
    --checkpoint results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --datasets APTOS:data/aptos/test.csv:data/aptos/test_images
```

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [RETFound Paper](https://www.nature.com/articles/s41586-023-06555-x)
- [Hyperparameter Optimization Best Practices](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/index.html)

## Citation

If you use this hyperparameter optimization script in your research:

```bibtex
@software{retfound_lora_hpo,
  title={Hyperparameter Optimization for RETFound + LoRA},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

**Last Updated:** 2025-10-12
**Version:** 1.0.0
