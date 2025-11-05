# Implementation Notes & Technical Reference

**Purpose:** Technical details, gotchas, and best practices for implementing the testing suite
**Audience:** Developers implementing the test scripts

---

## Table of Contents

1. [PyTorch Testing Patterns](#pytorch-testing-patterns)
2. [Memory Profiling Best Practices](#memory-profiling-best-practices)
3. [RETFound-Specific Considerations](#retfound-specific-considerations)
4. [LoRA Testing Patterns](#lora-testing-patterns)
5. [Common Gotchas](#common-gotchas)
6. [Debugging Tips](#debugging-tips)
7. [Performance Optimization](#performance-optimization)
8. [Integration with Existing Code](#integration-with-existing-code)

---

## PyTorch Testing Patterns

### Deterministic Testing
```python
# Always set seeds for reproducible tests
def set_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Device Handling
```python
# Robust device detection
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Always test on CPU fallback
if device.type == "cpu":
    print("⚠️  Running on CPU (slower but functional)")
```

### Gradient Flow Verification
```python
# Check gradients exist and are not NaN
for name, param in model.named_parameters():
    if param.requires_grad:
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        # Also check gradient magnitude
        grad_norm = param.grad.norm().item()
        assert grad_norm > 0, f"Zero gradient for {name}"
```

### Loss Convergence Pattern
```python
# Train on fixed batch to verify learning
losses = []
for step in range(20):
    outputs = model(fixed_batch)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# Check monotonic decrease
assert losses[-1] < losses[0] * 0.5, "Loss should decrease by 50%+"
assert all(losses[i] <= losses[i-1] * 1.1 for i in range(1, len(losses))), \
    "Loss should decrease monotonically (within 10% tolerance)"
```

---

## Memory Profiling Best Practices

### CUDA Memory Management
```python
# Always reset memory stats before profiling
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Measure peak memory after warmup
for _ in range(3):  # Warmup
    forward_backward_pass()

torch.cuda.reset_peak_memory_stats()  # Reset after warmup

# Actual measurement
forward_backward_pass()
peak_mb = torch.cuda.max_memory_allocated() / 1e6
```

### Memory Leak Detection
```python
# Check for memory leaks in loops
initial_memory = torch.cuda.memory_allocated()

for i in range(100):
    outputs = model(batch)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 10 == 0:
        current_memory = torch.cuda.memory_allocated()
        memory_growth = current_memory - initial_memory
        assert memory_growth < 100e6, f"Memory leak detected: {memory_growth/1e6:.1f} MB growth"
```

### Gradient Accumulation Testing
```python
# Verify gradient accumulation works correctly
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## RETFound-Specific Considerations

### Checkpoint Loading
```python
# RETFound checkpoints may have key mismatches
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Handle different checkpoint formats
if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Remove 'module.' prefix if present (from DataParallel)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load with strict=False to handle key mismatches
model.load_state_dict(state_dict, strict=False)
```

### RETFound vs RETFound_Green
```python
# Different normalization stats
RETFOUND_LARGE_STATS = {
    'mean': [0.485, 0.456, 0.406],  # ImageNet
    'std': [0.229, 0.224, 0.225]
}

RETFOUND_GREEN_STATS = {
    'mean': [0.5, 0.5, 0.5],  # Custom
    'std': [0.5, 0.5, 0.5]
}

# Different image sizes
RETFOUND_LARGE_SIZE = 224
RETFOUND_GREEN_SIZE = 392
```

### ViT-Specific Testing
```python
# Verify patch embeddings work correctly
img_size = 224
patch_size = 16
num_patches = (img_size // patch_size) ** 2  # 196 for 224x224

# Check attention dimensions
batch_size = 4
num_heads = 16
seq_length = num_patches + 1  # +1 for class token

# Attention shape should be (batch, heads, seq_len, seq_len)
expected_attn_shape = (batch_size, num_heads, seq_length, seq_length)
```

---

## LoRA Testing Patterns

### Verify Only LoRA Parameters are Trainable
```python
# Count trainable vs frozen parameters
lora_params = []
frozen_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        lora_params.append(name)
    else:
        frozen_params.append(name)

# LoRA should have ~800K trainable params (0.26% of 303M)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert total_trainable < 1_000_000, "Too many trainable params for LoRA"

# Verify 'lora_' in trainable param names
assert all('lora_' in name or 'head' in name for name in lora_params), \
    "Only LoRA and head parameters should be trainable"
```

### LoRA Rank Testing
```python
# Test different ranks
for r in [4, 8, 16, 32]:
    model = RETFoundLoRA(
        checkpoint_path=checkpoint_path,
        num_classes=5,
        lora_r=r,
        lora_alpha=r * 4  # Alpha typically 4x rank
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Rank {r}: {trainable:,} trainable params")

    # Higher rank = more parameters
    # r=4: ~400K, r=8: ~800K, r=16: ~1.6M, r=32: ~3.2M
```

### LoRA Merge Testing
```python
# Verify LoRA can be merged back into base model
model_with_lora = RETFoundLoRA(...)
# Train...

# Merge LoRA weights
merged_model = model_with_lora.merge_and_unload()

# Verify outputs are identical
with torch.no_grad():
    out1 = model_with_lora(test_input)
    out2 = merged_model(test_input)
    assert torch.allclose(out1, out2, atol=1e-5), "Merge changed outputs"
```

---

## Common Gotchas

### 1. Batch Normalization in Eval Mode
```python
# BN behaves differently in train vs eval
model.train()  # Uses batch statistics
outputs_train = model(batch)

model.eval()  # Uses running statistics
with torch.no_grad():
    outputs_eval = model(batch)

# Outputs will differ! This is expected.
```

### 2. DataLoader num_workers on macOS
```python
# macOS MPS doesn't work well with num_workers > 0
if torch.backends.mps.is_available():
    num_workers = 0  # Force single-process
else:
    num_workers = 4
```

### 3. Gradient Accumulation Pitfalls
```python
# WRONG: Loss not scaled
loss = criterion(outputs, labels)
loss.backward()  # Accumulates gradients

# CORRECT: Scale loss by accumulation steps
loss = criterion(outputs, labels) / accumulation_steps
loss.backward()
```

### 4. Mixed Precision Training
```python
# When using AMP, wrap forward/backward
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5. Pin Memory with Non-Tensors
```python
# pin_memory only works with tensors
# If dataset returns dict with non-tensor metadata, set pin_memory=False
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=False  # If returning non-tensor data
)
```

---

## Debugging Tips

### Enable Anomaly Detection
```python
# Detect NaN/Inf during backprop
torch.autograd.set_detect_anomaly(True)

# This will show exactly where NaN originates
# WARNING: Slows down training significantly, only use for debugging
```

### Profile Slow Operations
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Visualize Gradient Flow
```python
def plot_grad_flow(named_parameters):
    """Plot gradient flow through model layers."""
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gradient_flow.png")

# Usage
plot_grad_flow(model.named_parameters())
```

### Check Data Distribution
```python
# Verify label distribution is balanced
from collections import Counter

labels = [dataset[i]['label'].item() for i in range(len(dataset))]
distribution = Counter(labels)

print("Class distribution:")
for cls, count in sorted(distribution.items()):
    print(f"  Class {cls}: {count} ({count/len(labels)*100:.1f}%)")

# Check for class imbalance
min_count = min(distribution.values())
max_count = max(distribution.values())
imbalance_ratio = max_count / min_count

if imbalance_ratio > 10:
    print(f"⚠️  Warning: Class imbalance ratio {imbalance_ratio:.1f}x")
    print("   Consider using weighted loss or oversampling")
```

---

## Performance Optimization

### DataLoader Optimization
```python
# Optimal settings for different scenarios
if torch.cuda.is_available():
    # CUDA: Use multiple workers + pin_memory
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Test 0,2,4,8 to find optimal
        pin_memory=True,
        persistent_workers=True,  # Avoid worker restart
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )
elif torch.backends.mps.is_available():
    # MPS: Single worker (multiprocessing issues)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
else:
    # CPU: Moderate workers
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Smaller batch for CPU
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
```

### Reduce Augmentation Overhead
```python
# Profile individual transforms
import time

transforms_to_test = [
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.ColorJitter(p=1.0),
    A.ElasticTransform(p=1.0),  # Often slow
]

image = np.array(PIL.Image.open("test.jpg"))

for transform in transforms_to_test:
    start = time.time()
    for _ in range(100):
        augmented = transform(image=image)['image']
    elapsed = time.time() - start
    print(f"{transform.__class__.__name__}: {elapsed*10:.1f} ms per image")

# Remove slow transforms if they're bottleneck
```

### Compilation (PyTorch 2.0+)
```python
# Compile model for faster training (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='default')
    print("✓ Model compiled with torch.compile")
```

---

## Integration with Existing Code

### Using Existing Fixtures
```python
# tests/conftest.py has 40+ fixtures
# Reuse them in test scripts

from tests.conftest import sample_csv, sample_images, sample_dataset

def test_with_fixtures(sample_dataset):
    """Use existing pytest fixtures in standalone scripts."""
    dataloader = DataLoader(sample_dataset, batch_size=4)
    batch = next(iter(dataloader))
    assert 'image' in batch
    assert 'label' in batch
```

### Calling Validation Scripts
```python
# Reuse validate_data.py logic
from scripts.validate_data import DataValidator

validator = DataValidator(
    csv_file="data/aptos/train.csv",
    img_dir="data/aptos/train_images"
)

is_valid = validator.validate_all()
if not is_valid:
    print("Data validation failed, fix issues first")
    sys.exit(1)
```

### Config System Integration
```python
# All test scripts should use config system
from scripts.config import load_config

config = load_config("configs/retfound_lora_config.yaml")

# Access config values
batch_size = config.training.batch_size
learning_rate = config.training.learning_rate
checkpoint_path = config.model.checkpoint_path

# Override from command line
if args.batch_size:
    config.training.batch_size = args.batch_size
```

---

## Testing Checklist

Before submitting test implementation, verify:

- [ ] Works on CUDA, MPS, and CPU
- [ ] Sets random seeds for reproducibility
- [ ] Has proper error handling and messages
- [ ] Cleans up resources (close files, clear CUDA cache)
- [ ] Provides clear pass/fail status
- [ ] Includes timing information
- [ ] Generates useful outputs (plots, reports)
- [ ] Exit code 0 on success, 1 on failure
- [ ] Documented usage in docstring
- [ ] Compatible with existing config system
- [ ] No hardcoded paths (use config)
- [ ] Handles missing dependencies gracefully
- [ ] Provides actionable recommendations on failure

---

## External References

### PyTorch Documentation
- Memory profiling: https://pytorch.org/docs/stable/torch.cuda.html#memory-management
- Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- Anomaly detection: https://pytorch.org/docs/stable/autograd.html#anomaly-detection

### PEFT (LoRA)
- Library: https://github.com/huggingface/peft
- LoRA paper: https://arxiv.org/abs/2106.09685
- Examples: https://huggingface.co/docs/peft/main/en/index

### RETFound
- Original paper: https://www.nature.com/articles/s41586-023-06555-x
- GitHub: https://github.com/rmaphoh/RETFound_MAE
- RETFound_Green: https://github.com/justinengelmann/RETFound_Green

### Best Practices
- PyTorch Performance Tuning: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Testing ML Models: https://madewithml.com/courses/mlops/testing/
- Data Validation: https://research.google/pubs/pub47967/

---

## Getting Help

If stuck during implementation:

1. **Check existing tests:** `tests/` directory has 287 examples
2. **Review validation scripts:** `scripts/validate_*.py` show patterns
3. **Read docstrings:** Most functions are documented
4. **Test incrementally:** Build and test each component separately
5. **Use verbose mode:** Add `--verbose` flags for debugging

---

**Document Version:** 1.0
**Last Updated:** 2025-11-04
**Maintainer:** Research Team
