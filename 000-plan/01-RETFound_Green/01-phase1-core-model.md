# Phase 1: Core Model Infrastructure

## Overview

**Objective**: Add RETFound_Green model loading functions to the codebase without modifying existing RETFound code.

**Files Modified**: `scripts/retfound_model.py`, `scripts/retfound_lora.py`

**Estimated Effort**: 2-3 days

**Risk Level**: **LOW** - All changes are additive

**Validation**: Unit tests pass, both variants load correctly

---

## File 1: scripts/retfound_model.py

### Current State
The file contains functions to load the original RETFound model:
- `load_retfound_model()` - Loads custom ViT-Large from checkpoint
- Handles ImageNet normalization, 224×224 input
- Parameter count: ~303M

### Required Changes

#### 1. Add Import for timm
```python
# Add to imports section
import timm
```

#### 2. Add Function: get_retfound_green()
Add this new function (don't modify existing load_retfound_model):

```python
def get_retfound_green(
    pretrained: bool = False,
    num_classes: int = 0,
    img_size: int = 392,
    **kwargs
) -> nn.Module:
    """
    Create RETFound_Green model using timm.

    RETFound_Green is a ViT-Small (21.3M params) trained with Token Reconstruction
    on 75K retinal images. It produces 384-dimensional feature embeddings.

    Args:
        pretrained: If True, will attempt to load from timm (not available yet)
        num_classes: Number of output classes. Use 0 for feature extraction mode.
        img_size: Input image size (default 392x392)
        **kwargs: Additional arguments passed to timm.create_model

    Returns:
        Vision Transformer model configured for retinal image analysis

    Notes:
        - Output embedding dimension: 384 (fixed by architecture)
        - Requires separate loading of pretrained weights
        - Uses mean=0.5, std=0.5 normalization (different from ImageNet)
        - Input size: 392x392 (larger than original RETFound's 224x224)
    """
    model = timm.create_model(
        'vit_small_patch14_reg4_dinov2',
        img_size=(img_size, img_size),
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )

    # Verify architecture
    assert hasattr(model, 'embed_dim'), "Model must have embed_dim attribute"
    assert model.embed_dim == 384, f"Expected embed_dim=384, got {model.embed_dim}"

    return model


def load_retfound_green_model(
    checkpoint_path: Union[str, Path],
    num_classes: int = 5,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load RETFound_Green model with pretrained weights.

    Args:
        checkpoint_path: Path to pretrained weights (statedict format)
        num_classes: Number of output classes for downstream classification
        device: Device to place model on (auto-detected if None)
        strict: If True, requires exact key match. If False, allows missing keys.

    Returns:
        Model with pretrained weights and classification head

    Example:
        >>> model = load_retfound_green_model(
        ...     'models/retfoundgreen_statedict.pth',
        ...     num_classes=5
        ... )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model in feature extraction mode (num_classes=0)
    backbone = get_retfound_green(num_classes=0)

    # Load pretrained weights
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle potential key mismatches (model may have been saved with different key names)
        load_result = backbone.load_state_dict(state_dict, strict=strict)

        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            logger.warning(
                f"Load result - Missing keys: {load_result.missing_keys}, "
                f"Unexpected keys: {load_result.unexpected_keys}"
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load RETFound_Green checkpoint from {checkpoint_path}: {e}"
        )

    # Add classification head on top of frozen backbone
    backbone = backbone.to(device)

    # Create wrapper that adds classification head
    class RETFoundGreenWithHead(nn.Module):
        def __init__(self, backbone: nn.Module, num_classes: int):
            super().__init__()
            self.backbone = backbone
            self.embed_dim = backbone.embed_dim  # 384
            self.num_classes = num_classes

            # Classification head
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Extract features from backbone
            features = self.backbone(x)  # Shape: [batch_size, 384]

            # Pass through classification head
            logits = self.head(features)  # Shape: [batch_size, num_classes]
            return logits

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract 384-dimensional features without classification."""
            return self.backbone(x)

    model = RETFoundGreenWithHead(backbone, num_classes)
    return model.to(device)


def detect_model_variant(checkpoint_path: Union[str, Path]) -> str:
    """
    Detect whether a checkpoint is from RETFound (large) or RETFound_Green (green).

    Uses heuristics based on:
    1. Metadata in checkpoint (if available)
    2. Parameter count
    3. State dict keys

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        'large' or 'green'

    Raises:
        ValueError: If variant cannot be determined
    """
    checkpoint_path = Path(checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise ValueError(f"Cannot load checkpoint {checkpoint_path}: {e}")

    # Check for explicit variant metadata
    if isinstance(checkpoint, dict):
        if 'lora_config' in checkpoint and 'variant' in checkpoint['lora_config']:
            return checkpoint['lora_config']['variant']

        # Fallback: check state dict structure
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Count parameters to estimate variant
        param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))

        # RETFound Large: ~303M params, Green: ~21.3M params
        if param_count > 100_000_000:  # >100M = Large
            return 'large'
        else:
            return 'green'

    raise ValueError(f"Cannot determine model variant from {checkpoint_path}")
```

#### 3. Update Logger (if not already imported)
```python
# Add to imports if missing
import logging
logger = logging.getLogger(__name__)
```

#### 4. Add to __main__ section for testing
```python
if __name__ == "__main__":
    # Test loading RETFound_Green
    try:
        print("Testing RETFound_Green loading...")
        model = get_retfound_green()
        print(f"✓ Created RETFound_Green: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test feature extraction
        x = torch.randn(2, 3, 392, 392)
        features = model(x)
        print(f"✓ Feature extraction works: output shape = {features.shape}")
        assert features.shape == (2, 384), f"Expected shape (2, 384), got {features.shape}"

    except Exception as e:
        print(f"✗ Error: {e}")
```

### Backward Compatibility
- ✅ Existing `load_retfound_model()` function unchanged
- ✅ All new functions use different names
- ✅ Old code continues to work exactly as before

---

## File 2: scripts/retfound_lora.py

### Current State
The `RETFoundLoRA` class currently:
- Takes `checkpoint_path` pointing to RETFound weights
- Hardcoded `embed_dim = 1024` (for ViT-Large)
- Uses fixed LoRA configuration
- Creates classification head on top of frozen backbone

### Required Changes

#### 1. Update Class: RETFoundLoRA.__init__

Modify the `__init__` method signature and add variant support:

```python
class RETFoundLoRA(nn.Module):
    """
    RETFound (or RETFound_Green) with LoRA fine-tuning.

    Freezes the foundation model backbone and adds low-rank adapters to attention layers.
    Only adapters and classification head are trainable (~0.26% of total parameters).

    Args:
        checkpoint_path: Path to pretrained foundation model weights
        num_classes: Number of output classes for classification
        model_variant: 'large' (ViT-Large, 303M) or 'green' (ViT-Small, 21.3M)
        lora_r: Rank of low-rank matrices (default: 8)
        lora_alpha: LoRA scaling factor (default: 32)
        head_dropout: Dropout in classification head (default: 0.1)
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        num_classes: int = 5,
        model_variant: str = 'large',  # NEW PARAMETER
        lora_r: int = 8,
        lora_alpha: int = 32,
        head_dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.model_variant = model_variant
        self.num_classes = num_classes
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # Load backbone based on variant
        if model_variant == 'large':
            logger.info("Loading RETFound (ViT-Large) backbone...")
            self.backbone = load_retfound_model(
                checkpoint_path=checkpoint_path,
                num_classes=0,  # Feature extraction mode
                device=device
            )
            self.embed_dim = 1024

        elif model_variant == 'green':
            logger.info("Loading RETFound_Green (ViT-Small) backbone...")
            self.backbone = load_retfound_green_model(
                checkpoint_path=checkpoint_path,
                num_classes=0,  # Feature extraction mode
                device=device
            )
            self.embed_dim = 384

        else:
            raise ValueError(f"Unknown model_variant: {model_variant}. "
                           f"Must be 'large' or 'green'.")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA to attention layers
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],  # Generic for both ViT-Large and ViT-Small
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        self.backbone = get_peft_model(self.backbone, peft_config)

        # Classification head (always trainable)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.embed_dim, num_classes)
        )

        logger.info(f"RETFoundLoRA initialized:")
        logger.info(f"  Variant: {model_variant} (embed_dim={self.embed_dim})")
        logger.info(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        logger.info(f"  Frozen parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}")

        self.to(device)
```

#### 2. Update forward() method (if needed)
The forward method may need updating to handle both model types. Check current implementation and ensure it works for both `embed_dim=1024` and `embed_dim=384`.

Current implementation likely looks like:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through backbone and classification head."""
    features = self.backbone(x)  # Extract features
    logits = self.head(features)  # Classify
    return logits
```

**No changes needed** - The forward pass is dimension-agnostic.

#### 3. Add utility method
```python
def get_trainable_params(self) -> int:
    """Return count of trainable parameters."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_frozen_params(self) -> int:
    """Return count of frozen parameters."""
    return sum(p.numel() for p in self.parameters() if not p.requires_grad)

def print_trainable_summary(self) -> None:
    """Print summary of trainable vs frozen parameters."""
    trainable = self.get_trainable_params()
    frozen = self.get_frozen_params()
    total = trainable + frozen
    pct = 100.0 * trainable / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Model: RETFound{'' if self.model_variant == 'large' else '_Green'}")
    print(f"{'='*50}")
    print(f"Total parameters:     {total:>12,} (100.0%)")
    print(f"Trainable:            {trainable:>12,} ({pct:>5.2f}%)")
    print(f"Frozen (backbone):    {frozen:>12,} ({100-pct:>5.2f}%)")
    print(f"{'='*50}\n")
```

### Backward Compatibility
- ✅ Default `model_variant='large'` preserves existing behavior
- ✅ Old code calling `RETFoundLoRA(...)` without variant parameter works unchanged
- ✅ New code can specify variant: `RETFoundLoRA(..., model_variant='green')`

---

## Implementation Checklist

### Code Implementation
- [ ] Add imports for timm in `retfound_model.py`
- [ ] Implement `get_retfound_green()` function
- [ ] Implement `load_retfound_green_model()` function with classification head
- [ ] Implement `detect_model_variant()` helper function
- [ ] Add test code to `__main__` block
- [ ] Update `RETFoundLoRA.__init__()` with `model_variant` parameter
- [ ] Add logging to show trainable parameter counts
- [ ] Add utility methods for parameter summaries

### Testing
- [ ] Create `tests/test_retfound_green.py` with:
  - [ ] Test `get_retfound_green()` creates correct architecture
  - [ ] Test output shape: (batch, 384)
  - [ ] Test parameter count: ~21.3M
  - [ ] Test `RETFoundLoRA` with variant='green'
  - [ ] Test LoRA adapters apply correctly
  - [ ] Test trainable param count (~800K)
  - [ ] Test forward pass returns logits

### Validation
- [ ] Run `python scripts/retfound_model.py` - should complete without errors
- [ ] Run `pytest tests/test_retfound_green.py -v`
- [ ] Verify old `RETFoundLoRA(...)` still works (variant='large' default)
- [ ] Verify new `RETFoundLoRA(..., model_variant='green')` works
- [ ] Check parameter counts match expectations:
  - Large: ~303M frozen, ~800K trainable
  - Green: ~21.3M frozen, ~800K trainable

### Performance Baseline
- [ ] Single forward pass time:
  - Large: ~100-200ms
  - Green: ~30-50ms (expected 3-4x faster)
- [ ] GPU memory (batch=1):
  - Large: ~4-5GB
  - Green: ~1.5-2GB
- [ ] Parameter count match documentation

---

## Code Location Reference

### Files Modified
1. `scripts/retfound_model.py`
   - Add ~150 lines of new code
   - No modifications to existing code
   - Location: After existing `load_retfound_model()` function

2. `scripts/retfound_lora.py`
   - Modify `__init__()` method (+30 lines)
   - Add utility methods (~15 lines)
   - Location: In RETFoundLoRA class definition

### Test Files to Create
1. `tests/test_retfound_green.py` (~100 lines)
   - Model loading tests
   - Output shape validation
   - Parameter count verification
   - LoRA adapter tests

---

## Dependency Management

### External Dependencies
- `timm` (already in requirements.txt)
- `torch` (already in requirements.txt)
- `peft` (already in requirements.txt for LoRA)

### Internal Dependencies
- `scripts/retfound_model.py` (loaded by `train_retfound_lora.py`)
- `scripts/retfound_lora.py` (used in training scripts)
- `scripts/config.py` (will be updated in Phase 2)

---

## Potential Issues & Solutions

### Issue 1: timm Model Not Available
**Problem**: `vit_small_patch14_reg4_dinov2` not found in timm

**Solution**:
```bash
pip install timm --upgrade
# If still missing, may need to use older architecture name
# Check timm docs or RETFound_Green repo for correct model name
```

### Issue 2: Checkpoint Loading Fails
**Problem**: State dict keys don't match model architecture

**Solution**:
```python
# Use strict=False to load despite key mismatches
state_dict = torch.load(...)
model.load_state_dict(state_dict, strict=False)  # Allows missing keys
```

### Issue 3: LoRA Target Modules Not Found
**Problem**: `["qkv"]` targets don't match timm ViT layer names

**Solution**:
```python
# Print available module names for debugging
for name, module in model.named_modules():
    print(name)

# Use more specific pattern if needed
target_modules = [".*qkv"]  # Regex pattern
```

### Issue 4: Feature Dimension Mismatch
**Problem**: Output features are not 384-dimensional

**Solution**:
```python
# Verify embedding dimension
features = backbone(torch.randn(1, 3, 392, 392))
print(features.shape)  # Should be [1, 384]
assert features.shape[-1] == 384
```

---

## Success Criteria for Phase 1

When this phase is complete, you should be able to:

1. ✅ Import and create `RETFound_Green` model
   ```python
   from scripts.retfound_model import get_retfound_green
   model = get_retfound_green()
   ```

2. ✅ Load pretrained weights
   ```python
   from scripts.retfound_model import load_retfound_green_model
   model = load_retfound_green_model('models/retfoundgreen_statedict.pth')
   ```

3. ✅ Extract 384-dimensional features
   ```python
   x = torch.randn(2, 3, 392, 392)
   features = model.extract_features(x)
   assert features.shape == (2, 384)
   ```

4. ✅ Use with LoRA
   ```python
   from scripts.retfound_lora import RETFoundLoRA
   lora_model = RETFoundLoRA(
       checkpoint_path='models/retfoundgreen_statedict.pth',
       model_variant='green'
   )
   ```

5. ✅ Verify trainable parameters
   ```python
   trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
   # Should be ~800K regardless of variant
   ```

6. ✅ Run unit tests
   ```bash
   pytest tests/test_retfound_green.py -v
   ```

---

## Transition to Phase 2

Once Phase 1 is complete and validated, proceed to Phase 2 which will:
- Add `--model_variant` argument to training script
- Update config system to handle both models
- Implement variant-aware transforms
- Full end-to-end training support

See `02-phase2-training-pipeline.md` for details.
