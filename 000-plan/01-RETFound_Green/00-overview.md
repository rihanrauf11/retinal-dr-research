# RETFound_Green Migration: Overview & Strategy

## Executive Summary

This document outlines the migration strategy to add **RETFound_Green** (ViT-Small, 21.3M params) support to the diabetic retinopathy classification project while maintaining full backward compatibility with the existing **RETFound** (ViT-Large, 303M params) implementation.

**Approach**: **Parallel Support** - Both models available simultaneously, user chooses via configuration.

**Timeline**: ~2 weeks (5-8 days implementation + 2-3 days testing)

**Risk Level**: **LOW** - All changes are additive, zero breaking changes

---

## Model Comparison

| Aspect | RETFound (Large) | RETFound_Green (Small) |
|--------|------------------|----------------------|
| **Base Architecture** | Custom ViT-Large | timm ViT-Small |
| **Total Parameters** | 303M | 21.3M |
| **Patch Size** | 16×16 | 14×14 |
| **Embedding Dimension** | 1024 | 384 |
| **Transformer Depth** | 24 blocks | 12 blocks |
| **Attention Heads** | 16 | 6 |
| **Input Size** | 224×224 | 392×392 |
| **Normalization** | ImageNet (0.485, 0.456, 0.406) | Custom (0.5, 0.5, 0.5) |
| **Weights Source** | Custom checkpoint | timm + GitHub release |
| **Training Data** | 1.6M images | 75K images |
| **Training Method** | Masked Autoencoding (MAE) | Token Reconstruction |
| **GPU Memory (batch=32)** | ~11-12GB | ~6-8GB |
| **Training Time per epoch** | ~10-15 min | ~3-5 min |
| **Downstream Output** | 384D embeddings | 384D embeddings |
| **Cross-Dataset Perf** | ✅ Proven | ⏳ To be validated |

### Key Insight
Both models produce **384-dimensional feature embeddings**, making the downstream LoRA adaptation layer compatible with minimal changes.

---

## Why RETFound_Green?

### Advantages
1. **Computational Efficiency**: 400× less compute to train, 50× less data needed
2. **Memory Efficient**: ~6-8GB GPU vs 11-12GB for Large
3. **Faster Inference**: Smaller model for deployment scenarios
4. **Faster Training**: ~3-5 min/epoch vs 10-15 min/epoch
5. **Comparable Performance**: Achieves competitive accuracy on BRSET, IDRiD, ROP benchmarks
6. **Research Value**: Validates LoRA's effectiveness across model scales

### When to Use Each
| Use Case | Model |
|----------|-------|
| Resource-constrained research | ✅ Green |
| Limited GPU memory (<8GB) | ✅ Green |
| Rapid prototyping | ✅ Green |
| Published benchmarks/comparison | ✅ Large (proven track record) |
| Hyperparameter optimization | ✅ Green (faster iteration) |
| Production deployment | ✅ Green (lower latency) |
| Cross-dataset generalization study | ✅ Both (compare) |

---

## Architecture: Parallel Support Design

```
┌─────────────────────────────────────────────────────────┐
│                   Configuration (YAML)                   │
│            model_variant: "large" | "green"             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Training Pipeline (Main Entry)              │
│         scripts/train_retfound_lora.py                  │
│         --model_variant [large|green]                   │
└─────────────────────────────────────────────────────────┘
                          ↓
                ┌─────────┴─────────┐
                ↓                   ↓
        ┌──────────────┐     ┌──────────────┐
        │ Large Path   │     │ Green Path   │
        │ (303M)       │     │ (21.3M)      │
        └──────────────┘     └──────────────┘
                │                   │
                ├─→ Load Backbone   ├─→ Load Backbone
                │   (custom ViT)    │   (timm ViT-Small)
                │                   │
                ├─→ ImageNet norm   ├─→ 0.5 norm
                │   224×224 input   │   392×392 input
                │                   │
                └─→ LoRA adapters   └─→ LoRA adapters
                    (embed_dim=1024)     (embed_dim=384)
                │                   │
                └─────────┬─────────┘
                          ↓
                ┌─────────────────┐
                │ Unified Training │
                │ Validation       │
                │ Evaluation       │
                └─────────────────┘
```

### Key Design Principles

1. **Variant Agnostic Downstream**: Both variants feed 384D embeddings to LoRA classifier
2. **Config-Driven**: All differences handled via config, minimal code branching
3. **Backward Default**: `model_variant="large"` is the default (no behavior change)
4. **Checkpoint Aware**: Variant stored in checkpoint metadata for correct loading
5. **Auto-Detection**: Evaluation scripts auto-detect variant from checkpoint

---

## Implementation Phases

### Phase 1: Core Model Infrastructure (2-3 days)
**Files**: `scripts/retfound_model.py`, `scripts/retfound_lora.py`

Add functions to create and load RETFound_Green models via timm without modifying existing Large code.

**Deliverables**:
- `get_retfound_green()` function
- `load_retfound_green_model()` function
- Updated `RETFoundLoRA.__init__()` with `model_variant` parameter
- Unit tests for Green model loading

**Risk**: LOW

---

### Phase 2: Training Pipeline (1-2 days)
**Files**: `scripts/train_retfound_lora.py`, `scripts/config.py`

Add variant parameter to config system and training script with proper transform handling.

**Deliverables**:
- `--model_variant` CLI argument
- `model_variant` field in config dataclass
- Variant-aware transform generation (normalization + size)
- Proper initialization based on variant

**Risk**: LOW

---

### Phase 3: Evaluation & Optimization (1-2 days)
**Files**: `scripts/evaluate_cross_dataset.py`, `scripts/hyperparameter_search.py`

Update evaluation to detect and handle both variants correctly.

**Deliverables**:
- Checkpoint variant auto-detection
- Variant-aware transform loading
- Hyperparameter search with variant support
- Checkpoint metadata including variant

**Risk**: MEDIUM (variant detection must be robust)

---

### Phase 4: Configuration & Documentation (1 day)
**Files**: `configs/`, `docs/`

Create new config template and update all documentation.

**Deliverables**:
- `configs/retfound_green_lora_config.yaml`
- Updated `configs/retfound_lora_config.yaml`
- Updated guides and README

**Risk**: LOW

---

## Critical Success Factors

### Before Implementation
- [ ] Download RETFound_Green weights: https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
- [ ] Verify timm is installed and can load `vit_small_patch14_reg4_dinov2`
- [ ] Understand current RETFound loading logic
- [ ] Have test APTOS dataset ready (APTOS only, no cross-dataset yet)

### During Implementation
- [ ] Validate transforms match model expectations
- [ ] Test LoRA loading doesn't crash (PEFT target module compatibility)
- [ ] Verify checkpoint saving/loading with variant metadata
- [ ] Check memory usage on target GPU
- [ ] Validate accuracy baseline (>70% on APTOS)

### After Implementation
- [ ] Run cross-dataset evaluation with Green variant
- [ ] Compare Green vs Large performance on same data
- [ ] Document performance characteristics
- [ ] Validate backward compatibility (old Large checkpoints still load)

---

## Decision Rationale: Why Parallel Support?

### Option A: Parallel Support (CHOSEN ✅)
**Pros**:
- ✅ Zero breaking changes
- ✅ Backward compatible with existing checkpoints
- ✅ Enables research comparison (Large vs Green)
- ✅ Gradual migration path
- ✅ Users choose model based on constraints
- ✅ Can run A/B tests

**Cons**:
- ✗ Slightly more code complexity
- ✗ Two checkpoints to manage

### Option B: Full Migration (NOT CHOSEN)
**Pros**:
- ✅ Simpler codebase
- ✅ Lower memory by default
- ✅ One model to maintain

**Cons**:
- ✗ Breaking change for existing workflows
- ✗ Requires re-training all models
- ✗ Research reproducibility compromised
- ✗ Users lose Large model option

**Verdict**: Parallel Support provides maximum flexibility and research value with minimal risk.

---

## Backward Compatibility Guarantee

All existing workflows will continue unchanged:

```bash
# This command still works exactly as before (uses Large)
python3 scripts/train_retfound_lora.py \
    --checkpoint_path models/RETFound_cfp_weights.pth \
    --config configs/retfound_lora_config.yaml

# This is the new capability
python3 scripts/train_retfound_lora.py \
    --checkpoint_path models/retfoundgreen_statedict.pth \
    --config configs/retfound_green_lora_config.yaml \
    --model_variant green
```

Checkpoint format compatibility:
- Old checkpoints (without `variant` metadata) → Assume "large"
- New checkpoints → Include variant for auto-detection
- Loading is version-aware and handles both formats

---

## File Summary

### Modified Files (7)
1. `scripts/retfound_model.py` - Add Green loading functions
2. `scripts/retfound_lora.py` - Add model_variant parameter
3. `scripts/config.py` - Add model_variant field
4. `scripts/train_retfound_lora.py` - Add CLI argument + transforms
5. `scripts/evaluate_cross_dataset.py` - Add variant detection
6. `configs/retfound_lora_config.yaml` - Add model_variant field
7. Various docs files - Update architecture descriptions

### New Files (2)
1. `configs/retfound_green_lora_config.yaml` - Template config
2. Phase documentation in `000-plan/` - Implementation guide

### Total Code Changes
~700 lines of Python + 200 lines of docs + 170 lines of config

---

## Success Metrics

After implementation, we will measure:

1. **Correctness**:
   - RETFound_Green trains without errors
   - LoRA adapters apply and are trainable
   - Checkpoints save/resume correctly

2. **Performance**:
   - Training accuracy >70% on APTOS (both variants)
   - Inference speed: Green ~2-3x faster than Large
   - Memory usage: Green uses ~35-40% of Large memory

3. **Compatibility**:
   - Old Large checkpoints still load
   - Cross-dataset evaluation works with both variants
   - Backward compatibility 100%

4. **Usability**:
   - Clear documentation for variant selection
   - Easy switching between models via config
   - Auto-detection in evaluation scripts

---

## Next Steps

1. **Review this overview** - Confirm approach and timeline
2. **Read Phase 1 document** - Understand core changes
3. **Download RETFound_Green weights** - Required for implementation
4. **Implement Phase 1** - Start with model loading infrastructure
5. **Validate Phase 1** - Unit tests pass, model loads
6. **Continue through phases** - Iterate through 2-4
7. **Integration testing** - Run full training pipeline
8. **Cross-dataset validation** - Confirm generalization works

---

## Resources & References

### RETFound_Green Repository
- GitHub: https://github.com/justinengelmann/RETFound_Green
- Weights: https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
- Model: timm's `vit_small_patch14_reg4_dinov2`

### Related Documentation
- PEFT LoRA: https://huggingface.co/docs/peft/
- timm ViT Models: https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py
- Original RETFound: https://github.com/rmaphoh/RETFound_MAE

### Project Context
- Current RETFound implementation: `scripts/retfound_model.py`
- Current LoRA integration: `scripts/retfound_lora.py`
- Current training pipeline: `scripts/train_retfound_lora.py`
- Current configs: `configs/retfound_lora_config.yaml`
