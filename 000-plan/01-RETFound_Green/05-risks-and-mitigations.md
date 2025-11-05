# Phase 5: Risks & Mitigations

## Overview

**Objective**: Identify and document all risks associated with RETFound_Green integration and provide mitigation strategies.

**Scope**: Technical risks, data risks, and integration risks

**Risk Assessment**: Comprehensive analysis across all implementation phases

---

## Critical Risk Analysis

### Risk 1: Normalization Mismatch

**Severity**: 🔴 CRITICAL

**Description**: Using incorrect normalization (mean/std) for a model variant causes severe accuracy degradation.

**Scenario**:
- Config specifies `model_variant: green` but uses ImageNet normalization
- Model trained on mean=0.5 fails when fed ImageNet-normalized images
- Accuracy drops 20-40%

**Root Causes**:
- RETFound Large: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- RETFound_Green: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- Config system might not properly update ImageConfig based on variant

**Detection**:
- Early validation accuracy is very low (<50%)
- Loss stops improving after first epoch
- Feature distributions are out of range

**Mitigation**:

1. **Config Validation** (Primary)
   ```python
   def validate_normalization(config: Config) -> None:
       """Validate normalization matches variant."""
       if config.model.model_variant == 'large':
           assert config.image.mean == [0.485, 0.456, 0.406]
           assert config.image.std == [0.229, 0.224, 0.225]
       elif config.model.model_variant == 'green':
           assert config.image.mean == [0.5, 0.5, 0.5]
           assert config.image.std == [0.5, 0.5, 0.5]
       else:
           raise ValueError(f"Unknown variant: {config.model.model_variant}")
   ```

2. **Automatic Update** (Fallback)
   - Call `config.image.update_for_variant()` before training
   - Automatically sets correct normalization

3. **Logging & Warnings**
   - Log actual normalization values before training
   - Warn if normalization doesn't match detected variant

4. **Testing**
   - Unit tests verify normalization is correct for each variant
   - Forward pass tests check output range is reasonable

---

### Risk 2: Input Size Mismatch

**Severity**: 🔴 CRITICAL

**Description**: Feeding wrong image size causes model failure or severe accuracy loss.

**Scenario**:
- Config specifies `model_variant: green` (expects 392×392)
- Images are resized to 224×224 (from old config)
- Model processes images with incorrect aspect ratio/content loss
- Results are unreliable

**Root Causes**:
- RETFound Large: 224×224 input
- RETFound_Green: 392×392 input
- User copies old config and forgets to update image size

**Detection**:
- Training accuracy much lower than baseline
- Validation accuracy plateau at low value
- Model not learning meaningful features

**Mitigation**:

1. **Config Validation**
   ```python
   def validate_image_size(config: Config) -> None:
       """Validate image size matches variant."""
       if config.model.model_variant == 'large':
           assert config.image.input_size == 224, \
               f"Large variant expects size=224, got {config.image.input_size}"
       elif config.model.model_variant == 'green':
           assert config.image.input_size == 392, \
               f"Green variant expects size=392, got {config.image.input_size}"
   ```

2. **Automatic Update**
   - `config.image.update_for_variant()` sets correct size

3. **Explicit Documentation**
   - Config templates clearly state image size for each variant
   - Comments warn about size importance

4. **Tests**
   - Unit test: forward pass with correct size works
   - Unit test: forward pass with wrong size fails gracefully

---

### Risk 3: Checkpoint Loading Failure

**Severity**: 🟠 HIGH

**Description**: Cannot load checkpoint due to variant mismatch or weight format issues.

**Scenario**:
- User has a RETFound_Green checkpoint from another training run
- Tries to load with `RETFoundLoRA(..., model_variant='large')`
- Model architecture doesn't match checkpoint weights
- Training fails immediately or runs but doesn't converge

**Root Causes**:
- User specifies wrong variant when loading
- Checkpoint format changed between versions
- Checkpoint corrupted or from different training

**Detection**:
- Model loading raises exception
- Shape mismatch errors during load_state_dict()
- Training runs but metrics are nonsensical

**Mitigation**:

1. **Auto-Detection** (Primary)
   ```python
   # Auto-detect variant from checkpoint before loading
   variant = detect_model_variant(checkpoint_path)
   model = RETFoundLoRA(..., model_variant=variant)
   ```

2. **Metadata Storage**
   - Save variant in checkpoint metadata
   - Store embed_dim and other model specs
   - Include full config for reproducibility

3. **Error Handling**
   ```python
   try:
       model.load_state_dict(checkpoint['model_state_dict'], strict=True)
   except RuntimeError as e:
       if 'shape' in str(e):
           logger.error(f"Shape mismatch - variant might be wrong: {e}")
           raise ValueError(
               f"Cannot load checkpoint. Detected variant: {variant}. "
               f"If this is wrong, specify --model_variant explicitly."
           )
       raise
   ```

4. **Manual Override**
   - Allow `--model_variant` CLI flag to override detection
   - Useful for edge cases or corrupted metadata

---

### Risk 4: LoRA Target Module Mismatch

**Severity**: 🟠 HIGH

**Description**: LoRA adapters don't apply to the correct layers in different model variants.

**Scenario**:
- LoRA targets `["qkv"]` modules
- RETFound Large and Green have different layer names
- LoRA doesn't apply correctly
- Model becomes trainable with 300M parameters instead of adapters only

**Root Causes**:
- timm and custom ViT implementations have different naming conventions
- PEFT's `get_peft_model()` can't find target modules

**Detection**:
- Trainable parameter count is wrong (300M instead of 800K)
- Training is very slow (not using adapters)
- Memory usage doesn't match expectations

**Mitigation**:

1. **Debug Logging**
   ```python
   # Print all module names
   for name, module in backbone.named_modules():
       print(f"{name}: {type(module).__name__}")

   # Verify LoRA applied
   trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
   logger.info(f"Trainable parameters: {trainable:,}")
   assert trainable < 1_000_000, "Too many trainable params - LoRA not applied?"
   ```

2. **Flexible Target Pattern**
   ```python
   # Use regex patterns instead of exact names
   target_modules = [".*qkv.*"] if flexible else ["qkv"]
   ```

3. **Validation Tests**
   ```python
   def test_lora_applied():
       model = RETFoundLoRA(..., model_variant='green')
       trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
       assert 500_000 < trainable < 1_500_000, f"Got {trainable}"
   ```

4. **Fallback**
   - If target modules not found, print warning and list available modules
   - Let user know to adjust configuration

---

### Risk 5: Performance Degradation of Green

**Severity**: 🟡 MEDIUM

**Description**: RETFound_Green may not achieve expected accuracy on cross-dataset tasks.

**Scenario**:
- User trains RETFound_Green expecting similar performance to Large
- Cross-dataset accuracy gap is larger (5% instead of 3%)
- Results are disappointing

**Root Causes**:
- Smaller model capacity for complex patterns
- Less training data (75K vs 1.6M)
- Different training objective (token reconstruction vs MAE)
- Haven't optimized hyperparameters for Green

**Detection**:
- Cross-dataset accuracy drop is 5-7% instead of expected 3-4%
- Model reaches plateau earlier than Large
- Validation overfitting is more pronounced

**Mitigation**:

1. **Hyperparameter Optimization**
   - Run Optuna search specifically for Green variant
   - May need different learning rates, batch sizes, LoRA ranks
   - Set variant-specific search spaces

2. **Documentation**
   - Clearly document expected performance of each variant
   - Set realistic expectations
   - Show performance comparison table

3. **Augmentation**
   - Green may benefit from stronger augmentation
   - Larger input size (392×392) helps with context
   - Experiment with augmentation intensity

4. **Training Adjustments**
   - Longer training (more epochs) might help
   - Different learning rate scheduling
   - Ensemble or knowledge distillation from Large

---

### Risk 6: Backward Compatibility Break

**Severity**: 🟡 MEDIUM

**Description**: Existing code/configs/checkpoints break when RETFound_Green is integrated.

**Scenario**:
- User has checkpoint from old codebase
- New code requires `model_variant` field in config
- Old checkpoint doesn't have variant metadata
- Training with old config fails

**Root Causes**:
- Default assumptions change
- Required fields added to config
- Checkpoint format changes

**Detection**:
- Old training commands fail
- Old checkpoints can't be loaded
- User gets cryptic error messages

**Mitigation**:

1. **Default Values**
   - `model_variant = 'large'` preserves old behavior
   - Missing config fields get sensible defaults
   - Old checkpoints assume `variant = 'large'`

2. **Graceful Degradation**
   ```python
   def load_config_with_defaults(config_file):
       config = load_yaml(config_file)
       # Set defaults if missing
       if 'model_variant' not in config.get('model', {}):
           config['model']['model_variant'] = 'large'
           logger.info("Using default model_variant='large'")
       return config
   ```

3. **Migration Guide**
   - Document how to migrate old checkpoints
   - Provide script to add variant to old checkpoints
   - Clear upgrade path

4. **Testing**
   - Unit test: old config loads correctly
   - Unit test: old checkpoint can be loaded
   - Integration test: old training pipeline works

---

### Risk 7: Download/Installation Issues

**Severity**: 🟡 MEDIUM

**Description**: RETFound_Green weights fail to download or installation has missing dependencies.

**Scenario**:
- User tries to download RETFound_Green weights
- GitHub releases API is slow/unreliable
- Weight file corrupts during download
- User can't find pretrained weights

**Root Causes**:
- Network issues
- GitHub API limits
- File corruption
- User confusion about where to download

**Detection**:
- Weights file doesn't exist at expected path
- File size is wrong (corrupted download)
- Training fails with "checkpoint not found" error

**Mitigation**:

1. **Clear Documentation**
   - Explicit download instructions in all relevant docs
   - Step-by-step guide with example commands
   - Multiple ways to obtain weights (wget, curl, browser download)

2. **Validation Script**
   ```python
   def validate_weights(checkpoint_path, expected_size=None):
       """Check weights file exists and is valid."""
       if not Path(checkpoint_path).exists():
           raise FileNotFoundError(
               f"Weights not found: {checkpoint_path}\n"
               f"Download from: https://github.com/justinengelmann/RETFound_Green/"
               f"releases/download/v0.1/retfoundgreen_statedict.pth"
           )

       size = Path(checkpoint_path).stat().st_size
       if expected_size and abs(size - expected_size) > 1_000_000:
           logger.warning(f"File size unexpected. File might be corrupted.")
   ```

3. **Fallback Options**
   - Provide alternative download sources if possible
   - Cache weights locally for team/lab
   - Use pre-computed features as fallback

4. **Initialization Script**
   ```bash
   #!/bin/bash
   # scripts/download_weights.sh

   mkdir -p models
   echo "Downloading RETFound_Green weights..."
   wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth \
       -O models/retfoundgreen_statedict.pth

   echo "Validating file..."
   python3 -c "
   import torch
   weights = torch.load('models/retfoundgreen_statedict.pth', map_location='cpu')
   print(f'Loaded weights: {len(weights)} keys')
   print('✓ Weights validated successfully')
   "
   ```

---

### Risk 8: Memory Management Issues

**Severity**: 🟡 MEDIUM

**Description**: GPU memory issues during training with incorrect batch sizes or configurations.

**Scenario**:
- User copies Large config and sets `batch_size=64`
- Training with Green variant hits OOM error
- Model crashes mid-epoch
- Training data is lost

**Root Causes**:
- Green model uses less memory, user assumes larger batches are safe
- Normalization is different, might affect tensor shapes
- Config not validated for feasibility on target hardware

**Detection**:
- CUDA OOM error during forward pass
- Training crashes unpredictably
- Memory usage spikes suddenly

**Mitigation**:

1. **Config-Based Limits**
   ```python
   MAX_BATCH_SIZES = {
       'large': 32,  # ~12GB at batch=32
       'green': 64,  # ~8GB at batch=64
   }

   def validate_memory_config(config):
       max_batch = MAX_BATCH_SIZES.get(config.model.model_variant)
       if config.training.batch_size > max_batch:
           logger.warning(
               f"Batch size {config.training.batch_size} might exceed memory "
               f"for {config.model.model_variant} (recommended max: {max_batch})"
           )
   ```

2. **Memory Profiling**
   - Run short validation pass to estimate memory
   - Warn user before full training starts
   - Suggest batch size adjustments

3. **Gradient Accumulation**
   - Use gradient accumulation to simulate larger batches
   - Less memory, same effective batch size

4. **Mixed Precision**
   - Enable mixed precision training
   - Reduces memory usage by ~50%

---

### Risk 9: Convergence Issues with Green

**Severity**: 🟡 MEDIUM

**Description**: RETFound_Green training doesn't converge or overfits differently than Large.

**Scenario**:
- User trains Green with Large's hyperparameters
- Model plateaus early or diverges
- Results are poor
- User blames the model instead of hyperparameters

**Root Causes**:
- Different architecture (12 vs 24 layers) requires different LR
- Smaller model capacity leads to different learning dynamics
- LoRA might need different rank for small models

**Detection**:
- Validation loss increases after initial improvement
- Training accuracy plateaus at 70-75%
- Huge gap between training and validation loss

**Mitigation**:

1. **Variant-Specific Hyperparameters**
   - Different configs for Large and Green
   - Include recommended ranges in comments
   - Document defaults for each variant

2. **Learning Rate Scheduling**
   - Green may need higher LR (more aggressive updates)
   - Shorter warmup period
   - Experiment with schedules

3. **Hyperparameter Search**
   - Run Optuna search for Green variant
   - Use variant-specific search space
   - Document best parameters

4. **Monitoring & Early Stopping**
   - Monitor training/validation loss ratio
   - Early stop if overfitting detected
   - Log learning rate during training

---

### Risk 10: Integration Testing Gaps

**Severity**: 🟡 MEDIUM

**Description**: Edge cases or integration issues are not caught before deployment.

**Scenario**:
- Phase 1-3 tests pass individually
- Integration test with all phases together fails
- Edge case: old Large checkpoint + Green config
- Bug only manifests in full pipeline

**Root Causes**:
- Unit tests don't cover all combinations
- Integration between phases not tested
- Edge cases not anticipated

**Detection**:
- Integration test fails mysteriously
- One phase works but breaks another
- Behavior changes in unexpected ways

**Mitigation**:

1. **Comprehensive Integration Tests**
   ```python
   def test_full_training_green():
       """End-to-end test with Green variant."""
       # Setup
       config = load_config('configs/retfound_green_lora_config.yaml')

       # Train
       model = RETFoundLoRA(..., model_variant='green')
       train_one_epoch(model)
       checkpoint = save_checkpoint(model, ...)

       # Load and evaluate
       loaded = load_checkpoint(checkpoint)
       evaluate_cross_dataset(loaded, ...)

   def test_backward_compatibility():
       """Old Large checkpoint still works."""
       old_checkpoint = create_old_format_checkpoint()
       loaded = load_checkpoint(old_checkpoint)
       assert loaded.model_variant == 'large'  # Auto-detected
   ```

2. **Test All Combinations**
   - Large config + Large checkpoint
   - Green config + Green checkpoint
   - Cross-variant edge cases
   - With/without variant metadata

3. **CI/CD Pipeline**
   - Run integration tests on every commit
   - Catch regressions early
   - Test on multiple GPU configurations

---

## Risk Matrix

| Risk | Severity | Probability | Mitigation | Testing |
|------|----------|-------------|-----------|---------|
| Normalization Mismatch | Critical | High | Config validation, auto-update | Unit + integration |
| Input Size Mismatch | Critical | High | Config validation, auto-update | Unit + integration |
| Checkpoint Loading | High | Medium | Auto-detect, metadata, error handling | Unit + integration |
| LoRA Modules | High | Medium | Debug logging, validation tests | Unit |
| Green Performance | Medium | Medium | Hyperparameter tuning, documentation | Integration |
| Backward Compatibility | Medium | Low | Defaults, migration guide | Integration |
| Downloads/Installation | Medium | Low | Documentation, validation script | Manual testing |
| Memory Issues | Medium | Medium | Config limits, profiling, warnings | Integration |
| Convergence | Medium | Medium | Variant-specific params, monitoring | Integration |
| Integration Gaps | Medium | Low | Comprehensive tests, CI/CD | Integration |

---

## Testing Strategy

### Unit Tests (Phase-Specific)
- Phase 1: Model loading, shape validation, parameter counts
- Phase 2: Config loading, transform correctness, backward compatibility
- Phase 3: Variant detection, checkpoint loading, evaluation
- Phase 4: Configuration validity, documentation accuracy

### Integration Tests
- Full training pipeline with both variants
- Cross-dataset evaluation
- Checkpoint save and resume
- Old checkpoint compatibility
- Mixed variant scenarios

### Stress Tests
- Large batch sizes near memory limit
- Extended training (check for memory leaks)
- Rapid checkpoint save/load cycles
- Parallel data loading

---

## Contingency Plan

If critical issues arise during implementation:

1. **Normalization Issue**
   - Revert to hardcoded transforms
   - Remove auto-update, require explicit config
   - Add extensive validation

2. **LoRA Not Applying**
   - Print all module names for debugging
   - Implement custom LoRA attachment
   - Fall back to full fine-tuning as escape hatch

3. **Performance Below Expectations**
   - Document the performance gap
   - Suggest Large variant for mission-critical
   - Run hyperparameter optimization campaign

4. **Backward Compatibility Broken**
   - Implement migration script
   - Support old checkpoint format indefinitely
   - Provide clear upgrade documentation

---

## Monitoring & Validation

After deployment, monitor:

1. **User Issues**
   - Track GitHub issues related to variants
   - Monitor accuracy reports from users
   - Collect feedback on usability

2. **Performance Metrics**
   - Compare Large vs Green accuracy in the wild
   - Track memory usage patterns
   - Monitor inference latency

3. **Reliability**
   - Track checkpoint loading failures
   - Monitor training convergence rates
   - Check for memory leaks

4. **Documentation Quality**
   - User success rate with getting started guide
   - Support requests for clarification
   - Quality of user feedback

---

## Success Criteria for Risk Mitigation

Phase 5 is complete when:

- [ ] All critical risks have documented mitigations
- [ ] Unit tests cover all identified risks
- [ ] Integration tests verify mitigations work
- [ ] Backward compatibility is proven
- [ ] Edge cases are handled gracefully
- [ ] Error messages are helpful
- [ ] Documentation is clear and complete
- [ ] Contingency plans are documented
- [ ] Monitoring strategy is in place

See `06-validation-checklist.md` for detailed validation procedures.
