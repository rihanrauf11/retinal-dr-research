# Testing Implementation Plan - Overview

**Project:** Diabetic Retinopathy Classification with RETFound + LoRA
**Purpose:** Comprehensive pre-training validation to catch issues before expensive GPU training runs
**Created:** 2025-11-04
**Status:** Planning Phase

---

## Executive Summary

This testing plan establishes a multi-phase validation strategy to ensure training pipelines work correctly before committing to expensive GPU training runs. The plan builds on existing test infrastructure (287 tests, 5,376 lines) while filling critical gaps in pre-training validation.

**Key Goals:**
1. **Catch errors early** - Find issues in minutes, not hours into training
2. **Predict resource needs** - Know GPU memory requirements before OOM crashes
3. **Verify learning capability** - Confirm model can learn before full training
4. **Optimize performance** - Identify data loading bottlenecks
5. **Visual debugging** - Inspect data augmentation and model predictions

---

## Implementation Phases

### Phase 1: Sanity Tests (Priority: **CRITICAL**)
**File:** [01-SANITY_TESTS.md](01-SANITY_TESTS.md)
**Time Estimate:** 3-4 hours
**Dependencies:** None

Quick smoke tests to verify basic training loop functionality:
- Single forward pass (model can process input)
- Single backward pass (gradients flow correctly)
- Loss computation (loss decreases on same batch)
- Configuration validation (all paths exist, parameters valid)

**Deliverable:** `scripts/test_sanity.py` - runs in < 2 minutes

---

### Phase 2: GPU Memory Tests (Priority: **CRITICAL**)
**File:** [02-GPU_MEMORY_TESTS.md](02-GPU_MEMORY_TESTS.md)
**Time Estimate:** 4-5 hours
**Dependencies:** Phase 1 complete

Memory profiling to predict OOM before training:
- Peak memory usage measurement
- Batch size recommendations
- RETFound Large vs Green comparison
- LoRA memory savings quantification

**Deliverable:** `scripts/test_gpu_memory.py` - provides memory budget report

---

### Phase 3: Overfitting Tests (Priority: **HIGH**)
**File:** [03-OVERFITTING_TESTS.md](03-OVERFITTING_TESTS.md)
**Time Estimate:** 3-4 hours
**Dependencies:** Phase 1 complete

Intentional overfitting on tiny dataset to verify learning:
- 10-sample dataset creation
- Train to 100% accuracy
- Confirm loss convergence
- Validate optimizer/scheduler work

**Deliverable:** `scripts/test_overfitting.py` - trains in < 5 minutes

---

### Phase 4: Data Loading Tests (Priority: **MEDIUM**)
**File:** [04-DATA_LOADING_TESTS.md](04-DATA_LOADING_TESTS.md)
**Time Estimate:** 4-5 hours
**Dependencies:** Phase 1 complete

Performance profiling of data pipeline:
- DataLoader bottleneck detection
- Augmentation timing analysis
- Multi-worker optimization
- I/O vs GPU utilization

**Deliverable:** `scripts/test_data_loading.py` - identifies bottlenecks

---

### Phase 5: Visual Inspection (Priority: **MEDIUM**)
**File:** [05-VISUAL_INSPECTION.md](05-VISUAL_INSPECTION.md)
**Time Estimate:** 5-6 hours
**Dependencies:** Phase 1 complete

Visual debugging tools for data and predictions:
- Augmentation visualization grids
- Sample prediction displays
- Attention map visualization (ViT models)
- Training curve plotting

**Deliverable:** `scripts/visualize_pipeline.py` - generates HTML reports

---

### Phase 6: Pre-Flight Orchestrator (Priority: **LOW**)
**File:** [06-PREFLIGHT_ORCHESTRATOR.md](06-PREFLIGHT_ORCHESTRATOR.md)
**Time Estimate:** 4-5 hours
**Dependencies:** Phases 1-5 complete

Master test runner for integrated validation:
- Orchestrates all test phases
- Parallel execution where possible
- HTML report generation
- Pass/fail criteria with exit codes

**Deliverable:** `scripts/preflight_check.py` - one command to rule them all

---

## Timeline and Priorities

### Critical Path (Must Have Before Training)
**Week 1:**
- Day 1-2: Phase 1 (Sanity Tests)
- Day 3-4: Phase 2 (GPU Memory Tests)
- Day 5: Phase 3 (Overfitting Tests)

**Total:** 5 days, ~15 hours implementation

### Extended Validation (Nice to Have)
**Week 2:**
- Day 1-2: Phase 4 (Data Loading Tests)
- Day 3-4: Phase 5 (Visual Inspection)

**Week 3:**
- Day 1-2: Phase 6 (Pre-Flight Orchestrator)

**Total:** 10 days, ~30 hours for complete suite

### Recommended Approach
**Option A - Rapid Start (Minimum Viable Testing):**
- Implement Phases 1-3 only (Critical path)
- Start training with confidence
- Add Phases 4-6 iteratively as needed

**Option B - Comprehensive (Full Testing Suite):**
- Implement all phases before first training run
- Maximum confidence, but delays training by 2-3 weeks

**Recommendation:** **Option A** - Get critical tests in place first, then iterate.

---

## Integration with Existing Infrastructure

### Current Testing Landscape
```
Existing Tests (287 tests, 5,376 lines):
├── tests/test_dataset.py       (43 tests - data loading)
├── tests/test_model.py          (52 tests - model architecture)
├── tests/test_transforms.py     (38 tests - augmentation)
├── tests/test_retfound_lora.py  (45 tests - LoRA integration)
├── tests/test_utils.py          (31 tests - utility functions)
├── tests/test_config.py         (24 tests - configuration)
├── tests/test_training_loop.py  (42 tests - training logic)
└── tests/test_integration.py    (12 tests - end-to-end)

Validation Scripts:
├── scripts/validate_all.py      (1,073 lines - comprehensive validation)
└── scripts/validate_data.py     (376 lines - data integrity)
```

### How New Tests Complement Existing Infrastructure

**Existing tests focus on:** Unit testing individual components
**New tests focus on:** Pre-training validation and performance profiling

**Relationship:**
```
pytest tests/           → Unit tests (verify components work)
scripts/validate_*.py   → Data validation (verify data integrity)
scripts/test_*.py       → Pre-training tests (verify training readiness) ← NEW
scripts/preflight_check.py → Orchestrator (run all pre-training tests) ← NEW
```

**Workflow Integration:**
```bash
# 1. Initial setup (one time)
pytest tests/                    # Verify codebase works (5 min)
python scripts/validate_data.py  # Verify data integrity (10 min)

# 2. Before each experiment (every time)
python scripts/preflight_check.py --config configs/my_experiment.yaml  # Pre-training validation (5 min)

# 3. Start training (with confidence)
python scripts/train_retfound_lora.py --config configs/my_experiment.yaml
```

---

## Success Criteria

### Phase 1 Success
- ✅ Can run single forward pass without errors
- ✅ Gradients flow through all trainable parameters
- ✅ Loss decreases on repeated batch training
- ✅ All paths in config exist and are valid

### Phase 2 Success
- ✅ Accurate peak memory measurement (±5% of actual)
- ✅ Batch size recommendations prevent OOM
- ✅ Memory comparison between RETFound variants
- ✅ LoRA memory savings quantified

### Phase 3 Success
- ✅ Model achieves 100% accuracy on 10-sample dataset
- ✅ Loss converges to near-zero (< 0.01)
- ✅ Training completes in < 5 minutes
- ✅ Confirms learning capability before expensive training

### Phase 4 Success
- ✅ Identifies data loading bottlenecks (if any)
- ✅ Recommends optimal num_workers setting
- ✅ Quantifies augmentation overhead
- ✅ Confirms GPU is not starved for data

### Phase 5 Success
- ✅ Generates augmentation visualization grids
- ✅ Shows sample predictions with confidence scores
- ✅ Visualizes attention maps for ViT models
- ✅ Plots training curves in real-time

### Phase 6 Success
- ✅ Runs all test phases in correct order
- ✅ Generates comprehensive HTML report
- ✅ Clear pass/fail status with exit codes
- ✅ Completes full check in < 10 minutes

---

## Quick Start Guide

### For Impatient Researchers (Fast Track)

**Step 1:** Implement Phase 1 (Sanity Tests)
```bash
# Create test script
vim scripts/test_sanity.py  # Use template from 01-SANITY_TESTS.md

# Run sanity check
python scripts/test_sanity.py --config configs/retfound_lora_config.yaml
```

**Step 2:** Implement Phase 2 (GPU Memory Tests)
```bash
# Create memory profiler
vim scripts/test_gpu_memory.py  # Use template from 02-GPU_MEMORY_TESTS.md

# Profile memory
python scripts/test_gpu_memory.py --config configs/retfound_lora_config.yaml
```

**Step 3:** Start Training (with confidence)
```bash
python scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
```

**Total time investment:** 6-8 hours implementation, 3-5 minutes per run

---

## File Organization

### Directory Structure
```
000-plan/02-Testing/
├── 00-OVERVIEW.md              ← You are here
├── 01-SANITY_TESTS.md          # Phase 1 implementation guide
├── 02-GPU_MEMORY_TESTS.md      # Phase 2 implementation guide
├── 03-OVERFITTING_TESTS.md     # Phase 3 implementation guide
├── 04-DATA_LOADING_TESTS.md    # Phase 4 implementation guide
├── 05-VISUAL_INSPECTION.md     # Phase 5 implementation guide
├── 06-PREFLIGHT_ORCHESTRATOR.md # Phase 6 implementation guide
└── 99-IMPLEMENTATION_NOTES.md  # Technical references and gotchas
```

### Generated Scripts (After Implementation)
```
scripts/
├── test_sanity.py              # Phase 1 deliverable
├── test_gpu_memory.py          # Phase 2 deliverable
├── test_overfitting.py         # Phase 3 deliverable
├── test_data_loading.py        # Phase 4 deliverable
├── visualize_pipeline.py       # Phase 5 deliverable
└── preflight_check.py          # Phase 6 deliverable
```

---

## Key Design Principles

### 1. **Fast Feedback**
All tests designed to complete in < 5 minutes (except overfitting test at ~5-10 min)

### 2. **Minimal Dependencies**
Each phase can be implemented independently (except Phase 6 which orchestrates all)

### 3. **Copy-Paste Ready**
Complete code examples in each phase file - no guessing required

### 4. **Research-Focused**
Designed for experimentation workflow, not production deployment

### 5. **Integration-Aware**
Works with existing pytest suite, validation scripts, and configuration system

### 6. **Progressive Enhancement**
Start with critical tests (Phases 1-3), add others as needed

---

## Risk Mitigation

### What Could Go Wrong Without These Tests?

**Without Phase 1 (Sanity Tests):**
- ❌ Discover model architecture errors 2 hours into training
- ❌ Find config typos after wasting GPU time
- ❌ Realize gradients aren't flowing after full epoch

**Without Phase 2 (GPU Memory Tests):**
- ❌ OOM crash 80% through first epoch
- ❌ Don't know if batch_size=32 will fit in memory
- ❌ Can't compare RETFound Large vs Green memory requirements

**Without Phase 3 (Overfitting Tests):**
- ❌ Train for hours only to find model can't learn
- ❌ Debug learning issues after expensive training
- ❌ Uncertainty if poor performance is model or data issue

**Without Phase 4 (Data Loading Tests):**
- ❌ GPU starved for data (low utilization)
- ❌ Don't know optimal num_workers setting
- ❌ Augmentation bottleneck slows training 2x

**Without Phase 5 (Visual Inspection):**
- ❌ Augmentation bugs go unnoticed (e.g., corrupted images)
- ❌ Can't visually verify model predictions
- ❌ Hard to debug why model makes certain errors

**Without Phase 6 (Pre-Flight Orchestrator):**
- ❌ Forget to run all validation checks
- ❌ Manual test execution prone to human error
- ❌ No standardized validation workflow

---

## Expected Outcomes

### Time Savings
**Scenario:** You want to train RETFound + LoRA for 20 epochs (~5 hours on RTX 3090)

**Without pre-training tests:**
- Config typo → Crash at startup → Fix → Restart (15 min wasted)
- OOM error → Reduce batch_size → Restart (30 min wasted)
- Model can't learn → Debug → Fix → Restart (5 hours wasted)
- **Total wasted:** 6+ hours

**With pre-training tests:**
- Run preflight_check.py (5 minutes)
- Fix all issues before training
- Training runs successfully on first try
- **Total wasted:** 0 hours

**ROI:** 6+ hours saved per experiment × multiple experiments = **massive time savings**

### Confidence Boost
- **Before:** "I hope this works... 🤞"
- **After:** "I know this will work. ✅"

### Debugging Speed
- **Before:** "Why isn't it working?" (hours of investigation)
- **After:** "The sanity test failed, here's exactly why." (minutes)

---

## Maintenance and Evolution

### Future Enhancements (Post-Implementation)

**Integration with CI/CD:**
```yaml
# .github/workflows/test.yml
- name: Run pre-flight checks
  run: python scripts/preflight_check.py --config configs/test_config.yaml
```

**W&B Integration:**
```python
# Log test results to Weights & Biases
wandb.log({
    "preflight/sanity_test": "PASS",
    "preflight/gpu_memory_mb": 6843,
    "preflight/overfitting_final_loss": 0.003
})
```

**Automated Memory Budgets:**
```python
# Auto-adjust batch_size based on GPU memory
recommended_batch_size = estimate_max_batch_size(
    model=model,
    gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9
)
```

---

## References

### Existing Documentation
- [TESTING_LANDSCAPE_ANALYSIS.md](../../TESTING_LANDSCAPE_ANALYSIS.md) - Comprehensive analysis of current tests
- [TESTING_QUICK_REFERENCE.md](../../TESTING_QUICK_REFERENCE.md) - Copy-paste command reference
- [TRAINING_GUIDE.md](../../docs/TRAINING_GUIDE.md) - Training pipeline documentation
- [CONFIGURATION_GUIDE.md](../../docs/CONFIGURATION_GUIDE.md) - Config system reference

### Existing Test Infrastructure
- `tests/` - 287 unit tests (pytest suite)
- `scripts/validate_all.py` - Comprehensive validation
- `scripts/validate_data.py` - Data integrity checks
- `tests/conftest.py` - 40+ pytest fixtures

### External Resources
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Memory Profiling: https://pytorch.org/docs/stable/torch.cuda.html#memory-management
- DataLoader Performance: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

## Questions and Discussion

### Before Implementation
1. **Scope:** Should we implement all phases or just critical ones (1-3)?
2. **Timeline:** How much time can we invest before starting training?
3. **RETFound variant:** Test both Large and Green, or focus on one?
4. **Integration:** One-time validation or permanent workflow addition?

### During Implementation
- Adapt based on actual needs
- Prioritize what provides most value
- Skip phases that don't apply to your use case

### After Implementation
- Document lessons learned
- Share what worked / didn't work
- Refine for future experiments

---

## Getting Started

**Next Steps:**
1. Review this overview document
2. Discuss priorities and timeline
3. Start with [01-SANITY_TESTS.md](01-SANITY_TESTS.md) (most critical)
4. Implement Phase 1, test it, iterate
5. Move to Phase 2, then Phase 3
6. Evaluate if Phases 4-6 are needed

**Ready to implement?** Open [01-SANITY_TESTS.md](01-SANITY_TESTS.md) and start coding!

---

**Document Version:** 1.0
**Last Updated:** 2025-11-04
**Maintainer:** Research Team
**Status:** Ready for Implementation
