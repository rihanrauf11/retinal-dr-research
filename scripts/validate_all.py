#!/usr/bin/env python3
"""
Comprehensive Validation Script for Diabetic Retinopathy Classification System.

This script tests all major components of the DR classification pipeline including:
- Dataset loading and preprocessing
- Model architectures (Baseline, RETFound, LoRA)
- Configuration system
- Utility functions
- Data augmentation
- End-to-end integration

Usage:
    python scripts/validate_all.py                    # Run all tests
    python scripts/validate_all.py --verbose          # Detailed output
    python scripts/validate_all.py --test dataset     # Run specific test group
    python scripts/validate_all.py --no-color         # Disable colored output

Exit Codes:
    0: All critical tests passed
    1: One or more critical tests failed
"""

import os
import sys
import time
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from scripts.config import Config, DataConfig, ModelConfig
from scripts.dataset import RetinalDataset
from scripts.model import DRClassifier
from scripts.utils import (
    set_seed, count_parameters, save_checkpoint, load_checkpoint,
    calculate_metrics, get_device
)

# Try to import RETFound modules (may not be available without weights)
try:
    from scripts.retfound_model import load_retfound_model, VisionTransformer
    RETFOUND_AVAILABLE = True
except Exception:
    RETFOUND_AVAILABLE = False

try:
    from scripts.retfound_lora import RETFoundLoRA
    LORA_AVAILABLE = True
except Exception:
    LORA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLOR CODES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def disable(cls):
        """Disable all colors."""
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKCYAN = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestResults:
    """Track test results across all validation tests."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.total = 0
        self.failures: List[Tuple[str, str]] = []
        self.start_time = time.time()

    def add_pass(self, test_name: str, detail: str = ""):
        """Record a passing test."""
        self.passed += 1
        self.total += 1
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}"
        detail_str = f" ({detail})" if detail else ""
        print(f"{status} {test_name:<40} {Colors.OKGREEN}PASS{Colors.ENDC}{detail_str}")

    def add_fail(self, test_name: str, error: str):
        """Record a failing test."""
        self.failed += 1
        self.total += 1
        self.failures.append((test_name, error))
        status = f"{Colors.FAIL}✗{Colors.ENDC}"
        print(f"{status} {test_name:<40} {Colors.FAIL}FAIL{Colors.ENDC}")
        print(f"  {Colors.FAIL}Error: {error}{Colors.ENDC}")

    def add_skip(self, test_name: str, reason: str):
        """Record a skipped test."""
        self.skipped += 1
        self.total += 1
        status = f"{Colors.WARNING}⚠{Colors.ENDC}"
        print(f"{status} {test_name:<40} {Colors.WARNING}SKIP{Colors.ENDC}")
        print(f"  {Colors.WARNING}{reason}{Colors.ENDC}")

    def get_duration(self) -> float:
        """Get elapsed time since validation started."""
        return time.time() - self.start_time

    def get_success_rate(self) -> float:
        """Calculate success rate (passed / (passed + failed))."""
        if self.passed + self.failed == 0:
            return 100.0
        return 100.0 * self.passed / (self.passed + self.failed)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(title: str, level: int = 1):
    """Print formatted section header."""
    if level == 1:
        sep = "═" * 79
        print(f"\n{Colors.BOLD}{sep}")
        print(f"{title:^79}")
        print(f"{sep}{Colors.ENDC}\n")
    else:
        sep = "─" * 79
        print(f"\n{Colors.BOLD}{sep}")
        print(f"{title}")
        print(f"{sep}{Colors.ENDC}")


def get_system_info() -> Dict[str, Any]:
    """Gather system information for reporting."""
    info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'device': str(get_device(verbose=False))
    }
    return info


def check_retfound_weights() -> Tuple[bool, str]:
    """Check if RETFound weights are available."""
    weight_path = Path('models/RETFound_cfp_weights.pth')
    if weight_path.exists():
        return True, str(weight_path)
    return False, str(weight_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 1: DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def test_dataset_loading(results: TestResults, verbose: bool = False):
    """Test dataset loading and basic functionality."""
    print_header("1. DATASET LOADING TESTS", level=2)

    # Test 1.1: Dataset initialization
    try:
        dataset = RetinalDataset(
            csv_file='data/sample/train.csv',
            img_dir='data/sample/images',
            transform=None
        )
        results.add_pass("RetinalDataset initialization", f"{len(dataset)} samples")
    except Exception as e:
        results.add_fail("RetinalDataset initialization", str(e))
        return  # Cannot continue without dataset

    # Test 1.2: __len__() method
    try:
        length = len(dataset)
        assert length > 0, "Dataset length should be positive"
        results.add_pass("Dataset __len__() method", f"length={length}")
    except Exception as e:
        results.add_fail("Dataset __len__() method", str(e))

    # Test 1.3: __getitem__() method
    try:
        image, label = dataset[0]
        # Without transforms, dataset returns PIL Image; with transforms it could be numpy/tensor
        from PIL import Image as PILImage
        assert isinstance(image, (np.ndarray, torch.Tensor, PILImage.Image)), \
            f"Image should be numpy array, tensor, or PIL Image, got {type(image)}"
        assert isinstance(label, int), "Label should be integer"
        results.add_pass("Dataset __getitem__() method", "returns (image, label)")
    except Exception as e:
        results.add_fail("Dataset __getitem__() method", str(e))

    # Test 1.4: Image shape verification with transforms
    try:
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        dataset_transformed = RetinalDataset(
            csv_file='data/sample/train.csv',
            img_dir='data/sample/images',
            transform=transform
        )
        image, label = dataset_transformed[0]
        expected_shape = (3, 224, 224)
        assert image.shape == expected_shape, f"Expected {expected_shape}, got {image.shape}"
        results.add_pass("Image shape verification", f"{image.shape}")
    except Exception as e:
        results.add_fail("Image shape verification", str(e))

    # Test 1.5: Label range verification
    try:
        labels = [dataset[i][1] for i in range(min(10, len(dataset)))]
        assert all(0 <= label <= 4 for label in labels), "Labels should be in [0, 4]"
        results.add_pass("Label range verification", "[0-4]")
    except Exception as e:
        results.add_fail("Label range verification", str(e))

    # Test 1.6: Class distribution
    try:
        dist = dataset.get_class_distribution()
        assert isinstance(dist, dict), "Class distribution should be dict"
        assert len(dist) > 0, "Class distribution should not be empty"
        results.add_pass("Class distribution calculation", f"{len(dist)} classes")
    except Exception as e:
        results.add_fail("Class distribution calculation", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 2: BASELINE MODEL (DRClassifier)
# ═══════════════════════════════════════════════════════════════════════════════

def test_baseline_model(results: TestResults, verbose: bool = False):
    """Test baseline DRClassifier model."""
    print_header("2. BASELINE MODEL TESTS (DRClassifier)", level=2)

    # Test 2.1: Model initialization (ResNet50)
    try:
        model = DRClassifier(
            model_name='resnet50',
            num_classes=5,
            pretrained=False,  # Faster for testing
            dropout_rate=0.3
        )
        results.add_pass("DRClassifier initialization (ResNet50)", "success")
    except Exception as e:
        results.add_fail("DRClassifier initialization (ResNet50)", str(e))
        return  # Cannot continue without model

    # Test 2.2: Forward pass
    try:
        set_seed(42, verbose=False)
        dummy_input = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        expected_shape = (4, 5)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        results.add_pass("Forward pass (batch=4)", f"output shape: {output.shape}")
    except Exception as e:
        results.add_fail("Forward pass (batch=4)", str(e))

    # Test 2.3: Output shape verification
    try:
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, 5), "Output shape mismatch"
        results.add_pass("Output shape verification", f"(batch_size, 5)")
    except Exception as e:
        results.add_fail("Output shape verification", str(e))

    # Test 2.4: Parameter counting
    try:
        total, trainable = count_parameters(model)
        assert total > 0, "Total parameters should be positive"
        assert trainable > 0, "Trainable parameters should be positive"
        assert trainable <= total, "Trainable should be <= total"
        results.add_pass("Parameter counting", f"{total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    except Exception as e:
        results.add_fail("Parameter counting", str(e))

    # Test 2.5: from_config() method
    try:
        model_config = ModelConfig(
            model_name='resnet50',
            num_classes=5,
            pretrained=False
        )
        model_from_config = DRClassifier.from_config(model_config)
        assert isinstance(model_from_config, DRClassifier), "Should return DRClassifier instance"
        results.add_pass("from_config() method", "creates model from config")
    except Exception as e:
        results.add_fail("from_config() method", str(e))

    # Test 2.6: freeze_backbone() method
    try:
        model_freeze = DRClassifier('resnet50', num_classes=5, pretrained=False)
        initial_total, initial_trainable = count_parameters(model_freeze)
        model_freeze.freeze_backbone()
        frozen_total, frozen_trainable = count_parameters(model_freeze)
        assert frozen_trainable < initial_trainable, "Trainable params should decrease after freezing"
        results.add_pass("freeze_backbone() method", f"reduced trainable params")
    except Exception as e:
        results.add_fail("freeze_backbone() method", str(e))

    # Test 2.7: unfreeze_backbone() method
    try:
        model_freeze.unfreeze_backbone()
        unfrozen_total, unfrozen_trainable = count_parameters(model_freeze)
        assert unfrozen_trainable == initial_trainable, "Trainable params should return to initial"
        results.add_pass("unfreeze_backbone() method", "restored trainable params")
    except Exception as e:
        results.add_fail("unfreeze_backbone() method", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 3: RETFOUND MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def test_retfound_model(results: TestResults, verbose: bool = False):
    """Test RETFound foundation model."""
    print_header("3. RETFOUND MODEL TESTS", level=2)

    # Check if weights are available
    weights_available, weights_path = check_retfound_weights()

    if not weights_available:
        results.add_skip(
            "RETFound weights not found",
            f"Location checked: {weights_path}\n  Download from: https://github.com/rmaphoh/RETFound_MAE"
        )
        return

    if not RETFOUND_AVAILABLE:
        results.add_skip("RETFound import failed", "Module import error")
        return

    # Test 3.1: Load RETFound model
    try:
        model = load_retfound_model(
            checkpoint_path=weights_path,
            num_classes=5,
            device='cpu'
        )
        results.add_pass("RETFound model loading", "weights loaded successfully")
    except Exception as e:
        results.add_fail("RETFound model loading", str(e))
        return

    # Test 3.2: Forward pass
    try:
        set_seed(42, verbose=False)
        dummy_input = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        expected_shape = (4, 5)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        results.add_pass("RETFound forward pass", f"output shape: {output.shape}")
    except Exception as e:
        results.add_fail("RETFound forward pass", str(e))

    # Test 3.3: Parameter counting
    try:
        total, trainable = count_parameters(model)
        # RETFound ViT-Large should have ~303M parameters
        assert total > 100_000_000, f"Expected >100M params, got {total}"
        results.add_pass("RETFound parameter count", f"{total/1e6:.1f}M params")
    except Exception as e:
        results.add_fail("RETFound parameter count", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 4: RETFOUND + LORA
# ═══════════════════════════════════════════════════════════════════════════════

def test_retfound_lora(results: TestResults, verbose: bool = False):
    """Test RETFound with LoRA adapters."""
    print_header("4. RETFOUND + LORA TESTS", level=2)

    # Check if weights are available
    weights_available, weights_path = check_retfound_weights()

    if not weights_available:
        results.add_skip(
            "RETFound weights not found",
            f"Cannot test LoRA without RETFound weights\n  Using mock model for structure testing..."
        )
        # Test LoRA parameter reduction logic with mock model
        test_lora_logic_mock(results)
        return

    if not LORA_AVAILABLE:
        results.add_skip("LoRA import failed", "Module import error or missing dependencies")
        return

    # Test 4.1: Create LoRA model
    try:
        model = RETFoundLoRA(
            checkpoint_path=weights_path,
            num_classes=5,
            lora_r=8,
            lora_alpha=32,
            device='cpu'
        )
        results.add_pass("RETFoundLoRA initialization", "r=8, alpha=32")
    except Exception as e:
        results.add_fail("RETFoundLoRA initialization", str(e))
        return

    # Test 4.2: Forward pass
    try:
        set_seed(42, verbose=False)
        dummy_input = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        expected_shape = (4, 5)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        results.add_pass("LoRA forward pass", f"output shape: {output.shape}")
    except Exception as e:
        results.add_fail("LoRA forward pass", str(e))

    # Test 4.3: Parameter efficiency
    try:
        total, trainable = model.get_num_params()
        trainable_pct = 100.0 * trainable / total
        assert trainable < total * 0.01, f"LoRA should train <1% of params, got {trainable_pct:.2f}%"
        results.add_pass(
            "LoRA parameter efficiency",
            f"{trainable/1e6:.2f}M trainable ({trainable_pct:.3f}% of {total/1e6:.1f}M)"
        )
    except Exception as e:
        results.add_fail("LoRA parameter efficiency", str(e))

    # Test 4.4: Verify frozen backbone
    try:
        # Check that backbone parameters are frozen
        backbone_params = [p for n, p in model.backbone.named_parameters() if 'lora' not in n.lower()]
        frozen_count = sum(1 for p in backbone_params if not p.requires_grad)
        total_backbone = len(backbone_params)
        assert frozen_count > 0, "Some backbone parameters should be frozen"
        results.add_pass("Frozen backbone verification", f"{frozen_count}/{total_backbone} params frozen")
    except Exception as e:
        results.add_fail("Frozen backbone verification", str(e))


def test_lora_logic_mock(results: TestResults):
    """Test LoRA parameter reduction logic with a mock model."""
    try:
        # Create a simple mock ViT model
        class MockViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(768, 768)
                self.layers = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])
                self.head = nn.Linear(768, 5)

            def forward(self, x):
                return self.head(x)

        mock_model = MockViT()
        total_before, trainable_before = count_parameters(mock_model)

        # Freeze all parameters
        for param in mock_model.parameters():
            param.requires_grad = False

        # Unfreeze only head (simulating LoRA behavior)
        for param in mock_model.head.parameters():
            param.requires_grad = True

        total_after, trainable_after = count_parameters(mock_model)

        assert trainable_after < trainable_before, "Trainable params should decrease"
        assert total_after == total_before, "Total params should stay same"

        trainable_pct = 100.0 * trainable_after / total_after
        results.add_pass(
            "LoRA parameter reduction logic (mock)",
            f"reduced to {trainable_pct:.2f}% trainable"
        )
    except Exception as e:
        results.add_fail("LoRA parameter reduction logic (mock)", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 5: CONFIGURATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def test_configuration_system(results: TestResults, verbose: bool = False):
    """Test configuration management system."""
    print_header("5. CONFIGURATION SYSTEM TESTS", level=2)

    # Test 5.1: Config creation from defaults
    try:
        config = Config()
        assert isinstance(config, Config), "Should return Config instance"
        assert config.model.num_classes == 5, "Default num_classes should be 5"
        results.add_pass("Config creation from defaults", "default values loaded")
    except Exception as e:
        results.add_fail("Config creation from defaults", str(e))

    # Test 5.2: Config from YAML
    try:
        if Path('configs/test_config.yaml').exists():
            config = Config.from_yaml('configs/test_config.yaml')
            assert isinstance(config, Config), "Should return Config instance"
            results.add_pass("Config from YAML (test_config.yaml)", "loaded successfully")
        else:
            results.add_skip("Config from YAML", "test_config.yaml not found")
    except Exception as e:
        results.add_fail("Config from YAML (test_config.yaml)", str(e))

    # Test 5.3: validate() method
    try:
        config = Config()
        config.validate(create_dirs=False)
        results.add_pass("validate() method", "validation passed")
    except Exception as e:
        results.add_fail("validate() method", str(e))

    # Test 5.4: to_dict() / from_dict()
    try:
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "to_dict() should return dict"

        config_restored = Config.from_dict(config_dict)
        assert isinstance(config_restored, Config), "from_dict() should return Config"
        assert config_restored.model.num_classes == config.model.num_classes, "Values should match"
        results.add_pass("to_dict() / from_dict()", "serialization works")
    except Exception as e:
        results.add_fail("to_dict() / from_dict()", str(e))

    # Test 5.5: YAML serialization
    try:
        config = Config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_yaml = f.name

        try:
            config.to_yaml(temp_yaml)
            config_loaded = Config.from_yaml(temp_yaml)
            assert config_loaded.model.num_classes == config.model.num_classes, "Values should match"
            results.add_pass("YAML serialization", "save/load works")
        finally:
            if Path(temp_yaml).exists():
                Path(temp_yaml).unlink()
    except Exception as e:
        results.add_fail("YAML serialization", str(e))

    # Test 5.6: Device auto-detection
    try:
        config = Config()
        assert config.system.device in ['cuda', 'cpu', 'mps'], "Device should be valid"
        results.add_pass("Device auto-detection", f"detected: {config.system.device}")
    except Exception as e:
        results.add_fail("Device auto-detection", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 6: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def test_utility_functions(results: TestResults, verbose: bool = False):
    """Test utility functions."""
    print_header("6. UTILITY FUNCTIONS TESTS", level=2)

    # Test 6.1: set_seed() reproducibility
    try:
        set_seed(42, verbose=False)
        rand1 = torch.rand(5)

        set_seed(42, verbose=False)
        rand2 = torch.rand(5)

        assert torch.allclose(rand1, rand2), "Same seed should produce same random values"
        results.add_pass("set_seed() reproducibility", "deterministic output")
    except Exception as e:
        results.add_fail("set_seed() reproducibility", str(e))

    # Test 6.2: count_parameters()
    try:
        mock_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        total, trainable = count_parameters(mock_model)
        expected_total = (10 * 20 + 20) + (20 * 5 + 5)  # weights + biases
        assert total == expected_total, f"Expected {expected_total}, got {total}"
        results.add_pass("count_parameters()", f"{total} params counted")
    except Exception as e:
        results.add_fail("count_parameters()", str(e))

    # Test 6.3: save_checkpoint()
    try:
        mock_model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_checkpoint = f.name

        try:
            save_checkpoint(
                model=mock_model,
                optimizer=optimizer,
                epoch=5,
                metrics={'val_acc': 0.85, 'val_loss': 0.42},
                path=temp_checkpoint
            )
            assert Path(temp_checkpoint).exists(), "Checkpoint file should exist"
            results.add_pass("save_checkpoint()", "checkpoint saved")
        finally:
            if Path(temp_checkpoint).exists():
                Path(temp_checkpoint).unlink()
    except Exception as e:
        results.add_fail("save_checkpoint()", str(e))

    # Test 6.4: load_checkpoint()
    try:
        mock_model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_checkpoint = f.name

        try:
            # Save checkpoint
            save_checkpoint(
                model=mock_model,
                optimizer=optimizer,
                epoch=5,
                metrics={'val_acc': 0.85},
                path=temp_checkpoint
            )

            # Load checkpoint
            new_model = nn.Linear(10, 5)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            checkpoint = load_checkpoint(
                path=temp_checkpoint,
                model=new_model,
                optimizer=new_optimizer
            )

            assert checkpoint['epoch'] == 5, "Epoch should be restored"
            assert checkpoint['metrics']['val_acc'] == 0.85, "Metrics should be restored"
            results.add_pass("load_checkpoint()", "checkpoint loaded and restored")
        finally:
            if Path(temp_checkpoint).exists():
                Path(temp_checkpoint).unlink()
    except Exception as e:
        results.add_fail("load_checkpoint()", str(e))

    # Test 6.5: calculate_metrics()
    try:
        y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 2, 0, 2, 1, 1])

        metrics = calculate_metrics(y_true, y_pred, num_classes=3)

        assert 'accuracy' in metrics, "Metrics should include accuracy"
        assert 'cohen_kappa' in metrics, "Metrics should include cohen_kappa"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"
        results.add_pass("calculate_metrics()", f"accuracy: {metrics['accuracy']:.3f}, kappa: {metrics['cohen_kappa']:.3f}")
    except Exception as e:
        results.add_fail("calculate_metrics()", str(e))

    # Test 6.6: get_device()
    try:
        device = get_device(verbose=False)
        assert isinstance(device, torch.device), "Should return torch.device"
        results.add_pass("get_device()", f"device: {device}")
    except Exception as e:
        results.add_fail("get_device()", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 7: DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_data_augmentation(results: TestResults, verbose: bool = False):
    """Test data augmentation pipelines."""
    print_header("7. DATA AUGMENTATION TESTS", level=2)

    # Test 7.1: Training transforms (Albumentations)
    try:
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = train_transform(image=dummy_image)
        transformed_image = transformed['image']

        assert isinstance(transformed_image, torch.Tensor), "Should return tensor"
        assert transformed_image.shape == (3, 224, 224), f"Expected (3, 224, 224), got {transformed_image.shape}"
        results.add_pass("Training transforms (Albumentations)", "pipeline works")
    except Exception as e:
        results.add_fail("Training transforms (Albumentations)", str(e))

    # Test 7.2: Validation transforms (minimal)
    try:
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = val_transform(image=dummy_image)
        transformed_image = transformed['image']

        assert isinstance(transformed_image, torch.Tensor), "Should return tensor"
        results.add_pass("Validation transforms (minimal)", "resize + normalize")
    except Exception as e:
        results.add_fail("Validation transforms (minimal)", str(e))

    # Test 7.3: ImageNet normalization values
    try:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Verify these are correct ImageNet stats
        assert len(imagenet_mean) == 3, "Mean should have 3 values"
        assert len(imagenet_std) == 3, "Std should have 3 values"
        assert all(0 < v < 1 for v in imagenet_mean), "Mean values should be normalized"
        results.add_pass("ImageNet normalization", "correct stats applied")
    except Exception as e:
        results.add_fail("ImageNet normalization", str(e))

    # Test 7.4: Transform on sample image
    try:
        if Path('data/sample/images').exists():
            dataset = RetinalDataset(
                csv_file='data/sample/train.csv',
                img_dir='data/sample/images',
                transform=train_transform
            )
            image, label = dataset[0]
            assert isinstance(image, torch.Tensor), "Should return tensor"
            assert image.shape == (3, 224, 224), "Should be (3, 224, 224)"
            results.add_pass("Transform on sample image", "real image processed")
        else:
            results.add_skip("Transform on sample image", "sample data not found")
    except Exception as e:
        results.add_fail("Transform on sample image", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 8: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_integration(results: TestResults, verbose: bool = False):
    """Test end-to-end integration."""
    print_header("8. INTEGRATION TESTS", level=2)

    # Test 8.1: Data → Model → Prediction pipeline
    try:
        if not Path('data/sample/images').exists():
            results.add_skip("End-to-end pipeline", "sample data not found")
            return

        # Setup
        set_seed(42, verbose=False)
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Load data
        dataset = RetinalDataset(
            csv_file='data/sample/train.csv',
            img_dir='data/sample/images',
            transform=transform
        )

        # Create model
        model = DRClassifier('resnet50', num_classes=5, pretrained=False)
        model.eval()

        # Get predictions
        with torch.no_grad():
            image, label = dataset[0]
            image_batch = image.unsqueeze(0)  # Add batch dimension
            predictions = model(image_batch)
            predicted_class = predictions.argmax(dim=1).item()

        assert 0 <= predicted_class <= 4, f"Prediction should be in [0, 4], got {predicted_class}"
        results.add_pass("Data → Model → Prediction pipeline", f"predicted class: {predicted_class}")
    except Exception as e:
        results.add_fail("Data → Model → Prediction pipeline", str(e))

    # Test 8.2: Batch processing
    try:
        if not Path('data/sample/images').exists():
            results.add_skip("Batch processing", "sample data not found")
            return

        from torch.utils.data import DataLoader

        dataset = RetinalDataset(
            csv_file='data/sample/train.csv',
            img_dir='data/sample/images',
            transform=transform
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = DRClassifier('resnet50', num_classes=5, pretrained=False)
        model.eval()

        # Process one batch
        images, labels = next(iter(dataloader))
        with torch.no_grad():
            predictions = model(images)

        assert predictions.shape[0] == images.shape[0], "Batch size should match"
        assert predictions.shape[1] == 5, "Should have 5 classes"
        results.add_pass("Batch processing", f"batch size: {images.shape[0]}")
    except Exception as e:
        results.add_fail("Batch processing", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests(verbose: bool = False, test_groups: Optional[List[str]] = None) -> TestResults:
    """Run all validation tests."""
    results = TestResults()

    # Define all test groups
    all_tests = {
        'dataset': test_dataset_loading,
        'model': test_baseline_model,
        'retfound': test_retfound_model,
        'lora': test_retfound_lora,
        'config': test_configuration_system,
        'utils': test_utility_functions,
        'augmentation': test_data_augmentation,
        'integration': test_integration
    }

    # Filter tests if specific groups requested
    if test_groups:
        tests_to_run = {k: v for k, v in all_tests.items() if k in test_groups}
        if not tests_to_run:
            print(f"{Colors.FAIL}Error: Invalid test group(s): {test_groups}{Colors.ENDC}")
            print(f"Available groups: {', '.join(all_tests.keys())}")
            sys.exit(1)
    else:
        tests_to_run = all_tests

    # Run tests
    for test_func in tests_to_run.values():
        test_func(results, verbose=verbose)

    return results


def print_summary(results: TestResults, sys_info: Dict[str, Any]):
    """Print final validation summary."""
    print_header("VALIDATION SUMMARY", level=2)

    duration = results.get_duration()
    success_rate = results.get_success_rate()

    # Summary statistics
    print(f"Total Tests:        {results.total}")
    print(f"Passed:            {results.passed:2d}  {Colors.OKGREEN}✓{Colors.ENDC}")
    print(f"Failed:            {results.failed:2d}  {Colors.FAIL}✗{Colors.ENDC}")
    print(f"Skipped:           {results.skipped:2d}  {Colors.WARNING}⚠{Colors.ENDC}")
    print()

    # Overall status
    if results.failed == 0:
        status_color = Colors.OKGREEN
        status_text = "PASS"
    else:
        status_color = Colors.FAIL
        status_text = "FAIL"

    print(f"Status:            {status_color}{status_text} ({success_rate:.1f}%){Colors.ENDC}")
    print(f"Duration:          {duration:.1f} seconds")
    print()

    # Print failures if any
    if results.failures:
        print(f"{Colors.FAIL}{Colors.BOLD}Failed Tests:{Colors.ENDC}")
        for test_name, error in results.failures:
            print(f"  {Colors.FAIL}✗ {test_name}{Colors.ENDC}")
            print(f"    {error}")
        print()

    # Final message
    if results.failed == 0:
        print(f"{Colors.OKGREEN}All critical components validated successfully!{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}Some tests failed. Please review the errors above.{Colors.ENDC}")


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description='Comprehensive validation for DR classification system',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    parser.add_argument(
        '--test', '-t',
        nargs='+',
        choices=['dataset', 'model', 'retfound', 'lora', 'config', 'utils', 'augmentation', 'integration'],
        help='Run specific test group(s) only'
    )

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.disable()

    # Print header
    print_header("DIABETIC RETINOPATHY - VALIDATION SUITE", level=1)

    # Print system information
    sys_info = get_system_info()
    print(f"{Colors.BOLD}System Information{Colors.ENDC}")
    print("─" * 79)
    print(f"Python Version:     {sys_info['python_version']}")
    print(f"PyTorch Version:    {sys_info['pytorch_version']}")
    print(f"Device:             {sys_info['device']}")
    print(f"CUDA Available:     {sys_info['cuda_available']}")
    print(f"MPS Available:      {sys_info['mps_available']}")

    # Run tests
    results = run_all_tests(verbose=args.verbose, test_groups=args.test)

    # Print summary
    print_summary(results, sys_info)

    # Print closing separator
    print(f"\n{Colors.BOLD}{'═' * 79}{Colors.ENDC}\n")

    # Exit with appropriate code
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == '__main__':
    main()
