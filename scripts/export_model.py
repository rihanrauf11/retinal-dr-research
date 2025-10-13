"""
Model Export Script for Diabetic Retinopathy Classification

This script exports trained PyTorch models to various deployment formats:
- ONNX: Universal format for deployment (ONNX Runtime, TensorRT, CoreML)
- TorchScript: Native PyTorch format for mobile/production (PyTorch Mobile, C++)
- LoRA Adapters: Separate weight export for adapter swapping

Features:
- Auto-detect model type (baseline/LoRA)
- Verify model before export
- Validate exported model against original
- Optional quantization for optimization
- Save deployment metadata
- Generate deployment guide
- Comprehensive CLI

Usage Examples:
    # Export to ONNX
    python scripts/export_model.py --checkpoint model.pth --format onnx --output model.onnx

    # Export to TorchScript
    python scripts/export_model.py --checkpoint model.pth --format torchscript --output model.pt

    # Export LoRA adapters separately
    python scripts/export_model.py --checkpoint lora_model.pth --format lora --output lora_adapters.pth

    # Export all formats with quantization
    python scripts/export_model.py --checkpoint model.pth --format all --output_dir exports/ --quantize

Author: Generated with Claude Code
Date: 2025-10-13
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.model import DRClassifier
from scripts.retfound_lora import RETFoundLoRA


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}

CLASS_DESCRIPTIONS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

# ImageNet normalization (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default input size for RETFound and most vision models
DEFAULT_INPUT_SIZE = 224

# ONNX configuration
DEFAULT_ONNX_OPSET = 14  # Compatible with most deployment platforms
ONNX_INPUT_NAME = 'input'
ONNX_OUTPUT_NAME = 'output'

# Export formats
EXPORT_FORMATS = ['onnx', 'torchscript', 'lora', 'all']

# Optimization levels
OPTIMIZATION_LEVELS = {
    0: "None - Direct export without optimization",
    1: "Basic - Standard optimizations for inference",
    2: "Moderate - Includes quantization (dynamic)",
    3: "Aggressive - Full optimization with static quantization"
}


# =============================================================================
# Model Type Detection and Loading
# =============================================================================

def detect_model_type(checkpoint_path: Path) -> str:
    """
    Auto-detect model type from checkpoint structure.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file

    Returns
    -------
    str
        Model type: 'baseline', 'lora', or 'retfound'
    """
    logger.info(f"Detecting model type from checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check for LoRA model
        if 'lora_adapters' in checkpoint or 'lora_config' in checkpoint:
            logger.info("✓ Detected LoRA model")
            return 'lora'

        # Check for baseline model
        if 'model_state_dict' in checkpoint:
            logger.info("✓ Detected baseline model (DRClassifier)")
            return 'baseline'

        # Check for RETFound base model
        if 'model' in checkpoint and 'pos_embed' in checkpoint:
            logger.info("✓ Detected RETFound foundation model")
            return 'retfound'

        # Default to baseline
        logger.warning("Could not definitively detect model type, defaulting to baseline")
        return 'baseline'

    except Exception as e:
        logger.error(f"Error detecting model type: {e}")
        raise ValueError(f"Failed to detect model type from checkpoint: {e}")


def load_baseline_model(
    checkpoint_path: Path,
    device: torch.device,
    verbose: bool = True
) -> nn.Module:
    """
    Load a baseline DRClassifier model from checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file
    device : torch.device
        Device to load model on
    verbose : bool
        Whether to print loading information

    Returns
    -------
    nn.Module
        Loaded model in eval mode
    """
    if verbose:
        logger.info(f"Loading baseline model from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model configuration
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'efficientnet_b0')
        num_classes = config.get('num_classes', 5)
        dropout_rate = config.get('dropout_rate', 0.3)

        if verbose:
            logger.info(f"  Model architecture: {model_name}")
            logger.info(f"  Number of classes: {num_classes}")
            logger.info(f"  Dropout rate: {dropout_rate}")

        # Initialize model
        model = DRClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,  # We're loading weights
            dropout_rate=dropout_rate
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Total parameters: {total_params:,}")

        return model

    except Exception as e:
        logger.error(f"Error loading baseline model: {e}")
        raise


def load_lora_model(
    checkpoint_path: Path,
    device: torch.device,
    retfound_checkpoint: Optional[Path] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Load a RETFoundLoRA model from checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the LoRA checkpoint file
    device : torch.device
        Device to load model on
    retfound_checkpoint : Path, optional
        Path to RETFound foundation model checkpoint
    verbose : bool
        Whether to print loading information

    Returns
    -------
    nn.Module
        Loaded model in eval mode
    """
    if verbose:
        logger.info(f"Loading LoRA model from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        config = checkpoint.get('config', {})
        num_classes = config.get('num_classes', 5)
        lora_r = config.get('lora_r', 8)
        lora_alpha = config.get('lora_alpha', 32)
        lora_dropout = config.get('lora_dropout', 0.1)

        # Get RETFound checkpoint path
        if retfound_checkpoint is None:
            retfound_checkpoint = config.get('pretrained_path', 'models/RETFound_cfp_weights.pth')

        if verbose:
            logger.info(f"  Number of classes: {num_classes}")
            logger.info(f"  LoRA rank (r): {lora_r}")
            logger.info(f"  LoRA alpha: {lora_alpha}")
            logger.info(f"  LoRA dropout: {lora_dropout}")
            logger.info(f"  RETFound checkpoint: {retfound_checkpoint}")

        # Initialize model
        model = RETFoundLoRA(
            checkpoint_path=str(retfound_checkpoint),
            num_classes=num_classes,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Load LoRA adapters
        if 'lora_adapters' in checkpoint:
            model.load_state_dict(checkpoint['lora_adapters'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        model = model.to(device)
        model.eval()

        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")

        return model

    except Exception as e:
        logger.error(f"Error loading LoRA model: {e}")
        raise


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_type: str = 'auto',
    retfound_checkpoint: Optional[Path] = None,
    verbose: bool = True
) -> Tuple[nn.Module, str, Dict]:
    """
    Load a model from checkpoint with auto-detection.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file
    device : torch.device
        Device to load model on
    model_type : str
        Model type: 'auto', 'baseline', or 'lora'
    retfound_checkpoint : Path, optional
        Path to RETFound foundation model (for LoRA)
    verbose : bool
        Whether to print loading information

    Returns
    -------
    tuple
        (model, model_type, config) where:
        - model: Loaded PyTorch model
        - model_type: Detected/specified model type
        - config: Model configuration dict
    """
    # Auto-detect model type if needed
    if model_type == 'auto':
        model_type = detect_model_type(checkpoint_path)

    # Load checkpoint to extract config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Load model based on type
    if model_type == 'baseline':
        model = load_baseline_model(checkpoint_path, device, verbose)
    elif model_type == 'lora':
        model = load_lora_model(checkpoint_path, device, retfound_checkpoint, verbose)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, model_type, config


# =============================================================================
# Model Verification
# =============================================================================

def verify_model(
    model: nn.Module,
    input_size: int = DEFAULT_INPUT_SIZE,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Verify that the model works correctly before export.

    Parameters
    ----------
    model : nn.Module
        Model to verify
    input_size : int
        Input image size
    device : torch.device
        Device to run verification on

    Returns
    -------
    dict
        Verification results including output shape and sample prediction
    """
    logger.info("Verifying model...")

    try:
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

        # Run inference
        with torch.no_grad():
            start_time = time.time()
            output = model(dummy_input)
            inference_time = time.time() - start_time

        # Verify output shape
        expected_shape = (1, 5)  # Batch size 1, 5 classes
        if output.shape != expected_shape:
            raise ValueError(f"Unexpected output shape: {output.shape}, expected {expected_shape}")

        # Get prediction
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        verification_results = {
            'status': 'success',
            'input_shape': list(dummy_input.shape),
            'output_shape': list(output.shape),
            'inference_time_ms': inference_time * 1000,
            'sample_prediction': {
                'class': predicted_class,
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': confidence
            }
        }

        logger.info("✓ Model verification successful")
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Inference time: {inference_time*1000:.2f} ms")

        return verification_results

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: int = DEFAULT_INPUT_SIZE,
    opset_version: int = DEFAULT_ONNX_OPSET,
    dynamic_axes: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Export model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        Model to export
    output_path : Path
        Output path for ONNX model
    input_size : int
        Input image size
    opset_version : int
        ONNX opset version
    dynamic_axes : bool
        Whether to use dynamic batch size
    verbose : bool
        Whether to print export information

    Returns
    -------
    dict
        Export results and metadata
    """
    if verbose:
        logger.info(f"Exporting to ONNX format...")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Opset version: {opset_version}")
        logger.info(f"  Dynamic axes: {dynamic_axes}")

    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Configure dynamic axes if requested
        if dynamic_axes:
            dynamic_axes_config = {
                ONNX_INPUT_NAME: {0: 'batch_size'},
                ONNX_OUTPUT_NAME: {0: 'batch_size'}
            }
        else:
            dynamic_axes_config = None

        # Export to ONNX
        start_time = time.time()
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[ONNX_INPUT_NAME],
            output_names=[ONNX_OUTPUT_NAME],
            dynamic_axes=dynamic_axes_config,
            verbose=False
        )
        export_time = time.time() - start_time

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        export_results = {
            'status': 'success',
            'output_path': str(output_path),
            'format': 'onnx',
            'opset_version': opset_version,
            'input_size': input_size,
            'dynamic_batch': dynamic_axes,
            'file_size_mb': file_size_mb,
            'export_time_seconds': export_time
        }

        if verbose:
            logger.info(f"✓ ONNX export successful")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Export time: {export_time:.2f} seconds")

        return export_results

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def validate_onnx_model(
    onnx_path: Path,
    original_model: nn.Module,
    input_size: int = DEFAULT_INPUT_SIZE,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate ONNX model against original PyTorch model.

    Parameters
    ----------
    onnx_path : Path
        Path to ONNX model
    original_model : nn.Module
        Original PyTorch model
    input_size : int
        Input image size
    tolerance : float
        Numerical tolerance for comparison
    verbose : bool
        Whether to print validation information

    Returns
    -------
    dict
        Validation results
    """
    if verbose:
        logger.info("Validating ONNX model...")

    try:
        import onnx
        import onnxruntime as ort

        # Load and check ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Create test input
        test_input = torch.randn(1, 3, input_size, input_size)

        # Get PyTorch output
        original_model.eval()
        with torch.no_grad():
            pytorch_output = original_model(test_input).numpy()

        # Get ONNX output
        ort_session = ort.InferenceSession(str(onnx_path))
        onnx_output = ort_session.run(
            None,
            {ONNX_INPUT_NAME: test_input.numpy()}
        )[0]

        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()

        # Check if within tolerance
        is_valid = max_diff < tolerance

        validation_results = {
            'status': 'success' if is_valid else 'failed',
            'is_valid': is_valid,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance,
            'pytorch_output_sample': pytorch_output[0].tolist(),
            'onnx_output_sample': onnx_output[0].tolist()
        }

        if verbose:
            if is_valid:
                logger.info(f"✓ ONNX validation successful")
            else:
                logger.warning(f"✗ ONNX validation failed (difference > tolerance)")
            logger.info(f"  Max difference: {max_diff:.2e}")
            logger.info(f"  Mean difference: {mean_diff:.2e}")
            logger.info(f"  Tolerance: {tolerance:.2e}")

        return validation_results

    except ImportError as e:
        logger.error(f"Missing required package for ONNX validation: {e}")
        logger.error("Install with: pip install onnx onnxruntime")
        return {
            'status': 'skipped',
            'reason': 'Missing onnx or onnxruntime package'
        }
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


# =============================================================================
# TorchScript Export
# =============================================================================

def export_to_torchscript(
    model: nn.Module,
    output_path: Path,
    input_size: int = DEFAULT_INPUT_SIZE,
    method: str = 'trace',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Export model to TorchScript format.

    Parameters
    ----------
    model : nn.Module
        Model to export
    output_path : Path
        Output path for TorchScript model
    input_size : int
        Input image size
    method : str
        Export method: 'trace' or 'script'
    verbose : bool
        Whether to print export information

    Returns
    -------
    dict
        Export results and metadata
    """
    if verbose:
        logger.info(f"Exporting to TorchScript format...")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Method: {method}")

    try:
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Export based on method
        start_time = time.time()
        if method == 'trace':
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
        elif method == 'script':
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown export method: {method}. Use 'trace' or 'script'.")

        # Save to file
        torch.jit.save(traced_model, str(output_path))
        export_time = time.time() - start_time

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        export_results = {
            'status': 'success',
            'output_path': str(output_path),
            'format': 'torchscript',
            'method': method,
            'input_size': input_size,
            'file_size_mb': file_size_mb,
            'export_time_seconds': export_time
        }

        if verbose:
            logger.info(f"✓ TorchScript export successful")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Export time: {export_time:.2f} seconds")

        return export_results

    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def validate_torchscript_model(
    torchscript_path: Path,
    original_model: nn.Module,
    input_size: int = DEFAULT_INPUT_SIZE,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate TorchScript model against original PyTorch model.

    Parameters
    ----------
    torchscript_path : Path
        Path to TorchScript model
    original_model : nn.Module
        Original PyTorch model
    input_size : int
        Input image size
    tolerance : float
        Numerical tolerance for comparison
    verbose : bool
        Whether to print validation information

    Returns
    -------
    dict
        Validation results
    """
    if verbose:
        logger.info("Validating TorchScript model...")

    try:
        # Load TorchScript model
        loaded_model = torch.jit.load(str(torchscript_path))
        loaded_model.eval()

        # Create test input
        test_input = torch.randn(1, 3, input_size, input_size)

        # Get PyTorch output
        original_model.eval()
        with torch.no_grad():
            pytorch_output = original_model(test_input).numpy()

        # Get TorchScript output
        with torch.no_grad():
            torchscript_output = loaded_model(test_input).numpy()

        # Compare outputs
        max_diff = np.abs(pytorch_output - torchscript_output).max()
        mean_diff = np.abs(pytorch_output - torchscript_output).mean()

        # Check if within tolerance
        is_valid = max_diff < tolerance

        validation_results = {
            'status': 'success' if is_valid else 'failed',
            'is_valid': is_valid,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance,
            'pytorch_output_sample': pytorch_output[0].tolist(),
            'torchscript_output_sample': torchscript_output[0].tolist()
        }

        if verbose:
            if is_valid:
                logger.info(f"✓ TorchScript validation successful")
            else:
                logger.warning(f"✗ TorchScript validation failed (difference > tolerance)")
            logger.info(f"  Max difference: {max_diff:.2e}")
            logger.info(f"  Mean difference: {mean_diff:.2e}")
            logger.info(f"  Tolerance: {tolerance:.2e}")

        return validation_results

    except Exception as e:
        logger.error(f"TorchScript validation failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


# =============================================================================
# LoRA Adapter Export
# =============================================================================

def export_lora_adapters(
    model: nn.Module,
    output_path: Path,
    checkpoint: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Export LoRA adapters separately for adapter swapping.

    Parameters
    ----------
    model : nn.Module
        LoRA model
    output_path : Path
        Output path for LoRA adapters
    checkpoint : dict
        Original checkpoint with config
    verbose : bool
        Whether to print export information

    Returns
    -------
    dict
        Export results and metadata
    """
    if verbose:
        logger.info(f"Exporting LoRA adapters...")
        logger.info(f"  Output path: {output_path}")

    try:
        # Extract LoRA-specific state dict
        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_state[name] = param.cpu()

        # Get LoRA configuration
        config = checkpoint.get('config', {})
        lora_config = {
            'lora_r': config.get('lora_r', 8),
            'lora_alpha': config.get('lora_alpha', 32),
            'lora_dropout': config.get('lora_dropout', 0.1),
            'target_modules': config.get('target_modules', ['qkv']),
            'num_classes': config.get('num_classes', 5)
        }

        # Create adapter checkpoint
        adapter_checkpoint = {
            'lora_adapters': lora_state,
            'lora_config': lora_config,
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__
        }

        # Save adapters
        start_time = time.time()
        torch.save(adapter_checkpoint, str(output_path))
        export_time = time.time() - start_time

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        # Calculate adapter parameter count
        adapter_params = sum(p.numel() for p in lora_state.values())

        export_results = {
            'status': 'success',
            'output_path': str(output_path),
            'format': 'lora_adapters',
            'lora_config': lora_config,
            'adapter_parameters': adapter_params,
            'file_size_mb': file_size_mb,
            'export_time_seconds': export_time
        }

        if verbose:
            logger.info(f"✓ LoRA adapter export successful")
            logger.info(f"  Adapter parameters: {adapter_params:,}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Export time: {export_time:.2f} seconds")

        return export_results

    except Exception as e:
        logger.error(f"LoRA adapter export failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


# =============================================================================
# Quantization
# =============================================================================

def quantize_model(
    model: nn.Module,
    output_path: Path,
    quantization_type: str = 'dynamic',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quantize model for optimization.

    Parameters
    ----------
    model : nn.Module
        Model to quantize
    output_path : Path
        Output path for quantized model
    quantization_type : str
        Type of quantization: 'dynamic' or 'static'
    verbose : bool
        Whether to print quantization information

    Returns
    -------
    dict
        Quantization results and metadata
    """
    if verbose:
        logger.info(f"Quantizing model ({quantization_type})...")
        logger.info(f"  Output path: {output_path}")

    try:
        model.eval()

        # Dynamic quantization (easiest, no calibration needed)
        if quantization_type == 'dynamic':
            start_time = time.time()
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            quantization_time = time.time() - start_time

            # Save quantized model
            torch.save(quantized_model.state_dict(), str(output_path))

        elif quantization_type == 'static':
            # Static quantization requires calibration data
            logger.warning("Static quantization requires calibration data")
            logger.warning("Using dynamic quantization instead")
            return quantize_model(model, output_path, 'dynamic', verbose)

        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        # Get file sizes
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

        quantization_results = {
            'status': 'success',
            'output_path': str(output_path),
            'quantization_type': quantization_type,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'quantization_time_seconds': quantization_time
        }

        if verbose:
            logger.info(f"✓ Quantization successful")
            logger.info(f"  Original size: {original_size:.2f} MB")
            logger.info(f"  Quantized size: {quantized_size:.2f} MB")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"  Quantization time: {quantization_time:.2f} seconds")

        return quantization_results

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


# =============================================================================
# Metadata Generation
# =============================================================================

def generate_metadata(
    model_type: str,
    config: Dict,
    verification_results: Dict,
    export_results: List[Dict],
    output_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for exported model.

    Parameters
    ----------
    model_type : str
        Type of model (baseline/lora)
    config : dict
        Model configuration
    verification_results : dict
        Results from model verification
    export_results : list
        List of export results for each format
    output_path : Path
        Output path for metadata JSON
    verbose : bool
        Whether to print metadata information

    Returns
    -------
    dict
        Complete metadata dictionary
    """
    if verbose:
        logger.info("Generating export metadata...")

    metadata = {
        'export_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__,
            'python_version': sys.version,
            'model_type': model_type
        },
        'model_info': {
            'type': model_type,
            'config': config,
            'num_classes': config.get('num_classes', 5),
            'architecture': config.get('model_name', 'unknown')
        },
        'input_spec': {
            'format': 'NCHW',
            'channels': 3,
            'height': DEFAULT_INPUT_SIZE,
            'width': DEFAULT_INPUT_SIZE,
            'dtype': 'float32',
            'normalization': {
                'mean': IMAGENET_MEAN,
                'std': IMAGENET_STD
            }
        },
        'output_spec': {
            'format': 'logits',
            'num_classes': 5,
            'class_names': CLASS_NAMES,
            'class_descriptions': CLASS_DESCRIPTIONS
        },
        'verification': verification_results,
        'exports': export_results
    }

    # Save metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        logger.info(f"✓ Metadata saved to: {output_path}")

    return metadata


# =============================================================================
# Deployment Guide Generation
# =============================================================================

def generate_deployment_guide(
    output_path: Path,
    model_type: str,
    export_results: List[Dict],
    verbose: bool = True
) -> None:
    """
    Generate deployment guide with usage examples.

    Parameters
    ----------
    output_path : Path
        Output path for deployment guide markdown
    model_type : str
        Type of model (baseline/lora)
    export_results : list
        List of export results for each format
    verbose : bool
        Whether to print generation information
    """
    if verbose:
        logger.info("Generating deployment guide...")

    # Create deployment guide content
    guide = f"""# Deployment Guide for {model_type.upper()} Model

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This guide provides instructions for deploying the exported diabetic retinopathy classification model in various production environments.

## Model Information

- **Model Type**: {model_type.upper()}
- **Task**: 5-class diabetic retinopathy classification
- **Input Size**: 224x224x3 (RGB images)
- **Output**: 5 class logits (No DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR)

## Preprocessing Requirements

All input images must be preprocessed as follows:

```python
# Python preprocessing example
import numpy as np
from PIL import Image

# Load image
image = Image.open('retinal_image.jpg').convert('RGB')
image = image.resize((224, 224))

# Convert to numpy array and normalize
image_np = np.array(image).astype(np.float32) / 255.0

# Apply ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_np = (image_np - mean) / std

# Transpose to NCHW format
image_np = np.transpose(image_np, (2, 0, 1))

# Add batch dimension
image_np = np.expand_dims(image_np, axis=0)
```

## Exported Formats

"""

    # Add format-specific sections
    for export_result in export_results:
        if export_result['status'] != 'success':
            continue

        format_type = export_result['format']

        if format_type == 'onnx':
            guide += """
### ONNX Runtime Deployment

**File**: `{}`
**Size**: {:.2f} MB
**Opset Version**: {}

#### Python Example

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('{}')

# Run inference (image_np from preprocessing above)
outputs = session.run(None, {{'input': image_np}})
logits = outputs[0]

# Get prediction
probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
predicted_class = np.argmax(logits, axis=1)[0]
confidence = probabilities[0][predicted_class]

class_names = {{
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}}

print(f"Prediction: {{class_names[predicted_class]}} (confidence: {{confidence:.2%}})")
```

#### C++ Example

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// Initialize ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DR_Classification");
Ort::SessionOptions session_options;
Ort::Session session(env, "model.onnx", session_options);

// Prepare input tensor
std::vector<float> input_tensor_values = ...; // preprocessed image data
std::vector<int64_t> input_shape = {{1, 3, 224, 224}};

auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_tensor_values.data(), input_tensor_values.size(),
    input_shape.data(), input_shape.size());

// Run inference
const char* input_names[] = {{"input"}};
const char* output_names[] = {{"output"}};
auto output_tensors = session.Run(Ort::RunOptions{{nullptr}}, input_names, &input_tensor, 1, output_names, 1);

// Get output
float* output_data = output_tensors[0].GetTensorMutableData<float>();
```

""".format(
                export_result['output_path'],
                export_result['file_size_mb'],
                export_result.get('opset_version', DEFAULT_ONNX_OPSET),
                export_result['output_path']
            )

        elif format_type == 'torchscript':
            guide += """
### TorchScript Deployment

**File**: `{}`
**Size**: {:.2f} MB
**Method**: {}

#### Python Example

```python
import torch
import numpy as np

# Load model
model = torch.jit.load('{}')
model.eval()

# Run inference (image_np from preprocessing above)
input_tensor = torch.from_numpy(image_np)

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

class_names = {{
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}}

print(f"Prediction: {{class_names[predicted_class]}} (confidence: {{confidence:.2%}})")
```

#### C++ (LibTorch) Example

```cpp
#include <torch/script.h>
#include <torch/torch.h>

// Load model
torch::jit::script::Module module = torch::jit::load("model.pt");
module.eval();

// Prepare input tensor
std::vector<float> input_data = ...; // preprocessed image data
auto input_tensor = torch::from_blob(input_data.data(), {{1, 3, 224, 224}});

// Run inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);
at::Tensor output = module.forward(inputs).toTensor();

// Get prediction
auto probabilities = torch::softmax(output, 1);
auto [max_prob, predicted_class] = torch::max(probabilities, 1);
```

#### Mobile Deployment (iOS/Android)

For mobile deployment, use PyTorch Mobile:

```python
# Optimize for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile

mobile_model = optimize_for_mobile(model)
mobile_model._save_for_lite_interpreter("model_mobile.ptl")
```

""".format(
                export_result['output_path'],
                export_result['file_size_mb'],
                export_result.get('method', 'trace'),
                export_result['output_path']
            )

        elif format_type == 'lora_adapters':
            guide += """
### LoRA Adapter Deployment

**File**: `{}`
**Size**: {:.2f} MB
**Adapter Parameters**: {:,}

LoRA adapters can be loaded separately and swapped without reloading the base model:

```python
import torch
from scripts.retfound_lora import RETFoundLoRA

# Load base RETFound model
base_model = RETFoundLoRA(
    checkpoint_path='models/RETFound_cfp_weights.pth',
    num_classes=5,
    lora_r=8,
    lora_alpha=32
)

# Load and apply adapters
adapter_checkpoint = torch.load('{}')
base_model.load_state_dict(adapter_checkpoint['lora_adapters'], strict=False)
base_model.eval()

# Run inference
with torch.no_grad():
    output = base_model(input_tensor)
```

**Adapter Swapping**: Load different adapters without reinitializing the base model:

```python
# Swap to different adapter
new_adapter = torch.load('adapter_v2.pth')
base_model.load_state_dict(new_adapter['lora_adapters'], strict=False)
```

""".format(
                export_result['output_path'],
                export_result['file_size_mb'],
                export_result.get('adapter_parameters', 0),
                export_result['output_path']
            )

    # Add performance optimization section
    guide += """
## Performance Optimization

### GPU Optimization

For NVIDIA GPUs, consider using TensorRT for optimal performance:

```python
# ONNX to TensorRT conversion
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX model
with open('model.onnx', 'rb') as f:
    parser.parse(f.read())

# Build TensorRT engine
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)
```

### Batch Processing

For higher throughput, process multiple images in batches:

```python
# Batch inference
batch_images = np.stack([img1, img2, img3, img4])  # Shape: (4, 3, 224, 224)
outputs = session.run(None, {{'input': batch_images}})
```

### Model Quantization

If quantized model is available, it offers:
- 2-4x faster inference
- 2-4x smaller model size
- Minimal accuracy loss (~1-2%)

## Production Checklist

- [ ] Verify preprocessing matches training (mean/std normalization)
- [ ] Test model with sample images before deployment
- [ ] Implement error handling for invalid inputs
- [ ] Set up logging for prediction monitoring
- [ ] Consider confidence thresholds for high-stakes predictions
- [ ] Implement fallback mechanism for uncertain predictions
- [ ] Monitor inference latency and throughput
- [ ] Set up A/B testing infrastructure if needed
- [ ] Document model version and update procedures

## Medical Disclaimer

This model is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult with qualified healthcare professionals for medical decisions.

## Support

For issues or questions, refer to the project documentation or contact the development team.

---

**Last Updated**: {}
""".format(time.strftime('%Y-%m-%d %H:%M:%S'))

    # Save deployment guide
    with open(output_path, 'w') as f:
        f.write(guide)

    if verbose:
        logger.info(f"✓ Deployment guide saved to: {output_path}")


# =============================================================================
# Main Export Pipeline
# =============================================================================

def export_model_pipeline(
    checkpoint_path: Path,
    output_dir: Path,
    formats: List[str],
    model_type: str = 'auto',
    retfound_checkpoint: Optional[Path] = None,
    optimization_level: int = 1,
    quantize: bool = False,
    validate: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete model export pipeline.

    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    output_dir : Path
        Output directory for exported models
    formats : list
        List of export formats: ['onnx', 'torchscript', 'lora', 'all']
    model_type : str
        Model type: 'auto', 'baseline', or 'lora'
    retfound_checkpoint : Path, optional
        Path to RETFound foundation model (for LoRA)
    optimization_level : int
        Optimization level (0-3)
    quantize : bool
        Whether to quantize the model
    validate : bool
        Whether to validate exported models
    verbose : bool
        Whether to print detailed information

    Returns
    -------
    dict
        Complete export results
    """
    logger.info("=" * 80)
    logger.info("DIABETIC RETINOPATHY MODEL EXPORT PIPELINE")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Step 1: Load model
    logger.info("\n[Step 1/6] Loading model...")
    try:
        model, detected_type, config = load_model(
            checkpoint_path, device, model_type, retfound_checkpoint, verbose
        )
        original_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {'status': 'failed', 'error': str(e)}

    # Step 2: Verify model
    logger.info("\n[Step 2/6] Verifying model...")
    verification_results = verify_model(model, DEFAULT_INPUT_SIZE, device)
    if verification_results['status'] != 'success':
        logger.error("Model verification failed!")
        return {
            'status': 'failed',
            'verification': verification_results
        }

    # Expand 'all' format to individual formats
    if 'all' in formats:
        if detected_type == 'lora':
            formats = ['onnx', 'torchscript', 'lora']
        else:
            formats = ['onnx', 'torchscript']

    # Step 3: Export to requested formats
    logger.info(f"\n[Step 3/6] Exporting to formats: {formats}")
    export_results = []

    if 'onnx' in formats:
        onnx_path = output_dir / f"{checkpoint_path.stem}.onnx"
        export_result = export_to_onnx(
            model, onnx_path, DEFAULT_INPUT_SIZE,
            DEFAULT_ONNX_OPSET, True, verbose
        )
        export_results.append(export_result)

        # Validate ONNX if requested
        if validate and export_result['status'] == 'success':
            validation_result = validate_onnx_model(
                onnx_path, model, DEFAULT_INPUT_SIZE, 1e-5, verbose
            )
            export_result['validation'] = validation_result

    if 'torchscript' in formats:
        ts_path = output_dir / f"{checkpoint_path.stem}_torchscript.pt"
        export_result = export_to_torchscript(
            model, ts_path, DEFAULT_INPUT_SIZE, 'trace', verbose
        )
        export_results.append(export_result)

        # Validate TorchScript if requested
        if validate and export_result['status'] == 'success':
            validation_result = validate_torchscript_model(
                ts_path, model, DEFAULT_INPUT_SIZE, 1e-5, verbose
            )
            export_result['validation'] = validation_result

    if 'lora' in formats and detected_type == 'lora':
        lora_path = output_dir / f"{checkpoint_path.stem}_lora_adapters.pth"
        export_result = export_lora_adapters(
            model, lora_path, original_checkpoint, verbose
        )
        export_results.append(export_result)

    # Step 4: Quantization (if requested)
    if quantize and optimization_level >= 2:
        logger.info("\n[Step 4/6] Quantizing model...")
        quant_path = output_dir / f"{checkpoint_path.stem}_quantized.pth"
        quant_result = quantize_model(model, quant_path, 'dynamic', verbose)
        export_results.append(quant_result)
    else:
        logger.info("\n[Step 4/6] Skipping quantization")

    # Step 5: Generate metadata
    logger.info("\n[Step 5/6] Generating metadata...")
    metadata_path = output_dir / f"{checkpoint_path.stem}_metadata.json"
    metadata = generate_metadata(
        detected_type, config, verification_results,
        export_results, metadata_path, verbose
    )

    # Step 6: Generate deployment guide
    logger.info("\n[Step 6/6] Generating deployment guide...")
    guide_path = output_dir / f"{checkpoint_path.stem}_deployment_guide.md"
    generate_deployment_guide(
        guide_path, detected_type, export_results, verbose
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nExported files:")
    for result in export_results:
        if result['status'] == 'success':
            logger.info(f"  ✓ {result['output_path']} ({result['file_size_mb']:.2f} MB)")
        else:
            logger.warning(f"  ✗ {result.get('format', 'unknown')} export failed")
    logger.info(f"  ✓ {metadata_path}")
    logger.info(f"  ✓ {guide_path}")
    logger.info("=" * 80)

    return {
        'status': 'success',
        'model_type': detected_type,
        'verification': verification_results,
        'exports': export_results,
        'metadata_path': str(metadata_path),
        'guide_path': str(guide_path)
    }


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export trained PyTorch models to deployment formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX
  python scripts/export_model.py --checkpoint model.pth --format onnx --output model.onnx

  # Export to TorchScript
  python scripts/export_model.py --checkpoint model.pth --format torchscript --output model.pt

  # Export LoRA adapters
  python scripts/export_model.py --checkpoint lora_model.pth --format lora --output lora_adapters.pth

  # Export all formats to directory
  python scripts/export_model.py --checkpoint model.pth --format all --output_dir exports/

  # Export with quantization
  python scripts/export_model.py --checkpoint model.pth --format all --output_dir exports/ --quantize

  # Skip validation for faster export
  python scripts/export_model.py --checkpoint model.pth --format onnx --output model.onnx --no_validate
        """
    )

    # Input arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='auto',
        choices=['auto', 'baseline', 'lora'],
        help='Model type (default: auto-detect)'
    )

    parser.add_argument(
        '--retfound_checkpoint',
        type=str,
        default=None,
        help='Path to RETFound foundation model (required for LoRA models)'
    )

    # Export format arguments
    parser.add_argument(
        '--format',
        type=str,
        required=True,
        choices=EXPORT_FORMATS,
        help='Export format(s): onnx, torchscript, lora, or all'
    )

    # Output arguments
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output',
        type=str,
        help='Output file path (for single format export)'
    )

    output_group.add_argument(
        '--output_dir',
        type=str,
        help='Output directory (for multiple format export)'
    )

    # Optimization arguments
    parser.add_argument(
        '--optimization_level',
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help='Optimization level (0=none, 1=basic, 2=moderate, 3=aggressive)'
    )

    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply quantization for model compression'
    )

    # Validation arguments
    parser.add_argument(
        '--no_validate',
        action='store_true',
        help='Skip validation of exported models'
    )

    # ONNX-specific arguments
    parser.add_argument(
        '--onnx_opset',
        type=int,
        default=DEFAULT_ONNX_OPSET,
        help=f'ONNX opset version (default: {DEFAULT_ONNX_OPSET})'
    )

    # TorchScript-specific arguments
    parser.add_argument(
        '--torchscript_method',
        type=str,
        default='trace',
        choices=['trace', 'script'],
        help='TorchScript export method (default: trace)'
    )

    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during export'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Convert paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    retfound_checkpoint = Path(args.retfound_checkpoint) if args.retfound_checkpoint else None

    # Determine output path(s)
    if args.output:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Single format export
        if args.format == 'all':
            logger.error("Cannot use --output with --format all. Use --output_dir instead.")
            sys.exit(1)

        # Manual export for single format
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        model, model_type, config = load_model(
            checkpoint_path, device, args.model_type,
            retfound_checkpoint, args.verbose
        )

        # Verify model
        verification_results = verify_model(model, DEFAULT_INPUT_SIZE, device)
        if verification_results['status'] != 'success':
            logger.error("Model verification failed!")
            sys.exit(1)

        # Export
        output_path = Path(args.output)
        if args.format == 'onnx':
            result = export_to_onnx(
                model, output_path, DEFAULT_INPUT_SIZE,
                args.onnx_opset, True, args.verbose
            )
            if not args.no_validate and result['status'] == 'success':
                validate_onnx_model(output_path, model, DEFAULT_INPUT_SIZE, 1e-5, args.verbose)

        elif args.format == 'torchscript':
            result = export_to_torchscript(
                model, output_path, DEFAULT_INPUT_SIZE,
                args.torchscript_method, args.verbose
            )
            if not args.no_validate and result['status'] == 'success':
                validate_torchscript_model(output_path, model, DEFAULT_INPUT_SIZE, 1e-5, args.verbose)

        elif args.format == 'lora':
            if model_type != 'lora':
                logger.error("Cannot export LoRA adapters from non-LoRA model")
                sys.exit(1)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            result = export_lora_adapters(model, output_path, checkpoint, args.verbose)

        if result['status'] != 'success':
            logger.error(f"Export failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

        logger.info(f"\n✓ Export successful: {output_path}")

    else:
        # Multiple format export
        output_dir = Path(args.output_dir)

        # Run complete pipeline
        results = export_model_pipeline(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            formats=[args.format],
            model_type=args.model_type,
            retfound_checkpoint=retfound_checkpoint,
            optimization_level=args.optimization_level,
            quantize=args.quantize,
            validate=not args.no_validate,
            verbose=args.verbose
        )

        if results['status'] != 'success':
            logger.error(f"Export pipeline failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == '__main__':
    main()
