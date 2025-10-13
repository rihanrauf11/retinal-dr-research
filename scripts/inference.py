#!/usr/bin/env python3
"""
Production-Ready Inference Script for Diabetic Retinopathy Classification

This script provides a complete inference pipeline for DR classification models,
supporting both baseline models (DRClassifier) and LoRA-adapted models (RETFoundLoRA).

Key Features:
    - Auto-detection of model type from checkpoint
    - Single image and batch inference
    - Confidence scores for all classes
    - Timing/performance metrics
    - JSON output format
    - Optional visualization
    - Comprehensive error handling
    - Optimized for speed (eval mode, no_grad)

Usage:
    # Single image inference
    python scripts/inference.py \\
        --image path/to/image.jpg \\
        --checkpoint checkpoints/best_model.pth

    # Batch inference
    python scripts/inference.py \\
        --images data/test/*.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --batch_size 32

    # With JSON output
    python scripts/inference.py \\
        --image path/to/image.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --output_json results/prediction.json

    # With visualization
    python scripts/inference.py \\
        --image path/to/image.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --visualize \\
        --output_image results/prediction_viz.png

Author: Generated with Claude Code
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Project modules
try:
    from scripts.model import DRClassifier
    from scripts.retfound_lora import RETFoundLoRA
    from scripts.retfound_model import load_retfound_model
except ModuleNotFoundError:
    from model import DRClassifier
    from retfound_lora import RETFoundLoRA
    from retfound_model import load_retfound_model


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

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

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate that image file exists and is a valid format.

    Args:
        image_path: Path to image file

    Returns:
        True if valid, False otherwise
    """
    image_path = Path(image_path)

    # Check file exists
    if not image_path.exists():
        return False

    # Check extension
    if image_path.suffix not in SUPPORTED_FORMATS:
        return False

    # Try to open image
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Load image from file with error handling.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image if successful, None otherwise
    """
    try:
        image = Image.open(image_path)
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def get_transform(img_size: int = 224) -> A.Compose:
    """
    Get preprocessing transform for inference.

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def preprocess_single_image(
    image_path: Union[str, Path],
    transform: A.Compose
) -> Optional[torch.Tensor]:
    """
    Preprocess a single image for inference.

    Args:
        image_path: Path to image
        transform: Preprocessing transform

    Returns:
        Preprocessed tensor of shape (1, 3, H, W) or None if failed
    """
    # Load image
    image = load_image(image_path)
    if image is None:
        return None

    try:
        # Convert to numpy array
        image_np = np.array(image)

        # Apply transforms
        transformed = transform(image=image_np)
        image_tensor = transformed['image']

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    except Exception as e:
        print(f"[ERROR] Failed to preprocess {image_path}: {e}")
        return None


def preprocess_batch(
    image_paths: List[Union[str, Path]],
    transform: A.Compose
) -> Tuple[torch.Tensor, List[Union[str, Path]]]:
    """
    Preprocess a batch of images for inference.

    Args:
        image_paths: List of image paths
        transform: Preprocessing transform

    Returns:
        (batch_tensor, valid_paths) - only includes successfully loaded images
    """
    batch_tensors = []
    valid_paths = []

    for image_path in image_paths:
        tensor = preprocess_single_image(image_path, transform)
        if tensor is not None:
            batch_tensors.append(tensor)
            valid_paths.append(image_path)

    if not batch_tensors:
        return torch.empty(0), []

    # Stack into batch
    batch_tensor = torch.cat(batch_tensors, dim=0)
    return batch_tensor, valid_paths


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def detect_model_type(checkpoint_path: str) -> str:
    """
    Auto-detect model type from checkpoint structure.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Model type: 'lora', 'baseline', or 'retfound'

    Raises:
        ValueError: If checkpoint format is unknown
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Check for LoRA model
    if 'lora_adapters' in checkpoint or 'lora_config' in checkpoint:
        return 'lora'

    # Check for baseline model
    if 'model_state_dict' in checkpoint:
        return 'baseline'

    # Check for raw RETFound model
    if 'model' in checkpoint:
        return 'retfound'

    raise ValueError(
        f"Unknown checkpoint format. Expected keys: 'lora_adapters', "
        f"'model_state_dict', or 'model'. Found: {list(checkpoint.keys())}"
    )


def load_baseline_model(
    checkpoint_path: str,
    device: torch.device,
    verbose: bool = True
) -> nn.Module:
    """
    Load DRClassifier model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        verbose: Print loading info

    Returns:
        Loaded model in evaluation mode
    """
    if verbose:
        print(f"[INFO] Loading baseline model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model configuration
    model_name = checkpoint.get('model_name', 'resnet50')
    num_classes = checkpoint.get('num_classes', 5)

    if verbose:
        print(f"[INFO] Architecture: {model_name}")
        print(f"[INFO] Classes: {num_classes}")

    # Create model
    model = DRClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False  # We're loading weights
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    if verbose:
        total, trainable = model.get_num_params()
        print(f"[INFO] Parameters: {total:,}")

    return model


def load_lora_model(
    checkpoint_path: str,
    device: torch.device,
    retfound_checkpoint: Optional[str] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Load RETFoundLoRA model from checkpoint.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        device: Device to load model on
        retfound_checkpoint: Optional path to RETFound base weights
        verbose: Print loading info

    Returns:
        Loaded model in evaluation mode
    """
    if verbose:
        print(f"[INFO] Loading LoRA model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get LoRA configuration
    lora_config = checkpoint.get('lora_config', {})
    retfound_path = retfound_checkpoint or lora_config.get('checkpoint_path')
    lora_r = lora_config.get('r', 8)
    lora_alpha = lora_config.get('alpha', 32)
    num_classes = checkpoint.get('num_classes', 5)

    if not retfound_path:
        raise ValueError(
            "LoRA checkpoint does not contain RETFound path. "
            "Please provide it via --retfound_checkpoint argument."
        )

    if verbose:
        print(f"[INFO] RETFound base: {retfound_path}")
        print(f"[INFO] LoRA rank: {lora_r}, alpha: {lora_alpha}")

    # Create model (suppress verbose output during creation)
    import io
    import contextlib

    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            model = RETFoundLoRA(
                checkpoint_path=retfound_path,
                num_classes=num_classes,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                device=device
            )
    else:
        model = RETFoundLoRA(
            checkpoint_path=retfound_path,
            num_classes=num_classes,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=device
        )

    # Load LoRA adapters
    if 'lora_adapters' in checkpoint:
        model.backbone.load_state_dict(checkpoint['lora_adapters'], strict=False)

    # Load classifier
    if 'classifier' in checkpoint:
        model.classifier.load_state_dict(checkpoint['classifier'])

    # Set to eval mode
    model.eval()

    return model


def load_model(
    checkpoint_path: str,
    device: torch.device,
    model_type: str = 'auto',
    retfound_checkpoint: Optional[str] = None,
    verbose: bool = True
) -> Tuple[nn.Module, str]:
    """
    Load model from checkpoint with automatic type detection.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
        model_type: 'auto', 'baseline', or 'lora'
        retfound_checkpoint: Optional RETFound base weights path
        verbose: Print loading info

    Returns:
        (model, detected_model_type)
    """
    # Detect model type if auto
    if model_type == 'auto':
        model_type = detect_model_type(checkpoint_path)
        if verbose:
            print(f"[INFO] Detected model type: {model_type}")

    # Load appropriate model
    if model_type == 'baseline':
        model = load_baseline_model(checkpoint_path, device, verbose)
    elif model_type == 'lora':
        model = load_lora_model(checkpoint_path, device, retfound_checkpoint, verbose)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, model_type


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference_single(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> Tuple[int, np.ndarray, float]:
    """
    Run inference on a single image.

    Args:
        model: Model in eval mode
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        device: Device to run inference on

    Returns:
        (predicted_class, confidence_scores, inference_time)
    """
    # Move to device
    image_tensor = image_tensor.to(device)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence_scores = probabilities[0].cpu().numpy()

    inference_time = time.time() - start_time

    return predicted_class, confidence_scores, inference_time


def run_inference_batch(
    model: nn.Module,
    batch_tensor: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference on a batch of images.

    Args:
        model: Model in eval mode
        batch_tensor: Batch of preprocessed images (B, 3, H, W)
        device: Device to run inference on

    Returns:
        (predicted_classes, confidence_scores, inference_time)
    """
    # Move to device
    batch_tensor = batch_tensor.to(device)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        logits = model(batch_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
        confidence_scores = probabilities.cpu().numpy()

    inference_time = time.time() - start_time

    return predicted_classes, confidence_scores, inference_time


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_prediction(
    image_path: Union[str, Path],
    predicted_class: int,
    confidence_scores: np.ndarray,
    inference_time: float
) -> Dict[str, Any]:
    """
    Format prediction results as dictionary.

    Args:
        image_path: Path to image
        predicted_class: Predicted class index
        confidence_scores: Confidence scores for all classes
        inference_time: Time taken for inference

    Returns:
        Dictionary with prediction results
    """
    # Get class name
    predicted_name = CLASS_NAMES[predicted_class]
    predicted_description = CLASS_DESCRIPTIONS[predicted_class]

    # Format confidence scores
    all_confidences = {
        CLASS_NAMES[i]: float(confidence_scores[i])
        for i in range(len(confidence_scores))
    }

    return {
        'image_path': str(image_path),
        'predicted_class': int(predicted_class),
        'predicted_class_name': predicted_name,
        'predicted_class_description': predicted_description,
        'confidence': float(confidence_scores[predicted_class]),
        'all_confidences': all_confidences,
        'inference_time_ms': float(inference_time * 1000)
    }


def format_batch_predictions(
    image_paths: List[Union[str, Path]],
    predicted_classes: np.ndarray,
    confidence_scores: np.ndarray,
    total_time: float
) -> Dict[str, Any]:
    """
    Format batch prediction results.

    Args:
        image_paths: List of image paths
        predicted_classes: Array of predicted classes
        confidence_scores: Array of confidence scores
        total_time: Total inference time

    Returns:
        Dictionary with all predictions and timing info
    """
    predictions = []
    for i, image_path in enumerate(image_paths):
        pred = format_prediction(
            image_path=image_path,
            predicted_class=predicted_classes[i],
            confidence_scores=confidence_scores[i],
            inference_time=total_time / len(image_paths)  # Average per image
        )
        predictions.append(pred)

    return {
        'predictions': predictions,
        'timing': {
            'total_time_seconds': float(total_time),
            'per_image_ms': float((total_time / len(image_paths)) * 1000),
            'throughput_images_per_second': float(len(image_paths) / total_time)
        },
        'summary': {
            'num_images': len(image_paths),
            'timestamp': datetime.now().isoformat()
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_prediction(
    image_path: Union[str, Path],
    predicted_class: int,
    confidence_scores: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> None:
    """
    Visualize prediction with annotated image and confidence bar chart.

    Args:
        image_path: Path to original image
        predicted_class: Predicted class index
        confidence_scores: Confidence scores for all classes
        output_path: Optional path to save visualization
        show: Whether to display the plot
    """
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"[ERROR] Cannot visualize - failed to load {image_path}")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Image with prediction
    ax1.imshow(image)
    ax1.axis('off')

    # Add prediction text
    predicted_name = CLASS_NAMES[predicted_class]
    confidence = confidence_scores[predicted_class]

    title = f"Prediction: {predicted_name}\nConfidence: {confidence:.1%}"
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Plot 2: Confidence bar chart
    class_names = [CLASS_NAMES[i] for i in range(len(confidence_scores))]
    colors = ['green' if i == predicted_class else 'lightblue'
              for i in range(len(confidence_scores))]

    ax2.barh(class_names, confidence_scores, color=colors, edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence Scores', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)

    # Add percentage labels
    for i, v in enumerate(confidence_scores):
        ax2.text(v + 0.02, i, f'{v:.1%}', va='center', fontsize=10)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Visualization saved to: {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Production-ready inference engine for DR classification.

    This class provides a high-level interface for model inference with
    automatic preprocessing, batch processing, and result formatting.

    Attributes:
        model: Loaded PyTorch model
        device: Device (cuda/cpu)
        model_type: Type of model ('baseline' or 'lora')
        transform: Preprocessing transform
        checkpoint_path: Path to model checkpoint
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        model_type: str = 'auto',
        retfound_checkpoint: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            model_type: 'auto', 'baseline', or 'lora'
            retfound_checkpoint: Optional RETFound base weights
            verbose: Print initialization info
        """
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        # Setup device
        self.device = torch.device(
            device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )

        if verbose:
            print("=" * 70)
            print("INITIALIZING INFERENCE ENGINE")
            print("=" * 70)
            print(f"Device: {self.device}")

        # Load model
        start_time = time.time()
        self.model, self.model_type = load_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            model_type=model_type,
            retfound_checkpoint=retfound_checkpoint,
            verbose=verbose
        )
        load_time = time.time() - start_time

        # Setup preprocessing
        self.transform = get_transform(img_size=224)

        if verbose:
            print(f"\n[INFO] Model loaded in {load_time:.2f}s")
            print("=" * 70)
            print("READY FOR INFERENCE")
            print("=" * 70 + "\n")

    def predict_single(
        self,
        image_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image

        Returns:
            Prediction dictionary or None if failed
        """
        # Validate image
        if not validate_image(image_path):
            print(f"[ERROR] Invalid image: {image_path}")
            return None

        # Preprocess
        image_tensor = preprocess_single_image(image_path, self.transform)
        if image_tensor is None:
            return None

        # Run inference
        predicted_class, confidence_scores, inference_time = run_inference_single(
            model=self.model,
            image_tensor=image_tensor,
            device=self.device
        )

        # Format results
        result = format_prediction(
            image_path=image_path,
            predicted_class=predicted_class,
            confidence_scores=confidence_scores,
            inference_time=inference_time
        )

        return result

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference

        Returns:
            Dictionary with all predictions
        """
        all_predictions = []
        total_time = 0.0

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Preprocess batch
            batch_tensor, valid_paths = preprocess_batch(batch_paths, self.transform)

            if batch_tensor.shape[0] == 0:
                continue

            # Run inference
            predicted_classes, confidence_scores, inference_time = run_inference_batch(
                model=self.model,
                batch_tensor=batch_tensor,
                device=self.device
            )

            total_time += inference_time

            # Format results
            for j, image_path in enumerate(valid_paths):
                pred = format_prediction(
                    image_path=image_path,
                    predicted_class=predicted_classes[j],
                    confidence_scores=confidence_scores[j],
                    inference_time=inference_time / len(valid_paths)
                )
                all_predictions.append(pred)

        # Format batch results
        result = {
            'predictions': all_predictions,
            'timing': {
                'total_time_seconds': float(total_time),
                'per_image_ms': float((total_time / len(all_predictions)) * 1000) if all_predictions else 0,
                'throughput_images_per_second': float(len(all_predictions) / total_time) if total_time > 0 else 0
            },
            'model_info': {
                'type': self.model_type,
                'checkpoint': str(self.checkpoint_path),
                'device': str(self.device)
            },
            'summary': {
                'num_images': len(image_paths),
                'num_successful': len(all_predictions),
                'num_failed': len(image_paths) - len(all_predictions),
                'timestamp': datetime.now().isoformat()
            }
        }

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Production inference for diabetic retinopathy classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python scripts/inference.py --image path/to/image.jpg --checkpoint best_model.pth

  # Batch inference
  python scripts/inference.py --images data/test/*.jpg --checkpoint best_model.pth

  # With JSON output
  python scripts/inference.py --image test.jpg --checkpoint model.pth --output_json results.json

  # With visualization
  python scripts/inference.py --image test.jpg --checkpoint model.pth --visualize --output_image viz.png

  # LoRA model
  python scripts/inference.py --image test.jpg --checkpoint lora.pth --model_type lora --retfound_checkpoint RETFound_cfp_weights.pth
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to single image'
    )
    input_group.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Paths to multiple images or glob pattern'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing images'
    )

    # Model options
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
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
        help='Path to RETFound base weights (for LoRA models)'
    )

    # Inference options
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    # Output options
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization'
    )
    parser.add_argument(
        '--output_image',
        type=str,
        default=None,
        help='Path to save visualization (requires --visualize)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display visualization (requires --visualize)'
    )

    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output (except errors)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine verbosity
    verbose = not args.quiet and (args.verbose or True)

    try:
        # Initialize inference engine
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
            model_type=args.model_type,
            retfound_checkpoint=args.retfound_checkpoint,
            verbose=verbose
        )

        # Collect image paths
        image_paths = []

        if args.image:
            image_paths = [args.image]
        elif args.images:
            # Handle glob patterns
            for pattern in args.images:
                if '*' in pattern or '?' in pattern:
                    image_paths.extend(glob.glob(pattern))
                else:
                    image_paths.append(pattern)
        elif args.image_dir:
            image_dir = Path(args.image_dir)
            for ext in SUPPORTED_FORMATS:
                image_paths.extend(image_dir.glob(f'*{ext}'))

        if not image_paths:
            print("[ERROR] No images found")
            sys.exit(1)

        if verbose:
            print(f"[INFO] Found {len(image_paths)} images")

        # Run inference
        if len(image_paths) == 1:
            # Single image
            result = engine.predict_single(image_paths[0])

            if result is None:
                print("[ERROR] Inference failed")
                sys.exit(1)

            # Print results
            if not args.quiet:
                print("\n" + "=" * 70)
                print("PREDICTION RESULTS")
                print("=" * 70)
                print(f"Image: {result['image_path']}")
                print(f"Prediction: {result['predicted_class_name']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Inference time: {result['inference_time_ms']:.1f} ms")
                print("\nAll confidences:")
                for class_name, conf in result['all_confidences'].items():
                    print(f"  {class_name:20s}: {conf:.1%}")
                print("=" * 70 + "\n")

            # Wrap in standard format
            final_result = {
                'predictions': [result],
                'model_info': {
                    'type': engine.model_type,
                    'checkpoint': str(engine.checkpoint_path)
                },
                'summary': {
                    'num_images': 1,
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Visualization
            if args.visualize:
                visualize_prediction(
                    image_path=image_paths[0],
                    predicted_class=result['predicted_class'],
                    confidence_scores=np.array(list(result['all_confidences'].values())),
                    output_path=args.output_image,
                    show=args.show
                )

        else:
            # Batch inference
            if verbose:
                print(f"\n[INFO] Running batch inference on {len(image_paths)} images...")

            final_result = engine.predict_batch(
                image_paths=image_paths,
                batch_size=args.batch_size
            )

            # Print summary
            if not args.quiet:
                print("\n" + "=" * 70)
                print("BATCH INFERENCE RESULTS")
                print("=" * 70)
                print(f"Total images: {final_result['summary']['num_images']}")
                print(f"Successful: {final_result['summary']['num_successful']}")
                print(f"Failed: {final_result['summary']['num_failed']}")
                print(f"\nTiming:")
                print(f"  Total time: {final_result['timing']['total_time_seconds']:.2f}s")
                print(f"  Per image: {final_result['timing']['per_image_ms']:.1f} ms")
                print(f"  Throughput: {final_result['timing']['throughput_images_per_second']:.1f} images/s")
                print("=" * 70 + "\n")

        # Save JSON output
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(final_result, f, indent=2)

            if verbose:
                print(f"[INFO] Results saved to: {output_path}")

        # Print JSON to stdout if no output file specified
        if not args.output_json and not args.quiet:
            print("\nJSON Output:")
            print(json.dumps(final_result, indent=2))

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
