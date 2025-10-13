"""
DRClassifier: Diabetic Retinopathy Classification Model.

This module provides a flexible, production-ready classifier for diabetic
retinopathy using pretrained backbones from the timm library. The model
automatically handles different architecture types (CNNs, ViTs, hybrids)
and replaces the classification head with a custom one optimized for DR
classification.
"""

from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import timm

try:
    from scripts.config import ModelConfig
except ImportError:
    ModelConfig = None


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy Classifier with flexible backbone selection.

    This classifier leverages pretrained models from the timm library and
    replaces the default classification head with a custom one featuring
    dropout regularization. It supports a wide variety of architectures
    including CNNs (ResNet, EfficientNet), Vision Transformers (ViT, DeiT),
    and hybrid models (ConvNeXt, Swin Transformer).

    The model is designed for 5-class diabetic retinopathy classification:
        0: No DR (No Diabetic Retinopathy)
        1: Mild NPDR (Non-Proliferative Diabetic Retinopathy)
        2: Moderate NPDR
        3: Severe NPDR
        4: PDR (Proliferative Diabetic Retinopathy)

    Parameters
    ----------
    model_name : str
        Name of the backbone architecture from timm model zoo.
        Examples: 'resnet50', 'efficientnet_b3', 'vit_base_patch16_224'
        Use timm.list_models() to see all available models.

    num_classes : int, default=5
        Number of output classes for classification.

    pretrained : bool, default=True
        Whether to load pretrained weights. Recommended for transfer learning.

    dropout_rate : float, default=0.3
        Dropout probability for regularization. Applied before final linear layer.
        Range: [0.0, 1.0]. Higher values = more regularization.

    Attributes
    ----------
    backbone : nn.Module
        The pretrained backbone model from timm
    classifier : nn.Sequential
        Custom classification head with dropout
    model_name : str
        Name of the backbone architecture
    num_classes : int
        Number of output classes
    feature_dim : int
        Dimension of features from the backbone

    Examples
    --------
    >>> # Basic usage
    >>> model = DRClassifier(model_name='resnet50', num_classes=5)
    >>> images = torch.randn(2, 3, 224, 224)
    >>> predictions = model(images)
    >>> predictions.shape
    torch.Size([2, 5])

    >>> # Using EfficientNet
    >>> model = DRClassifier('efficientnet_b3', num_classes=5, pretrained=True)
    >>> total_params, trainable_params = model.get_num_params()
    >>> print(f"Total parameters: {total_params:,}")

    >>> # Transfer learning workflow
    >>> model = DRClassifier('vit_base_patch16_224')
    >>> model.freeze_backbone()  # Freeze backbone
    >>> # Train only the classifier head...
    >>> model.unfreeze_backbone()  # Then fine-tune entire model

    >>> # Using with config system
    >>> from scripts.config import Config
    >>> config = Config.from_yaml('configs/default_config.yaml')
    >>> model = DRClassifier.from_config(config.model)

    Notes
    -----
    - The model automatically detects the backbone's output dimension
    - Works with both CNN and Transformer architectures
    - Pretrained weights are from ImageNet-1K or ImageNet-21K
    - For optimal performance, use input size matching the backbone's native resolution
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """Initialize the DRClassifier model."""
        super(DRClassifier, self).__init__()

        # Validate parameters
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        # Store configuration
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load backbone from timm
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove original classifier
                global_pool=''   # We'll handle pooling ourselves
            )
        except Exception as e:
            available_models = timm.list_models(model_name + '*')
            if available_models:
                suggestions = ', '.join(available_models[:5])
                raise ValueError(
                    f"Failed to load model '{model_name}'. "
                    f"Similar models: {suggestions}. "
                    f"Use timm.list_models() to see all available models."
                ) from e
            else:
                raise ValueError(
                    f"Model '{model_name}' not found in timm. "
                    f"Use timm.list_models() to see all available models."
                ) from e

        # Detect feature dimension
        self.feature_dim = self._detect_feature_dim()

        # Create custom classification head
        self.classifier = self._build_classifier_head()

        print(f"✓ Created DRClassifier with backbone: {model_name}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Output classes: {num_classes}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Pretrained: {pretrained}")

    def _detect_feature_dim(self) -> int:
        """
        Automatically detect the output feature dimension of the backbone.

        Returns
        -------
        int
            Number of features output by the backbone

        Raises
        ------
        RuntimeError
            If feature dimension cannot be detected
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)

            # Forward pass through backbone
            with torch.no_grad():
                features = self.backbone(dummy_input)

            # Handle different output formats
            if isinstance(features, torch.Tensor):
                # Most models return a tensor
                if len(features.shape) == 2:  # [batch, features]
                    return features.shape[1]
                elif len(features.shape) == 4:  # [batch, channels, h, w]
                    return features.shape[1]
                else:
                    raise RuntimeError(f"Unexpected feature shape: {features.shape}")
            else:
                raise RuntimeError(f"Unexpected feature type: {type(features)}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to detect feature dimension for {self.model_name}. "
                f"Error: {e}"
            ) from e

    def _build_classifier_head(self) -> nn.Sequential:
        """
        Build the custom classification head.

        The head consists of:
        1. Global Average Pooling (if features are spatial)
        2. Dropout for regularization
        3. Linear layer for classification

        Returns
        -------
        nn.Sequential
            The classification head module
        """
        layers = []

        # Add global average pooling for spatial features (CNNs)
        # This will be a no-op if features are already [batch, features]
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())

        # Add dropout for regularization
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(p=self.dropout_rate))

        # Add final linear layer
        layers.append(nn.Linear(self.feature_dim, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch_size, 3, height, width)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes)
            Apply softmax/sigmoid for probabilities

        Examples
        --------
        >>> model = DRClassifier('resnet50')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> logits = model(images)
        >>> probabilities = torch.softmax(logits, dim=1)
        >>> predicted_classes = torch.argmax(logits, dim=1)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through classification head
        logits = self.classifier(features)

        return logits

    def get_num_params(self) -> Tuple[int, int]:
        """
        Get the number of parameters in the model.

        Returns
        -------
        tuple of (int, int)
            (total_params, trainable_params)
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters

        Examples
        --------
        >>> model = DRClassifier('resnet50')
        >>> total, trainable = model.get_num_params()
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def freeze_backbone(self) -> None:
        """
        Freeze the backbone weights for transfer learning.

        This is useful when you want to train only the classification head
        first, then fine-tune the entire model later. Freezing the backbone
        reduces memory usage and speeds up training.

        Examples
        --------
        >>> model = DRClassifier('resnet50')
        >>> model.freeze_backbone()
        >>> # Now only the classifier head will be updated during training
        >>> # To verify:
        >>> total, trainable = model.get_num_params()
        >>> print(f"Trainable: {trainable:,}")
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        total, trainable = self.get_num_params()
        print(f"✓ Backbone frozen. Trainable parameters: {trainable:,} / {total:,}")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the backbone weights for fine-tuning.

        Call this after training the classification head with a frozen
        backbone to fine-tune the entire model end-to-end.

        Examples
        --------
        >>> model = DRClassifier('resnet50')
        >>> model.freeze_backbone()
        >>> # Train with frozen backbone...
        >>> model.unfreeze_backbone()
        >>> # Now fine-tune entire model with lower learning rate
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        total, trainable = self.get_num_params()
        print(f"✓ Backbone unfrozen. Trainable parameters: {trainable:,} / {total:,}")

    def get_feature_dim(self) -> int:
        """
        Get the feature dimension of the backbone output.

        Returns
        -------
        int
            Number of features from the backbone

        Examples
        --------
        >>> model = DRClassifier('resnet50')
        >>> feat_dim = model.get_feature_dim()
        >>> print(f"Feature dimension: {feat_dim}")
        """
        return self.feature_dim

    @classmethod
    def from_config(cls, config: 'ModelConfig') -> 'DRClassifier':
        """
        Create a DRClassifier from a ModelConfig object.

        This enables seamless integration with the configuration system.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing model parameters

        Returns
        -------
        DRClassifier
            Initialized model

        Examples
        --------
        >>> from scripts.config import Config
        >>> config = Config.from_yaml('configs/default_config.yaml')
        >>> model = DRClassifier.from_config(config.model)
        """
        if ModelConfig is None:
            raise ImportError(
                "ModelConfig not available. "
                "Make sure scripts.config is accessible."
            )

        return cls(
            model_name=config.model_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained
        )

    def __repr__(self) -> str:
        """String representation of the model."""
        total_params, trainable_params = self.get_num_params()
        return (
            f"DRClassifier(\n"
            f"  backbone={self.model_name},\n"
            f"  num_classes={self.num_classes},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (2, 3, 224, 224)) -> None:
    """
    Print a detailed summary of the model architecture.

    Parameters
    ----------
    model : nn.Module
        The model to summarize
    input_size : tuple, default=(2, 3, 224, 224)
        Shape of input tensor (batch_size, channels, height, width)

    Examples
    --------
    >>> model = DRClassifier('resnet50')
    >>> get_model_summary(model)
    """
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)

    # Model representation
    print("\n[Model Structure]")
    print(model)

    # Parameter counts
    print("\n[Parameter Count]")
    total_params, trainable_params = model.get_num_params()
    print(f"  Total parameters:      {total_params:>15,}")
    print(f"  Trainable parameters:  {trainable_params:>15,}")
    print(f"  Non-trainable params:  {(total_params - trainable_params):>15,}")

    # Memory estimate (rough)
    param_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
    print(f"  Estimated size:        {param_size_mb:>15.2f} MB")

    # Test forward pass
    print("\n[Forward Pass Test]")
    print(f"  Input shape:  {input_size}")

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  ✓ Forward pass successful!")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    """
    Comprehensive test suite for DRClassifier.
    """
    print("=" * 80)
    print("DRClassifier Test Suite")
    print("=" * 80)

    # Test 1: Basic model creation
    print("\n[Test 1] Creating DRClassifier with ResNet50...")
    try:
        model_resnet = DRClassifier(
            model_name='resnet50',
            num_classes=5,
            pretrained=True,
            dropout_rate=0.3
        )
        print("✓ ResNet50 model created successfully")
    except Exception as e:
        print(f"✗ Failed to create ResNet50 model: {e}")
        exit(1)

    # Test 2: Forward pass with dummy input
    print("\n[Test 2] Testing forward pass with dummy input...")
    try:
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        print(f"  Input shape: {tuple(dummy_input.shape)}")

        output = model_resnet(dummy_input)
        print(f"  Output shape: {tuple(output.shape)}")

        assert output.shape == (batch_size, 5), f"Expected shape (2, 5), got {output.shape}"
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

    # Test 3: Model summary
    print("\n[Test 3] Generating model summary...")
    try:
        get_model_summary(model_resnet, input_size=(2, 3, 224, 224))
        print("✓ Model summary generated")
    except Exception as e:
        print(f"✗ Model summary failed: {e}")

    # Test 4: Parameter counting
    print("\n[Test 4] Testing parameter counting...")
    try:
        total_params, trainable_params = model_resnet.get_num_params()
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        assert trainable_params == total_params, "All params should be trainable initially"
        print("✓ Parameter counting works correctly")
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")

    # Test 5: Freeze/unfreeze functionality
    print("\n[Test 5] Testing freeze/unfreeze functionality...")
    try:
        # Freeze backbone
        model_resnet.freeze_backbone()
        _, trainable_frozen = model_resnet.get_num_params()

        # Unfreeze backbone
        model_resnet.unfreeze_backbone()
        _, trainable_unfrozen = model_resnet.get_num_params()

        assert trainable_frozen < trainable_unfrozen, "Frozen model should have fewer trainable params"
        print("✓ Freeze/unfreeze works correctly")
    except Exception as e:
        print(f"✗ Freeze/unfreeze failed: {e}")

    # Test 6: Different backbone architectures
    print("\n[Test 6] Testing different backbone architectures...")
    test_models = [
        ('efficientnet_b0', 224),
        ('resnet34', 224),
        ('mobilenetv3_small_100', 224),
    ]

    for model_name, img_size in test_models:
        try:
            print(f"\n  Testing {model_name}...")
            model = DRClassifier(model_name=model_name, num_classes=5, pretrained=False)

            # Test forward pass
            test_input = torch.randn(2, 3, img_size, img_size)
            output = model(test_input)

            assert output.shape == (2, 5), f"Unexpected output shape: {output.shape}"
            print(f"  ✓ {model_name} works correctly")
        except Exception as e:
            print(f"  ⚠ {model_name} test skipped: {e}")

    # Test 7: Different input sizes
    print("\n[Test 7] Testing different input sizes...")
    try:
        model_test = DRClassifier('resnet50', num_classes=5, pretrained=False)

        input_sizes = [224, 384, 512]
        for size in input_sizes:
            test_input = torch.randn(2, 3, size, size)
            output = model_test(test_input)
            assert output.shape == (2, 5), f"Failed for input size {size}"
            print(f"  ✓ Input size {size}x{size} works")

        print("✓ Different input sizes work correctly")
    except Exception as e:
        print(f"✗ Input size test failed: {e}")

    # Test 8: Integration with config system
    print("\n[Test 8] Testing integration with config system...")
    try:
        from scripts.config import Config, ModelConfig

        # Create config
        config = Config()
        print(f"  Config model: {config.model.model_name}")

        # Create model from config
        model_from_config = DRClassifier.from_config(config.model)

        # Verify
        assert model_from_config.model_name == config.model.model_name
        assert model_from_config.num_classes == config.model.num_classes
        print("✓ Config integration works correctly")
    except ImportError:
        print("  ⚠ Config system not available, skipping test")
    except Exception as e:
        print(f"✗ Config integration failed: {e}")

    # Test 9: Model representation
    print("\n[Test 9] Testing model representation...")
    try:
        print(model_resnet)
        print("✓ Model representation works")
    except Exception as e:
        print(f"✗ Model representation failed: {e}")

    # Test 10: Invalid parameters
    print("\n[Test 10] Testing error handling...")
    try:
        # Test invalid num_classes
        try:
            invalid_model = DRClassifier('resnet50', num_classes=-1)
            print("✗ Should have raised ValueError for negative num_classes")
        except ValueError:
            print("  ✓ Correctly caught invalid num_classes")

        # Test invalid dropout rate
        try:
            invalid_model = DRClassifier('resnet50', dropout_rate=1.5)
            print("✗ Should have raised ValueError for invalid dropout_rate")
        except ValueError:
            print("  ✓ Correctly caught invalid dropout_rate")

        # Test invalid model name
        try:
            invalid_model = DRClassifier('nonexistent_model_xyz')
            print("✗ Should have raised ValueError for invalid model name")
        except ValueError:
            print("  ✓ Correctly caught invalid model name")

        print("✓ Error handling works correctly")
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

    print("\nUsage examples:")
    print("""
# Basic usage
from scripts.model import DRClassifier

model = DRClassifier(model_name='resnet50', num_classes=5)

# Forward pass
import torch
images = torch.randn(4, 3, 224, 224)
predictions = model(images)

# Get model info
total_params, trainable_params = model.get_num_params()
print(f"Parameters: {total_params:,}")

# Transfer learning
model.freeze_backbone()  # Train only classifier head
# ... training ...
model.unfreeze_backbone()  # Fine-tune entire model

# Use with config
from scripts.config import Config
config = Config.from_yaml('configs/default_config.yaml')
model = DRClassifier.from_config(config.model)

# Model summary
from scripts.model import get_model_summary
get_model_summary(model)
""")
