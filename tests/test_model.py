"""
Unit Tests for DRClassifier Model.

Tests cover:
- Model creation with different backbones
- Forward pass and output shapes
- Parameter management (freeze/unfreeze)
- Transfer learning workflows
- Integration with config system
- Error handling

Author: Generated with Claude Code
"""

import pytest
import torch
import torch.nn as nn

from model import DRClassifier, get_model_summary


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CREATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelCreation:
    """Test model initialization with different backbones and parameters."""

    def test_model_creation_resnet50(self):
        """Test creating model with ResNet50 backbone."""
        model = DRClassifier(
            model_name='resnet50',
            num_classes=5,
            pretrained=False,
            dropout_rate=0.3
        )

        assert model is not None
        assert model.model_name == 'resnet50'
        assert model.num_classes == 5
        assert model.dropout_rate == 0.3

    def test_model_creation_efficientnet(self):
        """Test creating model with EfficientNet backbone."""
        model = DRClassifier(
            model_name='efficientnet_b0',
            num_classes=5,
            pretrained=False
        )

        assert model.model_name == 'efficientnet_b0'
        assert model.num_classes == 5

    def test_model_creation_mobilenet(self):
        """Test creating model with MobileNet backbone."""
        model = DRClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=5,
            pretrained=False
        )

        assert model.model_name == 'mobilenetv3_small_100'

    def test_model_creation_resnet18(self):
        """Test creating smaller ResNet18 for faster tests."""
        model = DRClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=False
        )

        assert model.model_name == 'resnet18'
        total_params, _ = model.get_num_params()
        assert total_params > 0

    def test_model_invalid_name(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="not found in timm"):
            DRClassifier(
                model_name='nonexistent_model_xyz123',
                num_classes=5
            )

    def test_model_invalid_num_classes_negative(self):
        """Test that negative num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            DRClassifier(
                model_name='resnet18',
                num_classes=-1
            )

    def test_model_invalid_num_classes_zero(self):
        """Test that zero num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            DRClassifier(
                model_name='resnet18',
                num_classes=0
            )

    def test_model_invalid_dropout_negative(self):
        """Test that negative dropout raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DRClassifier(
                model_name='resnet18',
                dropout_rate=-0.1
            )

    def test_model_invalid_dropout_too_large(self):
        """Test that dropout > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DRClassifier(
                model_name='resnet18',
                dropout_rate=1.5
            )

    def test_model_pretrained_vs_random(self):
        """Test that pretrained and random models have same structure."""
        model_pretrained = DRClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=True
        )

        model_random = DRClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=False
        )

        # Both should have same number of parameters
        total1, _ = model_pretrained.get_num_params()
        total2, _ = model_random.get_num_params()
        assert total1 == total2


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelForwardPass:
    """Test model forward pass with various inputs."""

    def test_forward_pass_basic(self, simple_model):
        """Test basic forward pass with standard input."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        output = simple_model(input_tensor)

        assert output.shape == (batch_size, 5)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_output_shape(self, simple_model):
        """Test that output shape matches (batch_size, num_classes)."""
        test_inputs = [
            (1, 3, 224, 224),
            (4, 3, 224, 224),
            (8, 3, 224, 224),
        ]

        for input_shape in test_inputs:
            batch_size = input_shape[0]
            input_tensor = torch.randn(*input_shape)
            output = simple_model(input_tensor)

            assert output.shape == (batch_size, 5), \
                f"Expected shape ({batch_size}, 5), got {output.shape}"

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_forward_different_batch_sizes(self, simple_model, batch_size):
        """Test forward pass with different batch sizes."""
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        output = simple_model(input_tensor)

        assert output.shape == (batch_size, 5)

    @pytest.mark.parametrize("img_size", [224, 384, 512])
    def test_forward_different_image_sizes(self, simple_model, img_size):
        """Test forward pass with different image sizes."""
        input_tensor = torch.randn(2, 3, img_size, img_size)
        output = simple_model(input_tensor)

        # Output shape should still be (batch_size, num_classes)
        assert output.shape == (2, 5)

    def test_forward_single_image(self, simple_model):
        """Test forward pass with batch size = 1."""
        input_tensor = torch.randn(1, 3, 224, 224)
        output = simple_model(input_tensor)

        assert output.shape == (1, 5)

    def test_forward_gradient_flow(self, simple_model):
        """Test that gradients can flow through the model."""
        simple_model.train()

        input_tensor = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = simple_model(input_tensor)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)

    def test_forward_eval_mode(self, simple_model):
        """Test forward pass in evaluation mode."""
        simple_model.eval()

        with torch.no_grad():
            input_tensor = torch.randn(2, 3, 224, 224)
            output = simple_model(input_tensor)

        assert output.shape == (2, 5)

    def test_forward_determinism(self, simple_model):
        """Test that forward pass is deterministic in eval mode."""
        simple_model.eval()

        input_tensor = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output1 = simple_model(input_tensor)
            output2 = simple_model(input_tensor)

        assert torch.allclose(output1, output2)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelParameterManagement:
    """Test parameter counting, freezing, and unfreezing."""

    def test_get_num_params(self, simple_model):
        """Test parameter counting."""
        total_params, trainable_params = simple_model.get_num_params()

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_all_params_trainable_initially(self, simple_model):
        """Test that all parameters are trainable initially."""
        total_params, trainable_params = simple_model.get_num_params()
        assert total_params == trainable_params

    def test_freeze_backbone(self, simple_model):
        """Test freezing backbone reduces trainable parameters."""
        _, trainable_before = simple_model.get_num_params()

        simple_model.freeze_backbone()

        _, trainable_after = simple_model.get_num_params()

        assert trainable_after < trainable_before

    def test_unfreeze_backbone(self, simple_model):
        """Test unfreezing backbone restores trainable parameters."""
        total_params, trainable_initial = simple_model.get_num_params()

        # Freeze then unfreeze
        simple_model.freeze_backbone()
        simple_model.unfreeze_backbone()

        _, trainable_after = simple_model.get_num_params()

        assert trainable_after == trainable_initial
        assert trainable_after == total_params

    def test_freeze_unfreeze_cycle(self, simple_model):
        """Test multiple freeze/unfreeze cycles."""
        total_params, trainable_initial = simple_model.get_num_params()

        for _ in range(3):
            simple_model.freeze_backbone()
            _, trainable_frozen = simple_model.get_num_params()
            assert trainable_frozen < total_params

            simple_model.unfreeze_backbone()
            _, trainable_unfrozen = simple_model.get_num_params()
            assert trainable_unfrozen == total_params

    def test_classifier_always_trainable(self, simple_model):
        """Test that classifier head remains trainable when backbone is frozen."""
        simple_model.freeze_backbone()

        # Check that classifier parameters are trainable
        for param in simple_model.classifier.parameters():
            assert param.requires_grad

        # Check that backbone parameters are frozen
        for param in simple_model.backbone.parameters():
            assert not param.requires_grad

    def test_parameter_counts_match(self, simple_model):
        """Test that manual parameter count matches get_num_params()."""
        total_counted = sum(p.numel() for p in simple_model.parameters())
        total_reported, _ = simple_model.get_num_params()

        assert total_counted == total_reported


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY METHODS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelUtilityMethods:
    """Test utility methods and helpers."""

    def test_get_feature_dim(self, simple_model):
        """Test getting feature dimension."""
        feat_dim = simple_model.get_feature_dim()

        assert isinstance(feat_dim, int)
        assert feat_dim > 0

    def test_model_repr(self, simple_model):
        """Test model string representation."""
        repr_str = repr(simple_model)

        assert 'DRClassifier' in repr_str
        assert 'resnet18' in repr_str
        assert 'num_classes=5' in repr_str

    def test_model_summary(self, simple_model, capsys):
        """Test model summary generation."""
        get_model_summary(simple_model, input_size=(2, 3, 224, 224))

        captured = capsys.readouterr()
        assert 'MODEL ARCHITECTURE SUMMARY' in captured.out
        assert 'Parameter Count' in captured.out
        assert 'Forward Pass Test' in captured.out

    @pytest.mark.gpu
    def test_model_device_transfer_cuda(self, simple_model):
        """Test transferring model to CUDA (requires GPU)."""
        device = torch.device('cuda')
        simple_model.to(device)

        # Test forward pass on GPU
        input_tensor = torch.randn(2, 3, 224, 224, device=device)
        output = simple_model(input_tensor)

        assert output.device.type == 'cuda'

    def test_model_device_transfer_cpu(self, simple_model):
        """Test that model works on CPU."""
        device = torch.device('cpu')
        simple_model.to(device)

        input_tensor = torch.randn(2, 3, 224, 224, device=device)
        output = simple_model(input_tensor)

        assert output.device.type == 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE-SPECIFIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelArchitectures:
    """Test different backbone architectures."""

    @pytest.mark.parametrize("model_name", [
        'resnet18',
        'resnet34',
        'efficientnet_b0',
        'mobilenetv3_small_100',
    ])
    def test_different_backbones(self, model_name):
        """Test creating models with different backbones."""
        model = DRClassifier(
            model_name=model_name,
            num_classes=5,
            pretrained=False
        )

        # Test forward pass
        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)

        assert output.shape == (2, 5)

    def test_cnn_backbone_spatial_features(self):
        """Test that CNN backbones handle spatial features correctly."""
        model = DRClassifier('resnet18', num_classes=5, pretrained=False)

        # Feed through backbone only
        input_tensor = torch.randn(2, 3, 224, 224)
        features = model.backbone(input_tensor)

        # Features should be spatial (B, C, H, W) or flat (B, C)
        assert features.dim() in [2, 4]

    @pytest.mark.parametrize("num_classes", [2, 5, 10, 100])
    def test_different_num_classes(self, num_classes):
        """Test models with different numbers of output classes."""
        model = DRClassifier(
            model_name='resnet18',
            num_classes=num_classes,
            pretrained=False
        )

        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)

        assert output.shape == (2, num_classes)

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.3, 0.5, 0.7])
    def test_different_dropout_rates(self, dropout_rate):
        """Test models with different dropout rates."""
        model = DRClassifier(
            model_name='resnet18',
            num_classes=5,
            dropout_rate=dropout_rate
        )

        assert model.dropout_rate == dropout_rate


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelIntegration:
    """Test model integration with training workflows."""

    def test_model_with_optimizer(self, simple_model):
        """Test creating optimizer with model parameters."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        assert optimizer is not None
        assert len(optimizer.param_groups) > 0

    def test_model_with_loss_function(self, simple_model):
        """Test model with loss function."""
        criterion = nn.CrossEntropyLoss()

        input_tensor = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])

        output = simple_model(input_tensor)
        loss = criterion(output, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_model_training_step(self, simple_model):
        """Test complete training step."""
        simple_model.train()

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        input_tensor = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])

        # Forward pass
        output = simple_model(input_tensor)
        loss = criterion(output, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without errors
        assert loss.item() > 0

    def test_model_inference_step(self, simple_model):
        """Test inference mode."""
        simple_model.eval()

        with torch.no_grad():
            input_tensor = torch.randn(2, 3, 224, 224)
            output = simple_model(input_tensor)

            # Get predictions
            probabilities = torch.softmax(output, dim=1)
            predicted_classes = torch.argmax(output, dim=1)

        assert probabilities.shape == (2, 5)
        assert predicted_classes.shape == (2,)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))

    @pytest.mark.slow
    def test_model_overfitting_small_batch(self, simple_model):
        """Test that model can overfit a small batch (sanity check)."""
        simple_model.train()

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Create small batch
        input_tensor = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])

        initial_loss = None

        # Train for a few iterations
        for _ in range(50):
            output = simple_model(input_tensor)
            loss = criterion(output, labels)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelConfigIntegration:
    """Test integration with configuration system."""

    def test_model_from_config(self):
        """Test creating model from config."""
        try:
            from config import Config, ModelConfig

            config = Config()
            model = DRClassifier.from_config(config.model)

            assert model.model_name == config.model.model_name
            assert model.num_classes == config.model.num_classes

        except ImportError:
            pytest.skip("Config module not available")

    def test_model_from_yaml_config(self):
        """Test creating model from YAML config."""
        try:
            from config import Config
            import os

            # Try to find a config file
            config_path = 'configs/default_config.yaml'
            if not os.path.exists(config_path):
                pytest.skip("Config file not found")

            config = Config.from_yaml(config_path)
            model = DRClassifier.from_config(config.model)

            assert model is not None

        except ImportError:
            pytest.skip("Config module not available")


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES AND ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_model_with_very_small_image(self, simple_model):
        """Test model with very small input (may fail gracefully)."""
        # Some architectures may not support very small inputs
        # Use eval mode to avoid batch norm issues with batch size 1
        simple_model.eval()
        input_tensor = torch.randn(1, 3, 32, 32)

        try:
            with torch.no_grad():
                output = simple_model(input_tensor)
            # If it works, check output shape
            assert output.shape == (1, 5)
        except (RuntimeError, ValueError):
            # Some models may not support small inputs or batch size 1
            pytest.skip("Model doesn't support small input sizes or batch size 1")

    def test_model_state_dict_save_load(self, simple_model, temp_data_dir):
        """Test saving and loading model state dict."""
        checkpoint_path = temp_data_dir / "model.pth"

        # Save
        torch.save(simple_model.state_dict(), checkpoint_path)

        # Create new model and load
        new_model = DRClassifier('resnet18', num_classes=5, pretrained=False)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Test that outputs match
        input_tensor = torch.randn(2, 3, 224, 224)

        simple_model.eval()
        new_model.eval()

        with torch.no_grad():
            output1 = simple_model(input_tensor)
            output2 = new_model(input_tensor)

        assert torch.allclose(output1, output2, atol=1e-5)

    def test_model_with_nan_input(self, simple_model):
        """Test model behavior with NaN input."""
        input_tensor = torch.randn(2, 3, 224, 224)
        input_tensor[0, 0, 0, 0] = float('nan')

        output = simple_model(input_tensor)

        # Output will likely contain NaN
        assert torch.isnan(output).any()

    def test_model_gradient_clipping(self, simple_model):
        """Test that gradient clipping works."""
        simple_model.train()

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        input_tensor = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])

        output = simple_model(input_tensor)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_norm=1.0)

        # Check that gradients are clipped
        total_norm = 0
        for p in simple_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Total norm should be <= 1.0 (with some tolerance)
        assert total_norm <= 1.1  # Small tolerance for numerical precision
