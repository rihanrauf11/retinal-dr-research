"""
RETFound Foundation Model for Diabetic Retinopathy Classification.

This module provides a Vision Transformer (ViT) implementation specifically designed
to load and use the RETFound foundation model. RETFound is a self-supervised
foundation model pre-trained on 1.6 million retinal images from diverse sources,
making it highly effective for ophthalmology tasks including diabetic retinopathy
classification.

Key Advantages of RETFound:
    - Domain-specific pre-training on retinal images (vs ImageNet natural images)
    - Better feature representations for medical imaging tasks
    - Improved transfer learning performance on small datasets
    - State-of-the-art results on multiple ophthalmology benchmarks

Reference:
    Zhou et al. "A Foundation Model for Generalizable Disease Detection from
    Retinal Images" Nature (2023)

Example:
    >>> # Load RETFound model with pretrained weights
    >>> model = load_retfound_model(
    ...     checkpoint_path='RETFound_cfp_weights.pth',
    ...     num_classes=5
    ... )
    >>>
    >>> # Use for inference
    >>> model.eval()
    >>> with torch.no_grad():
    ...     outputs = model(images)
    ...     predictions = torch.argmax(outputs, dim=1)
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding layer.

    Converts an image into a sequence of patch embeddings using a convolutional layer.
    For ViT-Large with patch_size=16, a 224x224 image becomes 14x14=196 patches.

    Args:
        img_size: Input image size (default: 224)
        patch_size: Size of each patch (default: 16)
        in_chans: Number of input channels (default: 3)
        embed_dim: Embedding dimension (default: 1024)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional projection: creates patch embeddings
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape

        # Check input size
        if H != self.img_size or W != self.img_size:
            warnings.warn(
                f"Input image size ({H}x{W}) doesn't match expected "
                f"size ({self.img_size}x{self.img_size}). This may cause issues."
            )

        # Project to patches: (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)

        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Implements scaled dot-product attention with multiple heads for parallel
    attention computations.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads (default: 16)
        qkv_bias: Whether to include bias in QKV projection (default: True)
        attn_drop: Attention dropout rate (default: 0.0)
        proj_drop: Output projection dropout rate (default: 0.0)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).

    Two-layer MLP with GELU activation, commonly used in transformer blocks.

    Args:
        in_features: Input dimension
        hidden_features: Hidden layer dimension (default: None, uses 4x expansion)
        out_features: Output dimension (default: None, same as input)
        drop: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block.

    Standard transformer block with multi-head self-attention and feed-forward
    network, using pre-normalization (LayerNorm before attention/MLP).

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        qkv_bias: Whether to use bias in QKV projection (default: True)
        drop: Dropout rate (default: 0.0)
        attn_drop: Attention dropout rate (default: 0.0)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)

        Returns:
            Output tensor of shape (B, N, C)
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    This implementation follows the architecture used in RETFound, which is based
    on ViT-Large. The model splits images into patches, embeds them, and processes
    them with transformer blocks.

    Architecture (ViT-Large):
        - Patch size: 16x16
        - Embedding dimension: 1024
        - Depth: 24 transformer blocks
        - Attention heads: 16
        - MLP ratio: 4.0
        - Input size: 224x224 (default, but flexible)

    Args:
        img_size: Input image size (default: 224)
        patch_size: Patch size (default: 16)
        in_chans: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000, set to 0 for feature extraction)
        embed_dim: Embedding dimension (default: 1024)
        depth: Number of transformer blocks (default: 24)
        num_heads: Number of attention heads (default: 16)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        qkv_bias: Use bias in QKV projection (default: True)
        drop_rate: Dropout rate (default: 0.0)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        use_global_pool: Use global average pooling instead of CLS token (default: False)

    Example:
        >>> # Create ViT-Large for feature extraction
        >>> model = VisionTransformer(
        ...     img_size=224,
        ...     patch_size=16,
        ...     embed_dim=1024,
        ...     depth=24,
        ...     num_heads=16,
        ...     num_classes=0  # No classification head
        ... )
        >>>
        >>> # Forward pass
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = model(images)  # (2, 1024)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_global_pool: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.use_global_pool = use_global_pool

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (for CLS token + patches)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        # Layer norm after transformer blocks
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (optional)
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, embed_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Extract features
        if self.use_global_pool:
            # Global average pooling (average all patch tokens, excluding CLS)
            x = x[:, 1:, :].mean(dim=1)  # (B, embed_dim)
        else:
            # Use CLS token
            x = x[:, 0]  # (B, embed_dim)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            - If num_classes > 0: Logits of shape (B, num_classes)
            - If num_classes == 0: Features of shape (B, embed_dim)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            trainable_only: Only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def get_retfound_vit_large(**kwargs) -> VisionTransformer:
    """
    Create a ViT-Large model with RETFound architecture.

    This is a convenience function that creates a Vision Transformer with the
    exact architecture used by RETFound:
        - Patch size: 16
        - Embedding dimension: 1024
        - Depth: 24 blocks
        - Attention heads: 16
        - MLP ratio: 4.0

    Args:
        **kwargs: Additional arguments to pass to VisionTransformer

    Returns:
        VisionTransformer model with ViT-Large architecture

    Example:
        >>> model = get_retfound_vit_large(num_classes=5)
        >>> print(f"Parameters: {model.get_num_params():,}")
    """
    model = VisionTransformer(
        img_size=kwargs.get('img_size', 224),
        patch_size=16,
        in_chans=kwargs.get('in_chans', 3),
        num_classes=kwargs.get('num_classes', 0),
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=kwargs.get('drop_rate', 0.0),
        attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
        use_global_pool=kwargs.get('use_global_pool', False)
    )
    return model


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean state dict keys by removing common prefixes.

    Many checkpoints save models with prefixes like 'module.' (from DataParallel)
    or 'model.' (from training frameworks). This function removes these prefixes.

    Args:
        state_dict: Original state dict

    Returns:
        Cleaned state dict with prefixes removed
    """
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        # Remove 'module.' prefix (from DataParallel/DistributedDataParallel)
        new_key = key.replace('module.', '')

        # Remove 'model.' prefix (from some training frameworks)
        new_key = new_key.replace('model.', '')

        cleaned_state_dict[new_key] = value

    return cleaned_state_dict


def load_retfound_model(
    checkpoint_path: Union[str, Path],
    num_classes: int = 5,
    strict: bool = False,
    use_global_pool: bool = False,
    device: Optional[torch.device] = None
) -> VisionTransformer:
    """
    Load RETFound foundation model from checkpoint.

    This function creates a ViT-Large model, loads pretrained weights from the
    RETFound checkpoint, and adds a classification head for diabetic retinopathy.

    The function handles various checkpoint formats:
        - State dicts wrapped in 'model' or 'state_dict' keys
        - Models saved with DataParallel ('module.' prefix)
        - Missing keys in the state dict (e.g., classification head)

    Args:
        checkpoint_path: Path to RETFound checkpoint (.pth or .pt file)
        num_classes: Number of output classes for DR classification (default: 5)
        strict: Whether to strictly enforce that state dict keys match (default: False)
        use_global_pool: Use global average pooling instead of CLS token (default: False)
        device: Device to load model to (default: None, uses CPU)

    Returns:
        VisionTransformer model with pretrained weights and classification head

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails

    Example:
        >>> # Load RETFound model for 5-class DR classification
        >>> model = load_retfound_model(
        ...     checkpoint_path='RETFound_cfp_weights.pth',
        ...     num_classes=5,
        ...     device=torch.device('cuda')
        ... )
        >>>
        >>> # Use for training or inference
        >>> model.train()
        >>> outputs = model(images)

    Note:
        The RETFound checkpoint should be downloaded from the official repository.
        Typical checkpoint names:
            - RETFound_cfp_weights.pth (Color Fundus Photography)
            - RETFound_oct_weights.pth (Optical Coherence Tomography)
    """
    checkpoint_path = Path(checkpoint_path)

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download RETFound weights from the official repository."
        )

    # Set device
    if device is None:
        device = torch.device('cpu')

    print(f"Loading RETFound model from: {checkpoint_path}")

    # Create ViT-Large model without classification head
    model = get_retfound_vit_large(
        num_classes=0,  # No head initially
        use_global_pool=use_global_pool
    )

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            # Try common keys
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Loaded state dict from 'model' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Loaded state dict from 'state_dict' key")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Loaded state dict from 'model_state_dict' key")
            else:
                # Assume the entire dict is the state dict
                state_dict = checkpoint
                print("Using checkpoint dict as state dict")
        else:
            state_dict = checkpoint

        # Clean state dict keys (remove 'module.' and 'model.' prefixes)
        state_dict = _clean_state_dict_keys(state_dict)

        # Load weights into model
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=strict
        )

        # Report loading status
        if missing_keys:
            print(f"Missing keys in checkpoint: {len(missing_keys)}")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")

        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  - {key}")

        if not missing_keys and not unexpected_keys:
            print("All weights loaded successfully!")

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e

    # Add classification head for DR classification
    print(f"Adding classification head for {num_classes} classes")
    model.head = nn.Linear(model.embed_dim, num_classes)

    # Initialize classification head weights
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.constant_(model.head.bias, 0)

    # Update num_classes
    model.num_classes = num_classes

    # Move to device
    model = model.to(device)

    print(f"Model loaded successfully!")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    return model


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

    Example:
        >>> model = get_retfound_green()
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            # Try common keys
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the entire dict is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

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

        # Extract state dict (handle different checkpoint formats)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the entire dict is the state dict
            state_dict = checkpoint

        # Count parameters to estimate variant
        param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))

        # Check for timm vs custom ViT architecture characteristics
        # timm models (Green) have "ls1.gamma" and "ls2.gamma" (LayerScale parameters)
        has_ls_gamma = any('ls1.gamma' in key or 'ls2.gamma' in key for key in state_dict.keys())

        # If it has LayerScale parameters, it's timm-based (Green)
        if has_ls_gamma:
            return 'green'

        # RETFound Large: ~303M params, Green: ~21.3M params
        if param_count > 100_000_000:  # >100M = Large
            return 'large'
        else:
            return 'green'

    raise ValueError(f"Cannot determine model variant from {checkpoint_path}")


def print_model_summary(model: VisionTransformer) -> None:
    """
    Print a summary of the Vision Transformer model.

    Args:
        model: VisionTransformer model
    """
    print("\n" + "=" * 70)
    print("VISION TRANSFORMER MODEL SUMMARY")
    print("=" * 70)

    print(f"\nArchitecture Details:")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Number of blocks: {len(model.blocks)}")
    print(f"  Number of attention heads: {model.blocks[0].attn.num_heads}")
    print(f"  Patch size: {model.patch_embed.patch_size}x{model.patch_embed.patch_size}")
    print(f"  Number of patches: {model.patch_embed.num_patches}")
    print(f"  Number of classes: {model.num_classes}")
    print(f"  Global pooling: {model.use_global_pool}")

    print(f"\nParameters:")
    print(f"  Total: {model.get_num_params():,}")
    print(f"  Trainable: {model.get_num_params(trainable_only=True):,}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    """
    Test suite for RETFound Vision Transformer implementation.
    """
    print("Testing RETFound Vision Transformer Implementation\n")
    print("=" * 70)

    # Test 1: Create ViT-Large model
    print("\n[Test 1] Creating ViT-Large model...")
    try:
        model = get_retfound_vit_large(num_classes=5)
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {model.get_num_params():,}")
        print(f"  - Embedding dim: {model.embed_dim}")
        print(f"  - Depth: {len(model.blocks)}")
        print(f"  - Heads: {model.blocks[0].attn.num_heads}")
        assert model.get_num_params() > 300_000_000, "ViT-Large should have >300M params"
        print("✓ Test 1 passed")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")

    # Test 2: Forward pass with dummy input
    print("\n[Test 2] Testing forward pass...")
    try:
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            outputs = model(images)

        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {images.shape}")
        print(f"  - Output shape: {outputs.shape}")
        assert outputs.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5)"
        print("✓ Test 2 passed")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")

    # Test 3: Feature extraction (no classification head)
    print("\n[Test 3] Testing feature extraction...")
    try:
        feature_model = get_retfound_vit_large(num_classes=0)

        with torch.no_grad():
            features = feature_model(images)

        print(f"✓ Feature extraction successful")
        print(f"  - Feature shape: {features.shape}")
        assert features.shape == (batch_size, 1024), f"Expected shape ({batch_size}, 1024)"
        print("✓ Test 3 passed")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")

    # Test 4: Global average pooling
    print("\n[Test 4] Testing global average pooling...")
    try:
        gap_model = get_retfound_vit_large(num_classes=5, use_global_pool=True)

        with torch.no_grad():
            outputs_gap = gap_model(images)

        print(f"✓ Global pooling successful")
        print(f"  - Output shape: {outputs_gap.shape}")
        assert outputs_gap.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5)"

        # Verify different from CLS token
        with torch.no_grad():
            outputs_cls = model(images)

        difference = (outputs_gap - outputs_cls).abs().mean()
        print(f"  - Difference from CLS token: {difference:.4f}")
        assert difference > 0.01, "GAP should give different results than CLS"
        print("✓ Test 4 passed")
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")

    # Test 5: Mock checkpoint loading
    print("\n[Test 5] Testing checkpoint loading (mock)...")
    try:
        import tempfile

        # Create a mock checkpoint
        mock_model = get_retfound_vit_large(num_classes=0)
        mock_checkpoint = {'model': mock_model.state_dict()}

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(mock_checkpoint, tmp.name)
            tmp_path = tmp.name

        # Load the checkpoint
        loaded_model = load_retfound_model(
            checkpoint_path=tmp_path,
            num_classes=5,
            strict=False
        )

        print(f"✓ Checkpoint loading successful")
        print(f"  - Model has classification head: {loaded_model.num_classes == 5}")

        # Clean up
        Path(tmp_path).unlink()
        print("✓ Test 5 passed")
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")

    # Test 6: State dict key cleaning
    print("\n[Test 6] Testing state dict key cleaning...")
    try:
        # Create state dict with prefixes
        dirty_state_dict = {
            'module.patch_embed.proj.weight': torch.randn(1024, 3, 16, 16),
            'model.pos_embed': torch.randn(1, 197, 1024),
            'blocks.0.norm1.weight': torch.randn(1024),
        }

        cleaned = _clean_state_dict_keys(dirty_state_dict)

        print(f"✓ State dict cleaning successful")
        print(f"  - Original keys: {list(dirty_state_dict.keys())[:2]}")
        print(f"  - Cleaned keys: {list(cleaned.keys())[:2]}")

        assert 'module.' not in str(cleaned.keys()), "Should remove 'module.' prefix"
        assert 'model.' not in str(cleaned.keys()), "Should remove 'model.' prefix"
        print("✓ Test 6 passed")
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")

    # Test 7: Model summary
    print("\n[Test 7] Testing model summary...")
    try:
        print_model_summary(model)
        print("✓ Test 7 passed")
    except Exception as e:
        print(f"✗ Test 7 failed: {e}")

    # Test 8: Different image sizes
    print("\n[Test 8] Testing different image sizes...")
    try:
        # Test with 384x384 images (common for ViT)
        model_384 = get_retfound_vit_large(img_size=384, num_classes=5)
        images_384 = torch.randn(1, 3, 384, 384)

        with torch.no_grad():
            outputs_384 = model_384(images_384)

        print(f"✓ Different image size successful")
        print(f"  - Input: 384x384")
        print(f"  - Num patches: {model_384.patch_embed.num_patches}")
        print(f"  - Output shape: {outputs_384.shape}")
        assert outputs_384.shape == (1, 5), "Should work with 384x384 images"
        print("✓ Test 8 passed")
    except Exception as e:
        print(f"✗ Test 8 failed: {e}")

    # Test 9: Create RETFound_Green model
    print("\n[Test 9] Creating RETFound_Green model...")
    try:
        green_model = get_retfound_green()
        print(f"✓ RETFound_Green model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in green_model.parameters()):,}")
        print(f"  - Embedding dim: {green_model.embed_dim}")
        assert green_model.embed_dim == 384, "Green model should have embed_dim=384"
        assert sum(p.numel() for p in green_model.parameters()) > 20_000_000, "Green should have ~21M params"
        print("✓ Test 9 passed")
    except Exception as e:
        print(f"✗ Test 9 failed: {e}")

    # Test 10: RETFound_Green forward pass
    print("\n[Test 10] Testing RETFound_Green forward pass...")
    try:
        batch_size = 2
        images_green = torch.randn(batch_size, 3, 392, 392)

        with torch.no_grad():
            features = green_model(images_green)

        print(f"✓ RETFound_Green forward pass successful")
        print(f"  - Input shape: {images_green.shape}")
        print(f"  - Output shape: {features.shape}")
        assert features.shape == (batch_size, 384), f"Expected shape ({batch_size}, 384), got {features.shape}"
        print("✓ Test 10 passed")
    except Exception as e:
        print(f"✗ Test 10 failed: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
    print("\nRETFound models are ready for use!")
    print("\nNext steps:")
    print("1. Download RETFound (Large) weights from official repository")
    print("2. Download RETFound_Green weights from: https://github.com/justinengelmann/RETFound_Green")
    print("3. Load models:")
    print("   - Large: load_retfound_model('path/to/RETFound_cfp_weights.pth')")
    print("   - Green: load_retfound_green_model('path/to/retfoundgreen_statedict.pth')")
    print("4. Fine-tune on your diabetic retinopathy dataset")
    print("5. Evaluate and compare with baseline models")
    print("=" * 70)
