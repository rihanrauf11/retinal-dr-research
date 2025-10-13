"""
Configuration management system for retinal DR research.

This module provides a comprehensive, type-safe configuration system using
dataclasses with support for YAML serialization and automatic validation.
"""

import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import yaml


@dataclass
class DataConfig:
    """
    Data-related configuration parameters.

    Attributes
    ----------
    train_csv : Optional[str]
        Path to training CSV file with 'id_code' and 'diagnosis' columns
    train_img_dir : Optional[str]
        Directory containing training images
    test_csv : Optional[str]
        Path to test/validation CSV file
    test_img_dir : Optional[str]
        Directory containing test/validation images
    """
    train_csv: Optional[str] = None
    train_img_dir: Optional[str] = None
    test_csv: Optional[str] = None
    test_img_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Attributes
    ----------
    model_name : str
        Name of the model architecture (e.g., 'resnet50', 'vit_base_patch16_224')
    num_classes : int
        Number of output classes (5 for DR: 0-4)
    pretrained : bool
        Whether to use pretrained weights
    """
    model_name: str = "resnet50"
    num_classes: int = 5
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """
    Training hyperparameters configuration.

    Attributes
    ----------
    batch_size : int
        Batch size for training and validation
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Initial learning rate for optimizer
    weight_decay : float
        Weight decay (L2 penalty) for optimizer
    """
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class ImageConfig:
    """
    Image processing configuration.

    Attributes
    ----------
    img_size : int
        Target image size (images will be resized to img_size x img_size)
    """
    img_size: int = 224


@dataclass
class SystemConfig:
    """
    System and hardware configuration.

    Attributes
    ----------
    num_workers : int
        Number of worker processes for data loading
    seed : int
        Random seed for reproducibility
    device : str
        Device to use for training ('cuda', 'cpu', or 'mps' for Apple Silicon)
        Will be auto-detected if not explicitly set
    """
    num_workers: int = 4
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PathConfig:
    """
    Output paths configuration.

    Attributes
    ----------
    checkpoint_dir : str
        Directory to save model checkpoints
    log_dir : str
        Directory to save training logs
    """
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"


@dataclass
class Config:
    """
    Main configuration class aggregating all sub-configurations.

    This class provides a complete configuration management system with:
    - Type-safe parameter definitions
    - YAML serialization/deserialization
    - Automatic validation and directory creation
    - Device auto-detection

    Attributes
    ----------
    data : DataConfig
        Data paths and dataset configuration
    model : ModelConfig
        Model architecture configuration
    training : TrainingConfig
        Training hyperparameters
    image : ImageConfig
        Image processing settings
    system : SystemConfig
        System and hardware settings
    paths : PathConfig
        Output directory paths

    Examples
    --------
    >>> # Create config with defaults
    >>> config = Config()
    >>> print(config.model.model_name)
    'resnet50'

    >>> # Create config with custom parameters
    >>> config = Config(
    ...     data=DataConfig(
    ...         train_csv='data/aptos/train.csv',
    ...         train_img_dir='data/aptos/train_images'
    ...     ),
    ...     training=TrainingConfig(batch_size=32, num_epochs=50)
    ... )

    >>> # Load from YAML
    >>> config = Config.from_yaml('configs/experiment.yaml')

    >>> # Save to YAML
    >>> config.to_yaml('configs/saved_config.yaml')

    >>> # Validate and create directories
    >>> config.validate()
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def __post_init__(self):
        """Post-initialization: auto-detect device and perform basic validation."""
        # Auto-detect device if still default
        if self.system.device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU.",
                UserWarning
            )
            self.system.device = "cpu"

        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if self.system.device == "cpu":
                # Suggest MPS but don't force it
                warnings.warn(
                    "Apple Silicon GPU (MPS) detected but using CPU. "
                    "Consider setting device='mps' for better performance.",
                    UserWarning
                )

    def validate(self, create_dirs: bool = True) -> None:
        """
        Validate configuration and optionally create output directories.

        This method performs the following checks:
        1. Validates numeric parameters (positive values)
        2. Checks that input paths exist (if specified)
        3. Creates output directories if they don't exist

        Parameters
        ----------
        create_dirs : bool, default=True
            If True, create output directories (checkpoint_dir, log_dir)
            if they don't exist. If False, only validate existing paths.

        Raises
        ------
        ValueError
            If any validation check fails
        FileNotFoundError
            If required input files/directories don't exist

        Examples
        --------
        >>> config = Config()
        >>> config.validate()  # Creates output directories
        >>> config.validate(create_dirs=False)  # Only validates, no creation
        """
        # Validate model parameters
        if self.model.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.model.num_classes}")

        # Validate training parameters
        if self.training.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.training.batch_size}")

        if self.training.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.training.num_epochs}")

        if self.training.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.training.learning_rate}")

        # Validate image parameters
        if self.image.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.image.img_size}")

        # Validate system parameters
        if self.system.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.system.num_workers}")

        # Check device validity
        valid_devices = ['cuda', 'cpu', 'mps']
        device_base = self.system.device.split(':')[0]  # Handle 'cuda:0', 'cuda:1', etc.
        if device_base not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.system.device}'. "
                f"Must be one of {valid_devices} (optionally with device index)"
            )

        # Validate input paths (only if they are specified)
        if self.data.train_csv is not None:
            train_csv_path = Path(self.data.train_csv)
            if not train_csv_path.exists():
                raise FileNotFoundError(f"Training CSV not found: {self.data.train_csv}")

        if self.data.train_img_dir is not None:
            train_img_path = Path(self.data.train_img_dir)
            if not train_img_path.exists():
                raise FileNotFoundError(f"Training image directory not found: {self.data.train_img_dir}")

        if self.data.test_csv is not None:
            test_csv_path = Path(self.data.test_csv)
            if not test_csv_path.exists():
                raise FileNotFoundError(f"Test CSV not found: {self.data.test_csv}")

        if self.data.test_img_dir is not None:
            test_img_path = Path(self.data.test_img_dir)
            if not test_img_path.exists():
                raise FileNotFoundError(f"Test image directory not found: {self.data.test_img_dir}")

        # Create output directories if requested
        if create_dirs:
            checkpoint_path = Path(self.paths.checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            log_path = Path(self.paths.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            print(f"✓ Created/verified directory: {checkpoint_path}")
            print(f"✓ Created/verified directory: {log_path}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a nested dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration

        Examples
        --------
        >>> config = Config()
        >>> config_dict = config.to_dict()
        >>> print(config_dict['model']['model_name'])
        'resnet50'
        """
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'image': asdict(self.image),
            'system': asdict(self.system),
            'paths': asdict(self.paths)
        }

    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Parameters
        ----------
        filepath : str or Path
            Path where the YAML file will be saved

        Examples
        --------
        >>> config = Config()
        >>> config.to_yaml('configs/my_config.yaml')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        print(f"✓ Configuration saved to: {filepath}")

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        filepath : str or Path
            Path to the YAML configuration file

        Returns
        -------
        Config
            Configuration object loaded from YAML

        Raises
        ------
        FileNotFoundError
            If the YAML file doesn't exist
        ValueError
            If the YAML file is malformed or missing required fields

        Examples
        --------
        >>> config = Config.from_yaml('configs/experiment.yaml')
        >>> print(config.model.model_name)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)

            if config_dict is None:
                config_dict = {}

            # Create sub-configurations
            data_config = DataConfig(**config_dict.get('data', {}))
            model_config = ModelConfig(**config_dict.get('model', {}))
            training_config = TrainingConfig(**config_dict.get('training', {}))
            image_config = ImageConfig(**config_dict.get('image', {}))
            system_config = SystemConfig(**config_dict.get('system', {}))
            paths_config = PathConfig(**config_dict.get('paths', {}))

            config = cls(
                data=data_config,
                model=model_config,
                training=training_config,
                image=image_config,
                system=system_config,
                paths=paths_config
            )

            print(f"✓ Configuration loaded from: {filepath}")
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameters: {e}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from a dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration parameters

        Returns
        -------
        Config
            Configuration object created from dictionary

        Examples
        --------
        >>> config_dict = {
        ...     'model': {'model_name': 'vit_base_patch16_224'},
        ...     'training': {'batch_size': 32}
        ... }
        >>> config = Config.from_dict(config_dict)
        """
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        image_config = ImageConfig(**config_dict.get('image', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        paths_config = PathConfig(**config_dict.get('paths', {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            image=image_config,
            system=system_config,
            paths=paths_config
        )

    def update(self, **kwargs) -> 'Config':
        """
        Create a new Config with updated parameters.

        This method creates a copy of the current configuration and updates
        specific parameters. Supports nested updates using dot notation in keys.

        Parameters
        ----------
        **kwargs
            Parameters to update. Can use nested keys like 'model.model_name'
            or pass sub-config dictionaries

        Returns
        -------
        Config
            New configuration with updated parameters

        Examples
        --------
        >>> config = Config()
        >>> new_config = config.update(
        ...     training={'batch_size': 32},
        ...     model={'model_name': 'vit_base_patch16_224'}
        ... )
        """
        config_dict = self.to_dict()

        # Update with provided kwargs
        for key, value in kwargs.items():
            if key in config_dict and isinstance(value, dict):
                config_dict[key].update(value)
            elif key in config_dict:
                config_dict[key] = value

        return Config.from_dict(config_dict)

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["Configuration:"]
        lines.append("=" * 70)

        # Data config
        lines.append("\n[Data]")
        lines.append(f"  train_csv      : {self.data.train_csv}")
        lines.append(f"  train_img_dir  : {self.data.train_img_dir}")
        lines.append(f"  test_csv       : {self.data.test_csv}")
        lines.append(f"  test_img_dir   : {self.data.test_img_dir}")

        # Model config
        lines.append("\n[Model]")
        lines.append(f"  model_name     : {self.model.model_name}")
        lines.append(f"  num_classes    : {self.model.num_classes}")
        lines.append(f"  pretrained     : {self.model.pretrained}")

        # Training config
        lines.append("\n[Training]")
        lines.append(f"  batch_size     : {self.training.batch_size}")
        lines.append(f"  num_epochs     : {self.training.num_epochs}")
        lines.append(f"  learning_rate  : {self.training.learning_rate}")
        lines.append(f"  weight_decay   : {self.training.weight_decay}")

        # Image config
        lines.append("\n[Image]")
        lines.append(f"  img_size       : {self.image.img_size}")

        # System config
        lines.append("\n[System]")
        lines.append(f"  num_workers    : {self.system.num_workers}")
        lines.append(f"  seed           : {self.system.seed}")
        lines.append(f"  device         : {self.system.device}")

        # Paths config
        lines.append("\n[Paths]")
        lines.append(f"  checkpoint_dir : {self.paths.checkpoint_dir}")
        lines.append(f"  log_dir        : {self.paths.log_dir}")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    """
    Test suite for the configuration management system.
    """
    print("=" * 70)
    print("Configuration Management System Test Suite")
    print("=" * 70)

    # Test 1: Create default configuration
    print("\n[Test 1] Creating default configuration...")
    config = Config()
    print(config)
    print("✓ Default configuration created successfully")

    # Test 2: Create configuration with custom parameters
    print("\n[Test 2] Creating custom configuration...")
    custom_config = Config(
        data=DataConfig(
            train_csv="data/aptos/train.csv",
            train_img_dir="data/aptos/train_images",
            test_csv="data/aptos/test.csv",
            test_img_dir="data/aptos/test_images"
        ),
        model=ModelConfig(
            model_name="vit_base_patch16_224",
            num_classes=5,
            pretrained=True
        ),
        training=TrainingConfig(
            batch_size=32,
            num_epochs=50,
            learning_rate=3e-4
        ),
        image=ImageConfig(img_size=384)
    )
    print("✓ Custom configuration created")
    print(f"  Model: {custom_config.model.model_name}")
    print(f"  Batch size: {custom_config.training.batch_size}")
    print(f"  Image size: {custom_config.image.img_size}")

    # Test 3: Test to_dict conversion
    print("\n[Test 3] Testing dictionary conversion...")
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict), "to_dict should return a dictionary"
    assert 'model' in config_dict, "Dictionary should contain 'model' key"
    assert config_dict['model']['model_name'] == 'resnet50', "Model name mismatch"
    print("✓ Dictionary conversion works correctly")

    # Test 4: Test from_dict creation
    print("\n[Test 4] Testing from_dict creation...")
    test_dict = {
        'model': {'model_name': 'efficientnet_b0'},
        'training': {'batch_size': 64}
    }
    config_from_dict = Config.from_dict(test_dict)
    assert config_from_dict.model.model_name == 'efficientnet_b0'
    assert config_from_dict.training.batch_size == 64
    print("✓ from_dict creation works correctly")

    # Test 5: Test YAML save/load
    print("\n[Test 5] Testing YAML serialization...")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_yaml = f.name

    try:
        # Save to YAML
        custom_config.to_yaml(temp_yaml)

        # Load from YAML
        loaded_config = Config.from_yaml(temp_yaml)
        assert loaded_config.model.model_name == custom_config.model.model_name
        assert loaded_config.training.batch_size == custom_config.training.batch_size
        print("✓ YAML save/load works correctly")
    finally:
        # Cleanup
        os.remove(temp_yaml)

    # Test 6: Test validation with valid parameters
    print("\n[Test 6] Testing validation...")
    try:
        config.validate(create_dirs=True)
        print("✓ Validation passed for valid configuration")
    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Test 7: Test validation with invalid parameters
    print("\n[Test 7] Testing validation with invalid parameters...")
    invalid_config = Config(
        training=TrainingConfig(batch_size=-1)
    )
    try:
        invalid_config.validate(create_dirs=False)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught invalid parameter: {str(e)[:60]}...")

    # Test 8: Test update method
    print("\n[Test 8] Testing config update...")
    updated_config = config.update(
        model={'model_name': 'resnet101'},
        training={'batch_size': 48}
    )
    assert updated_config.model.model_name == 'resnet101'
    assert updated_config.training.batch_size == 48
    assert config.model.model_name == 'resnet50'  # Original unchanged
    print("✓ Config update works correctly")

    # Test 9: Test device detection
    print("\n[Test 9] Testing device detection...")
    print(f"  Detected device: {config.system.device}")
    if torch.cuda.is_available():
        print("  ✓ CUDA available")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✓ Apple Silicon MPS available")
    else:
        print("  ✓ Using CPU")

    # Test 10: Pretty print
    print("\n[Test 10] Testing pretty print...")
    print(config)

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)

    print("\nUsage examples:")
    print("""
# Create configuration with defaults
from scripts.config import Config

config = Config()

# Create with custom parameters
config = Config(
    data=DataConfig(
        train_csv='data/aptos/train.csv',
        train_img_dir='data/aptos/train_images'
    ),
    training=TrainingConfig(batch_size=32, num_epochs=50)
)

# Load from YAML
config = Config.from_yaml('configs/experiment.yaml')

# Validate and create directories
config.validate()

# Save to YAML
config.to_yaml('configs/saved_config.yaml')

# Update specific parameters
new_config = config.update(
    model={'model_name': 'vit_base_patch16_224'},
    training={'batch_size': 64}
)
""")
